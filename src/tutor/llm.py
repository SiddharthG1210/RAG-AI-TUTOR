"""The one place the tutor talks to the LLM: prompts, the call, rate limits, the token log.

M1 concept "Structured output (question writing)". Every LLM call in the
tutor goes through ``complete_structured()`` (CLAUDE.md › Guardrails).

**The SDK.** Groq's API is OpenAI-compatible, so the official ``openai`` SDK
talks to it once ``base_url`` points at Groq. Switching provider means editing
``.env``, not this code (PLAN.md › Tech).

**Structured output.** The request carries a JSON Schema built from a Pydantic
model (such as ``QuestionSet``), with ``strict: true``. In strict mode Groq
constrains the model's sampling so its answer always fits the schema: every
field present, nothing extra. Pydantic then turns that JSON into a Python
object and checks the rules the schema can't carry, such as "exactly 3".

**The free tier.** Each call's tokens are logged in the ``llm_calls`` table. On
HTTP 429 ("too many requests") the call waits as long as Groq's ``retry-after``
header says and tries once more; if that fails too, or the wait is longer than
a minute (a daily limit), it raises ``FreeLimitReached``, which the tutor shows
as "free limit reached" (PLAN.md › Free-tier guard).
"""

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import openai
from openai import OpenAI
from pydantic import BaseModel

from tutor.config import get_settings
from tutor.db import log_llm_call

PROMPTS_DIR = Path("prompts")
"""Where the prompt files live. Prompts are never strings in code (CLAUDE.md › Guardrails)."""

MAX_WAIT_SECONDS = 60.0
"""The longest the tutor waits after a 429. Longer means a daily limit: give up."""

UNSUPPORTED_KEYWORDS = ("minItems", "maxItems", "minLength", "maxLength", "pattern", "default")
"""Schema rules Groq's strict mode doesn't document, so they're left out of the
request. Pydantic still enforces them on the answer."""


class FreeLimitReached(Exception):
    """Groq's free-tier limit was hit, and waiting didn't help.

    The tutor shows this as "free limit reached" (PLAN.md › Free-tier guard).
    """


@dataclass(frozen=True)
class Prompt:
    """A prompt file's text and version.

    Attributes:
        text: The file's contents: the instructions sent to the LLM.
        version: The first 12 characters of the text's SHA-256 hash. Any edit
            to the file changes it, and it's saved with every question and
            grade, so a change in behavior traces back to a prompt change.
    """

    text: str
    version: str


def load_prompt(name: str) -> Prompt:
    """Read a prompt file from ``prompts/``.

    Args:
        name: The file's name without ``.md``, e.g. "question_writer".

    Returns:
        The prompt's text and version.
    """
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    # SHA-256 turns any text into a fixed 64-character fingerprint; one changed
    # character changes it completely. 12 characters are plenty to tell versions apart.
    return Prompt(text, hashlib.sha256(text.encode("utf-8")).hexdigest()[:12])


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    """Create the SDK client once, pointed at Groq.

    ``max_retries=0`` turns off the SDK's own silent retries, so the 429
    handling below is the only retry and stays visible.

    Returns:
        The client.
    """
    settings = get_settings()
    return OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key, max_retries=0)


def _make_strict(schema: dict[str, Any]) -> None:
    """Change one JSON Schema node, and everything inside it, to fit Groq's strict mode, in place.

    Strict mode needs every object to list all its properties as required and
    to set ``additionalProperties: false``.

    Args:
        schema: A schema node (a dict), changed in place.
    """
    for keyword in UNSUPPORTED_KEYWORDS:
        schema.pop(keyword, None)
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        schema["additionalProperties"] = False
        schema["required"] = list(properties)
        for child in properties.values():
            _make_strict(child)
    if isinstance(schema.get("items"), dict):
        _make_strict(schema["items"])
    for child in schema.get("anyOf", []):
        _make_strict(child)
    # Pydantic puts nested models (e.g. Question inside QuestionSet) in "$defs".
    for child in schema.get("$defs", {}).values():
        _make_strict(child)


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    """Build the JSON Schema for a Pydantic model, in the form strict mode accepts.

    The schema includes every field's ``description``: the LLM reads those as
    instructions, so they're part of the prompt.

    Args:
        model: The Pydantic model the answer must fit, e.g. ``QuestionSet``.

    Returns:
        The schema, as a dict ready to send.
    """
    schema = model.model_json_schema()
    _make_strict(schema)
    return schema


def _retry_after(error: openai.RateLimitError) -> float | None:
    """Read how many seconds Groq asks us to wait, from the 429 response's header.

    Args:
        error: The rate-limit error.

    Returns:
        The seconds, or None if the header is missing or unreadable.
    """
    value = error.response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _create(**request: Any) -> Any:
    """Send one chat-completion request, waiting and retrying once after a 429.

    Args:
        **request: The request's fields: model, messages, response_format, ….

    Returns:
        The SDK's response object.

    Raises:
        FreeLimitReached: If the second try also hits the limit, or the wait
            Groq asks for is longer than MAX_WAIT_SECONDS.
    """
    try:
        return _client().chat.completions.create(**request)
    except openai.RateLimitError as error:
        wait = _retry_after(error)
        if wait is None or wait > MAX_WAIT_SECONDS:
            raise FreeLimitReached(
                f"Groq's free limit was reached (retry-after: {wait})."
            ) from error
        time.sleep(wait)
    try:
        return _client().chat.completions.create(**request)
    except openai.RateLimitError as error:
        raise FreeLimitReached("Groq's free limit was reached again after waiting.") from error


def complete_structured[T: BaseModel](
    conn: sqlite3.Connection,
    messages: list[dict[str, str]],
    schema: type[T],
    *,
    model: str,
    reasoning_effort: str,
    purpose: str,
) -> T:
    """Ask the LLM for an answer in a fixed shape, log its tokens, and return it as an object.

    ``[T: BaseModel]`` makes this a generic function (Python 3.12): pass
    ``QuestionSet`` as the schema and you get a ``QuestionSet`` back.

    Args:
        conn: An open connection, for the token log.
        messages: The chat messages, e.g. ``[{"role": "system", "content": …},
            {"role": "user", "content": …}]``.
        schema: The Pydantic model the answer must fit.
        model: The model's name, e.g. ``get_settings().question_model``.
        reasoning_effort: "low", "medium" or "high". gpt-oss models think
            before answering, and their thinking tokens count toward the
            free-tier limits, so every call sets this (PLAN.md › Decisions › LLM).
        purpose: What the call is for, e.g. "questions", saved in the token log.

    Returns:
        The answer, as an instance of ``schema``.

    Raises:
        FreeLimitReached: If Groq's free-tier limit stops the call.
        ValueError: If the answer comes back empty.
        pydantic.ValidationError: If the answer breaks a rule of the schema.
    """
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": strict_schema(schema), "strict": True},
    }
    start = time.perf_counter()
    response = _create(
        model=model,
        messages=messages,
        response_format=response_format,
        reasoning_effort=reasoning_effort,
    )
    seconds = time.perf_counter() - start
    usage = response.usage
    log_llm_call(
        conn,
        model,
        purpose,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.total_tokens,
        seconds,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError(f"The {purpose} call to {model} came back empty.")
    # model_validate_json parses the JSON text and checks every rule in the model.
    return schema.model_validate_json(content)
