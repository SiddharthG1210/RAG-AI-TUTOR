"""The one place the tutor talks to the LLM: prompts, the call, rate limits, the token log.

M1 concept "Structured output (question writing)", plus tool calls from the M2
concept "Tool calling". Every LLM call in the tutor goes through
``complete_structured()`` or ``complete_with_tools()`` (CLAUDE.md › Guardrails).

**The SDK.** Groq's API is OpenAI-compatible, so the official ``openai`` SDK
talks to it once ``base_url`` points at Groq. Switching provider means editing
``.env``, not this code (PLAN.md › Tech).

**Structured output.** The request carries a JSON Schema built from a Pydantic
model (such as ``QuestionSet``), with ``strict: true``. In strict mode Groq
checks the model's answer against the schema (every field present, nothing
extra) before sending it. For gpt-oss that check can fail: the check script
once got a question with no ``quote``, and Groq answered HTTP 400
``json_validate_failed`` instead of the JSON. So the call is sent once more
after such a failure, and a second failure raises. Pydantic then turns the
JSON into a Python object and checks the rules the schema can't carry, such
as "exactly 3". Every way the answer itself can be unusable raises the same
error, ``UnusableAnswer``.

**Tool calls.** The request carries a list of tools: for each one a name, a
description and a JSON Schema of its arguments (``function_tool()`` builds
one from a Pydantic model). The LLM never runs a tool. When it wants one, its
reply holds a request instead of text: the tool's name and the arguments as
a JSON string. The caller runs the tool and sends the result back in the
next call (see ``tutor.ask``). Groq can't combine tools with structured
output in one request, so ``complete_with_tools()`` sends no
``response_format``.

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
from pydantic import BaseModel, ValidationError

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


class UnusableAnswer(ValueError):
    """The LLM answered, but its answer can't be used: it broke the schema, or came back empty.

    One error for every cause, so a caller can treat "the LLM's answer was
    bad" as one case: Groq's ``json_validate_failed`` or ``tool_use_failed``
    twice in a row, an empty answer, Pydantic refusing the JSON (for example
    2 questions where ``QuestionSet`` needs exactly 3), or a tool request
    naming a tool that doesn't exist (``tutor.ask``). The study loop counts
    it as a failed try at writing questions (M1 concept "Conditional
    edges"). A refused
    request, such as an unknown model name, is a different problem and stays
    an ``openai.BadRequestError``. It's a ``ValueError`` because the value
    that came back is the problem.
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


def function_tool(model: type[BaseModel], name: str) -> dict[str, Any]:
    """Build a tool definition, the entry in ``tools`` that the LLM reads, from a Pydantic model.

    The model's fields are the tool's arguments, and its docstring is the
    tool's description: the sentence the LLM reads to decide whether to call
    it. Pydantic puts the docstring at the top of the schema, as
    ``"description"``. The API wants it one level up, next to the name, so
    it's moved there. The same model later checks the arguments the LLM
    sends (``model_validate_json``), so what the LLM is told and what code
    accepts can't drift apart.

    Groq documents strict mode only for ``response_format``, not for tools, so
    nothing here asks Groq to check the arguments; Pydantic does it on our side.

    Args:
        model: The arguments' shape, e.g. ``SearchBook`` with its one field ``query``.
        name: The name the LLM writes back to ask for this tool, e.g. "search_book".

    Returns:
        ``{"type": "function", "function": {"name", "description", "parameters"}}``.

    Raises:
        ValueError: If the model has no docstring, since the LLM would get no
            description of the tool.
    """
    parameters = strict_schema(model)
    description = parameters.pop("description", None)
    if not description:
        raise ValueError(f"{model.__name__} needs a docstring: it's the tool's description.")
    # "type": "function" means a tool the caller runs. (Groq also has
    # built-in tools, such as web search, that run on its own servers.)
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


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


def _create_valid(malformed_code: str, **request: Any) -> Any:
    """Send one chat-completion request, and once more if the LLM's answer came out malformed.

    Groq checks what the model wrote before sending it on. When it doesn't
    fit, Groq answers HTTP 400 with a code instead:

    - ``json_validate_failed``: a strict-mode answer that broke its schema.
    - ``tool_use_failed``: a tool request Groq couldn't read. (Not in Groq's
      error docs, but it's the code Groq sends.)

    The model samples its answer, so a second try usually comes out right.
    The failed try's tokens can't be logged: the error carries no token counts.

    Args:
        malformed_code: The code that means "the answer came out malformed"
            for this kind of call.
        **request: The request's fields: model, messages, response_format, ….

    Returns:
        The SDK's response object.

    Raises:
        UnusableAnswer: If the second try comes out malformed too.
        openai.BadRequestError: If the request was refused for any other reason.
    """
    try:
        return _create(**request)
    except openai.BadRequestError as error:
        # Any other 400 (a bad model name, a malformed schema) won't fix
        # itself on a second try, so it's raised at once.
        if error.code != malformed_code:
            raise
    try:
        return _create(**request)
    except openai.BadRequestError as error:
        if error.code != malformed_code:
            raise
        # "from error" keeps Groq's error attached, so the traceback shows both.
        raise UnusableAnswer(
            f"The answer came out malformed twice in a row ({malformed_code})."
        ) from error


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
        UnusableAnswer: If the answer breaks the schema twice in a row
            (``json_validate_failed``), comes back empty, or breaks a rule of
            the schema that only Pydantic checks.
        openai.BadRequestError: If Groq refuses the request itself.
    """
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": strict_schema(schema), "strict": True},
    }
    start = time.perf_counter()
    response = _create_valid(
        "json_validate_failed",
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
        raise UnusableAnswer(f"The {purpose} call to {model} came back empty.")
    # model_validate_json parses the JSON text and checks every rule in the
    # model. A rule it refuses raises pydantic's ValidationError, which is
    # passed on as UnusableAnswer, with pydantic's message saying which rule.
    try:
        return schema.model_validate_json(content)
    except ValidationError as error:
        raise UnusableAnswer(f"The {purpose} answer from {model} broke a rule: {error}") from error


def complete_with_tools(
    conn: sqlite3.Connection,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    *,
    model: str,
    reasoning_effort: str,
    purpose: str,
) -> dict[str, Any]:
    """Send the messages with tools, log the tokens, and return the LLM's reply as a plain dict.

    The reply is one of two things. When the LLM wants a tool, it's a request:
    ``content`` is usually empty and ``tool_calls`` holds the tool's name and
    its arguments, as a JSON string. Otherwise it's text in ``content`` and
    there's no ``tool_calls``. gpt-oss on Groq asks for at most one tool per
    reply, but ``tool_calls`` is still a list, because that's the API's shape.

    The reply is copied into a plain dict, with only the fields the API reads
    back, so it can go straight into the next call's ``messages``: a tool's
    result must follow the request it answers (``tool_call_id``). The SDK's
    own object carries more, such as gpt-oss's reasoning, and a plain dict is
    also what LangGraph's checkpointer can store (the M2 concept "The agent
    loop as a graph" keeps this history in the graph's state).

    Args:
        conn: An open connection, for the token log.
        messages: The chat so far. Besides ``system`` and ``user`` messages
            it can hold earlier replies from this function and the tool
            results that answer them.
        tools: Tool definitions, each built by ``function_tool()``.
        model: The model's name, e.g. ``get_settings().ask_model``.
        reasoning_effort: "low", "medium" or "high" (see ``complete_structured()``).
        purpose: What the call is for, e.g. "ask", saved in the token log.

    Returns:
        ``{"role": "assistant", "content": ...}``, plus ``"tool_calls": [{"id",
        "type": "function", "function": {"name", "arguments"}}]`` when the LLM
        asked for a tool.

    Raises:
        FreeLimitReached: If Groq's free-tier limit stops the call.
        UnusableAnswer: If the tool request comes out malformed twice in a row
            (``tool_use_failed``), or the reply has neither text nor a request.
        openai.BadRequestError: If Groq refuses the request itself.
    """
    start = time.perf_counter()
    # No response_format: Groq can't combine tools with structured output.
    response = _create_valid(
        "tool_use_failed",
        model=model,
        messages=messages,
        tools=tools,
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
    message = response.choices[0].message
    # The SDK gives None, not an empty list, when there's no request.
    tool_calls = message.tool_calls or []
    if not message.content and not tool_calls:
        raise UnusableAnswer(f"The {purpose} call to {model} came back empty.")
    reply: dict[str, Any] = {"role": "assistant", "content": message.content}
    if tool_calls:
        reply["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in tool_calls
        ]
    return reply
