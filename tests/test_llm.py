"""Tests for ``tutor.llm``: prompts, the strict schema, the structured call, 429s and the token log.

No request ever reaches Groq: ``monkeypatch`` swaps the SDK client for a fake
whose ``chat.completions.create`` returns made-up responses or raises a
made-up 429 (CI has no API key). ``time.sleep`` is swapped too, so a test
never really waits.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from pydantic import BaseModel, Field

from tutor import llm
from tutor.db import connect
from tutor.llm import FreeLimitReached, complete_structured, load_prompt, strict_schema


class Item(BaseModel):
    """A small nested model, standing in for Question."""

    name: str = Field(description="a name")


class Box(BaseModel):
    """A small outer model, standing in for QuestionSet."""

    items: list[Item] = Field(min_length=1, max_length=2, description="one or two items")


def response(content: str) -> SimpleNamespace:
    """Build a fake SDK response with the given message text and some token counts.

    Args:
        content: The text the "model" answered.

    Returns:
        An object shaped like the SDK's response.
    """
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    message = SimpleNamespace(content=content)
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=message)])


def rate_limit_error(retry_after: str | None) -> openai.RateLimitError:
    """Build the error the SDK raises on HTTP 429, with an optional retry-after header.

    Args:
        retry_after: The header's value in seconds, or None for no header.

    Returns:
        The error.
    """
    headers = {"retry-after": retry_after} if retry_after else {}
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return openai.RateLimitError(
        "rate limited", response=httpx.Response(429, headers=headers, request=request), body=None
    )


class FakeClient:
    """Plays the SDK client: hands out the given outcomes one call at a time.

    Each outcome is either a response, which is returned, or an exception,
    which is raised. Every request is recorded.
    """

    def __init__(self, outcomes: list[Any]) -> None:
        """Remember the outcomes, in order.

        Args:
            outcomes: What each successive call returns or raises.
        """
        self.outcomes = outcomes
        self.requests: list[dict[str, Any]] = []
        # client.chat.completions.create(...) must exist, so build that path.
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request: Any) -> Any:
        """Record the request and hand out the next outcome.

        Args:
            **request: The request's fields.

        Returns:
            The next response.
        """
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace time.sleep with a recorder, so tests never wait.

    Returns:
        The list each would-be wait is appended to.
    """
    recorded: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", recorded.append)
    return recorded


def use(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> None:
    """Make the wrapper use the fake client.

    Args:
        monkeypatch: pytest's fixture.
        client: The fake.
    """
    monkeypatch.setattr(llm, "_client", lambda: client)


def call(conn: Any) -> Box:
    """Run complete_structured with the test schema.

    Args:
        conn: An open connection.

    Returns:
        The parsed answer.
    """
    messages = [{"role": "user", "content": "hi"}]
    return complete_structured(
        conn, messages, Box, model="m", reasoning_effort="low", purpose="test"
    )


def test_strict_schema_requires_everything_and_forbids_extras() -> None:
    """Every object lists all its fields as required, allows nothing extra, and drops minItems."""
    schema = strict_schema(Box)
    assert schema["required"] == ["items"]
    assert schema["additionalProperties"] is False
    assert "minItems" not in schema["properties"]["items"]
    item = schema["$defs"]["Item"]
    assert item["required"] == ["name"]
    assert item["additionalProperties"] is False
    # Descriptions stay: the LLM reads them as instructions.
    assert item["properties"]["name"]["description"] == "a name"


def test_answer_is_parsed_and_its_tokens_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The JSON answer comes back as a Box, and the call is recorded in llm_calls."""
    client = FakeClient([response('{"items": [{"name": "a"}]}')])
    use(monkeypatch, client)
    conn = connect(":memory:")

    box = call(conn)

    assert box == Box(items=[Item(name="a")])
    assert client.requests[0]["response_format"]["json_schema"]["strict"] is True
    assert client.requests[0]["reasoning_effort"] == "low"
    row = conn.execute("SELECT model, purpose, total_tokens FROM llm_calls").fetchone()
    assert tuple(row) == ("m", "test", 150)


def test_a_rule_the_schema_cant_carry_is_still_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three items break max_length=2: Pydantic refuses the answer."""
    items = ", ".join('{"name": "a"}' for _ in range(3))
    use(monkeypatch, FakeClient([response(f'{{"items": [{items}]}}')]))
    with pytest.raises(ValueError, match="at most 2"):
        call(connect(":memory:"))


def test_a_429_waits_as_asked_then_retries(
    monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    """After a 429 asking for 2 seconds, the call waits 2 seconds and the retry succeeds."""
    client = FakeClient([rate_limit_error("2"), response('{"items": [{"name": "a"}]}')])
    use(monkeypatch, client)
    assert call(connect(":memory:")).items[0].name == "a"
    assert waits == [2.0]
    assert len(client.requests) == 2


def test_a_long_wait_means_free_limit_reached(
    monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    """A retry-after of an hour is a daily limit: give up at once instead of waiting."""
    use(monkeypatch, FakeClient([rate_limit_error("3600")]))
    with pytest.raises(FreeLimitReached):
        call(connect(":memory:"))
    assert waits == []


def test_a_second_429_means_free_limit_reached(
    monkeypatch: pytest.MonkeyPatch, waits: list[float]
) -> None:
    """If the retry is rate-limited too, the tutor stops with "free limit reached"."""
    use(monkeypatch, FakeClient([rate_limit_error("1"), rate_limit_error("1")]))
    with pytest.raises(FreeLimitReached):
        call(connect(":memory:"))


def test_a_prompts_version_changes_when_its_text_does(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Editing a prompt file gives it a new version; the same text gives the same version."""
    monkeypatch.setattr(llm, "PROMPTS_DIR", tmp_path)
    (tmp_path / "p.md").write_text("Write 3 questions.", encoding="utf-8")
    first = load_prompt("p")
    assert first.text == "Write 3 questions."
    assert len(first.version) == 12
    assert load_prompt("p").version == first.version
    (tmp_path / "p.md").write_text("Write 3 questions!", encoding="utf-8")
    assert load_prompt("p").version != first.version
