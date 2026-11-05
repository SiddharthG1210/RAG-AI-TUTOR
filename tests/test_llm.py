"""Tests for ``tutor.llm``: prompts, schemas, the structured call, tool calls, 429s, the token log.

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
from tutor.llm import (
    FreeLimitReached,
    UnusableAnswer,
    complete_structured,
    complete_with_tools,
    function_tool,
    load_prompt,
    strict_schema,
)


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


def tool_response(content: str | None, calls: list[tuple[str, str, str]]) -> SimpleNamespace:
    """Build a fake SDK response to a call with tools.

    Args:
        content: The reply's text, or None (as in a reply that only asks for a tool).
        calls: (id, name, arguments) for each tool request; empty for none.

    Returns:
        An object shaped like the SDK's response. Like the SDK, it gives
        ``tool_calls=None``, not an empty list, when there's no request.
    """
    usage = SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    tool_calls = [
        SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=arguments))
        for call_id, name, arguments in calls
    ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls or None)
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


def bad_request(code: str) -> openai.BadRequestError:
    """Build the error the SDK raises on HTTP 400, with Groq's error code.

    Args:
        code: Groq's code, e.g. "json_validate_failed" when the answer broke the schema.

    Returns:
        The error. The SDK reads ``code`` from the error body.
    """
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    body = {"message": "refused", "type": "invalid_request_error", "code": code}
    return openai.BadRequestError(
        "refused", response=httpx.Response(400, request=request), body=body
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
    """Three items break max_length=2: Pydantic refuses the answer, raised as UnusableAnswer."""
    items = ", ".join('{"name": "a"}' for _ in range(3))
    use(monkeypatch, FakeClient([response(f'{{"items": [{items}]}}')]))
    with pytest.raises(UnusableAnswer, match="at most 2"):
        call(connect(":memory:"))


def test_an_answer_that_broke_the_schema_is_asked_for_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After Groq's json_validate_failed, the request is sent once more and its answer used."""
    client = FakeClient(
        [bad_request("json_validate_failed"), response('{"items": [{"name": "a"}]}')]
    )
    use(monkeypatch, client)
    assert call(connect(":memory:")).items[0].name == "a"
    assert len(client.requests) == 2


def test_a_second_broken_answer_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two json_validate_failed in a row: UnusableAnswer is raised, with no third try."""
    client = FakeClient([bad_request("json_validate_failed"), bad_request("json_validate_failed")])
    use(monkeypatch, client)
    with pytest.raises(UnusableAnswer, match="twice"):
        call(connect(":memory:"))
    assert len(client.requests) == 2


def test_an_empty_answer_is_unusable(monkeypatch: pytest.MonkeyPatch) -> None:
    """An answer with no content raises UnusableAnswer instead of returning nothing."""
    use(monkeypatch, FakeClient([response("")]))
    with pytest.raises(UnusableAnswer, match="came back empty"):
        call(connect(":memory:"))


def test_any_other_400_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused request (e.g. an unknown model) won't fix itself, so it's raised at once."""
    client = FakeClient([bad_request("model_not_found")])
    use(monkeypatch, client)
    with pytest.raises(openai.BadRequestError):
        call(connect(":memory:"))
    assert len(client.requests) == 1


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


TOOLS = [function_tool(Item, "find_item")]
"""One tool built from the small test model, for the calls with tools."""


def call_with_tools(conn: Any) -> dict[str, Any]:
    """Run complete_with_tools with the test tool.

    Args:
        conn: An open connection.

    Returns:
        The reply, as a plain dict.
    """
    messages = [{"role": "user", "content": "hi"}]
    return complete_with_tools(
        conn, messages, TOOLS, model="m", reasoning_effort="low", purpose="ask"
    )


def test_a_tool_is_built_from_a_model() -> None:
    """The docstring becomes the tool's description; the fields become strict parameters."""
    tool = function_tool(Item, "find_item")
    assert tool["type"] == "function"
    function = tool["function"]
    assert function["name"] == "find_item"
    assert function["description"] == "A small nested model, standing in for Question."
    assert "description" not in function["parameters"]
    assert function["parameters"]["required"] == ["name"]
    assert function["parameters"]["additionalProperties"] is False
    # The field's description stays: the LLM reads it to fill the argument.
    assert function["parameters"]["properties"]["name"]["description"] == "a name"


def test_a_tool_needs_a_docstring() -> None:
    """A model with no docstring would give the LLM no description, so it fails loudly."""

    class Bare(BaseModel):
        x: int

    with pytest.raises(ValueError, match="docstring"):
        function_tool(Bare, "bare")


def test_a_tool_request_comes_back_as_a_plain_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tools go out with no response_format; the request comes back as a dict; tokens are logged."""
    client = FakeClient([tool_response(None, [("call_1", "find_item", '{"name": "a"}')])])
    use(monkeypatch, client)
    conn = connect(":memory:")

    reply = call_with_tools(conn)

    assert reply == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "find_item", "arguments": '{"name": "a"}'},
            }
        ],
    }
    assert client.requests[0]["tools"] == TOOLS
    assert "response_format" not in client.requests[0]
    row = conn.execute("SELECT model, purpose, total_tokens FROM llm_calls").fetchone()
    assert tuple(row) == ("m", "ask", 150)


def test_a_text_reply_has_no_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the LLM answers instead of asking for a tool, the dict has no tool_calls key."""
    use(monkeypatch, FakeClient([tool_response("The book says so.", [])]))
    reply = call_with_tools(connect(":memory:"))
    assert reply == {"role": "assistant", "content": "The book says so."}


def test_a_reply_with_neither_text_nor_request_is_unusable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No text and no tool request: there's nothing to use, so UnusableAnswer is raised."""
    use(monkeypatch, FakeClient([tool_response(None, [])]))
    with pytest.raises(UnusableAnswer, match="came back empty"):
        call_with_tools(connect(":memory:"))


def test_a_malformed_tool_request_is_asked_for_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """After Groq's tool_use_failed, the request is sent once more and its reply used."""
    client = FakeClient(
        [
            bad_request("tool_use_failed"),
            tool_response(None, [("call_1", "find_item", '{"name": "a"}')]),
        ]
    )
    use(monkeypatch, client)
    assert call_with_tools(connect(":memory:"))["tool_calls"][0]["id"] == "call_1"
    assert len(client.requests) == 2


def test_a_second_malformed_tool_request_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two tool_use_failed in a row: UnusableAnswer is raised, with no third try."""
    client = FakeClient([bad_request("tool_use_failed"), bad_request("tool_use_failed")])
    use(monkeypatch, client)
    with pytest.raises(UnusableAnswer, match="tool_use_failed"):
        call_with_tools(connect(":memory:"))
    assert len(client.requests) == 2
