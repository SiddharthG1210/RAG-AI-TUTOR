"""Tests for ``tutor.graph.agent``: Ask the book's loop, run through LangGraph.

No LLM and no database search: ``complete_with_tools`` is swapped for a fake
that hands out scripted replies (an order, or a text answer), and
``run_search_request`` for a fake that hands back made-up passage ids. The
settings and ``start_messages`` are swapped too, since CI has no ``.env``
and the tests shouldn't depend on the prompt's wording.

The routing tests check the two routing functions on their own, with plain
dicts. The loop tests run the compiled graph the way ``scripts/ask.py`` does.
"""

from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.graph import END

from tutor.db import connect
from tutor.graph import agent
from tutor.graph.agent import (
    LIMIT_REACHED,
    MAX_SEARCHES,
    ask_the_book,
    build_agent_graph,
    final_text,
    route_after_llm,
    route_after_search,
    run_search,
)
from tutor.graph.state import Context

START = [
    {"role": "system", "content": "the rules"},
    {"role": "user", "content": "How is tagging different from classification?"},
]
"""What the fake start_messages() hands back."""


def order(query: str, call_id: str = "call_1") -> dict[str, Any]:
    """Build an LLM reply that orders one search, shaped like complete_with_tools() returns.

    Args:
        query: What it asks to search for.
        call_id: The order's id.

    Returns:
        The reply.
    """
    arguments = '{"query": "' + query + '"}'
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "search_book", "arguments": arguments},
            }
        ],
    }


def answer(text: str) -> dict[str, Any]:
    """Build an LLM reply that answers in text.

    Args:
        text: The answer.

    Returns:
        The reply.
    """
    return {"role": "assistant", "content": text}


class FakeLLM:
    """Plays complete_with_tools(): hands out scripted replies, one per call, and records each."""

    def __init__(self, replies: list[dict[str, Any]]) -> None:
        """Remember the replies, in order.

        Args:
            replies: What each successive call returns.
        """
        self.replies = list(replies)
        self.sent: list[list[dict[str, Any]]] = []

    def __call__(
        self, conn: Any, messages: list[dict[str, Any]], tools: list, **kwargs: Any
    ) -> dict:
        """Record the chat sent, and hand out the next reply.

        Args:
            conn: Ignored.
            messages: The chat sent.
            tools: Ignored.
            **kwargs: Ignored.

        Returns:
            The next scripted reply.
        """
        self.sent.append(list(messages))
        return self.replies.pop(0)


def fake_search(
    conn: Any, tool_call: dict[str, Any], shown_ids: list[int]
) -> tuple[dict, list[int]]:
    """Play run_search_request(): two new passage ids per search, numbered after those shown.

    Args:
        conn: Ignored.
        tool_call: The order being answered.
        shown_ids: The ids already shown.

    Returns:
        A tool message naming the order's id, and two new ids.
    """
    first = len(shown_ids) + 1
    message = {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "name": "search_book",
        "content": "found",
    }
    return message, [first, first + 1]


@pytest.fixture
def use(monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201 (a fixture factory)
    """Swap the LLM, the search, the settings and the first messages inside ``tutor.graph.agent``.

    Returns:
        A function: give it the scripted replies, get back the fake LLM.
    """

    def install(replies: list[dict[str, Any]]) -> FakeLLM:
        llm = FakeLLM(replies)
        monkeypatch.setattr(agent, "complete_with_tools", llm)
        monkeypatch.setattr(agent, "run_search_request", fake_search)
        monkeypatch.setattr(agent, "get_settings", lambda: SimpleNamespace(ask_model="fake/model"))
        monkeypatch.setattr(agent, "start_messages", lambda question: list(START))
        return llm

    return install


def ask(question: str = "How is tagging different from classification?") -> dict:
    """Run the compiled loop on one question, with an in-memory database.

    Args:
        question: The question.

    Returns:
        The final state.
    """
    context = Context(conn=connect(":memory:"), session_id="session-1")
    return ask_the_book(build_agent_graph(), context, question)


# --- The routing functions -------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "expected"),
    [(order("tagging"), "run_search"), (answer("The book says…"), END)],
)
def test_route_after_llm(reply: dict, expected: str) -> None:
    """An order → run the search; a text answer → the end."""
    assert route_after_llm({"messages": [*START, reply]}) == expected


@pytest.mark.parametrize(
    ("searches_done", "expected"), [(1, "call_llm"), (2, "call_llm"), (MAX_SEARCHES, END)]
)
def test_route_after_search(searches_done: int, expected: str) -> None:
    """Under the limit → ask the LLM again; at the limit → the end, with no more LLM calls."""
    assert route_after_search({"searches_done": searches_done}) == expected


def test_every_name_a_route_returns_is_a_node() -> None:
    """A routing function naming a node that doesn't exist would end the loop silently.

    LangGraph only logs a warning ("wrote to unknown channel") and stops, so
    this checks that every name the routing functions can return is a node of
    the compiled graph (``END`` included, as ``"__end__"``).
    """
    node_names = set(build_agent_graph().get_graph().nodes)
    returned = {
        route_after_llm({"messages": [*START, order("tagging")]}),
        route_after_llm({"messages": [*START, answer("The book says…")]}),
        route_after_search({"searches_done": 1}),
        route_after_search({"searches_done": MAX_SEARCHES}),
    }
    assert returned <= node_names


# --- The whole loop --------------------------------------------------------------


def test_an_answer_without_searching_ends_after_one_call(use: Any) -> None:
    """The LLM answers at once: one call, no search, and the answer is the last message."""
    llm = use([answer("Tagging is…")])

    final = ask()

    assert len(llm.sent) == 1
    assert final["searches_done"] == 0
    assert final["shown_ids"] == []
    assert final_text(final) == "Tagging is…"


def test_two_searches_then_an_answer(use: Any) -> None:
    """The tagging example: search, search, answer. 3 calls, each sending the whole chat so far."""
    llm = use([order("tagging", "c1"), order("classification", "c2"), answer("The book says…")])

    final = ask()

    roles = [message["role"] for message in final["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    # Each call rereads everything before it: 2, then 4, then 6 messages.
    assert [len(sent) for sent in llm.sent] == [2, 4, 6]
    # Each result answers its own order, by id.
    assert [m["tool_call_id"] for m in final["messages"] if m["role"] == "tool"] == ["c1", "c2"]
    assert final["searches_done"] == 2
    assert final["shown_ids"] == [1, 2, 3, 4]
    assert final_text(final) == "The book says…"


def test_an_llm_that_keeps_ordering_searches_is_stopped_after_3(use: Any) -> None:
    """PLAN.md's M2 "Done when": a fake LLM that keeps asking for searches stops after 3.

    And code stops without asking the LLM a 4th time: 3 calls, 3 searches.
    """
    llm = use([order(f"search {n}", f"c{n}") for n in range(1, 10)])

    final = ask()

    assert final["searches_done"] == MAX_SEARCHES
    assert len(llm.sent) == MAX_SEARCHES
    assert final["messages"][-1]["role"] == "tool"
    assert final_text(final) is None


# --- run_search on its own --------------------------------------------------------


def run_search_on(state: dict, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Call the run_search node directly, with the fake search.

    Args:
        state: The state to hand it.
        monkeypatch: pytest's fixture.

    Returns:
        What the node returns.
    """
    monkeypatch.setattr(agent, "run_search_request", fake_search)
    runtime = SimpleNamespace(context=Context(conn=None, session_id="s"))
    return run_search(state, runtime)


def test_run_search_hands_the_ids_already_shown_to_the_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second search gets the first search's ids, so it can leave those passages out."""
    state = {"messages": [*START, order("classification")], "shown_ids": [1, 2], "searches_done": 1}

    changes = run_search_on(state, monkeypatch)

    assert changes["shown_ids"] == [1, 2, 3, 4]
    assert changes["searches_done"] == 2
    assert changes["messages"][-1]["tool_call_id"] == "call_1"


def test_an_order_past_the_limit_is_answered_but_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reply with two orders when one search is left: one runs, the other gets LIMIT_REACHED."""
    two_orders = order("a", "c1")
    two_orders["tool_calls"] += order("b", "c2")["tool_calls"]
    state = {"messages": [*START, two_orders], "shown_ids": [], "searches_done": MAX_SEARCHES - 1}

    changes = run_search_on(state, monkeypatch)

    results = changes["messages"][-2:]
    assert [m["tool_call_id"] for m in results] == ["c1", "c2"]
    assert results[0]["content"] == "found"
    assert results[1]["content"] == LIMIT_REACHED
    assert changes["searches_done"] == MAX_SEARCHES
