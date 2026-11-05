"""Tests for ``tutor.ask``: the search_book tool, reading a request, and running its search.

No LLM is called: each tool request is written by hand, shaped like one entry
of the ``tool_calls`` that ``complete_with_tools()`` returns. ``search()`` is
swapped for a fake with ``monkeypatch``, so no database or embedding model is
needed, and ``load_prompt`` too, so the tests don't depend on the prompt's
wording.
"""

from typing import Any

import pytest

from tutor import ask
from tutor.ask import (
    NOTHING_FOUND,
    NOTHING_NEW,
    SEARCH_BOOK_TOOL,
    SEARCH_K,
    SearchBook,
    format_passages,
    new_passages,
    read_search_request,
    run_search_request,
    start_messages,
    tool_result,
)
from tutor.llm import Prompt, UnusableAnswer


def request(arguments: str, name: str = "search_book") -> dict[str, Any]:
    """Build a tool request like one entry of a reply's ``tool_calls``.

    Args:
        arguments: The arguments as the LLM writes them: a JSON string.
        name: The tool it asks for.

    Returns:
        The request, with the id "call_1".
    """
    return {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}


def passage(passage_id: int, pages: tuple[str, str], text: str) -> dict[str, Any]:
    """Build a stand-in for a passage row.

    ``format_passages()`` reads a row by column name, which a dict allows
    just as a ``sqlite3.Row`` does.

    Args:
        passage_id: The passage's id.
        pages: The printed pages it starts and ends on.
        text: Its text.

    Returns:
        The columns ``format_passages()`` reads.
    """
    return {
        "id": passage_id,
        "number": "1.3.1.3",
        "title": "Tagging",
        "page_label": pages[0],
        "end_page_label": pages[1],
        "text": text,
    }


RESULTS = [
    (0.71, passage(22, ("12", "13"), "Tagging predicts every label that applies.")),
    (0.66, passage(21, ("12", "12"), "Classification picks one label.")),
]
"""Two search results, best first: one passage over two pages, one on a single page."""


def test_the_tool_definition_comes_from_search_book() -> None:
    """The tool is called search_book, described by SearchBook's docstring, with one argument."""
    function = SEARCH_BOOK_TOOL["function"]
    assert function["name"] == "search_book"
    assert function["description"].startswith("Search the book Dive into Deep Learning")
    assert list(function["parameters"]["properties"]) == ["query"]
    assert function["parameters"]["required"] == ["query"]


def test_the_chat_starts_with_the_prompt_then_the_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The system message is prompts/ask_book.md, and the user message is the question."""
    monkeypatch.setattr(ask, "load_prompt", lambda name: Prompt(f"the {name} prompt", "v1"))
    assert start_messages("How is tagging different from classification?") == [
        {"role": "system", "content": "the ask_book prompt"},
        {"role": "user", "content": "How is tagging different from classification?"},
    ]


def test_a_request_is_read_into_search_book() -> None:
    """The JSON string of arguments becomes a checked SearchBook."""
    assert read_search_request(request('{"query": "tagging"}')) == SearchBook(query="tagging")


def test_a_request_for_another_tool_is_unusable() -> None:
    """Only search_book exists; a request naming anything else can't be run."""
    with pytest.raises(UnusableAnswer, match="doesn't exist"):
        read_search_request(request('{"query": "tagging"}', name="web_search"))


@pytest.mark.parametrize(
    "arguments",
    [
        '{"q": "tagging"}',  # the wrong name: query is missing
        '{"query": 3}',  # a number, not text
        "tagging",  # not JSON at all
    ],
)
def test_arguments_that_dont_fit_are_unusable(arguments: str) -> None:
    """Arguments that don't fit SearchBook raise UnusableAnswer instead of searching for nothing."""
    with pytest.raises(UnusableAnswer, match="broke a rule"):
        read_search_request(request(arguments))


def test_passages_are_shown_with_their_id_section_and_pages() -> None:
    """Each passage gets a header line the LLM can cite, then its text; blank lines between."""
    assert format_passages(RESULTS) == (
        "[passage 22] 1.3.1.3 Tagging, pages 12–13\n"
        "Tagging predicts every label that applies.\n"
        "\n"
        "[passage 21] 1.3.1.3 Tagging, page 12\n"
        "Classification picks one label."
    )


def test_no_passages_say_nothing_was_found() -> None:
    """An empty search tells the LLM so, instead of handing it an empty text."""
    assert format_passages([]) == NOTHING_FOUND


def test_a_tool_result_names_the_request_it_answers() -> None:
    """The result carries the request's id and the tool's name."""
    assert tool_result(request('{"query": "tagging"}'), "found") == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "search_book",
        "content": "found",
    }


def test_a_request_runs_the_search_and_answers_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The LLM's query reaches search() with SEARCH_K; the passages and their ids come back."""
    searched: list[tuple[str, int]] = []

    def fake_search(conn: Any, query: str, k: int) -> list[tuple[float, dict[str, Any]]]:
        """Record what was searched, and return the two fixed results."""
        searched.append((query, k))
        return RESULTS

    monkeypatch.setattr(ask, "search", fake_search)

    message, passage_ids = run_search_request(None, request('{"query": "tagging"}'), [])

    assert searched == [("tagging", SEARCH_K)]
    assert message == tool_result(request('{"query": "tagging"}'), format_passages(RESULTS))
    assert passage_ids == [22, 21]


def test_a_search_that_finds_nothing_hands_out_no_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing above the cut-off: the LLM reads NOTHING_FOUND, and no passage id is handed out."""
    monkeypatch.setattr(ask, "search", lambda conn, query, k: [])
    message, passage_ids = run_search_request(None, request('{"query": "pizza recipes"}'), [])
    assert message["content"] == NOTHING_FOUND
    assert passage_ids == []


def test_new_passages_drops_what_was_shown_and_keeps_at_most_search_k() -> None:
    """Shown ids are left out, the order stays best first, and no more than SEARCH_K remain."""
    results = [
        *RESULTS,
        (0.60, passage(20, ("11", "11"), "x")),
        (0.58, passage(19, ("10", "10"), "y")),
    ]
    assert [pair[1]["id"] for pair in new_passages(results, [21])] == [22, 20]


def test_a_later_search_brings_new_passages_instead_of_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With passage 22 already shown, one more is asked for, and only new ones are handed out."""
    results = [*RESULTS, (0.60, passage(20, ("11", "11"), "Regression predicts a number."))]
    asked_for: list[int] = []

    def fake_search(conn: Any, query: str, k: int) -> list[tuple[float, dict[str, Any]]]:
        """Record how many passages were asked for, and return the three fixed results."""
        asked_for.append(k)
        return results

    monkeypatch.setattr(ask, "search", fake_search)

    message, passage_ids = run_search_request(None, request('{"query": "tagging"}'), [22])

    assert asked_for == [SEARCH_K + 1]
    assert passage_ids == [21, 20]
    assert "[passage 22]" not in message["content"]


def test_a_search_that_finds_only_repeats_says_nothing_new(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every passage found was shown already: the LLM reads NOTHING_NEW, and no id is handed out."""
    monkeypatch.setattr(ask, "search", lambda conn, query, k: RESULTS)
    message, passage_ids = run_search_request(None, request('{"query": "tagging"}'), [22, 21])
    assert message["content"] == NOTHING_NEW
    assert passage_ids == []


def test_nothing_found_stays_apart_from_nothing_new(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passages shown earlier don't change it: a search that finds nothing says NOTHING_FOUND."""
    monkeypatch.setattr(ask, "search", lambda conn, query, k: [])
    message, _ = run_search_request(None, request('{"query": "pizza recipes"}'), [22])
    assert message["content"] == NOTHING_FOUND
