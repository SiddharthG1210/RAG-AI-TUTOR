"""Ask the book: the ``search_book`` tool, and running a search the LLM asked for.

M2 concept "Tool calling". After a grade you can ask a question of your own,
and an agent answers it from the book (PLAN.md › How the tutor behaves ›
Ask the book). The LLM can't read the book, and it can't run anything: all
it can do is write text. So it's given one tool, ``search_book``, described
by ``SearchBook``. When it wants a search, its reply is a request in a fixed
shape (``search_book``, ``{"query": "tagging"}``), and code runs the real
search, M1's ``search()``, and sends the passages back in the next call.
Code stays in charge of what actually runs.

One round, from the question to the passages handed back:

    "How is tagging different from classification?"
       │ start_messages()
    [system, user]
       │ complete_with_tools(..., [SEARCH_BOOK_TOOL], ...)     (tutor.llm)
    the LLM's request: tool_calls = [{id, name: "search_book",
                                      arguments: '{"query": "tagging"}'}]
       │ run_search_request(), which does the five steps below
       │   read_search_request()   the name checked, the arguments parsed
       │   search()                M1's search by meaning (tutor.retrieval.search)
       │   new_passages()          passages already shown in this chat dropped
       │   format_passages()       the passages as the text the LLM reads
       │   tool_result()           the message that answers the request, by its id
    {"role": "tool", "tool_call_id": ..., "content": "[passage 22] 1.3.1.3 Tagging, …"}
    + the ids of the passages handed out, e.g. [22, 21]

The caller appends the request, then the result, to the messages, adds the
ids to the ones already shown, and calls the LLM again: it can ask for another
search or answer. gpt-oss on Groq asks for one tool at a time, so "tagging vs
classification" can take two rounds. Repeating the round, at most 3 searches,
is the M2 concept "The agent loop as a graph"; the cited answer, checked
against the ids handed out, is the M2 concept "Answers grounded in several
passages".

**A passage goes to the LLM only once per chat.** Two searches often land on
the same passage ("tagging" and "classification" are neighbours). Every call
resends the whole chat, so a repeat would cost its tokens again on every later
call and tell the LLM nothing new. ``run_search_request()`` can't remember
what it sent, because it forgets everything when it returns, so the caller
passes in the ids already shown.
"""

import sqlite3
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from tutor.llm import UnusableAnswer, function_tool, load_prompt
from tutor.retrieval.search import search

SEARCH_K = 2
"""How many passages one search hands back to the LLM.

Every call resends the whole chat, so the passages from earlier searches are
sent again with each later call. A chapter 1 passage averages about 250 words
(about 330 tokens). With 3 passages a search, a question that uses all 3
searches sends roughly 10K tokens in under a minute, over gpt-oss-20b's free
limit of 8K tokens a minute: the 429 handling in ``tutor.llm`` would make it
wait. With 2, it's about 8K. A typed topic in the study loop still uses
``search()``'s default of 3, since it only needs the best one."""

ASK_REASONING = "low"
"""How hard gpt-oss thinks before each Ask the book call. Choosing a search
needs little, and its thinking tokens count toward the free-tier limits."""

NOTHING_FOUND = "No passage in the book is close enough to this search."
"""The tool's result when every passage scores below ``search()``'s cut-off."""

NOTHING_NEW = "Nothing new: every passage close to this search is already above."
"""The tool's result when the search found passages, but all of them were shown already.

It's kept apart from ``NOTHING_FOUND`` because they mean different things: this
one says the book does cover the search, and the LLM already has the passages."""


class SearchBook(BaseModel):
    """Search the book Dive into Deep Learning for the passages closest in meaning to a topic.

    Returns each passage with its id, section number and title, pages and
    text. Search for one idea at a time; to compare two ideas, search for
    each one separately.
    """

    # Pydantic copies the docstring above into the schema, and function_tool()
    # makes it the tool's description: the LLM reads it to decide whether and
    # when to search. Notes for developers go in comments, like this one.
    # The field's description steers what the LLM searches for.
    query: str = Field(
        description="a few words naming one idea, such as 'tagging' or 'gradient descent'; "
        "not the whole question"
    )


SEARCH_BOOK_TOOL = function_tool(SearchBook, "search_book")
"""The tool definition sent with every Ask the book call, built once from ``SearchBook``."""


def start_messages(question: str) -> list[dict[str, Any]]:
    """Build the first messages of an Ask the book chat: the instructions, then your question.

    Args:
        question: What you typed, e.g. "How is tagging different from classification?".

    Returns:
        ``[{"role": "system", ...}, {"role": "user", ...}]``, the system
        message from ``prompts/ask_book.md``.
    """
    prompt = load_prompt("ask_book")
    return [
        {"role": "system", "content": prompt.text},
        {"role": "user", "content": question},
    ]


def read_search_request(tool_call: dict[str, Any]) -> SearchBook:
    """Check that a tool request asks for ``search_book``, and turn its arguments into a SearchBook.

    The arguments arrive as a JSON string, written by the LLM.
    ``model_validate_json`` parses that string and checks it in one go: a
    missing ``query``, a number instead of text, or text that isn't JSON at
    all is refused, where ``json.loads`` would let a wrong shape through.

    Args:
        tool_call: One entry of a reply's ``tool_calls`` from
            ``complete_with_tools()``.

    Returns:
        The arguments, e.g. ``SearchBook(query="tagging")``.

    Raises:
        UnusableAnswer: If the request names another tool, or its arguments
            don't fit ``SearchBook``. Either way the LLM's reply can't be
            used, like any other broken answer (fail loudly).
    """
    name = tool_call["function"]["name"]
    if name != "search_book":
        raise UnusableAnswer(f"The LLM asked for a tool that doesn't exist: {name!r}.")
    try:
        return SearchBook.model_validate_json(tool_call["function"]["arguments"])
    except ValidationError as error:
        raise UnusableAnswer(f"The LLM's search_book arguments broke a rule: {error}") from error


def new_passages(
    results: list[tuple[float, sqlite3.Row]], shown_ids: list[int]
) -> list[tuple[float, sqlite3.Row]]:
    """Keep the closest passages the LLM hasn't been shown yet, at most ``SEARCH_K`` of them.

    Args:
        results: What ``search()`` returned: (score, passage row) pairs, best first.
        shown_ids: The ids already handed to the LLM in this chat.

    Returns:
        The pairs to hand out, still best first. Empty when every passage in
        ``results`` was shown already, or ``results`` is empty.
    """
    unseen = [(score, passage) for score, passage in results if passage["id"] not in shown_ids]
    return unseen[:SEARCH_K]


def format_passages(results: list[tuple[float, sqlite3.Row]]) -> str:
    """Turn a search's passages into the text the LLM reads as the tool's result.

    Each passage starts with a header line: its id, section and printed pages,
    e.g. ``[passage 22] 1.3.1.3 Tagging, pages 12–13``. The LLM only knows
    what's in this text, so the id must be there for it to say which passage
    it used (the M2 concept "Answers grounded in several passages" checks
    every citation by id). The scores stay out: the LLM doesn't need them.

    Args:
        results: What ``search()`` returned: (score, passage row) pairs, best
            first. A row needs ``id``, ``number``, ``title``, ``page_label``,
            ``end_page_label`` and ``text``.

    Returns:
        The passages, separated by blank lines, or ``NOTHING_FOUND`` when
        there are none.
    """
    if not results:
        return NOTHING_FOUND
    blocks = []
    for _score, passage in results:
        # The printed page numbers, e.g. "page 12", or "pages 12–13" when the
        # passage runs onto the next page (the same pages the study page shows).
        if passage["end_page_label"] == passage["page_label"]:
            pages = f"page {passage['page_label']}"
        else:
            pages = f"pages {passage['page_label']}–{passage['end_page_label']}"
        header = f"[passage {passage['id']}] {passage['number']} {passage['title']}, {pages}"
        blocks.append(f"{header}\n{passage['text']}")
    return "\n\n".join(blocks)


def tool_result(tool_call: dict[str, Any], content: str) -> dict[str, Any]:
    """Build the message that answers one tool request.

    ``tool_call_id`` must be the request's own id: that's how the API, and
    the LLM, tie this result to the request it answers.

    Args:
        tool_call: The request being answered.
        content: The tool's result, as text.

    Returns:
        ``{"role": "tool", "tool_call_id": ..., "name": ..., "content": ...}``.
    """
    return {
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "name": tool_call["function"]["name"],
        "content": content,
    }


def run_search_request(
    conn: sqlite3.Connection, tool_call: dict[str, Any], shown_ids: list[int]
) -> tuple[dict[str, Any], list[int]]:
    """Run the search one tool request asks for, and build the message that answers it.

    The wiring of one tool request: read it, search, drop repeats, format,
    answer. In the M2 concept "The agent loop as a graph" this becomes the
    tool node's work, and the graph's state keeps ``shown_ids``.

    Args:
        conn: An open connection to the database with the embedded passages.
        tool_call: One entry of a reply's ``tool_calls``.
        shown_ids: The ids already handed to the LLM in this chat; ``[]`` for
            the first search. It has no default on purpose: a caller that
            forgot it would quietly send repeats.

    Returns:
        The tool message, ready to append to the messages, and the ids of the
        passages newly handed to the LLM, for the caller to add to
        ``shown_ids``. Empty when nothing was close enough or nothing was new.

    Raises:
        UnusableAnswer: If the request can't be read (``read_search_request()``).
    """
    # 1. The request → its arguments, checked.
    request = read_search_request(tool_call)
    # 2. The real search, the same one a typed topic uses. It asks for one extra
    #    passage per id already shown, so SEARCH_K new ones can remain even if
    #    every shown passage is among the closest.
    results = search(conn, request.query, k=SEARCH_K + len(shown_ids))
    # 3. Passages the LLM already has → dropped.
    passages = new_passages(results, shown_ids)
    # 4. The passages → the text the LLM reads. Found but all repeats is its
    #    own message: format_passages() would say the book has nothing close.
    content = NOTHING_NEW if results and not passages else format_passages(passages)
    # 5. The ids handed out, so a later citation can be checked against them.
    passage_ids = [passage["id"] for _score, passage in passages]
    return tool_result(tool_call, content), passage_ids
