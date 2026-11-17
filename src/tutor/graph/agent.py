"""Ask the book's agent loop: the LLM picks each next step, and code runs it and stops it.

M2 concept "The agent loop as a graph". The M2 concept "Tool calling" built
the pieces of one round (``tutor.ask``): the first messages, one LLM call
with the ``search_book`` tool, and running one search the LLM ordered. This
file runs those pieces in order and repeats them, because nobody knows in
advance how many searches a question needs: only the LLM finds out, as it
reads the passages.

It's a LangGraph graph like the study loop (``build.py``), with two nodes and
two forks:

    START → call_llm ─┬─ an order ──► run_search ─┬─ under 3 searches ──► call_llm
                      └─ text ──────► END         └─ 3 searches done ───► END

- ``call_llm`` sends the whole chat, with the tool, and adds the LLM's reply.
- ``run_search`` runs the search the reply ordered and adds the passages.
- The two routing functions read the state and name the next node.

**This is what makes it an agent.** In the study loop, code chooses every
step. Here the LLM chooses the next one (search again, or answer), and code
only runs what it ordered and enforces the limit of 3 searches (PLAN.md ›
Design rules: Ask the book is the one agent, and it's fenced in). After the
third search, code stops without asking the LLM again, so a question never
costs more than 3 calls here. The M2 concept "Answers grounded in several
passages" replaces both ``END``s with the node that writes the cited answer,
the 4th and last call.

**The chat grows by every step and is resent in full.** The LLM remembers
nothing between calls, so each call rereads ``messages`` from the top. That's
why a search hands back 2 passages and never repeats one (``tutor.ask``), and
why the limit matters for the free tier as well as for safety.
"""

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from tutor.ask import (
    ASK_REASONING,
    SEARCH_BOOK_TOOL,
    run_search_request,
    start_messages,
    tool_result,
)
from tutor.config import get_settings
from tutor.graph.state import Context
from tutor.llm import complete_with_tools

MAX_SEARCHES = 3
"""The most searches one question may run (PLAN.md › Decisions › Ask the book)."""

LIMIT_REACHED = "Not run: the limit of 3 searches for one question is used up."
"""The result sent for an order the limit doesn't allow.

The API refuses a chat where an order has no result after it, so even an
order that isn't run gets an answer. gpt-oss on Groq orders one search at a
time, so this only matters if a reply ever holds more than one order."""


class AgentState(TypedDict):
    """The keys one Ask the book chat carries from node to node.

    A ``TypedDict`` is a plain dict at run time; the class only lists the keys
    and their types. Each node returns the keys it changed, and LangGraph
    replaces those keys in the state (the same merge as the study loop).
    """

    # The chat so far, as plain dicts: the rules, your question, then every
    # reply from the LLM and every search result, in order. start_messages()
    # makes the first two; call_llm and run_search each add to it. It's the
    # LLM's whole memory: every call sends all of it.
    messages: list[dict[str, Any]]
    # The ids of every passage handed to the LLM in this chat (run_search →
    # run_search, so no passage is sent twice; later the citation check in
    # "Answers grounded in several passages").
    shown_ids: list[int]
    # How many searches have run (run_search → route_after_search).
    searches_done: int


def call_llm(state: AgentState, runtime: Runtime[Context]) -> dict:
    """Ask the LLM for its next move, with the ``search_book`` tool, and add its reply to the chat.

    Args:
        state: Reads ``messages``.
        runtime: Gives the database connection, for the token log.

    Returns:
        ``{"messages": ...}``: the chat with the LLM's reply added. The reply
        is either an order (it has a ``"tool_calls"`` key) or a text answer.
    """
    reply = complete_with_tools(
        runtime.context.conn,
        state["messages"],
        [SEARCH_BOOK_TOOL],
        model=get_settings().ask_model,
        reasoning_effort=ASK_REASONING,
        purpose="ask",
    )
    # A new list, not .append(): a node returns its changes instead of
    # editing the state it was handed.
    return {"messages": [*state["messages"], reply]}


def run_search(state: AgentState, runtime: Runtime[Context]) -> dict:
    """Run the search the LLM's last reply ordered, and add the result to the chat.

    Every order gets a result, because the API refuses a chat where an order
    is left unanswered. An order past ``MAX_SEARCHES`` isn't run: its result
    says so (``LIMIT_REACHED``).

    Args:
        state: Reads ``messages`` (the last one is the LLM's order),
            ``shown_ids`` and ``searches_done``.
        runtime: Gives the database connection, for the search.

    Returns:
        ``messages`` with a result per order, ``shown_ids`` with the newly
        shown passages, and ``searches_done`` plus the searches that ran.
    """
    messages = list(state["messages"])
    shown_ids = list(state["shown_ids"])
    searches_done = state["searches_done"]
    for tool_call in state["messages"][-1]["tool_calls"]:
        if searches_done < MAX_SEARCHES:
            tool_message, new_ids = run_search_request(runtime.context.conn, tool_call, shown_ids)
            shown_ids += new_ids
            searches_done += 1
        else:
            tool_message = tool_result(tool_call, LIMIT_REACHED)
        messages.append(tool_message)
    return {"messages": messages, "shown_ids": shown_ids, "searches_done": searches_done}


def route_after_llm(state: AgentState) -> str:
    """Choose what follows ``call_llm``: run the search it ordered, or finish.

    Args:
        state: The state just after ``call_llm``; reads the LLM's reply, the
            last message in ``messages``.

    Returns:
        ``"run_search"`` when the reply is an order, or ``END`` when it's a
        text answer.
    """
    # The LLM's reply is the last message in the chat.
    reply = state["messages"][-1]
    # 1. The reply has a "tool_calls" key: it's an order → run the search.
    if "tool_calls" in reply:
        choice = "run_search"
    # 2. Otherwise it's a text answer → END.
    else:
        choice = END
    return choice


def route_after_search(state: AgentState) -> str:
    """Choose what follows ``run_search``: ask the LLM again, or stop at the limit.

    Args:
        state: The state just after ``run_search``; reads ``searches_done``.

    Returns:
        ``"call_llm"`` while fewer than ``MAX_SEARCHES`` searches have run,
        or ``END`` once the limit is reached.
    """
    # 1. Fewer than MAX_SEARCHES searches done → ask the LLM again: it reads
    #    the new passages and picks its next move.
    if state["searches_done"] < MAX_SEARCHES:
        choice = "call_llm"
    # 2. The limit is reached → END, without asking the LLM again, so a
    #    question never costs a 4th call here.
    else:
        choice = END
    return choice


def build_agent_graph() -> CompiledStateGraph:
    """Build Ask the book's loop: add its two nodes, draw its edge and two forks, and compile it.

    No checkpointer: the loop never pauses for you, so ``invoke()`` runs it
    from start to end in one go.

    Returns:
        The compiled graph. Run it with ``ask_the_book()``.
    """
    builder = StateGraph(AgentState, context_schema=Context)
    builder.add_node("call_llm", call_llm)
    builder.add_node("run_search", run_search)
    builder.add_edge(START, "call_llm")
    builder.add_conditional_edges("call_llm", route_after_llm)
    builder.add_conditional_edges("run_search", route_after_search)
    return builder.compile()


def ask_the_book(graph: CompiledStateGraph, context: Context, question: str) -> AgentState:
    """Run the agent loop on one question, from your question to the LLM's last move.

    Args:
        graph: The compiled loop, from ``build_agent_graph()``.
        context: The database connection and the session id.
        question: What you typed, e.g. "How is tagging different from classification?".

    Returns:
        The final state: the whole chat, the ids shown and the searches run.
    """
    start = {"messages": start_messages(question), "shown_ids": [], "searches_done": 0}
    return graph.invoke(start, context=context)


def final_text(state: AgentState) -> str | None:
    """Give the LLM's text answer, if the chat ended with one.

    Args:
        state: A final state from ``ask_the_book()``.

    Returns:
        The answer's text, or ``None`` when the loop stopped at the search
        limit (the last message is then a search result). Until the M2
        concept "Answers grounded in several passages", that case has no
        answer.
    """
    last = state["messages"][-1]
    if last["role"] == "assistant" and "tool_calls" not in last:
        return last["content"]
    return None
