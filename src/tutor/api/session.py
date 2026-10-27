"""Drive one study session from a web page: start the graph, then resume it one click at a time.

M1 concept "API and UI". In the terminal, ``scripts/study.py`` drives a
session with one loop: invoke the graph, read the pause note, ask you,
resume, until the graph ends. A web page can't keep a loop running while you
think: every click is a separate web request, which must finish and send its
answer back. So the loop is cut into its passes, one per click:

    Start  → start_session(topic)                              → the first note
    Submit → resume_session(id, answer)                        → the next note
    Next   → resume_session(id, {"stop": False, "topic": ...}) → the next note
    Stop   → resume_session(id, {"stop": True, "topic": None}) → None: it ended

What ``input()`` did inside the loop (waiting for you) now happens between
clicks, with the graph paused in the checkpointer under the session id
(LangGraph's thread id). The page keeps that id and sends it with every click.

**A connection per call.** Each call opens its own database connection and
closes it before returning. Gradio runs each click's function in a worker
thread, and a ``sqlite3`` connection refuses to be used from any thread but
the one that opened it. A fresh connection per click works because the state
holds ids, not rows (M1 concept "The study loop: state, nodes, edges"), and
because the context is handed over again with every call anyway (LangGraph
never saves it).
"""

import uuid
from pathlib import Path
from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from tutor.db import connect
from tutor.graph.state import Context


class SessionLost(Exception):
    """The page asked to resume a session the server doesn't have.

    The checkpointer lives in memory until M6, so a server restart forgets
    every paused session, while the browser tab still holds the old id.
    """


def start_session(graph: CompiledStateGraph, db_path: Path, topic: str | None) -> tuple[str, dict]:
    """Start a new session: make its id, and run the graph to its first pause.

    Args:
        graph: The compiled study graph, built once for the whole server.
        db_path: The tutor's database file.
        topic: The first turn's topic, or ``None`` for "You choose".

    Returns:
        The new session id, and the first pause's note: ``{"pause":
        "answer", ...}`` with the question, or ``{"pause": "next", ...}`` when
        the topic wasn't covered.

    Raises:
        RuntimeError: If the graph ended without pausing, which the graph's
            wiring never allows.
        FreeLimitReached: From the LLM wrapper, when Groq's free tier runs out.
    """
    # uuid4() makes a random id that's practically never repeated; .hex is it
    # as 32 letters and digits, e.g. "3f9a…".
    session_id = uuid.uuid4().hex
    note = _run(graph, db_path, session_id, {"topic": topic})
    if note is None:
        raise RuntimeError("The study graph ended before its first pause.")
    return session_id, note


def resume_session(
    graph: CompiledStateGraph, db_path: Path, session_id: str, reply: Any
) -> dict | None:
    """Resume a paused session with your reply, and run the graph to its next pause or its end.

    Args:
        graph: The compiled study graph, built once for the whole server.
        db_path: The tutor's database file.
        session_id: The id ``start_session()`` gave, sent back by the page.
        reply: What the paused node's ``interrupt()`` returns: your answer
            (text) at the "answer" pause, ``{"stop": ..., "topic": ...}`` at
            the "next" pause.

    Returns:
        The next pause's note, or ``None`` when the session ended (you stopped).

    Raises:
        SessionLost: If the checkpointer has no paused session with that id.
        FreeLimitReached: From the LLM wrapper, when Groq's free tier runs out.
    """
    # get_state() reads what the checkpointer saved for this thread id.
    # .next names the node waiting to run (the paused one, e.g.
    # ("ask_question",)); it's empty () for an id the checkpointer has never
    # seen, or a session that already ended.
    if not graph.get_state(_config(session_id)).next:
        raise SessionLost(f"No paused study session has the id {session_id!r}.")
    # Command(resume=reply) resumes the paused node: its interrupt() returns reply.
    return _run(graph, db_path, session_id, Command(resume=reply))


def _config(session_id: str) -> dict:
    """LangGraph's config for one session: the thread id the checkpointer saves it under.

    Args:
        session_id: The session's id, which is also its thread id.

    Returns:
        ``{"configurable": {"thread_id": session_id}}``.
    """
    return {"configurable": {"thread_id": session_id}}


def _run(
    graph: CompiledStateGraph, db_path: Path, session_id: str, graph_input: Any
) -> dict | None:
    """Run the graph once, on a connection of its own, and return the pause note it stops at.

    One pass of the terminal's loop: ``invoke()`` runs the graph until it
    pauses or ends, then returns.

    Args:
        graph: The compiled study graph.
        db_path: The tutor's database file.
        session_id: The session's id: the thread id, and part of the context.
        graph_input: ``{"topic": ...}`` to start, or ``Command(resume=...)`` to resume.

    Returns:
        The note the pausing node handed to ``interrupt()``, or ``None`` when
        the graph ran to its end.
    """
    conn = connect(db_path)
    try:
        result = graph.invoke(
            graph_input,
            _config(session_id),
            context=Context(conn=conn, session_id=session_id),
        )
    finally:
        # Closed even when the run raises (e.g. FreeLimitReached), so no
        # connection is left open for this click's worker thread.
        conn.close()
    # A paused graph's result holds "__interrupt__": a list with one Interrupt
    # per pausing node (ours pause one at a time), whose .value is the note.
    if "__interrupt__" in result:
        return result["__interrupt__"][0].value
    return None
