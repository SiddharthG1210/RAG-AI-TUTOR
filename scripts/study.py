"""Study in the terminal: question, your answer, the grade and the passage, then the next turn.

Run from the repo root, after ingesting chapter 1 (it needs the Groq key in ``.env``):

    uv run python scripts/study.py                      # "You choose": the next new passage
    uv run python scripts/study.py --topic "gradient descent"

M1 concepts "The study loop: state, nodes, edges" and "Pausing for your
answer". This is the starter, and the caller of the graph. The graph stops at
each pause (``interrupt()``) and hands back what to show you; this script
shows it, reads what you type, and resumes the graph with it:

    pause "answer": it prints the question and reads your answer
    pause "next":   it prints what the turn shows (the grade and the passage,
                    or a message), then reads the next topic, Enter for
                    "You choose", or s to stop

The browser page (``src/tutor/api/``) is the other caller: it hands out and
reads the same notes, one click at a time.

One run is one study session, with its own id. Unlike the check scripts, it
works on the real ``data/tutor.db``: the questions it writes, your attempts
(with the session id) and the questions' "used" marks are kept.

Each turn usually makes two real LLM calls: one to write a passage's
questions (only when its pool is empty) and one to grade your answer (none
when you send the same answer again). A passage whose questions fail is
written once more, and after a second failure it's skipped and another
passage is picked, so a bad passage adds up to 2 more calls.
"""

import argparse
import sqlite3
import sys
import uuid
from pathlib import Path

from langgraph.types import Command

from tutor.db import connect
from tutor.graph.build import build_graph
from tutor.graph.state import Context
from tutor.llm import FreeLimitReached


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(description="Study in the terminal, one turn after another.")
    parser.add_argument(
        "--topic",
        help='what to study first, e.g. "gradient descent"; the next new passage if not given',
    )
    parser.add_argument("--db", type=Path, default=Path("data/tutor.db"), help="the database")
    return parser.parse_args()


def read_answer(question: str) -> str:
    """Show the question at pause 1 and read your answer, asking again until it isn't empty.

    ``grade_answer()`` refuses an empty answer, so asking again here means an
    empty answer never costs an LLM call.

    Args:
        question: The question's text, from what the pause handed out.

    Returns:
        Your answer, as typed.
    """
    print(f"\nQuestion: {question}")
    answer = input("Your answer: ")
    while not answer.strip():
        answer = input("Please type an answer: ")
    return answer


def print_shown(shown: dict) -> None:
    """At pause 2, print what the turn shows: the grade and the passage, or a message.

    ``shown`` is what the ``show`` or ``not_covered`` node packed (see their
    docstrings in ``tutor/graph/nodes.py`` for its keys). The page shows the
    same dict as boxes; this prints it as plain lines.

    Args:
        shown: ``note["shown"]``, from what the pause handed out.
    """
    print()
    # "no_support" and "not_covered" carry a message instead of a grade.
    if "message" in shown:
        print(shown["message"])
    if shown["kind"] == "graded":
        print(
            f"Verdict: {shown['verdict']}"
            + ("  (reused: you gave this answer before)" if shown["reused"] else "")
        )
        # Each list prints only when it has something in it.
        for label, key in (
            ("Missing", "missing"),
            ("Wrong", "wrong"),
            ("Extras (no penalty)", "extras"),
            ("Based on", "quotes"),
        ):
            if shown[key]:
                print(f"{label}:")
                for item in shown[key]:
                    print(f"  - {item}")
    # "not_covered" has no passage: nothing was found to show.
    if "passage" in shown:
        passage = shown["passage"]
        print(f"\nThe passage: {passage['section']}, page {passage['pages']}\n")
        print(passage["text"])


def read_next() -> dict:
    """At pause 2, read the next topic, "You choose" (Enter), or stop (s).

    Returns:
        The reply ``wait_for_next`` expects: ``{"stop": ..., "topic": ...}``,
        where ``"topic"`` is ``None`` for "You choose".
    """
    line = input('\nNext: type a topic, press Enter for "You choose", or s to stop: ').strip()
    if line.lower() == "s":
        return {"stop": True, "topic": None}
    # An empty line means "You choose"; anything else is a topic to search for.
    return {"stop": False, "topic": line or None}


def run_session(conn: sqlite3.Connection, topic: str | None) -> None:
    """Run one study session: start the graph, then resume it at every pause until it ends.

    Args:
        conn: The open connection to the tutor's database.
        topic: The first turn's topic, or ``None`` for "You choose".
    """
    graph = build_graph()
    # A new id for each run: one run is one session. uuid4() makes a random
    # id that's practically never repeated; .hex is it as 32 letters/digits.
    session_id = uuid.uuid4().hex
    # The thread id tells the checkpointer which paused graph to save and
    # resume. Ours is the session id.
    config = {"configurable": {"thread_id": session_id}}
    # The context isn't saved with the state, so every call hands it over.
    context = Context(conn=conn, session_id=session_id)

    # Start: the graph runs until its first pause (or its end), then returns.
    result = graph.invoke({"topic": topic}, config, context=context)

    while "__interrupt__" in result:
        note = result["__interrupt__"][0].value
        if note["pause"] == "answer":
            reply = read_answer(note["question"])
        else:
            print_shown(note["shown"])
            reply = read_next()
        result = graph.invoke(Command(resume=reply), config, context=context)


def main() -> None:
    """Run one study session, and print "free limit reached" if Groq's free tier runs out."""
    # Windows terminals sometimes can't print curly quotes in their default encoding.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    # connect() would create an empty database, which would only fail later
    # with a confusing message, so check that chapter 1 was ingested first.
    if not args.db.exists():
        sys.exit(f"No database at {args.db}. Ingest a chapter first: scripts/ingest.py --chapter 1")

    conn = connect(args.db)
    try:
        run_session(conn, args.topic)
    except FreeLimitReached:
        print("\nFree limit reached: Groq's free tier is used up for now. Try again later.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
