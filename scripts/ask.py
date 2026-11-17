"""Ask the book in the terminal: one question, every step of the agent loop, and the answer.

Run from the repo root, after ingesting chapter 1 (it needs the Groq key in ``.env``):

    uv run python scripts/ask.py "How is tagging different from classification?"

M2 concept "The agent loop as a graph". This is the starter for Ask the book's
loop (``tutor/graph/agent.py``): it runs the loop on your question, then
prints what happened, step by step, from the finished chat:

    1. LLM → search_book "tagging"
       search → [passage 23] 1.3.1.3 Tagging, pages 11–12
                [passage 16] 1.3.1 Supervised Learning, pages 7–8
    2. LLM → answers in text
    ...
    LLM calls: 2 · tokens in 1678 · out 519

It works on the real ``data/tutor.db``, and only writes the token log. Each
question makes up to 3 real LLM calls: one before each search, and none after
the third. The cited answer, with every quote checked, comes in the M2
concept "Answers grounded in several passages".
"""

import argparse
import json
import sqlite3
import sys
import uuid
from pathlib import Path

from tutor.db import connect
from tutor.graph.agent import AgentState, ask_the_book, build_agent_graph, final_text
from tutor.graph.state import Context
from tutor.llm import FreeLimitReached, UnusableAnswer


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(description="Ask the book one question, in the terminal.")
    parser.add_argument("question", help='your question, e.g. "What is tagging?"')
    parser.add_argument("--db", type=Path, default=Path("data/tutor.db"), help="the database")
    return parser.parse_args()


def last_call_id(conn: sqlite3.Connection) -> int:
    """Where the token log ends right now, so the calls of this run can be counted afterwards.

    Args:
        conn: An open connection.

    Returns:
        The highest row id in ``llm_calls``, or 0 when it's empty.
    """
    return conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM llm_calls").fetchone()[0]


def calls_since(conn: sqlite3.Connection, after_id: int) -> tuple[int, int, int]:
    """Count the LLM calls and tokens logged after a point in the token log.

    Args:
        conn: An open connection.
        after_id: A row id from ``last_call_id()``.

    Returns:
        (calls, prompt tokens, completion tokens).
    """
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0) "
        "FROM llm_calls WHERE rowid > ?",
        (after_id,),
    ).fetchone()
    return row[0], row[1], row[2]


def print_steps(state: AgentState) -> None:
    """Print the chat after the first two messages: each LLM move, and what each search found.

    Args:
        state: The final state from ``ask_the_book()``.
    """
    step = 0
    for message in state["messages"][2:]:
        if message["role"] == "assistant" and "tool_calls" in message:
            step += 1
            for call in message["tool_calls"]:
                query = json.loads(call["function"]["arguments"]).get("query")
                print(f"{step}. LLM → {call['function']['name']} {query!r}")
        elif message["role"] == "assistant":
            step += 1
            print(f"{step}. LLM → answers in text")
        else:
            # A search result: its header lines name the passages; otherwise
            # it's a one-line message such as "nothing close enough".
            headers = [line for line in message["content"].splitlines() if line.startswith("[")]
            lines = headers or [message["content"]]
            print(f"   search → {lines[0]}")
            for line in lines[1:]:
                print(f"            {line}")


def main() -> None:
    """Ask the book one question, print every step and the answer, then the calls and tokens."""
    # Windows terminals sometimes can't print "→" or curly quotes in their default encoding.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    # connect() would create an empty database, so check that a chapter was ingested first.
    if not args.db.exists():
        sys.exit(f"No database at {args.db}. Ingest a chapter first: scripts/ingest.py --chapter 1")

    conn = connect(args.db)
    try:
        before = last_call_id(conn)
        context = Context(conn=conn, session_id=uuid.uuid4().hex)
        print(f"Question: {args.question}\n")
        state = ask_the_book(build_agent_graph(), context, args.question)
        print_steps(state)
        answer = final_text(state)
        if answer is None:
            print("\nNo answer: the 3 searches are used up. Writing the answer from them comes in")
            print('the M2 concept "Answers grounded in several passages".')
        else:
            print(f"\nAnswer:\n{answer}")
        calls, tokens_in, tokens_out = calls_since(conn, before)
        print(f"\nLLM calls: {calls} · tokens in {tokens_in} · out {tokens_out}")
        print(f"Passages shown: {state['shown_ids']}")
    except FreeLimitReached:
        print("\nFree limit reached: Groq's free tier is used up for now. Try again later.")
    except UnusableAnswer as error:
        print(f"\nThe LLM's reply couldn't be used: {error}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
