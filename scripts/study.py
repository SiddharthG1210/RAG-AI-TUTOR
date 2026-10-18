"""Study one turn in the terminal: a question, your typed answer, the grade and the passage.

Run from the repo root, after ingesting chapter 1 (it needs the Groq key in ``.env``):

    uv run python scripts/study.py                      # "You choose": the next new passage
    uv run python scripts/study.py --topic "gradient descent"

M1 concept "The study loop: state, nodes, edges". This is the starter: it
opens the database, builds the graph and runs it once, from START to END.
Unlike the check scripts, it works on the real ``data/tutor.db``: the
questions it writes, your attempt and the question's "used" mark are kept.

It makes up to two real LLM calls: one to write a passage's questions (only
when its pool is empty) and one to grade your answer (none when you send the
same answer again).
"""

import argparse
import sys
from pathlib import Path

from tutor.db import connect
from tutor.graph.build import build_graph
from tutor.graph.nodes import TurnStopped
from tutor.graph.state import Context
from tutor.llm import FreeLimitReached


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(description="Study one turn in the terminal.")
    parser.add_argument(
        "--topic", help='what to study, e.g. "gradient descent"; the next new passage if not given'
    )
    parser.add_argument("--db", type=Path, default=Path("data/tutor.db"), help="the database")
    return parser.parse_args()


def main() -> None:
    """Run one study turn, and print the message of a turn that has to stop."""
    # Windows terminals sometimes can't print curly quotes in their default encoding.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    # connect() would create an empty database, which would only fail later
    # with a confusing message, so check that chapter 1 was ingested first.
    if not args.db.exists():
        sys.exit(f"No database at {args.db}. Ingest a chapter first: scripts/ingest.py --chapter 1")

    conn = connect(args.db)
    try:
        graph = build_graph()
        # The state starts as this dict. The connection goes in the context,
        # not the state: every node reads it as runtime.context.conn.
        graph.invoke({"topic": args.topic}, context=Context(conn=conn))
    except TurnStopped as stopped:
        # A case with no route yet (see tutor.graph.nodes): show its message.
        print(f"\n{stopped}")
    except FreeLimitReached:
        print("\nFree limit reached: Groq's free tier is used up for now. Try again later.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
