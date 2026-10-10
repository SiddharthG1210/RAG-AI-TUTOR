"""Check how well search finds chapter 1's passages, and what the cut-off lets through.

Run from the repo root, after ingesting and embedding chapter 1:

    uv run python scripts/check_retrieval.py

M1 concept "Embeddings and vector search". It runs four kinds of queries
through ``search()`` and prints, for each, the best scores and what the tutor
would do with the cut-off (``CUTOFF`` in ``tutor.retrieval.search``):

- **Test queries:** 10 topics typed in a learner's own words, each with the
  topic that should answer it. M1 is done when, for 8 of the 10, that topic is
  in the top 5 (PLAN.md › Roadmap › M1 › Done when).
- **Off-topic queries,** such as "pizza recipes": the book doesn't cover them,
  so the tutor should say "the material doesn't cover this". M1 needs this for
  "pizza recipes".
- **Computer topics the book doesn't cover,** such as "learning javascript":
  they share the book's vocabulary and score like real topics, so the cut-off
  lets them through. A known limit, shown so it stays visible.
- **Topics chapter 1 only mentions** (attention, GANs, GPUs): the chapter names
  them without explaining them. A cut-off measures closeness, not coverage, so
  these get passages that only name them (PLAN.md › How the tutor behaves in
  specific cases › Topic search).

Run it again after any change to the header, the query prefix, the model, the
chunking or the cut-off, and compare.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

from tutor.db import connect
from tutor.retrieval.search import CUTOFF, search

TOP = 5
"""How deep to look: the right topic must be in the top 5 (PLAN.md › M1 › Done when)."""

NO_CUTOFF = -1.0
"""The lowest cosine there is: passed as the cut-off to see every score."""

TEST_QUERIES = [
    ("gradient descent", "1.2.4"),
    ("what a loss function measures", "1.2.3"),
    ("predicting a house's sale price", "1.3.1.1"),
    ("is this image a cat or a dog", "1.3.1.2"),
    ("assigning several labels to one image", "1.3.1.3"),
    ("ranking web pages for a search query", "1.3.1.4"),
    ("suggesting products a customer might like", "1.3.1.5"),
    ("translating sentences from one language to another", "1.3.1.6"),
    ("learning from data without any labels", "1.3.2"),
    ("an agent learning from rewards", "1.3.4"),
]
"""Each query, and the topic number whose passages should answer it."""

OFF_TOPIC = ["pizza recipes", "how to fix a flat bike tyre", "the history of the Roman empire"]
"""Queries the book doesn't cover at all."""

COMPUTER_OFF_TOPIC = ["learning javascript", "python list comprehension", "computer viruses"]
"""Computer topics the book doesn't cover, which share its vocabulary."""

MENTIONED_ONLY = ["attention mechanisms", "generative adversarial networks", "training on GPUs"]
"""Topics chapter 1 names without explaining."""


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(description="Check search on chapter 1.")
    parser.add_argument("--db", type=Path, default=Path("data/tutor.db"), help="the database")
    return parser.parse_args()


def topics(results: list[tuple[float, sqlite3.Row]]) -> str:
    """Format the results' topics and scores, e.g. "1.2.4 0.783, 1.7 0.740".

    Args:
        results: ``search()``'s (score, passage) pairs.

    Returns:
        One short line.
    """
    return ", ".join(f"{passage['number']} {score:.3f}" for score, passage in results)


def tutor_says(conn: sqlite3.Connection, query: str) -> str:
    """Describe what the tutor does with a query, using the real cut-off.

    Args:
        conn: An open connection.
        query: The query.

    Returns:
        'says "the material doesn't cover this"', or how many passages it shows.
    """
    results = search(conn, query, k=3)
    return (
        'says "the material doesn\'t cover this"'
        if not results
        else f"shows {len(results)} passages"
    )


def main() -> None:
    """Run every query and print the ranks, the scores and what the cut-off does."""
    sys.stdout.reconfigure(encoding="utf-8")
    conn = connect(parse_args().db)
    print(f"Cut-off: {CUTOFF}\n")

    print(f"Test queries: is the right topic in the top {TOP}?\n")
    hits = 0
    for query, expected in TEST_QUERIES:
        results = search(conn, query, k=TOP, cutoff=NO_CUTOFF)
        numbers = [passage["number"] for _, passage in results]
        # The rank (1 = top) of the first passage from the expected topic, if any.
        rank = numbers.index(expected) + 1 if expected in numbers else None
        hits += rank is not None
        found = f"rank {rank}" if rank else "MISSED"
        print(f"  {query!r} → {expected}: {found}; the tutor {tutor_says(conn, query)}")
        print(f"      {topics(results)}")
    print(f"\n  {hits} of {len(TEST_QUERIES)} found in the top {TOP} (M1 needs 8).\n")

    for heading, queries in [
        ("Off-topic queries: should get 'doesn't cover this'", OFF_TOPIC),
        ("Computer topics the book doesn't cover: a known limit", COMPUTER_OFF_TOPIC),
        ("Topics chapter 1 only mentions", MENTIONED_ONLY),
    ]:
        print(f"{heading}\n")
        for query in queries:
            results = search(conn, query, k=3, cutoff=NO_CUTOFF)
            print(f"  {query!r}: the tutor {tutor_says(conn, query)}")
            print(f"      {topics(results)}")
        print()
    conn.close()


if __name__ == "__main__":
    main()
