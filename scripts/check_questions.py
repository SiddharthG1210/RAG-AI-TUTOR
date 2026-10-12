"""Check the question writer on 10 real passages, without touching the real question pools.

Run from the repo root, after ingesting chapter 1 (it needs the Groq key in ``.env``):

    uv run python scripts/check_questions.py

M1 concept "Structured output (question writing)". It copies the database to
a temporary file, runs ``write_questions()`` on 10 passages spread across the
chapter (10 real LLM calls), and prints every question the LLM wrote, with its
key points and its quote, marked kept or dropped. Then it checks M1's "Done
when" (PLAN.md › Roadmap › M1):

- **At least 90% of the questions pass the quote check.** Each dropped quote
  is printed, so you can tell an LLM that reworded the passage from a gap in
  ``normalize()``.
- **A made-up quote is rejected,** checked against every picked passage.
- **A real quote with one word changed is rejected,** made from every kept
  quote by adding a "not" after its first verb like "is" or "can", or, when it
  has none, by swapping its longest word.

The copy is deleted at the end, so the real ``data/tutor.db`` keeps its pools
exactly as they were. Run it again after any change to the prompt, the schema,
the model or the quote check, and compare.
"""

import argparse
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from tutor import questions
from tutor.db import connect, list_passages
from tutor.llm import complete_structured
from tutor.questions import QuestionSet, write_questions
from tutor.quotes import quote_in_passage

PASSAGES = 10
"""How many passages to write questions for (PLAN.md › M1 › Done when)."""

PASS_RATE = 0.9
"""The share of questions whose quote must pass the check."""

MADE_UP = "Deep learning models never need any data to learn from."
"""A quote no passage in the book contains, though it sounds like the book."""

PAUSE_SECONDS = 20.0
"""The wait between calls. Groq's free tier allows gpt-oss-20b 8,000 tokens per
minute, and one call uses about 1,700, so back-to-back calls hit the limit
after four or five; one call every 20 seconds stays under it."""

AUXILIARIES = {"is", "are", "was", "were", "can", "could", "will", "would", "should", "does", "do"}
"""Verbs a "not" can follow, turning a sentence into its opposite."""


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(description="Check the question writer on real passages.")
    parser.add_argument("--db", type=Path, default=Path("data/tutor.db"), help="the database")
    return parser.parse_args()


def spread(passages: list[sqlite3.Row], count: int) -> list[sqlite3.Row]:
    """Pick ``count`` passages spread evenly across the chapter, the same ones on every run.

    Args:
        passages: Every passage, in book order.
        count: How many to pick.

    Returns:
        The picked passages, in book order.
    """
    # Every (len / count)-th passage: with 49 passages and 10 wanted, about
    # every 5th one, from the introduction to the last section.
    step = len(passages) / count
    return [passages[int(i * step)] for i in range(count)]


def change_one_word(quote: str) -> str:
    """Make a real quote say something else, by changing a single word.

    Args:
        quote: A quote that's really in its passage.

    Returns:
        The same quote with a "not" added after its first verb like "is" or
        "can" (it now says the opposite), or, if it has none, with its longest
        word swapped for "nothing".
    """
    words = quote.split(" ")
    for i, word in enumerate(words):
        if word.lower() in AUXILIARIES:
            return " ".join([*words[: i + 1], "not", *words[i + 1 :]])
    # No such verb: swap the longest word, which carries the most meaning.
    longest = max(range(len(words)), key=lambda i: len(words[i]))
    words[longest] = "nothing"
    return " ".join(words)


def main() -> None:
    """Write questions for 10 passages on a copy of the database, and check M1's goals."""
    # Windows terminals sometimes can't print curly quotes in their default encoding.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    # write_questions() returns only the kept questions. To print the dropped
    # ones too, the script swaps the LLM call inside tutor.questions for a
    # version that also keeps a copy of each full answer. write_questions()
    # looks the name up in its module every time it runs, so it uses this one.
    answers: list[QuestionSet] = []

    def recording_call(*call_args: Any, **call_kwargs: Any) -> QuestionSet:
        """The real LLM call, plus a copy of its answer in ``answers``."""
        answer = complete_structured(*call_args, **call_kwargs)
        answers.append(answer)
        return answer

    questions.complete_structured = recording_call

    # A temporary folder for the copy. ignore_cleanup_errors: on Windows an
    # open file can't be deleted, and a crash mustn't hide the real error.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        copy = Path(folder) / "tutor.db"
        # copy2 copies the file byte for byte; a missing database fails loudly here.
        shutil.copy2(args.db, copy)
        conn = connect(copy)
        try:
            # The token log's last id before this run, so the totals count only its calls.
            before = conn.execute("SELECT COALESCE(MAX(id), 0) FROM llm_calls").fetchone()[0]
            picked = spread(list_passages(conn), PASSAGES)
            written = 0
            kept_quotes: list[tuple[str, str]] = []  # (quote, its passage's text)

            for i, passage in enumerate(picked):
                if i > 0:
                    time.sleep(PAUSE_SECONDS)  # stay under the tokens-per-minute limit
                print(f"## {passage['number']} {passage['title']}, passage {passage['position']}\n")
                kept = write_questions(conn, passage)
                kept_quotes += [(q.quote, passage["text"]) for q in kept]
                for q in answers[-1].questions:
                    written += 1
                    mark = "kept   " if q in kept else "DROPPED"
                    print(f"  {mark} Q: {q.question}")
                    for point in q.key_points:
                        print(f"            - {point}")
                    print(f"            quote: {q.quote!r}\n")

            passed = len(kept_quotes)
            print(
                f"Quote check: {passed} of {written} passed ({passed / written:.0%}; M1 needs 90%)."
            )

            made_up_rejected = not any(quote_in_passage(MADE_UP, p["text"]) for p in picked)
            print(f"Made-up quote rejected by all {len(picked)} passages: {made_up_rejected}")

            changed = [(change_one_word(q), text) for q, text in kept_quotes]
            caught = sum(not quote_in_passage(q, text) for q, text in changed)
            print(f"Kept quotes with one word changed, rejected: {caught} of {len(changed)}")

            calls, tokens, seconds = conn.execute(
                "SELECT COUNT(*), SUM(total_tokens), SUM(seconds) FROM llm_calls WHERE id > ?",
                (before,),
            ).fetchone()
            print(f"LLM calls: {calls}, {tokens} tokens, {seconds:.1f} seconds in all.")

            ok = passed / written >= PASS_RATE and made_up_rejected and caught == len(changed)
            print(f"\nM1's quote-check goals: {'met' if ok else 'NOT met'}.")
        finally:
            conn.close()
    # A non-zero exit code tells a shell (or CI) that the check failed.
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
