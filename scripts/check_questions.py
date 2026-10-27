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

It also shows what the wording check did (PLAN.md › How the tutor behaves in
specific cases › Questions never mention the passage): how many questions
mentioned the passage, the text or the author as the LLM wrote them, how many
had a phrase like "according to the passage" cut, and how many still mentioned
it after the cut, and so were dropped. The wording isn't one of M1's "Done
when" goals, so it doesn't change the exit code.

The copy is deleted at the end, so the real ``data/tutor.db`` keeps its pools
exactly as they were. Run it again after any change to the prompt, the schema,
the model or the quote check, and compare. To try another model without
editing ``.env``:

    uv run python scripts/check_questions.py --model qwen/qwen3.8-27b
"""

import argparse
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import openai
from pydantic import ValidationError

from tutor import questions
from tutor.config import get_settings
from tutor.db import connect, list_passages
from tutor.llm import complete_structured
from tutor.questions import (
    QuestionSet,
    cut_source_phrase,
    mentions_the_source,
    write_questions,
)
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
    parser.add_argument(
        "--model", help="the question model to try, e.g. qwen/qwen3.8-27b; .env's if not given"
    )
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
    if args.model:
        # pydantic-settings reads environment variables before .env, so this
        # overrides QUESTION_MODEL for this run only. It must happen before
        # anything calls get_settings(), which reads the settings once.
        os.environ["QUESTION_MODEL"] = args.model

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
            print(f"Model: {get_settings().question_model}\n")
            written = 0
            mentioned = 0  # questions that mentioned the passage as the LLM wrote them
            cut_count = 0  # questions whose "according to the passage" was cut
            still_mentioned = 0  # questions that still mentioned it after the cut: dropped
            failed_calls = 0
            # (quote, its passage's text) for every quote that passed the check
            kept_quotes: list[tuple[str, str]] = []

            for i, passage in enumerate(picked):
                if i > 0:
                    time.sleep(PAUSE_SECONDS)  # stay under the tokens-per-minute limit
                print(f"## {passage['number']} {passage['title']}, passage {passage['position']}\n")
                try:
                    write_questions(conn, passage)
                except (ValidationError, openai.BadRequestError) as error:
                    # The answer broke the schema: e.g. 1 question instead of
                    # exactly 3 (Pydantic), or Groq's json_validate_failed on
                    # both tries. write_questions() fails loudly, as it should;
                    # the script notes it and goes on with the next passage.
                    failed_calls += 1
                    first_line = str(error).splitlines()[0]
                    print(f"  CALL FAILED: {type(error).__name__}: {first_line}\n")
                    continue
                for q in answers[-1].questions:
                    written += 1
                    # The two checks write_questions() ran, repeated here one
                    # at a time, so each count shows which check did what.
                    # The quote check alone decides M1's pass rate.
                    quote_ok = quote_in_passage(q.quote, passage["text"])
                    if quote_ok:
                        kept_quotes.append((q.quote, passage["text"]))
                    cut = cut_source_phrase(q.question)
                    notes = [] if quote_ok else ["quote not in the passage"]
                    if mentions_the_source(q.question):
                        mentioned += 1
                    if cut != q.question:
                        cut_count += 1
                        notes.append(f"saved as: {cut}")
                    if mentions_the_source(cut):
                        still_mentioned += 1
                        notes.append("still mentions the passage")
                    mark = "kept   " if quote_ok and not mentions_the_source(cut) else "DROPPED"
                    print(f"  {mark} Q: {q.question}")
                    for note in notes:
                        print(f"            ({note})")
                    for point in q.key_points:
                        print(f"            - {point}")
                    print(f"            quote: {q.quote!r}\n")

            passed = len(kept_quotes)
            print(f"Calls whose answer broke the schema: {failed_calls} of {len(picked)}.")
            print(
                f"Quote check: {passed} of {written} passed ({passed / written:.0%}; M1 needs 90%)."
            )

            print(
                f"Wording check: {mentioned} of {written} mentioned the passage, the text or "
                f"the author; {cut_count} had the phrase cut, {still_mentioned} still mentioned "
                f"it and were dropped."
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
