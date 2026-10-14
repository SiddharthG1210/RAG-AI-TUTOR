"""Re-grade the regression set and list every grade that isn't the one it should get.

Run from the repo root, after ingesting chapter 1 (it needs the Groq key in ``.env``):

    uv run python scripts/regrade.py

M1 concept "Grading grounded in the passage". The regression set is a fixed
list of questions, each with test answers and the grade each answer should
get. Run this after any change to ``prompts/grader.md``, the ``Grade`` form,
the grading model or its settings, and read the summary at the end. Like
``check_questions.py``, it works on a copy of the database, so the real
``data/tutor.db`` never sees the test questions or their attempts. On the copy,
every answer is graded fresh, because the copy has no earlier attempts.

It also checks the M1 "Done when" line for the same-answer check (PLAN.md ›
Roadmap › M1): the first answer sent again (in capitals, with extra spaces)
reuses its grade with no LLM call, while the same answer with one word
changed is graded fresh. The token log proves both.

It makes one real LLM call per test answer, plus one for the changed word.
To try another grading model without editing ``.env``:

    uv run python scripts/regrade.py --model openai/gpt-oss-20b
"""

import argparse
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from tutor.config import get_settings
from tutor.db import connect, list_passages, list_questions, save_questions
from tutor.grading import GradedAnswer, grade_answer
from tutor.questions import Question
from tutor.quotes import quote_in_passage

PAUSE_SECONDS = 20.0
"""The wait between LLM calls. Groq's free tier allows gpt-oss-120b 8,000
tokens a minute, and a grade costs a few thousand, so back-to-back calls would
hit the limit. The wrapper in llm.py also waits and retries on a 429."""


@dataclass(frozen=True)
class RegressionAnswer:
    """One test answer and the grade it should get.

    Attributes:
        label: What the answer tests, e.g. "correct with an extra fact".
        text: The answer, as a learner would type it.
        expected: The grade it should get: "correct", "partly correct" or "incorrect".
    """

    label: str
    text: str
    expected: str


@dataclass(frozen=True)
class RegressionQuestion:
    """A question from a real passage, with its test answers.

    The passage is named by its topic number and position, not by its id, so
    the set stays readable and survives a fresh ingest.

    Attributes:
        topic: The passage's topic number, e.g. "1.3.1.2".
        position: The passage's position in its topic: 1, 2, 3 ….
        text: The question.
        key_points: What a good answer must contain.
        quote: The passage's words the question is based on.
        answers: The test answers.
    """

    topic: str
    position: int
    text: str
    key_points: tuple[str, ...]
    quote: str
    answers: tuple[RegressionAnswer, ...]


REGRESSION_SET = (
    RegressionQuestion(
        topic="1.3.1.2",
        position=2,
        text=(
            "Why is it usually easier to express a classifier's output as probabilities "
            "than as a firm label like cat or dog?"
        ),
        key_points=(
            "A model that can only output a firm category, such as cat or dog, is difficult "
            "to optimize.",
            "So the model is expressed in the language of probabilities: it assigns a "
            "probability to each possible class.",
        ),
        quote=(
            "For reasons that we will get into as the book gets more technical, it can be "
            "difficult to optimize a model that can only output a firm categorical assignment, "
            "e.g., either “cat” or “dog”."
        ),
        answers=(
            RegressionAnswer(
                "correct",
                "Because a model that can only output a firm label like cat or dog is hard "
                "to optimize, so instead it gives each class a probability.",
                "correct",
            ),
            RegressionAnswer(
                "partly correct",
                "It gives a probability for each class, like being 90% sure that an image "
                "is a cat.",
                "partly correct",
            ),
            RegressionAnswer(
                "wrong",
                "Firm labels are actually easier to optimize. Probabilities are only used "
                "because people find them easier to read.",
                "incorrect",
            ),
            RegressionAnswer(
                "correct with an extra fact",
                "A model that only outputs a firm label like cat or dog is difficult to "
                "optimize, so it gives a probability to each class instead. This is usually "
                "done with a softmax function.",
                "correct",
            ),
            RegressionAnswer(
                "off-topic",
                "Regression predicts numbers, such as the price of a house, by minimizing "
                "the squared error loss.",
                "incorrect",
            ),
        ),
    ),
)
"""The regression set: M1's five test answers, one of each kind PLAN.md names.
The M3 concept "Growing the regression set" adds about 10 more."""

ONE_WORD = ("hard", "easy")
"""The one-word change made to the set's first answer: "hard to optimize" becomes
"easy to optimize", which says the opposite."""


def parse_args() -> argparse.Namespace:
    """Read the command-line options.

    Returns:
        The parsed options.
    """
    parser = argparse.ArgumentParser(description="Re-grade the grading regression set.")
    parser.add_argument("--db", type=Path, default=Path("data/tutor.db"), help="the database")
    parser.add_argument(
        "--model", help="the grading model to try, e.g. openai/gpt-oss-20b; .env's if not given"
    )
    return parser.parse_args()


def add_question(conn: sqlite3.Connection, item: RegressionQuestion) -> sqlite3.Row:
    """Save a regression question into the copy's question pool, and return its row.

    Args:
        conn: The copy's connection.
        item: The regression question.

    Returns:
        The saved question's row, as ``grade_answer()`` expects it.

    Raises:
        ValueError: If the copy has no passage at that topic and position.
    """
    matches = [
        p
        for p in list_passages(conn)
        if (p["number"], p["position"]) == (item.topic, item.position)
    ]
    if not matches:
        raise ValueError(
            f"No passage {item.position} in topic {item.topic}: ingest chapter 1 first."
        )
    passage_id = matches[0]["id"]
    question = Question(question=item.text, key_points=list(item.key_points), quote=item.quote)
    save_questions(conn, passage_id, [question], "regression-set")
    # The question just saved is the passage's newest one.
    return list_questions(conn, passage_id)[-1]


def show(result: GradedAnswer, passage_text: str) -> None:
    """Print a grade's fields, marking any quote that isn't in the passage.

    Args:
        result: What ``grade_answer()`` returned.
        passage_text: The passage, to point out each failing quote.
    """
    grade = result.grade
    for quote in grade.quotes:
        mark = "" if quote_in_passage(quote, passage_text) else "  ← NOT IN THE PASSAGE"
        print(f"    quote:   {quote!r}{mark}")
    for name, items in (
        ("missing", grade.missing),
        ("wrong", grade.wrong),
        ("extra", grade.extras),
    ):
        for item in items:
            print(f"    {name}: {' ' * (7 - len(name))}{item}")
    if not result.supported:
        print("    → the tutor would show: I couldn't find support for this grade")


def calls_since(conn: sqlite3.Connection, after_id: int) -> tuple[int, int]:
    """Count the LLM calls logged after a given row of the token log.

    Args:
        conn: The copy's connection.
        after_id: The last ``llm_calls`` id before the calls to count.

    Returns:
        How many calls, and their tokens in all.
    """
    calls, tokens = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(total_tokens), 0) FROM llm_calls WHERE id > ?",
        (after_id,),
    ).fetchone()
    return calls, tokens


def last_call_id(conn: sqlite3.Connection) -> int:
    """The id of the newest row in the token log (0 if it's empty).

    Args:
        conn: The copy's connection.

    Returns:
        The id.
    """
    return conn.execute("SELECT COALESCE(MAX(id), 0) FROM llm_calls").fetchone()[0]


def main() -> None:
    """Grade every test answer on a copy of the database, then check the same-answer reuse."""
    # Windows terminals sometimes can't print curly quotes in their default encoding.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.model:
        # pydantic-settings reads environment variables before .env, so this
        # overrides GRADER_MODEL for this run only. It must happen before
        # anything calls get_settings(), which reads the settings once.
        os.environ["GRADER_MODEL"] = args.model

    # A temporary folder for the copy. ignore_cleanup_errors: on Windows an
    # open file can't be deleted, and a crash mustn't hide the real error.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        copy = Path(folder) / "tutor.db"
        # copy2 copies the file byte for byte; a missing database fails loudly here.
        shutil.copy2(args.db, copy)
        conn = connect(copy)
        try:
            print(f"Model: {get_settings().grader_model}\n")
            start = last_call_id(conn)
            graded = 0
            as_expected = 0
            first: tuple[sqlite3.Row, str, GradedAnswer] | None = None

            # 1. Grade every test answer, and compare with the grade it should get.
            for item in REGRESSION_SET:
                question = add_question(conn, item)
                passage = conn.execute(
                    "SELECT text FROM passages WHERE id = ?", (question["passage_id"],)
                ).fetchone()
                print(f"## {item.topic}, passage {item.position}: {item.text}\n")
                for answer in item.answers:
                    if graded > 0:
                        time.sleep(PAUSE_SECONDS)  # stay under the tokens-per-minute limit
                    before = last_call_id(conn)
                    result = grade_answer(conn, question, answer.text)
                    _, tokens = calls_since(conn, before)
                    graded += 1
                    got = result.grade.verdict
                    ok = got == answer.expected and result.supported
                    as_expected += ok
                    mark = "as expected" if ok else f"DIFFERS: expected {answer.expected}"
                    print(f"  [{answer.label}] {answer.text}")
                    print(f"    verdict: {got}  ({mark}; {tokens} tokens)")
                    show(result, passage["text"])
                    print()
                    if first is None:
                        first = (question, answer.text, result)

            # 2. The same-answer check, on the set's first answer.
            assert first is not None, "The regression set is empty."
            question, text, original = first
            print("## The same-answer check\n")

            before = last_call_id(conn)
            again = grade_answer(conn, question, "  " + text.upper().replace(" ", "  ") + "  ")
            again_calls, _ = calls_since(conn, before)
            reused = again.reused and again_calls == 0 and again.grade == original.grade
            print("  Same answer, in capitals with extra spaces:")
            print(
                f"    {again_calls} LLM calls; reused: {again.reused}; "
                f"same grade: {again.grade == original.grade}"
            )

            old, new = ONE_WORD
            assert old in text, f"The first answer must contain {old!r} to change it."
            time.sleep(PAUSE_SECONDS)
            before = last_call_id(conn)
            changed = grade_answer(conn, question, text.replace(old, new, 1))
            changed_calls, _ = calls_since(conn, before)
            fresh = not changed.reused and changed_calls == 1
            print(f"  One word changed ({old!r} → {new!r}):")
            print(
                f"    {changed_calls} LLM call; reused: {changed.reused}; "
                f"its verdict: {changed.grade.verdict}\n"
            )

            calls, tokens = calls_since(conn, start)
            print(f"Grades as expected (and supported): {as_expected} of {graded}.")
            print(f"Same answer reused with no LLM call: {reused}")
            print(f"One-word change graded fresh: {fresh}")
            print(f"LLM calls: {calls}, {tokens} tokens in all.")
            ok = as_expected == graded and reused and fresh
            print(f"\nM1's grading goals: {'met' if ok else 'NOT met'}.")
        finally:
            conn.close()
    # A non-zero exit code tells a shell (or CI) that the check failed.
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
