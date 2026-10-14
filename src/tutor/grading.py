"""Grading a typed answer against the passage its question came from.

M1 concept "Grading grounded in the passage". The LLM grades only what the
question asks, using only the passage, and backs its grade with quotes from
the passage that code checks (PLAN.md › How the tutor behaves › Quotes
checked in code, and › Same answer, same grade).

The chain:

    question row (id, text, key points, passage id) + your answer (a string)
       │ 0. look up the question's passage, so the answer is always graded
       │    against the passage that question came from
       │ 1. tidy the answer: lowercase, extra spaces removed
    tidied answer
       │ 2. the same-answer check: is there an earlier supported attempt with
       │    this question, this tidied answer, this prompt version and this
       │    model and settings? If so, hand back its grade and stop here,
       │    with no LLM call.
    no earlier attempt
       │ 3. build the messages: the grader prompt + the passage, the
       │    question, its key points and your answer
    messages for the LLM
       │ 4. one LLM call, forced into the Grade form
    Grade: quotes → missing → wrong → extras → verdict
       │ 5. the quote check, on every quote
    Grade + supported (True only if every quote is really in the passage)
       │ 6. save the attempt, with the prompt version and the model and settings
    the Grade, marked supported or not (not → the tutor shows
    "I couldn't find support for this grade")

An attempt whose quotes failed is saved, so failures can be counted, but
step 2 never reuses it: sending the same answer again grades it afresh.
"""

import json
import sqlite3
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from tutor.config import get_settings
from tutor.db import find_attempt, get_passage, save_attempt
from tutor.llm import complete_structured, load_prompt
from tutor.quotes import quote_in_passage

GRADER_REASONING = "medium"
"""How hard gpt-oss-120b thinks before grading. Its thinking tokens count
toward the free-tier limits (8,000 tokens a minute is the tightest). Grading
is a judgement, so it gets more thinking than question writing ("low");
``scripts/regrade.py`` prints the tokens each grade costs."""


class Grade(BaseModel):
    """A grade for one typed answer, decided only from the passage, written in field order."""

    # Pydantic copies this class's docstring and each Field(description=...)
    # into the JSON Schema that's sent to the LLM, so they're instructions for
    # it too. The rules themselves live in prompts/grader.md, whose version is
    # saved with every attempt.
    #
    # The ORDER of the fields matters. The LLM writes its answer left to
    # right, one token at a time, and each token is chosen by reading all the
    # text before it, including what it has already written. So the quotes
    # (the evidence) come first, and the verdict comes last: when the LLM
    # writes the verdict, its quotes and its missing/wrong lists are already
    # in front of it. A verdict written first could only be justified
    # afterwards, never shaped by them.
    #
    # Names: "Grade" is the whole form, all five fields. "verdict" is just
    # the one word: correct, partly correct or incorrect.
    quotes: list[str] = Field(
        description="the passage's sentences this grade rests on, each copied exactly"
    )
    missing: list[str] = Field(description="key points the answer leaves out; empty if none")
    wrong: list[str] = Field(
        description="statements in the answer the passage contradicts, and what it says instead"
    )
    extras: list[str] = Field(
        description="statements in the answer the passage doesn't cover; they cost nothing"
    )
    # Literal makes the schema an "enum": Groq's strict mode lets the LLM write
    # only one of these three exact strings, and Pydantic checks it again.
    verdict: Literal["correct", "partly correct", "incorrect"] = Field(
        description="decided last, from the fields above"
    )


@dataclass(frozen=True)
class GradedAnswer:
    """What ``grade_answer()`` hands back: the grade, and what the tutor should do with it.

    A plain dataclass, not a Pydantic model: it never goes to or comes from
    the LLM, so it needs no schema.

    Attributes:
        attempt_id: The attempt's row in the attempts table.
        grade: The LLM's Grade.
        supported: Whether every quote is really in the passage. If not, the
            tutor shows "I couldn't find support for this grade" instead.
        reused: True if the same-answer check found an earlier attempt, so
            no LLM call was made.
    """

    attempt_id: int
    grade: Grade
    supported: bool
    reused: bool


def tidy_answer(answer: str) -> str:
    """Tidy an answer for the same-answer check: lowercase, extra spaces removed.

    Only changes that can't alter the meaning are made (PLAN.md › Same answer,
    same grade). Nothing like "similar enough": one word ("is" → "isn't") can
    flip an answer's meaning.

    Args:
        answer: The answer as typed.

    Returns:
        The tidied answer, e.g. "  It gives   a Probability " → "it gives a probability".
    """
    # str.split() with no argument splits on any run of spaces, tabs or line
    # breaks and drops them at both ends; joining with " " leaves single spaces.
    return " ".join(answer.lower().split())


def grade_answer(
    conn: sqlite3.Connection,
    question: sqlite3.Row,
    answer: str,
    session_id: str | None = None,
) -> GradedAnswer:
    """Grade a typed answer against its question's passage, reusing the grade of a repeated answer.

    Args:
        conn: An open connection.
        question: A row of the questions table, with ``id``, ``passage_id``,
            ``text`` and ``key_points``.
        answer: The answer as typed.
        session_id: The study session, or None before sessions exist (until
            the M1 concept "Pausing for your answer").

    Returns:
        The grade, whether its quotes checked out, and whether it was reused.

    Raises:
        ValueError: If the answer is empty. The study loop asks again before
            calling this, so an empty answer never costs an LLM call.
        FreeLimitReached: If Groq's free tier stops the call.
    """
    # 0. The question's own passage, by the id stored with the question.
    passage = get_passage(conn, question["passage_id"])

    # 1. Tidy the answer.
    tidied = tidy_answer(answer)
    if not tidied:
        raise ValueError("The answer is empty: there's nothing to grade.")

    # 2. The same-answer check. Its key includes the prompt's version and the
    #    model with its settings, so the prompt is loaded first.
    prompt = load_prompt("grader")
    model = get_settings().grader_model
    settings = f"reasoning_effort={GRADER_REASONING}"
    earlier = find_attempt(conn, question["id"], tidied, prompt.version, model, settings)
    if earlier is not None:
        # The saved JSON turns back into a Grade: the same grade, for free.
        # Only supported attempts are ever found, so supported is True.
        grade = Grade.model_validate_json(earlier["grade_details"])
        return GradedAnswer(earlier["id"], grade, supported=True, reused=True)

    # 3. Build the messages. The prompt file is the "system" message; the
    #    "user" message carries the four things the grader needs. The key
    #    points are stored as a JSON list, so json.loads turns them back.
    key_points = "\n".join(f"- {point}" for point in json.loads(question["key_points"]))
    user_message = (
        f"Passage:\n{passage['text']}\n\n"
        f"Question:\n{question['text']}\n\n"
        f"Key points:\n{key_points}\n\n"
        f"Learner's answer:\n{answer}"
    )
    messages = [
        {"role": "system", "content": prompt.text},
        {"role": "user", "content": user_message},
    ]

    # 4. One LLM call, forced into the Grade form (Groq strict mode). It
    #    raises instead of returning something half-right.
    grade = complete_structured(
        conn,
        messages,
        Grade,
        model=model,
        reasoning_effort=GRADER_REASONING,
        purpose="grading",
    )

    # 5. The quote check: every quote must really be in the passage, and
    #    there must be at least one. A grade with no quotes rests on nothing
    #    code can check, so it isn't supported either.
    supported = bool(grade.quotes) and all(
        quote_in_passage(quote, passage["text"]) for quote in grade.quotes
    )

    # 6. Save the attempt, supported or not, and hand the result back.
    attempt_id = save_attempt(
        conn,
        question["id"],
        session_id,
        answer,
        tidied,
        grade,
        supported,
        prompt.version,
        model,
        settings,
    )
    return GradedAnswer(attempt_id, grade, supported, reused=False)
