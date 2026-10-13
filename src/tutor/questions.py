"""Writing questions about a passage: the shape of the LLM's answer, and the chain around it.

M1 concept "Structured output (question writing)". One LLM call writes 3
questions about one passage, each with the key points a good answer contains
and a quote from the passage. Code keeps only the questions whose quote is
really in the passage, and saves them as the passage's pool (PLAN.md ›
Decisions › Reviews).

The chain:

    one passage row (id, number, title, text)
       │ 1. build the messages: the prompt's instructions + the passage
       │    (+ its earlier questions, on a refill)
    messages for the LLM
       │ 2. one LLM call, forced into the QuestionSet shape
    QuestionSet: 3 × {question, key points, quote}
       │ 3. the quote check, per question
    the questions that passed (0 to 3)
       │ 4. save them, with the passage's id and the prompt's version
    up to 3 checked questions in the questions table

What happens when none pass (write once more, or skip the passage) is decided
by the study loop's routing, in the M1 concept "Conditional edges".
"""

import sqlite3

from pydantic import BaseModel, Field

from tutor.config import get_settings
from tutor.db import list_questions, save_questions
from tutor.llm import complete_structured, load_prompt
from tutor.quotes import quote_in_passage

QUESTION_REASONING = "low"
"""How hard gpt-oss thinks before writing. Its thinking tokens count toward
the free-tier limits, and writing questions from a given passage needs little.
"medium" (about 2,700 tokens a call instead of 1,600) was tried against
questions that say "according to the passage" and didn't reduce them."""


class Question(BaseModel):
    """One question about a passage, what a good answer contains, and the line it's based on."""

    # Pydantic copies this class's docstring and each Field(description=...)
    # into the JSON Schema, and the schema is sent to the LLM, so they're
    # instructions for it too. Notes for developers go in comments, like this.
    # The field names are fixed, because save_questions() and the grader read them.
    question: str = Field(description="the question itself")
    key_points: list[str] = Field(
        description="what a good answer must contain, as a list of short strings"
    )
    quote: str = Field(description="the passage's words the question is based on, copied exactly")


class QuestionSet(BaseModel):
    """What one call to the question writer returns: exactly 3 questions about one passage."""

    # min_length/max_length make Pydantic refuse any other number of questions.
    # strict_schema() drops them before the call, so the LLM learns "exactly 3"
    # from the docstring above, which is sent as the schema's description.
    questions: list[Question] = Field(min_length=3, max_length=3)


def write_questions(conn: sqlite3.Connection, passage: sqlite3.Row) -> list[Question]:
    """Write questions for one passage, keep the ones whose quote checks out, and save them.

    Args:
        conn: An open connection.
        passage: A row from ``list_passages()`` or ``get_passage()``, with
            ``id``, ``number``, ``title`` and ``text``.

    Returns:
        The questions that passed the quote check (0 to 3), already saved in
        the passage's pool.
    """
    # 1. Build the messages.
    #    The prompt file is the "system" message: the standing instructions.
    #    Its version (a hash of its text) is saved with each question below.
    prompt = load_prompt("question_writer")
    #    The "user" message carries the passage text only: no section number
    #    or title, so the LLM can't ask about them.
    user_message = f"Passage:\n{passage['text']}"
    #    First time or refill? There's no need to ask: list_questions() is
    #    simply empty the first time, so the block below is skipped. On a
    #    refill it holds the passage's earlier questions, used or not, and the
    #    prompt's "Earlier questions" section tells the LLM not to repeat them.
    earlier = list_questions(conn, passage["id"])
    if earlier:
        listed = "\n".join(f"- {row['text']}" for row in earlier)
        user_message += f"\n\nQuestions already written for this passage:\n{listed}"
    messages = [
        {"role": "system", "content": prompt.text},
        {"role": "user", "content": user_message},
    ]

    # 2. One LLM call, forced into the QuestionSet shape (Groq strict mode).
    #    It raises instead of returning something half-right: FreeLimitReached
    #    when the free tier runs out, and pydantic's ValidationError when the
    #    answer breaks a rule the schema couldn't carry, such as "exactly 3".
    question_set = complete_structured(
        conn,
        messages,
        QuestionSet,
        model=get_settings().question_model,
        reasoning_effort=QUESTION_REASONING,
        purpose="questions",
    )

    # 3. The quote check, per question: keep only the questions whose quote
    #    really is in the passage. Key points aren't checked; the prompt keeps
    #    them inside the passage.
    kept = [q for q in question_set.questions if quote_in_passage(q.quote, passage["text"])]

    # 4. Save the kept ones in the passage's pool, with the prompt's version,
    #    and return them. With none kept, nothing is saved, and the caller (the
    #    study loop's routing, in "Conditional edges") decides what to do.
    save_questions(conn, passage["id"], kept, prompt.version)
    return kept
