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
       │ 3. the wording check (cut "according to the passage", drop what
       │    still mentions the passage) and the quote check, per question
    the questions that passed (0 to 3)
       │ 4. save them, with the passage's id and the prompt's version
    up to 3 checked questions in the questions table

What happens when none pass (write once more, or skip the passage) is decided
by the study loop's routing, in the M1 concept "Conditional edges".

**The wording check** (PLAN.md › How the tutor behaves in specific cases ›
Questions never mention the passage). The learner sees only the question,
never the passage, so "According to the passage, why…" reads oddly. The
prompt forbids it, yet about 4 in 10 questions said it anyway: two prompt
wordings, reasoning "medium" and a bigger model didn't change that. So code
does the job, the way it checks quotes:

- ``cut_source_phrase()`` cuts a phrase that only points at the passage
  ("according to the passage", "as described in the passage"). The question
  still asks the same thing, so its key points and quote still fit.
- ``mentions_the_source()`` finds a question that still mentions the passage,
  the text or the author ("What does the passage say about…"). It's built
  around the passage, so it can't be cut, and it's dropped.

Both compare characters with regular expressions, never meaning (PLAN.md ›
Design rules › Three kinds of matching).
"""

import re
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
questions that say "according to the passage" and didn't reduce them, so the
wording check in code handles those instead."""

# Regular expressions (the ``re`` module) describe a pattern of characters to
# find in a text. The pieces used below:
#   \b        a word boundary, so "text" doesn't match inside "context"
#   \s+       one or more spaces
#   (?:a|b)   either a or b, as one group that isn't kept separately
#   ,?        an optional comma
#   s?        an optional "s", so "passage" and "passages" both match
# re.IGNORECASE makes "According" and "according" match alike. re.compile()
# turns a pattern into an object once, so it isn't parsed again on every call.

SOURCE_PHRASE = re.compile(
    # An optional comma and spaces before the phrase, so ", according to the
    # passage?" leaves "?" and not " ?".
    r",?\s*\b(?:"
    # "according to the passage", "as described in this text", "stated in the
    # passage", …: always about the source, whether it says passage or text.
    r"(?:according\s+to|(?:as\s+)?(?:described|stated|explained|mentioned|discussed|presented)\s+in)"
    r"\s+(?:the|this)\s+(?:passage|text)"
    # "in the passage", "from this passage": passage only. "from the text" is
    # left alone, because in a book about deep learning, text can be the data
    # ("What can a model learn from the text?").
    r"|(?:in|from)\s+(?:the|this)\s+passage"
    # An optional comma after it, so "According to the passage, why…" leaves "why…".
    r")\b,?",
    re.IGNORECASE,
)
"""A phrase that only points at the passage, which ``cut_source_phrase()`` cuts."""

MENTIONS_SOURCE = re.compile(r"\b(?:the|this)\s+(?:passages?|text|authors?)\b", re.IGNORECASE)
"""Any mention of the passage, the text or the author, which ``mentions_the_source()`` finds.

"the" or "this" must come first, so a topic like "speech-to-text" or "a
passage of text" doesn't count."""


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


def cut_source_phrase(question: str) -> str:
    """Cut a phrase that only points at the passage, such as "according to the passage".

    The learner never sees the passage, and the phrase doesn't change what's
    asked, so the question reads as a teacher would ask it in class.

    Args:
        question: A question as the LLM wrote it.

    Returns:
        The question without the phrase, with its spaces tidied and its first
        letter in capitals: "According to the passage, what is a label?" →
        "What is a label?". A question without such a phrase comes back
        unchanged, even one built around the passage ("What does the passage
        say about labels?"), which ``mentions_the_source()`` then finds.
    """
    # .sub() replaces every match of the pattern; a space in its place keeps
    # the words on either side apart.
    cut = SOURCE_PHRASE.sub(" ", question)
    if cut == question:
        return question
    # Tidy what the cut left: runs of spaces become one, and a space before
    # punctuation goes ("regression ?" → "regression?"). \1 puts back the
    # punctuation mark the group ([…]) matched.
    cut = re.sub(r"\s+", " ", cut).strip()
    cut = re.sub(r"\s+([?.,;:!])", r"\1", cut)
    # The cut can start the question with a small letter ("how many…").
    return cut[:1].upper() + cut[1:]


def mentions_the_source(question: str) -> bool:
    """Does a question mention the passage, the text or the author?

    Args:
        question: A question, after ``cut_source_phrase()``.

    Returns:
        True for "What does the passage say about labels?" or "How do the
        authors describe it?"; False for "What does a speech-to-text system
        output?".
    """
    # .search() looks for the pattern anywhere in the text, and gives None
    # when it isn't there.
    return MENTIONS_SOURCE.search(question) is not None


def write_questions(conn: sqlite3.Connection, passage: sqlite3.Row) -> list[Question]:
    """Write questions for one passage, keep the ones that pass the checks, and save them.

    Args:
        conn: An open connection.
        passage: A row from ``list_passages()`` or ``get_passage()``, with
            ``id``, ``number``, ``title`` and ``text``.

    Returns:
        The questions that passed the wording check and the quote check (0
        to 3), with any "according to the passage" cut, already saved in the
        passage's pool.
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

    # 3. The checks, per question.
    #    a. The wording: cut a phrase like "according to the passage". It
    #       only points at the source, so the question still asks the same
    #       thing. model_copy(update=...) is Pydantic's way to copy an object
    #       with one field changed, leaving the LLM's answer as it was.
    cut = [
        q.model_copy(update={"question": cut_source_phrase(q.question)})
        for q in question_set.questions
    ]
    #    b. Keep only the questions whose quote really is in the passage and
    #       that no longer mention it. A question built around the passage
    #       ("What does the passage say about…") can't be cut, so it's
    #       dropped. Key points aren't checked; the prompt keeps them inside
    #       the passage.
    kept = [
        q
        for q in cut
        if quote_in_passage(q.quote, passage["text"]) and not mentions_the_source(q.question)
    ]

    # 4. Save the kept ones in the passage's pool, with the prompt's version,
    #    and return them. With none kept, nothing is saved, and the caller (the
    #    study loop's routing, in "Conditional edges") decides what to do.
    save_questions(conn, passage["id"], kept, prompt.version)
    return kept
