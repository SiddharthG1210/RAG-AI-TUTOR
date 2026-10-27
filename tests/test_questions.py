"""Tests for ``tutor.questions``: the wiring from one passage to its saved, checked questions.

No request ever reaches Groq. ``monkeypatch`` (a pytest fixture that swaps an
attribute for the length of one test, then puts it back) replaces three
things inside ``tutor.questions``:

- ``complete_structured``, the LLM call, with a fake that records what it was
  sent and returns a fixed ``QuestionSet``;
- ``get_settings``, because CI has no ``.env`` and so no API key;
- ``load_prompt``, so the tests don't depend on the prompt's wording.

The database is in memory (``":memory:"``), so ``data/tutor.db`` is never touched.
"""

import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

from tutor import questions
from tutor.db import connect, list_passages, list_questions, save_chapter
from tutor.ingest.pdf import Paragraph, Topic
from tutor.llm import Prompt
from tutor.questions import (
    Question,
    QuestionSet,
    cut_source_phrase,
    mentions_the_source,
    write_questions,
)

# The real passage 2 of 1.3.1.2 Classification.
PASSAGE_TEXT = (
    "For reasons that we will get into as the book gets more technical, it can be "
    "difficult to optimize a model that can only output a firm categorical assignment, "
    "e.g., either “cat” or “dog”. In these cases, it is usually much easier to express "
    "our model in the language of probabilities."
)

PROMPT = Prompt(text="You write study questions.", version="abc123def456")


def question(text: str, quote: str) -> Question:
    """Build a question with one key point, as the fake LLM returns it.

    Args:
        text: The question.
        quote: The quote it's based on.

    Returns:
        The question.
    """
    return Question(question=text, key_points=["a point from the passage"], quote=quote)


# Two quotes copied from the passage (one typed with straight quotes, as an LLM
# would), and one the passage never says.
REAL = question(
    "Why use probabilities?",
    "In these cases, it is usually much easier to express our model in the language of "
    "probabilities.",
)
ALSO_REAL = question("What is hard to optimize?", 'either "cat" or "dog"')
MADE_UP = question("What do classifiers output?", "Classifiers always output exactly one label.")


class FakeLLM:
    """Stands in for ``complete_structured``: records each call and returns a fixed answer."""

    def __init__(self, answer: QuestionSet) -> None:
        """Remember the answer to give.

        Args:
            answer: The QuestionSet every call returns.
        """
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, conn: sqlite3.Connection, messages: list[dict[str, str]], schema: type, **kwargs: Any
    ) -> QuestionSet:
        """Record what was sent, the way ``complete_structured`` would receive it.

        Args:
            conn: The connection (unused).
            messages: The chat messages.
            schema: The schema asked for.
            **kwargs: model, reasoning_effort and purpose.

        Returns:
            The fixed answer.
        """
        self.calls.append({"messages": messages, "schema": schema, **kwargs})
        return self.answer


@pytest.fixture
def conn() -> sqlite3.Connection:
    """An in-memory database holding one passage: 1.3.1.2's passage 2."""
    conn = connect(":memory:")
    topic = Topic("1.3.1.2", "Classification", 4, 49, "9")
    paragraph = Paragraph("1.3.1.2", 49, "9", PASSAGE_TEXT, 49, "9")
    save_chapter(conn, [topic], {"1.3.1.2": [[paragraph]]})
    return conn


def use(monkeypatch: pytest.MonkeyPatch, answer: QuestionSet) -> FakeLLM:
    """Swap the LLM call, the settings and the prompt inside ``tutor.questions``.

    Args:
        monkeypatch: pytest's fixture.
        answer: What the fake LLM returns.

    Returns:
        The fake, to inspect what it was sent.
    """
    fake = FakeLLM(answer)
    monkeypatch.setattr(questions, "complete_structured", fake)
    monkeypatch.setattr(
        questions, "get_settings", lambda: SimpleNamespace(question_model="fake/model")
    )
    monkeypatch.setattr(questions, "load_prompt", lambda name: PROMPT)
    return fake


def test_only_questions_whose_quote_is_real_are_kept_and_saved(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The made-up quote's question is dropped; the other two are returned and saved."""
    use(monkeypatch, QuestionSet(questions=[REAL, MADE_UP, ALSO_REAL]))
    passage = list_passages(conn)[0]

    kept = write_questions(conn, passage)

    assert kept == [REAL, ALSO_REAL]
    rows = list_questions(conn, passage["id"])
    assert [(r["text"], r["prompt_version"]) for r in rows] == [
        ("Why use probabilities?", "abc123def456"),
        ("What is hard to optimize?", "abc123def456"),
    ]


def test_the_call_sends_the_prompt_and_the_passage_text_only(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """System = the prompt; user = the passage, without its number or title; one call."""
    fake = use(monkeypatch, QuestionSet(questions=[REAL, REAL, REAL]))

    write_questions(conn, list_passages(conn)[0])

    [call] = fake.calls
    assert call["messages"] == [
        {"role": "system", "content": "You write study questions."},
        {"role": "user", "content": f"Passage:\n{PASSAGE_TEXT}"},
    ]
    assert call["schema"] is QuestionSet
    assert (call["model"], call["reasoning_effort"], call["purpose"]) == (
        "fake/model",
        "low",
        "questions",
    )


def test_a_refill_lists_the_earlier_questions(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second call's user message lists the questions the first one saved."""
    fake = use(monkeypatch, QuestionSet(questions=[REAL, MADE_UP, ALSO_REAL]))
    passage = list_passages(conn)[0]

    write_questions(conn, passage)  # first time: the pool gets 2 questions
    write_questions(conn, passage)  # refill

    first, refill = (call["messages"][1]["content"] for call in fake.calls)
    assert "already written" not in first
    assert refill.endswith(
        "Questions already written for this passage:\n"
        "- Why use probabilities?\n"
        "- What is hard to optimize?"
    )


# pytest.mark.parametrize runs the test once per (before, after) pair, and
# reports each pair as its own test. Most pairs are real questions the LLM
# wrote for chapter 1.
@pytest.mark.parametrize(
    ("before", "after"),
    [
        # At the start, with a comma after it: the next word gets its capital.
        (
            "According to the passage, what distinguishes binary classification from "
            "other types of classification?",
            "What distinguishes binary classification from other types of classification?",
        ),
        # At the end, with a comma before it.
        (
            "Why do classification problems need different tools from regression, "
            "according to the passage?",
            "Why do classification problems need different tools from regression?",
        ),
        (
            "What is the primary output of a classifier, as described in the passage?",
            "What is the primary output of a classifier?",
        ),
        (
            "What are the typical steps involved in the training process described in the passage?",
            "What are the typical steps involved in the training process?",
        ),
        # In the middle, between two commas.
        ("What, according to this text, is a model?", "What is a model?"),
        ("Which example in the passage shows overfitting?", "Which example shows overfitting?"),
        # Built around the passage: nothing to cut, so mentions_the_source() drops it.
        ("What does the passage say about labels?", "What does the passage say about labels?"),
        # "from the text" can be the data the model learns from, so it stays.
        ("What can a model learn from the text?", "What can a model learn from the text?"),
        ("Why is a loss function needed?", "Why is a loss function needed?"),
    ],
)
def test_cut_source_phrase(before: str, after: str) -> None:
    """A phrase that only points at the passage is cut; anything else stays as it was."""
    assert cut_source_phrase(before) == after


@pytest.mark.parametrize(
    ("question", "mentions"),
    [
        ("What does the passage say about labels?", True),
        ("How does the author describe the wake-word problem?", True),
        ("How do the authors describe the wake-word problem?", True),
        ("What does this text claim about data?", True),
        # "the" or "this" must come first, and whole words only.
        ("What does a speech-to-text system output?", False),
        ("Why does a wake word need context?", False),
        ("Why is a loss function needed?", False),
    ],
)
def test_mentions_the_source(question: str, mentions: bool) -> None:
    """The passage, the text or the author counts; a topic with the same letters doesn't."""
    assert mentions_the_source(question) is mentions


def test_the_wording_check_cuts_or_drops_before_saving(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tacked-on "according to the passage" is cut; a question built around it is dropped.

    All three quotes are real, so only the wording decides.
    """
    tacked_on = question("According to the passage, why use probabilities?", REAL.quote)
    built_around = question("What does the passage say about optimizing?", ALSO_REAL.quote)
    use(monkeypatch, QuestionSet(questions=[tacked_on, built_around, ALSO_REAL]))
    passage = list_passages(conn)[0]

    kept = write_questions(conn, passage)

    assert [q.question for q in kept] == ["Why use probabilities?", "What is hard to optimize?"]
    rows = list_questions(conn, passage["id"])
    assert [r["text"] for r in rows] == ["Why use probabilities?", "What is hard to optimize?"]


def test_when_no_quote_passes_nothing_is_saved(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero kept is a normal result, not an error: the caller decides what comes next."""
    use(monkeypatch, QuestionSet(questions=[MADE_UP, MADE_UP, MADE_UP]))
    passage = list_passages(conn)[0]

    assert write_questions(conn, passage) == []
    assert list_questions(conn, passage["id"]) == []
