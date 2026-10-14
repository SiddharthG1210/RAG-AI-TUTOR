"""Tests for ``tutor.grading``: from one typed answer to a saved, checked grade.

No request ever reaches Groq. ``monkeypatch`` (a pytest fixture that swaps an
attribute for the length of one test, then puts it back) replaces three
things inside ``tutor.grading``:

- ``complete_structured``, the LLM call, with a fake that records what it was
  sent and returns a fixed ``Grade``;
- ``get_settings``, because CI has no ``.env`` and so no API key;
- ``load_prompt``, so the tests don't depend on the prompt's wording, and
  can change its version.

The database is in memory (``":memory:"``), so ``data/tutor.db`` is never touched.
"""

import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

from tutor import grading
from tutor.db import connect, list_passages, list_questions, save_chapter, save_questions
from tutor.grading import Grade, grade_answer, tidy_answer
from tutor.ingest.pdf import Paragraph, Topic
from tutor.llm import Prompt

# The real passage 2 of 1.3.1.2 Classification (its first two sentences).
PASSAGE_TEXT = (
    "For reasons that we will get into as the book gets more technical, it can be "
    "difficult to optimize a model that can only output a firm categorical assignment, "
    "e.g., either “cat” or “dog”. In these cases, it is usually much easier to express "
    "our model in the language of probabilities."
)

PROMPT = Prompt(text="You grade answers.", version="abc123def456")

ANSWER = "It gives a probability for each class, like 90% sure it's a cat."

# A grade whose quote is really in the passage, and one whose quote isn't.
SUPPORTED = Grade(
    quotes=[
        "In these cases, it is usually much easier to express our model in the language of "
        "probabilities."
    ],
    missing=["a firm label is hard to optimize"],
    wrong=[],
    extras=[],
    verdict="partly correct",
)
MADE_UP = SUPPORTED.model_copy(update={"quotes": ["Classifiers always use softmax."]})


class FakeLLM:
    """Stands in for ``complete_structured``: records each call and returns a fixed answer."""

    def __init__(self, answer: Grade) -> None:
        """Remember the answer to give.

        Args:
            answer: The Grade every call returns.
        """
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, conn: sqlite3.Connection, messages: list[dict[str, str]], schema: type, **kwargs: Any
    ) -> Grade:
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
    """An in-memory database holding 1.3.1.2's passage 2 and one question about it."""
    conn = connect(":memory:")
    topic = Topic("1.3.1.2", "Classification", 4, 49, "9")
    paragraph = Paragraph("1.3.1.2", 50, "10", PASSAGE_TEXT, 50, "10")
    save_chapter(conn, [topic], {"1.3.1.2": [[paragraph]]})
    question = SimpleNamespace(
        question="Why is it easier to output probabilities than a firm label?",
        key_points=["A firm label is hard to optimize.", "Each class gets a probability."],
        quote="it is usually much easier to express our model in the language of probabilities.",
    )
    save_questions(conn, list_passages(conn)[0]["id"], [question], "q-version")
    return conn


def the_question(conn: sqlite3.Connection) -> sqlite3.Row:
    """The fixture's one question, as a row of the questions table.

    Args:
        conn: The fixture's connection.

    Returns:
        The question's row.
    """
    return list_questions(conn, list_passages(conn)[0]["id"])[0]


def use(monkeypatch: pytest.MonkeyPatch, answer: Grade, prompt: Prompt = PROMPT) -> FakeLLM:
    """Swap the LLM call, the settings and the prompt inside ``tutor.grading``.

    Args:
        monkeypatch: pytest's fixture.
        answer: What the fake LLM returns.
        prompt: The prompt ``load_prompt`` returns.

    Returns:
        The fake, to inspect what it was sent.
    """
    fake = FakeLLM(answer)
    monkeypatch.setattr(grading, "complete_structured", fake)
    monkeypatch.setattr(
        grading, "get_settings", lambda: SimpleNamespace(grader_model="fake/grader")
    )
    monkeypatch.setattr(grading, "load_prompt", lambda name: prompt)
    return fake


def test_tidy_answer_lowercases_and_tidies_spaces() -> None:
    """Capitals and extra spaces or line breaks go; every word and character stays."""
    assert tidy_answer("  It gives\n a  Probability.  ") == "it gives a probability."


def test_the_call_sends_the_prompt_passage_question_key_points_and_answer(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """System = the prompt; user = the four things the grader needs; one call to the grader."""
    fake = use(monkeypatch, SUPPORTED)

    grade_answer(conn, the_question(conn), ANSWER)

    [call] = fake.calls
    assert call["messages"] == [
        {"role": "system", "content": "You grade answers."},
        {
            "role": "user",
            "content": (
                f"Passage:\n{PASSAGE_TEXT}\n\n"
                "Question:\nWhy is it easier to output probabilities than a firm label?\n\n"
                "Key points:\n- A firm label is hard to optimize.\n"
                "- Each class gets a probability.\n\n"
                f"Learner's answer:\n{ANSWER}"
            ),
        },
    ]
    assert call["schema"] is Grade
    assert (call["model"], call["reasoning_effort"], call["purpose"]) == (
        "fake/grader",
        "medium",
        "grading",
    )


def test_a_grade_with_real_quotes_is_supported_and_saved(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The grade comes back supported, and its attempt row holds everything it was based on."""
    use(monkeypatch, SUPPORTED)

    result = grade_answer(conn, the_question(conn), ANSWER, session_id="s1")

    assert (result.grade, result.supported, result.reused) == (SUPPORTED, True, False)
    row = conn.execute("SELECT * FROM attempts WHERE id = ?", (result.attempt_id,)).fetchone()
    assert (row["session_id"], row["answer"], row["verdict"], row["supported"]) == (
        "s1",
        ANSWER,
        "partly correct",
        1,
    )
    assert (row["prompt_version"], row["model"], row["settings"]) == (
        "abc123def456",
        "fake/grader",
        "reasoning_effort=medium",
    )
    assert Grade.model_validate_json(row["grade_details"]) == SUPPORTED


@pytest.mark.parametrize(
    "grade",
    [MADE_UP, SUPPORTED.model_copy(update={"quotes": []})],
    ids=["a made-up quote", "no quotes at all"],
)
def test_a_grade_without_real_quotes_is_not_supported_but_still_saved(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch, grade: Grade
) -> None:
    """The tutor will show "I couldn't find support"; the attempt is kept, to count failures."""
    use(monkeypatch, grade)

    result = grade_answer(conn, the_question(conn), ANSWER)

    assert result.supported is False
    row = conn.execute("SELECT supported FROM attempts WHERE id = ?", (result.attempt_id,))
    assert row.fetchone()["supported"] == 0


def test_the_same_answer_again_reuses_the_grade_with_no_llm_call(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Different capitals and spaces are still the same answer: one call, one attempt."""
    fake = use(monkeypatch, SUPPORTED)
    question = the_question(conn)

    first = grade_answer(conn, question, ANSWER)
    again = grade_answer(conn, question, "  " + ANSWER.upper() + "  ")

    assert len(fake.calls) == 1
    assert (again.attempt_id, again.grade, again.supported, again.reused) == (
        first.attempt_id,
        SUPPORTED,
        True,
        True,
    )
    assert conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 1


def test_a_one_word_change_is_graded_fresh(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One changed word means a different answer: a second LLM call."""
    fake = use(monkeypatch, SUPPORTED)
    question = the_question(conn)

    grade_answer(conn, question, ANSWER)
    changed = grade_answer(conn, question, ANSWER.replace("each", "one"))

    assert len(fake.calls) == 2
    assert changed.reused is False


def test_an_unsupported_grade_is_never_reused(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same answer after a failed quote check gets a fresh try, not "no support" again."""
    fake = use(monkeypatch, MADE_UP)
    question = the_question(conn)

    grade_answer(conn, question, ANSWER)
    grade_answer(conn, question, ANSWER)

    assert len(fake.calls) == 2


def test_a_new_prompt_version_grades_the_same_answer_afresh(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After an edit to the prompt file, earlier grades aren't reused."""
    question = the_question(conn)
    use(monkeypatch, SUPPORTED)
    grade_answer(conn, question, ANSWER)

    fake = use(monkeypatch, SUPPORTED, Prompt(text="You grade answers.", version="new000000000"))
    result = grade_answer(conn, question, ANSWER)

    assert len(fake.calls) == 1
    assert result.reused is False


def test_an_empty_answer_fails_before_any_llm_call(
    conn: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Spaces alone aren't an answer: a clear error, and no call."""
    fake = use(monkeypatch, SUPPORTED)

    with pytest.raises(ValueError, match="empty"):
        grade_answer(conn, the_question(conn), "   \n ")
    assert fake.calls == []
