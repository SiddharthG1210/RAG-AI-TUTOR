"""Tests for ``tutor.api.ui``: what each click would put on screen, checked without a browser.

The handlers (``on_study``, ``on_submit``, ``on_stop``) return a ``Screen``
of plain values, so a test calls them like a click would and checks the
screen. They drive a real graph over a database file in ``tmp_path``, whose
one passage already has a question, with the grading tool swapped for a fake:
no request reaches Groq. ``tests/test_main.py`` checks the page is served.
"""

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from tutor.api.ui import (
    FREE_LIMIT,
    SESSION_ENDED,
    SESSION_LOST,
    Screen,
    on_stop,
    on_study,
    on_submit,
    result_markdown,
    screen_after,
)
from tutor.db import connect, list_passages, save_attempt, save_chapter, save_questions
from tutor.grading import Grade, GradedAnswer
from tutor.graph import nodes
from tutor.graph.build import build_graph
from tutor.ingest.pdf import Paragraph, Topic
from tutor.llm import FreeLimitReached

GRADE = Grade(quotes=["much easier"], missing=[], wrong=[], extras=[], verdict="correct")

GRADED = {
    "kind": "graded",
    "verdict": "partly correct",
    "reused": False,
    "missing": ["a firm label is hard to optimize"],
    "wrong": [],
    "extras": [],
    "quotes": ["It is much easier"],
    "passage": {"section": "1.3.1.2 Classification", "pages": "10–11", "text": "It is much …"},
}
"""A ``shown`` dict as the ``show`` node packs it."""


def make_db(tmp_path: Path) -> Path:
    """Make a database file with one passage of 1.3.1.2 Classification and one question.

    Args:
        tmp_path: pytest's temporary folder for this test.

    Returns:
        The database file's path.
    """
    path = tmp_path / "tutor.db"
    conn = connect(path)
    topic = Topic("1.3.1.2", "Classification", 4, 49, "9")
    paragraph = Paragraph("1.3.1.2", 49, "9", "It is much easier …", 49, "9")
    save_chapter(conn, [topic], {"1.3.1.2": [[paragraph]]})
    passage_id = list_passages(conn)[0]["id"]
    question = SimpleNamespace(question="Why?", key_points=["a point"], quote="much easier")
    save_questions(conn, passage_id, [question], "v")
    conn.close()
    return path


@pytest.fixture
def fake_grading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the grading tool for a fake that saves the attempt like the real one, with no LLM call.

    Args:
        monkeypatch: pytest's fixture.
    """

    def fake_grade(
        conn: sqlite3.Connection, question: sqlite3.Row, answer: str, session_id: str | None
    ) -> GradedAnswer:
        attempt_id = save_attempt(
            conn, question["id"], session_id, answer, answer, GRADE, True, "v", "m", "s"
        )
        return GradedAnswer(attempt_id, GRADE, supported=True, reused=False)

    monkeypatch.setattr(nodes, "grade_answer", fake_grade)


def test_result_markdown_shows_the_verdict_and_only_the_non_empty_lists() -> None:
    """The verdict in bold, then Missing and Based on as bullets; the empty Wrong is left out."""
    assert result_markdown(GRADED) == (
        "**Verdict: partly correct**\n"
        "\nMissing:\n- a firm label is hard to optimize\n"
        "\nBased on:\n- It is much easier"
    )


def test_result_markdown_says_when_a_grade_was_reused() -> None:
    """A repeated answer's grade says it was reused."""
    reused = {**GRADED, "reused": True}
    assert result_markdown(reused).startswith(
        "**Verdict: partly correct** (reused: you gave this answer before)"
    )


@pytest.mark.parametrize("kind", ["no_support", "not_covered"])
def test_result_markdown_shows_the_message_instead_of_a_grade(kind: str) -> None:
    """No grade to show: the message the node packed, as it is."""
    assert result_markdown({"kind": kind, "message": "A message."}) == "A message."


def test_screen_after_the_answer_pause_shows_only_the_question() -> None:
    """Pause 1: the question with an open answer box; no grade, no passage, no choice."""
    screen = screen_after("s1", {"pause": "answer", "question": "Why?"})
    assert screen == Screen("s1", question="Why?", answering=True, choosing=False)


def test_screen_after_a_grade_keeps_the_question_and_adds_the_grade_and_passage() -> None:
    """Pause 2 after Submit: question and locked answer, the grade, the passage, the choice."""
    screen = screen_after("s1", {"pause": "next", "shown": GRADED}, "Why?", "my answer")
    assert screen.question == "Why?"
    assert screen.answer == "my answer"
    assert screen.answering is False
    assert screen.result == result_markdown(GRADED)
    assert screen.passage_label == "The passage: 1.3.1.2 Classification, page 10–11"
    assert screen.passage == "It is much …"
    assert screen.choosing is True


def test_screen_after_a_topic_not_covered_shows_only_the_message_and_the_choice() -> None:
    """No passage was found: no question, no passage; the message and the choice again."""
    shown = {"kind": "not_covered", "message": "The material doesn't cover this: 'pizza'."}
    screen = screen_after("s1", {"pause": "next", "shown": shown})
    assert screen == Screen("s1", result="The material doesn't cover this: 'pizza'.")


@pytest.mark.usefixtures("fake_grading")
def test_a_whole_session_click_by_click(tmp_path: Path) -> None:
    """You choose → the question → Submit → the grade → Stop → the start screen."""
    graph, path = build_graph(), make_db(tmp_path)

    asking = on_study(graph, path, None, None)
    assert asking.session_id is not None
    assert asking.question == "Why?"
    assert asking.answering is True

    graded = on_submit(graph, path, asking.session_id, "Why?", "it is easier")
    assert graded.session_id == asking.session_id
    assert graded.result.startswith("**Verdict: correct**")
    assert graded.passage == "It is much easier …"

    assert on_stop(graph, path, graded.session_id) == Screen(None, status=SESSION_ENDED)


def test_the_free_limit_ends_the_session_with_a_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Groq's free tier runs out while grading: the start screen says so, nothing crashes."""
    graph, path = build_graph(), make_db(tmp_path)

    def limited_grade(*_args: object) -> GradedAnswer:
        raise FreeLimitReached("free limit")

    monkeypatch.setattr(nodes, "grade_answer", limited_grade)
    asking = on_study(graph, path, None, None)
    assert on_submit(graph, path, asking.session_id, "Why?", "an answer") == Screen(
        None, status=FREE_LIMIT
    )


def test_a_lost_session_goes_back_to_the_start_with_a_message(tmp_path: Path) -> None:
    """A click in a tab whose session the server forgot (a restart) starts over, saying why."""
    screen = on_study(build_graph(), make_db(tmp_path), "forgotten-id", None)
    assert screen == Screen(None, status=SESSION_LOST)
