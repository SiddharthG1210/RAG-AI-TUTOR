"""Tests for ``tutor.api.session``: a study session driven one call at a time, as the page does.

Each call opens its own connection, so the database is a file in pytest's
``tmp_path`` (a fresh temporary folder for each test): an in-memory database
would be a new, empty one on every connection. Its one passage already has a
question in its pool, so no question is written, and the grading tool is
swapped for a fake inside ``tutor.graph.nodes``: no request reaches Groq.
"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from tutor.api.session import SessionLost, resume_session, start_session
from tutor.db import connect, list_passages, save_attempt, save_chapter, save_questions
from tutor.grading import Grade, GradedAnswer
from tutor.graph import nodes
from tutor.graph.build import build_graph
from tutor.ingest.pdf import Paragraph, Topic

GRADE = Grade(quotes=["much easier"], missing=[], wrong=[], extras=[], verdict="correct")
STOP = {"stop": True, "topic": None}
"""The reply to the "next" pause that ends the session."""


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


def test_start_session_makes_an_id_and_stops_at_the_first_pause(tmp_path: Path) -> None:
    """Start: a new 32-character id, and the "answer" pause's note with the question."""
    session_id, note = start_session(build_graph(), make_db(tmp_path), None)
    assert len(session_id) == 32
    assert note == {"pause": "answer", "question": "Why?"}


def test_every_session_gets_its_own_id(tmp_path: Path) -> None:
    """Two Starts, even on the same graph, are two separate sessions."""
    graph, path = build_graph(), make_db(tmp_path)
    first, _note = start_session(graph, path, None)
    second, _note = start_session(graph, path, None)
    assert first != second


@pytest.mark.usefixtures("fake_grading")
def test_resume_grades_the_answer_and_hands_out_what_to_show(tmp_path: Path) -> None:
    """Submit: the answer is graded, and the "next" pause hands out the grade and the passage."""
    graph, path = build_graph(), make_db(tmp_path)
    session_id, _note = start_session(graph, path, None)

    note = resume_session(graph, path, session_id, "it is easier")

    assert note["pause"] == "next"
    assert note["shown"]["verdict"] == "correct"
    assert note["shown"]["passage"]["text"] == "It is much easier …"
    # The attempt was saved with this session's id, by the call's own connection.
    conn = connect(path)
    assert conn.execute("SELECT session_id FROM attempts").fetchone()[0] == session_id
    conn.close()


@pytest.mark.usefixtures("fake_grading")
def test_stop_ends_the_session(tmp_path: Path) -> None:
    """Stop: the graph runs to its end, so there's no note, and nothing is left to resume."""
    graph, path = build_graph(), make_db(tmp_path)
    session_id, _note = start_session(graph, path, None)
    resume_session(graph, path, session_id, "it is easier")

    assert resume_session(graph, path, session_id, STOP) is None
    with pytest.raises(SessionLost):
        resume_session(graph, path, session_id, STOP)


def test_an_unknown_session_is_lost(tmp_path: Path) -> None:
    """An id the checkpointer has never seen raises SessionLost, instead of starting afresh."""
    with pytest.raises(SessionLost, match="no-such-id"):
        resume_session(build_graph(), make_db(tmp_path), "no-such-id", "an answer")


def test_a_new_graph_has_lost_every_session(tmp_path: Path) -> None:
    """A graph built again (as after a server restart) has an empty checkpointer.

    This is why the server builds its graph once: a graph per click would
    never find the session it should resume.
    """
    path = make_db(tmp_path)
    session_id, _note = start_session(build_graph(), path, None)
    with pytest.raises(SessionLost):
        resume_session(build_graph(), path, session_id, "an answer")


@pytest.mark.usefixtures("fake_grading")
def test_each_click_can_run_in_a_different_thread(tmp_path: Path) -> None:
    """Gradio runs each click in a worker thread; each call's own connection makes that work.

    A sqlite3 connection opened in one thread raises ProgrammingError when
    used in another. Start runs here, Submit in another thread, as two
    clicks would.
    """
    graph, path = build_graph(), make_db(tmp_path)
    session_id, _note = start_session(graph, path, None)

    # A pool with one worker thread; .result() waits for the call to finish
    # there, and re-raises here anything it raised.
    with ThreadPoolExecutor(max_workers=1) as pool:
        note = pool.submit(resume_session, graph, path, session_id, "it is easier").result()

    assert note["shown"]["verdict"] == "correct"
