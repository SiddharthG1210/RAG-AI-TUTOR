"""Tests for ``tutor.graph.nodes``: each step of a study turn, called on its own.

A node is a plain function, so a test can call it directly with a state dict
and a ``Runtime`` holding our ``Context``, exactly as LangGraph would. No
request ever reaches Groq: ``monkeypatch`` swaps the two tools that call the
LLM (``write_questions`` and ``grade_answer``) and ``search`` (which would load
the embedding model) for fakes, inside ``tutor.graph.nodes`` only. The
database is in memory.

The two pausing nodes (``ask`` and ``wait_for_next``) call ``interrupt()``,
which only works inside a running graph, so ``tests/test_build.py`` tests
them through the compiled graph.
"""

import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.runtime import Runtime

from tutor.db import (
    connect,
    get_question,
    list_passages,
    save_attempt,
    save_chapter,
    save_questions,
    save_skip,
)
from tutor.grading import Grade, GradedAnswer
from tutor.graph import nodes
from tutor.graph.nodes import (
    grade,
    not_covered,
    pick_passage,
    pick_question,
    show,
    skip_passage,
)
from tutor.graph.state import Context
from tutor.ingest.pdf import Paragraph, Topic
from tutor.llm import FreeLimitReached, UnusableAnswer

GRADE = Grade(
    quotes=["much easier"],
    missing=["a firm label is hard to optimize"],
    wrong=[],
    extras=["softmax turns scores into probabilities"],
    verdict="partly correct",
)

SESSION = "session-1"
"""The session id every test's Context carries."""


def a_question(text: str) -> SimpleNamespace:
    """Build something shaped like a ``Question``, for ``save_questions()``.

    Args:
        text: The question.

    Returns:
        An object with question, key_points and quote.
    """
    return SimpleNamespace(question=text, key_points=["a point"], quote="much easier")


def setup() -> tuple[sqlite3.Connection, Runtime[Context], list[int]]:
    """Make an in-memory database with two passages of 1.3.1.2 Classification.

    Returns:
        The connection, a Runtime carrying it (what LangGraph hands each
        node), and the two passages' ids in book order.
    """
    conn = connect(":memory:")
    topic = Topic("1.3.1.2", "Classification", 4, 49, "9")
    passages = [
        [Paragraph("1.3.1.2", 49, "9", "First passage …", 49, "9")],
        [Paragraph("1.3.1.2", 50, "10", "It is much easier …", 51, "11")],
    ]
    save_chapter(conn, [topic], {"1.3.1.2": passages})
    ids = [row["id"] for row in list_passages(conn)]
    return conn, Runtime(context=Context(conn=conn, session_id=SESSION)), ids


def test_you_choose_picks_the_next_new_passage() -> None:
    """With no topic, the first passage in book order is picked, and its write tries start at 0."""
    _conn, runtime, ids = setup()
    assert pick_passage({"topic": None}, runtime) == {
        "passage_id": ids[0],
        "write_tries": 0,
        "nothing_new": False,
    }


def test_you_choose_passes_over_a_skipped_passage() -> None:
    """A skipped passage is never new again, so "You choose" moves on to the next one."""
    conn, runtime, ids = setup()
    save_skip(conn, ids[0], "no question passed")
    assert pick_passage({"topic": None}, runtime)["passage_id"] == ids[1]


def test_a_typed_topic_picks_the_best_search_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first (best) pair search() returns is picked; the topic is passed on."""
    conn, runtime, ids = setup()
    second = list_passages(conn)[1]
    asked: list[str] = []

    def fake_search(_conn: sqlite3.Connection, query: str) -> list[tuple[float, Any]]:
        asked.append(query)
        return [(0.8, second)]

    monkeypatch.setattr(nodes, "search", fake_search)
    assert pick_passage({"topic": "classification"}, runtime) == {
        "passage_id": ids[1],
        "write_tries": 0,
        "nothing_new": False,
    }
    assert asked == ["classification"]


def test_a_typed_topic_passes_over_a_skipped_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the best match is skipped, the next best one is picked."""
    conn, runtime, ids = setup()
    first, second = list_passages(conn)
    save_skip(conn, ids[0], "no question passed")
    monkeypatch.setattr(nodes, "search", lambda _conn, _query: [(0.9, first), (0.8, second)])
    assert pick_passage({"topic": "classification"}, runtime)["passage_id"] == ids[1]


def test_a_typed_topic_passes_over_a_passage_asked_this_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typing the same topic again in a session moves on to the next-best passage.

    An answer in another session doesn't count: that passage can come back.
    """
    conn, runtime, ids = setup()
    first, second = list_passages(conn)
    save_questions(conn, ids[0], [a_question("Why?")], "v")
    question_id = list(conn.execute("SELECT id FROM questions"))[0]["id"]
    monkeypatch.setattr(nodes, "search", lambda _conn, _query: [(0.9, first), (0.8, second)])

    def answered_in(session_id: str) -> None:
        save_attempt(conn, question_id, session_id, "a", "a", GRADE, True, "v", "m", "s")

    answered_in("another session")
    assert pick_passage({"topic": "classification"}, runtime)["passage_id"] == ids[0]
    answered_in(SESSION)
    assert pick_passage({"topic": "classification"}, runtime)["passage_id"] == ids[1]


@pytest.mark.parametrize(
    "skip_both", [False, True], ids=["nothing close", "all close ones skipped"]
)
def test_no_passage_for_a_typed_topic_records_none(
    monkeypatch: pytest.MonkeyPatch, skip_both: bool
) -> None:
    """No passage close enough (or only skipped ones): passage_id is None, for routing."""
    conn, runtime, ids = setup()
    results: list[tuple[float, Any]] = []
    if skip_both:
        for passage_id in ids:
            save_skip(conn, passage_id, "no question passed")
        results = [(0.9, row) for row in list_passages(conn)]
    monkeypatch.setattr(nodes, "search", lambda _conn, _query: results)
    # Close passages that were all left out mean "nothing new", not "not covered".
    assert pick_passage({"topic": "pizza recipes"}, runtime) == {
        "passage_id": None,
        "write_tries": 0,
        "nothing_new": skip_both,
    }


def test_a_question_from_the_pool_needs_no_llm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A full pool: the oldest unused question is picked; nothing is written or counted."""
    conn, runtime, ids = setup()
    save_questions(conn, ids[0], [a_question("First?"), a_question("Second?")], "v")
    monkeypatch.setattr(nodes, "write_questions", lambda *_: pytest.fail("no LLM call expected"))
    result = pick_question({"passage_id": ids[0], "write_tries": 0}, runtime)
    assert get_question(conn, result["question_id"])["text"] == "First?"
    assert result["write_tries"] == 0


def test_an_empty_pool_is_written_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty pool calls write_questions() once, counts the try, then picks from what it saved."""
    conn, runtime, ids = setup()
    written: list[int] = []

    def fake_write(conn: sqlite3.Connection, passage: sqlite3.Row) -> list[Any]:
        # Like the real one: save the questions that passed, return them.
        written.append(passage["id"])
        questions = [a_question("Why?")]
        save_questions(conn, passage["id"], questions, "v")
        return questions

    monkeypatch.setattr(nodes, "write_questions", fake_write)
    result = pick_question({"passage_id": ids[1], "write_tries": 0}, runtime)
    assert written == [ids[1]]
    assert get_question(conn, result["question_id"])["text"] == "Why?"
    assert result["write_tries"] == 1


def test_no_question_passing_its_quote_check_records_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When write_questions() saves nothing, question_id is None and the try is counted."""
    _conn, runtime, ids = setup()
    monkeypatch.setattr(nodes, "write_questions", lambda *_: [])
    assert pick_question({"passage_id": ids[0], "write_tries": 1}, runtime) == {
        "question_id": None,
        "write_tries": 2,
    }


def test_an_unusable_llm_answer_counts_as_a_failed_try(monkeypatch: pytest.MonkeyPatch) -> None:
    """UnusableAnswer (e.g. 2 questions instead of 3) is caught: a failed try, not a crash."""
    _conn, runtime, ids = setup()

    def broken_write(_conn: sqlite3.Connection, _passage: sqlite3.Row) -> list[Any]:
        raise UnusableAnswer("The questions answer broke a rule: 2 questions")

    monkeypatch.setattr(nodes, "write_questions", broken_write)
    assert pick_question({"passage_id": ids[0], "write_tries": 0}, runtime) == {
        "question_id": None,
        "write_tries": 1,
    }


def test_the_free_limit_still_stops_the_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only an unusable answer counts as a failed try; FreeLimitReached goes on to the starter."""
    _conn, runtime, ids = setup()

    def limited_write(_conn: sqlite3.Connection, _passage: sqlite3.Row) -> list[Any]:
        raise FreeLimitReached("free limit")

    monkeypatch.setattr(nodes, "write_questions", limited_write)
    with pytest.raises(FreeLimitReached):
        pick_question({"passage_id": ids[0], "write_tries": 0}, runtime)


def test_grade_grades_then_marks_the_question_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """grade_answer() gets the question row and the answer; afterwards the question is used."""
    conn, runtime, ids = setup()
    save_questions(conn, ids[0], [a_question("Why?")], "v")
    question_id = list(conn.execute("SELECT id FROM questions"))[0]["id"]
    result = GradedAnswer(attempt_id=1, grade=GRADE, supported=True, reused=False)
    calls: list[tuple[int, str, str | None]] = []

    def fake_grade(
        _conn: sqlite3.Connection, question: sqlite3.Row, answer: str, session_id: str | None
    ) -> GradedAnswer:
        # The question isn't marked used yet while it's being graded.
        assert question["used"] == 0
        calls.append((question["id"], answer, session_id))
        return result

    monkeypatch.setattr(nodes, "grade_answer", fake_grade)
    assert grade({"question_id": question_id, "answer": "an answer"}, runtime) == {"result": result}
    # The session id goes along, to be saved with the attempt.
    assert calls == [(question_id, "an answer", SESSION)]
    assert get_question(conn, question_id)["used"] == 1


def test_show_prints_the_grade_and_the_passage_with_its_pages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A supported grade: the verdict, the non-empty lists, then the passage and page range."""
    _conn, runtime, ids = setup()
    result = GradedAnswer(attempt_id=1, grade=GRADE, supported=True, reused=False)
    assert show({"passage_id": ids[1], "result": result}, runtime) == {}
    out = capsys.readouterr().out
    assert "Verdict: partly correct" in out
    assert "Missing:\n  - a firm label is hard to optimize" in out
    assert "Wrong:" not in out  # an empty list isn't printed
    assert "1.3.1.2 Classification, page 10–11" in out
    assert "It is much easier …" in out


def test_show_an_unsupported_grade_says_so_instead(capsys: pytest.CaptureFixture[str]) -> None:
    """An unsupported grade isn't shown; PLAN.md's message is, and the passage still is."""
    _conn, runtime, ids = setup()
    result = GradedAnswer(attempt_id=1, grade=GRADE, supported=False, reused=False)
    show({"passage_id": ids[0], "result": result}, runtime)
    out = capsys.readouterr().out
    assert "I couldn't find support for this grade." in out
    assert "Verdict" not in out
    assert "First passage …" in out


def test_not_covered_says_so_with_the_topic(capsys: pytest.CaptureFixture[str]) -> None:
    """Nothing close: PLAN.md's fallback message, naming what you typed."""
    _conn, runtime, _ids = setup()
    assert not_covered({"topic": "pizza recipes", "nothing_new": False}, runtime) == {}
    assert "The material doesn't cover this: 'pizza recipes'." in capsys.readouterr().out


def test_not_covered_says_when_there_is_nothing_new(capsys: pytest.CaptureFixture[str]) -> None:
    """Close passages, all asked this session: it says that, not "doesn't cover this"."""
    _conn, runtime, _ids = setup()
    not_covered({"topic": "classification", "nothing_new": True}, runtime)
    out = capsys.readouterr().out
    assert "Nothing new on 'classification' this session" in out
    assert "doesn't cover" not in out


def test_skip_passage_saves_the_skip(capsys: pytest.CaptureFixture[str]) -> None:
    """The skip lands in the database, so neither picker chooses the passage again."""
    conn, runtime, ids = setup()
    assert skip_passage({"passage_id": ids[0]}, runtime) == {}
    row = conn.execute("SELECT passage_id, reason FROM skipped_passages").fetchone()
    assert tuple(row) == (ids[0], "no question passed its quote check in 2 tries")
    assert f"Skipping passage {ids[0]}" in capsys.readouterr().out
