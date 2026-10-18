"""Tests for ``tutor.graph.nodes``: each step of a study turn, called on its own.

A node is a plain function, so a test can call it directly with a state dict
and a ``Runtime`` holding our ``Context``, exactly as LangGraph would. No
request ever reaches Groq: ``monkeypatch`` swaps the two tools that call the
LLM (``write_questions`` and ``grade_answer``) and ``search`` (which would load
the embedding model) for fakes, inside ``tutor.graph.nodes`` only, and
``builtins.input`` for a list of typed lines. The database is in memory.
"""

import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.runtime import Runtime

from tutor.db import connect, get_question, list_passages, save_chapter, save_questions
from tutor.grading import Grade, GradedAnswer
from tutor.graph import nodes
from tutor.graph.nodes import TurnStopped, ask, grade, pick_passage, pick_question, show
from tutor.graph.state import Context
from tutor.ingest.pdf import Paragraph, Topic

GRADE = Grade(
    quotes=["much easier"],
    missing=["a firm label is hard to optimize"],
    wrong=[],
    extras=["softmax turns scores into probabilities"],
    verdict="partly correct",
)


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
    return conn, Runtime(context=Context(conn=conn)), ids


def test_you_choose_picks_the_next_new_passage() -> None:
    """With no topic, the first passage in book order is picked."""
    _conn, runtime, ids = setup()
    assert pick_passage({"topic": None}, runtime) == {"passage_id": ids[0]}


def test_a_typed_topic_picks_the_best_search_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first (best) pair search() returns is picked; the topic is passed on."""
    conn, runtime, ids = setup()
    second = list_passages(conn)[1]
    asked: list[str] = []

    def fake_search(_conn: sqlite3.Connection, query: str) -> list[tuple[float, Any]]:
        asked.append(query)
        return [(0.8, second)]

    monkeypatch.setattr(nodes, "search", fake_search)
    assert pick_passage({"topic": "classification"}, runtime) == {"passage_id": ids[1]}
    assert asked == ["classification"]


def test_a_topic_the_material_doesnt_cover_stops_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty search result stops the turn with PLAN.md's message."""
    _conn, runtime, _ids = setup()
    monkeypatch.setattr(nodes, "search", lambda _conn, _query: [])
    with pytest.raises(TurnStopped, match="The material doesn't cover this"):
        pick_passage({"topic": "pizza recipes"}, runtime)


def test_a_question_from_the_pool_needs_no_llm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """With questions in the pool, the oldest unused one is picked and nothing is written."""
    conn, runtime, ids = setup()
    save_questions(conn, ids[0], [a_question("First?"), a_question("Second?")], "v")
    monkeypatch.setattr(nodes, "write_questions", lambda *_: pytest.fail("no LLM call expected"))
    result = pick_question({"passage_id": ids[0]}, runtime)
    assert get_question(conn, result["question_id"])["text"] == "First?"


def test_an_empty_pool_is_written_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty pool calls write_questions() once, then picks from what it saved."""
    conn, runtime, ids = setup()
    written: list[int] = []

    def fake_write(conn: sqlite3.Connection, passage: sqlite3.Row) -> list[Any]:
        # Like the real one: save the questions that passed, return them.
        written.append(passage["id"])
        questions = [a_question("Why?")]
        save_questions(conn, passage["id"], questions, "v")
        return questions

    monkeypatch.setattr(nodes, "write_questions", fake_write)
    result = pick_question({"passage_id": ids[1]}, runtime)
    assert written == [ids[1]]
    assert get_question(conn, result["question_id"])["text"] == "Why?"


def test_no_question_passing_its_quote_check_stops_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When write_questions() saves nothing, the turn stops with a clear message."""
    _conn, runtime, ids = setup()
    monkeypatch.setattr(nodes, "write_questions", lambda *_: [])
    with pytest.raises(TurnStopped, match="passed the quote check"):
        pick_question({"passage_id": ids[0]}, runtime)


def test_ask_prints_the_question_and_asks_again_on_an_empty_answer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Blank lines are refused, so an empty answer never reaches the grader."""
    conn, runtime, ids = setup()
    save_questions(conn, ids[0], [a_question("Why probabilities?")], "v")
    question_id = list(conn.execute("SELECT id FROM questions"))[0]["id"]
    typed = iter(["", "   ", "it shows how sure the model is"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(typed))

    assert ask({"question_id": question_id}, runtime) == {
        "answer": "it shows how sure the model is"
    }
    assert "Question: Why probabilities?" in capsys.readouterr().out


def test_grade_grades_then_marks_the_question_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """grade_answer() gets the question row and the answer; afterwards the question is used."""
    conn, runtime, ids = setup()
    save_questions(conn, ids[0], [a_question("Why?")], "v")
    question_id = list(conn.execute("SELECT id FROM questions"))[0]["id"]
    result = GradedAnswer(attempt_id=1, grade=GRADE, supported=True, reused=False)
    calls: list[tuple[int, str]] = []

    def fake_grade(_conn: sqlite3.Connection, question: sqlite3.Row, answer: str) -> GradedAnswer:
        # The question isn't marked used yet while it's being graded.
        assert question["used"] == 0
        calls.append((question["id"], answer))
        return result

    monkeypatch.setattr(nodes, "grade_answer", fake_grade)
    assert grade({"question_id": question_id, "answer": "an answer"}, runtime) == {"result": result}
    assert calls == [(question_id, "an answer")]
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
