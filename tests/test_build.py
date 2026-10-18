"""Tests for ``tutor.graph.build``: the whole study turn, run through LangGraph.

``tests/test_nodes.py`` calls each node on its own; these tests run the
compiled graph from START to END, the way ``scripts/study.py`` does, to check
the wiring: every node runs, in order, and hands the next one what it needs.
The LLM tools, ``search`` and ``input`` are swapped for fakes (see
``tests/test_nodes.py``), and the database is in memory.
"""

import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

from tutor.db import connect, get_question, list_passages, save_chapter, save_questions
from tutor.grading import Grade, GradedAnswer
from tutor.graph import nodes
from tutor.graph.build import build_graph
from tutor.graph.nodes import TurnStopped
from tutor.graph.state import Context
from tutor.ingest.pdf import Paragraph, Topic

RESULT = GradedAnswer(
    attempt_id=1,
    grade=Grade(quotes=["much easier"], missing=[], wrong=[], extras=[], verdict="correct"),
    supported=True,
    reused=False,
)


def one_passage() -> sqlite3.Connection:
    """Make an in-memory database with one passage of 1.3.1.2 Classification and no questions.

    Returns:
        The connection.
    """
    conn = connect(":memory:")
    topic = Topic("1.3.1.2", "Classification", 4, 49, "9")
    passage = [Paragraph("1.3.1.2", 49, "9", "It is much easier …", 49, "9")]
    save_chapter(conn, [topic], {"1.3.1.2": [passage]})
    return conn


def test_one_turn_runs_every_step_in_order(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """You choose → questions written → asked → graded → shown; the final state has every key."""
    conn = one_passage()
    passage_id = list_passages(conn)[0]["id"]
    steps: list[str] = []

    def fake_write(conn: sqlite3.Connection, passage: sqlite3.Row) -> list[Any]:
        steps.append("write")
        questions = [SimpleNamespace(question="Why?", key_points=["a point"], quote="much easier")]
        save_questions(conn, passage["id"], questions, "v")
        return questions

    def fake_grade(_conn: sqlite3.Connection, question: sqlite3.Row, answer: str) -> GradedAnswer:
        steps.append(f"grade {question['text']} / {answer}")
        return RESULT

    def fake_input(_prompt: str = "") -> str:
        steps.append("input")
        return "it is easier"

    monkeypatch.setattr(nodes, "write_questions", fake_write)
    monkeypatch.setattr(nodes, "grade_answer", fake_grade)
    monkeypatch.setattr("builtins.input", fake_input)

    final = build_graph().invoke({"topic": None}, context=Context(conn=conn))

    # The steps ran in the chain's order, each with what the one before handed it.
    assert steps == ["write", "input", "grade Why? / it is easier"]
    assert final["passage_id"] == passage_id
    assert final["answer"] == "it is easier"
    assert final["result"] == RESULT
    # The question counts as asked only now, after its grade.
    assert get_question(conn, final["question_id"])["used"] == 1
    assert "Verdict: correct" in capsys.readouterr().out


def test_a_stopped_turn_reaches_the_starter(monkeypatch: pytest.MonkeyPatch) -> None:
    """TurnStopped raised inside a node comes out of invoke() as itself, for study.py to catch."""
    conn = one_passage()
    monkeypatch.setattr(nodes, "search", lambda _conn, _query: [])
    with pytest.raises(TurnStopped, match="doesn't cover this"):
        build_graph().invoke({"topic": "pizza recipes"}, context=Context(conn=conn))
