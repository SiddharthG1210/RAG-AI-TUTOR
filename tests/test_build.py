"""Tests for ``tutor.graph.build``: the whole study turn, run through LangGraph.

``tests/test_nodes.py`` calls each node on its own; these tests run the
compiled graph from START to END, the way ``scripts/study.py`` does, to check
the wiring: every node runs, in order, and hands the next one what it needs,
and every fork (M1 concept "Conditional edges") takes the right arrow. The
routing functions are also called on their own, with plain dicts. The LLM
tools, ``search`` and ``input`` are swapped for fakes (see
``tests/test_nodes.py``), and the database is in memory.
"""

import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest

from tutor.db import connect, get_question, list_passages, save_chapter, save_questions
from tutor.grading import Grade, GradedAnswer
from tutor.graph import nodes
from tutor.graph.build import build_graph, route_after_pick_passage, route_after_pick_question
from tutor.graph.state import Context
from tutor.ingest.pdf import Paragraph, Topic

RESULT = GradedAnswer(
    attempt_id=1,
    grade=Grade(quotes=["much easier"], missing=[], wrong=[], extras=[], verdict="correct"),
    supported=True,
    reused=False,
)


def one_passage(count: int = 1) -> sqlite3.Connection:
    """Make an in-memory database with passages of 1.3.1.2 Classification and no questions.

    Args:
        count: How many passages.

    Returns:
        The connection.
    """
    conn = connect(":memory:")
    topic = Topic("1.3.1.2", "Classification", 4, 49, "9")
    passages = [
        [Paragraph("1.3.1.2", 49, "9", f"It is much easier … ({n})", 49, "9")]
        for n in range(1, count + 1)
    ]
    save_chapter(conn, [topic], {"1.3.1.2": passages})
    return conn


def fake_tools(
    monkeypatch: pytest.MonkeyPatch, failing_writes: dict[int, int], steps: list[str]
) -> None:
    """Swap the LLM tools and input() for fakes that log each step.

    Args:
        monkeypatch: pytest's fixture.
        failing_writes: How many writes fail for each passage id, before one
            succeeds (a missing id never fails).
        steps: The list each fake appends what it did to.
    """

    def fake_write(conn: sqlite3.Connection, passage: sqlite3.Row) -> list[Any]:
        if failing_writes.get(passage["id"], 0) > 0:
            failing_writes[passage["id"]] -= 1
            steps.append(f"write {passage['id']}: none passed")
            return []
        steps.append(f"write {passage['id']}")
        questions = [SimpleNamespace(question="Why?", key_points=["a point"], quote="much easier")]
        save_questions(conn, passage["id"], questions, "v")
        return questions

    def fake_grade(_conn: sqlite3.Connection, question: sqlite3.Row, answer: str) -> GradedAnswer:
        steps.append(f"grade {question['passage_id']}")
        return RESULT

    def fake_input(_prompt: str = "") -> str:
        steps.append("input")
        return "it is easier"

    monkeypatch.setattr(nodes, "write_questions", fake_write)
    monkeypatch.setattr(nodes, "grade_answer", fake_grade)
    monkeypatch.setattr("builtins.input", fake_input)


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


@pytest.mark.parametrize(
    ("passage_id", "expected"), [(7, "pick_question"), (None, "topic_not_covered")]
)
def test_route_after_pick_passage(passage_id: int | None, expected: str) -> None:
    """A passage → pick its question; none → say the topic isn't covered."""
    assert route_after_pick_passage({"passage_id": passage_id}) == expected


@pytest.mark.parametrize(
    ("question_id", "write_tries", "expected"),
    [
        (12, 0, "ask_question"),  # from the pool, nothing written
        (12, 2, "ask_question"),  # the second write worked
        (None, 1, "pick_question"),  # the first write failed: write once more
        (None, 2, "skip_the_passage"),  # the second write failed too
    ],
)
def test_route_after_pick_question(
    question_id: int | None, write_tries: int, expected: str
) -> None:
    """A question → ask it; none after 1 write → write again; none after 2 → skip."""
    state = {"question_id": question_id, "write_tries": write_tries}
    assert route_after_pick_question(state) == expected


def test_every_name_a_route_returns_is_a_node() -> None:
    """A routing function naming a node that doesn't exist would end the turn silently.

    LangGraph only logs a warning ("wrote to unknown channel") and stops, so
    this checks the contract: every name the routing functions can return is
    a node of the compiled graph.
    """
    node_names = set(build_graph().get_graph().nodes)
    returned = {
        route_after_pick_passage({"passage_id": 7}),
        route_after_pick_passage({"passage_id": None}),
        route_after_pick_question({"question_id": 12, "write_tries": 0}),
        route_after_pick_question({"question_id": None, "write_tries": 1}),
        route_after_pick_question({"question_id": None, "write_tries": 2}),
    }
    assert returned <= node_names


def test_a_failed_write_is_tried_once_more(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first write saves nothing, the second works: the question is asked and graded."""
    conn = one_passage()
    passage_id = list_passages(conn)[0]["id"]
    steps: list[str] = []
    fake_tools(monkeypatch, {passage_id: 1}, steps)

    final = build_graph().invoke({"topic": None}, context=Context(conn=conn))

    assert steps == [
        f"write {passage_id}: none passed",
        f"write {passage_id}",
        "input",
        f"grade {passage_id}",
    ]
    assert final["write_tries"] == 2


def test_two_failed_writes_skip_the_passage_and_pick_another(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Passage 1 fails twice: it's saved as skipped, and passage 2 is studied instead."""
    conn = one_passage(count=2)
    first, second = (row["id"] for row in list_passages(conn))
    steps: list[str] = []
    fake_tools(monkeypatch, {first: 2}, steps)

    final = build_graph().invoke({"topic": None}, context=Context(conn=conn))

    assert steps == [
        f"write {first}: none passed",
        f"write {first}: none passed",
        f"write {second}",
        "input",
        f"grade {second}",
    ]
    assert final["passage_id"] == second
    # pick_passage started the count again for passage 2.
    assert final["write_tries"] == 1
    assert conn.execute("SELECT passage_id FROM skipped_passages").fetchone()[0] == first
    assert f"Skipping passage {first}" in capsys.readouterr().out


def test_a_topic_the_material_doesnt_cover_ends_the_turn(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing close enough: the message is shown, and no question is written or asked."""
    conn = one_passage()
    steps: list[str] = []
    fake_tools(monkeypatch, {}, steps)
    monkeypatch.setattr(nodes, "search", lambda _conn, _query: [])

    final = build_graph().invoke({"topic": "pizza recipes"}, context=Context(conn=conn))

    assert steps == []
    assert final["passage_id"] is None
    assert "The material doesn't cover this: 'pizza recipes'." in capsys.readouterr().out
