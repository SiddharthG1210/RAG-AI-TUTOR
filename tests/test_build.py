"""Tests for ``tutor.graph.build``: whole study sessions, run through LangGraph.

``tests/test_nodes.py`` calls each node on its own; these tests run the
compiled graph the way ``scripts/study.py`` does, to check the wiring: every
node runs in order and hands the next one what it needs, every fork (M1
concept "Conditional edges") takes the right arrow, and both pauses (M1
concept "Pausing for your answer") stop the graph and resume it in the right
place. The routing functions are also called on their own, with plain dicts.

The graph pauses at ``interrupt()``, so a test plays the caller: ``run()``
starts the graph, then answers each pause from a list of scripted replies
and resumes it, exactly like the loop in ``scripts/study.py``. The LLM tools
and ``search`` are swapped for fakes (see ``tests/test_nodes.py``), and the
database is in memory.
"""

import sqlite3
from types import SimpleNamespace
from typing import Any

import pytest
from langgraph.graph import END
from langgraph.types import Command

from tutor.db import (
    connect,
    get_question,
    list_passages,
    save_attempt,
    save_chapter,
    save_questions,
)
from tutor.grading import Grade, GradedAnswer
from tutor.graph import nodes
from tutor.graph.build import (
    build_graph,
    route_after_pick_passage,
    route_after_pick_question,
    route_after_wait_for_next,
)
from tutor.graph.state import Context
from tutor.ingest.pdf import Paragraph, Topic

GRADE = Grade(quotes=["much easier"], missing=[], wrong=[], extras=[], verdict="correct")
SESSION = "session-1"
STOP = {"stop": True, "topic": None}
"""The reply to the "next" pause that ends the session."""
YOU_CHOOSE = {"stop": False, "topic": None}
"""The reply to the "next" pause that starts another turn with "You choose"."""


def passages(count: int = 1) -> sqlite3.Connection:
    """Make an in-memory database with passages of 1.3.1.2 Classification and no questions.

    Args:
        count: How many passages.

    Returns:
        The connection.
    """
    conn = connect(":memory:")
    topic = Topic("1.3.1.2", "Classification", 4, 49, "9")
    paragraphs = [
        [Paragraph("1.3.1.2", 49, "9", f"It is much easier … ({n})", 49, "9")]
        for n in range(1, count + 1)
    ]
    save_chapter(conn, [topic], {"1.3.1.2": paragraphs})
    return conn


def fake_tools(
    monkeypatch: pytest.MonkeyPatch, failing_writes: dict[int, int], steps: list[str]
) -> None:
    """Swap the two LLM tools for fakes that log each step.

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

    def fake_grade(
        conn: sqlite3.Connection, question: sqlite3.Row, answer: str, session_id: str | None
    ) -> GradedAnswer:
        # Like the real one: save the attempt, with its session id.
        steps.append(f"grade {question['passage_id']}")
        attempt_id = save_attempt(
            conn, question["id"], session_id, answer, answer, GRADE, True, "v", "m", "s"
        )
        return GradedAnswer(attempt_id, GRADE, supported=True, reused=False)

    monkeypatch.setattr(nodes, "write_questions", fake_write)
    monkeypatch.setattr(nodes, "grade_answer", fake_grade)


def run(
    conn: sqlite3.Connection, topic: str | None, replies: list[Any], notes: list[dict] | None = None
) -> dict:
    """Play the caller: start a session, answer each pause with the next scripted reply, resume.

    The same loop as ``scripts/study.py``, with replies from a list instead of
    your keyboard.

    Args:
        conn: The database.
        topic: The first turn's topic, or None for "You choose".
        replies: One reply per pause, in order: an answer (text) for each
            "answer" pause, a ``{"stop": ..., "topic": ...}`` dict for each
            "next" pause.
        notes: If given, every note the pauses hand out is appended to it.

    Returns:
        The final state.

    Raises:
        AssertionError: If the graph pauses more often than there are replies,
            or a reply is left over at the end.
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": SESSION}}
    context = Context(conn=conn, session_id=SESSION)
    result = graph.invoke({"topic": topic}, config, context=context)
    replies = list(replies)
    while "__interrupt__" in result:
        note = result["__interrupt__"][0].value
        if notes is not None:
            notes.append(note)
        assert replies, f"the graph paused with {note}, but no reply is left"
        result = graph.invoke(Command(resume=replies.pop(0)), config, context=context)
    assert not replies, f"replies left over: {replies}"
    return result


@pytest.mark.parametrize(
    ("passage_id", "expected"), [(7, "pick_question"), (None, "topic_not_covered")]
)
def test_route_after_pick_passage(passage_id: int | None, expected: str) -> None:
    """A passage → pick its question; none → say why."""
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


@pytest.mark.parametrize(("stop", "expected"), [(True, END), (False, "pick_passage")])
def test_route_after_wait_for_next(stop: bool, expected: str) -> None:
    """Stop → the end of the session; anything else → the next turn."""
    assert route_after_wait_for_next({"stop": stop}) == expected


def test_every_name_a_route_returns_is_a_node() -> None:
    """A routing function naming a node that doesn't exist would end the session silently.

    LangGraph only logs a warning ("wrote to unknown channel") and stops, so
    this checks the contract: every name the routing functions can return is
    a node of the compiled graph (``END`` included, as ``"__end__"``).
    """
    node_names = set(build_graph().get_graph().nodes)
    returned = {
        route_after_pick_passage({"passage_id": 7}),
        route_after_pick_passage({"passage_id": None}),
        route_after_pick_question({"question_id": 12, "write_tries": 0}),
        route_after_pick_question({"question_id": None, "write_tries": 1}),
        route_after_pick_question({"question_id": None, "write_tries": 2}),
        route_after_wait_for_next({"stop": True}),
        route_after_wait_for_next({"stop": False}),
    }
    assert returned <= node_names


def test_the_first_pause_hands_out_the_question_and_waits() -> None:
    """Call 1 stops at ask_question, handing out the question; nothing is graded yet."""
    conn = passages()
    passage_id = list_passages(conn)[0]["id"]
    save_questions(
        conn, passage_id, [SimpleNamespace(question="Why?", key_points=[], quote="")], "v"
    )
    graph = build_graph()
    config = {"configurable": {"thread_id": SESSION}}

    result = graph.invoke({"topic": None}, config, context=Context(conn, SESSION))

    assert result["__interrupt__"][0].value == {"pause": "answer", "question": "Why?"}
    # The checkpointer saved where it stopped: resuming continues in ask_question.
    assert graph.get_state(config).next == ("ask_question",)
    assert conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0] == 0


def test_one_turn_then_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """You choose → questions written → pause for the answer → graded, shown → pause → stop."""
    conn = passages()
    passage_id = list_passages(conn)[0]["id"]
    steps: list[str] = []
    notes: list[dict] = []
    fake_tools(monkeypatch, {}, steps)

    final = run(conn, None, ["it is easier", STOP], notes)

    assert steps == [f"write {passage_id}", f"grade {passage_id}"]
    assert notes[0] == {"pause": "answer", "question": "Why?"}
    # The second pause hands out what show packed: the grade and the passage.
    assert notes[1]["pause"] == "next"
    assert notes[1]["shown"]["kind"] == "graded"
    assert notes[1]["shown"]["verdict"] == "correct"
    assert notes[1]["shown"]["passage"]["section"] == "1.3.1.2 Classification"
    assert final["answer"] == "it is easier"
    assert final["result"].grade.verdict == "correct"
    assert final["stop"] is True
    # The question counts as asked only now, after its grade.
    assert get_question(conn, final["question_id"])["used"] == 1
    # The attempt carries the session id.
    assert conn.execute("SELECT session_id FROM attempts").fetchone()[0] == SESSION


def test_next_starts_another_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a turn, Enter ("You choose") studies the next new passage, then s stops."""
    conn = passages(count=2)
    first, second = (row["id"] for row in list_passages(conn))
    steps: list[str] = []
    fake_tools(monkeypatch, {}, steps)

    final = run(conn, None, ["answer one", YOU_CHOOSE, "answer two", STOP])

    assert steps == [f"write {first}", f"grade {first}", f"write {second}", f"grade {second}"]
    assert final["passage_id"] == second


def test_the_same_topic_again_moves_on_to_the_next_best_passage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Typing a topic again in a session skips the passages already asked, then says so."""
    conn = passages(count=2)
    rows = list_passages(conn)
    first, second = (row["id"] for row in rows)
    steps: list[str] = []
    notes: list[dict] = []
    fake_tools(monkeypatch, {}, steps)
    monkeypatch.setattr(nodes, "search", lambda _conn, _query: [(0.9, rows[0]), (0.8, rows[1])])
    again = {"stop": False, "topic": "classification"}

    # Turn 1: the best match. Turn 2: the next best. Turn 3: nothing new left.
    run(conn, "classification", ["a", again, "b", again, STOP], notes)

    assert steps == [f"write {first}", f"grade {first}", f"write {second}", f"grade {second}"]
    assert notes[-1]["shown"]["kind"] == "not_covered"
    assert "Nothing new on 'classification' this session" in notes[-1]["shown"]["message"]


def test_a_failed_write_is_tried_once_more(monkeypatch: pytest.MonkeyPatch) -> None:
    """The first write saves nothing, the second works: the question is asked and graded."""
    conn = passages()
    passage_id = list_passages(conn)[0]["id"]
    steps: list[str] = []
    fake_tools(monkeypatch, {passage_id: 1}, steps)

    final = run(conn, None, ["it is easier", STOP])

    assert steps == [
        f"write {passage_id}: none passed",
        f"write {passage_id}",
        f"grade {passage_id}",
    ]
    assert final["write_tries"] == 2


def test_two_failed_writes_skip_the_passage_and_pick_another(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Passage 1 fails twice: it's saved as skipped, and passage 2 is studied instead."""
    conn = passages(count=2)
    first, second = (row["id"] for row in list_passages(conn))
    steps: list[str] = []
    fake_tools(monkeypatch, {first: 2}, steps)

    final = run(conn, None, ["it is easier", STOP])

    assert steps == [
        f"write {first}: none passed",
        f"write {first}: none passed",
        f"write {second}",
        f"grade {second}",
    ]
    assert final["passage_id"] == second
    # pick_passage started the count again for passage 2.
    assert final["write_tries"] == 1
    assert conn.execute("SELECT passage_id FROM skipped_passages").fetchone()[0] == first
    # Logged on the console, not shown on the page.
    assert f"Skipping passage {first}" in caplog.text


def test_a_topic_the_material_doesnt_cover_asks_for_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing close: the message is shown, nothing is written or asked, and the session goes on."""
    conn = passages()
    steps: list[str] = []
    notes: list[dict] = []
    fake_tools(monkeypatch, {}, steps)
    monkeypatch.setattr(nodes, "search", lambda _conn, _query: [])

    final = run(conn, "pizza recipes", [STOP], notes)

    assert steps == []
    # Straight to "what next?", not the end, handing out the message to show.
    assert notes == [
        {
            "pause": "next",
            "shown": {
                "kind": "not_covered",
                "message": "The material doesn't cover this: 'pizza recipes'.",
            },
        }
    ]
    assert final["passage_id"] is None
