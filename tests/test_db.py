"""Tests for ``tutor.db``: the topics, passages, questions, llm_calls and attempts tables.

Every test uses an in-memory database (``":memory:"``): SQLite builds it in
RAM, it starts empty, and it disappears when the connection closes, so tests
never touch ``data/tutor.db`` and never affect each other.
"""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from tutor.db import (
    connect,
    find_attempt,
    get_question,
    list_passages,
    list_questions,
    list_topics,
    log_llm_call,
    mark_question_used,
    next_new_passage,
    next_unused_question,
    save_attempt,
    save_chapter,
    save_questions,
)
from tutor.grading import Grade
from tutor.ingest.pdf import Paragraph, Topic

TOPICS = [
    Topic("1", "Introduction", 1, 41, "1"),
    Topic("1.3.1.2", "Classification", 4, 49, "9"),
]


def para(text: str, page: int, end_page: int | None = None) -> Paragraph:
    """Build a paragraph of topic 1.3.1.2, printed 40 pages below its PDF page, as in d2l.

    Args:
        text: The paragraph's text.
        page: The PDF page it starts on.
        end_page: The PDF page it ends on; the start page if not given.

    Returns:
        The paragraph.
    """
    end = end_page or page
    return Paragraph("1.3.1.2", page, str(page - 40), text, end, str(end - 40))


def test_topics_come_back_in_book_order() -> None:
    """Saved topics are read back with every column, in the order they were saved."""
    conn = connect(":memory:")
    save_chapter(conn, TOPICS, {})
    rows = list_topics(conn)
    assert [(r["number"], r["title"], r["level"], r["page"], r["page_label"]) for r in rows] == [
        ("1", "Introduction", 1, 41, "1"),
        ("1.3.1.2", "Classification", 4, 49, "9"),
    ]


def test_passages_keep_their_topic_order_text_and_page_range() -> None:
    """Each passage is saved under its topic, in order, from first to last page."""
    conn = connect(":memory:")
    passages = {
        "1.3.1.2": [
            [para("While regression models …", 49, 50), para("In classification, …", 50)],
            [para("Note that the most likely class …", 50, 51)],
        ]
    }
    save_chapter(conn, TOPICS, passages)
    rows = list_passages(conn)
    assert [(r["number"], r["position"], r["page_label"], r["end_page_label"]) for r in rows] == [
        ("1.3.1.2", 1, "9", "10"),
        ("1.3.1.2", 2, "10", "11"),
    ]
    # A passage's paragraphs are stored with a blank line between them.
    assert rows[0]["text"] == "While regression models …\n\nIn classification, …"


def test_ingesting_a_chapter_twice_fails_and_saves_nothing() -> None:
    """A repeated topic number raises, and the whole second save is rolled back."""
    conn = connect(":memory:")
    save_chapter(conn, TOPICS[:1], {})
    with pytest.raises(ValueError, match="already in the database"):
        # The new topic comes first, so it would be saved if there were no transaction.
        save_chapter(conn, [Topic("2", "Preliminaries", 1, 70, "30"), TOPICS[0]], {})
    assert [r["number"] for r in list_topics(conn)] == ["1"]


def test_questions_are_saved_in_their_passages_pool() -> None:
    """Saved questions come back for their passage, unused, with their key points as JSON."""
    conn = connect(":memory:")
    save_chapter(conn, TOPICS, {"1.3.1.2": [[para("In classification, …", 49)]]})
    passage_id = list_passages(conn)[0]["id"]
    # Any object with question, key_points and quote works; the real one is
    # tutor.questions.Question.
    question = SimpleNamespace(
        question="Why probabilities?", key_points=["easier to optimize"], quote="much easier"
    )
    save_questions(conn, passage_id, [question], "abc123def456")

    [row] = list_questions(conn, passage_id)
    assert (row["text"], row["quote"], row["prompt_version"], row["used"]) == (
        "Why probabilities?",
        "much easier",
        "abc123def456",
        0,
    )
    assert json.loads(row["key_points"]) == ["easier to optimize"]


def three_passages(conn: sqlite3.Connection) -> list[int]:
    """Save three passages of topic 1.3.1.2, each with one unused question.

    Args:
        conn: An empty in-memory database.

    Returns:
        The three passages' ids, in book order.
    """
    passages = [[para("First …", 49)], [para("Second …", 50)], [para("Third …", 51)]]
    save_chapter(conn, TOPICS, {"1.3.1.2": passages})
    ids = [row["id"] for row in list_passages(conn)]
    question = SimpleNamespace(question="Why?", key_points=["a point"], quote="…")
    for passage_id in ids:
        save_questions(conn, passage_id, [question], "q-version")
    return ids


def mark_asked(conn: sqlite3.Connection, passage_id: int) -> None:
    """Mark every question of a passage as used, as the study loop does after each grade.

    Args:
        conn: An open connection.
        passage_id: The passage whose questions were asked.
    """
    for row in list_questions(conn, passage_id):
        mark_question_used(conn, row["id"])


def test_the_next_new_passage_skips_passages_already_asked() -> None:
    """Passage 1 was asked, so passage 2 comes next; written-but-unasked questions don't count."""
    conn = connect(":memory:")
    first, second, _ = three_passages(conn)
    # Before anything is asked, the first passage in book order is new.
    assert next_new_passage(conn)["id"] == first
    mark_asked(conn, first)
    row = next_new_passage(conn)
    assert row["id"] == second
    # It carries its topic's number and title, like list_passages() rows.
    assert (row["number"], row["title"]) == ("1.3.1.2", "Classification")


def test_no_new_passage_left_fails_loudly() -> None:
    """When every passage has been asked about, it raises instead of returning None."""
    conn = connect(":memory:")
    for passage_id in three_passages(conn):
        mark_asked(conn, passage_id)
    with pytest.raises(ValueError, match="no new passage"):
        next_new_passage(conn)


def test_the_next_unused_question_is_the_oldest_one_not_yet_asked() -> None:
    """Questions come out in the order they were written, skipping used ones."""
    conn = connect(":memory:")
    save_chapter(conn, TOPICS, {"1.3.1.2": [[para("In classification, …", 49)]]})
    passage_id = list_passages(conn)[0]["id"]
    questions = [
        SimpleNamespace(question=text, key_points=["a point"], quote="…")
        for text in ("First?", "Second?", "Third?")
    ]
    save_questions(conn, passage_id, questions, "q-version")

    first = next_unused_question(conn, passage_id)
    assert first["text"] == "First?"
    mark_question_used(conn, first["id"])
    assert next_unused_question(conn, passage_id)["text"] == "Second?"


def test_an_empty_or_used_up_pool_gives_none() -> None:
    """None, not an error: it's the signal to write questions."""
    conn = connect(":memory:")
    save_chapter(conn, TOPICS, {"1.3.1.2": [[para("In classification, …", 49)]]})
    passage_id = list_passages(conn)[0]["id"]
    # No questions written yet.
    assert next_unused_question(conn, passage_id) is None
    # One written and asked.
    question = SimpleNamespace(question="Why?", key_points=["a point"], quote="…")
    save_questions(conn, passage_id, [question], "q-version")
    mark_asked(conn, passage_id)
    assert next_unused_question(conn, passage_id) is None


def test_a_question_is_read_back_by_its_id() -> None:
    """get_question gives the same row the pool has; a wrong id raises."""
    conn = connect(":memory:")
    question_id = a_question_id(conn)
    assert get_question(conn, question_id)["text"] == "Why?"
    with pytest.raises(ValueError, match="No question has the id 99"):
        get_question(conn, 99)


def test_marking_a_question_that_doesnt_exist_fails_loudly() -> None:
    """A wrong id raises instead of changing nothing in silence."""
    conn = connect(":memory:")
    with pytest.raises(ValueError, match="No question has the id 99"):
        mark_question_used(conn, 99)


def test_llm_calls_are_logged() -> None:
    """Each logged call is one row with its model, purpose and tokens."""
    conn = connect(":memory:")
    log_llm_call(conn, "openai/gpt-oss-20b", "questions", 900, 600, 1500, 2.5)
    row = conn.execute("SELECT model, purpose, total_tokens, seconds FROM llm_calls").fetchone()
    assert tuple(row) == ("openai/gpt-oss-20b", "questions", 1500, 2.5)


GRADE = Grade(quotes=["much easier"], missing=[], wrong=[], extras=[], verdict="correct")


def a_question_id(conn: sqlite3.Connection) -> int:
    """Save one passage with one question, and return the question's id.

    Args:
        conn: An empty in-memory database.

    Returns:
        The question's id, for attempts to point at.
    """
    save_chapter(conn, TOPICS, {"1.3.1.2": [[para("In classification, …", 49)]]})
    passage_id = list_passages(conn)[0]["id"]
    question = SimpleNamespace(question="Why?", key_points=["a point"], quote="much easier")
    save_questions(conn, passage_id, [question], "q-version")
    return list_questions(conn, passage_id)[0]["id"]


def test_a_saved_attempt_is_found_by_the_same_answer_check() -> None:
    """Same question, tidied answer, prompt version, model and settings: found, with its grade."""
    conn = connect(":memory:")
    question_id = a_question_id(conn)
    attempt_id = save_attempt(
        conn, question_id, None, "Much Easier", "much easier", GRADE, True, "v1", "m", "s"
    )

    row = find_attempt(conn, question_id, "much easier", "v1", "m", "s")

    assert row["id"] == attempt_id
    assert (row["answer"], row["verdict"], row["supported"]) == ("Much Easier", "correct", 1)
    assert Grade.model_validate_json(row["grade_details"]) == GRADE


@pytest.mark.parametrize(
    ("tidied", "version", "model", "settings"),
    [
        ("much harder", "v1", "m", "s"),
        ("much easier", "v2", "m", "s"),
        ("much easier", "v1", "other", "s"),
        ("much easier", "v1", "m", "other"),
    ],
    ids=["another answer", "another prompt version", "another model", "other settings"],
)
def test_any_difference_means_no_match(
    tidied: str, version: str, model: str, settings: str
) -> None:
    """The match is exact on all five parts of the key, never "close enough"."""
    conn = connect(":memory:")
    question_id = a_question_id(conn)
    save_attempt(conn, question_id, None, "x", "much easier", GRADE, True, "v1", "m", "s")

    assert find_attempt(conn, question_id, tidied, version, model, settings) is None


def test_an_unsupported_attempt_is_never_found() -> None:
    """An attempt whose quotes failed is saved, but the same-answer check skips it."""
    conn = connect(":memory:")
    question_id = a_question_id(conn)
    save_attempt(conn, question_id, None, "x", "much easier", GRADE, False, "v1", "m", "s")

    assert find_attempt(conn, question_id, "much easier", "v1", "m", "s") is None


def test_connect_needs_an_existing_folder(tmp_path: Path) -> None:
    """A database path in a missing folder fails with a clear error.

    ``tmp_path`` is a pytest fixture: a fresh temporary folder for this test.
    """
    with pytest.raises(FileNotFoundError, match="doesn't exist"):
        connect(tmp_path / "missing" / "tutor.db")
