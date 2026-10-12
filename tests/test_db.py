"""Tests for ``tutor.db``: the topics, passages, questions and llm_calls tables.

Every test uses an in-memory database (``":memory:"``): SQLite builds it in
RAM, it starts empty, and it disappears when the connection closes, so tests
never touch ``data/tutor.db`` and never affect each other.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tutor.db import (
    connect,
    list_passages,
    list_questions,
    list_topics,
    log_llm_call,
    save_chapter,
    save_questions,
)
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


def test_llm_calls_are_logged() -> None:
    """Each logged call is one row with its model, purpose and tokens."""
    conn = connect(":memory:")
    log_llm_call(conn, "openai/gpt-oss-20b", "questions", 900, 600, 1500, 2.5)
    row = conn.execute("SELECT model, purpose, total_tokens, seconds FROM llm_calls").fetchone()
    assert tuple(row) == ("openai/gpt-oss-20b", "questions", 1500, 2.5)


def test_connect_needs_an_existing_folder(tmp_path: Path) -> None:
    """A database path in a missing folder fails with a clear error.

    ``tmp_path`` is a pytest fixture: a fresh temporary folder for this test.
    """
    with pytest.raises(FileNotFoundError, match="doesn't exist"):
        connect(tmp_path / "missing" / "tutor.db")
