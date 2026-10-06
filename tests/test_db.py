"""Tests for ``tutor.db``: the topics table.

Every test uses an in-memory database (``":memory:"``): SQLite builds it in
RAM, it starts empty, and it disappears when the connection closes, so tests
never touch ``data/tutor.db`` and never affect each other.
"""

from pathlib import Path

import pytest

from tutor.db import connect, insert_topics, list_topics
from tutor.ingest.pdf import Topic

TOPICS = [
    Topic("1", "Introduction", 1, 41, "1"),
    Topic("1.3.1.2", "Classification", 4, 49, "9"),
]


def test_topics_come_back_in_book_order() -> None:
    """Saved topics are read back with every column, in the order they were saved."""
    conn = connect(":memory:")
    insert_topics(conn, TOPICS)
    rows = list_topics(conn)
    assert [(r["number"], r["title"], r["level"], r["page"], r["page_label"]) for r in rows] == [
        ("1", "Introduction", 1, 41, "1"),
        ("1.3.1.2", "Classification", 4, 49, "9"),
    ]


def test_ingesting_a_chapter_twice_fails_and_saves_nothing() -> None:
    """A repeated topic number raises, and the whole second insert is rolled back."""
    conn = connect(":memory:")
    insert_topics(conn, TOPICS[:1])
    with pytest.raises(ValueError, match="already in the database"):
        # The new topic comes first, so it would be saved if there were no transaction.
        insert_topics(conn, [Topic("2", "Preliminaries", 1, 70, "30"), TOPICS[0]])
    assert [r["number"] for r in list_topics(conn)] == ["1"]


def test_connect_needs_an_existing_folder(tmp_path: Path) -> None:
    """A database path in a missing folder fails with a clear error.

    ``tmp_path`` is a pytest fixture: a fresh temporary folder for this test.
    """
    with pytest.raises(FileNotFoundError, match="doesn't exist"):
        connect(tmp_path / "missing" / "tutor.db")
