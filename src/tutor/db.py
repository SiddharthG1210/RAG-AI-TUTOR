"""The tutor's SQLite database: the connection, the tables, and plain-SQL reads and writes.

SQLite keeps the whole database in one file (``data/tutor.db``), and Python's
built-in ``sqlite3`` module talks to it, so there's no server and nothing to
install. The SQL is written by hand, with no ORM, so every query is visible
(PLAN.md › Tech). SQLite is the tutor's source of truth: the Chroma index
added in the M1 concept "Embeddings and vector search" is rebuilt from it.

Tables arrive milestone by milestone (PLAN.md › Data). This module has:

- **topics** (M1 concept "Turning the PDF into prose with page numbers"):
  one row per section the tutor quizzes on.
"""

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from tutor.ingest.pdf import Topic

# "CREATE TABLE IF NOT EXISTS" makes the schema safe to run on every
# connection: it creates missing tables and leaves existing ones alone.
SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id         INTEGER PRIMARY KEY,   -- SQLite numbers rows itself; book order
    number     TEXT    NOT NULL UNIQUE,  -- section number, e.g. '1.3.1.2'
    title      TEXT    NOT NULL,      -- e.g. 'Classification'
    level      INTEGER NOT NULL,      -- 1 = chapter, 2 = section, and so on
    page       INTEGER NOT NULL,      -- PDF page of the heading, counted from 1
    page_label TEXT    NOT NULL       -- the page number printed in the book
);
"""


def connect(path: Path | str) -> sqlite3.Connection:
    """Open the database (creating the file and tables if needed).

    Args:
        path: The database file, or ``":memory:"`` for a temporary in-memory
            database (what the tests use).

    Returns:
        An open connection. Rows come back as ``sqlite3.Row``, so columns can
        be read by name (``row["title"]``) as well as by position.

    Raises:
        FileNotFoundError: If the folder the file should go in doesn't exist.
    """
    if str(path) != ":memory:" and not Path(path).parent.is_dir():
        raise FileNotFoundError(f"The folder for the database doesn't exist: {Path(path).parent}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # SQLite ignores foreign keys unless this is switched on for each
    # connection. No table uses them yet; passages (the M1 concept
    # "Chunking") will point at topics.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def insert_topics(conn: sqlite3.Connection, topics: Iterable[Topic]) -> None:
    """Save a chapter's topics, all or nothing.

    ``with conn:`` wraps the inserts in one transaction: it commits if every
    insert succeeds and rolls back if any fails, so a failed ingest never
    leaves half a chapter behind.

    Args:
        conn: An open connection.
        topics: The topics to save, in book order.

    Raises:
        ValueError: If a topic with the same number is already saved, which
            means the chapter was ingested before.
    """
    rows = [(t.number, t.title, t.level, t.page, t.page_label) for t in topics]
    try:
        with conn:
            # The "?" placeholders let sqlite3 insert the values safely,
            # instead of pasting them into the SQL text.
            conn.executemany(
                "INSERT INTO topics (number, title, level, page, page_label) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
    except sqlite3.IntegrityError as error:
        # UNIQUE on number: the same section can't be saved twice.
        raise ValueError(
            "This chapter's topics are already in the database. Studied passages are "
            "never re-cut (PLAN.md › Data), so to ingest it again before studying it, "
            "delete data/tutor.db first."
        ) from error


def list_topics(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return every saved topic, in book order.

    Args:
        conn: An open connection.

    Returns:
        One row per topic, with the columns of the topics table.
    """
    return conn.execute(
        "SELECT id, number, title, level, page, page_label FROM topics ORDER BY id"
    ).fetchall()
