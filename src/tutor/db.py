"""The tutor's SQLite database: the connection, the tables, and plain-SQL reads and writes.

SQLite keeps the whole database in one file (``data/tutor.db``), and Python's
built-in ``sqlite3`` module talks to it, so there's no server and nothing to
install. The SQL is written by hand, with no ORM, so every query is visible
(PLAN.md › Tech). SQLite is the tutor's source of truth, embeddings included.

Tables arrive milestone by milestone (PLAN.md › Data). This module has:

- **topics** (M1 concept "Turning the PDF into prose with page numbers"):
  one row per section the tutor quizzes on.
- **passages** (M1 concept "Chunking"): the 120–350-word pieces of each
  topic that questions are written from, with their page ranges.
- **embeddings** (M1 concept "Embeddings and vector search"): each
  passage's vector, written and read by ``tutor.retrieval.store``.
- **questions** (M1 concept "Structured output (question writing)"): each
  passage's pool of checked questions, asked one per review.
- **llm_calls** (same concept): one row per LLM call, with its tokens, to
  keep an eye on the free tier.
"""

import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from tutor.ingest.pdf import Paragraph, Topic

# TYPE_CHECKING is False when the program runs, so this import only happens
# for type checkers and editors. tutor.questions imports this module, so
# importing it back at run time would be a circular import.
if TYPE_CHECKING:
    from tutor.questions import Question

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

CREATE TABLE IF NOT EXISTS passages (
    id             INTEGER PRIMARY KEY,
    -- Which topic it belongs to. REFERENCES makes it a foreign key: SQLite
    -- refuses a passage whose topic_id isn't a real row in topics.
    topic_id       INTEGER NOT NULL REFERENCES topics(id),
    position       INTEGER NOT NULL,  -- 1, 2, 3 … in book order within its topic
    text           TEXT    NOT NULL,  -- its paragraphs, separated by blank lines
    page           INTEGER NOT NULL,  -- PDF page it starts on
    page_label     TEXT    NOT NULL,  -- printed page it starts on
    end_page       INTEGER NOT NULL,  -- PDF page it ends on
    end_page_label TEXT    NOT NULL,  -- printed page it ends on
    UNIQUE (topic_id, position)
);

CREATE TABLE IF NOT EXISTS embeddings (
    -- One vector per passage: the passage's id is also this table's key.
    passage_id INTEGER PRIMARY KEY REFERENCES passages(id),
    model      TEXT    NOT NULL,  -- the model that made it, e.g. 'BAAI/bge-small-en-v1.5'
    -- The vector's 384 numbers as raw bytes (4 bytes each, so 1,536 bytes).
    -- A BLOB ("binary large object") stores bytes exactly as given.
    vector     BLOB    NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
    id             INTEGER PRIMARY KEY,
    passage_id     INTEGER NOT NULL REFERENCES passages(id),
    text           TEXT    NOT NULL,  -- the question itself
    key_points     TEXT    NOT NULL,  -- what a good answer contains: a JSON list of strings
    quote          TEXT    NOT NULL,  -- the passage's words it's based on, as the LLM copied them
    -- The first 12 characters of the prompt file's SHA-256 hash, so a change
    -- in the questions can be traced to the prompt change behind it.
    prompt_version TEXT    NOT NULL,
    used           INTEGER NOT NULL DEFAULT 0,  -- 0 = still in the pool, 1 = already asked
    created_at     TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS llm_calls (
    id                INTEGER PRIMARY KEY,
    model             TEXT    NOT NULL,  -- e.g. 'openai/gpt-oss-20b'
    purpose           TEXT    NOT NULL,  -- what the call was for, e.g. 'questions'
    prompt_tokens     INTEGER NOT NULL,  -- tokens sent: instructions plus passage
    completion_tokens INTEGER NOT NULL,  -- tokens written back, reasoning included
    total_tokens      INTEGER NOT NULL,  -- what counts against the free-tier limits
    seconds           REAL    NOT NULL,  -- how long the call took
    called_at         TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
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
    # SQLite ignores foreign keys (the REFERENCES in passages) unless this is
    # switched on for each connection.
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def save_chapter(
    conn: sqlite3.Connection,
    topics: list[Topic],
    passages: dict[str, list[list[Paragraph]]],
) -> None:
    """Save a chapter's topics and passages, all or nothing.

    ``with conn:`` wraps every insert in one transaction: it commits if they
    all succeed and rolls back if any fails, so a failed ingest never leaves
    half a chapter behind.

    Args:
        conn: An open connection.
        topics: The chapter's topics, in book order.
        passages: Each topic's passages, keyed by topic number, as
            ``chunk_chapter()`` returns them. Each passage is a list of
            paragraphs.

    Raises:
        ValueError: If a topic with the same number is already saved, which
            means the chapter was ingested before.
    """
    try:
        with conn:
            for topic in topics:
                # The "?" placeholders let sqlite3 insert the values safely,
                # instead of pasting them into the SQL text.
                cursor = conn.execute(
                    "INSERT INTO topics (number, title, level, page, page_label) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (topic.number, topic.title, topic.level, topic.page, topic.page_label),
                )
                # lastrowid is the id SQLite just gave this topic's row; the
                # passages point at it.
                topic_id = cursor.lastrowid
                rows = [
                    (
                        topic_id,
                        position,
                        "\n\n".join(paragraph.text for paragraph in passage),
                        # A passage runs from its first paragraph's start page
                        # to its last paragraph's end page.
                        passage[0].page,
                        passage[0].page_label,
                        passage[-1].end_page,
                        passage[-1].end_page_label,
                    )
                    for position, passage in enumerate(passages.get(topic.number, []), start=1)
                ]
                conn.executemany(
                    "INSERT INTO passages (topic_id, position, text, page, page_label, "
                    "end_page, end_page_label) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
    except sqlite3.IntegrityError as error:
        # UNIQUE on topics.number: the same section can't be saved twice.
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


def get_passage(conn: sqlite3.Connection, passage_id: int) -> sqlite3.Row:
    """Return one passage with its topic's number and title.

    Args:
        conn: An open connection.
        passage_id: The passage's id, e.g. one that ``search()`` picked.

    Returns:
        The passage's row: the same columns as ``list_passages()`` gives.

    Raises:
        ValueError: If no passage has that id.
    """
    row = conn.execute(
        "SELECT passages.*, topics.number, topics.title "
        "FROM passages JOIN topics ON topics.id = passages.topic_id "
        "WHERE passages.id = ?",
        (passage_id,),
    ).fetchone()
    # fetchone() gives None when no row matched.
    if row is None:
        raise ValueError(f"No passage has the id {passage_id}.")
    return row


def list_passages(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return every saved passage with its topic's number and title, in book order.

    Args:
        conn: An open connection.

    Returns:
        One row per passage: its own columns plus ``number`` and ``title``
        from its topic. The JOIN matches each passage to its topic row.
    """
    return conn.execute(
        "SELECT passages.*, topics.number, topics.title "
        "FROM passages JOIN topics ON topics.id = passages.topic_id "
        "ORDER BY topics.id, passages.position"
    ).fetchall()


def save_questions(
    conn: sqlite3.Connection, passage_id: int, questions: list["Question"], prompt_version: str
) -> None:
    """Add checked questions to a passage's pool, all or nothing.

    Args:
        conn: An open connection.
        passage_id: The passage they were written from.
        questions: The questions whose quotes passed the check.
        prompt_version: The question-writing prompt's version (from ``load_prompt()``).
    """
    rows = [
        # json.dumps turns the list of key points into one string, e.g.
        # '["firm categories are hard to optimize", "..."]', since a column
        # holds one value. json.loads turns it back into a list.
        (passage_id, q.question, json.dumps(q.key_points), q.quote, prompt_version)
        for q in questions
    ]
    with conn:
        conn.executemany(
            "INSERT INTO questions (passage_id, text, key_points, quote, prompt_version) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )


def list_questions(conn: sqlite3.Connection, passage_id: int) -> list[sqlite3.Row]:
    """Return every question ever written for a passage, oldest first, used or not.

    A refill sends these to the question writer, so it doesn't repeat them.

    Args:
        conn: An open connection.
        passage_id: The passage.

    Returns:
        One row per question. ``key_points`` is a JSON string; ``json.loads``
        turns it back into a list.
    """
    return conn.execute(
        "SELECT * FROM questions WHERE passage_id = ? ORDER BY id", (passage_id,)
    ).fetchall()


def log_llm_call(
    conn: sqlite3.Connection,
    model: str,
    purpose: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    seconds: float,
) -> None:
    """Record one LLM call and the tokens it used (PLAN.md › Free-tier guard).

    Args:
        conn: An open connection.
        model: The model that answered.
        purpose: What the call was for, e.g. "questions".
        prompt_tokens: Tokens sent.
        completion_tokens: Tokens written back, reasoning included.
        total_tokens: The two together: what counts against the free-tier limits.
        seconds: How long the call took.
    """
    with conn:
        conn.execute(
            "INSERT INTO llm_calls (model, purpose, prompt_tokens, completion_tokens, "
            "total_tokens, seconds) VALUES (?, ?, ?, ?, ?, ?)",
            (model, purpose, prompt_tokens, completion_tokens, total_tokens, seconds),
        )
