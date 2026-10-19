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
- **attempts** (M1 concept "Grading grounded in the passage"): one row per
  graded answer, with the grade and what it was based on. The same-answer
  check looks here before calling the LLM.
- **skipped_passages** (M1 concept "Conditional edges"): passages whose
  questions failed twice, so the tutor never picks them again.
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
    from tutor.grading import Grade
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

CREATE TABLE IF NOT EXISTS attempts (
    id             INTEGER PRIMARY KEY,
    question_id    INTEGER NOT NULL REFERENCES questions(id),
    -- The study session it belongs to (LangGraph's thread id, from the M1
    -- concept "Pausing for your answer"). Empty (NULL) until sessions exist.
    -- It's here from the start because "CREATE TABLE IF NOT EXISTS" never
    -- adds a column to a table that already exists.
    session_id     TEXT,
    answer         TEXT    NOT NULL,  -- your answer, exactly as typed
    tidied_answer  TEXT    NOT NULL,  -- lowercase, spaces tidied: what the same-answer check uses
    verdict        TEXT    NOT NULL,  -- 'correct', 'partly correct' or 'incorrect'
    -- The LLM's whole Grade as JSON: quotes, missing, wrong, extras and grade.
    grade_details  TEXT    NOT NULL,
    -- 1 = every quote is really in the passage; 0 = the tutor showed
    -- "I couldn't find support for this grade", and the attempt is never reused.
    supported      INTEGER NOT NULL,
    prompt_version TEXT    NOT NULL,  -- the grader prompt's version (a hash of prompts/grader.md)
    model          TEXT    NOT NULL,  -- the grading model, e.g. 'openai/gpt-oss-120b'
    settings       TEXT    NOT NULL,  -- the call's settings, e.g. 'reasoning_effort=medium'
    created_at     TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS skipped_passages (
    -- A passage whose questions failed twice in a row, so it's never picked
    -- again. A table of its own, not a column on passages, because "CREATE
    -- TABLE IF NOT EXISTS" never adds a column to a table that already exists.
    passage_id INTEGER PRIMARY KEY REFERENCES passages(id),
    reason     TEXT    NOT NULL,  -- why, e.g. 'no question passed its quote check in 2 tries'
    skipped_at TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
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


def next_new_passage(conn: sqlite3.Connection) -> sqlite3.Row:
    """Return the first passage in book order that you haven't been asked about or skipped.

    "You choose" uses this when no topic was typed (PLAN.md › What "You
    choose" picks: the next new passage, in book order). A passage is new
    while none of its questions has ``used = 1``. A question is marked used
    only after its answer is graded, so a question you stopped before
    answering leaves its passage new, and it comes back next session.
    A passage whose questions were written but never asked is still new.
    A skipped passage (``save_skip()``) is never new again.

    Args:
        conn: An open connection.

    Returns:
        The passage's row: the same columns as ``list_passages()`` gives.

    Raises:
        ValueError: If every passage has been asked about or skipped.
    """
    # The same SELECT, JOIN and ORDER BY as list_passages(), plus two filters.
    # "NOT EXISTS (subquery)" keeps a passage only if the subquery finds no
    # row. The first: no question of this passage with used = 1. The second:
    # the passage isn't in skipped_passages. Each subquery runs once per
    # passage, reading that passage's id as passages.id.
    # LIMIT 1 keeps only the first passage left, so SQLite can stop there.
    row = conn.execute(
        "SELECT passages.*, topics.number, topics.title "
        "FROM passages JOIN topics ON topics.id = passages.topic_id "
        "WHERE NOT EXISTS ("
        "    SELECT 1 FROM questions "
        "    WHERE questions.passage_id = passages.id AND questions.used = 1"
        ") "
        "AND NOT EXISTS ("
        "    SELECT 1 FROM skipped_passages WHERE skipped_passages.passage_id = passages.id"
        ") "
        "ORDER BY topics.id, passages.position "
        "LIMIT 1"
    ).fetchone()
    # fetchone() gives None when no passage is left.
    if row is None:
        raise ValueError(
            "Every passage has been asked about or skipped: there's no new passage left."
        )
    return row


def save_skip(conn: sqlite3.Connection, passage_id: int, reason: str) -> None:
    """Skip a passage for good: neither "You choose" nor a typed topic picks it again.

    The study loop's ``skip_passage`` step calls this when no question about
    the passage passed its quote check in 2 tries (PLAN.md › Quotes checked in
    code). The passages stay listed, so a later check can show what went wrong
    with them (often code or math the PDF step turned into messy text).

    Args:
        conn: An open connection.
        passage_id: The passage to skip.
        reason: Why, e.g. "no question passed its quote check in 2 tries".

    Raises:
        sqlite3.IntegrityError: If the passage is already skipped (a skipped
            passage can't be picked, so skipping it twice is a bug), or no
            passage has that id.
    """
    with conn:
        conn.execute(
            "INSERT INTO skipped_passages (passage_id, reason) VALUES (?, ?)", (passage_id, reason)
        )


def skipped_passage_ids(conn: sqlite3.Connection) -> set[int]:
    """Return the ids of every skipped passage.

    ``pick_passage`` uses this to pass over skipped passages among a typed
    topic's search results.

    Args:
        conn: An open connection.

    Returns:
        The skipped passages' ids; an empty set if none is skipped.
    """
    return {row["passage_id"] for row in conn.execute("SELECT passage_id FROM skipped_passages")}


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


def get_question(conn: sqlite3.Connection, question_id: int) -> sqlite3.Row:
    """Return one question by its id.

    The study loop's state keeps only the question's id (a row can't be
    saved when the graph pauses, from the M1 concept "Pausing for your
    answer"), so each step that needs the question reads it here.

    Args:
        conn: An open connection.
        question_id: The question's id.

    Returns:
        The question's row, with ``id``, ``passage_id``, ``text``,
        ``key_points`` (a JSON string) and ``quote``.

    Raises:
        ValueError: If no question has that id.
    """
    row = conn.execute("SELECT * FROM questions WHERE id = ?", (question_id,)).fetchone()
    # fetchone() gives None when no row matched.
    if row is None:
        raise ValueError(f"No question has the id {question_id}.")
    return row


def next_unused_question(conn: sqlite3.Connection, passage_id: int) -> sqlite3.Row | None:
    """Return the passage's oldest question that hasn't been asked yet, or None if there's none.

    The study loop's ``get_question`` step calls this on every turn, once
    the passage is picked. ``None`` isn't an error: an empty pool is the
    normal signal to write questions (PLAN.md › Decisions › Reviews), the
    same way ``find_attempt()`` gives ``None`` for "no earlier answer".

    Args:
        conn: An open connection.
        passage_id: The passage picked for this turn.

    Returns:
        The question's row, with ``id``, ``passage_id``, ``text``,
        ``key_points`` (a JSON string) and ``quote``: what
        ``grade_answer()`` needs. ``None`` if every question of the passage
        is used, or it has none yet.
    """
    # Lowest id first: the order the questions were written in. LIMIT 1
    # keeps only that one.
    return conn.execute(
        "SELECT * FROM questions WHERE passage_id = ? AND used = 0 ORDER BY id LIMIT 1",
        (passage_id,),
    ).fetchone()


def mark_question_used(conn: sqlite3.Connection, question_id: int) -> None:
    """Mark a question as asked, so it never comes back and its passage is no longer new.

    The study loop's ``grade`` step calls this right after the answer is
    graded, never earlier: a question you stopped before answering stays
    unused, so it comes back next session.

    Args:
        conn: An open connection.
        question_id: The question that was just graded.

    Raises:
        ValueError: If no question has that id.
    """
    # "with conn" commits the change, or rolls it back if anything fails.
    with conn:
        cursor = conn.execute("UPDATE questions SET used = 1 WHERE id = ?", (question_id,))
    # rowcount is how many rows the UPDATE changed: 0 means the id matched
    # nothing, which would otherwise pass silently.
    if cursor.rowcount == 0:
        raise ValueError(f"No question has the id {question_id}.")


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


def save_attempt(
    conn: sqlite3.Connection,
    question_id: int,
    session_id: str | None,
    answer: str,
    tidied_answer: str,
    grade: "Grade",
    supported: bool,
    prompt_version: str,
    model: str,
    settings: str,
) -> int:
    """Save one graded answer.

    Args:
        conn: An open connection.
        question_id: The question that was answered.
        session_id: The study session, or None before sessions exist.
        answer: The answer exactly as typed.
        tidied_answer: The answer lowercased with its spaces tidied.
        grade: The LLM's Grade, saved whole as JSON.
        supported: Whether every quote in the grade is really in the passage.
        prompt_version: The grader prompt's version (from ``load_prompt()``).
        model: The grading model.
        settings: The call's settings, e.g. "reasoning_effort=medium".

    Returns:
        The new attempt's id.
    """
    with conn:
        cursor = conn.execute(
            "INSERT INTO attempts (question_id, session_id, answer, tidied_answer, verdict, "
            "grade_details, supported, prompt_version, model, settings) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                question_id,
                session_id,
                answer,
                tidied_answer,
                grade.verdict,
                # model_dump_json() turns the whole Grade into one JSON string;
                # Grade.model_validate_json() turns it back into a Grade.
                grade.model_dump_json(),
                # SQLite has no true/false type: int(True) is 1, int(False) is 0.
                int(supported),
                prompt_version,
                model,
                settings,
            ),
        )
    return cursor.lastrowid


def find_attempt(
    conn: sqlite3.Connection,
    question_id: int,
    tidied_answer: str,
    prompt_version: str,
    model: str,
    settings: str,
) -> sqlite3.Row | None:
    """Find an earlier supported attempt with exactly this answer to this question.

    This is the same-answer check (PLAN.md › How the tutor behaves › Same
    answer, same grade). The SQL "=" compares the tidied answers character
    by character, so one changed word means no match. The prompt version,
    model and settings must match too, so changing any of them grades
    answers afresh. Attempts whose quotes failed are skipped, so the same
    answer sent again gets a fresh try instead of "no support" forever.

    Args:
        conn: An open connection.
        question_id: The question being answered.
        tidied_answer: The answer, tidied by ``tidy_answer()``.
        prompt_version: The grader prompt's current version.
        model: The grading model about to be used.
        settings: The call's settings about to be used.

    Returns:
        The most recent matching attempt, or None if there is none. None
        isn't an error here: it just means the answer has to be graded.
    """
    return conn.execute(
        "SELECT * FROM attempts WHERE question_id = ? AND tidied_answer = ? "
        "AND prompt_version = ? AND model = ? AND settings = ? AND supported = 1 "
        "ORDER BY id DESC LIMIT 1",
        (question_id, tidied_answer, prompt_version, model, settings),
    ).fetchone()
