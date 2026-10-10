"""Save each passage's vector in SQLite, and load them all back for search.

M1 concept "Embeddings and vector search". The vectors live in the
``embeddings`` table of ``data/tutor.db`` (see ``tutor.db``), one row per
passage, so SQLite stays the single source of truth (PLAN.md › Decisions ›
Vector store). A vector database such as Chroma would only be a copy that can
be rebuilt from this table.

How a vector fits in a table: a NumPy vector is 384 float32 numbers, and
``vector.tobytes()`` turns it into 1,536 raw bytes (4 per number), which SQLite
stores as a BLOB. ``np.frombuffer()`` reads those bytes back into the same 384
numbers. Nothing is rounded or lost on the way.

Each row also records the model that made it. Vectors from two different
models can't be compared, so loading refuses a mix instead of returning
nonsense scores (PLAN.md › Data).
"""

import sqlite3

import numpy as np


def save_embeddings(
    conn: sqlite3.Connection,
    passage_ids: list[int],
    vectors: list[np.ndarray],
    model: str,
) -> None:
    """Save one vector per passage, replacing any vector a passage already had.

    Replacing is what makes re-embedding safe: after a change of header or
    model, embedding every passage again simply overwrites the old vectors.
    The passages' text is never touched.

    Args:
        conn: An open connection.
        passage_ids: The passages' ids, in the same order as ``vectors``.
        vectors: The passages' vectors, as ``embed_passages()`` returns them.
        model: The name of the model that made them.

    Raises:
        ValueError: If there aren't exactly as many vectors as passage ids.
    """
    if len(passage_ids) != len(vectors):
        raise ValueError(f"Got {len(passage_ids)} passage ids but {len(vectors)} vectors.")
    # float32 fixes each number at 4 bytes, so loading knows where each one starts.
    rows = [
        (passage_id, model, vector.astype(np.float32).tobytes())
        for passage_id, vector in zip(passage_ids, vectors, strict=True)
    ]
    # One transaction: all the vectors are saved, or none are.
    with conn:
        # "INSERT OR REPLACE" overwrites the row when this passage_id already has one.
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (passage_id, model, vector) VALUES (?, ?, ?)",
            rows,
        )


def embedded_ids(conn: sqlite3.Connection, model: str) -> set[int]:
    """Return the ids of the passages that already have a vector from this model.

    The ingest command uses it to embed only what's new, so adding a chapter
    doesn't embed every earlier chapter again.

    Args:
        conn: An open connection.
        model: The model's name.

    Returns:
        The passage ids, as a set (fast to check "is this id in it?").
    """
    rows = conn.execute("SELECT passage_id FROM embeddings WHERE model = ?", (model,))
    return {row["passage_id"] for row in rows}


def load_embeddings(conn: sqlite3.Connection, model: str) -> tuple[list[int], np.ndarray]:
    """Load every passage's vector, ready to compare with a query.

    Args:
        conn: An open connection.
        model: The model the query will be embedded with. Every saved vector
            must come from it.

    Returns:
        Two things, in book order:

        - the passages' ids;
        - their vectors, stacked into one 2-D array with one row per passage
          (shape: number of passages × 384). Row 0 belongs to the first id,
          and so on. Looping over the array gives one vector at a time.

    Raises:
        ValueError: If any passage has no vector yet, or a vector was made by
            a different model.
    """
    # Passages that were saved but never embedded would be invisible to search.
    (missing,) = conn.execute(
        "SELECT COUNT(*) FROM passages WHERE id NOT IN (SELECT passage_id FROM embeddings)"
    ).fetchone()
    if missing:
        raise ValueError(
            f"{missing} passages have no vector yet. Run the ingest command again to embed them."
        )
    rows = conn.execute(
        "SELECT passage_id, model, vector FROM embeddings ORDER BY passage_id"
    ).fetchall()
    if not rows:
        raise ValueError("No passages are saved yet. Ingest a chapter first.")
    other_models = {row["model"] for row in rows} - {model}
    if other_models:
        raise ValueError(
            f"Some vectors were made by {sorted(other_models)}, not {model}. "
            "Vectors from different models can't be compared: embed every passage again."
        )
    # np.frombuffer reads raw bytes back as float32 numbers; np.stack piles the
    # 1-D vectors into one 2-D array, one row each.
    vectors = np.stack([np.frombuffer(row["vector"], dtype=np.float32) for row in rows])
    return [row["passage_id"] for row in rows], vectors
