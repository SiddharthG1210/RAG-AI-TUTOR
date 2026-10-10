"""Tests for ``tutor.retrieval.store``: saving and loading passage vectors.

The vectors here are made up and only 3 numbers long, so no embedding model
is loaded: CI never downloads bge. Saving and loading don't care about the
length, only that the numbers come back exactly.
"""

import sqlite3

import numpy as np
import pytest

from tutor.db import connect, list_passages, save_chapter
from tutor.ingest.pdf import Paragraph, Topic
from tutor.retrieval.store import load_embeddings, save_embeddings

MODEL = "test-model"


def chapter_with_two_passages() -> tuple[sqlite3.Connection, list[int]]:
    """Save one topic with two passages in an in-memory database.

    Returns:
        The open connection, and the two passages' ids in book order.
    """
    conn = connect(":memory:")
    topic = Topic("1.2.3", "Objective Functions", 3, 46, "6")
    passages = {
        "1.2.3": [
            [Paragraph("1.2.3", 46, "6", "In machine learning, …", 46, "6")],
            [Paragraph("1.2.3", 47, "7", "During optimization, …", 47, "7")],
        ]
    }
    save_chapter(conn, [topic], passages)
    return conn, [row["id"] for row in list_passages(conn)]


def test_vectors_come_back_exactly_in_book_order() -> None:
    """Every number survives the trip through SQLite, and row i belongs to id i."""
    conn, ids = chapter_with_two_passages()
    vectors = [np.array([0.1, 0.2, 0.3], dtype=np.float32), np.array([-1.0, 0.0, 0.5])]
    save_embeddings(conn, ids, vectors, MODEL)

    loaded_ids, loaded = load_embeddings(conn, MODEL)
    assert loaded_ids == ids
    assert loaded.shape == (2, 3)
    assert np.array_equal(loaded, np.array(vectors, dtype=np.float32))


def test_saving_again_replaces_the_old_vectors() -> None:
    """Re-embedding overwrites each passage's vector instead of adding a second one."""
    conn, ids = chapter_with_two_passages()
    save_embeddings(conn, ids, [np.zeros(3), np.zeros(3)], MODEL)
    save_embeddings(conn, ids, [np.ones(3), np.ones(3)], MODEL)

    _, loaded = load_embeddings(conn, MODEL)
    assert np.array_equal(loaded, np.ones((2, 3)))


def test_a_passage_without_a_vector_fails() -> None:
    """Search would never find an unembedded passage, so loading refuses."""
    conn, ids = chapter_with_two_passages()
    save_embeddings(conn, ids[:1], [np.ones(3)], MODEL)
    with pytest.raises(ValueError, match="1 passages have no vector"):
        load_embeddings(conn, MODEL)


def test_vectors_from_another_model_fail() -> None:
    """Vectors from two models can't be compared, so a mix is refused."""
    conn, ids = chapter_with_two_passages()
    save_embeddings(conn, ids, [np.ones(3), np.ones(3)], "old-model")
    with pytest.raises(ValueError, match="old-model"):
        load_embeddings(conn, MODEL)


def test_an_empty_database_fails() -> None:
    """Loading before any chapter is ingested raises instead of returning nothing."""
    with pytest.raises(ValueError, match="Ingest a chapter first"):
        load_embeddings(connect(":memory:"), MODEL)


def test_ids_and_vectors_must_match_in_number() -> None:
    """A vector missing from the list would shift every later vector onto the wrong passage."""
    conn, ids = chapter_with_two_passages()
    with pytest.raises(ValueError, match="2 passage ids but 1 vectors"):
        save_embeddings(conn, ids, [np.ones(3)], MODEL)
