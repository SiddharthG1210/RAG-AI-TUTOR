"""Tests for ``tutor.retrieval.search``: cosine similarity and picking the best passages.

No model is loaded. The passages get made-up 2-number vectors, and
``monkeypatch`` (a pytest fixture) swaps ``embed_query`` for a function that
returns a chosen vector, so every score is known in advance.
"""

import sqlite3

import numpy as np
import pytest

from tutor.db import connect, list_passages, save_chapter
from tutor.ingest.pdf import Paragraph, Topic
from tutor.retrieval import search as search_module
from tutor.retrieval.embed import MODEL_NAME
from tutor.retrieval.search import cosine_similarity, search
from tutor.retrieval.store import save_embeddings


def test_cosine_similarity_measures_direction() -> None:
    """Same direction gives 1, a right angle 0, opposite -1, whatever the lengths."""
    assert cosine_similarity(np.array([3, 4]), np.array([6, 8])) == pytest.approx(1.0)
    assert cosine_similarity(np.array([1, 0]), np.array([0, 1])) == pytest.approx(0.0)
    assert cosine_similarity(np.array([1, 0]), np.array([-1, 0])) == pytest.approx(-1.0)


@pytest.fixture
def three_passages(monkeypatch: pytest.MonkeyPatch) -> sqlite3.Connection:
    """Save three passages whose scores against any query are 1.0, 0.0 and 0.6.

    The passages' vectors are [1, 0], [0, 1] and [0.6, 0.8], and the fake
    ``embed_query`` always returns [1, 0].

    Returns:
        The open in-memory connection.
    """
    conn = connect(":memory:")
    topic = Topic("1.2.4", "Optimization Algorithms", 3, 46, "6")
    texts = ["first", "second", "third"]
    passages = {"1.2.4": [[Paragraph("1.2.4", 46, "6", text, 46, "6")] for text in texts]}
    save_chapter(conn, [topic], passages)
    ids = [row["id"] for row in list_passages(conn)]
    vectors = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.6, 0.8])]
    save_embeddings(conn, ids, vectors, MODEL_NAME)
    monkeypatch.setattr(search_module, "embed_query", lambda query: np.array([1.0, 0.0]))
    return conn


def scores_and_texts(results: list[tuple[float, sqlite3.Row]]) -> list[tuple[float, str]]:
    """Reduce search results to (rounded score, text) pairs, easy to compare.

    Args:
        results: ``search()``'s (score, passage) pairs.

    Returns:
        The pairs, with scores rounded to 3 decimals.
    """
    return [(round(score, 3), passage["text"]) for score, passage in results]


def test_search_returns_the_closest_passages_best_first(
    three_passages: sqlite3.Connection,
) -> None:
    """The k passages pointing most like the query come back, highest score first."""
    results = search(three_passages, "anything", k=2, cutoff=-1.0)
    assert scores_and_texts(results) == [(1.0, "first"), (0.6, "third")]


def test_passages_below_the_cutoff_are_left_out(three_passages: sqlite3.Connection) -> None:
    """With a cut-off of 0.5, the passage scoring 0.0 isn't returned even though k allows it."""
    results = search(three_passages, "anything", k=3, cutoff=0.5)
    assert scores_and_texts(results) == [(1.0, "first"), (0.6, "third")]


def test_nothing_close_enough_gives_an_empty_list(three_passages: sqlite3.Connection) -> None:
    """When no passage reaches the cut-off, the empty list means "doesn't cover this"."""
    assert search(three_passages, "anything", cutoff=1.5) == []
