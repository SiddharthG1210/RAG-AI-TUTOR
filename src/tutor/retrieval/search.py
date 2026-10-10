"""Find the passages closest in meaning to a typed topic or question.

M1 concept "Embeddings and vector search". The whole vector search, in plain
NumPy: embed the query, score it against every saved passage vector with
cosine similarity, and keep the best few (PLAN.md › Decisions › Vector store).
Checking every vector this way takes about a millisecond even for the whole
book, so no vector database is needed.

How the pieces meet:

    "gradient descent" ── embed_query() ──► query vector ─┐
                                                           ├─ cosine, per passage ──► best k
    database ── load_embeddings() ──► ids + vectors ──────┘
"""

import sqlite3

import numpy as np

from tutor.db import get_passage
from tutor.retrieval.embed import MODEL_NAME, embed_query
from tutor.retrieval.store import load_embeddings

CUTOFF = 0.56
"""The lowest score a passage needs to be returned. Below it for every passage,
the tutor says "the material doesn't cover this" (PLAN.md › How the tutor
behaves in specific cases › Topic search).

Chosen from chapter 1's scores (``scripts/check_retrieval.py`` plus a wider
probe), which overlap, so no number separates everything:

- Real topics, typed vaguely, scored from 0.591 ("cats and dogs") and 0.619
  ("overfitting") upwards. The 10 test queries scored 0.644 or more.
- Queries with nothing to do with the book (pizza, pasta, guitar, football,
  the weather, the Roman empire, "asdfgh") scored 0.487 to 0.536.
- Computer topics the book doesn't cover ("learning javascript" 0.655,
  "python list comprehension" 0.637, "computer viruses" 0.609) score like
  real topics, because they share the vocabulary.

0.56 sits halfway between the clearly unrelated queries (up to 0.536) and the
lowest real topic (0.591). It leans low on purpose: turning away a topic the
book covers hides material you want to study, while letting a computer topic
through only shows a passage you can see is unrelated. Ask the book (M2)
checks every citation, so it doesn't rely on this number alone.
"""


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine of the angle between vectors ``a`` and ``b``.

    Your function from the tiny example (scratch/cosine.py), moved here
    because ``scratch/`` isn't part of the package. 1 means the same direction,
    0 unrelated, -1 opposite.

    Args:
        a: The first vector.
        b: The second vector, with the same number of dimensions.

    Returns:
        A number from -1 through 0 to 1.
    """
    # The dot product: multiply the vectors number by number and add up.
    dot_product = np.dot(a, b)
    # Each vector's length (its "norm").
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return float(dot_product / (norm_a * norm_b))


def search(
    conn: sqlite3.Connection, query: str, k: int = 3, cutoff: float = CUTOFF
) -> list[tuple[float, sqlite3.Row]]:
    """Return the k passages closest in meaning to the query, best first.

    Args:
        conn: An open connection to the database with the embedded passages.
        query: What the user typed, such as "gradient descent".
        k: How many passages to return.
        cutoff: The lowest score a passage needs to be returned. The retrieval
            check passes -1 (the lowest cosine there is) to see every score.

    Returns:
        Up to k pairs of (score, passage), highest score first, only those
        scoring at least ``cutoff``. Each passage is a row like
        ``list_passages()`` gives: its text, pages, and its topic's number and
        title. An empty list means the material doesn't cover the query.
    """
    # 1. Load every passage's id and vector from the database.
    #    ids[i] and vectors[i] belong to the same passage.
    ids, vectors = load_embeddings(conn, MODEL_NAME)

    # 2. Turn the query into a vector (with bge's query prefix).
    embedded_query = embed_query(query)

    # 3. Score every passage, keeping each score next to its passage's id.
    scored_ids = []
    for i in range(len(ids)):
        score = cosine_similarity(embedded_query, vectors[i])
        scored_ids.append((score, ids[i]))

    # 4. Sort the pairs, highest score first, and keep the first k.
    #    Python sorts pairs by their first item, the score, so no key is needed.
    #    (A tie would be settled by the id, a number, which Python can compare.)
    scored_ids.sort(reverse=True)
    best = scored_ids[:k]

    # 5. Turn each kept id into its passage, keeping the score next to it.
    #    Every passage that's returned may reach the LLM (a question each, or
    #    Ask the book's sources), so each must reach the cut-off on its own.
    #    The pairs are sorted, so the first one below it means all the rest
    #    are too. If even the best one is, the empty list means "the material
    #    doesn't cover this".
    results = []
    for score, passage_id in best:
        if score < cutoff:
            break  # this one is too far from the query, so every later one is too
        results.append((score, get_passage(conn, passage_id)))
    return results
