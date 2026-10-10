"""Retrieval: finding the passages that match a typed topic by meaning.

The modules (M1 concept "Embeddings and vector search"):

- ``embed.py``: the bge-small model that turns text into vectors.
- ``store.py``: the passages' vectors, saved in and loaded from SQLite.
- ``search.py``: ``search(query, k, topic)``, cosine similarity against every passage.
"""
