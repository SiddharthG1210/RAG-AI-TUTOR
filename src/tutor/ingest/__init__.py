"""Ingestion: turning study material into topics and prose the tutor can quiz on.

Adding material happens once per chapter, before any studying, and uses no
LLM (PLAN.md › Diagram 2). The modules arrive concept by concept:

- ``pdf.py`` (M1 concept "Turning the PDF into prose with page numbers"):
  reads the PDF, keeps only prose, and groups it into topics from the
  PDF's bookmarks.
- ``chunk.py`` (M1 concept "Chunking"): cuts each topic's prose into passages.
"""
