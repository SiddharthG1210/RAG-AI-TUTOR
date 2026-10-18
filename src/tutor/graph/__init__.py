"""The study loop: a LangGraph graph that runs one study turn, step by step.

M1 concept "The study loop: state, nodes, edges". LangGraph runs a list of
steps (nodes) in the order the edges give, carrying one dict (the state)
from step to step. Each node calls a tool built in an earlier concept.

- ``state.py``: ``State``, the keys carried from node to node, and
  ``Context``, what every node needs that isn't part of the state.
- ``nodes.py``: the five steps of a turn, each a thin wrapper around a tool.
- ``build.py``: ``build_graph()``, which adds the nodes, draws the edges and
  compiles the graph.
"""
