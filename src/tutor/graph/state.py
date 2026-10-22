"""What the study loop carries from step to step (``State``), and what every step can reach.

``Context`` is what every step can reach: the database connection and the
study session's id.

A value goes in the state when one node makes it and a later node needs it.
LangGraph starts the state from the dict given to ``graph.invoke()``, then
merges each node's returned dict into it, key by key: a new key is added, an
existing key is replaced. Each key here is written by exactly one node (only
``topic`` has two writers: you at the start, then ``wait_for_next`` after each
turn), so no node can overwrite another's work by accident.

The state keeps ids, not database rows. From the M1 concept "Pausing for your
answer", LangGraph saves the state whenever the graph pauses, and its saver
can store numbers, text, lists, dicts and dataclasses, but not a
``sqlite3.Row``. The database stays the one source of truth: a node that
needs a row reads it by its id.
"""

import sqlite3
from dataclasses import dataclass
from typing import TypedDict

from tutor.grading import GradedAnswer


class State(TypedDict):
    """The keys one study turn carries from node to node.

    A ``TypedDict`` is a plain ``dict`` at run time; the class only lists
    which keys it has and the type of each, so LangGraph (and the editor)
    know them.

    The turn: START → pick_passage → pick_question → ask → grade → show → END,
    with the forks from the M1 concept "Conditional edges" (see build.py).
    """

    # Each key: who writes it → who reads it later.
    # The topic you typed, or None for "You choose" (you, at START → pick_passage).
    topic: str | None
    # The picked passage's id, or None when nothing is close enough to the
    # typed topic (pick_passage → its routing function, pick_question, show).
    passage_id: int | None
    # The picked question's id, or None when no written question passed its
    # quote check (pick_question → its routing function, ask, grade).
    question_id: int | None
    # How many times this passage's questions were written this turn: reset
    # to 0 by pick_passage, plus 1 each time pick_question writes
    # (pick_question → its routing function, which allows 2 tries).
    write_tries: int
    # Your answer, as typed (ask → grade).
    answer: str
    # What grade_answer() handed back (grade → show).
    result: GradedAnswer
    # True when passages close to the typed topic exist but you've already
    # been asked about each one this session (or it's skipped), as opposed to
    # nothing close at all (pick_passage → not_covered, which says which).
    nothing_new: bool
    # Your reply at the pause after a turn: True to stop the session
    # (wait_for_next → its routing function). The same reply sets the next
    # turn's topic.
    stop: bool


@dataclass(frozen=True)
class Context:
    """What every node needs that isn't part of the state: the connection and the session id.

    No node makes either, and both stay the same for the whole session, so
    they fail the state's test. The connection also can't be saved: it's a
    live link to ``data/tutor.db``, not a value. LangGraph's "runtime
    context" exists for exactly this. The starter passes it with every call,
    as ``graph.invoke(..., context=Context(conn=conn, session_id=...))``:
    LangGraph saves the state when the graph pauses, but never the context,
    so a resumed call must hand it over again. Every node reads it as
    ``runtime.context``.

    Attributes:
        conn: The open connection to the tutor's database.
        session_id: The study session's id. It's also LangGraph's thread id,
            which names the paused turn to resume, and it's saved with each
            attempt, so a session's answers can be found again.
    """

    conn: sqlite3.Connection
    session_id: str


# graph.invoke({...}, context=Context(conn=..., session_id=...))  ← you hand LangGraph the Context
#                                                       with EVERY call (a resume too)
#         │
#         ▼
# LangGraph calls each node as  node(state, runtime)    ← it puts the Context inside "runtime"
#         │
#         ▼
# inside the node:
#     conn = runtime.context.conn                       ← the node takes conn out
#     next_new_passage(conn)                            ← and passes it to the tool's conn parameter
