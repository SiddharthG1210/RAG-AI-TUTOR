"""The five steps of a study turn: each one reads the state, calls a tool, and returns its changes.

M1 concept "The study loop: state, nodes, edges". A LangGraph node is a plain
function. LangGraph calls it with the current state (and, because it has a
``runtime`` parameter, the run's ``Runtime``, which carries our ``Context``),
and merges the dict it returns into the state. A node never calls another
node: the edges in ``build.py`` decide what runs next.

The turn, with the key each node adds:

    {"topic": ...}
       │ pick_passage   search() for a typed topic, else next_new_passage()
    + passage_id
       │ pick_question  next_unused_question(), writing questions first if the pool is empty
    + question_id
       │ ask            prints the question, reads your answer with input()
    + answer
       │ grade          grade_answer(), then mark_question_used()
    + result
       │ show           prints the grade and the passage (adds nothing)

The graph is one straight line for now. Two cases have no route yet, so they
stop the turn with ``TurnStopped``: a typed topic the material doesn't cover,
and a passage where no written question passed its quote check. The M1
concept "Conditional edges" turns them into routes.
"""

from langgraph.runtime import Runtime

from tutor.db import (
    get_passage,
    get_question,
    mark_question_used,
    next_new_passage,
    next_unused_question,
)
from tutor.grading import grade_answer
from tutor.graph.state import Context, State
from tutor.questions import write_questions
from tutor.retrieval.search import search


class TurnStopped(Exception):
    """A turn that can't go on, with the message to show you instead.

    Raised for the cases that have no route yet (see the module docstring).
    The starter catches it and prints the message.
    """


def pick_passage(state: State, runtime: Runtime[Context]) -> dict:
    """Pick the turn's passage: the closest one to a typed topic, or the next new one.

    Args:
        state: Needs ``topic``.
        runtime: Gives the database connection, as ``runtime.context.conn``.

    Returns:
        ``{"passage_id": ...}``.

    Raises:
        TurnStopped: If no passage is close enough to the typed topic.
        ValueError: From ``next_new_passage()``, if every passage has been asked about.
    """
    conn = runtime.context.conn
    topic = state["topic"]
    if topic is None:
        # "You choose": the first passage in book order with no question asked yet.
        passage = next_new_passage(conn)
    else:
        # A typed topic: search() gives up to 3 (score, passage) pairs, best
        # first, only those above the cut-off. An empty list means nothing
        # is close enough (PLAN.md › Topic search).
        results = search(conn, topic)
        if not results:
            raise TurnStopped(f"The material doesn't cover this: {topic!r}.")
        _score, passage = results[0]  # the best match; its score isn't needed here
    return {"passage_id": passage["id"]}


def pick_question(state: State, runtime: Runtime[Context]) -> dict:
    """Pick the passage's next unused question, writing a pool of questions first if it's empty.

    The question is only picked here, not marked used: ``grade`` marks it
    after the answer is graded, so a question you stop before answering
    comes back next session.

    Args:
        state: Needs ``passage_id``.
        runtime: Gives the database connection.

    Returns:
        ``{"question_id": ...}``.

    Raises:
        TurnStopped: If the pool was empty and no newly written question passed its quote check.
    """
    conn = runtime.context.conn
    passage_id = state["passage_id"]
    question = next_unused_question(conn, passage_id)
    if question is None:
        # The pool is empty: one LLM call writes 3 questions, and the ones
        # whose quote is really in the passage are saved. write_questions()
        # returns Question objects, which have no id, so the pool is read
        # again to get a saved row.
        write_questions(conn, get_passage(conn, passage_id))
        question = next_unused_question(conn, passage_id)
    if question is None:
        raise TurnStopped(
            f"No question about passage {passage_id} passed the quote check. "
            "(Writing again, then skipping the passage, comes in 'Conditional edges'.)"
        )
    return {"question_id": question["id"]}


def ask(state: State, runtime: Runtime[Context]) -> dict:
    """Print the question and read your answer from the terminal.

    ``input()`` makes the program wait while you type, which works only in a
    terminal. The M1 concept "Pausing for your answer" replaces it with
    LangGraph's ``interrupt()``, so the browser can get the question and
    send the answer back later.

    Args:
        state: Needs ``question_id``.
        runtime: Gives the database connection.

    Returns:
        ``{"answer": ...}``, never empty.
    """
    question = get_question(runtime.context.conn, state["question_id"])
    print(f"\nQuestion: {question['text']}")
    answer = input("Your answer: ")
    # grade_answer() refuses an empty answer, so ask again here: an empty
    # answer never costs an LLM call.
    while not answer.strip():
        answer = input("Please type an answer: ")
    return {"answer": answer}


def grade(state: State, runtime: Runtime[Context]) -> dict:
    """Grade the answer against the question's passage, then mark the question as asked.

    Args:
        state: Needs ``question_id`` and ``answer``.
        runtime: Gives the database connection.

    Returns:
        ``{"result": ...}``: the ``GradedAnswer`` (attempt id, grade, supported, reused).

    Raises:
        FreeLimitReached: From the LLM wrapper, when Groq's free tier runs out.
    """
    conn = runtime.context.conn
    question = get_question(conn, state["question_id"])
    result = grade_answer(conn, question, state["answer"])
    # Only now, with the grade saved, does the question count as asked.
    mark_question_used(conn, question["id"])
    return {"result": result}


def show(state: State, runtime: Runtime[Context]) -> dict:
    """Print the grade (or "no support found") and the passage with its section and pages.

    Args:
        state: Needs ``passage_id`` and ``result``.
        runtime: Gives the database connection.

    Returns:
        ``{}``: it adds nothing to the state.
    """
    passage = get_passage(runtime.context.conn, state["passage_id"])
    result = state["result"]
    print()
    if not result.supported:
        # The grade's quotes weren't all found in the passage (or it had none),
        # so the grade can't be trusted (PLAN.md › Quotes checked in code).
        print("I couldn't find support for this grade.")
    else:
        grade = result.grade
        print(
            f"Verdict: {grade.verdict}"
            + ("  (reused: you gave this answer before)" if result.reused else "")
        )
        # Each list prints only when it has something in it.
        for label, items in (
            ("Missing", grade.missing),
            ("Wrong", grade.wrong),
            ("Extras (no penalty)", grade.extras),
        ):
            if items:
                print(f"{label}:")
                for item in items:
                    print(f"  - {item}")
    pages = passage["page_label"]
    if passage["end_page_label"] != pages:
        pages += f"–{passage['end_page_label']}"
    print(f"\nThe passage: {passage['number']} {passage['title']}, page {pages}\n")
    print(passage["text"])
    return {}
