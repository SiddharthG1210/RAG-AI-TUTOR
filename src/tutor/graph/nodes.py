"""The steps of a study turn: each one reads the state, calls a tool, and returns its changes.

M1 concepts "The study loop: state, nodes, edges" and "Conditional edges". A
LangGraph node is a plain function. LangGraph calls it with the current state
(and, because it has a ``runtime`` parameter, the run's ``Runtime``, which
carries our ``Context``), and merges the dict it returns into the state. A
node never calls another node: the edges in ``build.py`` decide what runs
next, and its routing functions choose between edges by reading the state.

The turn, with the keys each node adds:

    {"topic": ...}
       │ pick_passage   search() for a typed topic, else next_new_passage()
    + passage_id (None: nothing close enough), write_tries = 0
       │ pick_question  next_unused_question(), writing questions first if the pool is empty
    + question_id (None: no question passed), write_tries + 1 if it wrote
       │ ask            prints the question, reads your answer with input()
    + answer
       │ grade          grade_answer(), then mark_question_used()
    + result
       │ show           prints the grade and the passage (adds nothing)

Two more nodes sit on the forks: ``not_covered`` prints "the material doesn't
cover this" (a typed topic with no passage close enough), and
``skip_passage`` saves a passage whose questions failed twice, so it's never
picked again. A node that can come up empty records ``None`` in the state
instead of raising, so the routing function can read what happened.
"""

from langgraph.runtime import Runtime

from tutor.db import (
    get_passage,
    get_question,
    mark_question_used,
    next_new_passage,
    next_unused_question,
    save_skip,
    skipped_passage_ids,
)
from tutor.grading import grade_answer
from tutor.graph.state import Context, State
from tutor.llm import UnusableAnswer
from tutor.questions import write_questions
from tutor.retrieval.search import search

MAX_WRITE_TRIES = 2
"""How many times a passage's questions are written before it's skipped
(PLAN.md › Quotes checked in code: write once more, then skip the passage)."""


def pick_passage(state: State, runtime: Runtime[Context]) -> dict:
    """Pick the turn's passage: the closest one to a typed topic, or the next new one.

    Skipped passages are passed over either way. Picking a passage starts its
    count of question-writing tries again at 0.

    Args:
        state: Needs ``topic``.
        runtime: Gives the database connection, as ``runtime.context.conn``.

    Returns:
        ``{"passage_id": ..., "write_tries": 0}``. ``passage_id`` is ``None``
        when no passage is close enough to the typed topic, or every close
        one is skipped.

    Raises:
        ValueError: From ``next_new_passage()``, if every passage has been
            asked about or skipped.
    """
    conn = runtime.context.conn
    topic = state["topic"]
    if topic is None:
        # "You choose": the first passage in book order with no question
        # asked yet; next_new_passage() already leaves skipped ones out.
        passage_id = next_new_passage(conn)["id"]
    else:
        # A typed topic: search() gives up to 3 (score, passage) pairs, best
        # first, only those above the cut-off (PLAN.md › Topic search). Take
        # the best one that isn't skipped. next(..., None) gives the first
        # item of the generator, or None when it has none: nothing close
        # enough, which the routing function sends to "topic_not_covered".
        skipped = skipped_passage_ids(conn)

        # next(..., None) means: give the first item from the list that search()
        # creates which is also not in the "skipped" table, and if no such
        # passage is found, return None.
        passage_id = next(
            (
                passage["id"]
                for _score, passage in search(conn, topic)
                if passage["id"] not in skipped
            ),
            None,
        )
    return {"passage_id": passage_id, "write_tries": 0}


def pick_question(state: State, runtime: Runtime[Context]) -> dict:
    """Pick the passage's next unused question, writing a pool of questions first if it's empty.

    Each time it writes, it adds 1 to ``write_tries``. A write that saves no
    question (no quote passed its check) and a write whose LLM answer was
    unusable (``UnusableAnswer``, e.g. 2 questions instead of 3) both leave
    the pool empty, so both count as a failed try. The routing function then
    sends the graph back here to write once more, or on to ``skip_passage``.

    The question is only picked here, not marked used: ``grade`` marks it
    after the answer is graded, so a question you stop before answering
    comes back next session.

    Args:
        state: Needs ``passage_id`` and ``write_tries``.
        runtime: Gives the database connection.

    Returns:
        ``{"question_id": ..., "write_tries": ...}``. ``question_id`` is
        ``None`` when the pool is still empty after writing.
    """
    conn = runtime.context.conn
    passage_id = state["passage_id"]
    write_tries = state["write_tries"]
    question = next_unused_question(conn, passage_id)
    if question is None:
        # The pool is empty: one LLM call writes 3 questions, and the ones
        # whose quote is really in the passage are saved. write_questions()
        # returns Question objects, which have no id, so the pool is read
        # again to get a saved row.
        write_tries += 1
        try:
            write_questions(conn, get_passage(conn, passage_id))
        except UnusableAnswer:
            # The LLM's answer couldn't be used, so nothing was saved: the
            # pool stays empty, and this try counts as failed. Any other
            # error (free limit reached, a bug) still stops the turn.
            pass
        question = next_unused_question(conn, passage_id)
    return {"question_id": None if question is None else question["id"], "write_tries": write_tries}


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


def not_covered(state: State, runtime: Runtime[Context]) -> dict:
    """Say the material doesn't cover the typed topic (PLAN.md's fallback message).

    Args:
        state: Needs ``topic``.
        runtime: Unused; every node takes it.

    Returns:
        ``{}``: it adds nothing to the state.
    """
    print(f"\nThe material doesn't cover this: {state['topic']!r}.")
    return {}


def skip_passage(state: State, runtime: Runtime[Context]) -> dict:
    """Skip a passage for good after its questions failed twice, so it's never picked again.

    Args:
        state: Needs ``passage_id``.
        runtime: Gives the database connection.

    Returns:
        ``{}``: the skip is saved in the database, not the state.
    """
    passage_id = state["passage_id"]
    reason = f"no question passed its quote check in {MAX_WRITE_TRIES} tries"
    save_skip(runtime.context.conn, passage_id, reason)
    print(f"\n(Skipping passage {passage_id}: {reason}. Picking another one.)")
    return {}
