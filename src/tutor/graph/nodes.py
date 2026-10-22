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
       │ ask            PAUSE 1: hands the question out with interrupt(), takes the answer back in
    + answer
       │ grade          grade_answer(), then mark_question_used()
    + result
       │ show           prints the grade and the passage (adds nothing)
       │ wait_for_next  PAUSE 2: asks for the next topic, "You choose", or stop
    + topic, stop       (not stopping: back to pick_passage for the next turn)

Two more nodes sit on the forks: ``not_covered`` prints "the material doesn't
cover this" (a typed topic with no passage close enough), and
``skip_passage`` saves a passage whose questions failed twice, so it's never
picked again. A node that can come up empty records ``None`` in the state
instead of raising, so the routing function can read what happened.

**Pausing (M1 concept "Pausing for your answer").** ``interrupt(value)`` stops
the graph and hands ``value`` out to whoever called ``graph.invoke()``; the
state is saved by the checkpointer. The caller resumes with
``graph.invoke(Command(resume=reply), ...)``, and LangGraph runs the paused
node again from its first line: this time ``interrupt()`` returns ``reply``.
So a pausing node does nothing before ``interrupt()`` except harmless reads
(PLAN.md › Pausing for your answer): a save or an LLM call there would happen
twice. What each pause hands out is a dict with a ``"pause"`` key, so the
caller can tell the two pauses apart.
"""

from langgraph.runtime import Runtime
from langgraph.types import interrupt

from tutor.db import (
    get_passage,
    get_question,
    mark_question_used,
    next_new_passage,
    next_unused_question,
    passages_asked_in_session,
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

    Skipped passages are passed over either way. For a typed topic, so are
    the passages you've been asked about in this session, so typing the same
    topic again moves on to the next-best passage (one question per passage
    per session). Picking a passage starts its count of question-writing
    tries again at 0.

    Args:
        state: Needs ``topic``.
        runtime: Gives the database connection, as ``runtime.context.conn``.

    Returns:
        ``{"passage_id": ..., "write_tries": 0, "nothing_new": ...}``.
        ``passage_id`` is ``None`` when no passage is close enough to the
        typed topic (``nothing_new`` False), or every close one was asked
        this session or skipped (``nothing_new`` True).

    Raises:
        ValueError: From ``next_new_passage()``, if every passage has been
            asked about or skipped.
    """
    conn = runtime.context.conn
    topic = state["topic"]
    nothing_new = False
    if topic is None:
        # "You choose": the first passage in book order with no question
        # asked yet; next_new_passage() already leaves skipped ones out.
        passage_id = next_new_passage(conn)["id"]
    else:
        # A typed topic: search() gives up to 3 (score, passage) pairs, best
        # first, only those above the cut-off (PLAN.md › Topic search). Take
        # the best one that isn't skipped or already asked this session.
        # "|" joins two sets. next(..., None) gives the first item of the
        # generator, or None when it has none, which the routing function
        # sends to "topic_not_covered".
        results = search(conn, topic)
        session_id = runtime.context.session_id  # this run's id, e.g. "3f9a…"
        asked = passages_asked_in_session(conn, session_id)  # a set, e.g. {12, 13}
        skipped = skipped_passage_ids(conn)  # a set, e.g. {40}
        left_out = asked | skipped  # "|" joins two sets: {12, 13, 40}

        # next(..., None) means: give the first item from the list that search()
        # creates which is also not left out, and if no such passage is found,
        # return None.
        passage_id = next(
            (passage["id"] for _score, passage in results if passage["id"] not in left_out),
            None,
        )
        # Close passages exist, but every one was left out: nothing NEW on
        # this topic this session (rather than nothing close at all).
        nothing_new = passage_id is None and bool(results)
    return {"passage_id": passage_id, "write_tries": 0, "nothing_new": nothing_new}


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
    """Pause 1: hand the question out to whoever runs the graph, and take your answer back in.

    The graph stops at ``interrupt()``, so it works the same whether the
    caller is the terminal (``scripts/study.py``) or, later, the browser.
    The caller shows the question and refuses an empty answer before
    resuming; ``grade_answer()`` refuses one too, loudly.

    Args:
        state: Needs ``question_id``.
        runtime: Gives the database connection.

    Returns:
        ``{"answer": ...}``: what the caller resumed with.
    """
    # A harmless read: it runs again on resume, and gives the same question.
    question = get_question(runtime.context.conn, state["question_id"])
    # pause 1.
    # 1. Call interrupt() with what the caller needs to show you:
    #    {"pause": "answer", "question": <the question's text>}.
    #    1st run: the graph stops here. On resume: interrupt() returns your answer.
    answer = interrupt({"pause": "answer", "question": question["text"]})
    # 2. Return the answer under the "answer" key.
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
    # The session id is saved with the attempt, so pick_passage can tell
    # which passages were asked this session.
    result = grade_answer(conn, question, state["answer"], runtime.context.session_id)
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
    """Say why the typed topic gave no passage: nothing close, or nothing new this session.

    "The material doesn't cover this" is PLAN.md's fallback message. When
    close passages exist but you've already been asked about each one this
    session, that message would be wrong, so it says so instead. Either way
    the session goes on: the next pause asks for another topic.

    Args:
        state: Needs ``topic`` and ``nothing_new``.
        runtime: Unused; every node takes it.

    Returns:
        ``{}``: it adds nothing to the state.
    """
    if state["nothing_new"]:
        print(f"\nNothing new on {state['topic']!r} this session: you've had every close passage.")
    else:
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


def wait_for_next(state: State, runtime: Runtime[Context]) -> dict:
    # no state required because it doesnt need anythinf from the state to execute its purpose
    """Pause 2: after a turn, ask for the next topic, "You choose", or stop.

    It's the same choice as at the start of a session. Your reply sets the
    next turn's ``topic``, so a new topic leads somewhere new (and the same
    topic typed again moves on to its next-best passage). In M2 and M3,
    Ask the book, "Grade is wrong" and "Flag" happen during this pause.

    Args:
        state: Unused before the pause; every node takes it.
        runtime: Unused; every node takes it.

    Returns:
        ``{"stop": ..., "topic": ...}``: from the caller's reply.
    """
    # pause 2.
    # 1. Call interrupt() with {"pause": "next"}.
    #    On resume, it returns the caller's reply: a dict like
    #    {"stop": False, "topic": "gradient descent"}. "topic": None means
    #    "You choose"; "stop": True means end the session.
    reply = interrupt({"pause": "next"})
    # 2. Return the reply's "stop" and "topic", each under the same key, so the
    #    routing function can read "stop" and pick_passage the new "topic".
    return {"stop": reply["stop"], "topic": reply["topic"]}
