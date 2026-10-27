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
       │ show           packs the grade and the passage, ready to show
    + shown
       │ wait_for_next  PAUSE 2: hands out shown; asks for the next topic, "You choose", or stop
    + topic, stop       (not stopping: back to pick_passage for the next turn)

Two more nodes sit on the forks: ``not_covered`` packs "the material doesn't
cover this" as ``shown`` (a typed topic with no passage close enough), and
``skip_passage`` saves a passage whose questions failed twice, so it's never
picked again. A node that can come up empty records ``None`` in the state
instead of raising, so the routing function can read what happened.

**Nothing is printed (M1 concept "API and UI").** In the web server, a node
runs inside the server program, so ``print()`` would only reach the server's
console, never the page. What a turn shows is put in the state as ``shown``
instead, and ``wait_for_next`` hands it out in its pause note: whoever runs
the graph (the page, or ``scripts/study.py``) shows it.

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

import logging

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

# Python's logging module: getLogger(__name__) gives this file its own named
# logger ("tutor.graph.nodes"). With nothing configured, logging still prints
# warnings and errors to the console (its "last resort" handler), so a skip
# shows up in the terminal and in the web server's window alike.
logger = logging.getLogger(__name__)


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
    """Pack what the turn shows: the grade (or "no support found") and the passage.

    It prints nothing (see "Nothing is printed" above). It copies the parts
    the screen needs, from two places, into one plain dict: the grade from
    the state (``result``), and the passage from the database (the state only
    holds its id). Everything in it is already in the form it's shown in, so
    the caller never looks anything up or formats a page number.

    Args:
        state: Needs ``passage_id`` and ``result``.
        runtime: Gives the database connection.

    Returns:
        ``{"shown": ...}``, one of two kinds, both with the passage:

        - ``"kind": "graded"``: ``verdict``, ``reused``, and the lists
          ``missing``, ``wrong``, ``extras`` and ``quotes`` (what the grade
          rests on).
        - ``"kind": "no_support"``: ``message``, PLAN.md's "I couldn't find
          support for this grade", shown instead of the grade.

        ``"passage"`` is ``{"section": "1.3.1.2 Classification", "pages":
        "23–24", "text": ...}``.
    """
    passage = get_passage(runtime.context.conn, state["passage_id"])
    result = state["result"]
    # The printed page numbers, e.g. "23", or "23–24" when the passage runs
    # onto the next page.
    pages = passage["page_label"]
    if passage["end_page_label"] != pages:
        pages += f"–{passage['end_page_label']}"
    shown_passage = {
        "section": f"{passage['number']} {passage['title']}",
        "pages": pages,
        "text": passage["text"],
    }
    if not result.supported:
        # The grade's quotes weren't all found in the passage (or it had none),
        # so the grade can't be trusted and isn't shown (PLAN.md › Quotes
        # checked in code). The passage still is.
        shown = {
            "kind": "no_support",
            "message": "I couldn't find support for this grade.",
            "passage": shown_passage,
        }
    else:
        grade = result.grade
        # list(...) copies each list, so the dict holds plain lists of its own.
        shown = {
            "kind": "graded",
            "verdict": grade.verdict,
            "reused": result.reused,
            "missing": list(grade.missing),
            "wrong": list(grade.wrong),
            "extras": list(grade.extras),
            "quotes": list(grade.quotes),
            "passage": shown_passage,
        }
    return {"shown": shown}


def not_covered(state: State, runtime: Runtime[Context]) -> dict:
    """Pack why the typed topic gave no passage: nothing close, or nothing new this session.

    "The material doesn't cover this" is PLAN.md's fallback message. When
    close passages exist but you've already been asked about each one this
    session, that message would be wrong, so it says so instead. Either way
    the session goes on: the next pause asks for another topic.

    The message is written here, once, so the page and the terminal say the
    same words.

    Args:
        state: Needs ``topic`` and ``nothing_new``.
        runtime: Unused; every node takes it.

    Returns:
        ``{"shown": {"kind": "not_covered", "message": ...}}``: no grade and
        no passage, only the message.
    """
    if state["nothing_new"]:
        message = f"Nothing new on {state['topic']!r} this session: you've had every close passage."
    else:
        message = f"The material doesn't cover this: {state['topic']!r}."
    return {"shown": {"kind": "not_covered", "message": message}}


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
    # A warning on the console, not something the page shows: the next
    # passage's question simply appears, and the skip is saved for good.
    # logging fills in the %s placeholders itself, only if it prints the line.
    logger.warning("Skipping passage %s: %s. Picking another one.", passage_id, reason)
    return {}


def wait_for_next(state: State, runtime: Runtime[Context]) -> dict:
    """Pause 2: hand out what the turn shows, then ask for the next topic, "You choose", or stop.

    It's the same choice as at the start of a session. Your reply sets the
    next turn's ``topic``, so a new topic leads somewhere new (and the same
    topic typed again moves on to its next-best passage). In M2 and M3,
    Ask the book, "Grade is wrong" and "Flag" happen during this pause.

    Both paths into this node write ``shown`` first (``show`` after a grade,
    ``not_covered`` otherwise), so the note always describes this turn.

    Args:
        state: Needs ``shown``, which it hands out with the pause.
        runtime: Unused; every node takes it.

    Returns:
        ``{"stop": ..., "topic": ...}``: from the caller's reply.
    """
    # pause 2.
    # 1. Call interrupt() with {"pause": "next", "shown": <what to show>}.
    #    Reading state["shown"] is a harmless read: on resume this line runs
    #    again and reads the same value.
    #    On resume, it returns the caller's reply: a dict like
    #    {"stop": False, "topic": "gradient descent"}. "topic": None means
    #    "You choose"; "stop": True means end the session.
    reply = interrupt({"pause": "next", "shown": state["shown"]})
    # 2. Return the reply's "stop" and "topic", each under the same key, so the
    #    routing function can read "stop" and pick_passage the new "topic".
    return {"stop": reply["stop"], "topic": reply["topic"]}
