"""The study page: its boxes and buttons, what each click does, and what the screen shows.

M1 concept "API and UI". Gradio builds a web page from Python: each
component (``gr.Textbox``, ``gr.Button``, ``gr.Markdown``) is a box on the
page, and ``button.click(fn, inputs=..., outputs=...)`` says which Python
function a click calls, which boxes' values it receives, and which boxes its
return value fills. Gradio's own JavaScript in the browser sends the click to
the server and puts the returned values on the page. Only what a function
*returns* reaches the page; anything it prints stays in the server's window.

Every click takes the same path:

    click → a handler: on_study(), on_submit() or on_stop()
          → start_session() or resume_session() (session.py) → the pause note
          → screen_after() → a Screen: the whole page as plain values
          → render() (inside build_page) → the boxes' new values

``Screen`` sits between the graph and Gradio on purpose. The handlers deal
only in plain values (no Gradio objects), so the tests can check what each
click would show without a browser, and ``render()`` is the one place that
knows the boxes.

Submit is greyed out from the moment you click it until the grade is shown
(PLAN.md › Same answer, same grade): a second click while the first answer is
being graded would otherwise reach the LLM twice.
"""

from dataclasses import dataclass
from pathlib import Path

import gradio as gr
from langgraph.graph.state import CompiledStateGraph

from tutor.api.session import SessionLost, resume_session, start_session
from tutor.llm import FreeLimitReached

CREDIT = (
    "From [*Dive into Deep Learning*](https://d2l.ai) by Aston Zhang, Zachary C. Lipton, "
    "Mu Li and Alexander J. Smola, licensed under "
    "[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)."
)
"""The book's credit, shown under every passage (PLAN.md › Citations)."""

START_PROMPT = "**What do you want to study?** Type a topic, or let the tutor choose."
NEXT_PROMPT = "**Next:** type a topic, let the tutor choose, or stop."

SESSION_ENDED = "Session ended. Type a topic or press You choose to start a new one."
FREE_LIMIT = "Free limit reached: Groq's free tier is used up for now. Try again later."
SESSION_LOST = (
    "This session was lost when the server restarted. "
    "Type a topic or press You choose to start a new one."
)


@dataclass(frozen=True)
class Screen:
    """What the page shows after a click, as plain values: no Gradio objects.

    ``render()`` turns it into the boxes' new values. A field left at its
    default hides its part of the page.

    Attributes:
        session_id: The session this tab is in, or ``None`` on the start
            screen (no session running).
        status: A line at the top, e.g. "Session ended.", or "" for none.
        question: The question's text, or ``None`` to hide the question, the
            answer box and Submit.
        answer: The text in the answer box.
        answering: True at the "answer" pause: the answer box can be typed in
            and Submit is shown. False after the grade: your answer stays,
            locked.
        result: Markdown for the grade, or for the message shown instead of
            one, or ``None`` to hide it.
        passage_label: The passage box's title, e.g. "The passage: 1.3.1.2
            Classification, page 23–24".
        passage: The passage's text, or ``None`` to hide it and the credit.
        choosing: True to show the topic box and the Study this topic / You
            choose buttons (plus Stop, during a session).
    """

    session_id: str | None
    status: str = ""
    question: str | None = None
    answer: str = ""
    answering: bool = False
    result: str | None = None
    passage_label: str = ""
    passage: str | None = None
    choosing: bool = True


def result_markdown(shown: dict) -> str:
    """Write the grade as Markdown, or the message that replaces it.

    Args:
        shown: What the ``show`` or ``not_covered`` node packed, from the
            "next" pause's note.

    Returns:
        For a grade: the verdict in bold (with "reused" when the same answer
        was graded before), then each non-empty list as bullets: Missing,
        Wrong, Extras and Based on (the grade's quotes). Otherwise the message.
    """
    if shown["kind"] != "graded":
        return shown["message"]
    reused = " (reused: you gave this answer before)" if shown["reused"] else ""
    lines = [f"**Verdict: {shown['verdict']}**{reused}"]
    # Each list shows only when it has something in it. In Markdown, a line
    # starting with "- " is a bullet.
    for label, key in (
        ("Missing", "missing"),
        ("Wrong", "wrong"),
        ("Extras (no penalty)", "extras"),
        ("Based on", "quotes"),
    ):
        if shown[key]:
            lines.append(f"\n{label}:")
            lines.extend(f"- {item}" for item in shown[key])
    return "\n".join(lines)


def screen_after(session_id: str, note: dict, question: str = "", answer: str = "") -> Screen:
    """Turn a pause note into the screen: the question to answer, or what the turn shows.

    Args:
        session_id: The session the note belongs to.
        note: What the pausing node handed to ``interrupt()``: ``{"pause":
            "answer", "question": ...}`` or ``{"pause": "next", "shown": ...}``.
        question: After Submit, the question you answered, so it stays on
            screen above its grade. "" when the turn had no question (a topic
            the material doesn't cover).
        answer: After Submit, your answer, shown locked under the question.

    Returns:
        The screen to render.
    """
    if note["pause"] == "answer":
        # Pause 1: only the question and an empty, open answer box.
        return Screen(session_id, question=note["question"], answering=True, choosing=False)
    # Pause 2: the grade (or the message), the passage if there is one, and the
    # choice of what's next. "not_covered" has no passage.
    shown = note["shown"]
    passage = shown.get("passage")
    return Screen(
        session_id,
        question=question or None,
        answer=answer,
        result=result_markdown(shown),
        passage_label=f"The passage: {passage['section']}, page {passage['pages']}"
        if passage
        else "",
        passage=passage["text"] if passage else None,
    )


def on_study(
    graph: CompiledStateGraph, db_path: Path, session_id: str | None, topic: str | None
) -> Screen:
    """Study this topic / You choose: start a session, or go on to the next turn of this one.

    Args:
        graph: The compiled study graph.
        db_path: The tutor's database file.
        session_id: The tab's session, or ``None`` on the start screen.
        topic: The typed topic, or ``None`` for "You choose".

    Returns:
        The question to answer, or the "doesn't cover this" message with the
        choice again. The start screen with a message if the free limit is
        reached or the session was lost.
    """
    try:
        if session_id is None:
            session_id, note = start_session(graph, db_path, topic)
        else:
            # The reply wait_for_next expects at the "next" pause.
            note = resume_session(graph, db_path, session_id, {"stop": False, "topic": topic})
    except FreeLimitReached:
        # The turn stopped halfway, so the session ends here, as in the
        # terminal. A question that wasn't graded stays unused and comes back.
        return Screen(None, status=FREE_LIMIT)
    except SessionLost:
        return Screen(None, status=SESSION_LOST)
    return screen_after(session_id, note)


def on_submit(
    graph: CompiledStateGraph, db_path: Path, session_id: str, question: str, answer: str
) -> Screen:
    """Submit: resume the paused session with your answer, and show its grade.

    The caller has already refused an empty answer, so ``grade_answer()``
    never gets one.

    Args:
        graph: The compiled study graph.
        db_path: The tutor's database file.
        session_id: The tab's session, paused at the "answer" pause.
        question: The question on screen, kept above the grade.
        answer: Your answer, as typed.

    Returns:
        The grade, the passage and the choice of what's next. The start screen
        with a message if the free limit is reached or the session was lost.
    """
    try:
        note = resume_session(graph, db_path, session_id, answer)
    except FreeLimitReached:
        return Screen(None, status=FREE_LIMIT)
    except SessionLost:
        return Screen(None, status=SESSION_LOST)
    return screen_after(session_id, note, question, answer)


def on_stop(graph: CompiledStateGraph, db_path: Path, session_id: str) -> Screen:
    """Stop: end the session, and go back to the start screen.

    Args:
        graph: The compiled study graph.
        db_path: The tutor's database file.
        session_id: The tab's session, paused at the "next" pause.

    Returns:
        The start screen, saying the session ended.
    """
    try:
        resume_session(graph, db_path, session_id, {"stop": True, "topic": None})
    except SessionLost:
        # The session is already gone (a restart): stopping it changes
        # nothing, and the start screen is right either way.
        pass
    return Screen(None, status=SESSION_ENDED)


def build_page(graph: CompiledStateGraph, db_path: Path) -> gr.Blocks:
    """Build the study page: lay out its boxes, and connect each button to its handler.

    Args:
        graph: The compiled study graph, built once for the whole server, so
            every click finds the paused sessions in the same checkpointer.
        db_path: The tutor's database file.

    Returns:
        The page, ready to mount in the FastAPI app (``main.py``).
    """
    # gr.Blocks builds a page from the components created inside the "with"
    # block, top to bottom. analytics_enabled=False stops Gradio from sending
    # usage statistics to its makers.
    with gr.Blocks(title="Study Tutor", analytics_enabled=False) as page:
        # gr.State holds a value for this browser tab only, kept on the server:
        # here the session id. Two tabs are two separate sessions.
        session_state = gr.State(None)
        gr.Markdown("# Study Tutor")
        status_text = gr.Markdown(visible=False)

        # The question, the answer box and Submit. visible=False hides a part
        # until render() shows it.
        with gr.Column(visible=False) as question_area:
            gr.Markdown("### Question")
            question_text = gr.Markdown()
            answer_box = gr.Textbox(label="Your answer", lines=4)
            submit_button = gr.Button("Submit", variant="primary")

        # The grade (or the message shown instead of one).
        result_text = gr.Markdown(visible=False)

        # The passage, in a read-only text box: it shows the text exactly as
        # stored, where Markdown would turn a stray "*" or "_" into formatting.
        with gr.Column(visible=False) as passage_area:
            passage_box = gr.Textbox(label="The passage", lines=10, interactive=False)
            gr.Markdown(CREDIT)

        # What to study: the same choice at the start and after every turn.
        with gr.Column() as choice_area:
            choice_prompt = gr.Markdown(START_PROMPT)
            topic_box = gr.Textbox(label="Topic", placeholder="e.g. gradient descent")
            with gr.Row():
                study_button = gr.Button("Study this topic", variant="primary")
                you_choose_button = gr.Button("You choose")
                stop_button = gr.Button("Stop", visible=False)

        # Every box a click can change. A handler returns a dict keyed by these
        # components, and Gradio updates only the boxes in that dict.
        everything = [
            session_state,
            status_text,
            question_area,
            question_text,
            answer_box,
            submit_button,
            result_text,
            passage_area,
            passage_box,
            choice_area,
            choice_prompt,
            topic_box,
            stop_button,
        ]

        def render(screen: Screen) -> dict:
            """Turn a Screen into every box's new value and visibility.

            ``gr.update(...)`` names the properties of a box to change, such
            as ``value``, ``visible`` or ``interactive`` (whether you can type
            in it or click it); a plain value only sets the box's value.
            """
            in_session = screen.session_id is not None
            return {
                session_state: screen.session_id,
                status_text: gr.update(value=screen.status, visible=bool(screen.status)),
                question_area: gr.update(visible=screen.question is not None),
                question_text: screen.question or "",
                answer_box: gr.update(value=screen.answer, interactive=screen.answering),
                submit_button: gr.update(
                    visible=screen.answering, interactive=True, value="Submit"
                ),
                result_text: gr.update(
                    value=screen.result or "", visible=screen.result is not None
                ),
                passage_area: gr.update(visible=screen.passage is not None),
                passage_box: gr.update(value=screen.passage or "", label=screen.passage_label),
                choice_area: gr.update(visible=screen.choosing),
                choice_prompt: NEXT_PROMPT if in_session else START_PROMPT,
                topic_box: "",
                stop_button: gr.update(visible=in_session),
            }

        def study_topic(session_id: str | None, topic: str) -> dict:
            """Study this topic (or Enter in the box): refuse an empty topic, else study it."""
            if not topic.strip():
                # gr.Warning shows a small notice on the page. A gr.update()
                # with nothing in it changes nothing. (Gradio refuses an empty
                # dict: it needs at least one box to "update".)
                gr.Warning("Type a topic first, or press You choose.")
                return {topic_box: gr.update()}
            return render(on_study(graph, db_path, session_id, topic.strip()))

        def you_choose(session_id: str | None) -> dict:
            """You choose: the tutor picks the next new passage."""
            return render(on_study(graph, db_path, session_id, None))

        def lock_submit() -> dict:
            """Grey out the answer box and Submit the moment Submit is clicked."""
            return {
                answer_box: gr.update(interactive=False),
                submit_button: gr.update(interactive=False, value="Grading…"),
            }

        def submit_answer(session_id: str, question: str, answer: str) -> dict:
            """Submit: refuse an empty answer, else grade it."""
            if not answer.strip():
                gr.Warning("Please type an answer first.")
                # Undo lock_submit, so you can type it.
                return {
                    answer_box: gr.update(interactive=True),
                    submit_button: gr.update(interactive=True, value="Submit"),
                }
            return render(on_submit(graph, db_path, session_id, question, answer))

        def stop(session_id: str) -> dict:
            """Stop: end the session."""
            return render(on_stop(graph, db_path, session_id))

        # The wiring: which function each click calls, with which boxes'
        # values (inputs, in the function's argument order), and which boxes
        # its return value may change (outputs).
        study_button.click(study_topic, inputs=[session_state, topic_box], outputs=everything)
        topic_box.submit(study_topic, inputs=[session_state, topic_box], outputs=everything)
        you_choose_button.click(you_choose, inputs=session_state, outputs=everything)
        stop_button.click(stop, inputs=session_state, outputs=everything)
        # Two steps, one after the other: lock_submit runs at once
        # (queue=False skips Gradio's queue of waiting clicks), so the button is
        # grey before grading starts; .then() runs submit_answer after it.
        submit_button.click(lock_submit, outputs=[answer_box, submit_button], queue=False).then(
            submit_answer,
            inputs=[session_state, question_text, answer_box],
            outputs=everything,
        )
    return page
