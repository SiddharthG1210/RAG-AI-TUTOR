"""The study loop's wiring: which steps exist, in what order they run, and where it forks.

M1 concepts "The study loop: state, nodes, edges" and "Conditional edges".
``StateGraph`` is LangGraph's builder. You tell it the state's shape
(``State``) and the context's shape (``Context``), add each node under a
name, and draw edges between the names. ``START`` and ``END`` are LangGraph's
two fixed ends. ``compile()`` checks the wiring (for example, that every edge
points at a node that exists) and gives back the runnable graph.

A plain edge (``add_edge``) always goes to the same node. A conditional edge
(``add_conditional_edges``) calls a routing function right after its node
finishes: the function reads the state and returns the name of the node to
run next. Routing is plain Python, so it costs no tokens (PLAN.md › Design
rules: code is in charge).

    START → pick_passage ─┬─ found one ──────────► pick_question
                          └─ nothing ────────────► topic_not_covered → END

            pick_question ─┬─ got one ───────────► ask_question → grades → show_results → END
                           ├─ none, 1st try ─────► pick_question (write once more)
                           └─ none, 2nd try ─────► skip_the_passage → pick_passage
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from tutor.graph.nodes import (
    MAX_WRITE_TRIES,
    ask,
    grade,
    not_covered,
    pick_passage,
    pick_question,
    show,
    skip_passage,
)
from tutor.graph.state import Context, State


def route_after_pick_passage(state: State) -> str:
    """Choose what follows ``pick_passage``: pick a question, or say the topic isn't covered.

    Args:
        state: The state just after ``pick_passage``; reads ``passage_id``.

    Returns:
        The next node's name: ``"pick_question"``, or ``"topic_not_covered"``.
    """
    # 1. pick_passage found no passage (passage_id is None) → the node that
    #    says the material doesn't cover this.
    if state["passage_id"] is None:
        choice = "topic_not_covered"
    # 2. Otherwise → the node that picks a question.
    else:
        choice = "pick_question"
    return choice


def route_after_pick_question(state: State) -> str:
    """Choose what follows ``pick_question``: ask it, write the questions again, or skip.

    Args:
        state: The state just after ``pick_question``; reads ``question_id``
            and ``write_tries``.

    Returns:
        The next node's name: ``"ask_question"``, ``"pick_question"`` or
        ``"skip_the_passage"``.
    """
    # 1. There's a question (question_id isn't None) → the node that asks it.
    if state["question_id"] is not None:
        choice = "ask_question"
    # 2. No question, and fewer than MAX_WRITE_TRIES writes so far → back to
    #    pick_question, which writes once more.
    elif state["write_tries"] < MAX_WRITE_TRIES:
        choice = "pick_question"
    # 3. No question after MAX_WRITE_TRIES writes → the node that skips the passage.
    else:
        choice = "skip_the_passage"
    return choice


def build_graph() -> CompiledStateGraph:
    """Build the study loop: add its nodes, draw its edges and forks, and compile it.

    Returns:
        The compiled graph. Run one turn with
        ``graph.invoke({"topic": ...}, context=Context(conn=conn))``.
    """
    builder = StateGraph(State, context_schema=Context)
    builder.add_node("pick_passage", pick_passage)
    builder.add_node("pick_question", pick_question)
    builder.add_node("ask_question", ask)
    builder.add_node("grades", grade)
    builder.add_node("show_results", show)
    builder.add_node("topic_not_covered", not_covered)
    builder.add_node("skip_the_passage", skip_passage)
    builder.add_edge(START, "pick_passage")
    builder.add_conditional_edges("pick_passage", route_after_pick_passage)
    builder.add_conditional_edges("pick_question", route_after_pick_question)
    builder.add_edge("topic_not_covered", END)
    builder.add_edge("skip_the_passage", "pick_passage")
    builder.add_edge("ask_question", "grades")
    builder.add_edge("grades", "show_results")
    builder.add_edge("show_results", END)
    return builder.compile()
