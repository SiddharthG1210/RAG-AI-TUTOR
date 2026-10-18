"""The study loop's wiring: which steps exist, and in what order they run.

M1 concept "The study loop: state, nodes, edges". ``StateGraph`` is
LangGraph's builder. You tell it the state's shape (``State``) and the
context's shape (``Context``), add each node under a name, and draw edges
between the names. ``START`` and ``END`` are LangGraph's two fixed ends.
``compile()`` checks the wiring (for example, that every edge points at a
node that exists) and gives back the runnable graph.

The turn is one straight line for now:

    START → pick_passage → pick_question → ask → grade → show → END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from tutor.graph.nodes import ask, grade, pick_passage, pick_question, show
from tutor.graph.state import Context, State


def build_graph() -> CompiledStateGraph:
    """Build the study loop: add its five nodes, draw its edges, and compile it.

    Returns:
        The compiled graph. Run one turn with
        ``graph.invoke({"topic": ...}, context=Context(conn=conn))``.
    """
    builder = StateGraph(State, context_schema=Context)
    builder.add_node("pick_passage", pick_passage)
    # Only adds questions when the pool is empty; otherwise picks directly from the pool.
    builder.add_node("pick_question", pick_question)
    builder.add_node("ask_question", ask)
    builder.add_node("grades", grade)
    builder.add_node("show_results", show)
    builder.add_edge(START, "pick_passage")
    builder.add_edge("pick_passage", "pick_question")
    builder.add_edge("pick_question", "ask_question")
    builder.add_edge("ask_question", "grades")
    builder.add_edge("grades", "show_results")
    builder.add_edge("show_results", END)
    return builder.compile()
