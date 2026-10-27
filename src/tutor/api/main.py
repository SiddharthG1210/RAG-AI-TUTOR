"""The web app: one FastAPI server, with the Gradio study page mounted at ``/``.

Run from the repo root, after ingesting chapter 1 (it needs the Groq key in ``.env``):

    uv run uvicorn tutor.api.main:create_app --factory

then open http://127.0.0.1:8000. Ctrl+C stops the server.

M1 concept "API and UI". uvicorn is the program that listens for web
requests on a port and hands each one to the app. FastAPI is the app: it
decides which code answers which web address. It has no routes of its own
yet: the Gradio page is *mounted* at ``/``, meaning every request under ``/``
(the page itself, and each click) goes to Gradio. The page's buttons call
Python functions directly (``ui.py``), so no JSON routes are needed. The
progress page (M6) and demo mode (M7) join this same app later.

``--factory`` tells uvicorn to call ``create_app()`` to get the app, instead
of importing a ready-made ``app`` variable. So importing this module (as the
tests do) doesn't build a graph or need ``data/tutor.db``.
"""

from pathlib import Path

import gradio as gr
from fastapi import FastAPI

from tutor.api.ui import build_page
from tutor.config import get_settings
from tutor.graph.build import build_graph

DB_PATH = Path("data/tutor.db")
"""The real database, relative to the repo root, where the server is started."""


def create_app(db_path: Path = DB_PATH) -> FastAPI:
    """Make the web app: check the setup, build the graph once, and mount the page.

    Args:
        db_path: The tutor's database file.

    Returns:
        The FastAPI app, with the study page at ``/``.

    Raises:
        FileNotFoundError: If there's no database yet (no chapter ingested).
        pydantic.ValidationError: If ``LLM_API_KEY`` is missing from ``.env``.
    """
    # Fail at start-up, with a clear message, rather than at the first click.
    # connect() would quietly create an empty database instead.
    if not db_path.exists():
        raise FileNotFoundError(
            f"No database at {db_path}. Ingest a chapter first: scripts/ingest.py --chapter 1"
        )
    get_settings()
    # Built once for the whole server: its in-memory checkpointer holds every
    # paused session. A graph built per click would start with an empty one,
    # and no click could find the session it should resume.
    graph = build_graph()
    app = FastAPI(title="Study Tutor")
    # mount_gradio_app() adds the page's routes to the FastAPI app, under path.
    return gr.mount_gradio_app(app, build_page(graph, db_path), path="/")
