"""The web layer: the study page in the browser, and the server that hosts it.

M1 concept "API and UI". The page is the graph's other caller, next to
``scripts/study.py``: it starts a session, shows what each pause hands out,
and resumes the graph with your click. Each click is a separate web request,
so the paused graph waits in the checkpointer between clicks.

- ``session.py``: ``start_session()`` and ``resume_session()``, the
  terminal's resume loop cut into one pass per click.
- ``ui.py``: the Gradio page: its boxes and buttons, which function each
  click calls, and how a pause note becomes what's on screen.
- ``main.py``: ``create_app()``, the FastAPI app with the page mounted at
  ``/``, which uvicorn runs.
"""
