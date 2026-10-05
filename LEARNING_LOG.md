# Learning log

The handoff between sessions: where the last session stopped, what's next, rebuild results and open questions. Read it at the start of every session, and update it at the end (see CLAUDE.md › Workflow). What the user has learned, and through what, is in `PROGRESS.md`.

## Where we are
- **Milestone:** M0 setup, in progress.
- **Done so far in M0:**
  - uv confirmed as the Python environment.
  - Claude wrote the M0 files: the docs (`LEARNING_LOG.md`, `PROGRESS.md`, `README.md`, `LICENSE`), the config (`.gitignore`, `.python-version`, `.env.example`, `pyproject.toml`) and the scaffold (`src/tutor/__init__.py`, `tests/test_smoke.py`, `.github/workflows/ci.yml`).
  - The user installed uv 0.12.23 with winget (M0 setup lesson "Installing uv", check questions done).
  - The user ran `uv sync`, which created `.venv` and `uv.lock` (M0 setup lesson "`uv sync`", check questions done). They already knew what a venv is.
  - The user ran `uv run pytest` and read a deliberate failure (M0 setup lesson "Running tests"). They skipped the check question, saying they already understood.
  - The user asked for tooling to be covered briefly from now on: what it does, how, why, and how it helps the project (see CLAUDE.md › Read first).
- **Next:** the user is running the ruff checks (the end of PLAN.md › M0 › "You run these", step 1) and the first commit of the docs on `main` (step 2). After that, step 3: creating the GitHub repo and pushing.

## Concepts done
None yet. The first concept is the M1 concept "Turning the PDF into prose with page numbers".

## Rebuild results
None yet. At the start of each session, the user rebuilds the core part they wrote in the previous session from memory, and the result goes here.

## Open questions
- Whether to plan a production stage (it would be `v2.0.0`). Decide after `v1.0.0`.
