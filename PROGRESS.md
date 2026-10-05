# Progress

What I've learned building the Study Tutor, and what I did to learn it, in plain words. Claude adds an entry after every concept (the M0 setup lessons count too). Newest entries are at the bottom.

## Before M0: planning

### Sorting out the project documents (2026-10-04)
**Did:** decided with Claude which file holds what.
**Learned:**
- `PLAN.md` says *what* to build and *why*.
- `CLAUDE.md` says *how* to work in the repo. Claude Code reads it automatically at the start of every session, so it holds the rules every session must follow.
- Write the plan first: CLAUDE.md's commands and rules depend on choices the plan makes, such as uv.

### Talking through versions and "production-grade" (2026-10-04)
**Did:** asked whether to build a rough MVP, then a full product, then a production version.
**Learned:**
- Git doesn't have stages. There's one repo with one history, and a version is just a label (a *tag*) on a commit.
- Version numbers carry the meaning: `0.x` means still changing, `1.0.0` means others can rely on it.
- Real teams build the thinnest version that works end to end, with decent code, and grow it. That's a *walking skeleton*, and here it's M1. "Rough now, rewrite later" rarely gets rewritten.
- Production quality goes where the AI is (checking the grader, checking quotes, storing prompt versions). Accounts, big databases and monitoring can wait.

## M0: setup

### Installing uv (2026-10-04)
**Did:** installed uv with `winget`.
**Learned:**
- uv is installed once for my user, outside any project, because it's the tool that creates each project's `.venv`. It has to exist before any venv does.
- ruff and pytest go *inside* the venv, pinned to exact versions, because a different version can give a different result: the code might pass on my laptop and fail on CI.

### `uv sync` and the lock file (2026-10-05)
**Did:** ran `uv sync` and `uv tree`, then worked through what happens when a teammate adds a package.
**Learned:**
- `pyproject.toml` lists what I chose, as ranges (`pytest>=8.0`). `uv.lock` lists every package at its exact version, and uv writes it from `pyproject.toml`. I always commit both together.
- `uv run <command>` first makes my `.venv` match `uv.lock`, then runs the command inside `.venv`. I don't need to activate anything.
- CI builds a brand-new `.venv` from `uv.lock` on GitHub's machine for every pull request, runs the tests, and then deletes it. My `.venv` stays on my laptop.
- `--locked` makes CI stop if `uv.lock` doesn't match `pyproject.toml`. That's how a forgotten `uv.lock` gets caught.

### Running tests (2026-10-05)
**Did:** ran `uv run pytest`, then broke the smoke test on purpose to see a failure.
**Learned:**
- pytest looks in `tests/`, runs every `test_*` function in every `test_*.py` file, and a failing `assert` makes the test fail.
- To read a failure: the `>` line is the line that failed, the `E` lines show what was expected and what the code actually gave, and `file::function` is the test's ID, which I can use to rerun just that one test.

### Commits and milestones (2026-10-05)
**Asked:** what to commit right now, and what the milestones are.
**Learned:**
- Commit only after the checks pass. The docs go straight onto `main`, since the first commit is what creates `main`. The code setup goes through a pull request, so CI checks it before it reaches `main`.
- The milestones (M0 to M6) aren't study material. They're stages of building the real tutor. Each one ends with a working tutor that does a bit more, and I learn the AI concepts by building them.

### ruff and `pyproject.toml` (2026-10-05)
**Asked:** what ruff checks, what it reads from `pyproject.toml`, and whether `pyproject.toml` is the same in most projects.
**Learned:**
- ruff scans the project's `.py` files and skips `.venv` and anything git ignores.
- Its rules live in `pyproject.toml` under `[tool.ruff]`: a maximum line length of 100, plus rule families that catch bugs (`F`, `B`), messy style (`E`, `W`), unsorted imports (`I`), outdated syntax (`UP`) and missing docstrings (`D`).
- Every `pyproject.toml` has the same *layout*, because it's a Python standard: `[project]`, `[build-system]` and `[tool.*]` sections. The *contents* differ, mostly the dependencies list and each team's tool settings.
