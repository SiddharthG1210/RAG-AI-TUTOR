# Progress

What I've learned building the Study Tutor, and what I did to learn it, in plain words. Claude adds an entry when I ask, usually after a concept (the M0 setup lessons count too). Newest entries are at the bottom.

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
- The milestones (M0 to M7) aren't study material. They're stages of building the real tutor. Each one ends with a working tutor that does a bit more, and I learn the AI concepts by building them.

### ruff and `pyproject.toml` (2026-10-05)
**Asked:** what ruff checks, what it reads from `pyproject.toml`, and whether `pyproject.toml` is the same in most projects.
**Learned:**
- ruff scans the project's `.py` files and skips `.venv` and anything git ignores.
- Its rules live in `pyproject.toml` under `[tool.ruff]`: a maximum line length of 100, plus rule families that catch bugs (`F`, `B`), messy style (`E`, `W`), unsorted imports (`I`), outdated syntax (`UP`) and missing docstrings (`D`).
- Every `pyproject.toml` has the same *layout*, because it's a Python standard: `[project]`, `[build-system]` and `[tool.*]` sections. The *contents* differ, mostly the dependencies list and each team's tool settings.

### Reviewing the plan (2026-10-05)
**Did:** went through a review of PLAN.md with Claude and agreed the changes.
**Learned:**
- An *agent* is an LLM with a tool (like "search the book") that decides by itself when to use it and what to search for. The plan had none, so I added M2, "Ask the book", and cut loading chapters from the web page (the terminal command does the same job) to keep the plan the same size.
- The embedding model learned language during its training and learns nothing from my book. It turns each passage into a vector by reading only that passage, so loading the whole book at once wouldn't change any vector.
- Chapters get added one at a time because later chapters are full of code and maths, and the PDF reader has to be checked on them first.
- Building a rough version of the whole loop first (the M1 concept "A rough first version in one file") gives every later concept a clear job: replace one rough piece with a real one.

### My first pull request (2026-10-05)
**Did:** opened and merged a pull request, added a rule protecting `main`, created `.env` and downloaded the PDF.
**Learned:**
- A pull request asks GitHub to add a branch's changes to `main`. Opening it starts CI on a fresh GitHub machine, and the ✓ or ✗ shows on the PR page. GitHub doesn't require a description; my CLAUDE.md asks for one so I can understand each change later.
- "Fast-forward" from `git pull` means my `main` just moved forward to match GitHub's.
- The rule on `main` blocks merging until ruff + pytest pass. It can only be added after the check has run once.
- `.env.example` is the empty template that git tracks. The real key goes in `.env`, which git ignores. `git restore <file>` undoes my changes to a tracked file.
- `data/` is ignored too: big files and files that can be regenerated don't belong in the repo.

## M1: one question, graded, with the passage shown

### A rough first version in one file (2026-10-05)
**Did:** made one test call to Groq, then wrote `main()` in `scratch/first_loop.py`. Claude wrote the two LLM calls: `write_question()` and `grade_answer()`. My `main()` shows a question about a pasted paragraph from section 1.3.1.2 "Classification", reads my answer with `input()`, and prints the grade and the passage. Then I answered three check questions.
**Learned:**
- `uv run --with openai --env-file .env <file>` adds `openai` for that one run only and loads `.env` into environment variables. `pyproject.toml` and `uv.lock` stay unchanged. VS Code underlines the import in red because `.venv` doesn't have `openai`. That's harmless.
- The gpt-oss models are reasoning models: they "think" before answering, and those hidden tokens count toward Groq's free limits. `reasoning_effort` controls how much they think. Every call also carries a built-in template: a 9-word question cost 80 prompt tokens.
- Every LLM call starts blank. The grading call has never seen the question-writing call, so the passage goes into *every* call. Grading against the passage, not the model's own knowledge, is what "grounded" means here.
- The grader needs three inputs: the passage, the question and my answer. Without the question, it can't grade "only what the question asks".
- Printing the passage after the grade is for *me*, so I can check the grade. Sending it to the grader is for the *LLM*, so it grades against the book. They're two different jobs.
- My `main()` is the glue: it passes values from one call to the next. In the concept "The study loop: state, nodes, edges", those values become the graph's state and each call becomes a node. Re-asking on an empty answer means an empty answer never costs a Groq call.
- The LLM wrote a question its passage can't answer. It asked *how* probabilities make optimizing easier, but the passage only says they do and leaves the reason for later chapters. Nothing in the rough file catches that. The quote check and the "not answerable from the passage" flag will.
- A free-text grade can't be trusted by code. It might come back as "Correct", "**Grade:**" in Markdown, or "mostly correct". The `Grade` schema with strict structured output forces one of three exact values.
