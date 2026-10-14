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

### Structured output (question writing) (2026-10-13)
**Did:** wrote the `Question` and `QuestionSet` models and saw the JSON Schema the LLM receives. Decided the quote check's rules with Claude, then ran the question writer on 10 real passages.
**Learned:**
- A Pydantic model becomes a JSON Schema that's sent with the call. Field descriptions and the class docstring are part of it, so the LLM reads them as instructions.
- Types in the schema are enforced; descriptions are only requests. Count rules like `min_length` aren't even sent (strict mode drops them), so "exactly 3" lives in two places: in words for the LLM, and in Pydantic to catch it when the LLM ignores them.
- `list[Question]`, not `list[str]` or `list[dict]`: a class I wrote can be a type, and that's what puts its fields into the schema.
- Rules belong in the prompt file, not in field descriptions, because only the prompt has a version saved with every question.
- The quote check normalizes both texts and then needs an exact match of whole words. A tidy-up step is only allowed if it can't change the meaning. A substring check alone would accept "supervised learning" inside "unsupervised learning".
- Key points must come only from the passage. An extra true point would mark me down for something the passage never taught, again at every review.
- Strict mode isn't a guarantee on gpt-oss: Groq can reject an answer that breaks the schema, so the wrapper retries once and then fails loudly.
- Small samples mislead. One experiment said reasoning "medium" fixed "according to the passage" (1 of 9 questions); the full run of 30 didn't confirm it.

### Grading grounded in the passage (2026-10-14)
**Did:** first asked Claude for the big picture of the project, because I couldn't see where anything lived. Then planned grading's two ends and its chain with Claude, and asked Claude to build it ("just build it"). Then went through the finished code line by line, renamed two things so the names make sense to me, and merged it.
**Learned:**
- The project is separate machines (loading a chapter, search, question writing, now grading) that share one database file. Nothing hands one machine's output to the next yet. Joining them is the study loop's job.
- Files in `src/tutor/` are tools: running one does nothing, because it only defines functions and classes. Files in `scripts/` are starters that call the tools. Only starters end with `if __name__ == "__main__":`.
- The grader doesn't care where my answer comes from. It's a plain string: today from a list in `scripts/regrade.py`, later from the terminal and then from a text box on the page. That's why the grader can be built before the text box exists.
- Test scripts work on a copy of the database, so test data never lands in my real question pools or attempts.
- An LLM writes one token at a time, and each token is chosen by reading all the text before it, including what it has already written. It has no other memory. So whatever comes first in the answer can shape what comes after, never the other way round. The `Grade` form puts the quotes first and the verdict last, so the verdict is decided with the evidence already written.
- A quote proves only one thing: that this exact text is in the passage. It catches the LLM making up book text, and it shows me the line so I can judge for myself. It doesn't prove the question or the grade really follows from that line.
- Key points only say what's missing. To tell what's wrong, or what's an extra, the grader needs the whole passage. For example, "the probability is the only way to show uncertainty" is wrong according to the passage, but no key point mentions uncertainty.
- The same-answer check is a plain text comparison after tidying (lowercase, spaces), done in SQL before any LLM call. It only finds answers that are already saved, so a second click while the first answer is still being graded would reach the LLM too. That's why the page will disable Submit while it grades (now in PLAN.md).
- A database cell holds one value, so the lists in a grade are stored as one piece of JSON text (`grade_details`), and the verdict gets its own column because later code searches by it. `model_dump_json()` turns the `Grade` into that text, and `model_validate_json()` turns it back into a `Grade`.
- A function hands back one thing. `GradedAnswer` is one object carrying four named values (attempt id, grade, supported, reused). A class is the blueprint; calling it builds one object; `return` hands back that object, and the caller reads `result.supported`. Names can't be mixed up the way positions in a tuple can.
- Naming matters: `Grade` is the whole form, and `verdict` is the one word in it (correct, partly correct or incorrect).
- `tests/test_grading.py` checks the code with a fake LLM, on every pull request. `scripts/regrade.py` checks the real LLM's judgement, and I run it by hand after changing the prompt.
- My own idea: after a break, "You choose" could give me nothing but overdue reviews, because reviews come before new passages. Many spaced-repetition apps cap reviews per day, so that's now a decision to make before M4 ("Daily review cap" in PLAN.md).
- `git add -A` stages every change at once. It's safe after `git status` shows exactly what "every" means, because `.gitignore` keeps `.env`, `data/` and `scratch/` out.
