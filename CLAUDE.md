# CLAUDE.md

How to work in this repo. Claude Code loads this file into every session.

**The project:** Study Tutor is an adaptive tutor. It writes questions from study material, grades typed answers against the passage each question came from, schedules reviews with FSRS, and tracks repeated mistakes. The first material is chapter 1 of *Dive into Deep Learning* (d2l.ai). The user is building it to learn RAG and agentic AI, so the user's understanding matters more than speed.

## Read first
- **`LEARNING_LOG.md`**, at the start of every session: where the last session stopped and what comes next. M0 creates it. Until then, start from the M0 section of PLAN.md.
- **`PLAN.md`**, for what to build and why: the scope, decisions, design rules, and the roadmap from M0 to M7. Before writing code for a concept, read two things in PLAN.md:
  - the concept's row in the roadmap (its branch, what the user writes, what Claude writes),
  - its milestone's "Done when" list.
- **The `study-tutor-teaching` skill:** invoke it at the start of every session that builds or explains a concept. It sets the teaching loop, the pacing, and which part of the code the user writes.
- **Tooling isn't taught in depth.** For anything that isn't an AI/ML concept (git, uv, CI, pytest, ruff, web and database plumbing), say briefly what it does, how it does it, why it's needed, and how it helps build the project. Skip step-by-step breakdowns and check questions; those are for the AI/ML concepts in PLAN.md's roadmap.

## Workflow
- One or two concepts per session.
- Update `PROGRESS.md` and `LEARNING_LOG.md` only when the user asks, and in the way they ask. Never add entries on your own initiative. This overrides the end-of-session update in the `study-tutor-teaching` skill.
- Each concept gets its own branch, named in its PLAN.md row. For example, the M1 concept "Chunking" is built on `feat/chunking`.
- `main` always works. A concept reaches `main` only through a pull request whose checks pass, with a description of what changed and why.
- Each milestone ends with a version tag and a GitHub Release. PLAN.md › "Finishing a milestone" says when.
- Decisions marked **Proposed** in PLAN.md › Decisions aren't final. Confirm each one with the user before the milestone that needs it, then update its row.

## Git
- The user runs every git command. Claude says when to commit, branch, open a pull request or merge, and explains any command the user hasn't used before. Claude never commits or pushes.
- Commit small, working steps. Messages follow Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Never committed: `.env`, the study material, generated data (`tutor.db`, `checkpoints.db`, `chroma/`), and large files. `.gitignore` lists them.

## Commands
These exist from M0 onward and use uv (PLAN.md › Decisions › Python environment).

| What | Command |
|---|---|
| Install or update dependencies | `uv sync` |
| Run the tests | `uv run pytest` |
| Check the lint rules | `uv run ruff check .` |
| Format the code | `uv run ruff format .` |
| Add a runtime package | `uv add <package>`, in the pull request of the concept that first needs it |
| Ingest a chapter (saves its topics to `data/tutor.db`) | `uv run python scripts/ingest.py --chapter 1` |
| Print the text an ingest would keep, saving nothing | `uv run python scripts/ingest.py --chapter 1 --show` |
| List a chapter's fonts, to decide `classify_block()`'s rules | `uv run python scripts/ingest.py --chapter 1 --fonts` |

Add each new command here (ingesting material, running the app, the retrieval check) in the same pull request that creates it.

## Coding style
- **Formatting and linting:** ruff, with its settings in `pyproject.toml`. Run `ruff format` and `ruff check` before each commit. CI runs both and fails the pull request if either fails.
- **Docstrings and comments:** a docstring on every module, class and function, plus comments explaining the library, algorithm or technique each piece of code uses. The user learns from this code, so err on the side of over-explaining.
- **Type hints** on every function signature.
- **Pydantic models** for every structured LLM output (such as `QuestionSet` and `Grade`) and for config (`pydantic-settings`, reading `.env`).
- **Fail loudly.** Raise a clear error instead of returning an empty or default value. The only fallbacks are the ones PLAN.md defines: "I couldn't find support for this grade", "I couldn't find support for this answer", "free limit reached" and "the material doesn't cover this".
- **Where files go:** see PLAN.md › "Repo layout".
- **Tests:**
  - One test file per module, so `src/tutor/grading.py` is tested in `tests/test_grading.py`.
  - Tests cover the deterministic part of each function.
  - Tests never call the real LLM. They use a fake in place of the wrapper in `src/tutor/llm.py`, because CI has no API key.

## Guardrails
The reason for each rule is in the PLAN.md section named in brackets.
- **The LLM does only three jobs:** writing questions, grading typed answers and answering Ask the book questions. Ask the book is the only agent: the LLM picks its searches, and code runs them, caps them at 3 and checks every citation. Plain Python does everything else: picking passages, routing between graph nodes, checking quotes and reusing grades. (Design rules)
- **`scratch/` is exempt from these guardrails.** It's ignored by git and holds throwaway code, such as the M1 concept "A rough first version in one file". (Roadmap › M1)
- **LLM calls go only through `src/tutor/llm.py`,** which uses the `openai` SDK pointed at Groq's OpenAI-compatible URL. No LangChain chat models. (Tech)
- **Database:** SQLite through the built-in `sqlite3` with plain SQL, and no ORM. SQLite is the source of truth. Chroma is a derived index that must stay rebuildable from it. (Tech; Decisions › Vector store)
- **Prompts live in `prompts/*.md`,** never as strings in code. The prompt file's hash is stored with every question and grade. (How the tutor behaves in specific cases › Prompt versions stored)
- **Compare characters wherever PLAN.md says characters.** The quote check and the same-answer check never use embedding similarity. (Design rules › Three kinds of matching)
- **Free tools only.** Ask before adding anything that needs a paid plan, and keep LLM usage inside Groq's free-tier limits. (Decisions › LLM)
- **License:** the repo is AGPL-3.0 because of PyMuPDF. Check that each new dependency's license is compatible; MIT, BSD and Apache-2.0 are. (Decisions › PDF library)

## Project map
The Study Tutor Map is an interactive picture of this plan: its concepts, files, tools and study turn, with a tick for every finished concept. Link: https://claude.ai/artifact/8gWtiwGskoX9DaF8nbYtNx
- **Where it comes from:** its content is `docs/project-map.json`. The personal `project-map` skill (`~/.claude/skills/project-map`) builds the page from that file and publishes it; its SKILL.md documents every field. The map is a view of PLAN.md and the repo: when they disagree, PLAN.md and the code win, and the map gets fixed.

Keep the map current without being asked:
- **A concept's pull request is merged:** tick that concept on the map. With the ArtifactData tool, `set` the document `progress/<concept id>` to `{"done": true, "on": "<YYYY-MM-DD>"}`. At the start of each session, also tick any merged concept the map hasn't caught up with.
- **The architecture changes:** when a branch adds, removes or renames a file, folder, tool or dependency, or changes a concept, a milestone or the study turn, update `docs/project-map.json` in that same branch, so the map changes in the same pull request as the code. Every change to PLAN.md gets the same treatment.
- **After such a pull request is merged:** rebuild the page and republish it to the same link with the project-map skill, leaving out `icon` and `capabilities` so the ticks and settings stay.
- **Concept ids never change:** ticks are stored under them. To rename a concept, change its `name`.

## Project documents
- Each fact lives in exactly one file:
  - `PLAN.md`: what to build and why.
  - `CLAUDE.md`: how to work.
  - `LEARNING_LOG.md`: the session handoff, meaning where we are, what's next, rebuild results and open questions.
  - `PROGRESS.md`: what the user has learned and through what, in plain words. It's written for the user to read.
  - `docs/project-map.json`: the content of the Study Tutor Map, a view of PLAN.md and the repo (see Project map).
  - Neither kind of note ever goes in PLAN.md.
- Write these files so that a fresh session understands them on their own:
  - Refer to things by name ("the M1 concept 'Chunking'"), never by a bare number.
  - Never describe a change relative to an earlier version the reader hasn't seen.
