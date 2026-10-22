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

## Who writes what
This overrides the `study-tutor-teaching` skill's "Claude writes most of the code" and PLAN.md's "Claude writes" column wherever they disagree.
- **At the start of each concept,** list its pieces and mark each one "user" or "Claude". The user can move any piece.
- **AI, ML and RAG logic is the user's, and so is the wiring that connects it:** chunking, embeddings and search, prompts and schemas, the quote check, grading, the graph's state and routing, tool calls, FSRS updates, mistake patterns, plus the code that calls the pieces in order (such as the steps inside `search()`). The user builds single functions easily but loses track of how they fit together, and writing the wiring is how that's learned.
- **Build each concept chain first, then fill the links:**
  1. **Ends first.** Agree on what goes in and what must come out, with a real example: `"gradient descent"` → 3 passages.
  2. **The chain.** The user writes the middle as a list of data shapes, working backwards from the end and forwards from the start, with one function per arrow. Claude corrects it, helps when the user is stuck, and names a library or function only where a link needs one, saying what it does and why.
  3. **A skeleton for the whole chain at once:** every link as an empty function (signature, docstring, one comment per step), plus the function that wires them. Keep giving skeletons: the user decided that starting from the goal alone takes too long.
  4. **Fill the links.** The user writes the AI/ML links and the wiring, and Claude writes the plumbing links. Claude reviews the user's code with hints instead of fixes.
  5. **Run one real example through,** printing what each link hands to the next, so a wrong link shows where the data stops looking right.
- **Questions go at the chain, not at every step.** The chain answers "what's the next move?" for every step at once. Ask only where a link isn't obvious from the chain, or what would go wrong if we did X. Use a quick run only when reasoning can't settle a question.
- **After each piece,** show where it sits in the chain: what feeds it and what it feeds.
- **Plumbing is Claude's:** database tables and SQL, command-line arguments and printing, config, the web layer and tests. Claude writes it in small pieces too, with a few lines on what each does.
- **A concept that's mostly plumbing,** such as the M1 concept "Turning the PDF into prose with page numbers": Claude writes all of it and gives its RAG takeaway in one line, with no explain-back.

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
| Ingest a chapter (saves its topics and passages to `data/tutor.db` and embeds the passages) | `uv run python scripts/ingest.py --chapter 1` |
| Embed every saved passage again, after changing the header or the embedding model | `uv run python scripts/ingest.py --embed` |
| Check search on chapter 1 (test queries, off-topic ones, the scores for the cut-off) | `uv run python scripts/check_retrieval.py` |
| Check the question writer on 10 passages, on a copy of the database (the quote-check pass rate; a made-up quote and one-word changes rejected). Makes 10 real LLM calls | `uv run python scripts/check_questions.py` |
| Re-grade the five test answers on a copy of the database, after any change to `prompts/grader.md`, the `Grade` form or the grading model (each grade against the one it should get; a repeated answer reused, a one-word change graded fresh). Makes 6 real LLM calls | `uv run python scripts/regrade.py` |
| Study in the terminal, on the real database: one session of turns, starting with the next new passage, or `--topic "gradient descent"` for the closest one. After each turn, type a topic, press Enter for "You choose", or type `s` to stop. Each turn usually makes 2 real LLM calls (writing a pool of questions, grading your answer); a passage whose questions fail adds up to 2 more before it's skipped | `uv run python scripts/study.py` |
| Print the passages an ingest would save, saving nothing | `uv run python scripts/ingest.py --chapter 1 --show` |
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
- **Database:** SQLite through the built-in `sqlite3` with plain SQL, and no ORM. SQLite is the source of truth, embeddings included. Search is a NumPy cosine over the stored vectors; Chroma, if the optional extra after v1.0.0 is built, is a derived index that must stay rebuildable from SQLite. (Tech; Decisions › Vector store)
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

## Code tree
The Study Tutor Code Tree is a plain-words picture of the code: every file, function and class under `src/tutor/`, `prompts/` and `scripts/`, plus the call chains that connect them. The user reads it to understand the code and to explain it to the next session. Link: https://claude.ai/artifact/3rxaNUDZxmVoZemwPdpfog
- **Where it comes from:** `docs/code-tree.html`, one self-contained page. Its content is the two lists at the top of its script: `CHAINS` (each chain's steps, in call order, with what each step hands to the next) and `TREE` (folder → file → items). Change only those lists, the eyebrow and the footer. Never restyle the page or change its rendering code, so every update looks the same.
- **The style of an entry:**
  - Names and arguments are copied exactly from the code. Long argument lists end in `…`.
  - Each description is one plain sentence, written from the docstring.
  - `kind` is `"fn"` for a function, `"shape"` for a class (methods go in `subs`), or `"helper"` for a name starting with `_`.
  - Tests stay out.

Keep it current without being asked:
- **A branch adds, removes or renames a file, function or class, or changes what one does or calls:** update `docs/code-tree.html` in that same branch, so the tree changes in the same pull request as the code. A concept that creates a new chain of calls (grading, the study loop) gets a new entry in `CHAINS`. Check each chain against the real calls in the code. Set the date in the eyebrow and footer to today.
- **After that pull request is merged:** republish `docs/code-tree.html` to the same link with the Artifact tool. Pass the link as `url` (read the artifact first), and leave out `icon`.

## Project documents
- Each fact lives in exactly one file:
  - `PLAN.md`: what to build and why.
  - `CLAUDE.md`: how to work.
  - `LEARNING_LOG.md`: the session handoff, meaning where we are, what's next, rebuild results and open questions.
  - `PROGRESS.md`: what the user has learned and through what, in plain words. It's written for the user to read.
  - `docs/project-map.json`: the content of the Study Tutor Map, a view of PLAN.md and the repo (see Project map).
  - `docs/code-tree.html`: the Study Tutor Code Tree, a view of the code (see Code tree). When it and the code disagree, the code wins and the tree gets fixed.
  - Neither kind of note ever goes in PLAN.md.
- Write these files so that a fresh session understands them on their own:
  - Refer to things by name ("the M1 concept 'Chunking'"), never by a bare number.
  - Never describe a change relative to an earlier version the reader hasn't seen.
