# Study Tutor: project plan (PLAN.md)

## Context
An adaptive study tutor. You give it study material and it quizzes you on it:
- it writes questions from the material,
- grades your answers against the source,
- schedules reviews so you revisit things before you forget them,
- remembers what you keep getting wrong.

You're the first user, and you're building it to learn RAG and agentic AI. Every tool must be free.

The first material is *Dive into Deep Learning* (the d2l.ai PDF, CC BY-SA 4.0, credit to Zhang, Lipton, Li & Smola), starting with chapter 1, which is mostly prose. The repo `D:\projects\RAG-AI-tutor` is initialized on `main`, with no commits yet.

Machine: Windows 11, Python 3.12.6, Git 2.53 (Git Credential Manager set up), a GTX 1650 with 4 GB of VRAM, 15 GB of RAM, and winget. Not installed: uv, the gh CLI, Ollama, Docker.

This file is the complete build plan, from setup (M0) to the shareable v1 (M6): what to build and why. How to work in the repo (workflow, git, coding style) is in `CLAUDE.md`. The session handoff is in `LEARNING_LOG.md`, and what you've learned is in `PROGRESS.md`.

## Scope
**Rules the tutor must follow:**
- Every question must be answerable from a single passage of the material, and that passage is stored with the question.
- Every grade shows the passage it was based on, with its page and section number.
- The tutor uses only the material. It never adds facts from outside it.
- If it can't find support in the material for a question or a grade, it says so instead of guessing.
- One question at a time. It waits for your answer before continuing.
- The same answer to the same question gets the same grade.
- It only asks about ideas explained in prose: never about formulas, code or figures.

**Not in the first version:** accounts or multiple users, more than one document at a time, images, diagrams and tables in the material, and voice.

## Decisions
| Question | Choice | Notes |
|---|---|---|
| Topic | The smallest numbered heading above a passage, read from the PDF's bookmarks | About 20 topics in chapter 1. The Summary and Exercises sections are skipped |
| Passage | 120–350 words, never crossing a topic boundary | Stores its topic and page range |
| Reviews | A new question from the same passage, taken from a pool of 3 | The scheduled item (the FSRS card) is the passage. One LLM call writes 3 questions the first time a passage is studied, and the pool is refilled when it runs out. A session asks only one question per passage. Questions are never written in advance for the whole book |
| Extra facts in an answer | Grade only what the question asks | Extras the passage doesn't cover get a note and no penalty. Extras that contradict it count as errors |
| "Grade is wrong" | You pick the right grade | Your grade is used for scheduling and for repeated answers. The grading report counts it as a disagreement |
| Answer type | Typed and multiple choice | MC arrives in M2. Each session: typed, MC or mixed (about every third question is MC). MC is graded in code, with no LLM call |
| Weak topic | Low recent score | At least 3 answers, and the average of the last 5 is below 0.6 (correct = 1, partly = 0.5, incorrect = 0; MC: 1 or 0) |
| LLM | Groq free tier | `openai/gpt-oss-20b` writes questions and `openai/gpt-oss-120b` grades. Each model is limited to 30 requests/min, 1K requests/day, 8K tokens/min and 200K tokens/day (roughly 100 typed answers a day). Re-check them at console.groq.com/settings/limits |
| PDF library | PyMuPDF | It's AGPL-3.0, so the repo is AGPL-3.0 and its source must be public once the app is hosted |
| Embeddings | **Proposed:** `BAAI/bge-small-en-v1.5` through sentence-transformers, local, on the CPU | Confirm before M1 "Embeddings and vector search" |
| Vector store | **Proposed:** Chroma, saved to a folder; a derived index that can be rebuilt from SQLite | Confirm before M1 "Embeddings and vector search" |
| UI | **Proposed:** Gradio, mounted inside FastAPI | Confirm before M1 "API and UI" |
| Python environment | uv | Installs the packages and keeps a lockfile (`uv.lock`), so this machine and CI install exactly the same versions |
| FSRS ratings | **Proposed:** typed: correct → Good, partly → Hard, incorrect → Again. MC: correct → Hard, wrong → Again | MC counts for less, because recognising an answer is easier than recalling it. Confirm before M3 |
| Mix-up follow-ups | **Proposed:** one single-passage question per topic of the pair, each about what makes that topic different | This keeps the single-passage rule. Confirm before M4 |
| Sharing (M6) | **Proposed:** a live demo on Hugging Face Spaces (free CPU) in demo mode, with a demo video as the backup | Confirm before M6 |

## Tech
- **Decided:** Python, FastAPI, LangGraph, SQLite, the `fsrs` library, Git and GitHub, PyMuPDF, the Groq free tier, uv, ruff (linting and formatting).
- **Proposed** (see Decisions): bge-small-en-v1.5 through sentence-transformers, Chroma, Gradio.
- **Implementation choices:**
  - LLM calls go through a thin wrapper around the `openai` SDK, pointed at Groq's OpenAI-compatible URL, so switching provider only means editing `.env`. Most LangGraph tutorials use LangChain chat models instead. Each of our calls is a single structured-output request, though, so a thin wrapper is enough and keeps the mechanism visible.
  - SQLite through Python's built-in `sqlite3` with plain SQL. A production app would usually add SQLAlchemy with migrations, which is overkill for one user.
  - Config from `.env` through `pydantic-settings`.
  - GitHub Actions runs ruff and pytest on every pull request. It's free for public repos, and a pull request can't be merged until the tests pass, which keeps `main` working.

## How the tutor behaves in specific cases
- **Flag reasons:** not answerable from the passage, ambiguous, incorrect, and messy formula/code. The last reason feeds the check on whether to switch to the book's Markdown files (see *Adding more chapters*).
- **Citations** show the section number as well as the page, so they still work if the source switches to Markdown, which has no pages.
- **Same answer, same grade:** before grading, the tutor looks up earlier answers to the same question.
  - The match must be exact after tidying (lowercase, extra spaces removed). It's never a similarity match, because one word ("is" → "isn't") can flip the meaning.
  - A hit reuses the earlier grade, and your corrected grade wins.
  - The lookup also includes the grading prompt's version.
  - This also stops double-submits from costing a second call.
- **Quotes checked in code:** for every question and every grade, the LLM returns a quote from the passage, and code checks that it's really there.
  - The comparison is about 90% of characters after normalizing hyphens and whitespace. It never compares meaning, because a made-up line that only sounds similar must fail.
  - If none of a passage's 3 questions pass, the LLM writes them once more, and a second failure skips the passage.
  - If a grade's quotes fail, the tutor says "I couldn't find support for this grade".
- **Prompt versions stored:** a hash of the prompt file is saved with every question and grade, so a change in behavior traces back to the commit that changed the prompt.
- **Free-tier guard:**
  - Tokens are logged per call and per model per day.
  - On HTTP 429 (rate limit reached), the tutor waits and retries, then shows "free limit reached" instead of crashing.
  - Questions are written only for passages you actually reach.
- **Pausing for your answer:** the study loop is split across web requests.
  - "Start" runs the graph until it pauses, saves the state and returns the question.
  - "Answer" loads the state and resumes from the same spot.
  - LangGraph's `interrupt` needs a checkpointer: an in-memory one in M1, SQLite from M5, so sessions survive a restart.
  - Each study session has its own id (LangGraph's thread id), which the page sends with every request.
- **Topic search:**
  - A typed topic is embedded and matched against the passages by meaning.
  - Picking a topic from the list uses plain SQL.
  - If even the closest passage is too far away, the tutor says "the material doesn't cover this". The cut-off is tuned in M1 "Embeddings and vector search" with off-topic test queries.
- **What "You choose" picks,** in this order:
  1. due reviews, most overdue first (from M3)
  2. follow-ups on active mix-ups (from M4)
  3. weak topics (from M4)
  4. the next new passage, in book order (from M1)

## Design rules
- **Code is in charge, and the LLM does only two jobs:** writing questions and grading typed answers. Plain code picks passages, decides the next step, checks quotes, reuses grades and chooses what to show. The routing functions are plain Python reading the graph's state. Many agent systems let an LLM choose the next step, but these rules are clear and code costs no tokens.
- **Every LLM call starts blank.** The passage is sent with every call. That reduces the use of outside knowledge but can't prevent it, which is why quotes are checked in code.
- **Three kinds of matching:**

| Where | What's compared | How strict |
|---|---|---|
| Finding passages for a typed topic, and spotting mix-ups (M4) | Meaning (embeddings) | The closest few, plus a distance cut-off |
| Checking the LLM's quotes | Characters | About 90% |
| Reusing the grade for a repeated answer | Characters, after tidying | 100% (exact) |

## How the tutor works (diagrams)
A `*` marks a proposed tool. The steps inside each diagram are lettered.

### Diagram 1: the parts, and which part calls which
```
  You, in a browser
     │  click / type
     ▼
  Gradio page* ........ the screen you use
     │
     ▼
  FastAPI ............. web server: passes your clicks to the loop
     │
     ▼
  LangGraph ........... the study loop: runs the steps in order
     │                  and pauses while it waits for you
     │
     ├──► Groq LLM ........... writes questions, grades typed answers
     ├──► bge-small* ......... turns a typed topic into an embedding
     │      └──► Chroma* ..... finds the passages closest to it
     ├──► fsrs ............... decides when each passage comes back
     └──► SQLite ............. stores everything: passages, questions,
                               answers, grades, marks, the schedule

  Not part of the running app:
  uv installs the Python packages · ruff checks the code style
  Git + GitHub keep the history
```

### Diagram 2: adding material (once per PDF, before any studying)
```
  d2l-en.pdf
     │
     ▼
  a. Read every page: text, fonts, page numbers ...... PyMuPDF
     │
     ▼
  b. Keep only prose; drop code, math, figure ........ Python (Claude writes)
     captions, page headers
     │
     ▼
  c. Group into topics using the PDF's bookmarks ..... PyMuPDF
     (1.2.1 Data, 1.3.1.2 Classification, ...)
     skip the Summary and Exercises sections
     │
     ▼
  d. Cut each topic into passages of ~120-350 ........ Python (you write)
     words, each with its page numbers
     │
     ├──► e. Save topics + passages .................. SQLite
     │
     └──► f. Turn each passage into an embedding ..... bge-small*
             and store it ............................ Chroma*

  No LLM runs here, so adding material uses none of the free quota.
  SQLite is the source of truth; Chroma can always be rebuilt from it.
```

### Diagram 3: a study session (WHEN = the milestone that adds each part)
```
                                                    TECH             WHEN
  a. Pick the answer type: typed / MC / mixed ..... Gradio*          M2
     Type a topic, or click "You choose" .......... Gradio*          M1
     │
     ▼
  b. Pick a passage
     • typed topic -> the most similar passages ... bge* + Chroma*   M1
       (nothing close enough -> "the material doesn't cover this")
     • "You choose" -> due reviews first .......... fsrs + SQLite    M3
                    -> mix-up follow-ups, weak .... SQLite           M4
                       topics
                    -> else the next new passage .. SQLite           M1
     │
     ▼
  c. Get a question about that passage
     • next unused one from its pool of 3 ......... SQLite           M1
     • pool empty? the LLM writes 3 more .......... Groq LLM         M1
     • code checks each question's quote is ....... Python           M1
       really in the passage, else drops it
       (none pass -> write once more, then skip the passage)
     │
     ▼
  d. PAUSE until you answer ....................... LangGraph        M1
     │
     ▼
  e. Grade the answer
     • same answer seen before? reuse its grade ... SQLite           M1
     • typed -> LLM grades it against the passage . Groq LLM         M1
     • multiple choice -> code checks your pick ... Python           M2
     • tag the mistake, spot mix-ups .............. LLM + Chroma*    M4
     │
     ▼
  f. Show the grade, what's missing or wrong, ..... Gradio*          M1
     and the passage with its page number
     │
     ▼
  g. You may click "Grade is wrong" or "Flag" ..... SQLite           M2
     │
     ▼
  h. Save the result .............................. SQLite           M1
     Schedule when this passage comes back ........ fsrs             M3
     │
     ▼
  i. PAUSE: "Next" or "Stop" ...................... LangGraph        M1
     • Next -> active mix-up? ask a follow-up ..... LangGraph        M4
               else back to b
     • Stop -> pick up later where you left off ... LangGraph+SQLite M5
```

### Diagram 4: which milestone needs which
```
  M1  Ask a question, grade it, show the passage
   │    needs: material added, the LLM, the study loop
   ▼
  M2  "Grade is wrong", flags, grading report, multiple choice
   │    needs M1: there have to be grades to mark
   ▼
  M3  Reviews come back when due
   │    needs grades (M1) and your corrections (M2)
   ▼
  M4  Weak topics + follow-up questions on repeated mistakes
   │    needs a history of graded answers (M1-M3)
   ▼
  M5  Pause/resume + progress page + adding material in the app
   │    the progress page shows M3's due reviews and M4's weak topics
   ▼
  M6  Someone else can try it: live link or demo video
        needs everything above working
```

## Data (SQLite tables, by the milestone that adds them)
| Milestone | Tables |
|---|---|
| M1 | **topics:** number, title, level, page. **passages:** topic, text, page range, order. **questions:** passage, text, key points, quote, prompt version, used or unused. **attempts:** question, session, your answer, tidied answer, grade, explanation, prompt version. **llm_calls:** model, tokens, time |
| M2 | **marks:** attempt, agree or disagree, your grade. **flags:** question, reason, note. **questions** gains: type (typed/MC), options, correct option |
| M3 | **cards:** passage, FSRS card (JSON), due date. **review_log:** passage, rating, date, attempt |
| M4 | **mistakes:** attempt, mistake type, wrong statement, mixed-up topic. **patterns:** topic pair (or topic plus mistake type), active or resolved |
| M5 | **sessions:** id, started, last active, open or closed. LangGraph keeps its own checkpoint tables in a separate `checkpoints.db` |

Chroma (from M1) has one collection, with one entry per passage. The entry's id is the passage id, and its labels are the topic and the pages.

## Repo layout (at v1.0.0)
```
CLAUDE.md          how to work here: workflow, git, coding style (loaded every session)
PLAN.md            this build plan
LEARNING_LOG.md    where we are, what's next
PROGRESS.md        what you've learned, and through what
README.md          what it is, how to run it, demo link, credit to the d2l authors (CC BY-SA 4.0)
LICENSE            AGPL-3.0
pyproject.toml, uv.lock, .python-version, .env.example, .gitignore
.github/workflows/ci.yml      ruff + pytest on every pull request
Dockerfile                    (M6)
prompts/           question_writer.md, grader.md (M1) · mc_writer.md (M2)
src/tutor/
  config.py, db.py, llm.py
  ingest/          pdf.py, chunk.py (M1) · markdown.py (only if the source switches)
  retrieval/       embed.py, store.py, search.py (M1)
  questions.py, grading.py (M1) · mc.py, feedback.py, report.py (M2)
  scheduling.py (M3) · mistakes.py, patterns.py (M4)
  graph/           state.py, nodes.py, build.py (M1; grows each milestone)
  api/             main.py, ui.py (M1) · progress and add-material pages (M5) · demo.py (M6)
scripts/           ingest.py, check_retrieval.py (M1) · regrade.py (M2, optional)
tests/             one test file per core function
data/              (ignored by git) the PDF, tutor.db, checkpoints.db, chroma/
```

## Roadmap

### How every milestone is built
- **Concepts:** each concept goes through the study-tutor-teaching loop: why it's needed, a tiny example, you write the core from a `TODO(human)` skeleton, a review, then you explain it back. Where the table says "Claude writes it all", it's: why, Claude writes, you explain back.
- **Finishing a milestone:** after the last merge, check its "Done when" list, then set `version` in `pyproject.toml` to the milestone's version, tag it (`v0.1.0` … `v0.5.0`, `v1.0.0`) and publish a GitHub Release with short notes.

### M0: setup (branch `chore/project-setup`)
- **Already written:** `CLAUDE.md` and `PLAN.md`.
- **Claude writes:**
  - The docs: `LEARNING_LOG.md`, `PROGRESS.md`, `README.md`, `LICENSE`.
  - The config: `.gitignore`, `.python-version` (pins Python 3.12, since this machine also has 3.9, 3.13 and 3.14), `.env.example`, and `pyproject.toml` (the dev tools pytest and ruff, plus ruff's lint and format settings). Each runtime package is added with `uv add` by the concept that first needs it.
  - The scaffold: `src/tutor/__init__.py`, `tests/test_smoke.py`, `.github/workflows/ci.yml` (ruff and pytest on every pull request).
- **You run these.** Each new command gets explained first.
  1. `winget install astral-sh.uv`, `uv sync`, `uv run pytest`, `uv run ruff check .`.
  2. The first commit (the docs: CLAUDE.md, PLAN.md, LEARNING_LOG.md, PROGRESS.md, README.md, LICENSE, .gitignore) on `main`.
  3. Create the repo on the GitHub website (public), then `git remote add origin …` and `git push -u origin main`.
  4. `git switch -c chore/project-setup`, commit the scaffold, push, open a pull request, and watch the test check pass.
  5. Merge it, then `git switch main` and `git pull`.
  6. In the GitHub repo's settings, add a rule for `main` that requires the `ruff + pytest` check to pass before a pull request can merge. The check only shows up as an option after it has run once, which is why this comes last.
- **Also you:** download https://d2l.ai/d2l-en.pdf into `data/`. Create a free Groq API key and put it in `.env` (first needed in M1 "Structured output").
- **Done when:**
  - `uv run pytest` passes.
  - `uv run ruff check .` and `uv run ruff format --check .` pass.
  - `git check-ignore -v data/d2l-en.pdf .env` shows both are ignored.
  - The PR is merged with a passing check.

### M1 (v0.1.0): one question, graded, with the passage shown
| # | Concept | Branch | You write | Claude writes |
|---|---|---|---|---|
| 1 | Turning the PDF into prose with page numbers | feat/pdf-extract | Claude writes it all | PDF reading, `classify_block()` (prose / code / math / caption / header, using the fonts found in chapter 1), bookmarks to topics, the topics table, the ingest command |
| 2 | Chunking | feat/chunking | `chunk_paragraphs()` | Joining paragraphs split across pages, the passages table, tests |
| 3 | Embeddings and vector search | feat/retrieval | `search(query, k, topic=None)`. Tiny example: cosine similarity in NumPy | Model loading. Embeddings computed by us, not by Chroma behind the scenes. The Chroma index and its rebuild. The distance cut-off. A retrieval check with 10 test queries plus off-topic ones |
| 4 | Structured output (question writing) | feat/question-gen | The question-writing prompt and the `QuestionSet` schema (3 questions, each with key points and a quote) | The LLM wrapper (Groq strict mode), the quote check, the question pool, the token log, 429 handling |
| 5 | Grading grounded in the passage | feat/grading | The grading prompt and the `Grade` schema (grade, what's missing, what's wrong, unchecked extras, quotes) | The same-answer check, the attempts table, the "no support found" path |
| 6 | The study loop: state, nodes, edges | feat/study-graph | The state and the `build_graph()` wiring | Node functions that call the code from concepts 1–5 |
| 7 | Conditional edges | feat/grounding-retry | The routing function: ask, write again (once), or skip the passage | Retry counters, tests |
| 8 | Pausing for your answer | feat/answer-interrupt | The node that calls `interrupt()`, and resuming | The in-memory checkpointer, session ids, a command-line driver |
| 9 | API and UI | feat/web-ui | Claude writes it all | FastAPI routes, the Gradio page |

**Done when:**
- Ingesting chapter 1 gives about 20 topics with pages, and no caption or header text is kept.
- Every passage stays inside one topic, with word counts in range.
- For 8 of the 10 test queries, the right passage is in the top 5, and "pizza recipes" gets "the material doesn't cover this".
- At least 90% of the questions written for 10 passages pass the quote check, and a made-up quote is rejected.
- Five test answers (correct, partly correct, wrong, correct with an extra fact, off-topic) get sensible grades.
- The same answer submitted twice uses one LLM call, while a one-word change is graded fresh.
- A command-line run and then a browser session both go: question → answer → grade with passage and page → next → stop.

### M2 (v0.2.0): your feedback, the grading report, multiple choice
| # | Concept | Branch | You write | Claude writes |
|---|---|---|---|---|
| 1 | Capturing your feedback | feat/marks-flags | Claude writes it all | "Grade is wrong" with the grade picker, "Flag" with the 4 reasons, the marks and flags tables. Flagged questions leave the pool. Your grade feeds the same-answer check |
| 2 | Measuring the grader | feat/grading-report | The agreement calculation: the share of marked grades you agreed with, and a table of the tutor's grade vs. yours | The report page: agreement, the grade table, flags by reason, the share of questions flagged |
| 3 | Multiple-choice questions | feat/multiple-choice | The MC prompt and schema: question, 4 options, the correct one, why each wrong option is wrong, a quote | Grading MC in code, an MC pool per passage, the typed/MC/mixed choice, the options UI |
| 4 | Grading regression set (optional) | test/grading-regression | Picking about 15 answers and the grade each should get | `scripts/regrade.py`: re-grades them after any change to `prompts/grader.md` and lists every grade that changed |

**Done when:**
- Correcting a grade changes what the same answer gets next time.
- A flagged question is never asked again.
- After 10 marked answers, the report shows the agreement, the grade table and flags by reason.
- 10 MC questions pass the quote check, and answering them makes zero LLM calls (according to the token log).

### M3 (v0.3.0): reviews come back when due
| # | Concept | Branch | You write | Claude writes |
|---|---|---|---|---|
| 1 | Scheduling with FSRS | feat/fsrs-scheduling | The update function: turn the final grade into an FSRS rating and update the passage's card | Card storage (JSON in SQLite), the review log, a card created the first time a passage is studied |
| 2 | Due reviews first | feat/due-queue | Claude writes it all | "You choose" puts the most overdue first. A review asks the passage's next unused question (refilling the pool when empty). A "reviews due" count on the start screen |

**When scheduling happens:** when you press Next or Stop, after your chance to correct the grade. So the final grade is yours if you corrected it. Default FSRS settings (90% target recall), to be revisited after a week of use.

**Done when** (using a fake clock):
- A correct answer schedules days ahead, and a wrong one schedules it sooner.
- On a later date, due passages come first.
- A corrected grade changes the rating used.
- A review never repeats a question you've already had on that passage.

### M4 (v0.4.0): weak topics and mistake patterns
| # | Concept | Branch | You write | Claude writes |
|---|---|---|---|---|
| 1 | Tagging mistakes | feat/mistake-tags | Extend the `Grade` schema and grading prompt: a mistake type (missing key point / wrong fact / mixed up with another idea / too vague), plus the wrong statements quoted from your answer | The mistakes table |
| 2 | Spotting mix-ups with retrieval | feat/confusion-detect | `find_confused_topic()`: search the passages with a wrong statement and return the closest *other* topic, if it's close enough | Storing the topic pair |
| 3 | Pattern and weak-topic rules | feat/patterns | The pattern rule: the same mix-up pair, or the same mistake type on one topic, twice in your last 10 answers | The weak-topic rule. A pattern is resolved after a correct follow-up on each topic of the pair |
| 4 | Routing to follow-ups | feat/follow-ups | The routing function after grading: follow-up or normal next question | The follow-up node (one single-passage question per topic of the pair), and "You choose" adding follow-ups and weak topics |

**Done when:**
- A tagging answer that describes classification is recorded as a classification ↔ tagging mix-up.
- Two such mix-ups make the next question a follow-up.
- Correct follow-ups resolve the pattern.
- A topic with answers ✗ ½ ✗ ✓ ✗ counts as weak.

### M5 (v0.5.0): pause and resume, progress, adding material
| # | Concept | Branch | You write | Claude writes |
|---|---|---|---|---|
| 1 | Resuming after a restart | feat/durable-sessions | Switching the checkpointer to SQLite, and the resume logic (find the open session, continue at its pause) | The sessions table, the "Resume" button, `checkpoints.db` |
| 2 | Progress view | feat/progress | Claude writes it all | The progress page: topics studied, reviews due today and this week, weak topics with their last answers, active mix-ups |
| 3 | Adding material in the app | feat/add-material | Claude writes it all | Upload a PDF, or add chapters of the current one, with a progress bar. Replacing the PDF archives the old database instead of deleting it (one document at a time). A PDF without bookmarks falls back to page-range topics |

**Done when:**
- Stopping the server mid-question and restarting it, then pressing Resume, brings back the same waiting question.
- The progress page's numbers match a direct database query.
- Adding chapter 2 in the app creates its topics and passages.

### Adding more chapters (after M2)
- Add chapters 2–10 one at a time, with the ingest command (or the M5 page).
- **Before chapter 11 (attention and transformers):** count flags by reason. If "messy formula/code" is the most common reason, switch the source to the book's Markdown files (github.com/d2l-ai/d2l-en). A Markdown reader (branch `feat/markdown-source`, Claude writes it all) fills the same topics and passages tables, and citations show section numbers instead of pages. Everything after ingestion stays the same.

### M6 (v1.0.0): someone else can try it
| # | Concept | Branch | You write | Claude writes |
|---|---|---|---|---|
| 1 | Demo mode | feat/demo-mode | The per-visitor limit check | A fresh throwaway database per visitor (copied from a ready-made chapter-1 database). A daily cap on total LLM calls, kept well under the free limits. A banner explaining the demo |
| 2 | Packaging and hosting | feat/deploy | Claude writes it all | Dockerfile, the Hugging Face Space (free CPU), the Groq key as a Space secret, a build that downloads the PDF from d2l.ai and ingests chapter 1 |
| 3 | Demo video (backup) | docs/demo-video | Recording a roughly 3-minute walkthrough (OBS Studio, free) | The script of what to show |

Then add the demo link and credit to the README, tag `v1.0.0` and publish the GitHub Release.

**Done when:**
- The link works in a private window.
- Two browsers get separate sessions.
- The limit message appears after the cap.
- The day's LLM usage stays under the cap.

## Verification
- Each milestone's **Done when** list is its acceptance check, run before tagging the release.
- pytest covers the deterministic part of every core function. GitHub Actions runs it, together with ruff, on every pull request, and a PR with a failing check isn't merged.
- The LLM steps (question writing, grading, mistake tagging) are checked by the scripted runs in the Done-when lists, plus the optional grading regression set from M2.

## Still open
- The proposed decisions, each confirmed before the milestone that needs it.
- Public repo, which is recommended because of the AGPL license and the M6 demo.
