# Study Tutor: project plan (PLAN.md)

## Context
An adaptive study tutor. You give it study material and it quizzes you on it:
- it writes questions from the material,
- grades your answers against the source,
- answers your own questions about the material, citing the passages it used,
- schedules reviews so you revisit things before you forget them,
- remembers what you keep getting wrong.

You're the first user, and you're building it to learn RAG and agentic AI. Every tool must be free.

The first material is *Dive into Deep Learning* (the d2l.ai PDF, CC BY-SA 4.0, credit to Zhang, Lipton, Li & Smola), starting with chapter 1, which is mostly prose. The repo `D:\projects\RAG-AI-tutor` is public on GitHub at https://github.com/SiddharthG1210/RAG-AI-TUTOR.

Machine: Windows 11, Python 3.12.6, Git 2.53 (Git Credential Manager set up), uv, a GTX 1650 with 4 GB of VRAM, 15 GB of RAM, and winget. Not installed: the gh CLI, Ollama, Docker.

This file is the complete build plan, from setup (M0) to the shareable v1 (M7): what to build and why. How to work in the repo (workflow, git, coding style) is in `CLAUDE.md`. The session handoff is in `LEARNING_LOG.md`, and what you've learned is in `PROGRESS.md`.

## Scope
**Rules the tutor must follow:**
- Every question the tutor asks must be answerable from a single passage of the material, and that passage is stored with the question.
- Every grade shows the passage it was based on, and every "Ask the book" answer cites the passages it used, each with its page and section number.
- The tutor uses only the material. It never adds facts from outside it.
- If it can't find support in the material for a question, a grade or an answer, it says so instead of guessing.
- One question at a time. It waits for your answer before continuing.
- The same answer to the same question gets the same grade.
- It only asks about ideas explained in prose: never about formulas, code or figures.

**Not in the first version:** accounts or multiple users, more than one document at a time, adding material from the web page (the ingest command does it), images, diagrams and tables in the material, and voice.

## Decisions
| Question | Choice | Notes |
|---|---|---|
| Topic | The smallest numbered heading above a passage, read from the PDF's bookmarks | About 20 topics in chapter 1. The Summary and Exercises sections are skipped |
| Passage | 120–350 words of whole paragraphs, never crossing a topic boundary | Stores its topic and page range. The cap of 350 keeps nearly every passage inside the embedding model's limit of 512 tokens (about 380 words of plain prose). The model drops anything past the limit, header included, so search can't see the end of a longer passage, though its stored text stays whole; the ingest command lists every such passage (chapter 1 has one, 15 tokens over). Edge cases, decided in the M1 concept "Chunking": a paragraph split by a page break is glued back first; a short last passage borrows paragraphs from the one before it; a topic under 120 words stays one short passage, listed in the ingest summary; a single paragraph over 350 words stops the ingest, and splitting it at a sentence end is written when a chapter first needs it |
| Reviews | A new question from the same passage, taken from a pool of 3 | The scheduled item (the FSRS card) is the passage. One LLM call writes 3 questions the first time a passage is studied, and the pool is refilled when it runs out, with the passage's earlier questions sent along so they aren't repeated. A session asks only one question per passage. Questions are never written in advance for the whole book |
| Extra facts in an answer | Grade only what the question asks | Extras the passage doesn't cover get a note and no penalty. Extras that contradict it count as errors |
| "Grade is wrong" | You pick the right grade | Your grade is used for scheduling and for repeated answers. The grading report lists it as a disagreement, but its agreement figure comes only from a fixed sample of grades (M3 "Measuring the grader") |
| Answer type | Typed and multiple choice | MC arrives in M3. Each session: typed, MC or mixed (about every third question is MC). MC is graded in code, with no LLM call |
| Weak topic | Low recent score | At least 3 answers, and the average of the last 5 is below 0.6 (correct = 1, partly = 0.5, incorrect = 0; MC: 1 or 0) |
| Ask the book | An agent with one tool, `search_book(query)`, used after a grade | The LLM decides what to search for, at most 3 searches. Code runs each search, and the answer must cite its passages with quotes that code checks. Up to 4 LLM calls per question, only when you use it. Arrives in M2 |
| LLM | Groq free tier | `openai/gpt-oss-20b` writes questions and `openai/gpt-oss-120b` grades. Each model is limited to 30 requests/min, 1K requests/day, 8K tokens/min and 200K tokens/day (roughly 100 typed answers a day). Re-check them at console.groq.com/settings/limits. Both are reasoning models: their reasoning tokens count toward these limits, 8K tokens/min is the tightest, and every call sets `reasoning_effort`. **Proposed:** `openai/gpt-oss-20b` also answers Ask the book, so it doesn't spend the grader's daily tokens. Confirm before M2 |
| PDF library | PyMuPDF | It's AGPL-3.0, so the repo is AGPL-3.0 and its source must be public once the app is hosted |
| Embeddings | `BAAI/bge-small-en-v1.5` (384 numbers per text, MIT) through `fastembed`, local, on the CPU | fastembed runs the model on ONNX Runtime with no PyTorch, so CI, which installs every dependency on each pull request, and the M7 image stay small. sentence-transformers runs the same model on PyTorch, which pulls several GB of GPU libraries on Linux; it's only worth it for fine-tuning, which isn't planned. Passages and queries must go through the same model: vectors from different models can't be compared, and changing the model means embedding every passage again |
| Vector store | No vector database: passage vectors are stored in SQLite, and search is a NumPy cosine against every passage | Checking every vector takes about a millisecond even for the whole book (about 2,000 passages); a vector database only pays off at hundreds of thousands of vectors. Only `search()` touches the vectors, so a database can replace its inside later without changing anything else. Chroma is planned as an optional extra after v1.0.0, to learn a vector database's API (Roadmap › After v1.0.0) |
| UI | **Proposed:** Gradio, mounted inside FastAPI | Confirm before M1 "API and UI" |
| Python environment | uv | Installs the packages and keeps a lockfile (`uv.lock`), so this machine and CI install exactly the same versions |
| FSRS ratings | **Proposed:** typed: correct → Good, partly → Hard, incorrect → Again. MC: correct → Hard, wrong → Again | MC counts for less, because recognising an answer is easier than recalling it. A side effect to weigh: in FSRS, Hard raises the passage's difficulty and only Easy lowers it (this mapping never gives Easy), so frequent MC answers would shorten its typed-review intervals. The alternative: MC wrong → Again, MC right → no FSRS update, because recognising an answer doesn't prove you can recall it. Confirm before M4 |
| Mix-up follow-ups | **Proposed:** one single-passage question per topic of the pair, each about what makes that topic different | This keeps the single-passage rule. Confirm before M5 |
| Sharing (M7) | **Proposed:** a live demo on Hugging Face Spaces (free CPU) in demo mode, preloaded with every chapter that passed its checks, with a demo video as the backup | Confirm before M7 |

## Tech
- **Decided:** Python, FastAPI, LangGraph, SQLite, the `fsrs` library, Git and GitHub, PyMuPDF, bge-small-en-v1.5 through fastembed, NumPy (vector search), the Groq free tier, uv, ruff (linting and formatting).
- **Proposed** (see Decisions): Gradio.
- **Optional, after v1.0.0:** Chroma.
- **Implementation choices:**
  - LLM calls go through a thin wrapper around the `openai` SDK, pointed at Groq's OpenAI-compatible URL, so switching provider only means editing `.env`. Most LangGraph tutorials use LangChain chat models instead. Most of our calls are a single structured-output request, though, and Ask the book's tool calls use the same SDK's `tools` parameter, so a thin wrapper is enough and keeps the mechanism visible.
  - SQLite through Python's built-in `sqlite3` with plain SQL. A production app would usually add SQLAlchemy with migrations, which is overkill for one user.
  - Config from `.env` through `pydantic-settings`.
  - GitHub Actions runs ruff and pytest on every pull request. It's free for public repos, and a pull request can't be merged until the tests pass, which keeps `main` working.

## How the tutor behaves in specific cases
- **Flag reasons:** not answerable from the passage, ambiguous, incorrect, and messy formula/code. The last reason feeds the check on whether to switch to the book's Markdown files (see *Adding more chapters*).
- **Citations** show the section number as well as the page, so they still work if the source switches to Markdown, which has no pages. The app credits the book (title, authors, CC BY-SA 4.0) wherever it shows the book's text, not only in the README.
- **Same answer, same grade:** before grading, the tutor looks up earlier answers to the same question.
  - The match must be exact after tidying (lowercase, extra spaces removed). It's never a similarity match, because one word ("is" → "isn't") can flip the meaning.
  - A hit reuses the earlier grade, and your corrected grade wins.
  - The lookup also includes the grading prompt's version and the grading model with its settings, so changing either one grades answers afresh.
  - This also stops double-submits from costing a second call.
- **Quotes checked in code:** for every question, every grade and every Ask the book citation, the LLM returns a quote from the passage, and code checks that it's really there.
  - Both texts are normalized first: Unicode NFKC (which turns PDF ligatures like "ﬁ" into "fi"), lowercase, straight quotes, words split across lines rejoined, and whitespace collapsed. Then the quote must appear exactly.
  - It's never a fuzzy match, because inserting one "not" into a real quote still scores about 95% similar. It never compares meaning either, because a made-up line that only sounds similar must fail. If fewer than 90% of questions pass, look at the failures before loosening the rule.
  - The check proves the quote is in the passage, not that it supports the question or the grade. Flags (M3) catch those.
  - If none of a passage's 3 questions pass, the LLM writes them once more, and a second failure skips the passage.
  - If a grade's quotes fail, the tutor says "I couldn't find support for this grade".
- **Prompt versions stored:** a hash of the prompt file is saved with every question and grade, so a change in behavior traces back to the commit that changed the prompt.
- **Free-tier guard:**
  - Tokens are logged per call and per model per day.
  - On HTTP 429 (rate limit reached), the tutor waits as long as the `retry-after` header says, retries, then shows "free limit reached" instead of crashing.
  - Questions are written only for passages you actually reach.
- **Pausing for your answer:** the study loop is split across web requests.
  - "Start" runs the graph until it pauses, saves the state and returns the question.
  - "Answer" loads the state and resumes from the same spot.
  - The loop pauses twice per question: for your answer, and after the grade until you press Next or Stop. "Grade is wrong", "Flag" and Ask the book happen during the second pause, so scheduling always uses your final grade.
  - On resume, LangGraph re-runs the paused node from its start. So a node that pauses does nothing before it calls `interrupt()`: a database write or an LLM call placed there would happen twice.
  - LangGraph's `interrupt` needs a checkpointer: an in-memory one in M1, SQLite from M6, so sessions survive a restart.
  - Each study session has its own id (LangGraph's thread id), which the page sends with every request.
- **Topic search:**
  - A typed topic is embedded and matched against the passages by meaning. It gets bge's query prefix ("Represent this sentence for searching relevant passages: "); passages don't.
  - Each passage is embedded with its section number and title in front (for example "1.3.1.2 Classification"), so a passage that says "this method" still carries its context. The stored text, which the quote check uses, stays unchanged.
  - Picking a topic from the list uses plain SQL.
  - If even the closest passage is too far away, the tutor says "the material doesn't cover this". The cut-off is tuned in M1 "Embeddings and vector search" with off-topic queries ("pizza recipes") and with topics chapter 1 only mentions without explaining (attention, GANs, GPUs). A cut-off measures closeness, not coverage, so the second kind shows where it falls short.
- **What "You choose" picks,** in this order:
  1. due reviews, most overdue first (from M4)
  2. follow-ups on active mix-ups (from M5)
  3. weak topics (from M5)
  4. the next new passage, in book order (from M1)
- **Ask the book** (from M2):
  - During the pause after a grade, you can type a question about the material.
  - An agent answers it. The LLM gets your question and one tool, `search_book(query)`, and decides what to search for, at most 3 times. Code runs each search (the same search as a typed topic) and hands back the passages with their ids.
  - The answer must cite the passages it used, each with a quote, and code checks every quote. A failed quote shows "I couldn't find support for this answer". If no search finds anything close enough, it says "the material doesn't cover this".
  - Then the loop pauses again, so you can ask another question or press Next or Stop.

## Design rules
- **Code is in charge, and the LLM does only three jobs:** writing questions, grading typed answers and answering Ask the book questions. Plain code picks passages, decides the next step, checks quotes, reuses grades and chooses what to show. The routing functions are plain Python reading the graph's state. Many agent systems let an LLM choose the next step, but these rules are clear and code costs no tokens.
- **Ask the book is the one agent, and it's fenced in.** The LLM chooses its searches, but code runs them, caps them at 3 and checks every citation. It's where you learn how agents work, in the one place where letting the LLM choose the next step is actually useful.
- **Every LLM call starts blank.** The passage is sent with every call (for Ask the book: your question, then the passages its searches return). That reduces the use of outside knowledge but can't prevent it, which is why quotes are checked in code.
- **Three kinds of matching:**

| Where | What's compared | How strict |
|---|---|---|
| Finding passages for a typed topic or an Ask the book search (M2), and spotting mix-ups (M5) | Meaning (embeddings) | The closest few, plus a distance cut-off |
| Checking the LLM's quotes | Characters, after normalizing | 100% (exact) |
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
     ├──► Groq LLM ........... writes questions, grades typed answers,
     │                         and answers Ask the book as an agent
     ├──► bge-small .......... turns a typed topic or a search into
     │      │                  an embedding (run by fastembed)
     │      └──► NumPy ....... compares it with every passage's
     │                         embedding and finds the closest
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
  b. Keep only prose; drop code, math, figure ........ Python (you write)
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
     └──► f. Turn each passage, with its section ..... bge-small
             number and title in front, into an
             embedding, and store it ................. SQLite

  No LLM runs here, so adding material uses none of the free quota.
  SQLite holds everything, embeddings included.
```

### Diagram 3: a study session (WHEN = the milestone that adds each part)
```
                                                    TECH             WHEN
  a. Pick the answer type: typed / MC / mixed ..... Gradio*          M3
     Type a topic, or click "You choose" .......... Gradio*          M1
     │
     ▼
  b. Pick a passage
     • typed topic -> the most similar passages ... bge + NumPy      M1
       (nothing close enough -> "the material doesn't cover this")
     • "You choose" -> due reviews first .......... fsrs + SQLite    M4
                    -> mix-up follow-ups, weak .... SQLite           M5
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
     • multiple choice -> code checks your pick ... Python           M3
     • tag the mistake, spot mix-ups .............. LLM + search     M5
     │
     ▼
  f. Show the grade, what's missing or wrong, ..... Gradio*          M1
     and the passage with its page number
     Save the attempt ............................. SQLite           M1
     │
     ▼
  g. PAUSE until "Next" or "Stop" ................. LangGraph        M1
     Meanwhile you may:
     • click "Grade is wrong" or "Flag" ........... SQLite           M3
     • ask the book a question: an agent .......... LLM agent        M2
       answers it, then it pauses here again
     │
     ▼
  h. Save your final grade ........................ SQLite           M1
     Schedule when this passage comes back ........ fsrs             M4
     │
     ▼
  i. Next -> active mix-up? ask a follow-up ....... LangGraph        M5
             else back to b
     Stop -> pick up later where you left off ..... LangGraph+SQLite M6
```

### Diagram 4: which milestone needs which
```
  M1  Ask a question, grade it, show the passage
   │    needs: material added, the LLM, the study loop
   ▼
  M2  Ask the book: an agent that searches the book to answer you
   │    needs M1: search, the LLM wrapper, the quote check, the graph
   ▼
  M3  "Grade is wrong", flags, grading report, multiple choice
   │    needs M1: there have to be grades to mark
   ▼
  M4  Reviews come back when due
   │    needs grades (M1) and your corrections (M3)
   ▼
  M5  Weak topics + follow-up questions on repeated mistakes
   │    needs a history of graded answers (M1, M3, M4)
   ▼
  M6  Pause/resume + progress page
   │    the progress page shows M4's due reviews and M5's weak topics
   ▼
  M7  Someone else can try it: live link or demo video
        needs everything above working
```

## Data (SQLite tables, by the milestone that adds them)
| Milestone | Tables |
|---|---|
| M1 | **topics:** number, title, level, page (both the PDF's page and the number printed on it, since the front matter makes them differ: printed page 10 is the PDF's page 50). **passages:** topic, text, page range (start and end, as PDF and printed pages), order. **embeddings:** passage, model, vector. **questions:** passage, text, key points, quote, prompt version, used or unused. **attempts:** question, session, your answer, tidied answer, grade, explanation, prompt version, model and settings. **llm_calls:** model, tokens, time |
| M2 | **asks:** session, your question, the answer, the cited passages with their quotes, number of searches, prompt version. Optional: **passages_fts**, an FTS5 keyword index of the passages |
| M3 | **marks:** attempt, agree or disagree, your grade, sampled or not. **flags:** question, reason, note. **questions** gains: type (typed/MC), options, correct option |
| M4 | **cards:** passage, FSRS card (JSON), due date. **review_log:** passage, rating, date, attempt |
| M5 | **mistakes:** attempt, mistake type, wrong statement, mixed-up topic. **patterns:** topic pair (or topic plus mistake type), active or resolved |
| M6 | **sessions:** id, started, last active, open or closed. LangGraph keeps its own checkpoint tables in a separate `checkpoints.db` |

Embeddings (from M1) are stored in SQLite, one per passage, with the name of the model that made them, so a change of model is noticed instead of mixing vectors from two models.

**Passages are frozen once studied.** Questions, attempts and FSRS cards point to passage ids, so re-chunking a chapter you've studied would cut them off from their history. A fix to extraction or chunking applies only to chapters not studied yet.

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
docs/project-map.json         the content of the Study Tutor Map (see CLAUDE.md › Project map)
Dockerfile                    (M7)
prompts/           question_writer.md, grader.md (M1) · ask_book.md (M2) · mc_writer.md (M3)
src/tutor/
  config.py, db.py, llm.py (tool calls added in M2)
  ingest/          pdf.py, chunk.py (M1) · markdown.py (only if the source switches)
  retrieval/       embed.py, store.py, search.py (M1) · keyword.py (M2, optional)
  questions.py, grading.py (M1) · ask.py (M2) · mc.py, feedback.py, report.py (M3)
  scheduling.py (M4) · mistakes.py, patterns.py (M5)
  graph/           state.py, nodes.py, build.py (M1; grows each milestone) · agent.py (M2)
  api/             main.py, ui.py (M1) · progress page (M6) · demo.py (M7)
scripts/           ingest.py, check_retrieval.py, regrade.py (M1)
tests/             one test file per core function
data/              (ignored by git) the PDF, tutor.db, checkpoints.db, the downloaded embedding model
scratch/           (ignored by git) first_loop.py, the rough first version (M1 concept 0)
```

## Roadmap

### How every milestone is built
- **Concepts:** each concept goes through the study-tutor-teaching loop: why it's needed, reasoning out the design, you write its AI/ML/RAG code one small function at a time from a `TODO(human)` skeleton, a review, then you explain it back. Claude writes the plumbing around it. CLAUDE.md › "Who writes what" has the details. Where the table says "Claude writes it all", it's: why, Claude writes, you explain back.
- **Blank-page exercise (M1 to M6):** after a milestone's last merge, you rebuild its core flow in a file in `scratch/`, without a skeleton and without looking at the real code, in about an hour. Then you compare the two. The session-start rebuild practises recalling one function; this practises laying out the structure yourself.
- **Finishing a milestone:** after the blank-page exercise, check its "Done when" list, then set `version` in `pyproject.toml` to the milestone's version, tag it (`v0.1.0` … `v0.6.0`, `v1.0.0`) and publish a GitHub Release with short notes.

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
  6. In the GitHub repo's settings, add a rule for `main` that requires the `ruff + pytest` check to pass before a pull request can merge. The check only shows up as an option after it has run once, which is why this comes last. Leave "Require approvals" off: you can't approve your own pull request.
- **Also you:** download https://d2l.ai/d2l-en.pdf into `data/`. Create a free Groq API key and put it in `.env` (first needed in the M1 concept "A rough first version in one file").
- **Done when:**
  - `uv run pytest` passes.
  - `uv run ruff check .` and `uv run ruff format --check .` pass.
  - `git check-ignore -v data/d2l-en.pdf .env` shows both are ignored.
  - The PR is merged with a passing check.

### M1 (v0.1.0): one question, graded, with the passage shown
**Concept 0** is a rough version of the whole loop in one file, with one chapter-1 paragraph pasted in. You see the tutor work in the first session, and each later concept replaces one rough piece with a real one.

| # | Concept | Branch | You write | Claude writes |
|---|---|---|---|---|
| 0 | A rough first version in one file | none: never committed | The loop: show the question, read your answer with `input()`, print the grade | The two Groq calls, with their prompts written inline. The file is `scratch/first_loop.py`, which git ignores. It runs with `uv run --with openai --env-file .env scratch/first_loop.py`, so nothing is added to the project |
| 1 | Turning the PDF into prose with page numbers | feat/pdf-extract | `classify_block()`: label each text block as prose, heading, code, math, caption, header or table, using the fonts found in chapter 1. It decides which text retrieval ever sees: only prose is kept, and headings mark where topics start | PDF reading, a listing of chapter 1's fonts and sizes to decide from, bookmarks to topics, the topics table, the ingest command, tests |
| 2 | Chunking | feat/chunking | `chunk_paragraphs()` (packing whole paragraphs, rebalancing a short last passage) and `join_page_breaks()` (gluing paragraphs split across pages) | The passages table, saving passages during ingest, tests |
| 3 | Embeddings and vector search | feat/retrieval | Tiny example: cosine similarity in NumPy. `embed_passages()` and `embed_query()` (the section title in front of each passage, bge's prefix on queries). `search(query, k)`: cosine against every passage, top k. The distance cut-off, chosen from the retrieval check's results | Model loading. The embeddings table in SQLite, saving and loading vectors. Embedding passages during ingest. A retrieval check with 10 test queries, plus off-topic ones and topics chapter 1 only mentions. Tests |
| 4 | Structured output (question writing) | feat/question-gen | The question-writing prompt and the `QuestionSet` schema (3 questions, each with key points and a quote) | The LLM wrapper (Groq strict mode), the quote check, the question pool, the token log, 429 handling |
| 5 | Grading grounded in the passage | feat/grading | The grading prompt and the `Grade` schema (grade, what's missing, what's wrong, unchecked extras, quotes). Five test answers (correct, partly correct, wrong, correct with an extra fact, off-topic) and the grade each should get | The same-answer check, the attempts table, the "no support found" path. `scripts/regrade.py`: it re-grades the test answers after any change to `prompts/grader.md` and lists every grade that changed |
| 6 | The study loop: state, nodes, edges | feat/study-graph | The state and the `build_graph()` wiring | Node functions that call the code from concepts 1–5 |
| 7 | Conditional edges | feat/grounding-retry | The routing function: ask, write again (once), or skip the passage | Retry counters, tests |
| 8 | Pausing for your answer | feat/answer-interrupt | The node that calls `interrupt()`, and resuming | The in-memory checkpointer, session ids, a command-line driver |
| 9 | API and UI | feat/web-ui | Claude writes it all | FastAPI routes, the Gradio page |

**Done when:**
- The rough first version asks one question about a pasted paragraph and grades a typed answer.
- Ingesting chapter 1 gives about 20 topics with pages, and no caption or header text is kept.
- Every passage stays inside one topic, with word counts in range apart from the edge cases decided in "Chunking".
- For 8 of the 10 test queries, the right passage is in the top 5, and "pizza recipes" gets "the material doesn't cover this". What the topics chapter 1 only mentions get is written down.
- At least 90% of the questions written for 10 passages pass the quote check, a made-up quote is rejected, and so is a real quote with one word flipped.
- The five test answers get the grades you expected, according to `scripts/regrade.py`.
- The same answer submitted twice uses one LLM call, while a one-word change is graded fresh.
- A command-line run and then a browser session both go: question → answer → grade with passage and page → next → stop.

### M2 (v0.2.0): Ask the book, your first agent
After a grade, you can ask a question of your own. An agent answers it from the book: the LLM decides what to search for, code runs the searches, and the answer cites the passages it used. Everything here reuses M1's search, LLM wrapper, quote check and graph.

| # | Concept | Branch | You write | Claude writes |
|---|---|---|---|---|
| 1 | Tool calling | feat/tool-calling | The `search_book(query)` tool definition: the description and parameters the LLM reads to decide when to call it | Tool calls in the LLM wrapper, running a requested search, tests with a fake LLM |
| 2 | The agent loop as a graph | feat/agent-loop | The routing function: the LLM asked for a search → run it and loop back; it answered → finish; 3 searches used → finish | The agent subgraph (an LLM node and a tool node), the message history in its state |
| 3 | Answers grounded in several passages | feat/ask-answers | The answer prompt and the `BookAnswer` schema (the answer, plus each cited passage with a quote) | The quote check on every citation, the "no support found" path, the Ask the book box after a grade, the asks table |
| 4 | Hybrid search (optional) | feat/hybrid-search | Reciprocal rank fusion: merging the keyword ranking with the embedding ranking | Keyword search with SQLite FTS5 (BM25 ranking), the hybrid `search_book`, a comparison on the M1 retrieval check |

**Done when:**
- "How is tagging different from classification?" gets an answer that cites passages from both topics, with their pages.
- "pizza recipes" gets "the material doesn't cover this".
- With a fake LLM that keeps asking for searches, the loop stops after 3.
- An answer whose citation quote isn't in its passage shows "I couldn't find support for this answer".
- The token log shows at most 4 LLM calls per question.

### M3 (v0.3.0): your feedback, the grading report, multiple choice
| # | Concept | Branch | You write | Claude writes |
|---|---|---|---|---|
| 1 | Capturing your feedback | feat/marks-flags | Claude writes it all | "Grade is wrong" with the grade picker, "Flag" with the 4 reasons, the marks and flags tables. Flagged questions leave the pool. Your grade feeds the same-answer check |
| 2 | Measuring the grader | feat/grading-report | The agreement calculation: the share of sampled grades you agreed with, and a table of the tutor's grade vs. yours | Asking you to agree or disagree on every 4th grade: a fixed sample, because counting only the grades you chose to correct would skew the agreement. The report page: agreement, the grade table, flags by reason, the share of questions flagged |
| 3 | Multiple-choice questions | feat/multiple-choice | The MC prompt and schema: question, 4 options, the correct one, why each wrong option is wrong, a quote | Grading MC in code, an MC pool per passage, the typed/MC/mixed choice, the options UI |
| 4 | Growing the regression set | test/grading-regression | About 10 more answers, picked from grades you disagreed with, and the grade each should get | Adding them to the set that `scripts/regrade.py` runs |

**Done when:**
- Correcting a grade changes what the same answer gets next time.
- A flagged question is never asked again.
- After 10 sampled grades, the report shows the agreement, the grade table and flags by reason.
- The regression set holds about 15 answers, and `scripts/regrade.py` reports no unexpected changes.
- 10 MC questions pass the quote check, and answering them makes zero LLM calls (according to the token log).

### M4 (v0.4.0): reviews come back when due
| # | Concept | Branch | You write | Claude writes |
|---|---|---|---|---|
| 1 | Scheduling with FSRS | feat/fsrs-scheduling | The update function: turn the final grade into an FSRS rating and update the passage's card | Card storage (JSON in SQLite), the review log, a card created the first time a passage is studied |
| 2 | Due reviews first | feat/due-queue | Claude writes it all | "You choose" puts the most overdue first. A review asks the passage's next unused question (refilling the pool when empty). A "reviews due" count on the start screen |

**When scheduling happens:** when you press Next or Stop, after your chance to correct the grade. So the final grade is yours if you corrected it. Default FSRS settings (90% target recall), to be revisited after a week of use, except that the learning steps are turned off with `Scheduler(learning_steps=(), relearning_steps=())`. The library's default steps (1 and 10 minutes) would bring a passage back within the same session.

**Done when** (using a fake clock):
- A correct answer schedules days ahead, and a wrong one schedules it sooner.
- On a later date, due passages come first.
- A corrected grade changes the rating used.
- A review never repeats a question you've already had on that passage.

### M5 (v0.5.0): weak topics and mistake patterns
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

### M6 (v0.6.0): pause and resume, progress
| # | Concept | Branch | You write | Claude writes |
|---|---|---|---|---|
| 1 | Resuming after a restart | feat/durable-sessions | Switching the checkpointer to SQLite, and the resume logic (find the open session, continue at its pause) | The sessions table, the "Resume" button, `checkpoints.db` |
| 2 | Progress view | feat/progress | Claude writes it all | The progress page: topics studied, reviews due today and this week, weak topics with their last answers, active mix-ups |

**Done when:**
- Stopping the server mid-question and restarting it, then pressing Resume, brings back the same waiting question.
- The progress page's numbers match a direct database query.

### Adding more chapters (after M3)
- Add chapters 2–10 one at a time, with the ingest command.
- **Before chapter 11 (attention and transformers):** count flags by reason. If "messy formula/code" is the most common reason, switch the source to the book's Markdown files (github.com/d2l-ai/d2l-en). A Markdown reader (branch `feat/markdown-source`, Claude writes it all) fills the same topics and passages tables, and citations show section numbers instead of pages. Everything after ingestion stays the same.
- **A different book (only if you switch):** the ingest command takes another PDF. One without bookmarks falls back to page-range topics, and replacing the PDF archives the old database instead of deleting it (one document at a time).

### M7 (v1.0.0): someone else can try it
| # | Concept | Branch | You write | Claude writes |
|---|---|---|---|---|
| 1 | Demo mode | feat/demo-mode | The per-visitor limit check | A fresh throwaway database per visitor (copied from a ready-made database with every chapter that passed its checks). A daily cap on total LLM calls, kept well under the free limits. A banner explaining the demo |
| 2 | Packaging and hosting | feat/deploy | Claude writes it all | Dockerfile, the Hugging Face Space (free CPU), the Groq key as a Space secret, a build that downloads the PDF from d2l.ai and ingests those chapters. Docker Desktop is installed first (`winget install Docker.DockerDesktop`), so the image is tested locally before it goes to the Space |
| 3 | Demo video (backup) | docs/demo-video | Recording a roughly 3-minute walkthrough (OBS Studio, free) | The script of what to show |

Then add the demo link and credit to the README, tag `v1.0.0` and publish the GitHub Release.

**Done when:**
- The link works in a private window.
- Two browsers get separate sessions.
- The limit message appears after the cap.
- The day's LLM usage stays under the cap.

### After v1.0.0 (optional): a vector database
Search works without one (Decisions › Vector store). This extra is for learning a vector database's API, and for the CV.

| # | Concept | Branch | You write | Claude writes |
|---|---|---|---|---|
| 1 | Chroma | feat/chroma | The inside of `search()`, now querying a Chroma collection (cosine distance; Chroma's default is L2) instead of the NumPy loop | The collection saved under `data/chroma/`, rebuilt from the embeddings in SQLite, which stays the source of truth |

**Done when:**
- The M1 retrieval check gives the same top 5 for every test query as the NumPy search.
- Deleting `data/chroma/` and running the rebuild restores it.

## Verification
- Each milestone's **Done when** list is its acceptance check, run before tagging the release.
- pytest covers the deterministic part of every core function. GitHub Actions runs it, together with ruff, on every pull request, and a PR with a failing check isn't merged.
- The LLM steps (question writing, grading, Ask the book, mistake tagging) are checked by the scripted runs in the Done-when lists, plus the grading regression set (`scripts/regrade.py`, from M1).

## Still open
- The proposed decisions, each confirmed before the milestone that needs it.
