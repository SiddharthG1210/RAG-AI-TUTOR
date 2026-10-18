# Learning log

The handoff between sessions: where the last session stopped, what's next, rebuild results and open questions. Read it at the start of every session. Claude updates it when the user asks (see CLAUDE.md › Workflow). What the user has learned, and through what, is in `PROGRESS.md`.

## Where we are
- **Milestone:** M1 (v0.1.0), "one question, graded, with the passage shown", in progress.
- **Done in M1:** concepts 0 to 5 (see Concepts done). Each one's pull request is merged and ticked on the Study Tutor Map.
- **Next session:** the M1 concept "The study loop: state, nodes, edges", on branch `feat/study-graph`. The branch's first commit is this log and progress update, plus the map's `start` moved to that concept in `docs/project-map.json`.
- **Proposed decisions:** none is needed for the study loop. Gradio is confirmed before the M1 concept "API and UI"; the daily review cap and the FSRS ratings before M4.

## Concepts done
- **"A rough first version in one file"** (2026-10-05): `scratch/first_loop.py`, never committed. The user wrote `main()`; Claude wrote the two inline Groq calls.
- **"Turning the PDF into prose with page numbers"** (PR #4, 2026-10-06): `src/tutor/ingest/pdf.py`. The PDF step also rejoins words split across lines; when a split falls on a real hyphen it glues the word ("multi-label" → "multilabel"), which the quote check later had to allow for.
- **"Chunking"** (PR #5, 2026-10-08): `src/tutor/ingest/chunk.py`. Chapter 1 is 49 passages of 120–350 words.
- **"Embeddings and vector search"** (PR #6, 2026-10-11): `src/tutor/retrieval/`, with `scripts/check_retrieval.py`.
- **"Structured output (question writing)"** (PR #7, 2026-10-13):
  - **The chain:** passage row → messages (the prompt + the passage text, plus earlier questions on a refill) → `complete_structured(..., QuestionSet)` → `quote_in_passage()` per question → `save_questions()` → 0 to 3 kept questions.
  - **Who wrote what:** the user wrote the `Question` and `QuestionSet` models. At the user's request Claude wrote the quote check, the prompt and `write_questions()`, and explained each step afterwards.
  - **Check script, gpt-oss-20b at reasoning "low":** 29 of 30 quotes passed (97%; M1 needs 90%). A made-up quote and one-word changes were all rejected.
- **"Grading grounded in the passage"** (PR #8, 2026-10-14):
  - **The chain:** question row + typed answer → `get_passage()` (the question's own passage) → `tidy_answer()` → `find_attempt()` (a hit returns the earlier grade, with no LLM call) → messages (the grader prompt + passage, question, key points, answer) → `complete_structured(..., Grade)` → `quote_in_passage()` on every quote → `save_attempt()` → a `GradedAnswer` (attempt id, grade, supported, reused).
  - **Who wrote what:** at the user's request ("just build it") Claude wrote all of it: the `Grade` form, `prompts/grader.md`, `grade_answer()`, the attempts table, `scripts/regrade.py` and its five test answers. The user shaped the design in discussion (quotes first and the verdict last; the names `verdict` and `grade_details`), then asked about the finished code line by line.
  - **Decisions:** the `Grade` fields are written in the order quotes, missing, wrong, extras, verdict. `grade_answer()` looks up the passage itself. A grade with no quotes, or a quote not in the passage, is saved as unsupported and never reused. A reused grade isn't saved again. `session_id` exists in the attempts table and stays empty until "Pausing for your answer".
  - **`scripts/regrade.py`, gpt-oss-120b at reasoning "medium":** 5 of 5 verdicts as expected, every quote found in the passage, the same answer sent again reused with 0 LLM calls, a one-word change graded fresh. About 2,200 tokens per grade, inside PLAN.md's estimate of roughly 100 typed answers a day.

## Rebuild results
- **Concept "A rough first version in one file":** the rebuild of `main()` was never recorded.
- **Rebuilds stopped** at the start of grading (2026-10-13): the user asked to skip rewriting last session's code and go straight to the new concept.

## Open questions
- **The five test answers in `scripts/regrade.py`** and the verdict each should get were written by Claude. PLAN.md makes them the user's call, and the user hasn't reviewed them yet.
- **For the M1 concept "The study loop: state, nodes, edges":** `grade_answer()` raises `ValueError` on an empty answer (the loop asks again before calling it) and `FreeLimitReached` when Groq's free tier runs out. `supported=False` means the screen shows "I couldn't find support for this grade" instead of the grade.
- **For the M1 concept "Conditional edges":** `write_questions()` can fail in two ways, and routing must handle both: no quote passes (0 kept), or the call itself fails, for example when the LLM returned 1 question instead of 3 and Pydantic refused it.
- **"According to the passage" in questions:** about 4 in 10 questions mention the passage, which the learner never sees. Two prompt wordings, reasoning "medium" and gpt-oss-120b didn't fix it. Accepted as a known limit. Gemini Flash wasn't tried during grading; it needs a free Google AI Studio key and a change to the PLAN.md decision "LLM: Groq free tier". `scripts/check_questions.py --model …` compares models.
- **The quote comes last in `Question`,** so it can't shape the question: the LLM has already written the question when it picks the quote. Moving `quote` first (as `Grade` does with its quotes) might cut questions the passage can't answer. Untested; `check_questions.py` can't measure it, because it only checks quotes.
- **Quotes cut mid-sentence** can drop a "not" and still pass. The prompt asks for whole sentences, and 1 quote in 29 still started mid-sentence. If the check script shows more, code could require a quote to start at a sentence start.
- **Not answerable from the passage:** a question can have a real quote and still ask something the passage doesn't explain. That's a case for the M3 flag.
- **The grader's "extras":** in the first regrade run it listed the off-topic answer's regression sentence as an extra, although the passage mentions regression. The verdict was right, and the second run didn't repeat it. Watch for it in later runs.
- Whether to plan a production stage (it would be `v2.0.0`). Decide after `v1.0.0`.
