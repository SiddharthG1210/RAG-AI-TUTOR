# Learning log

The handoff between sessions: where the last session stopped, what's next, rebuild results and open questions. Read it at the start of every session. Claude updates it when the user asks (see CLAUDE.md › Workflow). What the user has learned, and through what, is in `PROGRESS.md`.

## Where we are
- **Milestone:** M1 (v0.1.0), "one question, graded, with the passage shown", in progress.
- **Done in M1:** concepts 0 to 4 (see Concepts done). Each one's pull request is merged and ticked on the Study Tutor Map.
- **Next session:**
  1. Rebuild task (see Rebuild results).
  2. The M1 concept "Grading grounded in the passage", on branch `feat/grading`. The branch's first commit is the map's refreshed date in `docs/project-map.json`, which was left uncommitted when the concept "Structured output (question writing)" was merged.
- **Proposed decisions:** none is needed for grading. Gradio is confirmed before the M1 concept "API and UI".

## Concepts done
- **"A rough first version in one file"** (2026-10-05): `scratch/first_loop.py`, never committed. The user wrote `main()`; Claude wrote the two inline Groq calls.
- **"Turning the PDF into prose with page numbers"** (PR #4, 2026-10-06): `src/tutor/ingest/pdf.py`. The PDF step also rejoins words split across lines; when a split falls on a real hyphen it glues the word ("multi-label" → "multilabel"), which the quote check later had to allow for.
- **"Chunking"** (PR #5, 2026-10-08): `src/tutor/ingest/chunk.py`. Chapter 1 is 49 passages of 120–350 words.
- **"Embeddings and vector search"** (PR #6, 2026-10-11): `src/tutor/retrieval/`, with `scripts/check_retrieval.py`.
- **"Structured output (question writing)"** (PR #7, 2026-10-13):
  - **The chain:** passage row → messages (the prompt + the passage text, plus earlier questions on a refill) → `complete_structured(..., QuestionSet)` → `quote_in_passage()` per question → `save_questions()` → 0 to 3 kept questions.
  - **Who wrote what:** the user wrote the `Question` and `QuestionSet` models. At the user's request Claude wrote the quote check, the prompt and `write_questions()`, and explained each step afterwards.
  - **Check script, gpt-oss-20b at reasoning "low":** 29 of 30 quotes passed (97%; M1 needs 90%). A made-up quote and one-word changes were all rejected.

## Rebuild results
- **Concept "A rough first version in one file":** the rebuild of `main()` was never recorded.

**Next session's rebuild task:** write `Question` and `QuestionSet` again from memory, in a file in `scratch/`, then say where the rule "exactly 3 questions" has to live and why. Answer: in Pydantic (`min_length`/`max_length`, checked after the call) and in words the LLM reads (the docstring or the prompt), because `strict_schema()` drops count rules before the call.

## Open questions
- **"According to the passage" in questions:** about 4 in 10 questions mention the passage, which the learner never sees. Two prompt wordings, reasoning "medium" and gpt-oss-120b didn't fix it. Accepted as a known limit. Revisit with Google Gemini Flash while building grading: it needs a free Google AI Studio key and a change to the PLAN.md decision "LLM: Groq free tier". `scripts/check_questions.py --model …` compares models.
- **For the M1 concept "Conditional edges":** `write_questions()` can fail in two ways, and routing must handle both: no quote passes (0 kept), or the call itself fails, for example when the LLM returned 1 question instead of 3 and Pydantic refused it.
- **Quotes cut mid-sentence** can drop a "not" and still pass. The prompt asks for whole sentences, and 1 quote in 29 still started mid-sentence. If the check script shows more, code could require a quote to start at a sentence start.
- **Not answerable from the passage:** a question can have a real quote and still ask something the passage doesn't explain. That's a case for the M3 flag.
- Whether to plan a production stage (it would be `v2.0.0`). Decide after `v1.0.0`.
