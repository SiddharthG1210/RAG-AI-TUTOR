# Learning log

The handoff between sessions: where the last session stopped, what's next, rebuild results and open questions. Read it at the start of every session. Claude updates it when the user asks (see CLAUDE.md › Workflow). What the user has learned, and through what, is in `PROGRESS.md`.

## Where we are
- **Milestone:** M1 (v0.1.0), "one question, graded, with the passage shown", in progress.
- **Done in M1:** concepts 0 to 8 (see Concepts done). Each one's pull request is merged and ticked on the Study Tutor Map.
- **Next:** the M1 concept "API and UI", on branch `feat/web-ui`, the last concept of M1. PLAN.md: Claude writes it all, the user explains it back. The branch's first commit is this log and progress update, plus the map's `start` moved to that concept in `docs/project-map.json`. After it: the blank-page exercise, M1's "Done when" list, then `v0.1.0` and a GitHub Release (PLAN.md › Finishing a milestone).
- **Proposed decisions:** before "API and UI", confirm two: Gradio (mounted inside FastAPI), and "Read the topic first". Before M4: the FSRS ratings, the daily review cap, and "Questions on a passage's first study".

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
- **"The study loop: state, nodes, edges"** (PR #9, 2026-10-18):
  - **The chain:** `{"topic": ...}` → `pick_passage` (a typed topic → `search()`, none → `next_new_passage()`) adds `passage_id` → `pick_question` (`next_unused_question()`; an empty pool → `write_questions()`, then read the pool again) adds `question_id` → `ask` (`input()`) adds `answer` → `grade` (`grade_answer()`, then `mark_question_used()`) adds `result` → `show` prints the grade and the passage. `scripts/study.py` opens the database and runs one turn.
  - **Who wrote what:** the user wrote `State` and `build_graph()` (`src/tutor/graph/`). Claude wrote the nodes, `Context`, `TurnStopped`, the four `db.py` lookups, `scripts/study.py` and the tests. Before that, a throwaway `scratch/tiny_graph.py` showed the state growing node by node, and a returned key replacing an existing one.
  - **Decisions:** the state holds ids, not rows, because LangGraph's checkpoint saver can't store a `sqlite3.Row` (tested). The database connection travels as LangGraph's runtime context (`context_schema=Context`, `invoke(..., context=...)`), not in the state. A question is marked used only after its grade. "You choose" picks the first passage in book order with no used question. `input()` stands in for the pause until "Pausing for your answer".
  - **The user's node names:** `pick_passage`, `pick_question`, `ask_question`, `grades`, `show_results`. Routing functions return these names.
  - **The real run:** "You choose" picked passage 1 of "1 Introduction", wrote 3 questions (1,624 tokens, every quote passed), and graded the answer "i dont know" as incorrect, with 2 real quotes and the 3 key points as missing. Question 1 is used; questions 2 and 3 wait in the pool.
  - **Check questions:** skipped at the user's request.
- **"Conditional edges"** (PR #10, 2026-10-19):
  - **The forks:** `pick_passage` → `route_after_pick_passage` → `pick_question`, or `topic_not_covered` → END. `pick_question` → `route_after_pick_question` → `ask_question`; or `pick_question` again (write once more); or, after `MAX_WRITE_TRIES` (2) writes, `skip_the_passage` → `pick_passage`.
  - **Who wrote what:** the user wrote both routing functions and the conditional edges in `build_graph()`, after a throwaway `scratch/tiny_routing.py` showed one fork taking each arrow. Claude wrote the rest: the nodes recording `None` instead of raising, the `write_tries` counter, the `not_covered` and `skip_passage` nodes, the `skipped_passages` table, `UnusableAnswer` in `llm.py`, and the tests.
  - **Decisions:**
    - A skipped passage is skipped for good. The skip is saved in `skipped_passages`, and both pickers leave it out. A skip kept only in the state would put the same failing passage first in every "You choose" session.
    - "The material doesn't cover this" became a route in this concept.
    - An unusable LLM answer (`UnusableAnswer`: schema broken twice, empty, or refused by Pydantic) counts as a failed try. `FreeLimitReached` still ends the turn. `TurnStopped` is gone.
  - **The user's first routing function called the nodes** instead of returning their names. The fix: a routing function only names the next node, and LangGraph runs it. The second review caught a leftover plain edge next to a conditional one; LangGraph would follow both.
  - **Found by testing:** a routing function returning a name that isn't a node doesn't raise. LangGraph logs "wrote to unknown channel … ignoring it" and the turn ends silently. `tests/test_build.py` checks every returned name is a node.
  - **Real runs:** "You choose" studied passage 2 (3 questions written, 1,461 tokens; the answer was graded correct). `--topic "pizza recipes"` took the "doesn't cover this" route. The skip route is covered by tests only: it needs the LLM to fail twice.
- **"Pausing for your answer"** (PR #11, 2026-10-22):
  - **The chain:** `scripts/study.py`'s `run_session()` makes a session id (also LangGraph's thread id) and starts the graph. It pauses in `ask_question` (`interrupt({"pause": "answer", "question": ...})`) and in a new `wait_for_next` node (`interrupt({"pause": "next"})`). At each pause `invoke()` returns the note in `result["__interrupt__"][0].value`; the loop reads the note's `"pause"` label, asks you (`read_answer()` or `read_next()`), and resumes with `graph.invoke(Command(resume=reply), config, context=...)` until no `"__interrupt__"` is left. `route_after_wait_for_next`: stop → `END`, otherwise → `pick_passage`.
  - **Who wrote what:** the user wrote both pausing nodes, `route_after_wait_for_next`, the new edges and the checkpointer in `build_graph()`, and the resume loop. Claude wrote `make_checkpointer()`, the session id in `Context`, `passages_asked_in_session()`, `nothing_new`, `read_answer()`, `read_next()` and the tests. Throwaway `scratch/tiny_pause.py` showed the two calls, and the paused node running twice.
  - **Decisions:**
    - The second pause offers the same choice as the start (a topic, Enter for "You choose", `s` to stop). Claude first proposed switching to "You choose" after Next; the user proposed asking again, and asked Claude to compare honestly. Asking again alone left a hole, so it's combined with: for a typed topic, skip passages already asked this session. Typing a topic a third time when every close passage was asked prints "Nothing new on X this session".
    - A topic the material doesn't cover no longer ends the session: `topic_not_covered` leads to the second pause.
    - Each pause hands out a dict labelled with `"pause"`, so the caller knows what to ask for without knowing node names.
    - The checkpointer's serializer is allowed to rebuild `Grade` and `GradedAnswer` (`SAVED_TYPES`), which removes LangGraph's "unregistered type" warning.
  - **Found by testing:** LangGraph saves the state at a pause but not the context, so every call, a resume included, passes `context=` again.
  - **Where the user got stuck:** what `interrupt()` returns on each of the two runs; why the note needs a label when LangGraph knows the paused node (LangGraph resumes the node; the caller decides what to ask the human); `result["__interrupt__"][0].value` layer by layer; reassigning `result` inside the `while` loop; and a tuple made by a misplaced `)` in the resume line.
  - **The real run:** one session of 4 turns: "You choose" (passage 3), "classification" (passage 21), "classification" again (passage 20, a different one), "You choose" (passage 4), then `s`. All 4 attempts carry the session id.

## Rebuild results
- **Concept "A rough first version in one file":** the rebuild of `main()` was never recorded.
- **Rebuilds stopped** at the start of grading (2026-10-13): the user asked to skip rewriting last session's code and go straight to the new concept.

## Open questions
- **The five test answers in `scripts/regrade.py`** and the verdict each should get were written by Claude. PLAN.md makes them the user's call, and the user hasn't reviewed them yet.
- **Skipped passages can't be listed yet.** PLAN.md says they can be listed to see what went wrong with them; no command does that yet. A small listing fits the M3 grading report, or a check script.
- **Every passage asked or skipped:** `next_new_passage()` raises `ValueError`, and `scripts/study.py` doesn't catch it, so the end of chapter 1 shows a traceback. Chapter 1 has 49 passages, so it's far off.
- **For the M1 concept "API and UI":**
  - Nodes print (`show`, `not_covered`, `skip_passage`), which only reaches a terminal. The page needs the grade, the passage and the messages as data, so they'll have to travel in the state or in what the pauses hand out.
  - The in-memory checkpointer lives as long as the server process, and every request must send the session id. A server restart loses paused sessions until M6's SQLite checkpointer.
  - Submit stays disabled while an answer is graded (PLAN.md › Same answer, same grade).
- **A pasted question as an answer:** in the real session, one answer began with the question's own text and was graded incorrect. Nothing stops pasting; it's only worth noting if it happens by accident in the browser.
- **"According to the passage" in questions:** about 4 in 10 questions mention the passage, which the learner never sees. Two prompt wordings, reasoning "medium" and gpt-oss-120b didn't fix it. Accepted as a known limit. Gemini Flash wasn't tried during grading; it needs a free Google AI Studio key and a change to the PLAN.md decision "LLM: Groq free tier". `scripts/check_questions.py --model …` compares models.
- **The quote comes last in `Question`,** so it can't shape the question: the LLM has already written the question when it picks the quote. Moving `quote` first (as `Grade` does with its quotes) might cut questions the passage can't answer. Untested; `check_questions.py` can't measure it, because it only checks quotes.
- **Quotes cut mid-sentence** can drop a "not" and still pass. The prompt asks for whole sentences, and 1 quote in 29 still started mid-sentence. If the check script shows more, code could require a quote to start at a sentence start.
- **Not answerable from the passage:** a question can have a real quote and still ask something the passage doesn't explain. That's a case for the M3 flag.
- **The grader's "extras":** in the first regrade run it listed the off-topic answer's regression sentence as an extra, although the passage mentions regression. The verdict was right, and the second run didn't repeat it. Watch for it in later runs.
- Whether to plan a production stage (it would be `v2.0.0`). Decide after `v1.0.0`.
