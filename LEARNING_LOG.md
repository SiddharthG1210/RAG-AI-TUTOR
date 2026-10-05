# Learning log

The handoff between sessions: where the last session stopped, what's next, rebuild results and open questions. Read it at the start of every session. Claude updates it when the user asks (see CLAUDE.md › Workflow). What the user has learned, and through what, is in `PROGRESS.md`.

## Where we are
- **Milestone:** M1 (v0.1.0), "one question, graded, with the passage shown", in progress.
- **M0 setup is done:**
  - The scaffold pull request is merged.
  - `main` is protected: the `ruff + pytest` check has to pass before a pull request can merge.
  - `data/d2l-en.pdf` is downloaded, and `.env` holds a working Groq key.
  - `git check-ignore` confirms that `.env`, `data/` and `scratch/` are ignored.
- **Done in M1:** the concept "A rough first version in one file" (see Concepts done).
- **Next session:**
  1. Rebuild task (see Rebuild results).
  2. The M1 concept "Turning the PDF into prose with page numbers", on branch `feat/pdf-extract`. Start from an up-to-date `main` (`git switch main`, `git pull`, then `git switch -c feat/pdf-extract`). PyMuPDF is added with `uv add pymupdf` in that concept's pull request. The open questions below about bookmarks and page numbers belong to this concept.
- **Proposed decisions:** none is needed for the next concept. Embeddings and Chroma are confirmed before the M1 concept "Embeddings and vector search", Gradio before "API and UI".

## Concepts done
### M1 concept "A rough first version in one file" (2026-10-05)
- **Files:** both are in `scratch/`, which git ignores, so they're never committed.
  - `scratch/hello_groq.py`: the tiny example, one Groq call that prints its token usage.
  - `scratch/first_loop.py`: Claude wrote `write_question()` (gpt-oss-20b, `reasoning_effort="low"`) and `grade_answer()` (gpt-oss-120b, `reasoning_effort="medium"`), each with its prompt written inline and returning free text. The user wrote `main()`: question → `input()`, which asks again on an empty answer → `grade_answer(PASSAGE, question, answer)` → print the grade, `PASSAGE_SOURCE` and `PASSAGE`.
- **Run with:** `uv run --with openai --env-file .env scratch/first_loop.py`.
- **Passage:** two paragraphs from section 1.3.1.2 "Classification" (printed page 10, PDF page 50), 261 words.
- **Result:** one question was asked, and a typed answer was graded "correct" with the passage shown. That meets the first item of M1's "Done when" list.
- **Explain back:** three check questions, then a summary in the user's own words. Two gaps were corrected:
  - The user mixed up *printing* the passage (for the user) with *sending* it to the grader (so the grade is grounded).
  - The user left out the question as the grader's third input.
- **Skipped for time:** the optional step of trying different answers (correct, wrong, correct with an outside fact, the same answer twice).

## Rebuild results
None yet.

**Next session's rebuild task:** write `main()` of `scratch/first_loop.py` again from memory. Either clear the function, or write it in a new file in `scratch/`. The answer only needs to get the order of the calls and the inputs to `grade_answer()` right.

## Open questions
- Whether to plan a production stage (it would be `v2.0.0`). Decide after `v1.0.0`.
- **For the M1 concept "Turning the PDF into prose with page numbers":**
  - **Bookmarks have titles only.** They say "Classification", not "1.3.1.2 Classification", so section numbers have to be computed from the bookmark levels. PyMuPDF's `get_toc()` returns `[level, title, page]`. Chapter 1 is "Introduction", level 1 at PDF page 41. Its "Summary" and "Exercises" entries are level 2 at page 69.
  - **Each page has two numbers:** the printed one (page 10, from `page.get_label()`) and the PDF file's page number (50). Decide which one citations show.
  - **Raw page text mixes in what `classify_block()` must drop:** running headers (a section title plus the page number), figure captions ("Fig. 1.3.2 Death cap - do not eat!"), and words split across lines with a hyphen ("corre-" / "sponding").
- **A test case for the M1 concept "Structured output (question writing)":** the rough question writer asked a question its passage can't answer, even though the prompt said "answerable from this passage alone":
  - The question: "How does expressing a model's output as probabilities help when it is difficult to optimize for a firm categorical assignment?"
  - Why it fails: section 1.3.1.2 says probabilities are "much easier" and leaves the reason for later chapters.
  - The quote check alone won't reject it, because the quote would be real. It's a case for the "not answerable from the passage" flag (M3).
