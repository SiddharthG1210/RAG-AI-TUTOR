# Study Tutor

An adaptive study tutor that quizzes you on your own study material.

It reads the material, writes one question at a time from a single passage, and grades your typed answer against that passage, showing it to you with its page and section number. It never adds facts from outside the material. It also schedules reviews with spaced repetition (FSRS) so you revisit each passage just before you'd forget it, and it notices which ideas you keep mixing up.

> **Status:** in development. The first working version, `v0.1.0`, asks one question, grades it and shows the passage. See [PLAN.md](PLAN.md) for the roadmap to `v1.0.0`.

## How it's built
- **RAG:** PyMuPDF extracts the PDF's text, which is cut into passages. The passages are embedded locally and searched with a vector store.
- **Agentic loop:** LangGraph runs the study session and pauses while it waits for your answer. Plain code decides every step. The LLM only writes questions and grades answers.
- **LLM:** Groq's free tier, through the `openai` SDK.
- **Storage:** SQLite. **Web:** FastAPI.

## Run it locally
You need Python 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync                     # create .venv and install the exact versions in uv.lock
uv run pytest               # run the tests
copy .env.example .env      # then put your free Groq API key in .env
uv run python scripts/ingest.py --chapter 1   # load chapter 1 of the PDF in data/
```

More commands (starting the app) get added here as they're built.

## Study material
The first material is [*Dive into Deep Learning*](https://d2l.ai) by Aston Zhang, Zachary C. Lipton, Mu Li and Alexander J. Smola, licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The book isn't included in this repo. Download [the PDF](https://d2l.ai/d2l-en.pdf) into `data/`.

## License
[AGPL-3.0-or-later](LICENSE), because the project uses PyMuPDF, which is AGPL-3.0 licensed.
