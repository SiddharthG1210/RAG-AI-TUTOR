"""The tutor's settings, read from the ``.env`` file.

``pydantic-settings`` reads each field of ``Settings`` from an environment
variable of the same name (``llm_api_key`` ← ``LLM_API_KEY``), or from the
``.env`` file, and checks its type. A missing or empty key fails right away
with a clear message, instead of failing later inside an LLM call.

``.env`` holds the real values and is never committed; ``.env.example`` shows
which ones exist (PLAN.md › Tech).
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every setting the tutor reads from ``.env``.

    Attributes:
        llm_base_url: Where the LLM API lives. Groq's OpenAI-compatible URL,
            so the openai SDK can talk to it.
        llm_api_key: The Groq key. Required, and can't be empty.
        question_model: The model that writes questions (the smaller one).
        grader_model: The model that grades typed answers (the larger one).
    """

    # env_file: also read .env; extra="ignore": other variables in .env are fine.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = Field(min_length=1)
    question_model: str = "openai/gpt-oss-20b"
    grader_model: str = "openai/gpt-oss-120b"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Read the settings once, and return that same object on every later call.

    Returns:
        The settings.

    Raises:
        pydantic.ValidationError: If ``LLM_API_KEY`` is missing or empty.
    """
    return Settings()
