"""Tests for ``tutor.config``: settings come from the environment, and a missing key fails at once.

``_env_file=None`` tells pydantic-settings not to read the real ``.env``, so
the tests see only the variables they set with ``monkeypatch``.
"""

import pytest
from pydantic import ValidationError

from tutor.config import Settings


def test_settings_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each field comes from the variable with the same name in capitals."""
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("QUESTION_MODEL", "some/model")
    settings = Settings(_env_file=None)
    assert settings.llm_api_key == "test-key"
    assert settings.question_model == "some/model"
    assert settings.grader_model == "openai/gpt-oss-120b"  # the default


def test_an_empty_key_fails_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty LLM_API_KEY is refused when the settings are read, not later inside a call."""
    monkeypatch.setenv("LLM_API_KEY", "")
    with pytest.raises(ValidationError, match="llm_api_key"):
        Settings(_env_file=None)
