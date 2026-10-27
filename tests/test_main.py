"""Tests for ``tutor.api.main``: the app refuses a missing database, and serves the page.

FastAPI's ``TestClient`` sends web requests straight to the app, in the same
process, with no server and no port: a test can fetch ``/`` as a browser
would. Using it in a ``with`` block also runs the app's start-up steps, which
Gradio needs.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tutor.api.main import create_app
from tutor.config import get_settings
from tutor.db import connect


def test_create_app_refuses_a_missing_database(tmp_path: Path) -> None:
    """No chapter ingested yet: a clear error at start-up, not an empty database."""
    path = tmp_path / "tutor.db"
    with pytest.raises(FileNotFoundError, match="Ingest a chapter first"):
        create_app(path)
    assert not path.exists()


def test_the_app_serves_the_study_page(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """GET / answers with Gradio's page, titled Study Tutor."""
    path = tmp_path / "tutor.db"
    connect(path).close()
    # create_app() reads the settings, so a key must exist; CI has no .env.
    # get_settings() keeps the first settings it reads (lru_cache), so the
    # cache is emptied before and after, and no other test sees this key.
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    get_settings.cache_clear()
    try:
        with TestClient(create_app(path)) as client:
            response = client.get("/")
    finally:
        get_settings.cache_clear()
    assert response.status_code == 200
    assert "Study Tutor" in response.text
