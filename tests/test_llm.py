"""Tests for LLM backend selection.

The real client isn't called (no network) — we only verify the *selection* logic:
no key -> FakeLLM, key present -> OpenAILLM. We patch the settings accessor with
an explicit, .env-isolated Settings so a developer's real key can't affect it.
"""

import pytest

import app.main as main
from app.config import Settings
from crs.llm import FakeLLM, OpenAILLM


def _settings(**overrides: object) -> Settings:
    # _env_file=None isolates from any local .env; explicit kwargs win over env.
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_falls_back_to_fake_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "get_settings", lambda: _settings(llm_api_key=""))
    assert isinstance(main._build_llm(), FakeLLM)


def test_uses_openai_when_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        main, "get_settings", lambda: _settings(llm_api_key="sk-dummy-not-real")
    )
    assert isinstance(main._build_llm(), OpenAILLM)
