"""Tests for LLM backend selection.

The real OpenAI client isn't called (no network, no key) — we only verify the
*selection* logic: no key -> FakeLLM, key present -> OpenAILLM. Constructing
OpenAILLM with a dummy key is offline (the SDK just stores config).
"""

import pytest

from app.config import get_settings
from app.main import _build_llm
from crs.llm import FakeLLM, OpenAILLM


def test_falls_back_to_fake_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CRS_LLM_API_KEY", raising=False)
    get_settings.cache_clear()  # settings are cached; force a re-read
    try:
        assert isinstance(_build_llm(), FakeLLM)
    finally:
        get_settings.cache_clear()


def test_uses_openai_when_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CRS_LLM_API_KEY", "sk-dummy-not-a-real-key")
    get_settings.cache_clear()
    try:
        assert isinstance(_build_llm(), OpenAILLM)
    finally:
        get_settings.cache_clear()  # don't leak the dummy key to other tests
