import pytest

from app.config import Settings
from app.main import _EchoModel, build_model


def test_settings_defaults() -> None:
    # With no env overrides the app must boot on safe defaults.
    settings = Settings()
    assert settings.approach == "echo"
    assert settings.top_k == 5
    assert settings.llm_api_key == ""


def test_settings_read_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    # CRS_-prefixed env vars override defaults and are coerced to the field type.
    monkeypatch.setenv("CRS_APPROACH", "rag")
    monkeypatch.setenv("CRS_TOP_K", "10")
    settings = Settings()
    assert settings.approach == "rag"
    assert settings.top_k == 10  # parsed from string to int


def test_build_model_echo() -> None:
    assert isinstance(build_model("echo"), _EchoModel)


def test_build_model_unknown_raises() -> None:
    # An unimplemented approach must fail loudly, not serve the wrong model.
    with pytest.raises(ValueError):
        build_model("multi_agent")
