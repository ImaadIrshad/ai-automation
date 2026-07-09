"""Application settings, loaded once from the environment (and a local `.env`).

Why this exists: the running service should be reconfigurable by *environment*,
not by editing code. Which CRS approach serves `/chat`, which LLM model to call,
and how many documents retrieval pulls back are all operational knobs — the kind
of thing that changes between local dev, a demo, and production. Centralising
them here is also what makes the "swap RAG <-> multi-agent behind the same
endpoint" design decision real: the endpoint reads one setting instead of
hardcoding a model.

Secrets (API keys) come in the same way, from the environment, never committed.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated configuration for the CRS service.

    Every field has a safe default so the app boots with zero config. Values are
    overridden by environment variables prefixed with ``CRS_`` (e.g.
    ``CRS_APPROACH=rag``) or by entries in a local, gitignored ``.env`` file.
    Because it's a Pydantic model, a bad value (e.g. ``CRS_TOP_K=abc``) fails
    loudly at startup instead of surfacing as a confusing error deep in a
    request.
    """

    model_config = SettingsConfigDict(
        env_prefix="CRS_",
        env_file=".env",
        extra="ignore",
    )

    # Which CRS implementation is wired behind /chat. Only "echo" exists today;
    # "rag" and "multi_agent" arrive in later phases. This is the single switch
    # point so adding an approach never touches the endpoint code.
    approach: str = "echo"

    # LLM + retrieval knobs. Read now, before we need them, so later phases plug
    # in without reshaping config. `llm_model` is a placeholder name until we
    # commit to a provider in the modeling phase.
    llm_model: str = "gpt-4o-mini"
    top_k: int = 5

    # Provider API key. Blank by default and supplied only via the environment /
    # .env — it must never live in the codebase.
    llm_api_key: str = ""


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, parsed exactly once.

    The cache means we read the environment a single time rather than on every
    request, and every caller shares one consistent view of the config.
    """
    return Settings()
