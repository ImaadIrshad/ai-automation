"""Multi-agent CRS tests (offline: injected fake retriever + FakeLLM).

Covers the orchestration end-to-end, plus the two agents with logic worth
pinning down: the intent agent's JSON-parse-with-heuristic-fallback, and the
retrieval agent's disliked-title filtering.
"""

from collections.abc import AsyncIterator

import pytest

from crs.agents.intent import (
    IntentAgent,
    UserPreferences,
    _parse_preferences,
)
from crs.agents.retrieval import RetrievalAgent
from crs.base import Turn
from crs.llm import FakeLLM
from crs.multi_agent import MultiAgentModel
from crs.retrieval import RetrievedMovie
from data.loader import Movie

_INCEPTION = Movie("m1", "Inception", "Sci-Fi", "A thief enters a dream to plant an idea.")
_LALA = Movie("m2", "La La Land", "Musical", "A jazz pianist and an actress fall in love.")
_MATRIX = Movie("m3", "The Matrix", "Sci-Fi", "A hacker learns reality is a simulation.")


class _FakeRetriever:
    def __init__(self, movies: list[Movie]) -> None:
        self._movies = movies

    def search(self, query: str, top_k: int = 5) -> list[RetrievedMovie]:
        return [RetrievedMovie(m, 1.0) for m in self._movies[:top_k]]


async def _collect(stream: AsyncIterator[str]) -> str:
    return "".join([chunk async for chunk in stream])


# --- orchestration -----------------------------------------------------------


@pytest.mark.asyncio
async def test_pipeline_streams_grounded_reply() -> None:
    model = MultiAgentModel(_FakeRetriever([_INCEPTION, _MATRIX]), FakeLLM())
    text = await _collect(model.respond([], "a mind-bending sci-fi"))
    assert "Inception" in text  # the response agent grounded in a real candidate
    assert text.strip()


# --- intent agent ------------------------------------------------------------


@pytest.mark.asyncio
async def test_intent_falls_back_to_heuristics_on_non_json() -> None:
    # FakeLLM emits prose, not JSON -> the heuristic path must run and still
    # recover liked/disliked titles and genres from the user's words.
    agent = IntentAgent(FakeLLM())
    history = [
        Turn(role="user", content="I loved 'Inception'"),
        Turn(role="user", content="'La La Land' wasn't for me though"),
    ]
    prefs = await agent.extract(history, "something sci-fi please")
    assert "Inception" in prefs.liked_titles
    assert "La La Land" in prefs.disliked_titles
    assert "sci-fi" in prefs.genres
    assert prefs.free_text == "something sci-fi please"


def test_parse_preferences_valid_json() -> None:
    prefs = _parse_preferences('{"liked_titles": ["A"], "disliked_titles": [], "genres": ["drama"]}')
    assert prefs is not None
    assert prefs.liked_titles == ["A"]
    assert prefs.genres == ["drama"]


def test_parse_preferences_rejects_garbage() -> None:
    # Malformed or non-object JSON must return None so the caller can fall back.
    assert _parse_preferences("not json at all") is None
    assert _parse_preferences("[1, 2, 3]") is None


# --- retrieval agent ---------------------------------------------------------


def test_retrieval_agent_filters_disliked() -> None:
    agent = RetrievalAgent(_FakeRetriever([_INCEPTION, _LALA, _MATRIX]))
    prefs = UserPreferences(disliked_titles=["La La Land"], free_text="a movie")
    kept = agent.retrieve(prefs, top_k=5)
    titles = {c.movie.title for c in kept}
    assert "La La Land" not in titles
    assert "Inception" in titles
