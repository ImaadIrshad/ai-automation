"""RAGModel tests.

These inject a fake retriever and the FakeLLM so the whole flow (query -> context
-> grounded prompt -> stream) runs offline, with no embeddings and no API key.
"""

from collections.abc import AsyncIterator, Sequence

import pytest

from crs.base import Turn
from crs.llm import FakeLLM
from crs.rag import RAGModel, build_messages
from crs.retrieval import RetrievedMovie
from data.loader import Movie

_INCEPTION = Movie("m1", "Inception", "Sci-Fi", "A thief enters a dream to plant an idea.")
_MATRIX = Movie("m2", "The Matrix", "Sci-Fi", "A hacker learns reality is a simulation.")


class _FakeRetriever:
    """Returns a fixed candidate list, decoupling RAGModel from real embeddings."""

    def __init__(self, movies: list[Movie]) -> None:
        self._movies = movies

    def search(self, query: str, top_k: int = 5) -> list[RetrievedMovie]:
        return [RetrievedMovie(m, 1.0) for m in self._movies[:top_k]]


async def _collect(stream: AsyncIterator[str]) -> str:
    return "".join([chunk async for chunk in stream])


@pytest.mark.asyncio
async def test_respond_grounds_in_retrieved_movie() -> None:
    model = RAGModel(_FakeRetriever([_INCEPTION, _MATRIX]), FakeLLM())
    text = await _collect(model.respond([], "a dream heist movie"))
    assert "Inception" in text  # names the top retrieved candidate
    assert text.strip()  # non-empty stream


@pytest.mark.asyncio
async def test_respond_handles_empty_retrieval() -> None:
    model = RAGModel(_FakeRetriever([]), FakeLLM())
    text = await _collect(model.respond([], "something obscure"))
    # Degrades gracefully instead of crashing or naming a nonexistent movie.
    assert text.strip()
    assert "Inception" not in text


def test_build_messages_structure() -> None:
    history = [
        Turn(role="user", content="I like sci-fi"),
        Turn(role="assistant", content="Great, any preferences?"),
    ]
    candidates = [RetrievedMovie(_INCEPTION, 1.0)]
    messages = build_messages(history, "recommend something", candidates)

    # System message carries the grounding context...
    assert messages[0]["role"] == "system"
    assert "Inception" in messages[0]["content"]
    # ...history is replayed...
    assert {"role": "user", "content": "I like sci-fi"} in messages
    # ...and the new question is the final user turn.
    assert messages[-1] == {"role": "user", "content": "recommend something"}
