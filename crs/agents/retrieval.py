"""Retrieval agent — turn structured preferences into movie candidates.

This is the second agent. It's pure code (no LLM): it builds a query string from
the intent agent's output, calls the shared FAISS ``Retriever`` (the *same* one
RAG uses — retrieval isn't duplicated across approaches), and drops any movie the
user explicitly disliked before handing candidates to the response agent.
"""

from __future__ import annotations

from crs.agents.intent import UserPreferences
from crs.retrieval import RetrievedMovie, Retriever


class RetrievalAgent:
    """Preferences in, ranked movie candidates out."""

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def retrieve(
        self, preferences: UserPreferences, top_k: int = 5
    ) -> list[RetrievedMovie]:
        query = _preferences_to_query(preferences)
        # Over-fetch a little so that filtering out disliked titles still leaves
        # us with a full top_k of usable candidates.
        candidates = self.retriever.search(query, top_k=top_k + len(preferences.disliked_titles))
        disliked = {title.lower() for title in preferences.disliked_titles}
        kept = [c for c in candidates if c.movie.title.lower() not in disliked]
        return kept[:top_k]


def _preferences_to_query(preferences: UserPreferences) -> str:
    """Compose a retrieval query from the structured preferences.

    Genres and liked titles sharpen the search; the raw free text keeps whatever
    nuance the structured fields missed. Disliked titles are handled by filtering
    (below), not by putting them in the query — you don't want to *retrieve*
    things similar to what the user rejected.
    """
    parts = [*preferences.genres, *preferences.liked_titles, preferences.free_text]
    return " ".join(part for part in parts if part).strip()
