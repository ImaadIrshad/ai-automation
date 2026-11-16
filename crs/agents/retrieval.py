"""Retrieval agent — turn structured preferences into movie candidates.

This is the second agent. It's pure code (no LLM): it builds a query string from
the intent agent's output, calls the shared FAISS ``Retriever`` (the *same* one
RAG uses — retrieval isn't duplicated across approaches), and drops the movies the
user named (and their franchise) before handing candidates to the response agent.
"""

from __future__ import annotations

from crs.agents.intent import UserPreferences
from crs.retrieval import (
    RetrievedMovie,
    Retriever,
    filter_candidates,
    strip_negations,
)

# Over-fetch generously so a full top_k remains after dropping named movies.
_OVERFETCH = 25


class RetrievalAgent:
    """Preferences in, ranked movie candidates out."""

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever

    def retrieve(
        self, preferences: UserPreferences, top_k: int = 5
    ) -> list[RetrievedMovie]:
        query = _preferences_to_query(preferences)
        raw = self.retriever.search(query, top_k=top_k + _OVERFETCH)
        # Exclude the movies the user named (liked or disliked) and their
        # franchises — recommend things *like* them, not them (same rule as RAG).
        mentioned = " ".join(
            [
                preferences.free_text,
                *preferences.liked_titles,
                *preferences.disliked_titles,
            ]
        )
        return filter_candidates(raw, mentioned, top_k)


def _preferences_to_query(preferences: UserPreferences) -> str:
    """Compose a retrieval query from the structured preferences.

    Genres and liked titles sharpen the search; the raw free text keeps whatever
    nuance the structured fields missed, with negated clauses stripped so we
    don't retrieve what the user is ruling out. Disliked titles are handled by
    filtering (below), not by putting them in the query — you don't want to
    *retrieve* things similar to what the user rejected.
    """
    free_text = strip_negations(preferences.free_text)
    parts = [*preferences.genres, *preferences.liked_titles, free_text]
    return " ".join(part for part in parts if part).strip()
