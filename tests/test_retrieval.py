"""Retriever tests.

These use a tiny deterministic ``KeywordEmbedder`` instead of the real neural
model so the suite stays fast and offline (no torch, no model download). It
embeds text as counts over a fixed keyword vocabulary — crude, but enough to
prove the plumbing: the index builds, search returns the right shape, respects
top_k, and ranks an obviously-relevant movie first.
"""

import numpy as np

from crs.retrieval import Retriever, movie_to_document
from data.loader import Movie

_VOCAB = ["dream", "space", "crime", "animation", "music", "alien"]


class KeywordEmbedder:
    """Deterministic bag-of-keywords embedder (no external deps)."""

    def __init__(self) -> None:
        self.dim = len(_VOCAB)

    def embed(self, texts) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            lowered = text.lower()
            for col, word in enumerate(_VOCAB):
                vectors[row, col] = lowered.count(word)
        # L2-normalise so inner product behaves like cosine (skip zero rows).
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


_MOVIES = [
    Movie("m1", "Inception", "Sci-Fi", "A thief enters a dream to plant an idea."),
    Movie("m2", "Interstellar", "Sci-Fi", "Explorers travel through space to save humanity."),
    Movie("m3", "The Godfather", "Crime", "A crime family patriarch hands over his empire."),
    Movie("m4", "Spirited Away", "Animation", "An animation about a girl in a spirit world."),
    Movie("m5", "Whiplash", "Music", "A drummer chases greatness in music."),
]


def _retriever() -> Retriever:
    return Retriever.build(_MOVIES, KeywordEmbedder())


def test_movie_to_document_includes_title_and_genre() -> None:
    doc = movie_to_document(_MOVIES[0])
    assert "Inception" in doc and "Sci-Fi" in doc


def test_search_respects_top_k() -> None:
    results = _retriever().search("space", top_k=3)
    assert len(results) == 3


def test_search_ranks_relevant_movie_first() -> None:
    # "dream" should surface Inception ahead of everything else.
    results = _retriever().search("a dream within a dream", top_k=1)
    assert results[0].movie.title == "Inception"


def test_search_caps_at_catalogue_size() -> None:
    # Asking for more than we have must not error or pad with junk.
    results = _retriever().search("crime", top_k=99)
    assert len(results) == len(_MOVIES)


def test_save_and_load_roundtrip(tmp_path) -> None:
    built = _retriever()
    built.save(tmp_path)
    reloaded = Retriever.load(tmp_path, KeywordEmbedder())
    assert [m.item_id for m in reloaded.movies] == [m.item_id for m in _MOVIES]
    assert reloaded.search("space", top_k=1)[0].movie.title == "Interstellar"
