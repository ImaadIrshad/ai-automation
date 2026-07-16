"""Movie retrieval for the RAG pipeline.

The job here: given what a user is looking for ("a mind-bending sci-fi about
dreams"), find the most relevant movies from our catalogue *fast*, so the LLM can
ground its recommendation in real titles instead of inventing them.

How: turn every movie into a short text blurb, convert each blurb into a vector
(a list of numbers that captures its meaning), and store those vectors in a FAISS
index built for fast nearest-neighbour search. At query time we embed the query
the same way and ask the index for the closest movie vectors.

The embedding backend is deliberately behind the ``Embedder`` protocol so it can
be swapped (local model today, OpenAI later) without changing the ``Retriever`` —
the same "swap the implementation, keep the contract" idea as the CRS models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import faiss
import numpy as np

from data.loader import Movie


class Embedder(Protocol):
    """Turns text into fixed-length, L2-normalised vectors.

    ``dim`` is the vector length; ``embed`` maps N texts to an (N, dim) float32
    array. Normalisation matters because we use inner-product search, and inner
    product on normalised vectors *is* cosine similarity — the standard "how
    close in meaning" measure.
    """

    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class LocalEmbedder:
    """Local sentence-transformers embedder: real semantic vectors, no API key.

    The heavy import is done lazily inside ``__init__`` so that merely importing
    this module (e.g. in fast unit tests that inject a fake embedder) doesn't pay
    the cost of loading torch / the model.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.dim: int = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,  # so inner product == cosine similarity
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype="float32")


@dataclass
class RetrievedMovie:
    """A movie returned by search, with its similarity score (higher = closer)."""

    movie: Movie
    score: float


def movie_to_document(movie: Movie) -> str:
    """Render a movie as the single text blurb we embed and index."""
    return f"{movie.title} ({movie.genre}). {movie.description}"


class Retriever:
    """A FAISS index over movie blurbs, plus the movies it indexes.

    Build once with :meth:`build`, optionally :meth:`save` it to disk, and reload
    with :meth:`load` so we don't re-embed the whole catalogue on every run.
    """

    _INDEX_FILE = "movies.faiss"
    _MOVIES_FILE = "movies.json"

    def __init__(
        self, embedder: Embedder, index: faiss.Index, movies: list[Movie]
    ) -> None:
        self.embedder = embedder
        self.index = index
        # Row i of the index corresponds to movies[i]; keep them aligned.
        self.movies = movies

    @classmethod
    def build(cls, movies: list[Movie], embedder: Embedder) -> "Retriever":
        """Embed every movie blurb and load the vectors into a fresh index."""
        documents = [movie_to_document(m) for m in movies]
        vectors = embedder.embed(documents)
        # IndexFlatIP = exact inner-product search. "Flat" (brute force) is the
        # right call at this scale; swap for an approximate index only if the
        # catalogue grows into the millions.
        index = faiss.IndexFlatIP(embedder.dim)
        index.add(vectors)
        return cls(embedder, index, movies)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedMovie]:
        """Return the ``top_k`` movies most similar to ``query``."""
        query_vector = self.embedder.embed([query])
        # Never ask for more neighbours than we have movies.
        k = min(top_k, len(self.movies))
        scores, indices = self.index.search(query_vector, k)
        return [
            RetrievedMovie(movie=self.movies[idx], score=float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1  # FAISS pads with -1 if fewer than k results exist
        ]

    def save(self, directory: str | Path) -> None:
        """Persist the index and movie list so we can reload without re-embedding."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / self._INDEX_FILE))
        (path / self._MOVIES_FILE).write_text(
            json.dumps([m.__dict__ for m in self.movies])
        )

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder) -> "Retriever":
        """Reload a previously :meth:`save`d index. The embedder must match the
        one used to build it (same model, same dimension)."""
        path = Path(directory)
        index = faiss.read_index(str(path / cls._INDEX_FILE))
        movies = [
            Movie(**data)
            for data in json.loads((path / cls._MOVIES_FILE).read_text())
        ]
        return cls(embedder, index, movies)
