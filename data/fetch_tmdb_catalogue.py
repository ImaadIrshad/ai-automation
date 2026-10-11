"""Build a recognizable movie catalogue for the demo from TMDB.

The LLM-Redial item catalogue is mostly obscure Amazon VHS/DVD listings, which
makes live recommendations underwhelming. This fetches the most-voted (i.e. most
widely known) real movies from TMDB — classics and hits people recognize — and
builds the FAISS index over them, so "something like Interstellar" returns names
a viewer actually knows. The pipeline is dataset-agnostic; this just swaps the
retrieval catalogue. LLM-Redial is still used for the CRS approaches and eval.

    python -m data.fetch_tmdb_catalogue          # ~3000 movies (150 pages)
    python -m data.fetch_tmdb_catalogue 200      # ~4000 movies

Requires CRS_TMDB_API_KEY. Writes a catalogue to data/processed (gitignored) and
rebuilds data/processed/movie_index.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

from app.config import get_settings
from crs.retrieval import LocalEmbedder, Retriever
from data.loader import load_catalogue

_TMDB = "https://api.themoviedb.org/3"
_CATALOGUE_PATH = "data/processed/tmdb_catalogue.json"
_INDEX_DIR = "data/processed/movie_index"


def _get_json(
    client: httpx.Client, url: str, params: dict, retries: int = 4
) -> dict | None:
    """GET returning parsed JSON, tolerant of transient network/rate errors."""
    for attempt in range(retries):
        try:
            response = client.get(url, params=params)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429:  # rate limited — back off and retry
                time.sleep(1.0 + attempt)
                continue
            return None  # a real error (bad key, 404) — don't hammer it
        except httpx.HTTPError:
            time.sleep(0.5 * (attempt + 1))  # transient; back off and retry
    return None


def fetch_catalogue(api_key: str, pages: int) -> dict[str, dict[str, str]]:
    """Fetch the most-voted movies (recognizable classics + hits) from TMDB."""
    catalogue: dict[str, dict[str, str]] = {}
    with httpx.Client(timeout=15) as client:
        genre_data = _get_json(
            client, f"{_TMDB}/genre/movie/list", {"api_key": api_key}
        )
        if not genre_data:
            return {}  # couldn't even reach TMDB
        genres = {g["id"]: g["name"] for g in genre_data.get("genres", [])}

        for page in range(1, pages + 1):
            data = _get_json(
                client,
                f"{_TMDB}/discover/movie",
                {
                    "api_key": api_key,
                    "sort_by": "vote_count.desc",  # most-rated == most-known
                    "include_adult": "false",
                    "language": "en-US",
                    "page": page,
                },
            )
            if not data:
                continue  # skip a page that failed after retries
            for movie in data.get("results", []):
                overview = movie.get("overview") or ""
                if not overview:
                    continue  # skip entries with no plot to embed
                year = (movie.get("release_date") or "")[:4]
                title = movie.get("title", "")
                catalogue[str(movie["id"])] = {
                    "title": f"{title} ({year})" if year else title,
                    "genre": "/".join(
                        genres[g] for g in movie.get("genre_ids", []) if g in genres
                    ),
                    "description": overview,
                }
            if page % 25 == 0:
                print(f"  page {page}/{pages} ({len(catalogue)} movies)")
            time.sleep(0.03)  # be polite to the API
    return catalogue


def main(pages: int = 150) -> None:
    api_key = get_settings().tmdb_api_key
    if not api_key:
        print("CRS_TMDB_API_KEY not set — add a free key to .env and rerun.")
        return

    print(f"Fetching up to {pages * 20} movies from TMDB...")
    catalogue = fetch_catalogue(api_key, pages)
    if not catalogue:
        print("No movies fetched (network or key problem). Aborting.")
        return

    out = Path(_CATALOGUE_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(catalogue))
    print(f"Saved {len(catalogue)} movies to {_CATALOGUE_PATH}")

    movies = list(load_catalogue(_CATALOGUE_PATH).values())
    print(f"Embedding and indexing {len(movies)} movies...")
    Retriever.build(movies, LocalEmbedder()).save(_INDEX_DIR)
    print(f"Index saved to {_INDEX_DIR}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
