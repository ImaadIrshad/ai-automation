"""Enrich the movie catalogue with plots/genres from TMDB.

The real ``item_map`` gives only titles, so title embeddings match words, not
meaning ("dreams" -> Nightmare on Elm Street). This script looks each movie up on
TMDB (free API) and caches its genre + overview, which retrieval then embeds for
genuine semantic search.

    python -m data.enrich                 # enrich data/sample
    python -m data.enrich data/raw/Movie  # enrich the real catalogue

Requires ``CRS_TMDB_API_KEY`` in the environment / .env. Output is written to
``data/processed/enrichment.json`` (gitignored) and is **resumable** — rerun to
continue where it stopped.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

from app.config import get_settings
from data.loader import clean_title

_TMDB = "https://api.themoviedb.org/3"
_OUT = "data/processed/enrichment.json"


def parse_search_result(
    result: dict, genre_map: dict[int, str]
) -> dict[str, str]:
    """Turn a TMDB search hit into our ``{genre, description}`` shape.

    Pure and network-free so it can be unit-tested without a key.
    """
    genres = [genre_map.get(gid, "") for gid in result.get("genre_ids", [])]
    return {
        "genre": "/".join(g for g in genres if g),
        "description": result.get("overview", "") or "",
    }


def _genre_map(client: httpx.Client, api_key: str) -> dict[int, str]:
    response = client.get(f"{_TMDB}/genre/movie/list", params={"api_key": api_key})
    response.raise_for_status()
    return {g["id"]: g["name"] for g in response.json().get("genres", [])}


def _search(client: httpx.Client, api_key: str, title: str) -> dict | None:
    response = client.get(
        f"{_TMDB}/search/movie", params={"api_key": api_key, "query": title}
    )
    if response.status_code != 200:
        return None
    results = response.json().get("results", [])
    return results[0] if results else None  # top hit is TMDB's best match


def main(data_dir: str = "data/sample") -> None:
    api_key = get_settings().tmdb_api_key
    if not api_key:
        print(
            "CRS_TMDB_API_KEY not set. Get a free key at "
            "https://www.themoviedb.org (Settings -> API), add it to .env, and rerun."
        )
        return

    item_map: dict[str, str] = json.loads(
        Path(f"{data_dir}/item_map.json").read_text()
    )
    out_path = Path(_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume: keep whatever we've already fetched.
    enrichment: dict[str, dict] = (
        json.loads(out_path.read_text()) if out_path.exists() else {}
    )

    with httpx.Client(timeout=15) as client:
        genre_map = _genre_map(client, api_key)
        todo = [(a, t) for a, t in item_map.items() if a not in enrichment]
        print(f"Enriching {len(todo)} of {len(item_map)} movies (resuming)...")

        for i, (asin, raw_title) in enumerate(todo, start=1):
            hit = _search(client, api_key, clean_title(raw_title))
            enrichment[asin] = (
                parse_search_result(hit, genre_map)
                if hit
                else {"genre": "", "description": ""}
            )
            if i % 100 == 0:
                out_path.write_text(json.dumps(enrichment))  # checkpoint
                print(f"  {i}/{len(todo)}")
            time.sleep(0.03)  # be polite to the API

    out_path.write_text(json.dumps(enrichment))
    print(f"Saved enrichment for {len(enrichment)} movies to {_OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/sample")
