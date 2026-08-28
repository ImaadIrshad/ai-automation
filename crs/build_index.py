"""Build and cache the FAISS movie index from a dataset's item map.

Run against the sample (default) or the real data:

    python -m crs.build_index                  # data/sample
    python -m crs.build_index data/raw/Movie   # the real LLM-Redial data

It embeds every movie with the local model, saves the index under
``data/processed/`` (gitignored), and prints a sample query so you can eyeball
that retrieval returns sensible movies.
"""

import sys

from crs.retrieval import LocalEmbedder, Retriever
from data.loader import load_movie_metadata

_DEFAULT_DIR = "data/sample"
_INDEX_DIR = "data/processed/movie_index"
_DEMO_QUERY = "a mind-bending sci-fi about dreams and reality"


def main(data_dir: str = _DEFAULT_DIR) -> None:
    movies = list(load_movie_metadata(f"{data_dir}/item_map.json").values())
    print(f"Embedding {len(movies)} movies from {data_dir} with the local model...")

    embedder = LocalEmbedder()
    retriever = Retriever.build(movies, embedder)
    retriever.save(_INDEX_DIR)
    print(f"Index saved to {_INDEX_DIR}")

    print(f'\nSample query: "{_DEMO_QUERY}"')
    for hit in retriever.search(_DEMO_QUERY, top_k=5):
        print(f"  {hit.score:.3f}  {hit.movie.title}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_DIR)
