"""Build and cache the FAISS movie index from the dataset metadata.

Run this once (and again whenever the movie catalogue changes):

    python -m crs.build_index

It loads the movie metadata, embeds every movie with the local model, saves the
index under ``data/processed/`` (gitignored), and prints a sample query so you
can eyeball that retrieval returns sensible movies.
"""

from data.loader import load_movie_metadata
from crs.retrieval import LocalEmbedder, Retriever

_METADATA_PATH = "data/sample/movie_metadata.json"
_INDEX_DIR = "data/processed/movie_index"
_DEMO_QUERY = "a mind-bending sci-fi about dreams and reality"


def main() -> None:
    movies = list(load_movie_metadata(_METADATA_PATH).values())
    print(f"Embedding {len(movies)} movies with the local model...")

    embedder = LocalEmbedder()
    retriever = Retriever.build(movies, embedder)
    retriever.save(_INDEX_DIR)
    print(f"Index saved to {_INDEX_DIR}")

    print(f'\nSample query: "{_DEMO_QUERY}"')
    for hit in retriever.search(_DEMO_QUERY, top_k=3):
        print(f"  {hit.score:.3f}  {hit.movie.title} ({hit.movie.genre})")


if __name__ == "__main__":
    main()
