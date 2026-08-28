"""Print a summary of an LLM-Redial Movie dataset directory.

Purpose: sanity-check a dataset (real or sample) before wiring it in — how many
movies and conversations, and what one joined dialogue looks like. Run:

    python -m data.inspect                 # inspects data/sample
    python -m data.inspect data/raw/Movie  # inspects the real data
"""

import sys
from pathlib import Path

from data.loader import load_conversations, load_movie_metadata

_DEFAULT_DIR = "data/sample"


def inspect(directory: str | Path) -> None:
    """Summarise movies, conversations, and show one example dialogue."""
    path = Path(directory)
    movies = load_movie_metadata(path / "item_map.json")
    conversations = load_conversations(
        path / "final_data.jsonl", path / "Conversation.txt"
    )

    print(f"Directory: {path}")
    print(f"Movies: {len(movies)}")
    print(f"Conversations: {len(conversations)}")

    first = conversations[0]
    titles = lambda ids: [movies[i].title for i in ids if i in movies]
    print(f"\nExample conversation (id {first.conversation_id}, user {first.user_id}):")
    print(f"  liked:       {titles(first.liked_items)}")
    print(f"  disliked:    {titles(first.disliked_items)}")
    print(f"  recommended: {titles(first.recommended_items)}")
    print("  dialogue:")
    for turn in first.turns[:4]:
        print(f"    {turn.role}: {turn.content[:80]}")


if __name__ == "__main__":
    inspect(sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_DIR)
