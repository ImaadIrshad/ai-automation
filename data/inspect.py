"""Print the structure of an LLM-Redial conversation file.

Purpose: the build plan says *don't assume the schema — inspect it*. When the
real approved data arrives, run this against it to see whether its shape matches
what ``loader.py`` expects (especially how the dialogue turns are nested, which
is the one part we inferred). Run:

    python -m data.inspect                       # inspects the synthetic sample
    python -m data.inspect data/raw/movie.json   # inspects the real data
"""

import json
import sys
from pathlib import Path

_DEFAULT_PATH = "data/sample/movie_sample.json"


def inspect(path: str | Path) -> None:
    """Summarise users, conversations, fields, and show one example dialogue."""
    raw: dict[str, dict] = json.loads(Path(path).read_text())

    print(f"File: {path}")
    print(f"Users: {len(raw)}")

    total_convos = 0
    user_level_fields: set[str] = set()
    convo_level_fields: set[str] = set()

    for user_data in raw.values():
        user_level_fields.update(user_data.keys())
        for wrapper in user_data.get("Conversation", []):
            for convo in wrapper.values():
                total_convos += 1
                convo_level_fields.update(convo.keys())

    print(f"Conversations: {total_convos}")
    print(f"User-level fields: {sorted(user_level_fields)}")
    print(f"Conversation-level fields: {sorted(convo_level_fields)}")

    # Show the first dialogue so the turn structure is visible at a glance.
    first_user = next(iter(raw))
    first_wrapper = raw[first_user]["Conversation"][0]
    first_convo = next(iter(first_wrapper.values()))
    print(f"\nExample dialogue (user {first_user}):")
    for turn in first_convo.get("content", []):
        print(f"  {turn['role']}: {turn['text']}")


if __name__ == "__main__":
    inspect(sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_PATH)
