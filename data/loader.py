"""Load the LLM-Redial Movie data into clean, typed Python objects.

This module is the **boundary** between the raw dataset on disk and the rest of
the system. Everything downstream (retrieval, the RAG model, the agents) talks
to these dataclasses, never to the raw JSON. That indirection is what lets us
swap the synthetic sample for the approved real data by changing a path: as long
as this loader emits the same objects, nothing else notices.

The raw schema is documented in ``docs/dataset.md``. Two shapes are read:

* the **conversation file** — keyed by user id, each user has structured
  annotations (likes / dislikes / recommended item) plus User/Agent dialogue;
* the **movie metadata file** — item id (Amazon ASIN) -> title, genre, plot,
  standing in for Amazon product metadata.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from crs.base import Turn  # reuse the same turn type the CRS models consume

# The dataset labels speakers "User"/"Agent"; the CRSModel interface (and every
# LLM chat API) speaks in "user"/"assistant". Normalise once, here, so no
# downstream code has to know the dataset's vocabulary.
_ROLE_MAP = {"User": "user", "Agent": "assistant"}


@dataclass
class Movie:
    """One movie's human-readable metadata, keyed elsewhere by ``item_id``."""

    item_id: str
    title: str
    genre: str
    description: str


@dataclass
class Conversation:
    """A single dialogue plus the ground-truth recommendation signals.

    ``recommended_items`` is what a good CRS should surface — it's the label we
    evaluate against later. ``liked_items`` / ``disliked_items`` are the
    preference signals expressed during the chat.
    """

    user_id: str
    conversation_id: int
    turns: list[Turn]
    liked_items: list[str] = field(default_factory=list)
    disliked_items: list[str] = field(default_factory=list)
    recommended_items: list[str] = field(default_factory=list)


def load_movie_metadata(path: str | Path) -> dict[str, Movie]:
    """Read the ASIN -> movie metadata map into ``Movie`` objects."""
    raw: dict[str, dict[str, str]] = json.loads(Path(path).read_text())
    return {
        item_id: Movie(
            item_id=item_id,
            title=fields["title"],
            genre=fields["genre"],
            description=fields["description"],
        )
        for item_id, fields in raw.items()
    }


def load_conversations(path: str | Path) -> list[Conversation]:
    """Read the conversation file into a flat list of ``Conversation`` objects.

    The raw file nests conversations two levels deep (per-user, then a
    ``{"conversation_N": {...}}`` wrapper). We flatten that here so callers get a
    simple list and never have to walk the wrapper keys.
    """
    raw: dict[str, dict] = json.loads(Path(path).read_text())
    conversations: list[Conversation] = []

    for user_id, user_data in raw.items():
        for wrapper in user_data.get("Conversation", []):
            # Each wrapper is {"conversation_1": {...}} — one key we don't care
            # about by name, so take its single value.
            for convo in wrapper.values():
                turns = [
                    Turn(role=_normalize_role(t["role"]), content=t["text"])
                    for t in convo.get("content", [])
                ]
                conversations.append(
                    Conversation(
                        user_id=user_id,
                        conversation_id=convo["conversation_id"],
                        turns=turns,
                        liked_items=convo.get("user_likes", []),
                        disliked_items=convo.get("user_dislikes", []),
                        recommended_items=convo.get("rec_item", []),
                    )
                )

    return conversations


def _normalize_role(role: str) -> str:
    """Map dataset speaker labels to the "user"/"assistant" convention.

    Unknown labels pass through unchanged rather than being dropped, so a schema
    surprise in the real data is visible instead of silently swallowed.
    """
    return _ROLE_MAP.get(role, role.lower())
