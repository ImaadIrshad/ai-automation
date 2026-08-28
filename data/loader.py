"""Load the LLM-Redial Movie data into clean, typed Python objects.

This module is the **boundary** between the raw dataset on disk and the rest of
the system. Everything downstream (retrieval, the RAG model, the agents) talks
to these dataclasses, never to the raw files. That indirection is what lets us
point at the real data or a small sample by changing paths only — as long as this
loader emits the same objects, nothing else notices.

The real LLM-Redial Movie data (see ``docs/dataset.md``) is split across three
files, and this loader joins them:

* ``final_data.jsonl`` — one JSON object per line, keyed by user id, holding the
  structured signals (history, likes/dislikes, recommended item). **No dialogue
  text lives here.**
* ``Conversation.txt`` — the actual ``User:``/``Agent:`` dialogue, in blocks each
  headed by an integer ``conversation_id`` (0, 1, 2, ...).
* ``item_map.json`` — ``ASIN -> title`` (title only; the real data carries no
  genre/plot).

The join key is ``conversation_id``: it appears in the structured records and is
the block number in ``Conversation.txt``.
"""

import html
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from crs.base import Turn  # reuse the same turn type the CRS models consume

# Real titles carry store cruft like "[VHS]" / "(DVD)" or a trailing " VHS".
# Strip it so the title reads cleanly and matches better against movie databases.
_FORMAT_TAG = re.compile(r"\s*[\[(](?:VHS|DVD|Blu-?ray|4K|UHD)[\])]", re.IGNORECASE)
_TRAILING_FORMAT = re.compile(r"\s+(?:VHS|DVD|Blu-?ray)\s*$", re.IGNORECASE)

# The dataset labels speakers "User"/"Agent"; the CRSModel interface (and every
# LLM chat API) speaks in "user"/"assistant". Normalise once, here, so no
# downstream code has to know the dataset's vocabulary.
_ROLE_MAP = {"User": "user", "Agent": "assistant"}


@dataclass
class Movie:
    """One movie, keyed elsewhere by ``item_id`` (Amazon ASIN).

    The real dataset only provides a title; ``genre`` and ``description`` are
    optional so richer sources (or the synthetic sample) can fill them in when
    available, and retrieval degrades to title-only when they're empty.
    """

    item_id: str
    title: str
    genre: str = ""
    description: str = ""


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


def clean_title(raw: str) -> str:
    """Normalise a raw store title: decode HTML entities, drop format tags."""
    title = html.unescape(raw)
    title = _FORMAT_TAG.sub("", title)
    title = _TRAILING_FORMAT.sub("", title)
    return re.sub(r"\s+", " ", title).strip()


def load_movie_metadata(
    path: str | Path, enrichment_path: str | Path | None = None
) -> dict[str, Movie]:
    """Read the ``ASIN -> title`` map into ``Movie`` objects.

    If ``enrichment_path`` points at an existing ``ASIN -> {genre, description}``
    file (produced by ``data/enrich.py``), those fields are merged in so movie
    documents carry real plot/genre text instead of a bare title.
    """
    raw: dict[str, str] = json.loads(Path(path).read_text())

    enrichment: dict[str, dict[str, str]] = {}
    if enrichment_path is not None and Path(enrichment_path).exists():
        enrichment = json.loads(Path(enrichment_path).read_text())

    movies: dict[str, Movie] = {}
    for item_id, title in raw.items():
        extra = enrichment.get(item_id, {})
        movies[item_id] = Movie(
            item_id=item_id,
            title=clean_title(title),
            genre=extra.get("genre", ""),
            description=extra.get("description", ""),
        )
    return movies


def load_dialogues(path: str | Path) -> dict[int, list[Turn]]:
    """Parse ``Conversation.txt`` into ``{conversation_id: [Turn, ...]}``.

    Blocks are headed by a bare integer id; ``User:``/``Agent:`` lines are turns.
    A line that is neither (a wrapped continuation) is appended to the current
    turn so multi-line utterances aren't dropped.
    """
    dialogues: dict[int, list[Turn]] = {}
    current_id: int | None = None
    turns: list[Turn] = []

    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.isdigit():
            # New block: stash the previous one and start fresh.
            if current_id is not None:
                dialogues[current_id] = turns
            current_id = int(line)
            turns = []
            continue
        role = _line_role(line)
        if role is not None:
            turns.append(Turn(role=role, content=line.split(":", 1)[1].strip()))
        elif turns:
            # Continuation of the previous utterance.
            turns[-1].content = f"{turns[-1].content} {line}".strip()

    if current_id is not None:
        dialogues[current_id] = turns
    return dialogues


def load_conversations(
    structured_path: str | Path, dialogue_path: str | Path
) -> list[Conversation]:
    """Join structured records with their dialogue into ``Conversation`` objects.

    Reads ``final_data.jsonl`` line by line (memory-friendly for the full set),
    flattens the per-user ``{"conversation_N": {...}}`` nesting, and attaches the
    turns from ``Conversation.txt`` by ``conversation_id``.
    """
    dialogues = load_dialogues(dialogue_path)
    conversations: list[Conversation] = []

    for raw_line in Path(structured_path).read_text().splitlines():
        if not raw_line.strip():
            continue
        record: dict[str, dict] = json.loads(raw_line)
        for user_id, user_data in record.items():
            for wrapper in user_data.get("Conversation", []):
                # Each wrapper is {"conversation_1": {...}} — take its one value.
                for convo in wrapper.values():
                    conversation_id = convo["conversation_id"]
                    conversations.append(
                        Conversation(
                            user_id=user_id,
                            conversation_id=conversation_id,
                            turns=dialogues.get(conversation_id, []),
                            liked_items=convo.get("user_likes", []),
                            disliked_items=convo.get("user_dislikes", []),
                            recommended_items=convo.get("rec_item", []),
                        )
                    )

    return conversations


def _line_role(line: str) -> str | None:
    """Return the normalised role for a ``User:``/``Agent:`` line, else ``None``."""
    for label, role in _ROLE_MAP.items():
        if line.startswith(f"{label}:"):
            return role
    return None
