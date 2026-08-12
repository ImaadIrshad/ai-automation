"""Intent agent — extract structured preferences from the conversation.

This is the first agent in the pipeline. Its single job: read the dialogue and
turn the user's freeform words into a small structured object (liked titles,
disliked titles, genres) that the retrieval agent can act on.

Robustness matters here. In the real path it asks the LLM to return JSON, but an
LLM can return malformed or non-JSON text. Rather than crash the whole pipeline
on one bad agent output, we validate the response and, if it isn't usable, fall
back to a deterministic heuristic extraction. So a garbage LLM reply degrades to
"slightly less rich preferences", never to a failure. (With the FakeLLM, which
doesn't emit JSON, the fallback is what runs — and it works.)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from crs.base import Turn
from crs.llm import ChatLLM, Message

# Genre keywords we can recognise deterministically in the fallback path.
_GENRES = [
    "sci-fi", "science fiction", "crime", "drama", "animation", "musical",
    "comedy", "action", "romance", "thriller", "horror", "fantasy", "music",
]
# Whole-word cues that flip a mentioned title from "liked" to "disliked".
_NEGATIVE_CUE = re.compile(
    r"\b(not|don't|didn't|wasn't|isn't|hate|dislike|too|skip|avoid|no)\b"
)
# Titles appear in the dialogue wrapped in single or double quotes.
_QUOTED = re.compile(r"'([^']+)'|\"([^\"]+)\"")


@dataclass
class UserPreferences:
    """Structured signal the retrieval agent consumes."""

    liked_titles: list[str] = field(default_factory=list)
    disliked_titles: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    free_text: str = ""  # the raw latest ask, always kept for retrieval


class IntentAgent:
    """Turns a conversation into a ``UserPreferences`` object."""

    def __init__(self, llm: ChatLLM) -> None:
        self.llm = llm

    async def extract(self, history: list[Turn], question: str) -> UserPreferences:
        # Ask the LLM first; consume its whole reply (this agent doesn't stream
        # to the user — only the final response agent does).
        raw = await _consume(self.llm.stream(_intent_prompt(history, question)))
        prefs = _parse_preferences(raw) or _heuristic_preferences(history, question)
        # The latest question always drives retrieval, regardless of path.
        prefs.free_text = question
        return prefs


def _intent_prompt(history: list[Turn], question: str) -> list[Message]:
    conversation = "\n".join(f"{t.role}: {t.content}" for t in history)
    instruction = (
        "You extract a user's movie preferences from a conversation. "
        "Respond with ONLY a JSON object with these keys: "
        "liked_titles (list of strings), disliked_titles (list of strings), "
        "genres (list of strings). Output no prose, only the JSON."
    )
    user = f"Conversation so far:\n{conversation}\nLatest message: {question}"
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user},
    ]


def _parse_preferences(raw: str) -> UserPreferences | None:
    """Parse an LLM JSON reply into preferences, or ``None`` if it's unusable."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return UserPreferences(
        liked_titles=_as_str_list(data.get("liked_titles")),
        disliked_titles=_as_str_list(data.get("disliked_titles")),
        genres=_as_str_list(data.get("genres")),
    )


def _heuristic_preferences(history: list[Turn], question: str) -> UserPreferences:
    """Deterministic fallback extraction from the user's own words.

    Intentionally simple: recognise genre keywords, and classify quoted movie
    titles as liked/disliked by whether their turn carries a negative cue. It's a
    safety net, not a replacement for the LLM's richer understanding.
    """
    liked: list[str] = []
    disliked: list[str] = []
    for turn in [*history, Turn(role="user", content=question)]:
        if turn.role != "user":
            continue
        titles = [m[0] or m[1] for m in _QUOTED.findall(turn.content)]
        if not titles:
            continue
        bucket = disliked if _NEGATIVE_CUE.search(turn.content.lower()) else liked
        bucket.extend(titles)

    all_user_text = " ".join(
        t.content for t in history if t.role == "user"
    ) + " " + question
    genres = [g for g in _GENRES if g in all_user_text.lower()]
    return UserPreferences(liked_titles=liked, disliked_titles=disliked, genres=genres)


def _as_str_list(value: object) -> list[str]:
    """Coerce an arbitrary JSON value into a clean list of strings."""
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


async def _consume(stream) -> str:
    """Drain a token stream into a single string (used for non-streamed agents)."""
    return "".join([chunk async for chunk in stream])
