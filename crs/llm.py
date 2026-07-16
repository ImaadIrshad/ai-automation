"""LLM client abstraction — one place for chat completion, async and streaming.

Both CRS approaches (RAG and the multi-agent system) talk to the LLM through the
``ChatLLM`` protocol below, never to a provider SDK directly. That indirection is
what lets us swap providers (OpenAI, Anthropic, a local model) without touching
the models, and — right now — run the whole pipeline with a **fake** LLM so no
API key is required.

Messages use the standard chat shape (``{"role": ..., "content": ...}``) that
OpenAI and Anthropic both accept, so a real client drops in without reshaping
anything upstream.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Protocol

Message = dict[str, str]


class ChatLLM(Protocol):
    """Anything that can stream a chat completion for a list of messages."""

    def stream(self, messages: Sequence[Message]) -> AsyncIterator[str]: ...


class FakeLLM:
    """Deterministic, offline stand-in for a real LLM (no API key needed).

    It doesn't *generate* language — it streams a fixed, context-aware reply that
    names the top retrieved movie, so the end-to-end pipeline (retrieval ->
    grounded prompt -> streaming) is fully exercisable and demonstrable. Swap it
    for a real client to get genuine generation; nothing else changes.
    """

    async def stream(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        title = _first_candidate_title(messages)
        if title:
            reply = (
                f'Based on what you\'re looking for, I\'d suggest "{title}". '
                "It lines up well with the preferences you described. "
                "Want something in a similar vein?"
            )
        else:
            # Graceful path when retrieval found nothing to ground on.
            reply = (
                "I couldn't find a good match in the catalogue for that. "
                "Could you tell me a bit more about what you enjoy?"
            )

        # Stream word-by-word to mimic a real token stream. The sleep(0) yields
        # control to the event loop so a slow "generation" can't starve other
        # concurrent requests.
        for word in reply.split(" "):
            await asyncio.sleep(0)
            yield word + " "


def _first_candidate_title(messages: Sequence[Message]) -> str | None:
    """Pull the first retrieved movie title out of the grounded prompt.

    The RAG prompt lists candidates as ``- Title (Genre): description`` lines in a
    system message; the fake reads the first one so its reply feels grounded.
    """
    for message in messages:
        for line in message.get("content", "").splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                # "- Inception (Sci-Fi): ..." -> "Inception"
                return stripped[2:].split(" (")[0].strip()
    return None
