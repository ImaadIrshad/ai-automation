"""Response agent — write the final, streamed recommendation.

The third and last agent. It takes the retrieved candidates plus the
conversation and streams a friendly recommendation to the user. It reuses RAG's
grounded-prompt builder (``build_messages``) so the "recommend only from these
candidates" grounding is identical across both approaches — one source of truth
for the prompt, not two.

This is the only agent whose output streams to the user; the earlier two run to
completion first.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from crs.base import Turn
from crs.llm import ChatLLM
from crs.rag import build_messages
from crs.retrieval import RetrievedMovie


class ResponseAgent:
    """Candidates + conversation in, streamed recommendation out."""

    def __init__(self, llm: ChatLLM) -> None:
        self.llm = llm

    async def respond(
        self,
        history: list[Turn],
        question: str,
        candidates: Sequence[RetrievedMovie],
    ) -> AsyncIterator[str]:
        messages = build_messages(history, question, candidates)
        async for chunk in self.llm.stream(messages):
            yield chunk
