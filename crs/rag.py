"""Approach 1 — Retrieval-Augmented Generation (RAG) CRS.

Flow for each turn:

1. Build a retrieval query from the conversation (what the user is asking for).
2. Retrieve the top-k most relevant movies from the FAISS index.
3. Construct a *grounded* prompt: system instructions + the retrieved movies as
   context + the conversation so far + the new question.
4. Stream the LLM's reply back token-by-token through the async ``respond``
   contract.

The point of steps 2-3 is grounding: the model is told to recommend only from
the retrieved candidates, so it points at real movies from our catalogue instead
of hallucinating plausible-sounding ones.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from crs.base import CRSModel, Turn
from crs.llm import ChatLLM, Message
from crs.retrieval import RetrievedMovie, Retriever

_SYSTEM_PROMPT = (
    "You are a helpful movie recommender. Recommend ONLY from the candidate "
    "movies listed below, grounding every suggestion in their titles and "
    "details. If none of the candidates fit the user's request, say so honestly "
    "rather than inventing a movie."
)


class RAGModel(CRSModel):
    """Retrieval-grounded conversational recommender."""

    def __init__(
        self, retriever: Retriever, llm: ChatLLM, top_k: int = 5
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    async def respond(
        self, history: list[Turn], question: str
    ) -> AsyncIterator[str]:
        query = _build_query(history, question)
        candidates = self.retriever.search(query, top_k=self.top_k)
        messages = build_messages(history, question, candidates)
        async for chunk in self.llm.stream(messages):
            yield chunk


def _build_query(history: list[Turn], question: str) -> str:
    """Compose the retrieval query from recent user turns plus the new question.

    Preferences are expressed by the *user*, and often across turns ("I like
    sci-fi" ... later ... "something recent"). Including the last couple of user
    turns alongside the new question makes retrieval reflect the whole ask, not
    just the final line.
    """
    user_turns = [t.content for t in history if t.role == "user"]
    recent = user_turns[-2:]
    return " ".join([*recent, question])


def build_messages(
    history: list[Turn], question: str, candidates: Sequence[RetrievedMovie]
) -> list[Message]:
    """Assemble the grounded chat prompt sent to the LLM."""
    system_content = f"{_SYSTEM_PROMPT}\n\nCandidate movies:\n{_format_candidates(candidates)}"
    messages: list[Message] = [{"role": "system", "content": system_content}]
    # Replay the conversation so the model has the full context, then the new
    # question as the final user turn.
    messages.extend({"role": turn.role, "content": turn.content} for turn in history)
    messages.append({"role": "user", "content": question})
    return messages


def _format_candidates(candidates: Sequence[RetrievedMovie]) -> str:
    """Render retrieved movies as the context block for the prompt."""
    if not candidates:
        # Explicit signal so the model (and the fake) can degrade gracefully.
        return "(no matching movies found)"
    return "\n".join(
        f"- {c.movie.title} ({c.movie.genre}): {c.movie.description}"
        for c in candidates
    )
