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
from crs.retrieval import (
    RetrievedMovie,
    Retriever,
    filter_candidates,
    strip_negations,
)

_SYSTEM_PROMPT = (
    "You are a knowledgeable, friendly movie recommendation assistant. Below are "
    "relevant titles from our movie catalogue for this request — treat them as "
    "your available inventory and background knowledge, NOT as a list the user "
    "gave you.\n"
    "Guidelines:\n"
    "- Always recommend two or three of these titles that best fit the request, "
    "and say warmly why each fits. Pick the closest matches even if the fit is "
    "loose. Never refuse, never say nothing matches, never ask the user to pick.\n"
    "- Only name titles that appear below (those are the ones we actually have), "
    "but NEVER mention 'the list', 'the candidates', 'provided', or that your "
    "choices are limited. Just recommend naturally, like a film buff who knows "
    "the collection by heart.\n"
    "- Don't recommend a film the user says they've already seen; suggest other "
    "titles that share what they liked about it.\n"
    "- Reply in plain, natural, conversational prose. No Markdown: no asterisks, "
    "bold, headings, bullet points, numbered lists, or tables."
)

# Over-fetch generously: "like Dune" pulls back the whole Dune franchise, which
# we then drop — so we need plenty of other candidates left to recommend.
_OVERFETCH = 25


class RAGModel(CRSModel):
    """Retrieval-grounded conversational recommender."""

    def __init__(self, retriever: Retriever, llm: ChatLLM, top_k: int = 5) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

    async def respond(self, history: list[Turn], question: str) -> AsyncIterator[str]:
        query = _build_query(history, question)
        raw = self.retriever.search(query, top_k=self.top_k + _OVERFETCH)
        # Drop any movie named anywhere in the conversation (and its franchise),
        # so "like X" recommends films *similar to* X, never X itself — and a
        # movie named a few turns ago can't loop back as a recommendation.
        mentioned = " ".join(
            [t.content for t in history if t.role == "user"] + [question]
        )
        candidates = filter_candidates(raw, mentioned, self.top_k)
        messages = build_messages(history, question, candidates)
        async for chunk in self.llm.stream(messages):
            yield chunk


def _build_query(history: list[Turn], question: str) -> str:
    """Compose the retrieval query from recent user turns plus the new question.

    Preferences are expressed by the *user*, and often across turns ("I like
    sci-fi" ... later ... "something recent"). Including the last couple of user
    turns alongside the new question makes retrieval reflect the whole ask, not
    just the final line. Negated clauses are stripped from each segment so we
    don't retrieve the very things the user is ruling out — the full question
    still goes to the LLM, which honours the exclusion when it replies.
    """
    user_turns = [t.content for t in history if t.role == "user"]
    recent = user_turns[-2:]
    return " ".join(strip_negations(s) for s in [*recent, question])


def build_messages(
    history: list[Turn], question: str, candidates: Sequence[RetrievedMovie]
) -> list[Message]:
    """Assemble the grounded chat prompt sent to the LLM."""
    system_content = (
        f"{_SYSTEM_PROMPT}\n\nCandidate movies:\n{_format_candidates(candidates)}"
    )
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
    lines = []
    for c in candidates:
        line = f"- {c.movie.title}"
        if c.movie.genre:
            line += f" ({c.movie.genre})"
        if c.movie.description:
            line += f": {c.movie.description}"
        lines.append(line)
    return "\n".join(lines)
