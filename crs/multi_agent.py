"""Approach 2 — function-based multi-agent CRS.

One recommendation is produced by three cooperating agents, each with a single
responsibility, chained in a fixed pipeline:

    conversation
        -> IntentAgent      (extract structured preferences)      [completes]
        -> RetrievalAgent   (preferences -> movie candidates)     [completes]
        -> ResponseAgent    (candidates + chat -> reply)          [streams out]

Why split it up? Each agent has a narrow, testable job and its own focused
prompt, which is easier to reason about, debug, and improve than one giant
do-everything prompt. The agents are split by *pipeline role*, not by movie
genre — see docs/notes.md for that decision.

Data flow is plain Python: the orchestrator awaits each stage and passes its
typed output to the next. No orchestration framework (LangChain/LangGraph) is
used because a linear three-step handoff doesn't need one — it would add a
dependency and indirection without buying us anything here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from crs.agents import IntentAgent, ResponseAgent, RetrievalAgent
from crs.base import CRSModel, Turn
from crs.llm import ChatLLM
from crs.retrieval import Retriever


class MultiAgentModel(CRSModel):
    """Orchestrates intent -> retrieval -> response behind the CRSModel contract."""

    def __init__(
        self, retriever: Retriever, llm: ChatLLM, top_k: int = 5
    ) -> None:
        # The intent and response agents talk to the LLM; retrieval is pure code.
        self.intent_agent = IntentAgent(llm)
        self.retrieval_agent = RetrievalAgent(retriever)
        self.response_agent = ResponseAgent(llm)
        self.top_k = top_k

    async def respond(
        self, history: list[Turn], question: str
    ) -> AsyncIterator[str]:
        # Stage 1 + 2 run to completion (their outputs aren't shown to the user).
        preferences = await self.intent_agent.extract(history, question)
        candidates = self.retrieval_agent.retrieve(preferences, top_k=self.top_k)
        # Stage 3 is the only one that streams to the user, token by token.
        async for chunk in self.response_agent.respond(history, question, candidates):
            yield chunk
