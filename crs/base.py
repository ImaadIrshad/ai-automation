"""Common interface for the CRS approaches implemented in this repo."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass
class Turn:
    role: str  # "user" or "assistant"
    content: str


class CRSModel(ABC):
    """A conversational recommender that streams a reply token-by-token."""

    @abstractmethod
    async def respond(
        self, history: list[Turn], question: str
    ) -> AsyncIterator[str]:
        """Yield response chunks for `question` given prior `history`."""
        raise NotImplementedError
        yield  # pragma: no cover - makes this an async generator for type checkers
