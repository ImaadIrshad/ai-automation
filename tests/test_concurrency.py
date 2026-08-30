"""Prove /chat is genuinely concurrent: one slow request must not block another.

The brief calls out performance — this test demonstrates the async design pays
off. Two overlapping streaming requests to a deliberately slow model should
finish in roughly the time of *one*, not the sum of both, because each `await`
inside the stream yields the event loop to the other request.
"""

import asyncio
import time
from collections.abc import AsyncIterator

import httpx
import pytest
from httpx import ASGITransport

import app.main as main
from crs.base import CRSModel, Turn

_STEP = 0.05  # seconds per streamed chunk
_CHUNKS = 3
_ONE_STREAM = _STEP * _CHUNKS  # ~0.15s for a single request


class _SlowModel(CRSModel):
    """Streams a few chunks with a real await between each (simulates latency)."""

    async def respond(
        self, history: list[Turn], question: str
    ) -> AsyncIterator[str]:
        for i in range(_CHUNKS):
            await asyncio.sleep(_STEP)
            yield f"{question}{i} "


@pytest.mark.asyncio
async def test_chat_requests_run_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "model", _SlowModel())
    transport = ASGITransport(app=main.app)

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:

        async def call(q: str) -> str:
            response = await client.post("/chat", json={"question": q, "history": []})
            return response.text

        start = time.perf_counter()
        first, second = await asyncio.gather(call("a"), call("b"))
        elapsed = time.perf_counter() - start

    assert first.strip() and second.strip()  # both completed
    # Serial handling would take ~2 * _ONE_STREAM; concurrent stays close to one.
    assert elapsed < _ONE_STREAM * 1.8
