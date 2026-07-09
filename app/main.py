from collections.abc import AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.config import get_settings
from crs.base import CRSModel, Turn


class _EchoModel(CRSModel):
    """Placeholder CRS used until a real approach is wired in."""

    async def respond(self, history: list[Turn], question: str) -> AsyncIterator[str]:
        for word in f"echo: {question}".split(" "):
            yield word + " "


def build_model(approach: str) -> CRSModel:
    """Select the CRS implementation that serves /chat, driven by config.

    This is the single switch point: registering RAG or the multi-agent model
    later means adding a branch here, never editing the endpoint. Both will
    implement the same `CRSModel` contract, so they slot in interchangeably.
    """
    if approach == "echo":
        return _EchoModel()
    # Fail loudly rather than silently serving the wrong thing.
    raise ValueError(
        f"CRS approach {approach!r} is not implemented yet; available: 'echo'"
    )


app = FastAPI(title="ai-automation CRS API")
model: CRSModel = build_model(get_settings().approach)


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatTurn] = []


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    history = [Turn(role=t.role, content=t.content) for t in request.history]

    async def stream() -> AsyncIterator[str]:
        async for chunk in model.respond(history, request.question):
            yield chunk

    return StreamingResponse(stream(), media_type="text/plain")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
