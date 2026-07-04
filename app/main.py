from collections.abc import AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from crs.base import CRSModel, Turn


class _EchoModel(CRSModel):
    """Placeholder CRS used until a real approach is wired in."""

    async def respond(self, history: list[Turn], question: str) -> AsyncIterator[str]:
        for word in f"echo: {question}".split(" "):
            yield word + " "


app = FastAPI(title="ai-automation CRS API")
model: CRSModel = _EchoModel()


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
