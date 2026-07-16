from collections.abc import AsyncIterator

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from app.config import get_settings
from crs.base import CRSModel, Turn

# Where the cached FAISS index lives and the metadata to build it from if absent.
_INDEX_DIR = "data/processed/movie_index"
_METADATA_PATH = "data/sample/movie_metadata.json"


class _EchoModel(CRSModel):
    """Placeholder CRS used until a real approach is wired in."""

    async def respond(self, history: list[Turn], question: str) -> AsyncIterator[str]:
        for word in f"echo: {question}".split(" "):
            yield word + " "


def build_model(approach: str) -> CRSModel:
    """Select the CRS implementation that serves /chat, driven by config.

    This is the single switch point: registering the multi-agent model later
    means adding a branch here, never editing the endpoint. Every approach
    implements the same `CRSModel` contract, so they slot in interchangeably.
    """
    if approach == "echo":
        return _EchoModel()
    if approach == "rag":
        return _build_rag_model()
    # Fail loudly rather than silently serving the wrong thing.
    raise ValueError(
        f"CRS approach {approach!r} is not implemented yet; "
        "available: 'echo', 'rag'"
    )


def _build_rag_model() -> CRSModel:
    """Wire up the RAG model: retriever (cached index if present) + LLM.

    Imports are local so the heavy retrieval/embedding stack only loads when RAG
    is actually selected — the default 'echo' path stays lightweight.
    """
    from crs.llm import FakeLLM
    from crs.rag import RAGModel
    from crs.retrieval import LocalEmbedder, Retriever
    from data.loader import load_movie_metadata

    settings = get_settings()
    embedder = LocalEmbedder()

    # Prefer the cached index; fall back to building it from the sample metadata
    # so a fresh clone still works without a separate build step.
    try:
        retriever = Retriever.load(_INDEX_DIR, embedder)
    except (FileNotFoundError, RuntimeError):
        movies = list(load_movie_metadata(_METADATA_PATH).values())
        retriever = Retriever.build(movies, embedder)

    # FakeLLM keeps us runnable with no API key; swap for a real provider client
    # (same ChatLLM interface) once a key is configured. See docs/notes.md.
    return RAGModel(retriever=retriever, llm=FakeLLM(), top_k=settings.top_k)


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
