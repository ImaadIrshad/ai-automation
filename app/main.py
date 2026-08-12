from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import FileResponse, StreamingResponse

from app.config import get_settings
from crs.base import CRSModel, Turn

# Where the cached FAISS index lives and the metadata to build it from if absent.
_INDEX_DIR = "data/processed/movie_index"
_METADATA_PATH = "data/sample/movie_metadata.json"
# The chat UI lives next to this module so it's found regardless of the cwd.
_STATIC_DIR = Path(__file__).parent / "static"


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
    if approach == "multi_agent":
        return _build_multi_agent_model()
    # Fail loudly rather than silently serving the wrong thing.
    raise ValueError(
        f"CRS approach {approach!r} is not implemented yet; "
        "available: 'echo', 'rag', 'multi_agent'"
    )


def _load_retriever():
    """Load the cached FAISS retriever, building it from sample data if absent.

    Shared by the RAG and multi-agent builders. The heavy imports are local so
    the embedding stack only loads when a retrieval-based approach is selected —
    the default 'echo' path stays lightweight.
    """
    from crs.retrieval import LocalEmbedder, Retriever
    from data.loader import load_movie_metadata

    embedder = LocalEmbedder()
    try:
        return Retriever.load(_INDEX_DIR, embedder)
    except (FileNotFoundError, RuntimeError):
        movies = list(load_movie_metadata(_METADATA_PATH).values())
        return Retriever.build(movies, embedder)


def _build_rag_model() -> CRSModel:
    """Wire up Approach 1 (RAG): shared retriever + LLM.

    FakeLLM keeps us runnable with no API key; swap for a real provider client
    (same ChatLLM interface) once a key is configured. See docs/notes.md.
    """
    from crs.llm import FakeLLM
    from crs.rag import RAGModel

    return RAGModel(retriever=_load_retriever(), llm=FakeLLM(), top_k=get_settings().top_k)


def _build_multi_agent_model() -> CRSModel:
    """Wire up Approach 2 (multi-agent): shared retriever + LLM for the agents."""
    from crs.llm import FakeLLM
    from crs.multi_agent import MultiAgentModel

    return MultiAgentModel(
        retriever=_load_retriever(), llm=FakeLLM(), top_k=get_settings().top_k
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


@app.get("/")
async def index() -> FileResponse:
    """Serve the minimal chat UI."""
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/config")
async def config() -> dict[str, str]:
    """Expose the active approach so the UI can show which model it's talking to."""
    settings = get_settings()
    return {"approach": settings.approach, "model": settings.llm_model}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
