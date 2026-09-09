from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.responses import FileResponse, StreamingResponse

from app.config import get_settings
from crs.base import CRSModel, Turn
from crs.llm import ChatLLM

# Where the cached FAISS index lives and the metadata to build it from if absent.
_INDEX_DIR = "data/processed/movie_index"
_METADATA_PATH = "data/sample/item_map.json"
# Optional TMDB plots/genres (data/enrich.py); merged in when present.
_ENRICHMENT_PATH = "data/processed/enrichment.json"
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
        movies = list(load_movie_metadata(_METADATA_PATH, _ENRICHMENT_PATH).values())
        return Retriever.build(movies, embedder)


def _build_llm() -> ChatLLM:
    """Pick the LLM backend: the real OpenAI client if a key is configured,
    otherwise the offline FakeLLM.

    This is the swap seam. Set CRS_LLM_API_KEY in .env and the whole system —
    both approaches — starts generating real responses, with no code change.
    """
    from crs.llm import FakeLLM, OpenAILLM

    settings = get_settings()
    if settings.llm_api_key:
        return OpenAILLM(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url or None,
        )
    return FakeLLM()


def _build_rag_model() -> CRSModel:
    """Wire up Approach 1 (RAG): shared retriever + selected LLM backend."""
    from crs.rag import RAGModel

    return RAGModel(
        retriever=_load_retriever(), llm=_build_llm(), top_k=get_settings().top_k
    )


def _build_multi_agent_model() -> CRSModel:
    """Wire up Approach 2 (multi-agent): shared retriever + selected LLM backend."""
    from crs.multi_agent import MultiAgentModel

    return MultiAgentModel(
        retriever=_load_retriever(), llm=_build_llm(), top_k=get_settings().top_k
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
    """Liveness: the process is up and serving."""
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, str]:
    """Readiness: the CRS model (and its retriever/index, if any) is built.

    ``model`` is constructed at startup, so reaching this handler means the
    selected approach loaded successfully and the app can serve requests.
    """
    return {"status": "ready", "approach": get_settings().approach}
