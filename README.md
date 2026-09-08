# Movie Conversational Recommender System (CRS)

A conversational movie recommender built on the **LLM-Redial** dataset, exposing
**two** LLM-based approaches — **RAG** and a **function-based multi-agent**
system — behind one streaming, async **FastAPI** endpoint, with a minimal web UI.

Both approaches implement the same contract and are swappable by a single config
value, so the API and UI never change when you switch between them.

```
                     ┌──────────────────────────────────┐
   Browser ───────▶  │  FastAPI  (app/main.py)           │
   app/static        │  GET /   POST /chat   /config ... │
                     └─────────────────┬─────────────────┘
                                       │  CRS_APPROACH picks a model
                     ┌─────────────────▼─────────────────┐
                     │  CRSModel  (crs/base.py)           │  async respond() -> token stream
                     │   ├─ RAGModel        (crs/rag.py)  │
                     │   └─ MultiAgentModel (multi_agent) │
                     └───────┬───────────────────┬────────┘
                             │                   │
                   ┌─────────▼────────┐   ┌──────▼───────────┐
                   │ Retriever (FAISS)│   │ ChatLLM          │
                   │ crs/retrieval.py │   │ Fake / OpenAI    │
                   └─────────┬────────┘   └──────────────────┘
                             │
                   ┌─────────▼──────────────┐
                   │ loader  (data/loader.py)│  LLM-Redial (real) or sample
                   └────────────────────────┘
```

For the full guided tour and design rationale, see
[`docs/architecture.md`](docs/architecture.md).

## The two approaches (and why)

- **RAG** ([`crs/rag.py`](crs/rag.py)) — retrieve the movies relevant to the
  conversation, then have the LLM recommend **only from those**, so answers are
  grounded in the real catalogue instead of hallucinated.
- **Multi-agent** ([`crs/multi_agent.py`](crs/multi_agent.py)) — three
  single-responsibility agents by *pipeline role*: **intent** (extract
  preferences) → **retrieval** (reuses the RAG retriever, filters dislikes) →
  **response** (streams the reply). Split by role, not genre — see
  [`docs/notes.md`](docs/notes.md).

Streaming is one-way HTTP (`StreamingResponse`), not WebSockets: the client only
receives the reply, so two-way complexity isn't warranted.

## Quickstart

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run it (echo model needs nothing; rag/multi_agent load the retriever):
CRS_APPROACH=rag uvicorn app.main:app
# then open http://127.0.0.1:8000/
```

Runs out of the box with **no API key and no dataset**: a small committed sample
(`data/sample/`) and an offline `FakeLLM` keep the whole pipeline exercisable.

## Turning on the real thing (two independent "dials")

| Dial | How | Effect |
| --- | --- | --- |
| **Real LLM** | put `CRS_LLM_API_KEY` in `.env` | swaps `FakeLLM` → `OpenAILLM`, real generated replies |
| **Real data** | place LLM-Redial Movie files in `data/raw/Movie/` (gitignored) and point the loader there | 9,687 real movies instead of the sample |
| **Better retrieval** | put a free `CRS_TMDB_API_KEY` in `.env`, run `python -m data.enrich data/raw/Movie` | adds plots/genres so retrieval matches meaning, not just title words |

Copy `.env.example` to `.env` to configure. Nothing secret is ever committed.

```bash
python -m data.inspect data/raw/Movie      # summarise a dataset
python -m crs.build_index data/raw/Movie   # (re)build the FAISS index
```

## Requirements

- **Python 3.10** (the pinned `numpy`/`faiss-cpu` wheels target it).
- All dependencies pinned in [`requirements.txt`](requirements.txt).

## Quality

- **PEP 8** via `ruff` and **PEP 484** type hints via `mypy` — both clean
  (`ruff check . && mypy app crs data`).
- Tests: `pytest -q` (includes a concurrency test proving overlapping requests
  don't block each other).

## Limitations & next steps

- Without the TMDB key the real catalogue is **title-only**, so retrieval matches
  title words rather than meaning — enrichment fixes this.
- `FakeLLM` is a scripted stand-in; real reasoning needs the OpenAI key.
- See [`docs/architecture.md`](docs/architecture.md) for the full list.
