# Architecture & Walkthrough

A guided tour of the system: what each piece does, why it's shaped that way, and
where to look. Reads top-down — the request lifecycle first, then a file-by-file
tour, then the design decisions, then the questions this design invites.

## 1. The request lifecycle

What happens when the browser sends a message:

1. **`POST /chat`** ([`app/main.py`](../app/main.py)) receives `{question,
   history}`, validated by a Pydantic model.
2. The handler calls `model.respond(history, question)` on the **active
   `CRSModel`**, chosen at startup from `CRS_APPROACH` (`echo` | `rag` |
   `multi_agent`).
3. `respond()` is an **async generator** that yields the reply in chunks. The
   handler wraps it in a `StreamingResponse`, so chunks flush to the client as
   they're produced.
4. For **RAG**: build a retrieval query from the conversation → FAISS returns the
   top-k movies → assemble a grounded prompt → the LLM streams the reply.
5. For **multi-agent**: an intent agent extracts preferences → a retrieval agent
   turns them into candidates (reusing the RAG retriever) → a response agent
   streams the reply.

The two approaches are interchangeable because both implement the same
`CRSModel.respond` contract.

## 2. File-by-file tour

### The contract — `crs/base.py`
The heart of the design: `CRSModel`, an abstract base class with one method,
`async respond(history, question) -> AsyncIterator[str]`. Every approach
implements it, so they're swappable behind the API. Also defines `Turn`
(`role`, `content`) — the one conversation-turn type used everywhere.

### Serving — `app/`
- **`main.py`** — the FastAPI app. `build_model()` is the single switch that maps
  `CRS_APPROACH` to a model. `_build_llm()` picks the LLM backend (real OpenAI if
  a key is set, else `FakeLLM`). `_load_retriever()` loads the cached FAISS index
  (or builds it from the sample). Endpoints: `/` (UI), `/chat` (stream),
  `/config`, `/health` (liveness), `/ready` (readiness).
- **`config.py`** — typed settings via `pydantic-settings`, read from `CRS_`-
  prefixed env vars / `.env`, parsed once (`get_settings()` is cached). Bad
  values fail at startup, not mid-request.
- **`static/index.html`** — a dependency-free chat UI: sends history to `/chat`,
  reads the streamed response with `ReadableStream`, renders it live.

### The LLM seam — `crs/llm.py`
`ChatLLM` is a `Protocol` (structural interface) with `stream(messages)`. Two
implementations: `FakeLLM` (deterministic, offline, no key — streams a canned
but context-aware reply) and `OpenAILLM` (real streaming completions). Both take
the same `{role, content}` message shape, so swapping is a one-line change.

### Retrieval — `crs/retrieval.py`
- `Embedder` protocol → `LocalEmbedder` (sentence-transformers, no key; lazy
  import so tests stay light).
- `movie_to_document()` renders a movie as the text we embed (title, plus
  genre/plot when enrichment is present).
- `Retriever` builds a FAISS `IndexFlatIP` (exact cosine search on normalised
  vectors), searches top-k, and can `save`/`load` the index to disk so we don't
  re-embed every run.
- `crs/build_index.py` is the script that builds and caches that index.

### Approach 1 — `crs/rag.py`
`RAGModel`. `_build_query()` composes a search query from recent user turns +
the new question. `build_messages()` assembles the **grounded prompt**: a system
message listing the retrieved candidates with "recommend only from these", then
the replayed conversation, then the new question. Then the LLM streams.

### Approach 2 — `crs/multi_agent.py` + `crs/agents/`
`MultiAgentModel` orchestrates three agents in plain Python:
- **`agents/intent.py`** — `IntentAgent` extracts `UserPreferences`
  (liked/disliked titles, genres). It asks the LLM for JSON, and if the reply
  isn't valid JSON it falls back to a deterministic heuristic — so a bad agent
  output degrades gracefully instead of crashing.
- **`agents/retrieval.py`** — `RetrievalAgent` turns preferences into a query,
  calls the **same** `Retriever` RAG uses, and filters out disliked titles.
- **`agents/response.py`** — `ResponseAgent` reuses RAG's `build_messages()` and
  streams the final reply. It's the only agent whose output reaches the user.

### Data — `data/`
- **`loader.py`** — the boundary between raw files and the system. Joins the real
  three-file LLM-Redial schema (`final_data.jsonl` + `Conversation.txt` +
  `item_map.json`) on `conversation_id` into typed `Conversation`/`Movie`
  objects. `clean_title()` strips store cruft; `load_movie_metadata()` merges
  TMDB enrichment when present. Swapping datasets is a path change here.
- **`enrich.py`** — fetches plots/genres from TMDB (free key) and caches them.
- **`inspect.py`** — prints a dataset summary (sanity-check real vs sample).

### Config & quality
- **`pyproject.toml`** — ruff (PEP 8), mypy (PEP 484), pytest (asyncio) config.
- **`tests/`** — the contract per module, plus a concurrency test proving
  overlapping `/chat` requests don't block each other.

## 3. Design decisions (the "why")

- **RAG for grounding.** An ungrounded LLM invents plausible movies; retrieval
  ties every recommendation to a real catalogue entry.
- **Multi-agent split by pipeline role, not genre.** Each agent has one testable
  job with its own focused prompt. Genre-specialist agents would need a router
  and add complexity without capability. The second approach *reuses* the first
  (same retriever, same prompt builder) rather than duplicating it.
- **Streaming over HTTP, not WebSockets.** The client only receives the reply;
  WebSockets' two-way machinery would be cost without benefit.
- **Swappable seams everywhere** (`CRSModel`, `ChatLLM`, `Embedder`). Depend on a
  contract, not an implementation — that's what makes approach, LLM provider, and
  embedding backend all swap by config.
- **Plain-Python orchestration, no LangChain/LangGraph.** A linear three-step
  handoff doesn't need a framework. A graph library would earn its place only if
  the flow became non-linear (loops, branching, agents retrying each other).
- **Fake LLM by default.** The whole pipeline runs and demos with no API key;
  the real client drops in behind the same interface when a key is set.
- **Async throughout.** `respond()` is an async generator and every LLM stream
  awaits, so one slow request yields the event loop to others.

## 4. Honest limitations

- **Title-only retrieval** without TMDB enrichment: embeddings match title words,
  not meaning ("dreams" → *Nightmare on Elm Street*). Enrichment fixes it.
- **`FakeLLM`** doesn't reason — it can't honour "not superheroes" or interpret
  "yes". That's the OpenAI key's job.
- **Exact FAISS index** (`IndexFlatIP`) is right at this scale; a catalogue in the
  millions would want an approximate index.
- Heuristic intent fallback is deliberately simple; the LLM path is richer.

## 5. Questions an interviewer might ask → where the answer lives

| Question | Where |
| --- | --- |
| How do your two approaches differ, and why two? | [`crs/rag.py`](../crs/rag.py), [`crs/multi_agent.py`](../crs/multi_agent.py) |
| Why split the agents by role instead of genre? | [`docs/notes.md`](notes.md), [`crs/agents/`](../crs/agents/) |
| How does streaming work end-to-end? | `/chat` + `StreamingResponse` in [`app/main.py`](../app/main.py); async gen in [`crs/base.py`](../crs/base.py) |
| Why HTTP streaming and not WebSockets? | [`docs/notes.md`](notes.md) |
| How do you handle an agent returning garbage? | `_parse_preferences` + `_heuristic_preferences` in [`crs/agents/intent.py`](../crs/agents/intent.py) |
| How do you switch approaches or LLM providers? | `build_model` / `CRS_APPROACH`; `_build_llm` + `ChatLLM` in [`crs/llm.py`](../crs/llm.py) |
| What is RAG doing concretely (embeddings, FAISS)? | [`crs/retrieval.py`](../crs/retrieval.py) |
| Why no LangChain/LangGraph? | docstring of [`crs/multi_agent.py`](../crs/multi_agent.py) |
| How do you know it's genuinely concurrent? | [`tests/test_concurrency.py`](../tests/test_concurrency.py) |
| How does the real dataset map in, and what did you assume? | [`data/loader.py`](../data/loader.py), [`docs/dataset.md`](dataset.md) |
| What would you improve next? | Section 4 above |
