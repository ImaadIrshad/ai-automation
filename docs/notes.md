# Notes

## Task summary

Build at least two LLM-based CRS approaches over LLM-Redial (Movie category),
serve via a streaming FastAPI endpoint. Candidate approaches: few-shot, RAG,
agent, multi-agent. We commit to **two**: RAG and a function-based multi-agent
system (reasoning below).

## Design decisions

### Approach 1 — RAG (Retrieval-Augmented Generation)

At answer-time, fetch the movie/conversation data relevant to what the user
asked and ground the recommendation in it, instead of relying on the model's
fuzzy internal memory.

**Why:** a recommender must point at *real* movies with *real* attributes. An
ungrounded LLM will happily invent plausible-sounding titles or misattribute
genres. Retrieval keeps every recommendation anchored to something that actually
exists in our dataset.

### Approach 2 — Function-based multi-agent

Split the work into agents by their **role in the pipeline**, not by movie
genre or language:

1. **Intent agent** — reads the conversation and extracts structured user
   preferences (genres, mood, liked/disliked titles, constraints).
2. **Retrieval agent** — turns those preferences into a query and calls the RAG
   retriever. RAG is *reused* here, not duplicated.
3. **Response agent** — takes the retrieved candidates plus the conversation and
   writes the final friendly, streamed recommendation with a short rationale.

**Why role-based, not genre-based:** genre-specialised agents (a "horror agent",
a "comedy agent", ...) are impractical and hard to justify — you'd need a router,
the genres overlap, and it adds complexity without adding capability for a test
of this size. Splitting by pipeline stage gives each agent one clear job, which
is easier to reason about, test, and defend.

### Streaming — one-way HTTP streaming, not WebSockets

Responses stream back token-by-token over HTTP (`StreamingResponse`, SSE-style).

**Why not WebSockets:** WebSockets buy you *two-way* mid-connection messaging.
Here the client sends one question, then only *receives* the streamed answer — it
never needs to talk back mid-stream. So the two-way complexity (connection
lifecycle, upgrade handshake, heartbeats) is cost without benefit.

### Contract — the `CRSModel` abstract base class

Both approaches implement the same async streaming interface in
[`crs/base.py`](../crs/base.py):
`respond(history, question) -> AsyncIterator[str]`. Because they share one
contract, either approach is swappable behind the `/chat` endpoint via config —
no endpoint code changes to switch between RAG and multi-agent.

## What exists vs what's next

**Exists now:** FastAPI app with a streaming `/chat` endpoint (currently backed
by a placeholder `_EchoModel`), the `CRSModel` contract, a pinned
`requirements.txt`, and a passing health test. Baseline runs on **Python 3.10**.

**Next:** dataset loading + a synthetic sample in the real schema's shape (so the
pipeline can be built/tested before dataset approval lands), a FAISS-backed
retriever, then the RAG and multi-agent models, then serving/streaming
hardening, then the "make it real" layer (UI, Docker, eval, README).

## Open blocker

The **LLM-Redial** dataset (`LitGreenhand/LLM-Redial`) requires an **email
request to the authors for approval** before download; it also draws on Amazon
review data. That request is being started in parallel. To avoid blocking on it,
downstream work is built against a synthetic sample in the real schema's shape
and swaps to the real data by changing only a path/config.
