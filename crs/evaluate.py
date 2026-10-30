"""Offline evaluation of the two CRS approaches on held-out LLM-Redial dialogues.

Question answered: when a user asks the way they do in the real dataset, does our
system surface the movie the dataset actually recommended (its ``rec_item``)?

Two honest metrics:
- **retrieval recall@k** — is the ground-truth movie among the top-k retrieved?
  (LLM-free, so run on a larger sample.)
- **end-to-end hit rate** — does the final recommendation text actually name it?
  Reported per approach (RAG vs multi-agent), so we can compare them.

Evaluated on LLM-Redial (the dataset of record), NOT the TMDB demo catalogue, so
the ground-truth movies are actually in the index. Requires the LLM-Redial data
in data/raw/Movie and (for the end-to-end metric) a configured LLM key.

    python -m crs.evaluate
    python -m crs.evaluate --recall 200 --e2e 15
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections.abc import AsyncIterator

from app.config import get_settings
from crs.base import Turn
from crs.llm import ChatLLM, FakeLLM, OpenAILLM
from crs.multi_agent import MultiAgentModel
from crs.rag import RAGModel, _build_query
from crs.retrieval import LocalEmbedder, Retriever
from data.loader import (
    Conversation,
    Movie,
    load_conversations,
    load_movie_metadata,
)

_DATA_DIR = "data/raw/Movie"
_ENRICHMENT = "data/processed/enrichment.json"
_EVAL_INDEX = "data/processed/eval_index"
_TOP_K = 20

# A sample is (conversation, its ground-truth titles, the user-only context turns).
Sample = tuple[Conversation, list[str], list[Turn]]


def _make_llm() -> ChatLLM:
    """Build the LLM straight from settings (avoids importing app.main, which
    would eagerly build the demo model at import time)."""
    settings = get_settings()
    if settings.llm_api_key:
        return OpenAILLM(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url or None,
        )
    return FakeLLM()


def _rec_titles(conv: Conversation, movies: dict[str, Movie]) -> list[str]:
    return [movies[i].title for i in conv.recommended_items if i in movies]


def _user_context(conv: Conversation, rec_titles: list[str]) -> list[Turn]:
    """User turns that express preferences *without leaking the answer title*."""
    lowered = [t.lower() for t in rec_titles if t]
    return [
        turn
        for turn in conv.turns
        if turn.role == "user"
        and not any(rt in turn.content.lower() for rt in lowered)
    ]


def _title_hit(text: str, rec_titles: list[str]) -> bool:
    low = text.lower()
    return any(len(rt) >= 3 and rt.lower() in low for rt in rec_titles)


def _samples(
    conversations: list[Conversation], movies: dict[str, Movie], limit: int
) -> list[Sample]:
    """First `limit` conversations with a usable ground-truth and user context."""
    out: list[Sample] = []
    for conv in conversations:
        titles = _rec_titles(conv, movies)
        if not titles:
            continue
        context = _user_context(conv, titles)
        if not context:
            continue
        out.append((conv, titles, context))
        if len(out) >= limit:
            break
    return out


def _load_eval_retriever(movies: list[Movie]) -> Retriever:
    embedder = LocalEmbedder()
    try:
        return Retriever.load(_EVAL_INDEX, embedder)
    except (FileNotFoundError, RuntimeError):
        retriever = Retriever.build(movies, embedder)
        retriever.save(_EVAL_INDEX)
        return retriever


async def _collect(stream: AsyncIterator[str]) -> str:
    return "".join([chunk async for chunk in stream])


async def main_async(recall_n: int, e2e_n: int) -> None:
    movies = load_movie_metadata(f"{_DATA_DIR}/item_map.json", _ENRICHMENT)
    conversations = load_conversations(
        f"{_DATA_DIR}/final_data.jsonl", f"{_DATA_DIR}/Conversation.txt"
    )
    print(f"Loaded {len(conversations)} conversations over {len(movies)} movies.")

    retriever = _load_eval_retriever(list(movies.values()))

    # --- Retrieval recall@k (LLM-free) --------------------------------------
    recall_samples = _samples(conversations, movies, recall_n)
    ks = [5, 10, 20, 50]
    hits = dict.fromkeys(ks, 0)
    for _conv, titles, context in recall_samples:
        query = _build_query(context[:-1], context[-1].content)
        ranked = [c.movie.title for c in retriever.search(query, top_k=max(ks))]
        for k in ks:
            if any(t in ranked[:k] for t in titles):
                hits[k] += 1
    n_recall = len(recall_samples)
    print(f"\nRetrieval recall over {n_recall} conversations:")
    for k in ks:
        rate = hits[k] / n_recall if n_recall else 0.0
        print(f"  recall@{k:<2d}: {rate:5.1%}  ({hits[k]}/{n_recall})")

    # --- End-to-end hit rate: RAG vs multi-agent (uses the LLM) --------------
    llm = _make_llm()
    rag = RAGModel(retriever=retriever, llm=llm, top_k=_TOP_K)
    multi = MultiAgentModel(retriever=retriever, llm=llm, top_k=_TOP_K)
    e2e_samples = _samples(conversations, movies, e2e_n)

    rag_hits = multi_hits = scored = 0
    for _conv, titles, context in e2e_samples:
        history, question = context[:-1], context[-1].content
        try:
            rag_out = await _collect(rag.respond(history, question))
            multi_out = await _collect(multi.respond(history, question))
        except Exception as exc:  # network / rate limit — skip, don't fake a score
            print(f"  (skipped one conversation: {exc})")
            continue
        rag_hits += _title_hit(rag_out, titles)
        multi_hits += _title_hit(multi_out, titles)
        scored += 1
        time.sleep(2.0)  # stay under the free-tier rate limit

    print(f"\nEnd-to-end hit rate (ground-truth movie named), over {scored}:")
    if scored:
        print(f"  RAG:         {rag_hits / scored:.1%}  ({rag_hits}/{scored})")
        print(f"  Multi-agent: {multi_hits / scored:.1%}  ({multi_hits}/{scored})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the two CRS approaches.")
    parser.add_argument("--recall", type=int, default=200, help="recall sample size")
    parser.add_argument("--e2e", type=int, default=12, help="end-to-end sample size")
    args = parser.parse_args()
    asyncio.run(main_async(args.recall, args.e2e))


if __name__ == "__main__":
    main()
