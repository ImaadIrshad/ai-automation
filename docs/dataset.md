# Dataset: LLM-Redial (Movie)

## What it is

**LLM-Redial** is (per its ACL 2024 Findings paper) the largest multi-domain
conversational-recommendation dataset: ~46.9k multi-turn dialogues across 4
domains. We use the **Movie** category only. The dialogues are LLM-generated but
grounded in **real Amazon user behaviour**, so item IDs are Amazon **ASINs**.

- Paper: [ACL 2024 Findings](https://aclanthology.org/2024.findings-acl.529/)
- Official repo: [`LitGreenhand/LLM-Redial`](https://github.com/LitGreenhand/LLM-Redial)

## Access & licensing

The official data is **gated**: you email the authors an application and they
send it back on approval. It is licensed for **research / non-commercial** use
tied to the paper. Access was requested; approval is pending.

A public third-party repo has committed the Movie files, which is how they were
obtained for local development. Because of the license, **the raw data is
gitignored (`data/raw/`) and never committed to this repo** — we do not
redistribute it. Only the small synthetic sample (below) is committed.

## Real schema (confirmed by inspecting the data)

The Movie data is **three files**, joined on `conversation_id`:

**`final_data.jsonl`** — one JSON object per line, keyed by user ID:

| Field | Meaning |
| --- | --- |
| `history_interaction` | list of ASINs the user interacted with before |
| `user_might_like` | list of candidate ASINs |
| `Conversation` | list of `{ "conversation_N": {...} }` wrappers |

Each conversation object holds `conversation_id`, `user_likes`, `user_dislikes`,
and `rec_item` (the ground-truth recommendation) — **but no dialogue text**.

**`Conversation.txt`** — the dialogue, in blocks each headed by a bare integer
(the `conversation_id`), then `User:` / `Agent:` turns. Movies are referenced by
title in quotes. (Correction to an earlier assumption: the turns live here, in a
separate file, *not* nested inside the JSON.)

**`item_map.json`** — `ASIN -> title`. **Title only** — the real data has no
genre or plot. So a movie's search "document" is just its title, which means
title embeddings behave closer to keyword matching than true semantic search.
A clear future improvement is enriching the catalogue with plots/genres (e.g.
TMDB) so retrieval can match on meaning, not just title words.

Scale of the real Movie set: **3,131 users, 10,089 conversations, 9,687 movies.**

## Synthetic sample (committed, for offline tests)

`data/sample/` mirrors the real three-file layout with made-up content (so no
gated data is redistributed): `item_map.json`, `final_data.jsonl`, and
`Conversation.txt` — 3 users, 4 conversations, 13 movies.

Everything downstream reads through `data/loader.py`, so switching between the
sample and the real data is a **path change**, not a code change:

```bash
python -m data.inspect                 # the sample
python -m data.inspect data/raw/Movie  # the real data (once placed there)
python -m crs.build_index data/raw/Movie   # build the FAISS index on real movies
```
