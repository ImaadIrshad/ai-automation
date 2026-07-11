# Dataset: LLM-Redial (Movie)

## What it is

**LLM-Redial** is (per its ACL 2024 Findings paper) the largest multi-domain
conversational-recommendation dataset: ~46.9k multi-turn dialogues and ~465.9k
utterances across 4 domains. We use the **Movie** category only. The dialogues
are LLM-generated, but grounded in **real Amazon user behaviour** — so item IDs
are Amazon **ASINs**, and human-readable movie details (title, genre, plot) come
from separate Amazon product metadata.

- Paper: [ACL 2024 Findings](https://aclanthology.org/2024.findings-acl.529/)
- Repo: [`LitGreenhand/LLM-Redial`](https://github.com/LitGreenhand/LLM-Redial)

## Access status — BLOCKED (in progress)

The full data requires an **email request to the authors for approval**; the
public repo ships only documentation and examples, no data files. The request
has been sent and is pending. **We are not waiting on it** — see the synthetic
sample below.

## Schema (from the repo's `example.md` / `example_conversation.md`)

Top-level object keyed by **user ID**:

| Field | Meaning |
| --- | --- |
| `history_interaction` | list of item IDs (ASINs) the user interacted with before |
| `user_might_like` | list of item IDs the system may recommend |
| `Conversation` | list of conversation objects (see below) |

Each entry in `Conversation` is `{ "conversation_N": { ... } }` with:

| Field | Meaning |
| --- | --- |
| `conversation_id` | integer id for the dialogue |
| `user_likes` | item IDs the user approves of in this dialogue |
| `user_dislikes` | item IDs the user rejects |
| `rec_item` | the item ID(s) the system recommends (the ground truth) |
| *(dialogue turns)* | multi-turn **User**/**Agent** utterances; movies referenced by title in quotes |

**Confirmed vs inferred:** the field names above (`history_interaction`,
`user_might_like`, `conversation_id`, `user_likes`, `user_dislikes`, `rec_item`)
come straight from the repo's `example.md`. What is **inferred** is exactly how
the natural-language turns are nested alongside the structured annotations — the
two examples present the structured data and the dialogue text separately, so in
our synthetic sample we nest the turns under a `content` list of
`{"role", "text"}` objects. When real data arrives, `data/inspect.py` prints the
true structure and we adjust the loader if the nesting differs — nothing else
downstream needs to change.

## Synthetic sample (the unblock)

To build and test the whole pipeline before approval lands, `data/sample/`
contains a small dataset in this exact shape:

- `movie_sample.json` — 3 users, 4 conversations, with likes/dislikes/rec_item
  grounded in real ASINs from the metadata file below.
- `movie_metadata.json` — 15 movies (ASIN → title, genre, description), our
  stand-in for the Amazon product metadata.

Everything downstream reads through `data/loader.py`, so swapping to the real
data is a **path change**, not a code change. Think of the sample as a
crash-test dummy built to the real passenger's exact dimensions: we test the
whole car on it, and the real data slots in when it's approved.
