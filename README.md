# ai-automation

Exploratory repo for LLM-based systems work. Current focus: a conversational
recommender system (CRS) for movies, served via a streaming FastAPI endpoint.

## Task

Build at least two LLM-based CRS approaches over the
[LLM-Redial](https://arxiv.org/abs/2405.11706) dataset (Movie category only),
then serve one/both behind an API that streams a response given a new
question and its conversation history.

Approaches under consideration: few-shot prompting, RAG, single-agent, or
multi-agent systems. See [`docs/notes.md`](docs/notes.md) for decisions as
they're made.

## Requirements

- Python 3.10 (repo developed against 3.10; adjust `requirements.txt` pins if
  running on a different minor version)
- Dependencies in `requirements.txt`

## Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Download the LLM-Redial dataset (Movie category) and place it under
`data/raw/`. Not committed to the repo — see `.gitignore`.

## Running the API

```bash
uvicorn app.main:app --reload
```

## Project layout

```
app/        FastAPI service (streaming chat endpoint)
crs/        CRS implementations (few-shot / RAG / agent / multi-agent)
data/       Dataset (gitignored) + loading/preprocessing scripts
docs/       Notes and design decisions
tests/      Tests
```
