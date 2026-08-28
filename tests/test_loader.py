"""Loader tests against the sample, which mirrors the real three-file schema:
item_map.json (ASIN -> title), final_data.jsonl (structured), Conversation.txt
(dialogue joined on conversation_id).
"""

import json

from data.loader import (
    Conversation,
    Movie,
    clean_title,
    load_conversations,
    load_dialogues,
    load_movie_metadata,
)

_DIR = "data/sample"
_ITEM_MAP = f"{_DIR}/item_map.json"
_STRUCTURED = f"{_DIR}/final_data.jsonl"
_DIALOGUE = f"{_DIR}/Conversation.txt"


def test_load_movie_metadata_title_only() -> None:
    movies = load_movie_metadata(_ITEM_MAP)
    assert len(movies) == 13
    inception = movies["B00INCEP01"]
    assert isinstance(inception, Movie)
    assert inception.title == "Inception"
    # Real data carries no genre/plot — these default empty.
    assert inception.genre == ""
    assert inception.description == ""


def test_load_dialogues_blocks_and_roles() -> None:
    dialogues = load_dialogues(_DIALOGUE)
    assert set(dialogues) == {0, 1, 2, 3}  # four blocks, keyed by conversation_id
    roles = {turn.role for turns in dialogues.values() for turn in turns}
    assert roles == {"user", "assistant"}  # User/Agent normalized
    assert all(turns for turns in dialogues.values())


def test_load_conversations_joins_dialogue() -> None:
    convos = load_conversations(_STRUCTURED, _DIALOGUE)
    # 3 users, 4 conversations total (user 3 has two).
    assert len(convos) == 4
    assert all(isinstance(c, Conversation) for c in convos)
    # The join actually attached turns to every conversation.
    assert all(c.turns for c in convos)


def test_clean_title_strips_format_cruft() -> None:
    assert clean_title("Never a Dull Moment [VHS]") == "Never a Dull Moment"
    assert clean_title("No Highway In The Sky VHS") == "No Highway In The Sky"
    assert clean_title("Nightmares &amp; Dreamscapes") == "Nightmares & Dreamscapes"
    assert clean_title("Inception") == "Inception"  # already clean, untouched


def test_load_movie_metadata_merges_enrichment(tmp_path) -> None:
    item_map = tmp_path / "item_map.json"
    item_map.write_text(json.dumps({"B1": "Inception [VHS]"}))
    enrichment = tmp_path / "enrichment.json"
    enrichment.write_text(json.dumps({"B1": {"genre": "Sci-Fi", "description": "Dreams."}}))

    movies = load_movie_metadata(item_map, enrichment)
    assert movies["B1"].title == "Inception"  # cleaned
    assert movies["B1"].genre == "Sci-Fi"  # merged from enrichment
    assert movies["B1"].description == "Dreams."


def test_recommended_items_reference_real_movies() -> None:
    # Integrity: every ground-truth recommendation must exist in the item map.
    convos = load_conversations(_STRUCTURED, _DIALOGUE)
    movies = load_movie_metadata(_ITEM_MAP)
    for convo in convos:
        for item_id in convo.recommended_items:
            assert item_id in movies, f"rec_item {item_id} missing from item_map"
