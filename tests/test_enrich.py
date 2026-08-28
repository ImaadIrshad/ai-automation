"""Tests for the TMDB enrichment parsing (offline — no network, no key)."""

from data.enrich import parse_search_result

_GENRE_MAP = {878: "Science Fiction", 53: "Thriller"}


def test_parse_maps_genre_ids_and_overview() -> None:
    result = {"genre_ids": [878, 53], "overview": "A thief enters dreams."}
    parsed = parse_search_result(result, _GENRE_MAP)
    assert parsed["genre"] == "Science Fiction/Thriller"
    assert parsed["description"] == "A thief enters dreams."


def test_parse_handles_missing_fields() -> None:
    # Unknown genre ids are dropped; a missing overview becomes an empty string.
    parsed = parse_search_result({"genre_ids": [999]}, _GENRE_MAP)
    assert parsed["genre"] == ""
    assert parsed["description"] == ""
