from data.loader import Conversation, Movie, load_conversations, load_movie_metadata

_SAMPLE = "data/sample/movie_sample.json"
_METADATA = "data/sample/movie_metadata.json"


def test_load_movie_metadata() -> None:
    movies = load_movie_metadata(_METADATA)
    assert len(movies) == 15
    inception = movies["B00INCEP01"]
    assert isinstance(inception, Movie)
    assert inception.title == "Inception"
    assert inception.genre  # non-empty
    assert inception.description


def test_load_conversations_flattens_and_types() -> None:
    convos = load_conversations(_SAMPLE)
    # 3 users, 4 dialogues total (user 3 has two).
    assert len(convos) == 4
    assert all(isinstance(c, Conversation) for c in convos)
    assert all(c.turns for c in convos)  # every dialogue has turns


def test_roles_normalized() -> None:
    convos = load_conversations(_SAMPLE)
    roles = {turn.role for c in convos for turn in c.turns}
    # Dataset's "User"/"Agent" must be normalized to the chat convention.
    assert roles == {"user", "assistant"}


def test_recommended_items_reference_real_movies() -> None:
    # Integrity check: every ground-truth recommendation must exist in metadata,
    # otherwise retrieval/eval would point at a movie we know nothing about.
    convos = load_conversations(_SAMPLE)
    movies = load_movie_metadata(_METADATA)
    for convo in convos:
        for item_id in convo.recommended_items:
            assert item_id in movies, f"rec_item {item_id} missing from metadata"
