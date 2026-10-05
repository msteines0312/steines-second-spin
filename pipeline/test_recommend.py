"""
test_recommend.py
Unit tests for the recommendation logic in recommend.py.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from recommend import (
    build_rec_objects,
    build_tfidf_matrix,
    fill_missing_listeners,
    build_embedding_matrix,
    build_similarity_matrix,
    closeness_similarity,
)


def test_output_format():
    """Verify basic rec object keys and types."""
    albums = [
        {"slug": "a", "artist": "A", "genres": ["rock", "indie"]},
        {"slug": "b", "artist": "B", "genres": ["rock", "pop"]},
        {"slug": "c", "artist": "C", "genres": ["jazz"]},
    ]
    slugs = ["a", "b", "c"]
    sim_scores = np.array([1.0, 0.9, 0.1])

    recs = build_rec_objects(albums[0], albums, sim_scores, slugs, top_n=2)

    assert len(recs) == 2
    for r in recs:
        assert "slug" in r
        assert "score" in r
        assert "shared_tags" in r
        assert isinstance(r["slug"], str)
        assert isinstance(r["score"], float)
        assert isinstance(r["shared_tags"], list)


def test_shared_tags_is_intersection():
    """Tags should be the overlap between albums."""
    albums = [
        {"slug": "x", "artist": "X", "genres": ["post-punk", "art rock", "indie"]},
        {"slug": "y", "artist": "Y", "genres": ["post-punk", "art rock", "noise"]},
    ]
    slugs = ["x", "y"]
    sim_scores = np.array([1.0, 0.8])

    recs = build_rec_objects(albums[0], albums, sim_scores, slugs, top_n=1)

    assert set(recs[0]["shared_tags"]) == {"post-punk", "art rock"}


def test_shared_tags_case_insensitive():
    """Matching should ignore case."""
    albums = [
        {"slug": "x", "artist": "X", "genres": ["Indie Rock", "Art Rock"]},
        {"slug": "y", "artist": "Y", "genres": ["indie rock", "noise"]},
    ]
    slugs = ["x", "y"]
    sim_scores = np.array([1.0, 0.8])

    recs = build_rec_objects(albums[0], albums, sim_scores, slugs, top_n=1)

    # Match should succeed despite differing case between the two genre lists
    assert len(recs[0]["shared_tags"]) == 1
    assert recs[0]["shared_tags"][0].lower() == "indie rock"


def test_excludes_self():
    """Don't recommend the same album."""
    albums = [
        {"slug": "a", "artist": "A", "genres": ["rock"]},
        {"slug": "b", "artist": "B", "genres": ["rock"]},
    ]
    slugs = ["a", "b"]
    sim_scores = np.array([1.0, 0.9])

    recs = build_rec_objects(albums[0], albums, sim_scores, slugs, top_n=5)

    assert all(r["slug"] != "a" for r in recs)


def test_score_rounded_to_two_decimals():
    """Round scores to 2 places."""
    albums = [
        {"slug": "a", "artist": "A", "genres": ["rock"]},
        {"slug": "b", "artist": "B", "genres": ["rock"]},
    ]
    slugs = ["a", "b"]
    sim_scores = np.array([1.0, 0.876543])

    recs = build_rec_objects(albums[0], albums, sim_scores, slugs, top_n=1)

    assert recs[0]["score"] == 0.88


def test_top_n_respected():
    """Limit results to top_n."""
    albums = [{"slug": str(i), "artist": f"Artist{i}", "genres": ["rock"]} for i in range(5)]
    slugs = [str(i) for i in range(5)]
    sim_scores = np.array([1.0, 0.9, 0.8, 0.7, 0.6])

    recs = build_rec_objects(albums[0], albums, sim_scores, slugs, top_n=2)

    assert len(recs) == 2


def test_results_ordered_by_score_descending():
    """Sort by score desc."""
    albums = [
        {"slug": "a", "artist": "A", "genres": ["rock"]},
        {"slug": "b", "artist": "B", "genres": ["rock"]},
        {"slug": "c", "artist": "C", "genres": ["rock"]},
        {"slug": "d", "artist": "D", "genres": ["rock"]},
    ]
    slugs = ["a", "b", "c", "d"]
    sim_scores = np.array([1.0, 0.3, 0.8, 0.6])

    recs = build_rec_objects(albums[0], albums, sim_scores, slugs, top_n=3)

    scores = [r["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_tfidf_shared_tag_scores_higher_for_rarer_tag():
    """A tag shared by fewer albums should get a higher IDF weight."""
    albums = [
        {"tag_weights": {"common": 1.0, "rare": 1.0}},
        {"tag_weights": {"common": 1.0}},
        {"tag_weights": {"common": 1.0}},
    ]

    matrix, all_tags = build_tfidf_matrix(albums)

    common_idx = all_tags.index("common")
    rare_idx = all_tags.index("rare")
    assert matrix[0, rare_idx] > matrix[0, common_idx]


def test_tfidf_matrix_shape():
    """Matrix should be (num_albums, num_unique_tags)."""
    albums = [
        {"tag_weights": {"a": 1.0, "b": 1.0}},
        {"tag_weights": {"b": 1.0, "c": 1.0}},
    ]

    matrix, all_tags = build_tfidf_matrix(albums)

    assert matrix.shape == (2, 3)
    assert set(all_tags) == {"a", "b", "c"}


def test_fill_missing_listeners_uses_median():
    """None values should be replaced with the median of the known values."""
    filled = fill_missing_listeners([100, None, 300])

    assert filled == [100, 200, 300]


def test_fill_missing_listeners_all_missing_uses_fallback():
    """If every value is missing, fall back to the provided default."""
    filled = fill_missing_listeners([None, None], fallback=42)

    assert filled == [42, 42]


def test_build_embedding_matrix_no_embeddings_returns_single_zero_column():
    """With no embeddings at all, fall back to a single zero column."""
    albums = [{"slug": "a"}, {"slug": "b"}]

    matrix = build_embedding_matrix(albums, {})

    assert matrix.shape == (2, 1)
    assert np.all(matrix == 0)


def test_build_embedding_matrix_fills_missing_with_zeros():
    """Albums without an embedding should get a zero row, not skipped or misaligned."""
    albums = [{"slug": "a"}, {"slug": "b"}]
    raw_embeddings = {"a": np.array([1.0, 2.0, 3.0])}

    matrix = build_embedding_matrix(albums, raw_embeddings)

    assert matrix.shape == (2, 3)
    assert np.array_equal(matrix[0], [1.0, 2.0, 3.0])
    assert np.array_equal(matrix[1], [0.0, 0.0, 0.0])


def _album(slug, tags, year=2020, listeners=1000):
    """Minimal clean-album record for similarity tests."""
    return {"slug": slug, "tag_weights": tags, "release_year": year, "listeners": listeners}


def test_closeness_similarity_is_one_on_diagonal_and_zero_at_extremes():
    """Identical values score 1; the two most distant values in the corpus score 0."""
    sim = closeness_similarity(np.array([1970.0, 2000.0, 2020.0]))

    assert np.allclose(np.diag(sim), 1.0)
    assert sim[0, 2] == pytest.approx(0.0)
    assert sim[0, 1] == pytest.approx(0.4)


def test_closeness_similarity_constant_values_are_all_similar():
    """With no spread at all, every pair is a perfect match instead of dividing by zero."""
    sim = closeness_similarity(np.array([5.0, 5.0]))

    assert np.all(sim == 1.0)


def test_similarity_matrix_stays_in_unit_range():
    """The blended score is a weighted average, so it must land in [0, 1]."""
    albums = [
        _album("a", {"rock": 1.0}, 1970, 100),
        _album("b", {"rock": 1.0, "pop": 0.5}, 2020, 1_000_000),
        _album("c", {"jazz": 1.0}, 1995, 5000),
    ]

    sim = build_similarity_matrix(albums, {})

    assert sim.min() >= 0.0
    assert sim.max() <= 1.0 + 1e-9


def test_missing_review_is_not_a_penalty():
    """
    An album without a review should score the same whether or not other albums
    have reviews. This is the bug the late-fusion design fixes: zero-filled
    embeddings used to drag down any candidate that did have one.
    """
    albums = [
        _album("source", {"rock": 1.0}),
        _album("reviewed", {"rock": 1.0}),
        _album("unreviewed", {"rock": 1.0}),
    ]
    embeddings = {"reviewed": np.array([1.0, 0.0])}

    sim = build_similarity_matrix(albums, embeddings)

    # Source has no review, so the reviewed and unreviewed candidates tie
    assert sim[0, 1] == pytest.approx(sim[0, 2])


def test_shared_review_signal_changes_ranking():
    """When both albums have reviews, similar review text should lift the score."""
    albums = [
        _album("source", {"rock": 1.0}),
        _album("similar_review", {"rock": 1.0}),
        _album("different_review", {"rock": 1.0}),
    ]
    embeddings = {
        "source":           np.array([1.0, 0.0]),
        "similar_review":   np.array([1.0, 0.0]),
        "different_review": np.array([0.0, 1.0]),
    }

    sim = build_similarity_matrix(albums, embeddings)

    assert sim[0, 1] > sim[0, 2]
