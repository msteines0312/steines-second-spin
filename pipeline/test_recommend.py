"""
test_recommend.py
Unit tests for the recommendation logic in recommend.py.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from recommend import build_rec_objects


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
