"""
recommend.py
Builds content-based recommendations by blending four similarity signals:
genre tags (TF-IDF cosine), release year, Last.fm listener counts, and
SBERT embeddings of written review text.

Each signal produces its own album-by-album similarity matrix in [0, 1], and the
final score is a weighted average of those matrices. This "late fusion" design
replaces an earlier version that stacked every feature into one vector and took
a single cosine similarity. That version let the signal with the largest raw
magnitude (TF-IDF tags, row norm ~10) drown out the others: the review
embeddings (norm 1) moved scores by about 1% and never changed a single top-3
recommendation.
"""

import json
import math
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Config
ROOT           = Path(__file__).resolve().parent.parent
CLEAN_FILE     = ROOT / "data" / "albums_clean.json"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
OUT_FILE       = ROOT / "data" / "recommendations.json"

TOP_N = 10

# Signal weights. Because every signal is a similarity in [0, 1] before weighting,
# these are true relative importances: genre counts twice as much as review text,
# and six times as much as year or popularity.
TAG_WEIGHT       = 3.0
YEAR_WEIGHT      = 0.5
LISTENERS_WEIGHT = 0.5
REVIEW_WEIGHT    = 1.5

def build_tfidf_matrix(albums: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Build a TF-IDF weighted tag matrix for the album corpus."""
    all_tags = sorted({tag for a in albums for tag in a.get("tag_weights", {})})
    n = len(albums)

    df_per_tag = {
        tag: sum(1 for a in albums if tag in a.get("tag_weights", {}))
        for tag in all_tags
    }

    idf_per_tag = {
        tag: math.log((n + 1) / (df_per_tag[tag] + 1))
        for tag in all_tags
    }

    tag_idx = {tag: i for i, tag in enumerate(all_tags)}
    matrix  = np.zeros((n, len(all_tags)))

    for i, album in enumerate(albums):
        for tag, weight in album.get("tag_weights", {}).items():
            if tag in tag_idx:
                matrix[i, tag_idx[tag]] = weight * idf_per_tag[tag]

    return matrix, all_tags

def fill_missing_listeners(raw_listeners: list, fallback: int = 100_000) -> list:
    """Replace missing listener counts with the corpus median, or a fallback if none exist."""
    valid = [l for l in raw_listeners if l is not None]
    median = int(np.median(valid)) if valid else fallback
    return [l if l is not None else median for l in raw_listeners]

def build_embedding_matrix(albums: list[dict], raw_embeddings: dict) -> np.ndarray:
    """Stack per-album SBERT embeddings into a matrix, zero-filling albums with none."""
    embedding_dim = next(iter(raw_embeddings.values())).shape[0] if raw_embeddings else 0
    if embedding_dim == 0:
        return np.zeros((len(albums), 1))

    matrix = np.zeros((len(albums), embedding_dim))
    for i, album in enumerate(albums):
        if album["slug"] in raw_embeddings:
            matrix[i] = raw_embeddings[album["slug"]]
    return matrix

def closeness_similarity(values: np.ndarray) -> np.ndarray:
    """
    Pairwise similarity for a single numeric feature: 1 when two values are
    equal, falling linearly to 0 for the two most distant values in the corpus.

    Parameters
    ----------
    values : np.ndarray
        One value per album, shape (n,).

    Returns
    -------
    np.ndarray
        Symmetric (n, n) similarity matrix in [0, 1].
    """
    value_range = values.max() - values.min()
    if value_range == 0:
        return np.ones((len(values), len(values)))
    return 1.0 - np.abs(values[:, None] - values[None, :]) / value_range

def review_similarity(albums: list[dict], raw_embeddings: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Cosine similarity between review embeddings, plus a mask of which pairs it
    is defined for (both albums must have a published review).

    Returns
    -------
    sim : np.ndarray
        (n, n) cosine similarity matrix. Rows for unreviewed albums are zero.
    both_reviewed : np.ndarray
        (n, n) boolean mask, True where both albums have an embedding.
    """
    emb_matrix = build_embedding_matrix(albums, raw_embeddings)
    has_review = np.array([a["slug"] in raw_embeddings for a in albums])
    both_reviewed = has_review[:, None] & has_review[None, :]
    return cosine_similarity(emb_matrix), both_reviewed

def build_similarity_matrix(albums: list[dict], raw_embeddings: dict,
                            tag_weight: float = TAG_WEIGHT,
                            year_weight: float = YEAR_WEIGHT,
                            listeners_weight: float = LISTENERS_WEIGHT,
                            review_weight: float = REVIEW_WEIGHT) -> np.ndarray:
    """
    Blend per-signal similarity matrices into one weighted average.

    The review signal only counts for pairs where both albums have a review.
    For every other pair its weight is dropped from both the numerator and the
    denominator, so a missing review neither helps nor hurts an album's score.

    Parameters
    ----------
    albums : list[dict]
        Clean album records with tag_weights, release_year, listeners, slug.
    raw_embeddings : dict
        Mapping of slug to SBERT embedding for reviewed albums.
    tag_weight, year_weight, listeners_weight, review_weight : float
        Relative importance of each signal. Exposed so evaluate_baseline.py
        can ablate signals by setting their weight to 0.

    Returns
    -------
    np.ndarray
        (n, n) similarity matrix in [0, 1].
    """
    tfidf_matrix, _ = build_tfidf_matrix(albums)
    tag_sim = cosine_similarity(tfidf_matrix)

    years    = np.array([a["release_year"] for a in albums], dtype=float)
    year_sim = closeness_similarity(years)

    # Listener counts span four orders of magnitude, so compare them on a log
    # scale. Otherwise every album except the biggest few looks identical.
    listeners     = np.array(fill_missing_listeners([a.get("listeners") for a in albums]), dtype=float)
    listeners_sim = closeness_similarity(np.log10(np.maximum(listeners, 1)))

    rev_sim, both_reviewed = review_similarity(albums, raw_embeddings)
    review_weights = np.where(both_reviewed, review_weight, 0.0)

    weighted_sum = (tag_weight * tag_sim
                    + year_weight * year_sim
                    + listeners_weight * listeners_sim
                    + review_weights * rev_sim)
    total_weight = tag_weight + year_weight + listeners_weight + review_weights
    return weighted_sum / total_weight

def build_rec_objects(album, albums, sim_scores, slugs, top_n):
    """Build recommendation objects for one album, excluding same artist."""
    slug_to_idx = {s: i for i, s in enumerate(slugs)}
    i = slug_to_idx[album["slug"]]
    ranked = sorted(
        [
            (j, sim_scores[j])
            for j in range(len(albums))
            if j != i and albums[j]["artist"] != album["artist"]
        ],
        key=lambda x: x[1],
        reverse=True,
    )

    source_genres_lower = {g.lower() for g in album["genres"]}
    return [
        {
            "slug":        slugs[j],
            "score":       round(float(score), 2),
            "shared_tags": sorted({
                g for g in albums[j]["genres"] if g.lower() in source_genres_lower
            }),
        }
        for j, score in ranked[:top_n]
    ]

def load_embeddings(slugs: list[str]) -> dict:
    """Load SBERT review embeddings for any album that has one on disk."""
    raw_embeddings = {}
    for slug in slugs:
        emb_path = EMBEDDINGS_DIR / f"{slug}.npy"
        if emb_path.exists():
            raw_embeddings[slug] = np.load(emb_path)
    return raw_embeddings

if __name__ == "__main__":
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        albums = json.load(f)

    slugs = [a["slug"] for a in albums]
    print(f"Loaded {len(albums)} albums\n")

    raw_embeddings = load_embeddings(slugs)
    print(f"Loaded {len(raw_embeddings)} review embedding(s)\n")

    sim_matrix = build_similarity_matrix(albums, raw_embeddings)
    output = []

    print("Generating recommendations...")
    for i, album in enumerate(albums):
        recs = build_rec_objects(album, albums, sim_matrix[i], slugs, TOP_N)
        output.append({
            "slug":            album["slug"],
            "recommendations": recs,
        })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Written to {OUT_FILE.relative_to(ROOT)}")
