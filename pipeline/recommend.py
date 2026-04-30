"""
recommend.py
Builds content-based recommendations using cosine similarity on genre tags,
release year, and Last.fm listener counts.
"""

import json
import math
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Config
ROOT           = Path(__file__).resolve().parent.parent
CLEAN_FILE     = ROOT / "data" / "albums_clean.json"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
OUT_FILE       = ROOT / "data" / "recommendations.json"

TOP_N = 10

# Signal weights
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

if __name__ == "__main__":
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        albums = json.load(f)

    slugs = [a["slug"] for a in albums]
    print(f"Loaded {len(albums)} albums\n")

    # Load SBERT review embeddings if available
    raw_embeddings = {}
    for slug in slugs:
        emb_path = EMBEDDINGS_DIR / f"{slug}.npy"
        if emb_path.exists():
            raw_embeddings[slug] = np.load(emb_path)

    embedding_dim = next(iter(raw_embeddings.values())).shape[0] if raw_embeddings else 0
    print(f"Loaded {len(raw_embeddings)} review embedding(s)\n")

    # Build feature signals
    tfidf_matrix, all_tags = build_tfidf_matrix(albums)

    years = np.array([a["release_year"] for a in albums], dtype=float).reshape(-1, 1)
    year_scaled = MinMaxScaler().fit_transform(years)

    raw_listeners   = [a.get("listeners") for a in albums]
    valid_listeners = [l for l in raw_listeners if l is not None]
    median_listeners = int(np.median(valid_listeners)) if valid_listeners else 100_000
    filled_listeners = [l if l is not None else median_listeners for l in raw_listeners]

    listeners_scaled = MinMaxScaler().fit_transform(
        np.array(filled_listeners, dtype=float).reshape(-1, 1)
    )

    if embedding_dim > 0:
        emb_matrix = np.zeros((len(albums), embedding_dim))
        for i, album in enumerate(albums):
            if album["slug"] in raw_embeddings:
                emb_matrix[i] = raw_embeddings[album["slug"]]
    else:
        emb_matrix = np.zeros((len(albums), 1))

    # Combine signals
    feature_matrix = np.hstack([
        tfidf_matrix     * TAG_WEIGHT,
        year_scaled      * YEAR_WEIGHT,
        listeners_scaled * LISTENERS_WEIGHT,
        emb_matrix       * REVIEW_WEIGHT,
    ])

    # Compute similarity and find top recommendations
    sim_matrix = cosine_similarity(feature_matrix)
    output = []
    
    print("Generating recommendations...")
    for i, album in enumerate(albums):
        scores = sim_matrix[i]
        recs = build_rec_objects(album, albums, scores, slugs, TOP_N)
        output.append({
            "slug":            album["slug"],
            "recommendations": recs,
        })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Written to {OUT_FILE.relative_to(ROOT)}")
