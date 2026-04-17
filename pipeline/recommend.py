"""
recommend.py
Builds a content-based recommendation engine using cosine similarity with
TF-IDF weighted genre tags and Last.fm listener counts.

Improvements over the original binary genre approach:
  - Tags are weighted by their Last.fm association score (TF), multiplied by
    inverse document frequency (IDF) so rare shared tags score higher than
    ubiquitous ones like "pop" or "rock."
  - Last.fm listener count replaces Spotify popularity (which returns null
    under Client Credentials auth) as the mainstream/underground signal.

This is step 4 of the pipeline:
    fetch_spotify.py -> fetch_lastfm.py -> clean_data.py
                     -> recommend.py -> build_data.py

Run from the project root:
    python pipeline/recommend.py
"""

import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# ── Config ───────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
CLEAN_FILE     = ROOT / "data" / "albums_clean.json"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
OUT_FILE       = ROOT / "data" / "recommendations.json"

TOP_N = 10   # recommendations to store per album (review pages show 3, discover shows 5)

# Feature weights applied before cosine similarity.
# Tag signal is primary; review embeddings add tonal/prose signal where available.
TAG_WEIGHT       = 3.0
YEAR_WEIGHT      = 0.5
LISTENERS_WEIGHT = 0.5
REVIEW_WEIGHT    = 1.5   # blended in only when both albums have embeddings
# ─────────────────────────────────────────────────────────────────────────────


def build_tfidf_matrix(albums: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Build a TF-IDF weighted tag matrix for the album corpus.

    TF (term frequency): the Last.fm association weight for a tag on this
    album, normalized to [0, 1]. Stored in album["tag_weights"].

    IDF (inverse document frequency): log((N+1) / (df+1)) where N is the
    corpus size and df is the number of albums carrying that tag. This form
    guarantees non-negative values and naturally down-weights ubiquitous tags:
      - A tag in every album gets IDF = log(1) = 0 (no discriminating power).
      - A tag in only one album gets the maximum IDF value.

    Two albums sharing a rare tag like "afrobeat" score much higher than two
    albums that both carry "rock" — which is exactly what we want.

    Parameters
    ----------
    albums : list[dict]
        Albums loaded from albums_clean.json. Each must have a "tag_weights"
        key mapping tag name to float weight.

    Returns
    -------
    matrix : np.ndarray
        Shape (n_albums, n_unique_tags). Cell [i][j] = TF * IDF for album i,
        tag j. Zero if album i does not carry tag j.
    all_tags : list[str]
        Sorted list of all unique tags (column labels for the matrix).
    """
    all_tags = sorted({tag for a in albums for tag in a.get("tag_weights", {})})
    n = len(albums)

    # Document frequency: how many albums carry each tag
    df_per_tag = {
        tag: sum(1 for a in albums if tag in a.get("tag_weights", {}))
        for tag in all_tags
    }

    # IDF: higher for rare tags, 0 for tags present in every album
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
    """Build recommendation objects for one album.

    Parameters
    ----------
    album : dict
        The source album. Must have 'slug', 'artist', and 'genres' keys.
    albums : list[dict]
        All albums in the catalog.
    sim_scores : np.ndarray
        1-D array of similarity scores for this album against every album,
        including itself (self-similarity is excluded inside this function).
    slugs : list[str]
        Slug list parallel to albums (same order).
    top_n : int
        Maximum number of recommendations to return.

    Returns
    -------
    list[dict]
        Each dict has keys: slug (str), score (float, 2 dp), shared_tags (list[str]).
        Same-artist albums are excluded — recommending an artist's other work
        to someone already on that artist's page adds no discovery value.
    """
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
    return [
        {
            "slug":        slugs[j],
            "score":       round(float(score), 2),
            # shared_tags uses the display genres list (not tag_weights) so
            # the frontend shows human-readable genre labels, not ML internals
            "shared_tags": sorted(set(album["genres"]) & set(albums[j]["genres"])),
        }
        for j, score in ranked[:top_n]
    ]


if __name__ == "__main__":
    # ── Step 1: Load clean data ──────────────────────────────────────────────
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        albums = json.load(f)

    slugs = [a["slug"] for a in albums]
    print(f"Loaded {len(albums)} albums\n")

    # ── Step 1b: Load SBERT review embeddings (optional) ────────────────────
    # Albums without a published review have no embedding file; they get a zero
    # row so the review signal is neutral for them (doesn't help or hurt).
    raw_embeddings = {}
    for slug in slugs:
        emb_path = EMBEDDINGS_DIR / f"{slug}.npy"
        if emb_path.exists():
            raw_embeddings[slug] = np.load(emb_path)

    embedding_dim = next(iter(raw_embeddings.values())).shape[0] if raw_embeddings else 0
    print(f"Loaded {len(raw_embeddings)} review embedding(s) (dim={embedding_dim})\n")

    print("Tag weights per album (top 5 shown):")
    for a in albums:
        weights = a.get("tag_weights", {})
        top5 = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_str = ", ".join(f"{k}({v:.2f})" for k, v in top5) or "(none)"
        print(f"  {a['name']}: {top5_str}")
    print()


    # ── Step 2: Build the feature matrix ────────────────────────────────────
    # Three signals combined into one numeric vector per album:
    #
    #   A) TF-IDF tags  -- rare shared tags score higher than generic ones
    #   B) Release year -- captures era proximity (e.g. early 2010s vs. 2020s)
    #   C) Listeners    -- Last.fm listener count as a mainstream/underground proxy
    #                      (replaces Spotify popularity, which is null under
    #                       Client Credentials auth)

    # A) TF-IDF weighted tag matrix
    tfidf_matrix, all_tags = build_tfidf_matrix(albums)

    idf_values = [
        math.log((len(albums) + 1) / (
            sum(1 for a in albums if t in a.get("tag_weights", {})) + 1
        ))
        for t in all_tags
    ]
    print(f"TF-IDF tag matrix: {tfidf_matrix.shape[0]} albums x {tfidf_matrix.shape[1]} unique tags")
    print(f"  IDF range: {min(idf_values):.3f} – {max(idf_values):.3f}\n")

    # B) Release year: scale 4-digit years to [0, 1]
    years = np.array([a["release_year"] for a in albums], dtype=float).reshape(-1, 1)
    year_scaled = MinMaxScaler().fit_transform(years)

    # C) Last.fm listeners: use corpus median as fill value for albums without data.
    #    Listener count is a better proxy for mainstream vs. underground feel than
    #    Spotify popularity, which returns null under Client Credentials auth.
    raw_listeners   = [a.get("listeners") for a in albums]
    valid_listeners = [l for l in raw_listeners if l is not None]
    median_listeners = int(np.median(valid_listeners)) if valid_listeners else 100_000
    filled_listeners = [l if l is not None else median_listeners for l in raw_listeners]

    listeners_scaled = MinMaxScaler().fit_transform(
        np.array(filled_listeners, dtype=float).reshape(-1, 1)
    )

    # D) SBERT review embeddings: zero row for albums without a review.
    #    Weight is only meaningful when BOTH albums have embeddings; a zero row
    #    contributes nothing to cosine similarity for unreviewed albums.
    if embedding_dim > 0:
        emb_matrix = np.zeros((len(albums), embedding_dim))
        for i, album in enumerate(albums):
            if album["slug"] in raw_embeddings:
                emb_matrix[i] = raw_embeddings[album["slug"]]
    else:
        emb_matrix = np.zeros((len(albums), 1))

    # Stack all features, applying weights so tag similarity dominates
    feature_matrix = np.hstack([
        tfidf_matrix     * TAG_WEIGHT,
        year_scaled      * YEAR_WEIGHT,
        listeners_scaled * LISTENERS_WEIGHT,
        emb_matrix       * REVIEW_WEIGHT,
    ])
    emb_label = f"  |  {embedding_dim} SBERT review columns" if embedding_dim > 0 else "  |  no review embeddings"
    print(f"Feature matrix: {feature_matrix.shape[0]} albums x {feature_matrix.shape[1]} features")
    print(f"  {tfidf_matrix.shape[1]} TF-IDF tag columns  |  1 year column  |  1 listeners column{emb_label}\n")


    # ── Step 3: Compute pairwise cosine similarity ───────────────────────────
    # cosine_similarity measures the angle between vectors, not magnitude —
    # so an album with 8 genre tags isn't penalized vs. one with 3. Scores
    # range from 0 (nothing in common) to 1 (identical vectors).
    sim_matrix = cosine_similarity(feature_matrix)


    # ── Step 4: Find top-N recommendations and build output ─────────────────
    print("Recommendations\n" + "-" * 48)

    output = []
    slug_to_album = {a["slug"]: a for a in albums}

    for i, album in enumerate(albums):
        scores = sim_matrix[i]
        recs = build_rec_objects(album, albums, scores, slugs, TOP_N)

        output.append({
            "slug":            album["slug"],
            "recommendations": recs,
        })

        print(f"  {album['name']}")
        for r in recs:
            rec_album = slug_to_album[r["slug"]]
            print(f"    {r['score']:.2f}  {rec_album['name']}  "
                  f"[{', '.join(r['shared_tags']) or 'no shared tags'}]")
        print()


    # ── Step 5: Write output ─────────────────────────────────────────────────
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Written to {OUT_FILE.relative_to(ROOT)}")
