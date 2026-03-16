"""
recommend.py
Builds a content-based recommendation engine using cosine similarity.
For each album in albums_clean.json, finds the 3 most similar other albums
based on genre overlap, release year, and popularity.

Genres come from albums_clean.json, which merges the manually curated tags
from albums.json with any Last.fm tags fetched by fetch_lastfm.py. This gives
the recommender a richer signal than either source alone.

This is step 3 of the pipeline:
    fetch_spotify.py  ->  fetch_lastfm.py  ->  clean_data.py
                      ->  recommend.py  ->  build_data.py

Run from the project root:
    python pipeline/recommend.py
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer


# ── Config ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
CLEAN_FILE = ROOT / "data" / "albums_clean.json"   # merged genres + Spotify metadata
OUT_FILE   = ROOT / "data" / "recommendations.json"

TOP_N = 3   # number of recommendations to surface per album
# ─────────────────────────────────────────────────────────────────────────────


# ── Step 1: Load clean data ──────────────────────────────────────────────────
# albums_clean.json already contains merged genres (manual + Last.fm) from
# the clean_data.py step, so no secondary load of albums.json is needed here.

with open(CLEAN_FILE, "r", encoding="utf-8") as f:
    albums = json.load(f)   # list ordered by slug (alphabetical from clean step)

slugs = [a["slug"] for a in albums]
print(f"Loaded {len(albums)} albums\n")

print("Genre tags per album:")
for a in albums:
    print(f"  {a['name']}: {', '.join(a['genres']) or '(none)'}")
print()


# ── Step 2: Build the feature matrix ────────────────────────────────────────
# We combine three signals into one numeric vector per album:
#
#   A) Genre (one-hot)   — captures stylistic similarity; the primary signal
#   B) Release year      — captures era proximity (e.g. early 2010s vs. 2020s)
#   C) Popularity        — captures mainstream vs. underground feel
#
# All numeric features are normalized to [0, 1] so no single feature
# dominates due to scale differences (raw year ~2000 vs. a 0/1 genre flag).

# A) Genre: MultiLabelBinarizer converts lists of strings into a binary matrix.
#    Each column represents one unique genre tag across the whole corpus.
#    An album gets a 1 in column j if it has that genre tag, 0 otherwise.
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform([a["genres"] for a in albums])
# shape: (n_albums, n_unique_genres)

print(f"Unique genre tags ({len(mlb.classes_)}): {', '.join(mlb.classes_)}\n")

# B) Release year: scale the 4-digit years to [0, 1]
years = np.array([a["release_year"] for a in albums], dtype=float).reshape(-1, 1)
year_scaled = MinMaxScaler().fit_transform(years)
# shape: (n_albums, 1)

# C) Popularity: scale 0-100 score to [0, 1].
#    Client Credentials auth returns null for popularity — treat as 50 (neutral
#    midpoint) so it contributes a flat signal rather than skewing the scores.
raw_pop = [a["popularity"] if a["popularity"] is not None else 50 for a in albums]
pop = np.array(raw_pop, dtype=float).reshape(-1, 1)
pop_scaled = MinMaxScaler().fit_transform(pop)
# shape: (n_albums, 1)

# Stack all features side by side into one matrix
feature_matrix = np.hstack([genre_matrix, year_scaled, pop_scaled])
# shape: (n_albums, n_unique_genres + 2)

print(f"Feature matrix: {feature_matrix.shape[0]} albums x {feature_matrix.shape[1]} features")
print(f"  {genre_matrix.shape[1]} genre columns  |  1 year column  |  1 popularity column\n")


# ── Step 3: Compute pairwise cosine similarity ───────────────────────────────
# cosine_similarity returns an (n x n) matrix where entry [i][j] is the
# cosine similarity between album i and album j. Scores range from 0 (nothing
# in common) to 1 (identical vectors). The diagonal is always 1.0 (self).
#
# Cosine similarity is a good fit here because it measures the *angle* between
# vectors, not magnitude — so an album with 4 genre tags isn't penalized vs.
# one with 2 just because its vector is longer.
sim_matrix = cosine_similarity(feature_matrix)


# ── Step 4: Find top-N recommendations and build output ─────────────────────
print("Recommendations\n" + "-" * 48)

output = []

for i, album in enumerate(albums):
    scores = sim_matrix[i]

    # Rank all other albums by similarity score, descending, excluding self
    ranked = sorted(
        [(j, scores[j]) for j in range(len(albums)) if j != i],
        key=lambda x: x[1],
        reverse=True
    )

    top = ranked[:TOP_N]
    rec_slugs = [slugs[j] for j, _ in top]

    output.append({
        "slug": album["slug"],
        "recommendations": rec_slugs,
    })

    print(f"  {album['name']}")
    for j, score in top:
        print(f"    {score:.3f}  {albums[j]['name']} ({albums[j]['artist']})")
    print()


# ── Step 5: Write output ─────────────────────────────────────────────────────
with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"Written to {OUT_FILE.relative_to(ROOT)}")
