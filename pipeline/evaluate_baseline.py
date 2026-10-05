"""
evaluate_baseline.py
Measures how often the top-3 recommendations share a genre tag with the
source album, and compares that rate against a random baseline.

This is a read-only analysis script. It does not write to recommendations.json
or any file the live site reads; it's for answering "is the model actually
better than chance?" with a real number instead of an estimate.

Usage:
  python pipeline/evaluate_baseline.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from recommend import (
    build_rec_objects,
    build_similarity_matrix,
    load_embeddings,
    TAG_WEIGHT,
)

ROOT           = Path(__file__).resolve().parent.parent
CLEAN_FILE     = ROOT / "data" / "albums_clean.json"
OUT_FILE       = ROOT / "data" / "baseline_eval.json"

TOP_N = 3
RANDOM_TRIALS = 1000
RANDOM_SEED = 42


def load_albums():
    """Load the clean album corpus and its SBERT embeddings, same as recommend.py."""
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        albums = json.load(f)

    slugs = [a["slug"] for a in albums]
    return albums, slugs, load_embeddings(slugs)


def genre_match_rate(albums, slugs, sim_matrix):
    """Fraction of top-3 recs (across all albums) that share at least one genre tag with the source."""
    matches = 0
    total = 0

    for i, album in enumerate(albums):
        recs = build_rec_objects(album, albums, sim_matrix[i], slugs, TOP_N)
        for rec in recs:
            total += 1
            if rec["shared_tags"]:
                matches += 1

    return matches / total, total


def catalog_coverage(albums, slugs, sim_matrix):
    """
    How evenly recommendations spread across the catalog.

    Genre-match rate alone can't catch a model that sends everyone to the same
    handful of albums, so this reports the fraction of albums that appear in at
    least one top-3 list, plus the single most-recommended album.

    Returns
    -------
    coverage : float
        Share of the catalog appearing in any top-3 list.
    top_album : tuple[str, int]
        The most frequently recommended slug and how many top-3 lists it is in.
    """
    appearances = Counter(
        rec["slug"]
        for i, album in enumerate(albums)
        for rec in build_rec_objects(album, albums, sim_matrix[i], slugs, TOP_N)
    )
    return len(appearances) / len(albums), appearances.most_common(1)[0]


def random_baseline_match_rate(albums, trials, seed):
    """Average fraction of top-3 *randomly chosen* recs (excluding same artist, like production)
    that share at least one genre tag with the source, averaged over many trials for stability."""
    rng = random.Random(seed)
    genres_lower = [{g.lower() for g in a["genres"]} for a in albums]
    n = len(albums)

    rates = []
    for _ in range(trials):
        matches = 0
        total = 0
        for i, album in enumerate(albums):
            pool = [j for j in range(n) if j != i and albums[j]["artist"] != album["artist"]]
            if len(pool) < TOP_N:
                continue
            picks = rng.sample(pool, TOP_N)
            for j in picks:
                total += 1
                if genres_lower[i] & genres_lower[j]:
                    matches += 1
        rates.append(matches / total)

    return float(np.mean(rates)), float(np.std(rates))


def main():
    albums, slugs, raw_embeddings = load_albums()
    print(f"Loaded {len(albums)} albums, {len(raw_embeddings)} review embedding(s)\n")

    # 1. Real model: the actual production weights from recommend.py
    real_sim = build_similarity_matrix(albums, raw_embeddings)
    real_rate, real_total = genre_match_rate(albums, slugs, real_sim)
    coverage, (top_slug, top_count) = catalog_coverage(albums, slugs, real_sim)

    # 2. Genre-only baseline: same blend, but year/listeners/review weights zeroed out
    genre_only_sim = build_similarity_matrix(albums, raw_embeddings, TAG_WEIGHT, 0.0, 0.0, 0.0)
    genre_only_rate, _ = genre_match_rate(albums, slugs, genre_only_sim)

    # 3. Random baseline: random picks from the same same-artist-excluded pool
    random_rate, random_std = random_baseline_match_rate(albums, RANDOM_TRIALS, RANDOM_SEED)

    print("Fraction of top-3 recommendations sharing at least one genre tag with the source album:\n")
    print(f"  Production model (genre + year + listeners + review):  {real_rate:.1%}  ({real_total} recs checked)")
    print(f"  Genre-only baseline (tags weighted, nothing else):      {genre_only_rate:.1%}")
    print(f"  Random baseline ({RANDOM_TRIALS} trials):                        {random_rate:.1%}  (std: {random_std:.1%})")
    print(f"\n  Model vs. random: {real_rate / random_rate:.1f}x more likely to share a genre tag")
    print(f"\nCatalog coverage: {coverage:.1%} of albums appear in at least one top-3 list")
    print(f"Most recommended: {top_slug} ({top_count} top-3 lists)")

    results = {
        "top_n": TOP_N,
        "total_recs_checked": real_total,
        "production_model_rate": round(real_rate, 4),
        "genre_only_baseline_rate": round(genre_only_rate, 4),
        "random_baseline_rate": round(random_rate, 4),
        "random_baseline_std": round(random_std, 4),
        "random_baseline_trials": RANDOM_TRIALS,
        "random_baseline_seed": RANDOM_SEED,
        "catalog_coverage": round(coverage, 4),
        "most_recommended": {"slug": top_slug, "top3_appearances": top_count},
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWritten to {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
