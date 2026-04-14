"""
build_data.py
Merges three data sources into one flat JSON file for the frontend.

This is the final step of the pipeline:
    fetch_spotify.py -> clean_data.py -> recommend.py -> build_data.py

The output (site/data/albums_final.json) is what the frontend HTML pages
fetch and render. One array of complete album objects, nothing left to join.

Run from the project root:
    python pipeline/build_data.py
"""

import json
from pathlib import Path


# ── Config ───────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
EDITORIAL    = ROOT / "data" / "albums.json"           # stars, blurb, genres, etc.
CLEAN        = ROOT / "data" / "albums_clean.json"     # Spotify metadata
RECS         = ROOT / "data" / "recommendations.json"  # recommendation slugs
OUT_DIR      = ROOT / "data"
OUT_FILE     = OUT_DIR / "albums_final.json"
# ─────────────────────────────────────────────────────────────────────────────


# ── Step 1: Load all three sources ───────────────────────────────────────────
# Each source is keyed by the album's slug/id field so we can join them cleanly.

with open(EDITORIAL, "r", encoding="utf-8") as f:
    # albums.json uses "id" as the slug field
    editorial = {a["id"]: a for a in json.load(f)}

# Convention: when adding a new review, set "date" in albums.json to the
# date the review page goes live (YYYY-MM-DD). The "Newest" sort on the
# reviews page depends on this field being accurate.

with open(CLEAN, "r", encoding="utf-8") as f:
    # albums_clean.json uses "slug"
    clean = {a["slug"]: a for a in json.load(f)}

with open(RECS, "r", encoding="utf-8") as f:
    # recommendations.json uses "slug"
    recs = {a["slug"]: a["recommendations"] for a in json.load(f)}

# Use the editorial list as the source of truth for which albums exist and
# what order they appear in. It's the one we curate by hand.
slugs = list(editorial.keys())

print(f"Merging {len(slugs)} albums from 3 sources...\n")


# ── Step 2: Merge into one flat dict per album ───────────────────────────────
# Field order matches the spec: editorial fields first, then Spotify metadata,
# then recommendations. Missing fields from any source default to None rather
# than raising a KeyError so the script stays robust as the catalog grows.

merged = []

for slug in slugs:
    e = editorial.get(slug, {})
    c = clean.get(slug, {})
    r = recs.get(slug, [])

    # cover_art: use the locally downloaded file in site/assets/covers/.
    # Images are downloaded by pipeline/fetch_covers.py. The path is relative
    # to the site/ root so it works from any page depth via a leading slash or
    # relative traversal. Review pages use ../assets/covers/, index uses
    # assets/covers/. We store just the filename and let each page resolve it.
    cover_art = f"assets/covers/{slug}.jpg"

    album = {
        "id":              e.get("id"),
        "title":           e.get("title"),
        "artist":          e.get("artist"),
        "year":            e.get("year"),
        "stars":           e.get("stars"),
        "second_spin":     e.get("second_spin"),
        "blurb":           e.get("blurb"),
        "review_url":      e.get("review_url"),
        # Sort genres by tag weight so the most meaningful tags appear first.
        # Review pages slice to the first 3 for display; the full list is kept
        # here so the Discover page cluster filter still has complete genre data.
        "genres":          sorted(
                               c.get("genres", e.get("genres", [])),
                               key=lambda g: c.get("tag_weights", {}).get(g.lower(), 0.0),
                               reverse=True,
                           ),
        "cover_art":       cover_art,
        "spotify_id":      c.get("spotify_id"),
        "track_count":     c.get("track_count"),
        "duration_ms":     c.get("duration_ms"),
        "popularity":      c.get("popularity"),
        "listeners":       c.get("listeners"),
        "date":            e.get("date"),
        "favorite_track":  e.get("favorite_track"),
        "track_url":       e.get("track_url"),
        "preview_url":     e.get("preview_url"),
        "recommendations": r,
    }

    merged.append(album)


# ── Step 3: Write output ─────────────────────────────────────────────────────
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)


# ── Step 4: Print summary ────────────────────────────────────────────────────
print(f"{'Album':<40} {'Stars':<7} Recommendations")
print("-" * 72)

for album in merged:
    s       = album["stars"] or 0
    stars   = "*" * s + "-" * (5 - s)
    rec_str = " / ".join(r["slug"] for r in album["recommendations"]) if album["recommendations"] else "(none)"
    print(f"  {album['title']:<38} [{stars}]  {rec_str}")

print(f"\nWritten {len(merged)} albums to {OUT_FILE.relative_to(ROOT)}")
