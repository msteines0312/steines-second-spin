"""
build_data.py
Merges editorial, clean Spotify metadata, and recommendations into albums_final.json.
"""

import json
from pathlib import Path

# Config
ROOT         = Path(__file__).resolve().parent.parent

EDITORIAL    = ROOT / "data" / "albums.json"
CLEAN        = ROOT / "data" / "albums_clean.json"
RECS         = ROOT / "data" / "recommendations.json"
OUT_DIR      = ROOT / "data"
OUT_FILE     = OUT_DIR / "albums_final.json"
# ─────────────────────────────────────────────────────────────────────────────

def main():
    with open(EDITORIAL, "r", encoding="utf-8") as f:
        editorial = {a["id"]: a for a in json.load(f)}

    with open(CLEAN, "r", encoding="utf-8") as f:
        clean = {a["slug"]: a for a in json.load(f)}

    with open(RECS, "r", encoding="utf-8") as f:
        recs = {a["slug"]: a["recommendations"] for a in json.load(f)}

    slugs = list(editorial.keys())
    print(f"Merging {len(slugs)} albums from 3 sources...\n")

    merged = []
    for slug in slugs:
        e = editorial.get(slug, {})
        c = clean.get(slug, {})
        r = recs.get(slug, [])

        cover_art = f"assets/covers/{slug}.webp"

        album = {
            "id":              e.get("id"),
            "title":           e.get("title"),
            "artist":          e.get("artist"),
            "year":            e.get("year"),
            "stars":           e.get("stars"),
            "second_spin":     e.get("second_spin"),
            "blurb":           e.get("blurb"),
            "review_url":      e.get("review_url"),
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"Written {len(merged)} albums to {OUT_FILE.name}")

if __name__ == "__main__":
    main()
