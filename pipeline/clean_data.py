"""
clean_data.py
Processes raw JSON from data/raw/ into a flat data/albums_clean.json.
Merges Last.fm tags with manual genres for the recommender.
"""

import json
from pathlib import Path
from tag_filters import is_junk, normalize_tag

# Config
ROOT         = Path(__file__).resolve().parent.parent
RAW_DIR      = ROOT / "data" / "raw"
ALBUMS_FILE  = ROOT / "data" / "albums.json"
LASTFM_DIR   = ROOT / "data" / "lastfm"
OUT_FILE     = ROOT / "data" / "albums_clean.json"

def _parse_lastfm_tags(tags: list) -> dict[str, float]:
    """Parse lastfm tags into a normalized name-to-weight dict."""
    DEFAULT_WEIGHT = 0.5
    seen   = set()
    result = {}

    for tag in tags:
        if isinstance(tag, dict):
            name   = str(tag.get("name", "")).strip()
            weight = float(tag.get("weight", DEFAULT_WEIGHT))
        else:
            name   = str(tag).strip()
            weight = DEFAULT_WEIGHT

        if not name or is_junk(name):
            continue

        normalized = normalize_tag(name)
        if normalized not in seen:
            seen.add(normalized)
            result[normalized] = weight

    return result

def merge_genres(manual: list[str], lastfm: list[str]) -> list[str]:
    """Merge genre lists, deduplicating case-insensitively. Manual first."""
    seen  = {g.lower() for g in manual}
    extra = [t for t in lastfm if t.lower() not in seen]
    return manual + extra

def extract_album(slug: str, raw: dict, manual_genres: list[str],
                  lastfm_weights: dict[str, float],
                  listeners: int | None,
                  playcount: int | None) -> dict:
    """Flatten raw Spotify response into a clean dictionary."""
    meta   = raw["album_metadata"]
    tracks = raw["tracks"]

    spotify_id = meta["id"]
    name       = meta["name"]
    primary_artist = meta["artists"][0]
    artist         = primary_artist["name"]
    artist_id      = primary_artist["id"]

    release_date = meta.get("release_date") or ""
    release_year = int(release_date[:4]) if len(release_date) >= 4 else 1900
    popularity = meta.get("popularity")

    genres = merge_genres(manual_genres, list(lastfm_weights.keys()))

    # tag_weights used by recommend.py for TF-IDF
    MANUAL_DEFAULT = 0.7
    tag_weights = {normalize_tag(g): MANUAL_DEFAULT for g in manual_genres}
    for name, weight in lastfm_weights.items():
        tag_weights[name] = weight

    track_count = len(tracks)
    duration_ms = sum(t.get("duration_ms", 0) for t in tracks)

    images    = meta.get("images", [])
    cover_art = images[0]["url"] if images else None

    return {
        "slug":         slug,
        "spotify_id":   spotify_id,
        "name":         name,
        "artist":       artist,
        "artist_id":    artist_id,
        "release_year": release_year,
        "popularity":   popularity,
        "genres":       genres,
        "tag_weights":  tag_weights,
        "listeners":    listeners,
        "playcount":    playcount,
        "track_count":  track_count,
        "duration_ms":  duration_ms,
        "cover_art":    cover_art,
    }

def main():
    raw_files = sorted(RAW_DIR.glob("*.json"))
    if not raw_files:
        print(f"No raw files found in {RAW_DIR}.")
        return

    with open(ALBUMS_FILE, "r", encoding="utf-8") as f:
        manual_map = {a["id"]: a.get("genres", []) for a in json.load(f)}

    print(f"Cleaning {len(raw_files)} raw file(s)...\n")
    cleaned = []

    for path in raw_files:
        slug = path.stem
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        lastfm_path = LASTFM_DIR / f"{slug}.json"
        if lastfm_path.exists():
            with open(lastfm_path, "r", encoding="utf-8") as f:
                lastfm_data = json.load(f)
            lastfm_weights = _parse_lastfm_tags(lastfm_data.get("chosen_tags", []))
            listeners      = lastfm_data.get("listeners")
            playcount      = lastfm_data.get("playcount")
        else:
            lastfm_weights = {}
            listeners      = None
            playcount      = None

        manual_genres = manual_map.get(slug, [])
        album = extract_album(slug, raw, manual_genres, lastfm_weights, listeners, playcount)
        cleaned.append(album)

        duration_min = album["duration_ms"] / 60_000
        lastfm_note  = f"  |  +{len(lastfm_weights)} lastfm tags" if lastfm_weights else ""
        print(f"  {album['name']} -- {album['artist']}")
        print(f"    {album['track_count']} tracks  |  {duration_min:.1f} min  |  popularity {album['popularity']}{lastfm_note}")
        print(f"    genres: {', '.join(album['genres']) or '(none)'}")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(cleaned)} albums written.")

if __name__ == "__main__":
    main()
