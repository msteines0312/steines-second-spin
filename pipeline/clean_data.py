"""
clean_data.py
Reads every raw JSON file from data/raw/, extracts a flat, normalized
dictionary for each album, and writes the result to data/albums_clean.json.

Genre enrichment: if data/lastfm/{slug}.json exists (produced by
fetch_lastfm.py), its tags are merged with the manually curated genres from
data/albums.json to produce a richer combined genre list for the recommender.

This is step 2 of the pipeline:
    fetch_spotify.py  ->  fetch_lastfm.py  ->  clean_data.py
                      ->  recommend.py  ->  build_data.py

Run from the project root:
    python pipeline/clean_data.py
"""

import json
from pathlib import Path

from tag_filters import is_junk, normalize_tag


# ── Config ───────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent   # project root
RAW_DIR      = ROOT / "data" / "raw"
ALBUMS_FILE  = ROOT / "data" / "albums.json"   # manual editorial data
LASTFM_DIR   = ROOT / "data" / "lastfm"        # raw Last.fm responses
OUT_FILE     = ROOT / "data" / "albums_clean.json"
# ─────────────────────────────────────────────────────────────────────────────


def _parse_lastfm_tags(tags: list) -> dict[str, float]:
    """Parse lastfm chosen_tags into a normalized name-to-weight dict.

    Handles both the old string format (["r&b", "pop"]) and the new weighted
    dict format ([{"name": "r&b", "weight": 0.95}]) so that existing cached
    lastfm files continue to work before fetch_lastfm.py is re-run.

    Old-format tags receive a default weight of 0.5 since their original
    Last.fm counts were not stored. Re-applies normalization and character
    filtering so stale files are cleaned on the fly.

    Returns
    -------
    dict[str, float]
        Normalized tag name -> weight (0.0-1.0), deduped.
    """
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
    """Merge two genre lists, deduplicating case-insensitively.

    Manual genres are preserved first (editorial intent wins); Last.fm tags
    that aren't already represented are appended after.

    Parameters
    ----------
    manual : list[str]
        Genres from albums.json (hand-curated).
    lastfm : list[str]
        Tags extracted from the Last.fm API response.

    Returns
    -------
    list[str]
        Merged, deduplicated list with manual genres first.
    """
    seen  = {g.lower() for g in manual}
    extra = [t for t in lastfm if t.lower() not in seen]
    return manual + extra


def extract_album(slug: str, raw: dict, manual_genres: list[str],
                  lastfm_weights: dict[str, float],
                  listeners: int | None,
                  playcount: int | None) -> dict:
    """Flatten one raw Spotify response into a clean, analysis-ready dictionary.

    Parameters
    ----------
    slug : str
        The filename stem (e.g. "mr-morale"). Used as our internal identifier
        so records stay linkable back to data/albums.json.
    raw : dict
        The full JSON object loaded from data/raw/{slug}.json, with keys
        "album_metadata", "tracks", and "audio_features".
    manual_genres : list[str]
        Hand-curated genre tags from albums.json.
    lastfm_weights : dict[str, float]
        Tag name -> weight (0.0-1.0) from the Last.fm API. Empty dict if not available.
    listeners : int | None
        Last.fm unique listener count. None if unavailable.
    playcount : int | None
        Last.fm total play count. None if unavailable.

    Returns
    -------
    dict
        Flat dictionary with the fields documented below.
    """
    meta   = raw["album_metadata"]
    tracks = raw["tracks"]          # full paginated track list (our top-level key)

    # --- Identity fields ---
    spotify_id = meta["id"]
    name       = meta["name"]

    # Artists is a list (collaborations exist), but we always want the primary
    primary_artist = meta["artists"][0]
    artist         = primary_artist["name"]
    artist_id      = primary_artist["id"]

    # --- Release year ---
    # release_date can be "YYYY-MM-DD", "YYYY-MM", or just "YYYY" depending on
    # how Spotify stores it. Slicing the first 4 chars handles all three formats.
    # Guard against missing or malformed dates (Spotify occasionally omits this).
    release_date = meta.get("release_date") or ""
    release_year = int(release_date[:4]) if len(release_date) >= 4 else 1900

    # --- Popularity --- (0-100 score from Spotify, based on recent stream counts)
    popularity = meta.get("popularity")

    # --- Genres (display list) ---
    # Spotify's album-level genres field is almost always empty - genres are
    # attached to artists in Spotify's data model. We use the manually curated
    # tags from albums.json merged with any Last.fm tags fetched by
    # fetch_lastfm.py instead. Manual tags come first; Last.fm tags are
    # appended if they aren't already represented.
    genres = merge_genres(manual_genres, list(lastfm_weights.keys()))

    # --- Tag weights (ML feature dict) ---
    # Combines editorial genres (fixed default weight) with Last.fm weights
    # (actual association scores). This dict is used by recommend.py to build
    # a TF-IDF matrix. Stored separately from the display genres list so the
    # frontend pill count stays manageable while the ML model gets richer signal.
    MANUAL_DEFAULT = 0.7   # editorial intent is a strong but unquantified signal
    tag_weights = {normalize_tag(g): MANUAL_DEFAULT for g in manual_genres}
    for name, weight in lastfm_weights.items():
        # Last.fm weights override the manual default for the same tag
        tag_weights[name] = weight

    # --- Track-level aggregations ---
    # Use the top-level "tracks" list (fully paginated by fetch_spotify.py),
    # NOT the embedded meta["tracks"] object, which may be incomplete for
    # double albums that exceed Spotify's default 50-track page size.
    track_count = len(tracks)
    duration_ms = sum(t.get("duration_ms", 0) for t in tracks)

    # --- Cover art ---
    # images is sorted largest-to-smallest by Spotify. Index 0 is always 640x640.
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
        "genres":       genres,        # merged display list (manual + Last.fm)
        "tag_weights":  tag_weights,   # ML feature dict (name -> weight)
        "listeners":    listeners,     # Last.fm unique listener count
        "playcount":    playcount,     # Last.fm total play count
        "track_count":  track_count,
        "duration_ms":  duration_ms,
        "cover_art":    cover_art,
    }


def main():
    raw_files = sorted(RAW_DIR.glob("*.json"))

    if not raw_files:
        print(f"No raw files found in {RAW_DIR}. Run fetch_spotify.py first.")
        return

    # Load manual genres keyed by slug so we can look them up per album
    with open(ALBUMS_FILE, "r", encoding="utf-8") as f:
        manual_map = {a["id"]: a.get("genres", []) for a in json.load(f)}

    print(f"Cleaning {len(raw_files)} raw file(s)...\n")

    cleaned = []

    for path in raw_files:
        slug = path.stem   # filename without .json extension
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Load Last.fm data if available (produced by fetch_lastfm.py).
        # _parse_lastfm_tags handles both the old string format and the new
        # weighted dict format, so existing cached files work without re-fetching.
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
        lastfm_note  = f"  |  +{len(lastfm_weights)} lastfm tags (weighted)" if lastfm_weights else ""
        print(f"  {album['name']} -- {album['artist']}")
        print(f"    {album['track_count']} tracks  |  {duration_min:.1f} min  |  popularity {album['popularity']}{lastfm_note}")
        print(f"    genres: {', '.join(album['genres']) or '(none)'}")

    # Write the cleaned array, pretty-printed so it's readable as a reference file
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(cleaned)} albums written to {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
