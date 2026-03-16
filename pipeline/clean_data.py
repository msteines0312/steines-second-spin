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
import re
from pathlib import Path


# ── Config ───────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent   # project root
RAW_DIR      = ROOT / "data" / "raw"
ALBUMS_FILE  = ROOT / "data" / "albums.json"   # manual editorial data
LASTFM_DIR   = ROOT / "data" / "lastfm"        # raw Last.fm responses
OUT_FILE     = ROOT / "data" / "albums_clean.json"
# ─────────────────────────────────────────────────────────────────────────────


# ── Tag normalization (mirrors fetch_lastfm.py, applied again here so that
#    stale lastfm JSON files are cleaned even if they predate these rules) ────
_VALID_TAG_RE = re.compile(r"^[a-zA-Z0-9 \-&]+$")

# Known misspellings/garbage that pass the character filter — exact substrings
_JUNK_TAGS = {"synth fumk"}

def _normalize_tag(tag: str) -> str:
    """Lowercase, then apply canonical normalizations.

    Uses word-boundary substitution for "rnb" so that compound tags like
    "alternative rnb" are also corrected to "alternative r&b".
    """
    t = tag.lower().strip()
    t = t.replace("hip hop", "hip-hop")        # exact phrase replacement
    t = re.sub(r"\brnb\b", "r&b", t)           # word-boundary: catches "alternative rnb" too
    return t


def _clean_lastfm_tags(tags: list[str]) -> list[str]:
    """Re-apply normalization and character filtering to saved lastfm tags.

    Catches tags that were persisted before the current filtering rules were
    in place (e.g. typos like "synth fumk", un-normalized "hip hop").
    Deduplicates after normalization so two tags that collapse to the same
    string (e.g. "hip hop" and "Hip Hop") don't both appear.
    """
    seen    = set()
    cleaned = []
    for tag in tags:
        tag = tag.strip()
        if not _VALID_TAG_RE.match(tag):
            continue                            # drop tags with bad characters
        if tag.lower() in _JUNK_TAGS:
            continue                            # drop known misspellings
        normalized = _normalize_tag(tag)
        if normalized not in seen:
            seen.add(normalized)
            cleaned.append(normalized)
    return cleaned


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
                  lastfm_tags: list[str]) -> dict:
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
    lastfm_tags : list[str]
        Tags fetched from the Last.fm API (empty list if not available).

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
    release_year = int(meta["release_date"][:4])

    # --- Popularity --- (0-100 score from Spotify, based on recent stream counts)
    popularity = meta.get("popularity")

    # --- Genres ---
    # Spotify's album-level genres field is almost always empty — genres are
    # attached to artists in Spotify's data model. We use the manually curated
    # tags from albums.json merged with any Last.fm tags fetched by
    # fetch_lastfm.py instead. Manual tags come first; Last.fm tags are
    # appended if they aren't already represented.
    genres = merge_genres(manual_genres, lastfm_tags)

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
        "genres":       genres,
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

        # Load Last.fm tags if available (produced by fetch_lastfm.py).
        # Re-apply normalization so stale JSON files are cleaned on the fly.
        lastfm_path = LASTFM_DIR / f"{slug}.json"
        if lastfm_path.exists():
            with open(lastfm_path, "r", encoding="utf-8") as f:
                raw_tags = json.load(f).get("chosen_tags", [])
            lastfm_tags = _clean_lastfm_tags(raw_tags)
        else:
            lastfm_tags = []

        manual_genres = manual_map.get(slug, [])

        album = extract_album(slug, raw, manual_genres, lastfm_tags)
        cleaned.append(album)

        duration_min = album["duration_ms"] / 60_000
        lastfm_note  = f"  |  +{len(lastfm_tags)} lastfm tags" if lastfm_tags else ""
        print(f"  {album['name']} -- {album['artist']}")
        print(f"    {album['track_count']} tracks  |  {duration_min:.1f} min  |  popularity {album['popularity']}{lastfm_note}")
        print(f"    genres: {', '.join(album['genres']) or '(none)'}")

    # Write the cleaned array, pretty-printed so it's readable as a reference file
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(cleaned)} albums written to {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
