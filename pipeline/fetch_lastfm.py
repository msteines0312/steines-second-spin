"""
fetch_lastfm.py
Enriches each album in data/albums.json with tag data from the Last.fm API.

For each album we make two calls:
  1. album.gettoptags  — genre/style tags specific to this release
  2. artist.gettoptags — fallback for when album-level tags are sparse

The top 5 filtered tags are extracted from whichever call returns more useful
results. Raw API responses are saved to data/lastfm/{slug}.json for debugging.

This is an optional enrichment step that runs before clean_data.py so that
the merged genre tags are available for the recommendation engine:

    fetch_spotify.py  ->  fetch_lastfm.py  ->  clean_data.py
                      ->  recommend.py  ->  build_data.py

Run from the project root:
    python pipeline/fetch_lastfm.py
"""

import json
import time
from pathlib import Path

import re

import requests
from dotenv import load_dotenv
import os


# ── Config ───────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
ALBUMS_FILE = ROOT / "data" / "albums.json"
LASTFM_DIR  = ROOT / "data" / "lastfm"
API_BASE    = "http://ws.audioscrobbler.com/2.0/"

TOP_N        = 5     # max tags to keep per album
REQUEST_GAP  = 0.5   # seconds between API calls (be polite)

# Tags to always discard — garbage or meta tags that don't describe genre/style
JUNK_SUBSTRINGS = [
    "seen live",
    "favourite",
    "favorites",
    "under 2000 listeners",
]

# Exact-match tags to discard (case-insensitive).
# Covers mood tags, geographic tags, award tags, and overly generic labels
# that carry no useful signal for genre-based recommendations.
_JUNK_EXACT = {
    "short", "happy", "chill", "indie", "aoty",
    "usa", "american", "texas",
    "2020s", "90s", "80s", "70s", "60s",
}
# ─────────────────────────────────────────────────────────────────────────────


load_dotenv(ROOT / ".env")
API_KEY = os.getenv("LASTFM_API_KEY")

if not API_KEY:
    raise EnvironmentError(
        "LASTFM_API_KEY not found in .env. "
        "Add LASTFM_API_KEY=<your_key> and try again."
    )


# Only these characters are allowed in a clean tag.
# Anything else (typos like "fumk", stray punctuation, emoji, etc.) gets dropped.
_VALID_TAG_RE = re.compile(r"^[a-zA-Z0-9 \-&]+$")

# Exact-match normalizations applied after lowercasing.
# Add entries here as new inconsistencies surface.
_NORMALIZATIONS = {
    "hip hop": "hip-hop",
    "rnb":     "r&b",
}


def normalize_tag(name: str) -> str:
    """Lowercase and apply canonical normalizations to a tag name."""
    lower = name.lower().strip()
    return _NORMALIZATIONS.get(lower, lower)


def is_junk(tag_name: str) -> bool:
    """Return True if this tag should be filtered out.

    Filters:
    - Tags shorter than 3 characters
    - Purely numeric tags (bare numbers like "2006")
    - Tags containing known junk substrings (case-insensitive)
    - Tags with characters outside [a-z A-Z 0-9 space hyphen ampersand]
      — catches typos, stray punctuation, and non-ASCII garbage
    """
    name = tag_name.strip()
    if len(name) <= 2:
        return True
    if name.isdigit():
        return True
    # Decade tags ("2020s", "90s") and any other tags starting with a digit
    if name[0].isdigit():
        return True
    lower = name.lower()
    if lower in _JUNK_EXACT:
        return True
    if any(junk in lower for junk in JUNK_SUBSTRINGS):
        return True
    if not _VALID_TAG_RE.match(name):
        return True
    return False


def extract_tags(tag_list: list, top_n: int = TOP_N) -> list[str]:
    """Filter, normalize, and return the top N tag names from a Last.fm tag list.

    Parameters
    ----------
    tag_list : list
        List of {"name": str, "count": int, "url": str} dicts from Last.fm.
    top_n : int
        How many tags to keep after filtering.

    Returns
    -------
    list[str]
        Filtered and normalized tag names, sorted by count descending.
    """
    filtered = [
        t for t in tag_list
        if isinstance(t.get("count"), int) and not is_junk(t["name"])
    ]
    # Last.fm returns tags pre-sorted by count, but sort explicitly to be safe
    filtered.sort(key=lambda t: t["count"], reverse=True)
    return [normalize_tag(t["name"]) for t in filtered[:top_n]]


def fetch_album_tags(artist: str, album: str) -> tuple[list, dict]:
    """Call album.gettoptags and return (filtered_tags, raw_response).

    Returns ([], {}) on any network or API error rather than crashing.
    """
    params = {
        "method":  "album.gettoptags",
        "artist":  artist,
        "album":   album,
        "api_key": API_KEY,
        "format":  "json",
    }
    try:
        resp = requests.get(API_BASE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        tags = data.get("toptags", {}).get("tag", [])
        # Last.fm returns a dict (not list) when there's only one tag
        if isinstance(tags, dict):
            tags = [tags]
        return extract_tags(tags), data
    except Exception as exc:
        print(f"    [warn] album.gettoptags failed: {exc}")
        return [], {}


def fetch_artist_tags(artist: str) -> tuple[list, dict]:
    """Call artist.gettoptags and return (filtered_tags, raw_response).

    Returns ([], {}) on any network or API error rather than crashing.
    """
    params = {
        "method":  "artist.gettoptags",
        "artist":  artist,
        "api_key": API_KEY,
        "format":  "json",
    }
    try:
        resp = requests.get(API_BASE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        tags = data.get("toptags", {}).get("tag", [])
        if isinstance(tags, dict):
            tags = [tags]
        return extract_tags(tags), data
    except Exception as exc:
        print(f"    [warn] artist.gettoptags failed: {exc}")
        return [], {}


def main():
    LASTFM_DIR.mkdir(parents=True, exist_ok=True)

    with open(ALBUMS_FILE, "r", encoding="utf-8") as f:
        albums = json.load(f)

    print(f"Fetching Last.fm tags for {len(albums)} albums...\n")

    for album in albums:
        slug   = album["id"]
        title  = album["title"]
        artist = album["artist"]

        print(f"  {title} — {artist}")

        # ── Call 1: Album-level tags ─────────────────────────────────────────
        album_tags, album_raw = fetch_album_tags(artist, title)
        time.sleep(REQUEST_GAP)

        # ── Call 2: Artist-level tags (always fetch; use as fallback) ────────
        artist_tags, artist_raw = fetch_artist_tags(artist)
        time.sleep(REQUEST_GAP)

        # ── Choose the better set ────────────────────────────────────────────
        # Prefer album tags when we get at least 3 results; otherwise fall back
        # to artist tags which tend to be more populated.
        if len(album_tags) >= 3:
            chosen_tags = album_tags
            source = "album"
        else:
            chosen_tags = artist_tags
            source = "artist (fallback)"

        print(f"    source: {source}  |  tags: {', '.join(chosen_tags) or '(none)'}")

        # ── Save raw responses to data/lastfm/{slug}.json ────────────────────
        out = {
            "slug":         slug,
            "chosen_tags":  chosen_tags,
            "album_raw":    album_raw,
            "artist_raw":   artist_raw,
        }
        with open(LASTFM_DIR / f"{slug}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Raw responses saved to {LASTFM_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
