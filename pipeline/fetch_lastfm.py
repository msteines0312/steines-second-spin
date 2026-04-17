"""
fetch_lastfm.py
Enriches each album in data/albums.json with tag data from the Last.fm API.

For each album we make two calls:
  1. album.gettoptags  — genre/style tags specific to this release
  2. artist.gettoptags — fallback for when album-level tags are sparse

The top 5 filtered tags are extracted from whichever call returns more useful
results. Raw API responses are saved to data/lastfm/{slug}.json for debugging.

By default, albums that already have a file in data/lastfm/ are skipped so
that re-running the script only fetches new additions. Pass --force to
re-fetch every album regardless.

This is an optional enrichment step that runs before clean_data.py so that
the merged genre tags are available for the recommendation engine:

    fetch_spotify.py  ->  fetch_lastfm.py  ->  clean_data.py
                      ->  recommend.py  ->  build_data.py

Run from the project root:
    python pipeline/fetch_lastfm.py           # skip already-fetched albums
    python pipeline/fetch_lastfm.py --force   # re-fetch everything
"""

import json
import sys
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

TOP_N        = 10    # max tags to keep per album (more tags = richer TF-IDF signal)
REQUEST_GAP  = 0.5   # seconds between API calls (be polite)

# Tags to always discard — garbage or meta tags that don't describe genre/style.
# Substring matches are checked case-insensitively against the full tag name.
JUNK_SUBSTRINGS = [
    "seen live",
    "favourite",
    "favorites",
    "under 2000 listeners",
    "best of",           # ranking tags: "best of 2020", "best of all time"
    "album of the year",
    "i love",            # personal collection tags: "i love singing along"
    "i like",
    "i can",             # "albums i can listen to all the way through"
    "i own",             # "cd i own", "albums i own"
    "where is",          # joke tags: "where is my bong"
    "my top",            # list tags: "my top songs"
    "fav ",              # "fav albums" — trailing space avoids matching "favourite"
    "cum",               # vulgar joke tags: "cum metal", "cum music"
    "goodness",          # "shoegazing goodness" and similar praise-noise tags
]

# Exact-match tags to discard (case-insensitive).
_JUNK_EXACT = {
    # Vague genre labels (too broad to add useful signal)
    "indie", "alternative", "progressive",
    # Generic meta / collection tags
    "aoty", "albums", "all", "classic albums", "vinyl", "hiphop",
    # Quality / rating / hype tags
    "peak", "goated", "goat", "masterpiece", "hot", "hype",
    "fresh", "beautiful", "classic", "special", "good", "great",
    "good stuff", "full", "long", "new", "slaps", "drumless",
    # Mood / descriptor tags (not genre descriptors)
    "atmospheric", "melancholic", "melancholy", "melodic", "energetic",
    "aggressive", "sexual", "romantic", "hedonistic", "playful", "powerful",
    "positive", "warm", "complex", "yearning", "sweet", "soothing", "dreamy",
    "intense", "bombastic", "witty", "heartbreak", "drugs", "nocturnal",
    "emotional", "moody", "boisterous", "rhythmic", "bass", "heavy",
    "funky", "introspective", "sensual", "sexy", "fun", "rage", "slow jams",
    # Geographic tags
    "usa", "american", "texas", "california", "london", "los angeles",
    "toronto", "new jersey", "sweden", "swedish", "british", "canadian",
    "southern", "urban", "united states", "america", "colombian", "alabama",
    # Decade tags
    "2020s", "2010s", "90s", "80s", "70s", "60s",
    # Demographic tags
    "male vocalists", "female vocalists", "male vocals", "female vocalist", "diva",
    # Geographic tags (expanded)
    "maryland", "ohio", "georgia", "florida", "michigan", "chicago",
    "new york", "detroit", "atlanta", "houston", "brooklyn", "queens",
    "uk", "england", "united kingdom", "australian", "irish", "german", "french", "japanese",
    # Label / industry tags
    "domino", "domino records", "columbia", "interscope", "def jam",
    "atlantic", "republic", "sub pop", "matador", "4ad",
    # Community / random noise
    "happy", "short", "chill", "wikipedia", "fob", "mixtaperoom",
    "xotwod", "windmill scene", "post-darling-core", "elevator music",
    "science", "cover", "bbc", "deposito", "prosty", "synth fumk",
    "pussy music",
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
    # Tags starting with a digit (decade tags like "90s", "2020s")
    if name[0].isdigit():
        return True
    # Tags containing a 4-digit sequence (year-based community tags like "ch2023")
    if re.search(r'\d{4}', name):
        return True
    lower = name.lower()
    if lower in _JUNK_EXACT:
        return True
    if any(junk in lower for junk in JUNK_SUBSTRINGS):
        return True
    if not _VALID_TAG_RE.match(name):
        return True
    return False


def extract_tags(tag_list: list, top_n: int = TOP_N) -> list[dict]:
    """Filter, normalize, and return the top N tags with their weights.

    Parameters
    ----------
    tag_list : list
        List of {"name": str, "count": int, "url": str} dicts from Last.fm.
    top_n : int
        How many tags to keep after filtering.

    Returns
    -------
    list[dict]
        Each dict has "name" (normalized tag string) and "weight" (float 0.0-1.0).
        Weight is the Last.fm count normalized to [0, 1] — higher means stronger
        association. Sorted by weight descending.
    """
    filtered = [
        t for t in tag_list
        if isinstance(t.get("count"), int) and not is_junk(t["name"])
    ]
    # Last.fm returns tags pre-sorted by count, but sort explicitly to be safe
    filtered.sort(key=lambda t: t["count"], reverse=True)
    return [
        {"name": normalize_tag(t["name"]), "weight": round(t["count"] / 100.0, 4)}
        for t in filtered[:top_n]
    ]


def fetch_album_info(artist: str, album: str) -> tuple[int | None, int | None]:
    """Call album.getinfo and return (listeners, playcount).

    Last.fm scrobble counts serve as a proxy for mainstream reach in the
    recommendation engine — useful for distinguishing underground vs. widely-
    heard albums (replaces Spotify popularity, which returns null under
    Client Credentials auth).

    Returns (None, None) on any network or API error.
    """
    params = {
        "method":  "album.getinfo",
        "artist":  artist,
        "album":   album,
        "api_key": API_KEY,
        "format":  "json",
    }
    try:
        resp = requests.get(API_BASE, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        a = data.get("album", {})
        # Last.fm returns listener/playcount as strings, not ints
        listeners = int(a["listeners"]) if a.get("listeners") else None
        playcount = int(a["playcount"]) if a.get("playcount") else None
        return listeners, playcount
    except Exception as exc:
        print(f"    [warn] album.getinfo failed: {exc}")
        return None, None


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
    force = "--force" in sys.argv

    LASTFM_DIR.mkdir(parents=True, exist_ok=True)

    with open(ALBUMS_FILE, "r", encoding="utf-8") as f:
        albums = json.load(f)

    to_fetch = albums if force else [a for a in albums if not (LASTFM_DIR / f"{a['id']}.json").exists()]
    skipped  = len(albums) - len(to_fetch)

    if skipped:
        print(f"Skipping {skipped} already-fetched album(s). Pass --force to re-fetch all.\n")
    if not to_fetch:
        print("All albums already fetched. Nothing to do.")
        return

    print(f"Fetching Last.fm tags for {len(to_fetch)} album(s)...\n")

    for album in to_fetch:
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

        # ── Call 3: Album info (listeners, playcount) ────────────────────────
        listeners, playcount = fetch_album_info(artist, title)
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

        # Remove tags that are just the artist's name — common on Last.fm.
        # Normalize by stripping non-alphanumeric chars so "JAY-Z" matches "jay z".
        artist_key = re.sub(r'[^a-z0-9]', '', artist.lower())
        chosen_tags = [
            t for t in chosen_tags
            if re.sub(r'[^a-z0-9]', '', t["name"]) != artist_key
        ]

        tag_summary = ", ".join(f"{t['name']}({t['weight']:.2f})" for t in chosen_tags) or "(none)"
        print(f"    source: {source}  |  tags: {tag_summary}")
        print(f"    listeners: {listeners:,}" if listeners is not None else "    listeners: n/a")

        # ── Save raw responses to data/lastfm/{slug}.json ────────────────────
        out = {
            "slug":         slug,
            "chosen_tags":  chosen_tags,   # [{name, weight}, ...] — weights are Last.fm counts / 100
            "listeners":    listeners,
            "playcount":    playcount,
            "album_raw":    album_raw,
            "artist_raw":   artist_raw,
        }
        with open(LASTFM_DIR / f"{slug}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(to_fetch)} album(s) saved to {LASTFM_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
