"""
fetch_lastfm.py
Enriches albums with tag data and listener counts from Last.fm.
Saves to data/lastfm/{slug}.json.
"""

import json
import os
import re
import sys
import time
from pathlib import Path
import requests
from dotenv import load_dotenv
from tag_filters import is_junk, normalize_tag

# Config
ROOT        = Path(__file__).resolve().parent.parent
ALBUMS_FILE = ROOT / "data" / "albums.json"
LASTFM_DIR  = ROOT / "data" / "lastfm"
API_BASE    = "https://ws.audioscrobbler.com/2.0/"
TOP_N        = 10
REQUEST_GAP  = 0.5

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("LASTFM_API_KEY")

if not API_KEY:
    raise EnvironmentError("LASTFM_API_KEY not found in .env.")

def extract_tags(tag_list: list, top_n: int = TOP_N) -> list[dict]:
    """Filter and normalize top N tags with their weights."""
    filtered = [
        t for t in tag_list
        if isinstance(t.get("count"), int) and not is_junk(t["name"])
    ]
    filtered.sort(key=lambda t: t["count"], reverse=True)
    return [
        {"name": normalize_tag(t["name"]), "weight": round(t["count"] / 100.0, 4)}
        for t in filtered[:top_n]
    ]

def fetch_album_info(artist: str, album: str) -> tuple[int | None, int | None]:
    """Get listeners and playcount for an album."""
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
        listeners = int(a["listeners"]) if a.get("listeners") else None
        playcount = int(a["playcount"]) if a.get("playcount") else None
        return listeners, playcount
    except Exception as exc:
        print(f"    [warn] album.getinfo failed: {exc}")
        return None, None

def fetch_album_tags(artist: str, album: str) -> tuple[list, dict]:
    """Get tags for an album."""
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
        if isinstance(tags, dict):
            tags = [tags]
        return extract_tags(tags), data
    except Exception as exc:
        print(f"    [warn] album.gettoptags failed: {exc}")
        return [], {}

def fetch_artist_tags(artist: str) -> tuple[list, dict]:
    """Get tags for an artist."""
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
        print(f"Skipping {skipped} already-fetched album(s).\n")
    if not to_fetch:
        print("All albums already fetched.")
        return

    print(f"Fetching Last.fm tags for {len(to_fetch)} album(s)...\n")

    for album in to_fetch:
        slug   = album["id"]
        title  = album["title"]
        artist = album["artist"]

        print(f"  {title} - {artist}")

        album_tags, album_raw = fetch_album_tags(artist, title)
        time.sleep(REQUEST_GAP)
        artist_tags, artist_raw = fetch_artist_tags(artist)
        time.sleep(REQUEST_GAP)
        listeners, playcount = fetch_album_info(artist, title)
        time.sleep(REQUEST_GAP)

        # Fall back to artist tags if album tags are sparse
        if len(album_tags) >= 3:
            chosen_tags = album_tags
            source = "album"
        else:
            chosen_tags = artist_tags
            source = "artist"

        # Filter tags matching artist name
        artist_key = re.sub(r'[^a-z0-9]', '', artist.lower())
        chosen_tags = [
            t for t in chosen_tags
            if re.sub(r'[^a-z0-9]', '', t["name"]) != artist_key
        ]

        print(f"    source: {source}  |  tags: {len(chosen_tags)}")

        out = {
            "slug":         slug,
            "chosen_tags":  chosen_tags,
            "listeners":    listeners,
            "playcount":    playcount,
            "album_raw":    album_raw,
            "artist_raw":   artist_raw,
        }
        with open(LASTFM_DIR / f"{slug}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nDone.")

if __name__ == "__main__":
    main()
