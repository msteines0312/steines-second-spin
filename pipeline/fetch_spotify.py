"""
fetch_spotify.py
Fetches album metadata and track features from Spotify for albums in data/albums.json.
Saves to data/raw/{id}.json.
"""

import json
import sys
import time
from pathlib import Path
import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials

# Config
ROOT      = Path(__file__).resolve().parent.parent
ALBUMS_IN = ROOT / "data" / "albums.json"
RAW_OUT   = ROOT / "data" / "raw"
RATE_LIMIT_DELAY = 0.5

def extract_spotify_id(spotify_url: str) -> str:
    """Parse a Spotify album URL and return the album ID."""
    return spotify_url.rstrip("/").split("/")[-1]

def fetch_album(sp: spotipy.Spotify, spotify_album_id: str) -> dict:
    """Fetch metadata, tracks, and audio features for an album."""
    album_meta = sp.album(spotify_album_id)

    tracks = []
    page = sp.album_tracks(spotify_album_id, limit=50)
    while page:
        tracks.extend(page["items"])
        page = sp.next(page) if page["next"] else None

    track_ids = [t["id"] for t in tracks if t.get("id")]
    audio_features = []

    for i in range(0, len(track_ids), 100):
        batch = track_ids[i : i + 100]
        try:
            result = sp.audio_features(batch)
            if result:
                audio_features.extend(result)
        except spotipy.SpotifyException as e:
            print(f"     [WARNING] Audio features unavailable: {e.msg}")
            break

    return {
        "album_metadata": album_meta,
        "tracks": tracks,
        "audio_features": audio_features,
    }

def main():
    force = "--force" in sys.argv
    load_dotenv()
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

    with open(ALBUMS_IN, "r", encoding="utf-8") as f:
        albums = json.load(f)

    RAW_OUT.mkdir(parents=True, exist_ok=True)

    to_fetch = albums if force else [a for a in albums if not (RAW_OUT / f"{a['id']}.json").exists()]
    skipped  = len(albums) - len(to_fetch)

    if skipped:
        print(f"Skipping {skipped} already-fetched album(s).\n")
    if not to_fetch:
        print("All albums already fetched.")
        return

    print(f"Fetching Spotify data for {len(to_fetch)} album(s)...\n")

    for entry in to_fetch:
        slug       = entry["id"]
        spotify_id = extract_spotify_id(entry["spotify_url"])

        print(f"  -> {entry['title']} by {entry['artist']}")

        try:
            data = fetch_album(sp, spotify_id)
            out_path = RAW_OUT / f"{slug}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"     {len(data['tracks'])} tracks -> {out_path.name}")

        except spotipy.SpotifyException as e:
            print(f"     [ERROR] Spotify API error: {e}")

        time.sleep(RATE_LIMIT_DELAY)

if __name__ == "__main__":
    main()
