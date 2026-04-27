"""
fetch_spotify.py
Connects to the Spotify Web API and fetches album metadata and track-level
audio features for every album listed in data/albums.json. Saves one raw
JSON file per album to data/raw/{album-id}.json.

By default, albums that already have a file in data/raw/ are skipped so that
re-running the script only fetches new additions. Pass --force to re-fetch
every album regardless.

Run from the project root:
    python pipeline/fetch_spotify.py           # skip already-fetched albums
    python pipeline/fetch_spotify.py --force   # re-fetch everything
"""

import json
import sys
import time
from pathlib import Path

import spotipy
from dotenv import load_dotenv
from spotipy.oauth2 import SpotifyClientCredentials


# ── Config ───────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent   # project root
ALBUMS_IN = ROOT / "data" / "albums.json"
RAW_OUT   = ROOT / "data" / "raw"

# Polite pause between album requests (seconds) to avoid hammering the API
RATE_LIMIT_DELAY = 0.5
# ─────────────────────────────────────────────────────────────────────────────


def extract_spotify_id(spotify_url: str) -> str:
    """Parse a Spotify album URL and return the bare album ID.

    Parameters
    ----------
    spotify_url : str
        Full Spotify URL, e.g. "https://open.spotify.com/album/79ONNoS4M9tfIA1mYLBYVX"

    Returns
    -------
    str
        The 22-character Spotify album ID at the end of the URL.
    """
    return spotify_url.rstrip("/").split("/")[-1]


def fetch_album(sp: spotipy.Spotify, spotify_album_id: str) -> dict:
    """Fetch all available data for a single album from the Spotify API.

    Pulls three things:
      1. Full album metadata (name, release date, cover images, label, etc.)
      2. A flat list of every track on the album, handling pagination so that
         double albums (like Mr. Morale & The Big Steppers) are fully covered.
      3. Audio features per track - danceability, energy, valence, tempo, etc.
         These may come back as None if your Spotify app hasn't been granted
         extended quota mode. The rest of the pipeline treats missing features
         as optional, so this is non-fatal.

    Parameters
    ----------
    sp : spotipy.Spotify
        Authenticated Spotify client.
    spotify_album_id : str
        Bare Spotify album ID (not the full URL).

    Returns
    -------
    dict
        Keys: "album_metadata" (dict), "tracks" (list), "audio_features" (list)
    """
    # 1. Full album metadata
    album_meta = sp.album(spotify_album_id)

    # 2. Track list - Spotify paginates at 50 tracks per page, so we loop
    #    until response["next"] is None. Most albums fit in one page.
    tracks = []
    page = sp.album_tracks(spotify_album_id, limit=50)
    while page:
        tracks.extend(page["items"])
        page = sp.next(page) if page["next"] else None

    # 3. Audio features - batched in groups of 100 (Spotify's max per request).
    #    This has its own try/except so a tier-restriction error on this endpoint
    #    doesn't prevent the metadata and tracks (fetched above) from being saved.
    track_ids = [t["id"] for t in tracks if t.get("id")]
    audio_features = []

    for i in range(0, len(track_ids), 100):
        batch = track_ids[i : i + 100]
        try:
            result = sp.audio_features(batch)
            # NOTE: sp.audio_features returns None for the whole batch if your Spotify
            # app doesn't have extended quota mode access. As of 2024, Spotify restricts
            # this endpoint to approved apps. None here is an API tier issue, not a bug.
            if result:
                audio_features.extend(result)
        except spotipy.SpotifyException as e:
            print(f"     [WARNING] Audio features unavailable (HTTP {e.http_status}): {e.msg}")
            break  # no point retrying further batches if the endpoint is restricted

    return {
        "album_metadata": album_meta,
        "tracks": tracks,
        "audio_features": audio_features,
    }


def main():
    force = "--force" in sys.argv

    # Load SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET from .env
    load_dotenv()
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

    with open(ALBUMS_IN, "r", encoding="utf-8") as f:
        albums = json.load(f)

    RAW_OUT.mkdir(parents=True, exist_ok=True)

    to_fetch = albums if force else [a for a in albums if not (RAW_OUT / f"{a['id']}.json").exists()]
    skipped  = len(albums) - len(to_fetch)

    if skipped:
        print(f"Skipping {skipped} already-fetched album(s). Pass --force to re-fetch all.\n")
    if not to_fetch:
        print("All albums already fetched. Nothing to do.")
        return

    print(f"Fetching Spotify data for {len(to_fetch)} album(s)...\n")

    for entry in to_fetch:
        slug       = entry["id"]
        spotify_id = extract_spotify_id(entry["spotify_url"])

        print(f"  -> {entry['title']} by {entry['artist']}  [{spotify_id}]")

        try:
            data = fetch_album(sp, spotify_id)

            out_path = RAW_OUT / f"{slug}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            track_count   = len(data["tracks"])
            feature_count = sum(1 for feat in data["audio_features"] if feat is not None)
            print(f"     {track_count} tracks, {feature_count} audio feature rows  ->  {out_path.name}")

        except spotipy.SpotifyException as e:
            print(f"     [ERROR] Spotify API error: {e}")

        time.sleep(RATE_LIMIT_DELAY)

    print("\nDone. Raw files saved to data/raw/")


if __name__ == "__main__":
    main()
