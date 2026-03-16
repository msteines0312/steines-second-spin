"""
clean_data.py
Reads every raw JSON file from data/raw/, extracts a flat, normalized
dictionary for each album, and writes the result to data/albums_clean.json.

This is step 2 of the pipeline:
    fetch_spotify.py  ->  clean_data.py  ->  recommend.py  ->  build_data.py

Run from the project root:
    python pipeline/clean_data.py
"""

import json
from pathlib import Path


# ── Config ───────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent.parent   # project root
RAW_DIR   = ROOT / "data" / "raw"
OUT_FILE  = ROOT / "data" / "albums_clean.json"
# ─────────────────────────────────────────────────────────────────────────────


def extract_album(slug: str, raw: dict) -> dict:
    """Flatten one raw Spotify response into a clean, analysis-ready dictionary.

    Parameters
    ----------
    slug : str
        The filename stem (e.g. "mr-morale"). Used as our internal identifier
        so records stay linkable back to data/albums.json.
    raw : dict
        The full JSON object loaded from data/raw/{slug}.json, with keys
        "album_metadata", "tracks", and "audio_features".

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
    # Spotify attaches genres to artist objects, not albums. The album-level
    # genres field often comes back empty ([]) as a result. That's fine -- we
    # store it anyway for any albums where it does populate.
    genres = meta.get("genres", [])

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

    print(f"Cleaning {len(raw_files)} raw file(s)...\n")

    cleaned = []

    for path in raw_files:
        slug = path.stem   # filename without .json extension
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        album = extract_album(slug, raw)
        cleaned.append(album)

        duration_min = album["duration_ms"] / 60_000
        print(f"  {album['name']} -- {album['artist']}")
        print(f"    {album['track_count']} tracks  |  {duration_min:.1f} min  |  popularity {album['popularity']}")

    # Write the cleaned array, pretty-printed so it's readable as a reference file
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(cleaned)} albums written to {OUT_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
