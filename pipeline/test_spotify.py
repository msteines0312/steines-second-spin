import json
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

load_dotenv()

# Authenticate using Client Credentials flow (no user login required)
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

# --- 1. Search for the album ---
print("=" * 60)
print("ALBUM SEARCH: 'Ca$ino' by Baby Keem")
print("=" * 60)

results = sp.search(q="album:Ca$ino artist:Baby Keem", type="album", limit=1)
print(json.dumps(results, indent=2))

# --- 2. Pull audio features for the first track ---
print("\n" + "=" * 60)
print("AUDIO FEATURES: first track from Ca$ino")
print("=" * 60)

albums = results.get("albums", {}).get("items", [])
if not albums:
    print("No album found - check your search query.")
else:
    album_id = albums[0]["id"]
    tracks = sp.album_tracks(album_id)
    first_track = tracks["items"][0] if tracks["items"] else None

    if not first_track:
        print("No tracks found on this album.")
    else:
        print(f"Fetching audio features for: {first_track['name']} ({first_track['id']})")
        features = sp.audio_features([first_track["id"]])

        # NOTE: audio_features may return None or an empty list if your Spotify app
        # does not have access to this endpoint. As of 2024, Spotify restricts
        # audio features to apps that have been granted extended quota mode.
        # If you see None or [] below, this is an API access tier issue, not a code bug.
        print(json.dumps(features, indent=2))
