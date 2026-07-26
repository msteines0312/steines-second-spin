"""
test_fetch_spotify.py
Unit tests for extract_spotify_id() in fetch_spotify.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_spotify import extract_spotify_id


def test_plain_album_url():
    url = "https://open.spotify.com/album/0eeXb23yMW6EaIgm63xxPC"
    assert extract_spotify_id(url) == "0eeXb23yMW6EaIgm63xxPC"


def test_trailing_slash():
    url = "https://open.spotify.com/album/0eeXb23yMW6EaIgm63xxPC/"
    assert extract_spotify_id(url) == "0eeXb23yMW6EaIgm63xxPC"


def test_share_link_query_string():
    url = "https://open.spotify.com/album/0eeXb23yMW6EaIgm63xxPC?si=abc123"
    assert extract_spotify_id(url) == "0eeXb23yMW6EaIgm63xxPC"


def test_trailing_slash_and_query_string():
    url = "https://open.spotify.com/album/0eeXb23yMW6EaIgm63xxPC/?si=abc123"
    assert extract_spotify_id(url) == "0eeXb23yMW6EaIgm63xxPC"
