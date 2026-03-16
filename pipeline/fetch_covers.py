"""
fetch_covers.py
Downloads album cover art and saves each image locally to
site/assets/covers/{slug}.jpg.

Why we download instead of linking externally:
  - Wikipedia URLs rot (files move, pages get renamed).
  - Spotify's CDN (i.scdn.co) blocks browser cross-origin requests via CORS.
  - Downloading server-side with requests bypasses both problems: Python is
    not subject to browser CORS policy, and we capture a stable local copy.

Source priority:
  1. Spotify CDN URL from albums_clean.json  (640x640, reliable)
  2. Wikipedia art URL from albums.json      (fallback if clean data is absent)

Run from the project root:
    python pipeline/fetch_covers.py
"""

import json
import time
from pathlib import Path

import requests

# ── Config ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
ALBUMS_IN  = ROOT / "data" / "albums.json"
CLEAN_IN   = ROOT / "data" / "albums_clean.json"
COVERS_OUT = ROOT / "site" / "assets" / "covers"

DELAY   = 0.5   # seconds between requests
TIMEOUT = 15    # max seconds per download

HEADERS = {
    # Identify ourselves; some CDNs reject bare Python user-agent strings
    "User-Agent": "Mozilla/5.0 (compatible; steines-second-spin/1.0)"
}
# ─────────────────────────────────────────────────────────────────────────────

COVERS_OUT.mkdir(parents=True, exist_ok=True)

with open(ALBUMS_IN, "r", encoding="utf-8") as f:
    albums = {a["id"]: a for a in json.load(f)}

# albums_clean.json has the Spotify CDN URLs (640x640) — prefer these
with open(CLEAN_IN, "r", encoding="utf-8") as f:
    clean = {a["slug"]: a for a in json.load(f)}

slugs = list(albums.keys())
print(f"Downloading covers for {len(slugs)} album(s)...\n")

def try_download(url, out_path):
    """Attempt to download url to out_path. Returns True on success."""
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    return len(response.content) / 1024  # KB saved


for slug in slugs:
    ed  = albums[slug]
    cl  = clean.get(slug, {})
    out_path = COVERS_OUT / f"{slug}.jpg"

    # Try sources in priority order: Spotify CDN (reliable, server-side only),
    # then Wikipedia (fallback — URLs sometimes rot or rate-limit).
    candidates = []
    if cl.get("cover_art"):
        candidates.append(("Spotify CDN", cl["cover_art"]))
    if ed.get("art"):
        candidates.append(("Wikipedia", ed["art"]))

    if not candidates:
        print(f"  [SKIP] {slug} — no art URL found in either source")
        continue

    saved = False
    for source, url in candidates:
        print(f"  {slug}  ({source})")
        print(f"    {url}")
        try:
            size_kb = try_download(url, out_path)
            print(f"    Saved {size_kb:.1f} KB -> {out_path.relative_to(ROOT)}")
            saved = True
            break  # success — skip remaining candidates
        except requests.RequestException as e:
            print(f"    [WARNING] {source} failed: {e} — trying next source...")

    if not saved:
        print(f"    [ERROR] All sources failed for {slug}")

    time.sleep(DELAY)

print("\nDone.")
