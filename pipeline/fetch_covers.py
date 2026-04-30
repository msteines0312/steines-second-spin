"""
fetch_covers.py
Downloads album covers, resizes them, and saves as webp.
"""

import argparse
import io
import json
import time
from pathlib import Path
import requests
from PIL import Image

# Config
ROOT       = Path(__file__).resolve().parent.parent
ALBUMS_IN  = ROOT / "data" / "albums.json"
CLEAN_IN   = ROOT / "data" / "albums_clean.json"
COVERS_OUT = ROOT / "assets" / "covers"
DELAY   = 0.5
TIMEOUT = 15
SIZE    = (600, 600)
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; steines-second-spin/1.0)"}

def process_and_save(content, out_path):
    """Resize to 600x600 and save as webp."""
    img = Image.open(io.BytesIO(content))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(SIZE, Image.Resampling.LANCZOS)
    img.save(out_path, "WEBP", quality=82, method=6)
    return out_path.stat().st_size / 1024

def try_download(url, out_path):
    """Download, process, and save image."""
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    return process_and_save(response.content, out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    COVERS_OUT.mkdir(parents=True, exist_ok=True)

    with open(ALBUMS_IN, "r", encoding="utf-8") as f:
        albums = {a["id"]: a for a in json.load(f)}

    with open(CLEAN_IN, "r", encoding="utf-8") as f:
        clean = {a["slug"]: a for a in json.load(f)}

    slugs = list(albums.keys())
    print(f"Downloading covers for {len(slugs)} album(s)...\n")

    for slug in slugs:
        ed  = albums[slug]
        cl  = clean.get(slug, {})
        out_path = COVERS_OUT / f"{slug}.webp"

        candidates = []
        if cl.get("cover_art"):
            candidates.append(("Spotify", cl["cover_art"]))
        if ed.get("art"):
            candidates.append(("Wikipedia", ed["art"]))

        if not candidates:
            continue

        if out_path.exists() and out_path.stat().st_size > 0 and not args.force:
            continue

        saved = False
        for source, url in candidates:
            try:
                size_kb = try_download(url, out_path)
                print(f"  {slug} ({source}): {size_kb:.1f} KB")
                saved = True
                break
            except Exception as e:
                print(f"  [WARN] {source} failed for {slug}: {e}")

        time.sleep(DELAY)
    print("\nDone.")

if __name__ == "__main__":
    main()
