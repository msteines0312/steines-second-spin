"""
embed_reviews.py
Extracts review text from each published review page and produces
a sentence embedding using SBERT (all-MiniLM-L6-v2).

Embeddings are saved to data/embeddings/{slug}.npy. recommend.py
reads these to add a review-similarity signal alongside the TF-IDF
genre features. Albums without a published review are simply skipped
and receive no embedding contribution in the recommender.

By default, albums that already have a .npy file in data/embeddings/
are skipped. Pass --force to re-embed everything.

This is step 3.5 of the pipeline (run after clean_data.py):
    fetch_spotify.py -> fetch_lastfm.py -> clean_data.py
                     -> embed_reviews.py -> recommend.py -> build_data.py

Run from the project root:
    python pipeline/embed_reviews.py           # skip already-embedded
    python pipeline/embed_reviews.py --force   # re-embed everything
"""

import json
import sys
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


# ── Config ────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
ALBUMS_FILE    = ROOT / "data" / "albums.json"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
MODEL_NAME     = "all-MiniLM-L6-v2"
# ─────────────────────────────────────────────────────────────────────────────


def extract_review_text(html_path: Path) -> str:
    """Pull all paragraph text from the .review-body article element.

    Strips the Favorites line (last <p>, which is italicized) since it's
    just a track list and adds noise rather than semantic signal.
    """
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    body = soup.find("article", class_="review-body")
    if not body:
        return ""

    paragraphs = body.find_all("p")
    # Drop the last paragraph — it's always "Favorites: Track, Track, Track"
    text_paragraphs = paragraphs[:-1] if len(paragraphs) > 1 else paragraphs
    return " ".join(p.get_text(separator=" ", strip=True) for p in text_paragraphs)


def main():
    force = "--force" in sys.argv

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    with open(ALBUMS_FILE, "r", encoding="utf-8") as f:
        albums = json.load(f)

    # Only process albums that have a real review page (not "#")
    reviewed = [a for a in albums if a.get("review_url", "#") != "#"]

    if not force:
        reviewed = [a for a in reviewed if not (EMBEDDINGS_DIR / f"{a['id']}.npy").exists()]

    if not reviewed:
        print("All reviewed albums already embedded. Pass --force to re-embed.")
        return

    print(f"Loading {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Embedding {len(reviewed)} review(s)...\n")

    for album in reviewed:
        slug       = album["id"]
        html_path  = ROOT / album["review_url"]

        if not html_path.exists():
            print(f"  [skip] {album['title']} — HTML not found at {html_path}")
            continue

        text = extract_review_text(html_path)
        if not text:
            print(f"  [skip] {album['title']} — no .review-body found")
            continue

        embedding = model.encode(text, convert_to_numpy=True)
        np.save(EMBEDDINGS_DIR / f"{slug}.npy", embedding)

        preview = text[:80].replace("\n", " ")
        print(f"  {album['title']} ({slug})")
        print(f"    \"{preview}...\"")
        print(f"    embedding shape: {embedding.shape}\n")

    print(f"Done. Embeddings saved to {EMBEDDINGS_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
