"""
embed_reviews.py
Extracts review text and creates SBERT embeddings (all-MiniLM-L6-v2).
Saved to data/embeddings/{slug}.npy.
"""

import json
import sys
from pathlib import Path

import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

ROOT           = Path(__file__).resolve().parent.parent
ALBUMS_FILE    = ROOT / "data" / "albums.json"
EMBEDDINGS_DIR = ROOT / "data" / "embeddings"
MODEL_NAME     = "all-MiniLM-L6-v2"

def extract_review_text(html_path: Path) -> str:
    """Pull paragraphs from .review-body, skipping the 'Favorites' tracklist."""
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
    body = soup.find("article", class_="review-body")
    if not body:
        return ""

    paragraphs = body.find_all("p")
    # Drop last paragraph (usually tracklist)
    text_paragraphs = paragraphs[:-1] if len(paragraphs) > 1 else paragraphs
    return " ".join(p.get_text(separator=" ", strip=True) for p in text_paragraphs)


def main():
    force = "--force" in sys.argv

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    with open(ALBUMS_FILE, "r", encoding="utf-8") as f:
        albums = json.load(f)

    # Only albums with a review
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
            print(f"  [skip] {album['title']} - HTML not found at {html_path}")
            continue

        text = extract_review_text(html_path)
        if not text:
            print(f"  [skip] {album['title']} - no .review-body found")
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
