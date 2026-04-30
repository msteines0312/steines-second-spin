# Steines' Second Spin

A personal music review site. Every record I review gets a write-up, a star rating, and a favorite track pick. The site covers 116 albums across hip-hop, rock, R&B, pop, and indie, with more in the queue.

## What It Is

This started as a way to document the records I was actually spending time with. It turned into a project that connects a Python data pipeline to a dynamic frontend, pulling album metadata from Spotify and Last.fm and building a recommendation engine on top of it.

The name is a nod to the idea that a record worth owning deserves more than one listen.

## Tech Stack

- HTML, CSS, JavaScript (static frontend, no framework)
- Python data pipeline (Spotify API, Last.fm API, pandas, scikit-learn)
- TF-IDF weighted cosine similarity for album recommendations
- SBERT (sentence-transformers, all-MiniLM-L6-v2) for review-text embeddings

## Key Features

- **Reviews** with star ratings, favorite track callouts, and a vinyl disc animation
- **Genre filtering and sorting** across the full catalog
- **Hybrid recommendation engine** combining TF-IDF genre tags, release year, Last.fm listener counts, and SBERT embeddings of written review text
- **Discover page** with seed-based album discovery and collapsible genre cluster filtering
- **Coming Soon queue** for albums I'm working through
- **Gear page** covering turntable, speaker, and accessory recommendations

## Pipeline

To update the entire site (fetch data, recalculate recommendations, and build HTML), run:

```bash
python pipeline/run_all.py
```

This master script orchestrates the following flow:
1. `fetch_spotify.py` — Fetches metadata and audio features.
2. `fetch_lastfm.py` — Enriches genres and listener counts.
3. `clean_data.py` — Normalizes and merges raw JSON.
4. `embed_reviews.py` — Generates SBERT embeddings for review text.
5. `recommend.py` — Calculates cosine similarity matches.
6. `build_data.py` — Merges all signals into the final frontend JSON.
7. `fetch_covers.py` — Downloads missing album art.
8. `build_html.py` — Generates the static site from templates and content.

Adding a new album means dropping an entry in `data/albums.json` and running the pipeline. Spotify and Last.fm fetches skip albums that already have cached data.

## What I Learned

Building the full stack from API calls through data cleaning to a live frontend showed me how much work lives in the middle of a data project. Adding SBERT embeddings taught me how to blend heterogeneous feature types (sparse TF-IDF vs. dense neural embeddings) in a single cosine similarity model, and how to design for graceful degradation when only some items have a given signal.
