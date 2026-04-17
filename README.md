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

```
fetch_spotify.py -> fetch_lastfm.py -> clean_data.py -> embed_reviews.py -> recommend.py -> build_data.py
```

Adding a new album means dropping an entry in `data/albums.json` and running the pipeline. Spotify and Last.fm fetches skip albums that already have cached data. `embed_reviews.py` extracts text from published review HTML, encodes it with SBERT, and saves 384-dimensional embeddings. The recommender blends all signals into one feature matrix before computing cosine similarity.

## What I Learned

Building the full stack from API calls through data cleaning to a live frontend showed me how much work lives in the middle of a data project. Adding SBERT embeddings taught me how to blend heterogeneous feature types (sparse TF-IDF vs. dense neural embeddings) in a single cosine similarity model, and how to design for graceful degradation when only some items have a given signal.
