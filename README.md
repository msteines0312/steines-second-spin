# Steines' Second Spin

A personal music review site. Every record I review gets a write-up, a star rating, and a favorite track pick. The site covers 30+ albums across hip-hop, rock, R&B, pop, and indie, with more in the queue.

## What It Is

This started as a way to document the records I was actually spending time with. It turned into a project that connects a Python data pipeline to a dynamic frontend, pulling album metadata from Spotify and Last.fm and building a recommendation engine on top of it.

The name is a nod to the idea that a record worth owning deserves more than one listen.

## Tech Stack

- HTML, CSS, JavaScript (static frontend, no framework)
- Python data pipeline (Spotify API, Last.fm API, pandas, scikit-learn)
- Cosine similarity for album recommendations

## Key Features

- **Reviews** with star ratings, favorite track callouts, and a vinyl disc animation
- **Genre filtering and sorting** across the full catalog
- **Cosine similarity recommendations** that surface related albums based on tags, genre, and listening patterns
- **Coming Soon queue** for albums I'm working through
- **Gear page** covering turntable, speaker, and accessory recommendations

## Pipeline

Adding a new album means dropping an entry in `data/albums.json`, running the fetch and build scripts to pull metadata and cover art, and writing the review. The pipeline handles Spotify/Last.fm enrichment, data validation, and producing the `albums_final.json` file the frontend reads.

## What I Learned

Building the full stack from API calls through data cleaning to a live frontend showed me how much work lives in the middle of a data project. The recommendation feature was a practical way to apply cosine similarity to a domain I actually care about.
