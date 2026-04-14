# CLAUDE.md — Steines' Second Spin

> Auto-generated project memory. Last updated: 2026-04-13
> Do not delete — used by Claude Code for session continuity.

---

## What This Project Is

A personal music review site (30+ albums, growing) with a Python data pipeline that pulls metadata from Spotify and Last.fm, builds a cosine similarity recommendation engine, and produces a static JSON file the frontend reads. Reviews include star ratings, favorite track callouts, a vinyl disc animation, and algorithm-driven "You Might Also Like" sections.

---

## Stack

- **Language(s):** Python (pipeline), HTML/CSS/JS (frontend)
- **Key libraries:** spotipy, scikit-learn, numpy, pandas, python-dotenv
- **Data sources:** Spotify API (metadata, track URLs), Last.fm API (genre tags), manual curation (`data/albums.json`)
- **Output:** Static site served from project root; `data/albums_final.json` is the single file the frontend fetches

---

## Project Structure

```
steines-second-spin/
├── data/
│   ├── albums.json          # manually curated: stars, genres, review_url (no blurb field)
│   ├── albums_clean.json    # merged genres (manual + Last.fm) + Spotify metadata
│   ├── albums_final.json    # final merged file the frontend reads
│   ├── recommendations.json # pipeline output (slug/score/shared_tags per rec)
│   ├── raw/                 # per-album Spotify raw responses
│   └── lastfm/              # per-album Last.fm raw responses
├── pipeline/
│   ├── fetch_spotify.py     # step 1: Spotify metadata
│   ├── fetch_lastfm.py      # step 2: Last.fm tags
│   ├── clean_data.py        # step 3: merge and clean
│   ├── recommend.py         # step 4: cosine similarity recommendations
│   ├── build_data.py        # step 5: final merge into albums_final.json
│   ├── fetch_covers.py      # download cover art
│   └── test_recommend.py    # pytest unit tests for recommendation logic
├── reviews/
│   ├── getting-killed.html      # live review (Geese, 2025)
│   ├── hurry-up-tomorrow.html   # live review (The Weeknd, 5 stars, Second Spin)
│   ├── deadbeat.html            # live review (Tame Impala, 3 stars, Second Spin)
│   ├── bully.html               # live review (Kanye West, 2026)
│   ├── imaginal-disk.html       # live review (Magdalena Bay, 2024)
│   └── template.html            # template for new reviews
├── assets/covers/           # downloaded cover art JPGs
├── css/styles.css           # shared base CSS
├── js/main.js               # shared JS (nav toggle only)
├── index.html
├── reviews.html
├── discover.html            # Discover page (added 2026-03-24)
├── about.html
├── favorites.html
├── coming-soon.html
└── gear-page-v2.html
```

---

## Current Status

- **Phase:** Active development
- **Catalog size:** 32 albums
- **Reviews published:** 5 (getting-killed, hurry-up-tomorrow, deadbeat, bully, imaginal-disk)
- **What works:** Full pipeline, recommendation engine, all review/grid pages, Discover page (mobile-responsive)
- **What's in progress:** Adding new albums as they're reviewed
- **What's next:** More review pages

---

## Key Decisions and Conventions

- **No shared CSS/JS per page** — Every HTML file has its own `<style>` block (300-500 lines) and inline `<script>`. Any nav/footer change must be made in all 10 files manually (7 root pages + 3 review subpages).
- **Nav order (all pages):** Reviews, Discover, Coming Soon, Hardware Recommendations, About, Favorites, Spotify. Root pages use `discover.html`; review subpages use `../discover.html`.
- **`data/albums.json` is source of truth** — Add a new album here first; pipeline pulls everything else.
- **Pipeline run order:** `fetch_spotify.py` → `fetch_lastfm.py` → `clean_data.py` → `recommend.py` → `build_data.py`
- **Recommendation data format (as of 2026-03-24):** Each rec is an object `{slug, score, shared_tags}`, not a bare string. Read `rec.slug`, not `rec` directly. 10 recs stored per album; review pages show 3 (`.slice(0,3)`), Discover page shows up to 5 (backfill).
- **Recommendation weights:** genre x3.0, year x0.5, popularity x0.5 before cosine similarity.
- **Discover page genre filter (as of 2026-03-24 polish pass):** Filter chips are collapsed by default behind a toggle button. Chips are 8 keyword-based clusters (Rock, Hip-Hop, Electronic, Folk, R&B/Soul, Pop, Metal, Experimental), not individual genre tags. `activeChips` stores cluster labels; `albumMatchesCluster()` does substring matching against `CLUSTERS` keywords. Adding new albums requires no maintenance to the filter. The 130+ raw Last.fm genre tags are intentionally not exposed in the UI.
- **Discover page seed picker:** Clicking an already-selected album deselects it and hides the results section. Selecting a new seed always shows results.
- **No em dashes anywhere** — Not in copy, comments, commit messages, meta tags, or docstrings. Use a period, comma, or restructured sentence instead.
- **No "Co-Authored-By: Claude"** in commit messages. Plain human voice, no conventional commit prefixes, no emoji.
- **Stars and Second Spin badge** only shown on cards/results where `review_url !== "#"` (i.e., a real review page exists).
- **`albums.json` has no `blurb` field** — removed entirely. Do not add it back.
- **about.html quick stats are fully dynamic** — both "Albums Cataloged" and "Reviews Published" are driven by `albums_final.json` at page load. No manual edits needed after running `build_data.py`.
- **`button` elements in CSS Grid need `width: 100%`** — browsers don't auto-stretch form controls like divs. Always add `width: 100%; min-width: 0` to button grid items.
- **After adding albums:** run `build_data.py`. Stats update automatically.

---

## Known Issues / Gotchas

- `review_url` is `"#"` for albums in the catalog that don't have a written review yet. All conditional UI (stars, badge, links) gates on `review_url !== "#"`.
- Spotify Client Credentials auth returns `null` for `popularity` — treated as 50 (neutral midpoint) in the pipeline.
- `template.html` was historically out of sync with `getting-killed.html` (missing Coming Soon nav item, missing hamburger button). Fixed 2026-03-24; keep them in sync going forward.
- All `<title>` and `og:title` tags now use hyphens. `discover.html` was the original reference; all others were updated 2026-04-13.
- The site requires a local HTTP server to work (`python -m http.server 8000`); opening HTML files directly from the filesystem breaks `fetch()` calls.

---

## Portfolio Notes

- End-to-end data pipeline: API ingestion → cleaning → ML (cosine similarity) → static JSON → dynamic frontend
- The recommendation engine uses genre tags, release year, and Spotify popularity as features with explicit weighting to make genre dominant
- Discover page is a seed-based discovery UI with collapsible genre cluster filtering, all static (no backend). Result cards display a thin red score bar (scaled to match %) and shared genre tags as chips.
- 6 pytest unit tests for the recommendation logic in `pipeline/test_recommend.py`
