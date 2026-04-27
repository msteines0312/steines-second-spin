# Discover Page + Recommendation Algorithm Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add weighted genre-first recommendations with score/shared-tag output, and build a new "Dig Through The Crate" discovery page where visitors pick a seed album and get similar ones.

**Architecture:** The Python pipeline generates richer recommendation objects (slug + score + shared_tags) stored in `albums_final.json`. A new static `discover.html` page reads that file client-side, lets users filter by genre chips, pick a seed album, and surfaces the top 5 matches with scores and explanations.

**Tech Stack:** Python (numpy, scikit-learn), vanilla JS, static HTML/CSS, no backend.

**Spec:** `docs/superpowers/specs/2026-03-24-discover-page-design.md`

---

## File Map

| Action | File | What changes |
|--------|------|-------------|
| Modify | `pipeline/recommend.py` | Weight multipliers, TOP_N=10, richer output format, extract `build_rec_objects` function |
| Create | `pipeline/test_recommend.py` | Unit tests for recommendation logic |
| Modify | `pipeline/build_data.py` | Fix summary printer (line 108) |
| Modify | `reviews/getting-killed.html` | JS: `.slice(0,3).map(rec => rec.slug)`, add Discover to nav |
| Modify | `reviews/template.html` | Same JS fix, add Coming Soon + Discover to nav |
| Modify | `index.html` | Add Discover to nav |
| Modify | `reviews.html` | Add Discover to nav |
| Modify | `about.html` | Add Discover to nav |
| Modify | `favorites.html` | Add Discover to nav |
| Modify | `gear-page-v2.html` | Add Discover to nav |
| Modify | `coming-soon.html` | Add Discover to nav |
| Create | `discover.html` | Full discovery page |

---

## Task 1: Update recommend.py

Extract the per-album recommendation builder into a testable function, apply weight multipliers, increase TOP_N, and output richer objects.

**Files:**
- Modify: `pipeline/recommend.py`
- Create: `pipeline/test_recommend.py`

- [ ] **Step 1: Write failing tests**

Create `pipeline/test_recommend.py`:

```python
"""
test_recommend.py
Unit tests for the recommendation logic in recommend.py.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from recommend import build_rec_objects


def test_output_format():
    """Each recommendation should be a dict with slug, score, and shared_tags."""
    albums = [
        {"slug": "a", "genres": ["rock", "indie"]},
        {"slug": "b", "genres": ["rock", "pop"]},
        {"slug": "c", "genres": ["jazz"]},
    ]
    slugs = ["a", "b", "c"]
    sim_scores = np.array([1.0, 0.9, 0.1])

    recs = build_rec_objects(albums[0], albums, sim_scores, slugs, top_n=2)

    assert len(recs) == 2
    for r in recs:
        assert "slug" in r
        assert "score" in r
        assert "shared_tags" in r
        assert isinstance(r["slug"], str)
        assert isinstance(r["score"], float)
        assert isinstance(r["shared_tags"], list)


def test_shared_tags_is_intersection():
    """shared_tags should be the intersection of source and rec album genre lists."""
    albums = [
        {"slug": "x", "genres": ["post-punk", "art rock", "indie"]},
        {"slug": "y", "genres": ["post-punk", "art rock", "noise"]},
    ]
    slugs = ["x", "y"]
    sim_scores = np.array([1.0, 0.8])

    recs = build_rec_objects(albums[0], albums, sim_scores, slugs, top_n=1)

    assert set(recs[0]["shared_tags"]) == {"post-punk", "art rock"}


def test_excludes_self():
    """The source album should not appear in its own recommendations."""
    albums = [
        {"slug": "a", "genres": ["rock"]},
        {"slug": "b", "genres": ["rock"]},
    ]
    slugs = ["a", "b"]
    sim_scores = np.array([1.0, 0.9])

    recs = build_rec_objects(albums[0], albums, sim_scores, slugs, top_n=5)

    assert all(r["slug"] != "a" for r in recs)


def test_score_rounded_to_two_decimals():
    """Scores should be rounded to 2 decimal places."""
    albums = [
        {"slug": "a", "genres": ["rock"]},
        {"slug": "b", "genres": ["rock"]},
    ]
    slugs = ["a", "b"]
    sim_scores = np.array([1.0, 0.876543])

    recs = build_rec_objects(albums[0], albums, sim_scores, slugs, top_n=1)

    assert recs[0]["score"] == 0.88


def test_top_n_respected():
    """Returns at most top_n results."""
    albums = [{"slug": str(i), "genres": ["rock"]} for i in range(5)]
    slugs = [str(i) for i in range(5)]
    sim_scores = np.array([1.0, 0.9, 0.8, 0.7, 0.6])

    recs = build_rec_objects(albums[0], albums, sim_scores, slugs, top_n=2)

    assert len(recs) == 2
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd c:/Users/mstei/steines-second-spin
python -m pytest pipeline/test_recommend.py -v
```

Expected: `ImportError` or `AttributeError` because `build_rec_objects` does not exist yet.

- [ ] **Step 3: Update recommend.py**

Make three changes:

**a) Change TOP_N and add weight constants near the top of the config section:**

```python
TOP_N = 10   # number of recommendations to store per album (review pages show 3, discover page shows 5)

# Feature weights applied before cosine similarity.
# Genre is the primary signal; year and popularity are secondary.
GENRE_WEIGHT = 3.0
YEAR_WEIGHT  = 0.5
POP_WEIGHT   = 0.5
```

**b) Apply weights before `np.hstack` (replace the existing hstack line):**

```python
feature_matrix = np.hstack([
    genre_matrix * GENRE_WEIGHT,
    year_scaled  * YEAR_WEIGHT,
    pop_scaled   * POP_WEIGHT,
])
```

**c) Add `build_rec_objects` function before the main loop (Step 4):**

```python
def build_rec_objects(album, albums, sim_scores, slugs, top_n):
    """Build recommendation objects for one album.

    Parameters
    ----------
    album : dict
        The source album. Must have 'slug' and 'genres' keys.
    albums : list[dict]
        All albums in the catalog.
    sim_scores : np.ndarray
        1-D array of similarity scores for this album against every album,
        including itself (self-similarity is excluded inside this function).
    slugs : list[str]
        Slug list parallel to albums (same order).
    top_n : int
        Maximum number of recommendations to return.

    Returns
    -------
    list[dict]
        Each dict has keys: slug (str), score (float, 2 dp), shared_tags (list[str]).
    """
    i = slugs.index(album["slug"])
    ranked = sorted(
        [(j, sim_scores[j]) for j in range(len(albums)) if j != i],
        key=lambda x: x[1],
        reverse=True,
    )
    return [
        {
            "slug":        slugs[j],
            "score":       round(float(score), 2),
            "shared_tags": sorted(set(album["genres"]) & set(albums[j]["genres"])),
        }
        for j, score in ranked[:top_n]
    ]
```

**d) Replace the main loop's output building to use the new function:**

```python
for i, album in enumerate(albums):
    scores = sim_matrix[i]
    recs = build_rec_objects(album, albums, scores, slugs, TOP_N)

    output.append({
        "slug":            album["slug"],
        "recommendations": recs,
    })

    print(f"  {album['name']}")
    for r in recs:
        print(f"    {r['score']:.2f}  {albums[slugs.index(r['slug'])]['name']}  "
              f"[{', '.join(r['shared_tags']) or 'no shared tags'}]")
    print()
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
python -m pytest pipeline/test_recommend.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add pipeline/recommend.py pipeline/test_recommend.py
git commit -m "Smarter recommendation weights and richer output format"
```

---

## Task 2: Fix build_data.py and regenerate data

Fix the summary printer that breaks when recommendations are objects instead of strings, then re-run the full pipeline to update the JSON files.

**Files:**
- Modify: `pipeline/build_data.py` (line 108)

- [ ] **Step 1: Fix the summary printer in build_data.py**

Find line 108:
```python
rec_str = " / ".join(album["recommendations"]) if album["recommendations"] else "(none)"
```

Replace with:
```python
rec_str = " / ".join(r["slug"] for r in album["recommendations"]) if album["recommendations"] else "(none)"
```

- [ ] **Step 2: Run recommend.py to regenerate recommendations.json**

```bash
cd c:/Users/mstei/steines-second-spin
python pipeline/recommend.py
```

Expected output ends with: `Written to data/recommendations.json`

- [ ] **Step 3: Run build_data.py to regenerate albums_final.json**

```bash
python pipeline/build_data.py
```

Expected: summary table prints without TypeError, ends with `Written 32 albums to data/albums_final.json`

- [ ] **Step 4: Spot-check the output**

Open `data/albums_final.json` and verify the first album's `recommendations` field looks like:

```json
"recommendations": [
  {"slug": "projector", "score": 0.89, "shared_tags": ["post-punk", "indie rock"]},
  ...
]
```

Confirm there are 10 entries per album, not 3.

- [ ] **Step 5: Commit**

```bash
git add pipeline/build_data.py data/recommendations.json data/albums_final.json
git commit -m "Regenerate recommendation data with scores and shared tags"
```

---

## Task 3: Fix review page JS and template.html nav

Update both review pages to read `rec.slug` instead of treating the recommendation as a bare string, add `.slice(0, 3)` to keep showing exactly 3 cards, and fix `template.html`'s nav.

**Files:**
- Modify: `reviews/getting-killed.html`
- Modify: `reviews/template.html`

- [ ] **Step 1: Fix getting-killed.html recommendations JS**

Find (around line 666):
```js
const recAlbums = (album.recommendations || [])
  .map(recSlug => albums.find(a => a.id === recSlug))
  .filter(Boolean);
```

Replace with:
```js
const recAlbums = (album.recommendations || [])
  .slice(0, 3)
  .map(rec => albums.find(a => a.id === rec.slug))
  .filter(Boolean);
```

- [ ] **Step 2: Add Discover to getting-killed.html nav**

Find the nav `<ul class="nav-links">` block. Add after the Reviews `<li>`:
```html
<li><a href="../discover.html">Discover</a></li>
```

Result nav order: Reviews, Discover, Coming Soon, Hardware Recommendations, About, Favorites, Spotify.

- [ ] **Step 3: Fix template.html recommendations JS**

Same change as Step 1 - find and replace the `.map(recSlug => ...)` line with `.slice(0, 3).map(rec => albums.find(a => a.id === rec.slug))`.

- [ ] **Step 4: Fix template.html nav**

`template.html`'s nav is missing the hamburger button and "Coming Soon". Replace the entire `<nav class="site-nav">` block with:

```html
<!-- NAV — paths go up one level (../) since this file lives in reviews/ -->
<nav class="site-nav">
  <a class="wordmark" href="../index.html">Steines'<span> Second Spin</span></a>
  <button class="nav-toggle" aria-label="Open menu" aria-expanded="false">
    <span></span><span></span><span></span>
  </button>
  <ul class="nav-links">
    <li><a href="../reviews.html" class="here">Reviews</a></li>
    <li><a href="../discover.html">Discover</a></li>
    <li><a href="../coming-soon.html">Coming Soon</a></li>
    <li><a href="../gear-page-v2.html">Hardware Recommendations</a></li>
    <li><a href="../about.html">About</a></li>
    <li><a href="../favorites.html">Favorites</a></li>
    <li><a href="https://open.spotify.com/user/msteines12" target="_blank" rel="noopener">Spotify ↗</a></li>
  </ul>
</nav>
```

- [ ] **Step 5: Verify in browser**

Serve locally (`python -m http.server 8000` from project root), open `http://localhost:8000/reviews/getting-killed.html`, confirm:
- "You Might Also Like" shows exactly 3 cards (not 10)
- Cards load cover art correctly
- Nav includes "Discover" link

- [ ] **Step 6: Commit**

```bash
git add reviews/getting-killed.html reviews/template.html
git commit -m "Fix review page recs to read new data format, add Discover to nav"
```

---

## Task 4: Add Discover to nav on all 6 root pages

Each root page needs `<li><a href="discover.html">Discover</a></li>` added after the Reviews nav item.

**Files:** `index.html`, `reviews.html`, `about.html`, `favorites.html`, `gear-page-v2.html`, `coming-soon.html`

- [ ] **Step 1: Update all 6 root pages**

In each file, find the nav item for "Reviews" and add the Discover link immediately after it. The link uses `discover.html` (no leading `../` since these are root-level pages).

For `reviews.html`, the existing Reviews item has `class="here"`:
```html
<li><a href="reviews.html" class="here">Reviews</a></li>
<li><a href="discover.html">Discover</a></li>
```

For all others, Reviews does not have `class="here"` so it's just:
```html
<li><a href="reviews.html">Reviews</a></li>
<li><a href="discover.html">Discover</a></li>
```

When on `discover.html` itself, add `class="here"` to the Discover link.

- [ ] **Step 2: Quick visual check**

Open each of the 6 pages in the browser and confirm the nav shows Discover between Reviews and Coming Soon.

- [ ] **Step 3: Commit**

```bash
git add index.html reviews.html about.html favorites.html gear-page-v2.html coming-soon.html
git commit -m "Add Discover link to nav on all root pages"
```

---

## Task 5: Create discover.html

Build the full Discover page: page header, genre chips, album picker grid, and results panel. All data-driven from `albums_final.json`.

**Files:**
- Create: `discover.html`

- [ ] **Step 1: Create the HTML skeleton with styles**

Create `discover.html` at the project root. Base the structure on `reviews.html` (same nav, same footer, same CSS import). Add a page-specific `<style>` block with styles for chips, picker grid, and result cards.

Full file:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Discover — Steines' Second Spin</title>
<meta property="og:title" content="Discover — Steines' Second Spin">
<meta property="og:description" content="Pick an album you know. Find your next listen.">
<meta property="og:type" content="website">
<link rel="stylesheet" href="css/styles.css">
<style>

/* === PAGE HEADER === */
.page-strip {
  background: var(--ink);
  padding: clamp(36px, 7vh, 72px) clamp(1.5rem, 6vw, 5rem) clamp(24px, 4vh, 48px);
}
.page-eyebrow {
  font-family: var(--sans);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--red);
  margin-bottom: 10px;
  display: flex; align-items: center; gap: 10px;
}
.page-eyebrow::before { content: ''; display: block; width: 24px; height: 1px; background: var(--red); }
.page-title {
  font-family: var(--sans);
  font-size: clamp(28px, 5vw, 60px);
  font-weight: 900;
  letter-spacing: -0.03em;
  text-transform: uppercase;
  color: #fff;
  margin: 0;
  line-height: 1.05;
}

/* === MAIN CONTENT === */
.main-content {
  padding: clamp(32px, 5vh, 56px) clamp(1.5rem, 6vw, 5rem);
}

/* === GENRE CHIPS === */
.chips-section {
  margin-bottom: clamp(24px, 4vh, 40px);
}
.chips-label {
  font-family: var(--sans);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--mid);
  margin-bottom: 12px;
}
.chips-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.chip {
  font-family: var(--sans);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 4px 12px;
  border: 1px solid var(--faint);
  background: var(--bg);
  color: var(--mid);
  cursor: pointer;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
}
.chip:hover { border-color: var(--ink); color: var(--ink); }
.chip.active { background: var(--ink); color: #fff; border-color: var(--ink); }
.chip-clear {
  font-family: var(--sans);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  padding: 4px 12px;
  border: 1px solid rgba(232,24,30,0.3);
  background: none;
  color: var(--red);
  cursor: pointer;
  transition: background 0.15s;
}
.chip-clear:hover { background: rgba(232,24,30,0.06); }

/* === ALBUM PICKER === */
.section-label {
  font-family: var(--sans);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--mid);
  margin-bottom: 14px;
}
.picker-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: clamp(8px, 1.2vw, 14px);
  margin-bottom: clamp(40px, 6vh, 64px);
}
.picker-card {
  background: none;
  border: 2px solid transparent;
  padding: 0;
  cursor: pointer;
  text-align: left;
  transition: border-color 0.2s, transform 0.2s;
}
.picker-card:hover { transform: translateY(-2px); }
.picker-card.selected { border-color: var(--red); }
.picker-art {
  aspect-ratio: 1;
  background: var(--faint);
  overflow: hidden;
}
.picker-art img {
  width: 100%; height: 100%;
  object-fit: cover;
  display: block;
}
.picker-body {
  padding: 6px 2px 2px;
}
.picker-title {
  font-family: var(--sans);
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: -0.02em;
  color: var(--ink);
  margin: 0 0 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.picker-artist {
  font-family: var(--sans);
  font-size: 10px;
  font-weight: 300;
  color: var(--mid);
  margin: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* === RESULTS PANEL === */
.results-section {
  display: none;
  border-top: 1px solid var(--faint);
  padding-top: clamp(32px, 5vh, 48px);
}
.results-header {
  margin-bottom: clamp(20px, 3vh, 32px);
}
.results-seed {
  font-family: var(--sans);
  font-size: clamp(18px, 2.5vw, 28px);
  font-weight: 900;
  letter-spacing: -0.02em;
  text-transform: uppercase;
  color: var(--ink);
  margin: 0;
}
.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: clamp(14px, 2vw, 20px);
}
.result-card {
  display: block;
  text-decoration: none;
  color: inherit;
  border: 1px solid var(--faint);
  background: var(--bg);
  transition: transform 0.2s, box-shadow 0.2s;
}
.result-card:hover { transform: translateY(-3px); box-shadow: 0 6px 24px rgba(0,0,0,0.1); }
.result-card--locked { cursor: default; }
.result-card--locked:hover { transform: none; box-shadow: none; }
.result-art {
  aspect-ratio: 1;
  background: var(--faint);
  overflow: hidden;
}
.result-art img {
  width: 100%; height: 100%;
  object-fit: cover;
  display: block;
}
.result-body {
  padding: clamp(10px, 1.5vw, 16px);
  border-top: 1px solid var(--faint);
}
.result-score {
  font-family: var(--sans);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--red);
  margin-bottom: 6px;
}
.result-title {
  font-family: var(--sans);
  font-size: clamp(12px, 1.2vw, 15px);
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: -0.02em;
  color: var(--ink);
  margin: 0 0 3px;
}
.result-artist {
  font-family: var(--sans);
  font-style: italic;
  font-size: 12px;
  color: var(--mid);
  margin: 0 0 6px;
}
.result-tags {
  font-family: var(--sans);
  font-size: 10px;
  font-weight: 400;
  color: var(--mid);
  margin: 0 0 6px;
  line-height: 1.5;
}
.star-row { display: flex; gap: 2px; margin-bottom: 4px; }
.star { font-size: 12px; color: var(--faint); }
.star.filled { color: var(--ink); }
.spin-badge {
  font-family: var(--sans);
  font-size: 9px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  padding: 2px 7px;
  border: 1.5px solid var(--ink);
  color: var(--ink);
  display: inline-block;
}
.empty-state {
  font-family: var(--sans);
  font-size: 13px;
  font-weight: 300;
  color: var(--mid);
  padding: 24px 0;
}

/* === FOOTER === */
.site-footer {
  border-top: 1px solid var(--rule);
  background: var(--ink);
  padding: clamp(20px,4vh,32px) clamp(1.5rem,6vw,5rem);
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
}
.footer-wordmark { font-family: var(--sans); font-size: 13px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: rgba(255,255,255,0.3); }
.footer-wordmark span { color: var(--red); }

@media (max-width: 600px) {
  .picker-grid { grid-template-columns: repeat(3, 1fr); }
  .results-grid { grid-template-columns: repeat(2, 1fr); }
}

</style>
</head>
<body>

<nav class="site-nav">
  <a class="wordmark" href="index.html">Steines'<span> Second Spin</span></a>
  <button class="nav-toggle" aria-label="Open menu" aria-expanded="false">
    <span></span><span></span><span></span>
  </button>
  <ul class="nav-links">
    <li><a href="reviews.html">Reviews</a></li>
    <li><a href="discover.html" class="here">Discover</a></li>
    <li><a href="coming-soon.html">Coming Soon</a></li>
    <li><a href="gear-page-v2.html">Hardware Recommendations</a></li>
    <li><a href="about.html">About</a></li>
    <li><a href="favorites.html">Favorites</a></li>
    <li><a href="https://open.spotify.com/user/msteines12" target="_blank" rel="noopener">Spotify ↗</a></li>
  </ul>
</nav>

<header class="page-strip">
  <div class="page-eyebrow">Find Your Next Listen</div>
  <h1 class="page-title">Dig Through<br>The Crate</h1>
</header>

<main class="main-content">

  <!-- Genre filter chips -->
  <section class="chips-section">
    <p class="chips-label">Filter by genre</p>
    <div class="chips-wrap" id="chipContainer"></div>
  </section>

  <!-- Album picker grid -->
  <p class="section-label" id="pickerLabel">Pick an album to start</p>
  <div class="picker-grid" id="pickerGrid"></div>

  <!-- Results panel (hidden until seed selected) -->
  <section class="results-section" id="resultsSection">
    <div class="results-header">
      <p class="section-label">Similar albums</p>
      <h2 class="results-seed" id="resultsSeed"></h2>
    </div>
    <div class="results-grid" id="resultsGrid"></div>
  </section>

</main>

<footer class="site-footer">
  <span class="footer-wordmark">Steines'<span> Second Spin</span></span>
</footer>

<script>
// ── State ─────────────────────────────────────────────────────────────────────
let allAlbums = [];
let activeChips = new Set();
let currentSeed = null;

// ── Helpers ───────────────────────────────────────────────────────────────────

function starMarks(n) {
  return Array.from({ length: 5 }, (_, i) =>
    `<span class="star${i < n ? ' filled' : ''}">★</span>`
  ).join('');
}

function getFilteredAlbums() {
  if (activeChips.size === 0) return allAlbums;
  return allAlbums.filter(a => (a.genres || []).some(g => activeChips.has(g)));
}

// ── Genre chips ───────────────────────────────────────────────────────────────

function buildChips(albums) {
  const allGenres = new Set();
  albums.forEach(a => (a.genres || []).forEach(g => allGenres.add(g)));
  const sorted = [...allGenres].sort();

  const container = document.getElementById('chipContainer');
  container.innerHTML =
    sorted.map(g => `<button class="chip" data-genre="${g}">${g}</button>`).join('') +
    `<button class="chip-clear" id="clearChips">Clear</button>`;

  container.addEventListener('click', e => {
    if (e.target.classList.contains('chip')) {
      const genre = e.target.dataset.genre;
      if (activeChips.has(genre)) {
        activeChips.delete(genre);
        e.target.classList.remove('active');
      } else {
        activeChips.add(genre);
        e.target.classList.add('active');
      }
      renderPicker();
      if (currentSeed) renderResults(currentSeed);
    }
    if (e.target.id === 'clearChips') {
      activeChips.clear();
      container.querySelectorAll('.chip.active').forEach(c => c.classList.remove('active'));
      renderPicker();
      if (currentSeed) renderResults(currentSeed);
    }
  });
}

// ── Album picker ──────────────────────────────────────────────────────────────

function renderPicker() {
  const filtered = getFilteredAlbums();
  const grid = document.getElementById('pickerGrid');
  const label = document.getElementById('pickerLabel');

  label.textContent = activeChips.size > 0
    ? `${filtered.length} album${filtered.length !== 1 ? 's' : ''} match`
    : 'Pick an album to start';

  if (filtered.length === 0) {
    grid.innerHTML = '<p class="empty-state">No albums match this filter.</p>';
    return;
  }

  grid.innerHTML = filtered.map(a => {
    const isSelected = currentSeed && currentSeed.id === a.id;
    return `
      <button class="picker-card${isSelected ? ' selected' : ''}" data-slug="${a.id}" aria-label="${a.title} by ${a.artist}">
        <div class="picker-art">
          <img src="assets/covers/${a.id}.jpg" alt="${a.title}" loading="lazy" onerror="this.style.background='#888'">
        </div>
        <div class="picker-body">
          <p class="picker-title">${a.title}</p>
          <p class="picker-artist">${a.artist}</p>
        </div>
      </button>`;
  }).join('');

  // Attach click handler to the re-rendered grid
  grid.onclick = e => {
    const card = e.target.closest('.picker-card');
    if (!card) return;
    currentSeed = allAlbums.find(a => a.id === card.dataset.slug);
    renderPicker();
    renderResults(currentSeed);
    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
  };
}

// ── Results panel ─────────────────────────────────────────────────────────────

function renderResults(seed) {
  const section = document.getElementById('resultsSection');
  const grid    = document.getElementById('resultsGrid');
  const seedEl  = document.getElementById('resultsSeed');

  section.style.display = 'block';
  seedEl.textContent = `Because you picked: ${seed.title}`;

  const recs    = seed.recommendations || [];
  const results = [];

  for (const rec of recs) {
    if (results.length >= 5) break;
    const recAlbum = allAlbums.find(a => a.id === rec.slug);
    if (!recAlbum) continue;
    // If chips active, skip albums that don't match any selected genre
    if (activeChips.size > 0 && !(recAlbum.genres || []).some(g => activeChips.has(g))) continue;
    results.push({ rec, album: recAlbum });
  }

  if (results.length === 0) {
    grid.innerHTML = '<p class="empty-state">No matches in your filter - try clearing some genres.</p>';
    return;
  }

  grid.innerHTML = results.map(({ rec, album }) => {
    const hasReview = album.review_url && album.review_url !== '#';
    const pct       = Math.round(rec.score * 100);
    const sharedLine = rec.shared_tags && rec.shared_tags.length > 0
      ? `Both tagged: ${rec.shared_tags.join(', ')}`
      : '';

    return `
      <${hasReview ? 'a' : 'div'}
        class="result-card${hasReview ? '' : ' result-card--locked'}"
        ${hasReview ? `href="${album.review_url}"` : ''}
      >
        <div class="result-art">
          <img src="${album.cover_art}" alt="${album.title}" loading="lazy" onerror="this.style.background='#888'">
        </div>
        <div class="result-body">
          <div class="result-score">${pct}% match</div>
          <h3 class="result-title">${album.title}</h3>
          <p class="result-artist">${album.artist}</p>
          ${sharedLine ? `<p class="result-tags">${sharedLine}</p>` : ''}
          ${hasReview ? `<div class="star-row">${starMarks(album.stars || 0)}</div>` : ''}
          ${hasReview && album.second_spin ? `<span class="spin-badge">Second Spin</span>` : ''}
        </div>
      </${hasReview ? 'a' : 'div'}>`;
  }).join('');
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────

fetch('data/albums_final.json')
  .then(r => r.json())
  .then(albums => {
    allAlbums = albums;
    buildChips(albums);
    renderPicker();
  })
  .catch(() => {
    document.getElementById('pickerGrid').innerHTML =
      '<p class="empty-state">Could not load album data. Run a local server: python -m http.server 8000</p>';
  });
</script>
<script src="js/main.js"></script>
</body>
</html>
```

- [ ] **Step 2: Verify the page in browser**

Serve locally, open `http://localhost:8000/discover.html`. Check:
- Genre chips render and are clickable
- Selecting chips filters the album grid in real time
- Clicking an album shows results below
- Results show score badge and shared tags
- Results without a review don't show stars and are non-clickable
- Results with a review link to the review page
- "Clear" resets the chip filters

- [ ] **Step 3: Commit**

```bash
git add discover.html
git commit -m "Add Discover page with seed-based recommendations"
```

---

## Verification Checklist

After all tasks are complete, confirm the following end-to-end:

- [ ] `python -m pytest pipeline/test_recommend.py -v` - all 5 tests pass
- [ ] `data/albums_final.json` recommendations are objects with slug/score/shared_tags
- [ ] Each album has 10 recommendations stored
- [ ] `reviews/getting-killed.html` shows exactly 3 rec cards with correct covers
- [ ] All 8 nav-bearing pages have "Discover" between Reviews and Coming Soon
- [ ] `discover.html` genre chips filter the picker in real time
- [ ] Picking a seed album shows up to 5 results with score badges
- [ ] Results show "Both tagged: ..." line when shared tags exist
- [ ] Stars + Second Spin only appear on result cards with real reviews
- [ ] Non-reviewed result cards are non-clickable (cursor: default)
- [ ] Empty state message appears when chips filter out all candidates
