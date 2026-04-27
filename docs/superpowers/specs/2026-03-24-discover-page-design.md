# Discover Page + Recommendation Algorithm Improvements

**Date:** 2026-03-24
**Status:** Approved

---

## Overview

Two related features:

1. **Algorithm improvements** to `recommend.py` - smarter weighting, richer output (scores + shared tags), more stored candidates per album.
2. **New "Dig Through The Crate" page** (`discover.html`) - a seed-based discovery UI that lets visitors pick an album they know and surface similar ones from the catalog.

---

## Algorithm Changes (recommend.py + build_data.py)

### Feature matrix weighting

Apply explicit multipliers to each component before calling `np.hstack`:

- Genre matrix: multiply by `3.0` (genre is the dominant signal)
- `year_scaled`: multiply by `0.5`
- `pop_scaled`: multiply by `0.5`

The math (cosine similarity) stays identical. The multipliers just shift the relative influence of each feature.

### Increase TOP_N to 10

Currently stores 3 recommendations per album. Increasing to 10 gives the Discover page enough candidates to show 5 results after potential genre filtering, while review pages continue to display only the top 3. With 32 albums the JSON stays small.

### Richer recommendation output

Each recommendation entry changes from a bare slug string to an object:

```json
{
  "slug": "ants-from-up-there",
  "score": 0.91,
  "shared_tags": ["art rock", "indie rock", "post-punk"]
}
```

`score` is the raw cosine similarity value rounded to 2 decimal places.
`shared_tags` is the intersection of genre tag lists between the source album and the recommended album.

### Pipeline propagation

`build_data.py` passes the recommendation objects through as-is into `albums_final.json`. The `recommendations` field on each album changes from `["slug1", "slug2"]` to `[{slug, score, shared_tags}, ...]`.

**build_data.py summary printer:** the existing line that does `" / ".join(album["recommendations"])` will throw a TypeError after this change since recommendations are now dicts. Update it to `" / ".join(r["slug"] for r in album["recommendations"])`. The `recs` loader (`recs = {a["slug"]: a["recommendations"] for a in json.load(f)}`) does not need changing - it passes the list through as-is.

**Review page JS update:** in both `reviews/getting-killed.html` and `reviews/template.html`, the recommendations render loop currently does:
```js
.map(recSlug => albums.find(a => a.id === recSlug))
```
Change to:
```js
.slice(0, 3).map(rec => albums.find(a => a.id === rec.slug))
```
The `.slice(0, 3)` is required because TOP_N increases to 10 - without it, review pages would render up to 10 rec cards instead of 3.

---

## Discover Page (discover.html)

### Route and nav

- New file at project root: `discover.html`
- Nav link text: "Discover", placed between "Reviews" and "Coming Soon"
- Must be added to **all 8 pages** that contain the nav:
  - Root pages (6): `index.html`, `reviews.html`, `about.html`, `favorites.html`, `gear-page-v2.html`, `coming-soon.html` - use `discover.html`
  - Review subpages (2): `reviews/getting-killed.html`, `reviews/template.html` - use `../discover.html`
- **Note:** `reviews/template.html` currently has a 5-item nav missing "Coming Soon". As part of this task, reconcile it to match the other pages - add both "Coming Soon" (`../coming-soon.html`) and "Discover" (`../discover.html`) in the correct order: Reviews, Discover, Coming Soon, Hardware Recommendations, About, Favorites, Spotify.

### Page structure (three zones)

**1. Page header strip**
- Dark background, matches existing `.page-strip` style
- Eyebrow: "Find Your Next Listen"
- Title: "Dig Through The Crate"

**2. Genre filter chips**
- Row of toggleable pill buttons built dynamically from all unique genre tags across `albums_final.json`
- Multiple chips can be active simultaneously (OR logic: show albums that have any selected genre)
- No chips selected = all albums visible in the picker
- "Clear" button resets all active chips and re-shows the full picker (results panel stays visible if a seed is selected)
- Chips are sorted alphabetically
- All albums (reviewed and unreviewed) are eligible as seeds and appear in the picker

**3. Album picker grid**
- Same card style as `reviews.html` but slightly smaller
- Filtered in real time as genre chips are toggled
- Clicking a card sets it as the seed album and triggers results
- Selected card gets a highlighted red border to indicate it is active
- After seed selection, page smooth-scrolls to the results panel

**4. Results panel**
- Hidden until a seed album is selected
- Iterates through all 10 stored recommendations in order, skipping slugs filtered out by active genre chips, and stops once 5 passing results are collected (backfill approach - always tries to show 5)
- If active genre chips filter out all 10 candidates, shows: "No matches in your filter - try clearing some genres"
- Each result card includes:
  - Cover art, title, artist
  - Score badge: "91% match" (use `Math.round(rec.score * 100)` to avoid floating-point display bugs; "100% match" is a valid value)
  - Shared tags line: "Both tagged: art rock, post-punk" (from `shared_tags`)
  - Stars and Second Spin badge ONLY if `review_url !== "#"` (i.e., a real review exists)
  - Clicking the card navigates to the review page; if `review_url === "#"` the card is non-interactive (no link, cursor: default)
- Seed persistence on page reload is explicitly out of scope

### Data source

Reads `data/albums_final.json` via `fetch`. No backend. All computation happens at pipeline time; the page is pure static HTML + JS.

---

## What does NOT change

- `fetch_spotify.py`, `fetch_lastfm.py`, `clean_data.py` - untouched
- CSS architecture - `discover.html` uses the existing `css/styles.css` with page-specific `<style>` block inline, matching every other page

---

## Open questions (resolved)

- **Natural language LLM input vs. seed-based?** Seed-based. LLM requires a backend, and 32 albums is too small a catalog for freeform text matching to feel good.
- **How many results on the Discover page?** 5, using backfill from 10 stored candidates.
- **Genre chip logic AND vs OR?** OR - more forgiving for a small catalog.
- **Show stars/second spin on result cards always?** No - only when a real review exists (`review_url !== "#"`).
- **Clearing genre chips when results are showing?** Chips clear, picker resets to full catalog, results re-render with no genre filter applied.
- **Seed persistence on reload?** Out of scope.
