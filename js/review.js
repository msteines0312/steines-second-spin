/**
 * review.js
 * Shared logic for all review pages.
 *
 * Currently exposes:
 *   renderMoreByArtist(album, allAlbums)
 *     Populates the #more-by-artist section with other catalog entries
 *     by the same artist. Hides the section automatically when the artist
 *     has no other albums in the catalog.
 */

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function renderMoreByArtist(album, allAlbums) {
  const section = document.getElementById('more-by-artist');
  if (!section) return;

  const others = allAlbums.filter(a => a.artist === album.artist && a.id !== album.id);

  if (others.length === 0) {
    section.style.display = 'none';
    return;
  }

  function stars(n) {
    return Array.from({ length: 5 }, (_, i) =>
      `<span class="star${i < n ? ' filled' : ''}">★</span>`
    ).join('');
  }

  const cards = others.map(a => {
    const hasReview = a.review_url && a.review_url !== '#';
    const href      = hasReview ? `${a.id}.html` : `../coming-soon.html#${a.id}`;
    return `
    <a class="acard${hasReview ? '' : ' acard--locked'}" href="${esc(href)}">
      <div class="acard-art">
        <img src="../${esc(a.cover_art || '')}" alt="${esc(a.title)}" loading="lazy" onerror="this.style.background='#888'">
      </div>
      <div class="acard-body">
        <h3 class="acard-title">${esc(a.title)}</h3>
        <p class="acard-artist">${esc(a.year)}</p>
        ${hasReview ? `<div class="star-row">${stars(a.stars || 0)}</div>` : ''}
      </div>
    </a>`;
  }).join('');

  section.className = 'recs-section';
  section.innerHTML = `
    <div class="recs-header">
      <div class="recs-eyebrow">More by ${esc(album.artist)}</div>
      <h2 class="recs-title">From the Same Artist</h2>
    </div>
    <div class="recs-grid">${cards}</div>
  `;
}
