// main.js
// Shared client-side logic for Steines' Second Spin.

// Mobile nav toggle -- runs on every page
document.querySelector('.nav-toggle').addEventListener('click', function() {
  var nav = document.querySelector('.site-nav');
  var open = nav.classList.toggle('nav-open');
  this.setAttribute('aria-expanded', open);
});
