/* =========================================================
   Service Worker — AgriSaathi PWA
   Caches the app shell so the app opens fast and offline.

   RELEASING A NEW VERSION (every time you change the frontend):
     1. Change CACHE_NAME below (e.g. v9 -> v10).
     2. Upload the new files.
   Phones then download the new version in the background and
   show "New version available — Refresh". Tapping it switches
   to the new version (see _registerSW in app.js).

   HOSTING: a service worker only works on https:// (or on
   localhost while developing). Phones also need https:// for
   the microphone and the "Install app" button.
   ========================================================= */

const CACHE_NAME = "agrisaathi-v10";
const APP_SHELL = [
  "./",
  "./index.html",
  "./style.css",
  "./app.js",
  "./config.js",
  "./manifest.json",
  "./favicon.svg",
  "./assets/logo.svg",
  "./icon-192.png",
  "./icon-512.png",
  "./icon-maskable-512.png",
  "./fonts/mukta-400-devanagari.woff2",
  "./fonts/mukta-400-latin.woff2",
  "./fonts/mukta-700-devanagari.woff2",
  "./fonts/mukta-700-latin.woff2",
];

// ── Install: download and cache the app shell ─────────────
// No skipWaiting() here: a new version waits until the farmer
// taps "Refresh", so the page never changes under their finger.
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log("[SW] Caching AgriSaathi app shell", CACHE_NAME);
      return cache.addAll(APP_SHELL);
    })
  );
});

// ── The page asks the waiting version to take over ────────
self.addEventListener("message", (event) => {
  if (event.data === "SKIP_WAITING") {
    self.skipWaiting();
  }
});

// ── Activate: delete caches from older versions ───────────
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((key) => key !== CACHE_NAME)
          .map((key) => {
            console.log("[SW] Deleting old cache:", key);
            return caches.delete(key);
          })
      )
    )
  );
  self.clients.claim();
});

// Fonts and images never change without a new CACHE_NAME,
// so they can come straight from the cache (fastest).
function _isStaticAsset(url) {
  return /\.(woff2|png|svg|jpg|jpeg|webp|ico)$/i.test(url.pathname);
}

// ── Fetch ─────────────────────────────────────────────────
self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);

  // Only GET requests for files of this site. API calls (POST,
  // /api/..., or the backend on another server) always go
  // straight to the network and are never cached: answers must
  // always be fresh, never an old saved reply.
  if (request.method !== "GET") return;
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/")) return;

  // Fonts and images: cache first, network if missing
  if (_isStaticAsset(url)) {
    event.respondWith(
      caches.match(request).then(
        (cached) => cached || fetch(request).then((response) => _saveCopy(request, response))
      )
    );
    return;
  }

  // Pages, CSS, JS, config: network first so changes show up,
  // cached copy when offline
  event.respondWith(
    fetch(request)
      .then((response) => _saveCopy(request, response))
      .catch(() =>
        caches.match(request).then((cached) => {
          if (cached) return cached;
          // Offline and never cached: for a page, show the app itself
          if (request.mode === "navigate") return caches.match("./index.html");
          return Response.error();
        })
      )
  );
});

// Keep a copy of a good response for offline use
function _saveCopy(request, response) {
  if (response && response.status === 200 && response.type === "basic") {
    const copy = response.clone();
    caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
  }
  return response;
}
