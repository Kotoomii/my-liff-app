// Service Worker for Smart Mirror App
const CACHE_NAME = 'stress-mirror-v1';
const urlsToCache = [
    '/',
    '/static/style.css',
    '/static/script.js',
    'https://cdn.jsdelivr.net/npm/chart.js'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(urlsToCache))
    );
});

self.addEventListener('fetch', (event) => {
    event.respondWith(
        caches.match(event.request)
            .then((response) => {
                // キャッシュがあれば返す、なければネットワークから取得
                return response || fetch(event.request);
            })
    );
});