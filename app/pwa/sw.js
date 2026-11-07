/* Service worker: cache básico da casca do app (offline após a primeira visita).
   Rede primeiro; se falhar, o cache. Também guarda as dependências: a cópia local em ./vendor/ (mesma origem)
   e, quando o app usa a CDN como reserva, as respostas da CDN e do modelo do MediaPipe. */
const VERSION = "sonolencia-pwa-v7"; // v7 = interface "Painel de estrada", fontes e QR (07/11/2026): muda o nome para descartar o cache antigo
const SHELL = ["./", "./index.html", "./app.js", "./engine.js", "./tflite_io.js", "./styles.css", "./manifest.webmanifest", "./icon.svg", "./apple-touch-icon.png", "./icon-192.png", "./icon-512.png", "./fonts/barlow-semi-condensed-latin-600-normal.woff2", "./fonts/barlow-semi-condensed-latin-700-normal.woff2", "./qr_repositorio.png"];
const CDN = ["https://cdn.jsdelivr.net/npm/", "https://storage.googleapis.com/mediapipe-models/"];

self.addEventListener("install", (ev) => {
  ev.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (ev) => {
  ev.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k)))).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (ev) => {
  const req = ev.request;
  if (req.method !== "GET") return;
  // Rede primeiro (para pegar versões novas durante o desenvolvimento); cache como reserva.
  ev.respondWith(
    fetch(req)
      .then((res) => {
        const same = new URL(req.url).origin === self.location.origin;
        const cdn = CDN.some((p) => req.url.startsWith(p));
        if ((same && res.ok) || (cdn && (res.ok || res.type === "opaque"))) {
          const copy = res.clone();
          caches.open(VERSION).then((c) => c.put(req, copy)).catch(() => {});
        }
        return res;
      })
      .catch(() => caches.match(req).then((hit) => hit || new Response("offline", { status: 503 })))
  );
});
