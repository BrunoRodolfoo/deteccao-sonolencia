/* Service worker: cache básico da casca do app (offline após a primeira visita).
   Rede primeiro; se falhar, o cache. Também guarda as dependências: a cópia local em ./vendor/ (mesma origem)
   e, quando o app usa a CDN como reserva, as respostas da CDN e do modelo do MediaPipe. */
const VERSION = "sonolencia-pwa-v8"; // v8 = GitHub Pages (17/11/2026): EXTRA e HEAD offline; muda o nome para descartar o cache antigo
const SHELL = ["./", "./index.html", "./app.js", "./engine.js", "./tflite_io.js", "./styles.css", "./manifest.webmanifest", "./icon.svg", "./apple-touch-icon.png", "./icon-192.png", "./icon-512.png", "./fonts/barlow-semi-condensed-latin-600-normal.woff2", "./fonts/barlow-semi-condensed-latin-700-normal.woff2", "./qr_repositorio.png"];
// O app busca estes arquivos ao abrir a página, antes de o service worker assumir; sem guardá-los aqui, a
// primeira visita não os guardaria e o app só abriria sem internet a partir da segunda. São opcionais: sem
// vendor/ (servidor local sem a pasta), o app usa a CDN e a instalação segue. O WASM e o modelo do rosto vêm
// depois, no "Iniciar viagem", e entram no cache pelo fetch abaixo.
const EXTRA = [
  "./vendor/tasks-vision@1.0.1/vision_bundle.mjs",
  "./vendor/tfjs-core@4.22.0/dist/tf-core.min.js",
  "./vendor/tfjs-backend-cpu@4.22.0/dist/tf-backend-cpu.min.js",
  "./vendor/tfjs-tflite@0.0.1-alpha.10/dist/tf-tflite.min.js",
  "./models/eye_state.json",
  "./models/eye_state_int8.tflite",
  "./models/blink_temporal.json",
  "./models/blink_temporal.tflite",
];
const CDN = ["https://cdn.jsdelivr.net/npm/", "https://storage.googleapis.com/mediapipe-models/"];

self.addEventListener("install", (ev) => {
  ev.waitUntil(
    caches
      .open(VERSION)
      .then((c) => c.addAll(SHELL).then(() => Promise.all(EXTRA.map((u) => c.add(u).catch(() => {})))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (ev) => {
  ev.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k)))).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (ev) => {
  const req = ev.request;
  // O app pergunta por HEAD se um modelo existe (exists() no app.js); sem rede, responde pelo que está guardado.
  if (req.method === "HEAD") {
    ev.respondWith(fetch(req).catch(() => caches.match(req.url).then((hit) => new Response(null, { status: hit ? 200 : 503 }))));
    return;
  }
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
      // Abrir a página sem rede com parâmetros na URL (?pesquisa=1, ?autostart=1) usa a mesma página guardada.
      .catch(() => caches.match(req, { ignoreSearch: req.mode === "navigate" }).then((hit) => hit || new Response("offline", { status: 503 })))
  );
});
