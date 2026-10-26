/**
 * Roda a PWA no Chrome headless com câmera falsa e registra console, exceções, estado do app,
 * checagem de layout e capturas de tela (DevTools Protocol, sem dependências; Node >= 22).
 *
 *   node tools/app_validacao/teste_headless_pwa.mjs --url "https://localhost:8766/?autostart=1" \
 *        [--video clipe.mjpeg] [--seconds 60] [--width 390] [--height 844] [--out pasta] \
 *        [--shots 20,45] [--click-sound 15] [--click-at 70:btn-ack] [--open-tech]
 *
 * SILÊNCIO OBRIGATÓRIO: o Chrome headless novo toca som de verdade nos alto-falantes do computador.
 * Por isso este script SEMPRE (1) lança o Chrome com --mute-audio e (2) abre o app com ?volume=0
 * (qualquer outro valor de volume na URL é trocado por 0), e (3) aborta o teste se o app informar
 * volume diferente de 0 ou ganho de áudio diferente de 0.
 * Política de autoplay PADRÃO (desde o ajuste final): sem --autoplay-policy=no-user-gesture-required, como
 * no iPhone o áudio só é liberado por um gesto real. Os cliques de --click-sound e --click-at são cliques
 * CONFIÁVEIS (Input.dispatchMouseEvent do DevTools, isTrusted = true), não element.click(). --autoplay-livre
 * devolve a flag antiga (esconde a exigência de gesto; só para comparação).
 *
 * --video: arquivo .y4m ou .mjpeg para a câmera falsa (sem ele o Chrome gera um padrão sintético,
 * sem rosto). Os vídeos das bases ficam só na máquina local (nunca versionar nem publicar capturas).
 * Saída: <out>/relatorio.json (console, exceções, estado final, layout) e <out>/tela_<s>s.png.
 */
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, arr) => {
    if (a.startsWith("--")) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith("--") ? arr[i + 1] : "1"]);
    return acc;
  }, []),
);
const URL_ = (() => {
  const u = new URL(args.url || "https://localhost:8766/?autostart=1");
  if (u.searchParams.has("volume") && u.searchParams.get("volume") !== "0") console.warn(`volume=${u.searchParams.get("volume")} na URL trocado por 0 (teste sempre mudo)`);
  u.searchParams.set("volume", "0");
  return u.toString();
})();
const SECONDS = Number(args.seconds || 60);
const W = Number(args.width || 390);
const H = Number(args.height || 844);
const OUT = path.resolve(args.out || "headless_out");
const SHOTS = (args.shots || "").split(",").filter(Boolean).map(Number);
const PORT = 9333 + Math.floor(Math.random() * 500);
const CHROME = args.chrome || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
fs.mkdirSync(OUT, { recursive: true });
const profile = fs.mkdtempSync(path.join(OUT, "perfil-"));

const flags = [
  "--headless=new",
  "--mute-audio", // obrigatório: nunca tocar som no computador
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`,
  "--use-fake-ui-for-media-stream",
  "--use-fake-device-for-media-stream",
  "--ignore-certificate-errors",
  ...(args["autoplay-livre"] ? ["--autoplay-policy=no-user-gesture-required"] : []),
  "--no-first-run",
  "--no-default-browser-check",
  `--window-size=${W},${H}`,
  "--enable-logging=stderr",
  "--v=0",
];
if (args.video) flags.push(`--use-file-for-fake-video-capture=${path.resolve(args.video)}`);
const chrome = spawn(CHROME, [...flags, "about:blank"], { stdio: ["ignore", "ignore", "pipe"] });
let chromeErr = "";
chrome.stderr.on("data", (d) => { chromeErr += d.toString(); });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function getJSON(u) {
  for (let i = 0; i < 100; i++) {
    try {
      const r = await fetch(u);
      if (r.ok) return await r.json();
    } catch {
      /* ainda subindo */
    }
    await sleep(100);
  }
  throw new Error(`sem resposta de ${u}`);
}
const targets = await getJSON(`http://127.0.0.1:${PORT}/json/list`);
const page = targets.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let id = 0;
const pending = new Map();
const logs = [];
const t0 = Date.now();
const stamp = () => ((Date.now() - t0) / 1000).toFixed(1);
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
    return;
  }
  if (m.method === "Runtime.consoleAPICalled") {
    const txt = m.params.args.map((a) => (a.value !== undefined ? String(a.value) : a.description || a.type)).join(" ");
    logs.push({ t: stamp(), kind: `console.${m.params.type}`, text: txt });
  } else if (m.method === "Runtime.exceptionThrown") {
    const d = m.params.exceptionDetails;
    logs.push({ t: stamp(), kind: "EXCECAO", text: `${d.text} ${d.exception?.description || ""} @${d.url || ""}:${d.lineNumber}` });
  } else if (m.method === "Log.entryAdded") {
    logs.push({ t: stamp(), kind: `log.${m.params.entry.level}`, text: `${m.params.entry.text} ${m.params.entry.url || ""}` });
  }
});
function send(method, params = {}) {
  return new Promise((resolve) => {
    const k = ++id;
    pending.set(k, resolve);
    ws.send(JSON.stringify({ id: k, method, params }));
  });
}
async function evaluate(expr) {
  const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true });
  if (r.result?.exceptionDetails) return { erro: r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text };
  return r.result?.result?.value;
}
/** Clique confiável (gesto do usuário) no centro do elemento: rola até ele e dispara mousePressed/Released. */
async function trustedClick(elId) {
  const box = await evaluate(`(() => { const e = document.getElementById(${JSON.stringify(elId)}); if (!e) return null; e.scrollIntoView({ block: "center" }); const r = e.getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; })()`);
  if (!box || box.erro) return false;
  for (const type of ["mousePressed", "mouseReleased"]) await send("Input.dispatchMouseEvent", { type, x: box.x, y: box.y, button: "left", clickCount: 1 });
  return true;
}
async function shot(name) {
  const r = await send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false });
  fs.writeFileSync(path.join(OUT, name), Buffer.from(r.result.data, "base64"));
}
await send("Runtime.enable");
await send("Log.enable");
await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: W, height: H, deviceScaleFactor: 2, mobile: W < 700 });
await send("Page.navigate", { url: URL_ });

/** Aborta se o app não estiver mudo (volume 0 e ganho 0). */
async function assertSilent(when) {
  const v = await evaluate("({pct: window.__app?.volumePct ?? null, fonte: window.__app?.volumeSource ?? null, ganho: window.__sound ? window.__sound.gain : null, ganhoNo: window.__sound?.out ? window.__sound.out.gain.value : null})");
  if (!v || (v.pct === null && v.ganho === null)) return { carregando: true }; // app.js ainda não rodou: nenhum som possível
  const okSilent = v.pct === 0 && v.ganho === 0 && (v.ganhoNo === null || v.ganhoNo === 0);
  if (!okSilent) {
    console.error(`ABORTADO (${when}): app não está mudo: ${JSON.stringify(v)}`);
    try { ws.close(); } catch { /* ignora */ }
    chrome.kill("SIGKILL");
    process.exit(3);
  }
  return v;
}
const clicks = (args["click-at"] || "").split(",").filter(Boolean).map((s) => s.split(":"));
const soundAt = args["click-sound"] ? Number(args["click-sound"]) : null;
const samples = [];
let silentInfo = null;
for (let s = 1; s <= SECONDS; s++) {
  await sleep(1000);
  if (s === 2 || s % 5 === 0) silentInfo = await assertSilent(`${s} s`);
  if (soundAt === s) {
    await trustedClick("btn-sound");
    await sleep(300);
    await assertSilent("após Testar som");
    samples.push({ s, som: await evaluate("({ctx: window.__sound?.ctx?.state ?? null, sirene: !!window.__sound?.siren, ganhoFinal: window.__sound?.out ? window.__sound.out.gain.value : null, erro: window.__sound?.error ?? null})") });
  }
  for (const [when, el] of clicks) if (Number(when) === s) await trustedClick(el);
  if (args["open-tech"] && s === 2) await evaluate("document.getElementById('tech').open = true");
  if (s % 5 === 0) {
    const st = await evaluate(`(() => { const a = window.__app; const r = a && a.last; return r ? { fase: r.phase, estado: r.score?.state, nivel: Math.round(r.score?.level ?? 0), forcado: r.score?.forced ?? null, fps: a.fps, rosto: a.faceDetected, titulo: document.getElementById('status-title').textContent, sirene_pedida: !!a.sirenWanted, sirene_nos_audio: !!window.__sound?.siren, flash: document.body.classList.contains('flashing') } : { titulo: document.getElementById('status-title')?.textContent }; })()`);
    samples.push({ s, ...st });
  }
  if (SHOTS.includes(s)) await shot(`tela_${s}s.png`);
}
const final = await evaluate(`(() => {
  const a = window.__app;
  const vw = window.innerWidth;
  const over = [...document.querySelectorAll('body *')].filter((e) => { const r = e.getBoundingClientRect(); return r.width > 0 && r.right > vw + 0.5; }).map((e) => e.id || e.className || e.tagName).slice(0, 10);
  return {
    eventos: a ? a.eventLog.map((e) => ({ hora: e.wall, tipo: e.type, texto: e.text })) : null,
    ultimo: a && a.last ? { fase: a.last.phase, score: a.last.score && { level: a.last.score.level, state: a.last.score.state, forced: a.last.score.forced, contributions: a.last.score.contributions }, eyes: a.last.eyes, yawn: a.last.yawn, head: a.last.head } : null,
    base: a ? a.monitor.baseline : null,
    stats: a ? a.monitor.stats : null,
    modelos: { landmarker: document.getElementById('s-landmarker').textContent, eye: document.getElementById('s-eye').textContent, lstm: document.getElementById('s-temporal').textContent, audio: document.getElementById('s-audio').textContent },
    aviso: document.getElementById('notice').textContent,
    latencia: a ? Object.fromEntries(Object.entries(a.lat).map(([k, v]) => [k, { p50: v.p50, p95: v.p95, n: v.n }])) : null,
    layout: { innerWidth: vw, scrollWidth: document.documentElement.scrollWidth, elementosAlemDaBorda: over },
    bench: window.__benchSummary || null,
  };
})()`);
if (SHOTS.includes(SECONDS + 1)) await shot(`tela_final.png`);
const rel = { url: URL_, video: args.video ? path.basename(args.video) : "padrão sintético do Chrome (sem rosto)", segundos: SECONDS, viewport: `${W}x${H}`, chrome: (await getJSON(`http://127.0.0.1:${PORT}/json/version`)).Browser, flags, silencio: silentInfo, amostras: samples, final, console: logs };
fs.writeFileSync(path.join(OUT, "relatorio.json"), JSON.stringify(rel, null, 1));
const exc = logs.filter((l) => l.kind === "EXCECAO" || l.kind === "console.error");
console.log(`chrome ${rel.chrome}; ${logs.length} mensagens de console, ${exc.length} erros/exceções; relatório em ${path.join(OUT, "relatorio.json")}`);
ws.close();
chrome.kill("SIGTERM");
await sleep(500);
fs.rmSync(profile, { recursive: true, force: true });
process.exit(0);
