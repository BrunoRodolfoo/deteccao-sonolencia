/**
 * Roda a PWA no Chrome headless com um VÍDEO REAL como câmera falsa e grava, quadro a quadro, o que o
 * motor do app decidiu (fase, olhos, bocejo, cabeça, nível, estado, alarme) e os eventos, com o relógio
 * do vídeo (video.currentTime) para comparar com a anotação. DevTools Protocol, sem dependências (Node >= 22).
 *
 *   node tools/app_validacao/rodar_app_video.mjs --clip clipe.y4m --fim 196 --out saida.json \
 *        [--url https://127.0.0.1:8766/] [--perfil pasta]
 *
 * --fim: segundo do relógio do vídeo em que a coleta termina (fim do trecho útil do clipe; o Chrome
 * repete o arquivo em laço depois do fim, por isso o clipe tem tela preta no fim e a coleta para antes).
 *
 * SILÊNCIO OBRIGATÓRIO (o teste roda com gente perto do computador; o Chrome headless novo toca som de verdade):
 *  (1) o Chrome sempre sobe com --mute-audio; (2) o app sempre abre com ?volume=0 (qualquer outro valor é
 *  trocado por 0); (3) a cada 1 s o script confere volume 0 % e ganho 0 e ABORTA se não estiverem; (4) desde
 *  o ajuste final, o Chrome roda com a política de autoplay PADRÃO (sem --autoplay-policy=no-user-gesture-
 *  required): sem toque do usuário o AudioContext nem é criado pelo app (?autostart=1 não é gesto). O vídeo da
 *  câmera toca mesmo assim, porque o <video> é muted. --autoplay-livre devolve a flag antiga, se precisar.
 * Nada de imagem é salvo: só números por quadro e eventos.
 *
 * Como coleta: um script injetado antes do app (Page.addScriptToEvaluateOnNewDocument) envolve
 * window.__app.monitor.update (o DrowsinessMonitor do engine.js) e guarda o resultado de cada quadro.
 * O código do app não é alterado.
 */
import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, arr) => {
    if (a.startsWith("--")) acc.push([a.slice(2), arr[i + 1] && !arr[i + 1].startsWith("--") ? arr[i + 1] : "1"]);
    return acc;
  }, []),
);
if (!args.clip || !args.fim || !args.out) {
  console.error("uso: --clip clipe.y4m --fim <s> --out saida.json");
  process.exit(2);
}
const URL_ = (() => {
  const u = new URL(args.url || "https://127.0.0.1:8766/");
  u.searchParams.set("autostart", "1");
  u.searchParams.set("volume", "0"); // sempre mudo
  return u.toString();
})();
const FIM = Number(args.fim);
const MAX_WALL_S = Number(args["max-s"] || FIM + 120);
const PORT = 9400 + Math.floor(Math.random() * 400);
const CHROME = args.chrome || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const profile = args.perfil ? path.resolve(args.perfil) : fs.mkdtempSync(path.join(os.tmpdir(), "perfil-val-"));
fs.mkdirSync(profile, { recursive: true });

const flags = [
  "--headless=new",
  "--mute-audio", // obrigatório: nunca tocar som no computador
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${profile}`,
  "--use-fake-ui-for-media-stream",
  "--use-fake-device-for-media-stream",
  `--use-file-for-fake-video-capture=${path.resolve(args.clip)}`,
  "--ignore-certificate-errors",
  ...(args["autoplay-livre"] ? ["--autoplay-policy=no-user-gesture-required"] : []),
  "--no-first-run",
  "--no-default-browser-check",
  "--disable-background-timer-throttling",
  "--disable-renderer-backgrounding",
  "--window-size=390,844",
];
const chrome = spawn(CHROME, [...flags, "about:blank"], { stdio: ["ignore", "ignore", "ignore"] });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
function die(msg, code = 3) {
  console.error(msg);
  try { chrome.kill("SIGKILL"); } catch { /* ignora */ }
  process.exit(code);
}
async function getJSON(u) {
  for (let i = 0; i < 150; i++) {
    try {
      const r = await fetch(u);
      if (r.ok) return await r.json();
    } catch { /* subindo */ }
    await sleep(100);
  }
  die(`sem resposta de ${u}`);
}
const targets = await getJSON(`http://127.0.0.1:${PORT}/json/list`);
const page = targets.find((t) => t.type === "page");
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r, { once: true }));
let id = 0;
const pending = new Map();
const logs = [];
ws.addEventListener("message", (ev) => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
    return;
  }
  if (m.method === "Runtime.exceptionThrown") {
    const d = m.params.exceptionDetails;
    logs.push({ kind: "EXCECAO", text: `${d.text} ${d.exception?.description || ""}`.slice(0, 400) });
  } else if (m.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(m.params.type)) {
    logs.push({ kind: `console.${m.params.type}`, text: m.params.args.map((a) => (a.value !== undefined ? String(a.value) : a.description || a.type)).join(" ").slice(0, 400) });
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

// Coletor injetado antes do app: envolve monitor.update e guarda o resultado de cada quadro.
const HOOK = `(() => {
  const V = (window.__val = { rows: [], ev: [], upd: [], hooked: false });
  const iv = setInterval(() => {
    const a = window.__app;
    if (!a || !a.monitor || V.hooked) return;
    V.hooked = true;
    clearInterval(iv);
    const mon = a.monitor;
    const orig = mon.update.bind(mon);
    const vid = document.getElementById("video");
    mon.update = (t, f) => {
      const r = orig(t, f);
      const vt = vid ? vid.currentTime : null;
      const ph = r.phase === "ativo" ? 1 : r.phase === "calibrando" ? 0 : -1;
      V.rows.push([t, vt, f.face ? 1 : 0, ph, r.eyes && r.eyes.closed ? 1 : 0, r.eyes ? r.eyes.closedMs : 0,
        r.yawn && r.yawn.open ? 1 : 0, r.head ? r.head.dPitch : null, r.head ? r.head.dYaw : null, r.head ? r.head.dRoll : null,
        r.head ? r.head.state : "", r.score ? r.score.level : 0, r.score ? r.score.state : 0, r.score && r.score.forced ? 1 : 0,
        r.alarm && r.alarm.siren ? 1 : 0, r.eyes ? r.eyes.perclos : null, f.blend ? f.blend.jawOpen : null,
        f.pose ? f.pose.pitch : null, r.calib ? r.calib.blinks ?? null : null,
        f.blend ? f.blend.eyeBlinkLeft : null, f.blend ? f.blend.eyeBlinkRight : null, f.ear ? f.ear.mean : null,
        r.score ? (r.score.all || []).map((c) => c.key + ":" + Math.round(c.points)).join("|") : "",
        r.eyes && r.eyes.perclosAll !== undefined ? r.eyes.perclosAll : null, r.score && r.score.mlCounted ? 1 : 0,
        r.score && r.score.postForcedMs ? r.score.postForcedMs : 0]);
      for (const e of r.events) V.ev.push({ t, vt, type: e.type, id: e.id, durMs: e.durMs ?? null, peakDeg: e.peakDeg ?? null, peakRollDeg: e.peakRollDeg ?? null, byRoll: e.byRoll ?? null, lost: e.lost ?? null, motivo: e.motivo ?? null, baseline: e.baseline ?? null, baseMs: e.baseMs ?? null });
      for (const u of r.updates) V.upd.push({ t, vt, id: u.id, durMs: u.durMs ?? null });
      return r;
    };
  }, 5);
})();`;
await send("Runtime.enable");
await send("Page.enable");
await send("Page.addScriptToEvaluateOnNewDocument", { source: HOOK });
await send("Emulation.setDeviceMetricsOverride", { width: 390, height: 844, deviceScaleFactor: 2, mobile: true });
await send("Page.navigate", { url: URL_ });

const out = { url: URL_, clip: path.basename(args.clip), fim_s: FIM, chrome: null, colunas: ["t", "vt", "face", "fase", "fechado", "fechado_ms", "boca_aberta", "dpitch", "dyaw", "droll", "cabeca", "nivel", "estado", "forcado", "sirene", "perclos", "jaw", "pitch", "piscadas_calib", "blink_l", "blink_r", "ear", "pontos", "perclos_piscadas", "lstm_soma", "pos_evento_ms"], rows: [], ev: [], upd: [], silencio: [], console: logs, flags };
out.chrome = (await getJSON(`http://127.0.0.1:${PORT}/json/version`)).Browser;
const t0 = Date.now();
let lastVt = 0;
let stalls = 0;
for (;;) {
  await sleep(1000);
  const el = (Date.now() - t0) / 1000;
  // silêncio: volume 0 % e ganho 0 (se o app já carregou); aborta caso contrário
  const v = await evaluate("({pct: window.__app?.volumePct ?? null, fonte: window.__app?.volumeSource ?? null, ganho: window.__sound ? window.__sound.gain : null, ganhoNo: window.__sound?.out ? window.__sound.out.gain.value : null})");
  if (v && (v.pct !== null || v.ganho !== null)) {
    if (!(v.pct === 0 && v.ganho === 0 && (v.ganhoNo === null || v.ganhoNo === 0))) die(`ABORTADO: app não está mudo: ${JSON.stringify(v)}`);
    if (out.silencio.length < 3 || Math.round(el) % 30 === 0) out.silencio.push({ s: Math.round(el), ...v });
  }
  const chunk = await evaluate("(() => { const V = window.__val; if (!V) return null; return { rows: V.rows.splice(0), ev: V.ev.splice(0), upd: V.upd.splice(0), hooked: V.hooked, vt: document.getElementById('video')?.currentTime ?? 0 }; })()");
  if (chunk && !chunk.erro) {
    for (const r of chunk.rows) out.rows.push(r);
    for (const e of chunk.ev) out.ev.push(e);
    for (const u of chunk.upd) out.upd.push(u);
    if (chunk.vt === lastVt && chunk.vt > 0) stalls++;
    lastVt = chunk.vt;
    if (chunk.vt >= FIM) break;
  }
  if (stalls > 10) die("vídeo parado por > 10 s");
  if (el > MAX_WALL_S) die(`tempo esgotado (${el} s); vt=${lastVt}`);
}
out.final = await evaluate(`(() => { const a = window.__app; return { baseline: a.monitor.baseline, stats: a.monitor.stats, delegate: a.delegate, eyeFile: a.eyeFile, tmpFile: a.tmpFile, video: [document.getElementById('video').videoWidth, document.getElementById('video').videoHeight], fps: a.fps, eventLog: a.eventLog.map((e) => ({ hora: e.wall, tipo: e.type, texto: e.text })) }; })()`);
out.parede_s = (Date.now() - t0) / 1000;
fs.writeFileSync(args.out, JSON.stringify(out));
const exc = logs.filter((l) => l.kind === "EXCECAO").length;
console.log(`${out.clip}: ${out.rows.length} quadros, ${out.ev.length} eventos, vt final ${lastVt.toFixed(1)} s, exceções JS ${exc}, chrome ${out.chrome}`);
ws.close();
chrome.kill("SIGTERM");
await sleep(800);
if (!args.perfil) fs.rmSync(profile, { recursive: true, force: true });
process.exit(0);
