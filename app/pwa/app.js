/**
 * app.js — PWA de detecção de sonolência (iPhone/Android no navegador, também desktop).
 *
 * Pipeline por quadro (detector v2.1, ajuste final de 27/09/2026):
 *   câmera -> MediaPipe Face Landmarker (GPU/CPU): 478 marcos + 52 blendshapes + matriz de pose 4x4
 *          -> sinais interpretáveis (engine.js, DrowsinessMonitor):
 *               OLHOS  = EAR calibrado + eyeBlink (blendshape), com trava de olhar para baixo;
 *                        piscadas, fechamentos >= 0,5 s, microssono >= 1 s, PERCLOS 60 s (só fechamentos
 *                        >= 0,5 s: piscadas normais não contam)
 *               BOCEJO = jawOpen (blendshape) / MAR relativo sustentado >= 1,5 s
 *               CABEÇA = pitch/yaw/roll da matriz -> queda >= 18° por 0,8 s, cabeceio
 *          -> nível 0-100 (soma de pontos explicada em "Por que N", na tela Em viagem) com histerese
 *          -> ALERTA (verde) / ATENÇÃO (amarelo + ding) / SONOLENTO (vermelho + sirene + tela de alarme)
 *   Em paralelo, só como informação: recorte dos olhos 48x48 -> eye_state.tflite (CNN, Exp. 3) e
 *   4 características por piscada -> z-score -> janela 30x4 -> blink_temporal.tflite (LSTM, Exp. 1),
 *   que soma no máximo +10 pontos, e só quando os sinais visíveis já estão em ATENÇÃO (>= 40).
 * Som: sirene/ding pelo Web Audio, liberado no primeiro toque (iOS), com o controle "Volume do alarme"
 * (padrão 20 %, salvo no aparelho; ?volume=0..100 na URL sobrepõe sem gravar — testes usam ?volume=0).
 *
 * Sem etapa de build: vanilla JS + ES modules. Dependências de ./vendor/ ou, se faltarem, da CDN. Todos os
 * textos em pt-BR.
 *
 * Interface (27/10/2026): telas no estilo do iPhone, com tema claro e escuro (Ajustes > Aparência). Início ->
 * Calibração (câmera em tela cheia) -> Em viagem (mostrador do nível, sinais, "Por que N") -> alarme vermelho em
 * tela cheia -> Encerrar -> resumo da viagem (aba Viagens). Detector, parâmetros e benchmark iguais aos da 2.1.
 */
import * as E from "./engine.js";
import { adaptEyeMeta, tfliteIO, runTflite as runTfliteRaw } from "./tflite_io.js";

// ---------------------------------------------------------------------------
// Dependências: local primeiro, CDN como reserva. Cada item é [cópia em ./vendor/, CDN], nas mesmas versões
// fixas (URLs da CDN verificadas com HTTP 200 em 20/09/2026). A cópia local vem de
// tools/baixar_dependencias_app.sh (mude as versões lá e aqui juntas) e faz o app funcionar sem internet
// quando servido pelo serve_https.py (celular sem dados, ligado ao computador pelo Acesso Pessoal).
// ---------------------------------------------------------------------------
const NPM = "https://cdn.jsdelivr.net/npm/";
const local = (p) => new URL(`vendor/${p}`, location.href).href;
const MP_BUNDLE = [local("tasks-vision@1.0.1/vision_bundle.mjs"), `${NPM}@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs`];
const MP_WASM = [local("tasks-vision@1.0.1/wasm"), `${NPM}@mediapipe/tasks-vision@1.0.1/wasm`];
const MP_MODEL = [
  local("face_landmarker/float16/1/face_landmarker.task"),
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
];
// TensorFlow.js core + backend CPU + tfjs-tflite: scripts clássicos, nesta ordem (expõem window.tf e window.tflite).
const TF_SCRIPTS = [
  [local("tfjs-core@4.22.0/dist/tf-core.min.js"), `${NPM}@tensorflow/tfjs-core@4.22.0/dist/tf-core.min.js`],
  [local("tfjs-backend-cpu@4.22.0/dist/tf-backend-cpu.min.js"), `${NPM}@tensorflow/tfjs-backend-cpu@4.22.0/dist/tf-backend-cpu.min.js`],
  [local("tfjs-tflite@0.0.1-alpha.10/dist/tf-tflite.min.js"), `${NPM}@tensorflow/tfjs-tflite@0.0.1-alpha.10/dist/tf-tflite.min.js`],
];
// Na versão 0.0.1-alpha.10 os módulos WASM ficam em /wasm/ (não em /dist/).
const TFLITE_WASM = [local("tfjs-tflite@0.0.1-alpha.10/wasm/"), `${NPM}@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/`];
const MODELS_DIR = "./models/";

const APP_VERSION = "2.1";
const EYE_SIZE_DEFAULT = 48;
const UI_INTERVAL_MS = 100;
const BENCH_MS = 60000;
const MAX_LOG_ITEMS = 80;
// Sirene de dois tons (hi-lo): 880 <-> 1250 Hz a cada 0,35 s, onda quadrada (harmônicos na faixa em que o
// alto-falante do celular rende mais). Ganho alto na mistura; depois vêm o VOLUME DO ALARME (controle na
// tela, padrão 20 %) e o volume de mídia do aparelho.
const SIREN = { lo: 880, hi: 1250, stepS: 0.35, gain: 0.55 };
const BLEND_KEYS = ["eyeBlinkLeft", "eyeBlinkRight", "jawOpen", "eyeLookDownLeft", "eyeLookDownRight", "eyeSquintLeft", "eyeSquintRight"];

// ---------------------------------------------------------------------------
// Estado global
// ---------------------------------------------------------------------------
const S = {
  landmarker: null,
  delegate: "--",
  stream: null,
  running: false,
  lastVideoTime: -1,
  frameCount: 0,
  fps: 0,
  fpsWindow: [],
  // detector v2 (engine.js)
  monitor: new E.DrowsinessMonitor(),
  last: null, // último resultado do monitor
  faceLostSince: null,
  // caminho do LSTM (informativo)
  detector: null, // BlinkDetector no EAR, criado após a calibração (mesmo limiar)
  normalizer: new E.SubjectNormalizer({ nCalib: 20 }),
  blinkWindow: new E.BlinkWindow({ size: 30, nFeatures: 4, minBlinks: 10, padPartial: true }),
  smoother: new E.ClassSmoother({ alpha: 0.3, nClasses: 3, history: 10 }),
  lat: { lm: new E.LatencyStats(300), eye: new E.LatencyStats(300), tmp: new E.LatencyStats(300), mon: new E.LatencyStats(300), tot: new E.LatencyStats(300) },
  // modelos TFLite
  tf: null,
  tflite: null,
  eyeModel: null,
  eyeMeta: null,
  eyeIO: null,
  eyeFile: null,
  tmpModel: null,
  tmpMeta: null,
  tmpIO: null,
  tmpFile: null,
  // últimos valores
  ear: { left: 0, right: 0, mean: 0 },
  mar: 0,
  blend: null,
  pose: null,
  eyeProb: { left: null, right: null },
  temporal: null, // {probs, cls, vote, partial, nBlinks}
  lastBlink: null,
  blinkLog: [],
  predLog: [],
  eventLog: [], // {id, type, wall, text, kind}
  faceDetected: false,
  // benchmark
  bench: null,
  // alertas e som
  sirenWanted: false,
  soundTesting: false,
  soundTimers: [0, 0],
  volumePct: E.SOUND.defaultVolumePct, // volume do alarme (0-100 %)
  volumeSource: "padrão", // "padrão" | "salvo" | "url"
  userTapped: false, // houve toque real (isTrusted) na página: libera a vibração
  wakeLock: null,
  wakeWanted: false,
  lastUi: 0,
  wasmSimd: null,
  // dependências (./vendor/ ou CDN)
  visionReady: null, // promessa: origem do módulo do MediaPipe (0 = local, 1 = CDN)
  tfReady: null, // promessa: TensorFlow.js + tfjs-tflite carregados
  tfliteWasm: TFLITE_WASM[1],
  deps: { mediapipe: "--", tfjs: "--" }, // "local", "CDN" ou "local + CDN"
};
window.__app = S; // inspeção em testes automatizados (Chrome headless)

// Modelo do aparelho: o Chrome no Android reduz o user-agent (o modelo vira "K"), mas informa o
// modelo real pelos User-Agent Client Hints (ex.: "SM-S918B" = Galaxy S23 Ultra). No iOS não há
// equivalente; lá o nome vem de results/phone/aparelhos.json (tools/import_phone_benchmark.py).
if (navigator.userAgentData && navigator.userAgentData.getHighEntropyValues) {
  navigator.userAgentData
    .getHighEntropyValues(["model", "platformVersion"])
    .then((h) => { S.uaHints = h; })
    .catch(() => {});
}

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const octx = overlay.getContext("2d");
const cropCanvas = document.createElement("canvas");
const cctx = cropCanvas.getContext("2d", { willReadFrequently: true });
const cropLeft = $("crop-left").getContext("2d");
const cropRight = $("crop-right").getContext("2d");

// ---------------------------------------------------------------------------
// Utilidades de UI
// ---------------------------------------------------------------------------
function setBadge(text, kind = "") {
  const b = $("badge-state");
  b.textContent = text;
  b.className = `badge ${kind}`;
}
/** Faixa de aviso no topo da tela (toque para fechar). */
function notice(msg, kind = "") {
  const n = $("notice");
  if (!msg) {
    n.textContent = "";
    return n.classList.add("hidden");
  }
  n.textContent = msg;
  n.className = `aviso ${kind}`;
}
function appendNotice(msg) {
  const n = $("notice");
  n.classList.remove("hidden");
  n.textContent = n.textContent ? `${n.textContent}\n${msg}` : msg;
}
function fmt(v, d = 3) {
  return v === null || v === undefined || Number.isNaN(v) ? "--" : Number(v).toFixed(d);
}
const br = (v, d = 1) => E.fmtBR(v, d);
function fmtMs(v) {
  return fmt(v, 1);
}
function signed(v, d = 0) {
  if (!Number.isFinite(v)) return "--";
  const s = v.toFixed(d).replace(".", ",");
  return v > 0 ? `+${s}` : s;
}
function wallClock() {
  return new Date().toLocaleTimeString("pt-BR", { hour12: false });
}
function deviceInfo() {
  const nav = navigator;
  return {
    userAgent: nav.userAgent,
    hardwareConcurrency: nav.hardwareConcurrency ?? null,
    deviceMemoryGB: nav.deviceMemory ?? null,
    platform: nav.userAgentData?.platform ?? nav.platform ?? null,
    screen: `${screen.width}x${screen.height}@${window.devicePixelRatio}`,
    secureContext: window.isSecureContext,
    crossOriginIsolated: window.crossOriginIsolated ?? false,
    wasmSimd: null,
    webgl2: !!document.createElement("canvas").getContext("webgl2"),
    audioSession: "audioSession" in navigator,
    vibrate: "vibrate" in navigator,
    wakeLock: "wakeLock" in navigator,
    language: nav.language,
    date: new Date().toISOString(),
  };
}
async function detectSimd() {
  // Teste padrão de SIMD do WebAssembly (módulo mínimo com instrução v128).
  try {
    const bytes = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);
    return WebAssembly.validate(bytes);
  } catch {
    return false;
  }
}
function showDeviceInfo(extra = {}) {
  const info = { ...deviceInfo(), ...extra };
  $("device-info").textContent = Object.entries(info)
    .map(([k, v]) => `${k}: ${v}`)
    .join("\n");
  return info;
}

// ---------------------------------------------------------------------------
// Som (Web Audio) e volume do alarme.
// No iOS o AudioContext só toca se for criado/retomado DENTRO de um toque do usuário; por isso
// Sound.unlock() é chamado de forma síncrona (antes de qualquer await) no clique de "Iniciar viagem",
// "Testar som" e "Recalibrar" e, com a câmera ligada, em qualquer toque na página enquanto o áudio
// não estiver ativo (click/touchend/keydown, os eventos que o WebKit aceita como gesto).
// navigator.audioSession.type = "playback" (Audio Session API do WebKit, iOS 16.4+, também no Chrome
// do iOS) é definido ANTES de criar o AudioContext e faz o som tocar mesmo com a chave de silencioso
// ligada; sem essa API, um <audio> silencioso em loop tem o mesmo efeito. Efeito colateral do
// "playback": ao liberar o som, o iOS pausa outro áudio que não se mistura (ex.: música de outro app).
// Cadeia: sirene/ding -> mix -> compressor (evita saturar quando sirene e ding se sobrepõem) ->
// VOLUME DO ALARME (ganho = E.volumeGain(%), padrão 20 %, ?volume= na URL) -> alto-falante.
// O volume de mídia do aparelho continua valendo por cima: o app não consegue aumentá-lo.
// ---------------------------------------------------------------------------
function silentWavUrl(seconds = 0.5, rate = 8000) {
  const n = Math.round(seconds * rate);
  const buf = new ArrayBuffer(44 + n);
  const v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + n, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true);
  v.setUint32(28, rate, true); v.setUint16(32, 1, true); v.setUint16(34, 8, true); w(36, "data"); v.setUint32(40, n, true);
  for (let i = 0; i < n; i++) v.setUint8(44 + i, 128); // PCM 8 bits: 128 = silêncio
  return URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
}
const Sound = {
  ctx: null,
  mix: null,
  out: null, // ganho final = volume do alarme
  siren: null,
  htmlAudio: null,
  error: null,
  volumePct: E.SOUND.defaultVolumePct,
  get ready() {
    return !!this.ctx && this.ctx.state === "running";
  },
  get gain() {
    return E.volumeGain(this.volumePct);
  },
  /** Aplica o volume do alarme (0-100 %) com uma rampa curta (sem estalo). */
  setVolume(pct) {
    this.volumePct = pct;
    if (!this.out) return;
    const t = this.ctx.currentTime;
    const g = this.out.gain;
    g.cancelScheduledValues(t);
    g.setValueAtTime(g.value, t);
    g.linearRampToValueAtTime(this.gain, t + 0.03);
  },
  unlock() {
    try {
      if (navigator.audioSession && navigator.audioSession.type !== "playback") navigator.audioSession.type = "playback";
    } catch (e) {
      this.error = `audioSession: ${e.message}`;
    }
    try {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) {
        this.error = "Web Audio indisponível neste navegador";
        updateAudioStatus();
        return false;
      }
      if (!this.ctx) {
        const ctx = new AC();
        this.ctx = ctx;
        this.out = ctx.createGain();
        this.out.gain.value = this.gain; // volume aplicado ANTES de ligar ao alto-falante
        const comp = ctx.createDynamicsCompressor();
        comp.threshold.value = -8;
        comp.knee.value = 6;
        comp.ratio.value = 4;
        this.mix = ctx.createGain();
        this.mix.gain.value = 1;
        this.mix.connect(comp);
        comp.connect(this.out);
        this.out.connect(ctx.destination);
        ctx.onstatechange = () => updateAudioStatus();
      }
      if (this.ctx.state !== "running") this.ctx.resume().catch(() => {});
      // Buffer silencioso tocado dentro do gesto: é o que "libera" o áudio no iOS.
      const b = this.ctx.createBuffer(1, 1, this.ctx.sampleRate);
      const src = this.ctx.createBufferSource();
      src.buffer = b;
      src.connect(this.out);
      src.start(0);
    } catch (e) {
      this.error = e.message;
    }
    if (!("audioSession" in navigator) && /iPhone|iPad|iPod/.test(navigator.userAgent) && !this.htmlAudio) {
      try {
        const a = new Audio(silentWavUrl());
        a.loop = true;
        a.setAttribute("playsinline", "");
        a.play().catch(() => {});
        this.htmlAudio = a;
      } catch {
        /* ignora */
      }
    }
    updateAudioStatus();
    return true;
  },
  startSiren() {
    if (!this.ctx || this.siren) return;
    if (this.ctx.state !== "running") this.ctx.resume().catch(() => {});
    const ctx = this.ctx;
    const t = ctx.currentTime;
    const osc = ctx.createOscillator();
    osc.type = "square";
    osc.frequency.value = (SIREN.hi + SIREN.lo) / 2;
    const lfo = ctx.createOscillator();
    lfo.type = "square";
    lfo.frequency.value = 1 / (2 * SIREN.stepS);
    const depth = ctx.createGain();
    depth.gain.value = (SIREN.hi - SIREN.lo) / 2;
    lfo.connect(depth).connect(osc.frequency);
    const g = ctx.createGain();
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(SIREN.gain, t + 0.03);
    osc.connect(g).connect(this.mix);
    // Ao parar, desliga todos os nós da sirene do grafo: ligar/desligar a sirene muitas vezes numa
    // sessão longa não acumula nós pendurados no mixer.
    osc.onended = () => {
      for (const n of [osc, lfo, depth, g]) {
        try {
          n.disconnect();
        } catch {
          /* já desligado */
        }
      }
    };
    osc.start(t);
    lfo.start(t);
    this.siren = { osc, lfo, g };
    this.nodeStarts = (this.nodeStarts || 0) + 1;
  },
  stopSiren() {
    const s = this.siren;
    if (!s) return;
    this.siren = null;
    try {
      const t = this.ctx.currentTime;
      s.g.gain.cancelScheduledValues(t);
      s.g.gain.setValueAtTime(s.g.gain.value, t);
      s.g.gain.linearRampToValueAtTime(0, t + 0.06);
      s.osc.stop(t + 0.08);
      s.lfo.stop(t + 0.08);
    } catch {
      /* ignora */
    }
  },
  ding() {
    if (!this.ctx) return;
    if (this.ctx.state !== "running") this.ctx.resume().catch(() => {});
    const ctx = this.ctx;
    const t = ctx.currentTime + 0.01;
    // "ding-dong" curto: E6 e depois C6, com o 2º harmônico, decaimento exponencial de 0,7 s
    for (const [f, dt, a] of [[1318.5, 0, 0.6], [1046.5, 0.24, 0.5]]) {
      const g = ctx.createGain();
      g.gain.setValueAtTime(0.0001, t + dt);
      g.gain.exponentialRampToValueAtTime(a, t + dt + 0.012);
      g.gain.exponentialRampToValueAtTime(0.0001, t + dt + 0.7);
      g.connect(this.mix);
      for (const [mult, lvl] of [[1, 1], [2, 0.25]]) {
        const o = ctx.createOscillator();
        o.type = "sine";
        o.frequency.value = f * mult;
        const og = ctx.createGain();
        og.gain.value = lvl;
        o.connect(og).connect(g);
        o.onended = () => {
          try {
            o.disconnect();
            og.disconnect();
            if (mult === 1) g.disconnect(); // 1 ganho por nota, compartilhado pelos 2 harmônicos (mesmo fim)
          } catch {
            /* já desligado */
          }
        };
        o.start(t + dt);
        o.stop(t + dt + 0.75);
      }
    }
  },
};
window.__sound = Sound; // inspeção em testes automatizados

// ---- volume do alarme: controle na tela, salvo no aparelho; ?volume= na URL sobrepõe sem gravar ----
function readStoredVolume() {
  try {
    return window.localStorage.getItem(E.SOUND.storageKey);
  } catch {
    return null; // modo privado, dados bloqueados etc.
  }
}
function saveVolume(pct) {
  if (S.volumeSource === "url") return false; // ?volume= é só para esta visita (testes): nunca grava
  try {
    window.localStorage.setItem(E.SOUND.storageKey, String(pct));
    S.volumeSource = "salvo";
    return true;
  } catch {
    return false;
  }
}
function showVolume() {
  $("vol-out").textContent = `${S.volumePct} %`;
  $("vol").setAttribute("aria-valuetext", `${S.volumePct} %`);
  for (const id of ["vol-out2", "som-pct", "som-pct-inicio"]) setText(id, `${S.volumePct} %`);
  updateAudioStatus();
}
function initVolume() {
  const params = new URLSearchParams(location.search);
  const r = E.resolveVolume({ urlValue: params.has("volume") ? params.get("volume") : null, storedValue: readStoredVolume() });
  S.volumePct = r.pct;
  S.volumeSource = r.source;
  Sound.setVolume(r.pct);
  const el = $("vol");
  el.value = String(r.pct);
  el.addEventListener("input", () => {
    const v = E.parseVolumePct(el.value);
    if (v === null) return;
    S.volumePct = v;
    Sound.setVolume(v);
    showVolume();
  });
  el.addEventListener("change", () => {
    saveVolume(S.volumePct);
    // Prévia curta no volume escolhido (só se o som já foi liberado por um toque).
    if (S.volumePct > 0 && Sound.ready && !S.sirenWanted) Sound.ding();
    showVolume();
  });
  showVolume();
}

function updateAudioStatus() {
  const st = Sound.ctx ? Sound.ctx.state : "não iniciado";
  const sess = "audioSession" in navigator ? `audioSession=${navigator.audioSession.type}` : "sem audioSession (iOS antigo, Android ou computador)";
  $("s-audio").textContent = `${st}; ${sess}; volume do alarme ${S.volumePct} % (${S.volumeSource}; ganho ${fmt(Sound.gain, 3)})${Sound.error ? `; erro: ${Sound.error}` : ""}`;
  const hint = $("sound-hint");
  let txt;
  if (S.volumePct === 0) txt = "Volume do alarme em 0 %: o alarme NÃO vai tocar (a tela ainda pisca e, no Android, o celular vibra).";
  else if (!Sound.ctx) txt = 'Toque em "Testar som" uma vez para liberar o som (o iPhone exige um toque).';
  else if (Sound.ctx.state !== "running") txt = 'Som pausado pelo sistema: toque em "Testar som" para reativar.';
  else if (S.volumePct < 50) txt = `Volume baixo (${S.volumePct} %), bom para testar sem susto. Na demonstração, suba para 100 % e deixe o volume de mídia do aparelho alto.`;
  else txt = `Som ativo (${S.volumePct} %). O volume final também depende do volume de mídia do aparelho.`;
  if (S.volumeSource === "url") txt += " (Volume definido pela URL ?volume=; não é salvo.)";
  hint.textContent = txt;
  hint.classList.toggle("warn", S.volumePct === 0);
}
function testSound() {
  Sound.unlock();
  S.soundTesting = true;
  Sound.ding();
  clearTimeout(S.soundTimers[0]);
  clearTimeout(S.soundTimers[1]);
  S.soundTimers[0] = setTimeout(() => {
    Sound.startSiren();
    S.soundTimers[1] = setTimeout(() => {
      S.soundTesting = false;
      if (!S.sirenWanted) Sound.stopSiren();
    }, 1600);
  }, 900);
  vibrate("aviso");
}
function vibrate(kind) {
  try {
    // Só depois de um toque real: o Chrome bloqueia (e reclama no console) vibração sem gesto do usuário.
    if (!navigator.vibrate || !S.userTapped) return;
    navigator.vibrate(kind === "alarme" ? [400, 150, 400, 150, 400] : [120, 80, 120]);
  } catch {
    /* ignora */
  }
}
function applyAlarm(al) {
  S.sirenWanted = al.siren;
  if (al.siren) Sound.startSiren();
  else if (!S.soundTesting) Sound.stopSiren();
  if (al.ding) Sound.ding();
  if (al.vibrate) vibrate(al.vibrate);
  document.body.classList.toggle("flashing", !!al.flash);
}

// ---------------------------------------------------------------------------
// Dependências externas (MediaPipe e TensorFlow.js): ./vendor/ primeiro, CDN como reserva. Começam a
// carregar ao abrir a página, como os scripts do <head> antes, e o "Iniciar câmera" espera por elas.
// ---------------------------------------------------------------------------
let vision = null; // módulo do MediaPipe Tasks Vision (FaceLandmarker, FilesetResolver)

/** Tenta load(url) com cada URL de [local, CDN]; devolve o índice da que funcionou (0 = local). */
async function firstAvailable(urls, load) {
  let err = null;
  for (let i = 0; i < urls.length; i++) {
    try {
      await load(urls[i]);
      return i;
    } catch (e) {
      err = e;
      if (i === 0) console.info(`[dependências] sem cópia local (${urls[0]}); usando a CDN`);
    }
  }
  throw err;
}
function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.crossOrigin = "anonymous";
    s.onload = resolve;
    s.onerror = () => {
      s.remove();
      reject(new Error(`não carregou: ${src}`));
    };
    document.head.appendChild(s);
  });
}
async function loadVision() {
  try {
    return await firstAvailable(MP_BUNDLE, async (url) => {
      vision = await import(url);
    });
  } catch (e) {
    throw new Error(`MediaPipe não carregou de ./vendor/ nem da CDN (${e.message}). Sem internet, rode no computador: bash tools/baixar_dependencias_app.sh`);
  }
}
async function loadTfjs() {
  const src = [];
  for (const urls of TF_SCRIPTS) src.push(await firstAvailable(urls, loadScript));
  S.tfliteWasm = TFLITE_WASM[src[2]]; // módulos WASM da mesma origem do tf-tflite.min.js
  S.deps.tfjs = src.every((i) => i === 0) ? "local" : src.every((i) => i === 1) ? "CDN" : "local + CDN";
  showDeps();
}
function showDeps() {
  const name = (d) => (d === "local" ? "local (./vendor/)" : d === "CDN" ? "CDN (internet)" : d);
  $("s-deps").textContent = `MediaPipe ${name(S.deps.mediapipe)} · TensorFlow.js ${name(S.deps.tfjs)}`;
}

// ---------------------------------------------------------------------------
// Carregamento dos modelos TFLite (tolerante a 404 -> modo degradado)
// ---------------------------------------------------------------------------
async function fetchJSON(url) {
  try {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) return null;
    return await r.json();
  } catch {
    return null;
  }
}
async function exists(url) {
  try {
    const r = await fetch(url, { method: "HEAD", cache: "no-store" });
    return r.ok;
  } catch {
    return false;
  }
}

/** Executa um modelo TFLite (ver tflite_io.js) com o tf global carregado. */
function runTflite(model, io, inputF32, shape) {
  return runTfliteRaw(S.tf, model, io, inputF32, shape);
}

async function loadTfliteModels() {
  try {
    await S.tfReady;
  } catch (e) {
    console.warn("[dependências] TensorFlow.js", e.message);
  }
  S.tf = window.tf;
  S.tflite = window.tflite;
  if (!S.tf || !S.tflite) {
    $("s-eye").textContent = "tfjs/tflite não carregou (nem de ./vendor/ nem da CDN)";
    $("s-temporal").textContent = "tfjs/tflite não carregou (nem de ./vendor/ nem da CDN)";
    appendNotice("Bibliotecas TensorFlow.js/tfjs-tflite não carregaram (nem de ./vendor/ nem da CDN). O detector (olhos, bocejo, cabeça) funciona; só os modelos experimentais ficam desligados.");
    return;
  }
  try {
    await S.tf.setBackend("cpu");
    await S.tf.ready();
  } catch (e) {
    console.warn("tf.setBackend cpu", e);
  }
  S.tflite.setWasmPath(S.tfliteWasm);

  // ---- eye_state ----
  S.eyeMeta = adaptEyeMeta(await fetchJSON(MODELS_DIR + "eye_state.json"));
  const eyeFiles = S.eyeMeta?.files || {};
  const candidates = [];
  if (eyeFiles.int8 && (S.eyeMeta?.int8?.input?.scale > 0 || S.eyeMeta?.quant?.input?.scale > 0)) candidates.push([eyeFiles.int8, "int8"]);
  if (eyeFiles.fp32) candidates.push([eyeFiles.fp32, "fp32"]);
  candidates.push(["eye_state_fp32.tflite", "fp32"]);
  const seen = new Set();
  for (const [file, key] of candidates) {
    if (seen.has(file)) continue;
    seen.add(file);
    const url = MODELS_DIR + file;
    if (!(await exists(url))) continue;
    try {
      S.eyeModel = await S.tflite.loadTFLiteModel(url);
      S.eyeIO = tfliteIO(S.eyeModel, S.eyeMeta, key);
      const ms = warmupEye();
      S.eyeFile = file;
      $("s-eye").textContent = `${file} (${key}; entrada ${S.eyeIO.inDtype} ${JSON.stringify(S.eyeIO.inShape)}; aquecimento ${ms.toFixed(1)} ms)`;
      console.log(`[modelos] eye_state carregado: ${file} (${key}) aquecimento ${ms.toFixed(1)} ms`);
      break;
    } catch (e) {
      console.warn("falha ao carregar", file, e);
      S.eyeModel = null;
      $("s-eye").textContent = `erro ao carregar ${file}: ${e.message}${/Aborted/.test(e.message) ? ". Provável batch dinâmico (ops SHAPE/STRIDED_SLICE/PACK): reexporte com batch fixo 1 (ver models/README.md)" : ""}`;
    }
  }
  if (!S.eyeModel) {
    $("s-eye").textContent = "ausente (404): coloque em app/pwa/models/";
    appendNotice("eye_state TFLite ausente em ./models/: CNN dos olhos desligada (o detector usa EAR + blendshapes e continua funcionando).");
  }

  // ---- blink_temporal ----
  S.tmpMeta = await fetchJSON(MODELS_DIR + "blink_temporal.json");
  const tmpFile = S.tmpMeta?.files?.fp32 || S.tmpMeta?.file || "blink_temporal.tflite";
  const tmpUrl = MODELS_DIR + tmpFile;
  if (await exists(tmpUrl)) {
    try {
      S.tmpModel = await S.tflite.loadTFLiteModel(tmpUrl);
      S.tmpIO = tfliteIO(S.tmpModel, S.tmpMeta, "int8");
      if (S.tmpMeta?.window) S.blinkWindow = new E.BlinkWindow({ size: S.tmpMeta.window, nFeatures: S.tmpMeta.n_features || 4, minBlinks: 10, padPartial: true });
      const ms = warmupTemporal();
      S.tmpFile = tmpFile;
      $("s-temporal").textContent = `${tmpFile} (entrada ${S.tmpIO.inDtype} ${JSON.stringify(S.tmpIO.inShape)}; aquecimento ${ms.toFixed(1)} ms)`;
      console.log(`[modelos] blink_temporal carregado: ${tmpFile} aquecimento ${ms.toFixed(1)} ms`);
    } catch (e) {
      S.tmpModel = null;
      $("s-temporal").textContent = `erro ao carregar ${tmpFile}: ${e.message}${/Aborted/.test(e.message) ? ". Provável batch dinâmico: reexporte com batch fixo 1 (ver models/README.md)" : ""}`;
    }
  }
  if (!S.tmpModel) {
    $("s-temporal").textContent = "ausente (404): coloque em app/pwa/models/";
    appendNotice("blink_temporal.tflite ausente em ./models/: estimativa do LSTM desligada (o detector continua funcionando).");
  }
}

/**
 * Inferências de aquecimento: validam formato de entrada/saída e medem a latência.
 * São feitas DUAS execuções e só a segunda é validada: verificado em Chrome headless
 * (20/09/2026) que, no tfjs-tflite 0.0.1-alpha.10, a PRIMEIRA inferência do
 * blink_temporal.tflite (LSTM com lote dinâmico: SHAPE/PACK/FILL para o estado inicial)
 * devolve [0, 0, 0]; a partir da segunda o resultado coincide com o TFLite do Python
 * (diferença < 1e-6). Lança erro se a saída for inválida.
 */
function warmupEye() {
  const size = S.eyeMeta?.input_size || EYE_SIZE_DEFAULT;
  const t = performance.now();
  runEyeModel(new Float32Array(size * size));
  const p = runEyeModel(new Float32Array(size * size));
  if (!Number.isFinite(p) || p < 0 || p > 1) throw new Error(`saída inválida no aquecimento (p=${p})`);
  return (performance.now() - t) / 2;
}
function warmupTemporal() {
  const n = S.blinkWindow.size * S.blinkWindow.nFeatures;
  const shape = [1, S.blinkWindow.size, S.blinkWindow.nFeatures];
  const t = performance.now();
  runTflite(S.tmpModel, S.tmpIO, new Float32Array(n), shape);
  const out = runTflite(S.tmpModel, S.tmpIO, new Float32Array(n), shape);
  if (out.length < 3) throw new Error(`saída com ${out.length} valores (esperado 3 classes)`);
  const sum = Array.from(out).reduce((a, b) => a + b, 0);
  if (!out.every(Number.isFinite) || sum < 1e-6) throw new Error(`saída inválida no aquecimento: [${Array.from(out).map((v) => v.toFixed(3))}]`);
  return (performance.now() - t) / 2;
}

// ---------------------------------------------------------------------------
// Face Landmarker (marcos + blendshapes + matriz de pose)
// ---------------------------------------------------------------------------
async function createLandmarker() {
  const src = await S.visionReady; // 0 = módulo de ./vendor/, 1 = CDN
  const { FaceLandmarker, FilesetResolver } = vision;
  // WASM e modelo: os locais primeiro (se o módulo veio de ./vendor/); a CDN fica como reserva.
  for (let i = src; i < MP_WASM.length; i++) {
    const files = await FilesetResolver.forVisionTasks(MP_WASM[i]);
    for (const delegate of ["GPU", "CPU"]) {
      try {
        const lm = await FaceLandmarker.createFromOptions(files, {
          baseOptions: { modelAssetPath: MP_MODEL[i], delegate },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
        });
        S.delegate = delegate;
        S.deps.mediapipe = i === 0 ? "local" : src === 0 ? "local + CDN" : "CDN";
        showDeps();
        $("m-delegate").textContent = delegate;
        $("s-landmarker").textContent = `face_landmarker float16 (${delegate}; blendshapes + matriz de pose)`;
        return lm;
      } catch (e) {
        console.warn(`Landmarker com delegate ${delegate} falhou`, e);
        $("s-landmarker").textContent = `delegate ${delegate} falhou: ${e?.message || e}`;
      }
    }
  }
  throw new Error("Não foi possível criar o Face Landmarker (GPU nem CPU).");
}

// ---------------------------------------------------------------------------
// Câmera
// ---------------------------------------------------------------------------
/** Desliga a câmera (todas as trilhas) e solta o <video>. */
function stopStream() {
  if (S.stream) for (const t of S.stream.getTracks()) t.stop();
  S.stream = null;
  video.srcObject = null;
}
async function startCamera() {
  const btn = $("btn-start");
  if (S.running) return;
  // "Cancelar" (ou "Encerrar") durante o carregamento troca a ficha: depois de cada espera, uma ficha velha
  // desiste em silêncio, em vez de ligar a câmera de uma viagem que já foi cancelada.
  const ficha = (S.fichaInicio = (S.fichaInicio || 0) + 1);
  const cancelado = () => ficha !== S.fichaInicio;
  btn.disabled = true;
  btn.textContent = "Carregando...";
  setBadge("carregando modelos", "warn");
  setCalibCarregando();
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("getUserMedia indisponível (a página precisa de HTTPS ou localhost).");
    stopStream(); // um novo toque depois de uma falha não deixa um segundo stream aberto
    S.stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    if (cancelado()) {
      stopStream();
      btn.disabled = false;
      btn.textContent = "Iniciar viagem";
      return;
    }
    video.srcObject = S.stream;
    placeCamera(); // o vídeo aparece antes do play(): o iOS não toca vídeo escondido
    await new Promise((res) => (video.onloadedmetadata = res));
    await video.play();
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    if (!S.landmarker) S.landmarker = await createLandmarker();
    if (!S.tflite && !S.eyeModel) await loadTfliteModels();
    if (cancelado()) {
      stopStream();
      placeCamera();
      btn.disabled = false;
      btn.textContent = "Iniciar viagem";
      return;
    }
    S.running = true;
    btn.textContent = "Viagem em andamento";
    $("btn-calib").disabled = false;
    $("btn-bench").disabled = false;
    $("btn-export").disabled = false;
    document.body.classList.add("rodando");
    $("previa-vazia").classList.add("hidden");
    placeCamera();
    setBadge(`${S.delegate} · FPS --`, "ok");
    console.log(`[app] câmera ${video.videoWidth}x${video.videoHeight}, landmarker delegate=${S.delegate}, eye_state=${S.eyeFile || "ausente"}, temporal=${S.tmpFile || "ausente"}, dependências mediapipe=${S.deps.mediapipe} tfjs=${S.deps.tfjs}`);
    S.wasmSimd = await detectSimd();
    showDeviceInfo({ delegate: S.delegate, video: `${video.videoWidth}x${video.videoHeight}`, wasmSimd: S.wasmSimd });
    requestWakeLock();
    startCalibration();
    requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    S.running = false;
    stopStream(); // a câmera não fica ligada se o detector não subiu
    S.viagem = null;
    document.body.classList.remove("rodando");
    placeCamera();
    notice(`Erro ao iniciar: ${e.message}`, "error");
    setBadge("erro", "danger");
    btn.disabled = false;
    btn.textContent = "Iniciar viagem";
    ir("inicio", { historico: false });
  }
}

// ---------------------------------------------------------------------------
// Calibração: fase 1 (10 s olhando para a frente: limiar do EAR, pose, boca de repouso) ->
// fase 2 em segundo plano (20 piscadas: duração de base e z-score do LSTM). O detector já funciona
// durante a fase 2.
// ---------------------------------------------------------------------------
function startCalibration() {
  S.monitor.startCalibration(performance.now());
  S.normalizer.reset();
  S.blinkWindow.reset();
  S.smoother.reset();
  S.temporal = null;
  S.detector = null;
  S.faceLostSince = null;
  resetCalibUI();
  $("blink-calib").classList.add("hidden");
  $("m-calib").textContent = "fase 1: 10 s olhando para a frente";
  addLog({ id: `c${Date.now()}`, type: "info", text: "calibração iniciada" });
  if (UI.tela !== "calibracao") ir("calibracao");
}

// ---------------------------------------------------------------------------
// Recorte dos olhos -> CNN eye_state (informativo)
// ---------------------------------------------------------------------------
function eyeCropGray(pts, w, h, size) {
  const box = E.eyeCropBox(pts, w, h, 1.6);
  cropCanvas.width = size;
  cropCanvas.height = size;
  cctx.drawImage(video, box.x, box.y, box.side, box.side, 0, 0, size, size);
  const img = cctx.getImageData(0, 0, size, size).data;
  const gray = new Float32Array(size * size);
  for (let i = 0, j = 0; i < img.length; i += 4, j++) gray[j] = 0.299 * img[i] + 0.587 * img[i + 1] + 0.114 * img[i + 2];
  return { gray, box };
}
function drawCropPreview(ctx, gray, size) {
  const im = ctx.createImageData(size, size);
  for (let j = 0; j < gray.length; j++) {
    const v = gray[j];
    im.data[j * 4] = v;
    im.data[j * 4 + 1] = v;
    im.data[j * 4 + 2] = v;
    im.data[j * 4 + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}
function probOpenFromOutput(out) {
  const meta = S.eyeMeta || {};
  const order = meta.class_order || ["closed", "open"];
  const openIdx = Math.max(0, order.indexOf("open"));
  if (out.length === 1) {
    const v = out[0];
    const isLogit = meta.output === "logit_open" || v < 0 || v > 1;
    const p = isLogit ? 1 / (1 + Math.exp(-v)) : v;
    return meta.output === "prob_closed" ? 1 - p : p;
  }
  let probs = out;
  const sum = Array.from(out).reduce((a, b) => a + b, 0);
  if (meta.output === "logits" || Math.abs(sum - 1) > 0.05 || out.some((v) => v < 0)) probs = E.softmax(out);
  return probs[openIdx];
}
function runEyeModel(gray) {
  const size = S.eyeMeta?.input_size || EYE_SIZE_DEFAULT;
  const x = E.normalizeEyeInput(gray, S.eyeMeta || {});
  const shape = S.eyeIO.inShape && S.eyeIO.inShape.length === 4 ? [1, size, size, S.eyeIO.inShape[3] || 1] : [1, size, size, 1];
  let input = x;
  if (shape[3] === 3) {
    input = new Float32Array(x.length * 3);
    for (let i = 0; i < x.length; i++) input[i * 3] = input[i * 3 + 1] = input[i * 3 + 2] = x[i];
  }
  const out = runTflite(S.eyeModel, S.eyeIO, input, shape);
  return probOpenFromOutput(out);
}

// ---------------------------------------------------------------------------
// LSTM de piscadas (informativo; conselheiro de no máximo +10 pontos)
// ---------------------------------------------------------------------------
function runTemporal(tensor, nBlinks) {
  const size = S.blinkWindow.size;
  const nf = S.blinkWindow.nFeatures;
  const out = runTflite(S.tmpModel, S.tmpIO, tensor, [1, size, nf]);
  let probs = out;
  const sum = Array.from(out).reduce((a, b) => a + b, 0);
  if (S.tmpMeta?.output === "logits" || Math.abs(sum - 1) > 0.05 || out.some((v) => v < 0)) probs = E.softmax(out);
  const sm = S.smoother.update(Array.from(probs));
  S.temporal = { raw: Array.from(probs), probs: sm.probs, cls: sm.cls, vote: sm.vote, partial: nBlinks < size, nBlinks };
  S.predLog.push({ t: performance.now(), nBlinks, raw: S.temporal.raw, cls: sm.cls, vote: sm.vote, partial: S.temporal.partial });
  return S.temporal;
}

// ---------------------------------------------------------------------------
// Loop principal
// ---------------------------------------------------------------------------
function loop() {
  if (!S.running) return;
  if (video.readyState >= 2 && video.currentTime !== S.lastVideoTime) {
    S.lastVideoTime = video.currentTime;
    try {
      processFrame();
    } catch (e) {
      console.error("processFrame", e);
    }
  }
  requestAnimationFrame(loop);
}

function processFrame() {
  const t0 = performance.now();
  const w = overlay.width;
  const h = overlay.height;
  let res;
  try {
    res = S.landmarker.detectForVideo(video, t0);
  } catch (e) {
    console.warn("detectForVideo", e);
    return;
  }
  const t1 = performance.now();
  S.lat.lm.push(t1 - t0);
  S.frameCount += 1;

  // FPS (janela de 1 s)
  S.fpsWindow.push(t0);
  while (S.fpsWindow.length && t0 - S.fpsWindow[0] > 1000) S.fpsWindow.shift();
  S.fps = S.fpsWindow.length;

  octx.clearRect(0, 0, w, h);
  let eyeMs = 0;
  let tmpMs = 0;
  let blinkNow = null;
  let mlProbs = null;
  let lp = null;
  let rp = null;
  const lms = res?.faceLandmarks?.[0];
  S.faceDetected = !!lms;
  S.blend = null;
  S.pose = null;

  if (lms) {
    lp = E.pickPoints(lms, E.LEFT_EYE, w, h);
    rp = E.pickPoints(lms, E.RIGHT_EYE, w, h);
    S.ear = { left: E.eyeAspectRatio(lp), right: E.eyeAspectRatio(rp), mean: 0 };
    S.ear.mean = (S.ear.left + S.ear.right) / 2;
    S.mar = E.marFromLandmarks(lms, w, h);
    const cats = res.faceBlendshapes?.[0]?.categories;
    if (cats) {
      const bm = E.blendshapeMap(cats);
      S.blend = {};
      for (const k of BLEND_KEYS) S.blend[k] = bm[k];
    }
    const mat = res.facialTransformationMatrixes?.[0];
    if (mat?.data) S.pose = E.poseFromMatrix(mat.data);

    // Caminho do LSTM: detector de piscadas no EAR com o limiar calibrado pelo monitor
    if (S.monitor.calibrated && !S.detector) {
      S.detector = new E.BlinkDetector({ threshold: S.monitor.baseline.earThr, minClosedFrames: 2, maxClosedMs: 1500 });
      S.normalizer.start();
    }
    if (S.detector) {
      blinkNow = S.detector.update(S.ear.mean, t0);
      if (blinkNow) {
        S.lastBlink = blinkNow;
        const z = S.normalizer.zscore(blinkNow.features);
        S.blinkLog.push({ t: blinkNow.t, start: blinkNow.start, bottom: blinkNow.bottom, end: blinkNow.end, closedMs: blinkNow.closedMs, raw: blinkNow.features, z, calib: S.normalizer.collecting });
        if (S.normalizer.collecting) {
          if (S.normalizer.add(blinkNow.features)) for (const s of S.normalizer.samples) S.blinkWindow.push(S.normalizer.zscore(s));
        } else if (S.normalizer.ready) {
          S.blinkWindow.push(z);
          if (S.tmpModel && S.blinkWindow.ready) {
            const ta = performance.now();
            try {
              mlProbs = runTemporal(S.blinkWindow.tensor(), S.blinkWindow.length).probs;
            } catch (e) {
              console.warn("LSTM", e);
            }
            tmpMs = performance.now() - ta;
            S.lat.tmp.push(tmpMs);
          }
        }
      }
    }

    // CNN dos olhos (informativa, a cada quadro)
    if (S.eyeModel) {
      const size = S.eyeMeta?.input_size || EYE_SIZE_DEFAULT;
      const te = performance.now();
      try {
        const L = eyeCropGray(lp, w, h, size);
        const R = eyeCropGray(rp, w, h, size);
        S.eyeProb.left = runEyeModel(L.gray);
        S.eyeProb.right = runEyeModel(R.gray);
        eyeMs = performance.now() - te;
        S.lat.eye.push(eyeMs);
        if (S.frameCount % 5 === 0) {
          drawCropPreview(cropLeft, L.gray, size);
          drawCropPreview(cropRight, R.gray, size);
        }
      } catch (e) {
        console.warn("CNN olhos", e);
        S.eyeModel = null;
        $("s-eye").textContent = `desativado após erro: ${e.message}`;
      }
    }

    // Benchmark: força o LSTM a cada quadro para medir a latência (com a janela atual ou zeros)
    if (S.bench && S.tmpModel && !blinkNow) {
      const ta = performance.now();
      const tensor = S.blinkWindow.tensor() || new Float32Array(S.blinkWindow.size * S.blinkWindow.nFeatures);
      try {
        runTflite(S.tmpModel, S.tmpIO, tensor, [1, S.blinkWindow.size, S.blinkWindow.nFeatures]);
        tmpMs = performance.now() - ta;
        S.lat.tmp.push(tmpMs);
      } catch (e) {
        console.warn("temporal (benchmark)", e);
      }
    }
  }

  // ---- detector v2 ----
  const tm = performance.now();
  const r = S.monitor.update(t0, { face: !!lms, ear: S.ear, mar: S.mar, blend: S.blend, pose: S.pose, mlProbs });
  const monMs = performance.now() - tm;
  S.lat.mon.push(monMs);
  S.last = r;
  S.faceLostSince = lms ? null : S.faceLostSince ?? t0;
  handleEvents(r);
  applyAlarm(r.alarm);

  if (lms) {
    const eyeColor = r.eyes?.closed ? "#ff4d4f" : r.eyes?.gated && r.eyes?.byEar ? "#f5b041" : "#2ecc71";
    drawEye(lp, eyeColor);
    drawEye(rp, eyeColor);
    drawMouth(lms, w, h, r.yawn?.open);
    drawHeadAxis(lms, w, h, r.head);
  }

  const tot = performance.now() - t0;
  S.lat.tot.push(tot);

  if (S.bench) benchRow(t0, t1, eyeMs, tmpMs, monMs, tot, blinkNow, r);
  registrarViagem(r, t0); // resumo da viagem: fora do tempo medido por quadro

  if (t0 - S.lastUi > UI_INTERVAL_MS) {
    S.lastUi = t0;
    updateUI(r);
  }
}

// ---------------------------------------------------------------------------
// Registro de eventos
// ---------------------------------------------------------------------------
const EV_KIND = { microssono: "danger", alarme: "danger", bocejo: "warn", queda: "warn", cabeceio: "warn", fechamento: "warn", aviso: "warn" };
/** Direção da cabeça no evento: tombando para o lado (roll) ou caindo para a frente (pitch). */
function headDir(e) {
  return e.byRoll ? `${br(e.peakRollDeg, 0)}° para o lado` : `${br(e.peakDeg, 0)}° para baixo`;
}
function eventText(e, upd = null) {
  switch (e.type) {
    case "bocejo":
      return upd ? `bocejo (${br(upd.durMs / 1000, 1)} s)` : "bocejo";
    case "queda":
      return `cabeça ${e.byRoll ? "tombando" : "caindo"} com os olhos fechados (${headDir(e)}${e.lost ? ", saiu do quadro" : ""})`;
    case "cabeca_baixa":
      return `cabeça ${e.byRoll ? "inclinada" : "baixa"} com os olhos abertos (${headDir(e)}): distração, 0 ponto`;
    case "cabeceio":
      return `cabeceio (${br(e.peakDeg, 0)}°, ${br(e.durMs / 1000, 1)} s)`;
    case "fechamento":
      return `olhos fechados ${br(e.durMs / 1000, 1)} s`;
    case "microssono":
      return upd ? `microssono ${br(upd.durMs / 1000, 1)} s` : "microssono (olhos fechados >= 1 s)";
    case "alarme":
      return `ALARME: ${e.motivo}`;
    case "aviso":
      return "aviso: ATENÇÃO";
    case "calibrado":
      return `calibração concluída (limiar EAR ${br(e.baseline.earThr, 3)}${e.baseline.pitch0 !== null ? `, cabeça ${br(e.baseline.pitch0, 0)}°` : ""})`;
    case "base_piscadas":
      return `linha de base das piscadas: ${Math.round(e.baseMs)} ms`;
    default:
      return e.label || e.type;
  }
}
function addLog(entry) {
  const item = { ...entry, wall: entry.wall || wallClock(), kind: entry.kind || EV_KIND[entry.type] || "info" };
  S.eventLog.push(item);
  const ol = $("event-log");
  const empty = ol.querySelector(".muted");
  if (empty) empty.remove();
  const li = document.createElement("li");
  li.id = `ev-${item.id}`;
  li.className = `linha ev-${item.kind}`;
  const tm = document.createElement("span");
  tm.className = "time";
  tm.textContent = item.wall;
  const tx = document.createElement("span");
  tx.className = "txt";
  tx.textContent = item.text;
  li.append(tm, tx);
  ol.prepend(li);
  while (ol.children.length > MAX_LOG_ITEMS) ol.lastElementChild.remove();
}
function handleEvents(r) {
  for (const e of r.events) {
    addLog({ id: e.id, type: e.type, text: eventText(e), ev: e });
    if (e.type === "calibrado") {
      $("blink-calib").classList.remove("hidden");
      const b = e.baseline;
      $("m-calib").textContent = `fase 2: piscadas (limiar EAR ${fmt(b.earThr)}, pitch ${fmt(b.pitch0, 1)}°, jawOpen ${fmt(b.jaw0, 2)})`;
      $("m-thresh").textContent = `${fmt(b.earThr)} (mediana ${fmt(b.earMedian)})`;
      if (b.warnings.length) addLog({ id: `w${e.id}`, type: "info", kind: "warn", text: `calibração com aviso: ${b.warnings.join(" ")}` });
      mostrarCalibPronta(b.warnings);
    }
    if (e.type === "base_piscadas") {
      $("blink-calib").classList.add("hidden");
      $("m-calib").textContent = `calibrado (piscada típica ${Math.round(e.baseMs)} ms)`;
    }
  }
  for (const u of r.updates) {
    const item = S.eventLog.find((x) => x.id === u.id);
    if (!item) continue;
    item.text = eventText(item.ev, u);
    item.durMs = u.durMs;
    const li = document.getElementById(`ev-${u.id}`);
    if (li) li.querySelector(".txt").textContent = item.text;
  }
}

// ---------------------------------------------------------------------------
// Desenho do overlay
// ---------------------------------------------------------------------------
function drawEye(pts, color) {
  octx.strokeStyle = color;
  octx.lineWidth = 2;
  octx.beginPath();
  octx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) octx.lineTo(pts[i][0], pts[i][1]);
  octx.closePath();
  octx.stroke();
}
function drawMouth(lms, w, h, open) {
  const [c0, c1] = E.pickPoints(lms, E.MOUTH_CORNERS, w, h);
  const [v0, v1] = E.pickPoints(lms, E.MOUTH_VERTICAL, w, h);
  octx.strokeStyle = open ? "#f5b041" : "#9fb0d0";
  octx.lineWidth = open ? 3 : 2;
  octx.beginPath();
  octx.moveTo(c0[0], c0[1]);
  octx.lineTo(v0[0], v0[1]);
  octx.lineTo(c1[0], c1[1]);
  octx.lineTo(v1[0], v1[1]);
  octx.closePath();
  octx.stroke();
}
/** Linha da testa (10) ao queixo (152): cor pelo estado da cabeça. */
function drawHeadAxis(lms, w, h, head) {
  const [a, b] = E.pickPoints(lms, [10, 152], w, h);
  const st = head?.state;
  octx.strokeStyle = st === "caida" ? "#ff4d4f" : st === "baixando" ? "#f5b041" : "rgba(159,176,208,.7)";
  octx.lineWidth = st === "caida" ? 4 : 2;
  octx.setLineDash([6, 5]);
  octx.beginPath();
  octx.moveTo(a[0], a[1]);
  octx.lineTo(b[0], b[1]);
  octx.stroke();
  octx.setLineDash([]);
}

// ---------------------------------------------------------------------------
// Interface: telas (Início, Calibração, Em viagem, Viagens, Ajustes, Como funciona, Detalhes técnicos),
// tema claro/escuro, alarme em tela cheia e resumo da viagem. A tela só lê o resultado do monitor; nada
// aqui muda o detector, e a atualização da tela fica fora do tempo medido por quadro (benchmark).
// ---------------------------------------------------------------------------
const UI = { tela: "inicio", historico: [], autoAvanco: 0, ultimoGrafico: 0, alarmeVisivel: false, cameraAgendada: false };
const STATE_TITLES = ["Alerta", "Atenção", "Sonolento"];
const WHY_SHORT = { perclos: "PERCLOS", fechamentos: "olhos fechados", bocejos: "bocejos", cabeca: "cabeça", piscadas: "piscadas lentas", modelo: "modelo LSTM", pos_evento: "após microssono/cabeça" };
const HEAD_TXT = { neutra: "Erguida", baixando: "Baixando", caida: "Caída", lado: "Olhando para o lado", sem_rosto: "Sem rosto", sem_base: "Sem pose" };
const TEMA_KEY = "sonolencia.tema";
const RESUMO_KEY = "sonolencia.ultimaViagem";

function cap(s) {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}
function setText(id, txt) {
  const el = $(id);
  if (el && el.textContent !== txt) el.textContent = txt;
}
function horaCurta(d) {
  return d.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit", hour12: false });
}
function diaRelativo(d) {
  const hoje = new Date();
  const ontem = new Date(hoje);
  ontem.setDate(hoje.getDate() - 1);
  if (d.toDateString() === hoje.toDateString()) return "Hoje";
  if (d.toDateString() === ontem.toDateString()) return "Ontem";
  return d.toLocaleDateString("pt-BR", { day: "2-digit", month: "2-digit" });
}
function duracaoTexto(seg) {
  const min = Math.max(0, Math.round(seg / 60));
  return min < 60 ? `${min} min` : `${Math.floor(min / 60)} h ${String(min % 60).padStart(2, "0")} min`;
}
function plural(n, um, varios) {
  return `${n} ${n === 1 ? um : varios}`;
}
/** Rótulo dos pontos em linguagem simples (o do motor é técnico: "microssono(s)", ">= 0,5 s"). */
function rotulo(c) {
  if (c.key === "perclos") {
    const m = c.label.match(/PERCLOS ([\d,]+) %/);
    return m ? `PERCLOS de ${m[1]} % no último minuto` : "PERCLOS no último minuto";
  }
  if (c.key === "modelo") return "Modelo LSTM (experimental)";
  let t = c.label;
  if (c.key === "piscadas") t = t.replace(/\s*\([^)]*\)\s*$/, "");
  t = t
    .replace(/(\d+) ([a-zçãõáéíóú]+)\(s\)\/([a-zçãõáéíóú]+)\(s\)/gi, (_, n, a, b) => (Number(n) === 1 ? `${n} ${a} ou ${b}` : `${n} ${a}s ou ${b}s`))
    .replace(/(\d+) ([a-zçãõáéíóú]+)\(s\)/gi, (_, n, a) => `${n} ${Number(n) === 1 ? a : `${a}s`}`)
    .replace(/(\d),(\d)-(\d) s/g, "$1,$2 a $3 s")
    .replace(/ATENÇÃO/g, "Atenção")
    .replace(/>= /g, "de ");
  return cap(t);
}
function duracaoCurta(seg) {
  return seg < 60 ? `${Math.round(seg)} s` : `${Math.round(seg / 60)} min`;
}

// ---- navegação ----
function telaDaViagem() {
  return S.monitor.calibrated && S.last?.phase !== "calibrando" ? "viagem" : "calibracao";
}
function ir(nome, { historico = true } = {}) {
  const nova = $(`tela-${nome}`);
  if (!nova) return;
  if (nome !== UI.tela) {
    const atual = $(`tela-${UI.tela}`);
    if (historico) {
      UI.historico.push(UI.tela);
      if (UI.historico.length > 12) UI.historico.shift();
    }
    if (atual) atual.hidden = true;
    nova.hidden = false;
    nova.scrollTop = 0;
    UI.tela = nome;
  }
  const aba = nova.dataset.abas || "";
  document.body.classList.toggle("com-abas", !!aba);
  for (const b of document.querySelectorAll("#abas [data-aba]")) b.setAttribute("aria-current", b.dataset.aba === aba ? "page" : "false");
  if (nome === "tecnico" && S.last) updateTech(S.last);
  placeCamera();
}
function voltar() {
  let prev = UI.historico.pop();
  while (prev && prev === UI.tela) prev = UI.historico.pop();
  if ((prev === "calibracao" || prev === "viagem") && !S.running) prev = "inicio";
  ir(prev || (S.running ? telaDaViagem() : "inicio"), { historico: false });
}
function irAba(aba) {
  if (aba === "dirigir") ir(S.running ? telaDaViagem() : "inicio");
  else if (aba === "viagens") ir("resumo");
  else ir("ajustes");
}

// ---- câmera: o mesmo <video> fica sobre o encaixe da tela atual (cheio, cartão ou círculo) ----
function placeCamera() {
  const cam = $("camera");
  if (!S.stream) {
    cam.classList.add("hidden");
    return;
  }
  const tela = $(`tela-${UI.tela}`);
  const slot = tela ? [...tela.querySelectorAll("[data-camera]")].find((e) => e.getClientRects().length > 0) : null;
  cam.classList.remove("hidden");
  cam.dataset.forma = slot ? slot.dataset.camera : "flutuante";
  if (slot) {
    const r = slot.getBoundingClientRect();
    cam.style.left = `${r.left}px`;
    cam.style.top = `${r.top}px`;
    cam.style.width = `${r.width}px`;
    cam.style.height = `${r.height}px`;
    cam.style.right = "auto";
  } else {
    for (const k of ["left", "top", "width", "height", "right"]) cam.style[k] = "";
  }
}
function agendarCamera() {
  if (UI.cameraAgendada) return;
  UI.cameraAgendada = true;
  requestAnimationFrame(() => {
    UI.cameraAgendada = false;
    placeCamera();
  });
}

// ---- tema (Ajustes > Aparência) ----
function temaEscuroAtivo() {
  const t = document.documentElement.dataset.tema;
  return t === "escuro" || (!t && window.matchMedia("(prefers-color-scheme: dark)").matches);
}
function atualizarCorTema() {
  const m = document.querySelector('meta[name="theme-color"]');
  if (m) m.content = temaEscuroAtivo() ? "#000000" : "#F2F2F7";
}
function aplicarTema(t, salvar = true) {
  const valor = t === "claro" || t === "escuro" ? t : "auto";
  if (valor === "auto") delete document.documentElement.dataset.tema;
  else document.documentElement.dataset.tema = valor;
  if (salvar) {
    try {
      window.localStorage.setItem(TEMA_KEY, valor);
    } catch {
      /* modo privado: vale só nesta visita */
    }
  }
  for (const b of document.querySelectorAll(".segmentado [data-tema]")) b.setAttribute("aria-checked", String(b.dataset.tema === valor));
  atualizarCorTema();
}
function initTema() {
  let t = "auto";
  try {
    t = window.localStorage.getItem(TEMA_KEY) || "auto";
  } catch {
    /* modo privado */
  }
  aplicarTema(t, false);
}

// ---- diálogos ----
function abrirDialogo(id) {
  const d = $(id);
  if (typeof d.showModal === "function") {
    if (!d.open) d.showModal();
  } else d.setAttribute("open", "");
}
function fecharDialogo(d) {
  if (!d) return;
  if (typeof d.close === "function") {
    if (d.open) d.close();
  } else d.removeAttribute("open");
}
function abrirSom() {
  abrirDialogo("folha-som");
}

// ---- viagem: começa em "Iniciar viagem" e termina em "Encerrar" ----
function novaViagem() {
  return { inicio: new Date(), ultimoT: null, ativoMs: 0, tempo: [0, 0, 0], serie: [], prox: 0, nivelMax: 0, estado: 0, n: { alarme: 0, aviso: 0, bocejo: 0, fechamento: 0, microssono: 0, queda: 0, cabeceio: 0 }, momentos: [] };
}
function iniciarViagem() {
  if (S.running) {
    ir(telaDaViagem());
    return Promise.resolve();
  }
  if (!S.viagem) S.viagem = novaViagem();
  resetCalibUI();
  setCalibCarregando();
  ir("calibracao");
  return startCamera();
}
function pararCamera() {
  if (S.bench) finishBenchmark();
  S.running = false;
  stopStream();
  S.sirenWanted = false;
  Sound.stopSiren();
  document.body.classList.remove("flashing", "rodando");
  $("alarme").classList.add("hidden");
  UI.alarmeVisivel = false;
  clearTimeout(UI.autoAvanco);
  if (S.wakeLock) {
    S.wakeWanted = false;
    S.wakeLock.release().catch(() => {});
  }
  $("btn-start").disabled = false;
  $("btn-start").textContent = "Iniciar viagem";
  $("btn-calib").disabled = true;
  $("btn-bench").disabled = true;
  $("previa-vazia").classList.remove("hidden");
  placeCamera();
}
function cancelarViagem() {
  S.fichaInicio = (S.fichaInicio || 0) + 1; // um carregamento em andamento desiste
  S.viagem = null;
  pararCamera();
  addLog({ id: `x${Date.now()}`, type: "info", text: "viagem cancelada na calibração" });
  UI.historico = [];
  ir("inicio", { historico: false });
}
function encerrarViagem() {
  fecharDialogo($("dlg-encerrar"));
  const resumo = fecharResumo();
  S.viagem = null;
  pararCamera();
  addLog({ id: `f${Date.now()}`, type: "info", text: "viagem encerrada" });
  if (resumo) {
    salvarResumo(resumo);
    renderResumo(resumo);
    renderUltima(resumo);
  }
  UI.historico = [];
  ir("resumo", { historico: false });
}
/** Chamado a cada quadro, depois da medição de latência: tempo em cada estado, série do nível e momentos. */
function registrarViagem(r, t) {
  const v = S.viagem;
  if (!v) return;
  for (const e of r.events) if (e.type in v.n) v.n[e.type] += 1;
  const sc = r.score;
  if (r.phase !== "ativo" || !sc) {
    v.ultimoT = null; // calibração não conta como tempo de viagem monitorada
    return;
  }
  if (v.ultimoT !== null) {
    const dt = Math.min(1000, Math.max(0, t - v.ultimoT));
    v.ativoMs += dt;
    v.tempo[sc.state] += dt;
  }
  v.ultimoT = t;
  if (v.ativoMs >= v.prox) {
    v.serie.push([Math.round(v.ativoMs / 1000), Math.round(sc.level)]);
    v.prox += 5000;
  }
  if (sc.level > v.nivelMax) v.nivelMax = sc.level;
  if (sc.state !== v.estado) {
    registrarMomento(v, sc);
    v.estado = sc.state;
  }
}
function registrarMomento(v, sc) {
  const top = sc.contributions.slice(0, 2).map((c) => WHY_SHORT[c.key] || c.key).join(" e ");
  let texto;
  if (sc.state === 2) texto = sc.forced ? `Alarme: ${sc.forced}` : `Alarme: ${top || "sinais somados"}`;
  else if (sc.state === 1) texto = v.estado === 2 ? "Saiu do vermelho" : `Aviso: ${top || "sinais de cansaço"}`;
  else texto = "De volta ao verde";
  v.momentos.push({ hora: horaCurta(new Date()), s: Math.round(v.ativoMs / 1000), estado: sc.state, nivel: Math.round(sc.level), texto });
  if (v.momentos.length > 40) v.momentos.shift();
}
function fecharResumo() {
  const v = S.viagem;
  if (!v || v.ativoMs < 5000) return null;
  let serie = v.serie;
  if (serie.length > 240) {
    const passo = Math.ceil(serie.length / 240);
    serie = serie.filter((_, i) => i % passo === 0 || i === serie.length - 1);
  }
  return { v: 1, inicio: v.inicio.toISOString(), fim: new Date().toISOString(), ativoS: Math.round(v.ativoMs / 1000), tempoS: v.tempo.map((ms) => Math.round(ms / 1000)), serie, nivelMax: Math.round(v.nivelMax), n: { ...v.n }, momentos: v.momentos.slice(-20) };
}
function salvarResumo(res) {
  try {
    window.localStorage.setItem(RESUMO_KEY, JSON.stringify(res));
  } catch {
    /* sem armazenamento: o resumo fica só nesta visita */
  }
}
function carregarResumo() {
  try {
    const txt = window.localStorage.getItem(RESUMO_KEY);
    const r = txt ? JSON.parse(txt) : null;
    return r && r.v === 1 && Array.isArray(r.serie) && Array.isArray(r.tempoS) ? r : null;
  } catch {
    return null;
  }
}
function renderResumo(res) {
  $("resumo-vazio").classList.toggle("hidden", !!res);
  $("resumo-conteudo").classList.toggle("hidden", !res);
  if (!res) return;
  const ini = new Date(res.inicio);
  const fim = new Date(res.fim);
  setText("res-quando", `${diaRelativo(ini)}, ${horaCurta(ini)} a ${horaCurta(fim)}`);
  setText("res-duracao", duracaoTexto((fim - ini) / 1000));
  const maxS = Math.max(1, res.ativoS, ...res.serie.map((p) => p[0]));
  const X = (s) => ((s / maxS) * 322).toFixed(1);
  const Y = (n) => (104 - Math.max(0, Math.min(100, n))).toFixed(1);
  $("res-linha").setAttribute("points", res.serie.map(([s, n]) => `${X(s)},${Y(n)}`).join(" "));
  const g = $("res-pontos");
  g.textContent = "";
  for (const m of res.momentos) {
    if (m.estado === 0) continue;
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("cx", X(m.s));
    c.setAttribute("cy", Y(m.nivel ?? 50));
    c.setAttribute("r", "4.5");
    c.style.fill = m.estado === 2 ? "var(--ver)" : "#ffb224";
    g.appendChild(c);
  }
  setText("res-ini", horaCurta(ini));
  setText("res-fim", horaCurta(fim));
  const n = res.n;
  setText("res-alarmes", String(n.alarme));
  setText("res-avisos", String(n.aviso));
  $("res-alarmes").classList.toggle("bola-r", n.alarme > 0); // cor só quando houve alarme ou aviso
  $("res-avisos").classList.toggle("bola-a", n.aviso > 0);
  setText("res-bocejos", String(n.bocejo));
  setText("res-olhos", String(n.fechamento + n.microssono));
  setText("res-cabeca", String(n.queda + n.cabeceio));
  const total = res.tempoS.reduce((a, b) => a + b, 0) || 1;
  res.tempoS.forEach((s, i) => {
    const el = $(`res-b${i}`);
    el.style.width = `${(100 * s) / total}%`;
    el.style.display = s > 0 ? "" : "none";
    setText(`res-t${i}`, duracaoCurta(s));
  });
  const ol = $("res-momentos");
  ol.textContent = "";
  const momentos = res.momentos.length ? res.momentos : [{ hora: horaCurta(ini), estado: 0, texto: "Sem avisos nem alarmes" }];
  for (const m of momentos) {
    const li = document.createElement("li");
    li.className = "linha momento";
    const h = document.createElement("span");
    h.className = "hora";
    h.textContent = m.hora;
    const p = document.createElement("i");
    p.className = `ponto-cor ${["p-v", "p-a", "p-r"][m.estado]}`;
    const tx = document.createElement("span");
    tx.className = "texto";
    tx.textContent = m.texto;
    li.append(h, p, tx);
    ol.appendChild(li);
  }
  setText("res-nota", n.alarme ? "Houve alarme nesta viagem. Com sono, não continue: descanse antes de voltar a dirigir." : n.aviso ? "Houve sinais de cansaço nesta viagem. Em viagens longas, faça pausas." : "Nenhum aviso nesta viagem.");
}
function renderUltima(res) {
  $("ultima-bloco").classList.toggle("hidden", !res);
  if (!res) return;
  const ini = new Date(res.inicio);
  const n = res.n;
  const partes = [duracaoTexto((new Date(res.fim) - ini) / 1000), n.alarme ? `${n.alarme} ${n.alarme === 1 ? "alarme" : "alarmes"}` : "sem alarme"];
  if (n.aviso) partes.push(`${n.aviso} ${n.aviso === 1 ? "aviso" : "avisos"}`);
  setText("ultima-titulo", `${diaRelativo(ini)}, ${horaCurta(ini)}`);
  setText("ultima-detalhe", partes.join(" · "));
}

// ---- calibração (tela com a câmera em tela cheia) ----
function resetCalibUI() {
  clearTimeout(UI.autoAvanco);
  document.querySelector(".anel").classList.remove("pronto");
  $("calib-anel").setAttribute("stroke-dasharray", "0 100");
  $("calib-fill").style.width = "0%";
  setText("calib-passo", "Passo 1 de 2");
  setText("calib-seg", "");
  $("calib-progresso").classList.remove("hidden");
  $("calib-nota").classList.remove("hidden");
  $("calib-pronto").classList.add("hidden");
}
function setCalibCarregando() {
  setText("calib-title", "Preparando a câmera");
  setText("calib-text", "Baixando o detector de rosto. Na primeira vez, pode levar alguns segundos.");
}
function atualizarCalib(r) {
  const p = Math.max(0, Math.min(1, r.calib?.progress ?? 0));
  const total = E.CFG.calib.neutralMs / 1000;
  $("calib-anel").setAttribute("stroke-dasharray", `${(p * 100).toFixed(1)} 100`);
  $("calib-fill").style.width = `${Math.round(p * 100)}%`;
  setText("calib-seg", `${Math.floor(p * total)} de ${total} s`);
  if (S.faceDetected) {
    setText("calib-title", "Olhe para a frente");
    setText("calib-text", "Olhos abertos e boca fechada. Pode piscar normalmente.");
  } else {
    setText("calib-title", "Enquadre o rosto");
    setText("calib-text", "Rosto não encontrado. Mostre o rosto inteiro, com luz de frente.");
  }
}
function mostrarCalibPronta(avisos) {
  document.querySelector(".anel").classList.add("pronto");
  $("calib-anel").setAttribute("stroke-dasharray", "100 100");
  $("calib-fill").style.width = "100%";
  $("calib-progresso").classList.add("hidden");
  $("calib-nota").classList.add("hidden");
  $("calib-pronto").classList.remove("hidden");
  setText("calib-passo", "Passo 1 de 2 concluído");
  clearTimeout(UI.autoAvanco);
  if (avisos.length) {
    setText("calib-title", "Calibração com aviso");
    setText("calib-text", avisos.join(" "));
    return; // com aviso, a pessoa decide: recalibrar ou seguir
  }
  setText("calib-title", "Pronto");
  setText("calib-text", `No passo 2, já dirigindo, o app aprende como você pisca nas primeiras ${E.CFG.calib.nBlinks} piscadas.`);
  UI.autoAvanco = setTimeout(() => {
    if (UI.tela === "calibracao" && S.running) ir("viagem");
  }, 1500);
}

// ---- em viagem ----
function setStatus(estado, title, sub, level) {
  const tela = $("tela-viagem");
  const cls = estado === null ? "" : `e${estado}`;
  if (tela.dataset.e !== cls) {
    tela.classList.remove("e0", "e1", "e2");
    if (cls) tela.classList.add(cls);
    tela.dataset.e = cls;
  }
  setText("status-title", title);
  setText("status-sub", sub);
  // Arredonda para baixo: 69,6 aparece como 69 (Atenção), nunca como "70" numa tela amarela.
  setText("level-num", level === null ? "--" : String(Math.floor(level)));
  const lv = level === null ? 0 : Math.max(0, Math.min(100, level));
  const dash = `${lv.toFixed(1)} 200`; // vão maior que o arco: o traço não se repete no fim
  const arco = $("meter-fill");
  if (arco.getAttribute("stroke-dasharray") !== dash) arco.setAttribute("stroke-dasharray", dash);
  $("meter").setAttribute("aria-valuenow", String(Math.floor(lv)));
}
function setSig(id, cls, state, detail) {
  const el = $(id);
  const want = `linha sinal ${cls}`.trim();
  if (el.className !== want) el.className = want;
  const a = el.querySelector(".sig-state");
  const b = el.querySelector(".sig-detail");
  if (a.textContent !== state) a.textContent = state;
  if (b.textContent !== detail) b.textContent = detail;
}
function renderPorque(sc) {
  const mostra = !!sc.forced || sc.contributions.length > 0;
  $("porque").classList.toggle("hidden", !mostra);
  if (!mostra) return;
  setText("porque-titulo", `Por que ${Math.floor(sc.level)}`);
  const partes = [];
  if (sc.forced) partes.push(`<li class="linha forcado"><span class="texto">${esc(cap(sc.forced))}</span><span class="pontos">vermelho na hora</span></li>`);
  for (const c of sc.contributions) partes.push(`<li class="linha"><span class="texto">${esc(rotulo(c))}</span><span class="pontos">+${Math.round(c.points)}</span></li>`);
  const html = partes.join("");
  const ul = $("why-list");
  if (ul.dataset.html !== html) {
    ul.innerHTML = html;
    ul.dataset.html = html;
  }
}
function atualizarGrafico10() {
  const v = S.viagem;
  const linha = $("grafico-linha");
  if (!v || !v.serie.length) {
    linha.setAttribute("points", "");
    setText("grafico-nota", "O gráfico começa depois da calibração.");
    return;
  }
  const fimS = v.ativoMs / 1000;
  const iniS = Math.max(0, fimS - 600);
  const pts = v.serie.filter(([s]) => s >= iniS - 5);
  const X = (s) => Math.max(0, ((s - iniS) / 600) * 326).toFixed(1);
  const Y = (n) => (52 - 0.48 * Math.max(0, Math.min(100, n))).toFixed(1);
  linha.setAttribute("points", pts.map(([s, n]) => `${X(s)},${Y(n)}`).join(" "));
  const pico = pts.reduce((m, p) => Math.max(m, p[1]), 0);
  setText("grafico-nota", fimS < 60 ? "O gráfico enche conforme a viagem avança." : pico >= 1 ? `Pico de ${pico} nos últimos ${Math.min(10, Math.ceil(fimS / 60))} min.` : "Nível zero no período.");
}
function atualizarAlarme(r) {
  const sc = r.score;
  // Depois do "estou acordado", a tela volta para a viagem (ainda em vermelho, com o som silenciado); se o nível
  // seguir alto quando os 30 s acabarem, o alarme volta. O vermelho forçado (microssono, cabeça caída) sempre mostra.
  const on = S.running && r.phase === "ativo" && !!sc && sc.state === 2 && (!!sc.forced || !r.alarm?.snoozed);
  if (on !== UI.alarmeVisivel) {
    UI.alarmeVisivel = on;
    $("alarme").classList.toggle("hidden", !on);
    if (on) {
      fecharDialogo($("folha-som"));
      fecharDialogo($("dlg-encerrar"));
      // foco no alarme (leitor de tela): no "deslize" quando dá para silenciar, senão no título
      setTimeout(() => {
        const alvo = $("deslizar").classList.contains("hidden") ? $("alarme-titulo") : $("btn-ack");
        alvo.focus({ preventScroll: true });
      }, 0);
    }
  }
  if (!on) return;
  setText("alarme-titulo", sc.forced ? "Acorde!" : "Pare e descanse");
  setText("alarme-texto", sc.forced ? `${cap(sc.forced)}. Procure um lugar seguro para parar.` : `Sonolência alta: nível ${Math.floor(sc.level)} de 100. Procure um lugar seguro para parar.`);
  const html = sc.forced ? "" : sc.contributions.slice(0, 3).map((c) => `<li><span>${esc(rotulo(c))}</span><b>+${Math.round(c.points)}</b></li>`).join("");
  const ul = $("alarme-motivos");
  if (ul.dataset.html !== html) {
    ul.innerHTML = html;
    ul.dataset.html = html;
  }
  const silenciado = !!r.alarm?.snoozed;
  setText("alarme-som", silenciado ? "som silenciado" : S.volumePct === 0 ? "sem som: volume em 0 %" : !Sound.ready ? "toque na tela para liberar o som" : "alarme tocando");
  $("deslizar").classList.toggle("hidden", !!sc.forced || silenciado);
  setText("alarme-nota", sc.forced ? "O alarme para quando os olhos abrirem e a cabeça voltar." : silenciado ? "Som silenciado por 30 s. O alarme volta se os sinais continuarem." : "Silencia o som por 30 s e volta para a viagem. Num microssono, o alarme volta na hora.");
}
function estouAcordado() {
  Sound.unlock();
  S.monitor.acknowledge(performance.now());
  addLog({ id: `a${Date.now()}`, type: "info", text: "“Estou acordado”: som silenciado por 30 s" });
}
/** "Deslize: estou acordado": arrastar até o fim silencia; um toque sozinho não (evita silenciar sem querer).
 *  Pelo teclado (Enter ou espaço no botão), silencia direto. */
function setupDeslizar() {
  const trilho = $("deslizar");
  const knob = $("btn-ack");
  let x0 = null;
  let dx = 0;
  let max = 0;
  const soltar = () => {
    x0 = null;
    knob.style.transition = "";
    knob.style.transform = "";
  };
  knob.addEventListener("pointerdown", (ev) => {
    x0 = ev.clientX;
    dx = 0;
    max = trilho.clientWidth - knob.offsetWidth - 12;
    knob.style.transition = "none";
    try {
      knob.setPointerCapture(ev.pointerId);
    } catch {
      /* ignora */
    }
  });
  knob.addEventListener("pointermove", (ev) => {
    if (x0 === null) return;
    dx = Math.max(0, Math.min(max, ev.clientX - x0));
    knob.style.transform = `translateX(${dx}px)`;
  });
  knob.addEventListener("pointerup", () => {
    if (x0 === null) return;
    const chegou = max > 0 && dx >= max * 0.85;
    soltar();
    if (chegou) estouAcordado();
  });
  knob.addEventListener("pointercancel", soltar);
  knob.addEventListener("click", (ev) => {
    if (ev.detail === 0) estouAcordado(); // clique vindo do teclado
  });
}

function updateUI(r) {
  const face = S.faceDetected;
  setBadge(`${S.delegate} · ${S.fps} FPS`, S.fps >= 15 ? "ok" : "warn");
  setText("hud-face-tec", face ? "detectado" : "não detectado");
  setText("hud-face-texto", face ? "Rosto encontrado" : "Rosto não encontrado");
  $("hud-face").classList.toggle("achou", face);
  $("camera").style.setProperty("--cor-rosto", face ? "var(--verde)" : "var(--amb)");
  const noFaceMs = S.faceLostSince === null ? 0 : performance.now() - S.faceLostSince;
  if (S.viagem) setText("viagem-tempo", `${Math.floor((Date.now() - S.viagem.inicio) / 60000)} min · ${horaCurta(new Date())}`);

  // ---- estado e mostrador ----
  if (r.phase === "calibrando") {
    atualizarCalib(r);
    setStatus(null, "Calibrando", "Olhe para a frente por alguns segundos.", null);
    setText("status-why", "");
    $("pausa").classList.add("hidden");
    $("porque").classList.add("hidden");
  } else if (r.score) {
    const sc = r.score;
    let sub;
    if (sc.state === 2) sub = sc.forced ? `${cap(sc.forced)}. Acorde!` : "Sonolência alta: pare em local seguro e descanse.";
    else if (sc.state === 1) sub = sc.postForcedMs > 0 ? `Olhos fechados ou cabeça caída há pouco: atenção por mais ${Math.ceil(sc.postForcedMs / 1000)} s.` : "Sinais de cansaço. Faça uma pausa assim que possível.";
    else sub = sc.level >= 15 ? "Sinais leves registrados; seguem sendo acompanhados." : "Nenhum sinal relevante de sono.";
    if (r.alarm?.snoozed) sub += " Som silenciado por 30 s.";
    if (noFaceMs > 1500 && !sc.forced) sub = `Rosto não detectado há ${br(noFaceMs / 1000, 0)} s: enquadre o rosto.`;
    setStatus(sc.state, STATE_TITLES[sc.state], sub, sc.level);
    setText("status-why", sc.forced ? `Motivo: ${sc.forced}` : "");
    $("pausa").classList.toggle("hidden", sc.state < 1);
    renderPorque(sc);
  }

  // ---- passo 2 da calibração: piscadas de referência ----
  if (r.phase === "ativo" && r.calib) {
    $("blink-calib").classList.remove("hidden");
    setText("blink-calib-text", `${r.calib.blinks} de ${r.calib.need}`);
  }

  // ---- três sinais ----
  if (r.phase !== "ativo") {
    const txt = r.phase === "calibrando" ? "Calibrando" : "--";
    setSig("sig-eyes", "", txt, "PERCLOS --");
    setSig("sig-yawn", "", txt, "0 em 5 min");
    setSig("sig-head", "", txt, "0 quedas em 5 min");
  } else if (r.eyes) {
    const e = r.eyes;
    let cls = "s-ok";
    let st = "Abertos";
    if (!face) { cls = ""; st = "Sem rosto"; }
    else if (e.closedMs >= E.CFG.eyes.microsleepMs) { cls = "s-danger"; st = `Fechados ${br(e.closedMs / 1000, 1)} s`; }
    else if (e.closed) { cls = e.closedMs >= E.CFG.eyes.longMs ? "s-warn" : "s-ok"; st = e.closedMs >= 200 ? `Fechados ${br(e.closedMs / 1000, 1)} s` : "Piscando"; }
    else if (e.gated && e.byEar) st = "Olhar para baixo";
    if (cls === "s-ok" && (e.perclos >= 0.15 || e.longPerMin >= 2)) cls = "s-warn";
    setSig("sig-eyes", cls, st, `PERCLOS ${br(e.perclos * 100, 1)} %${e.microsleeps5min ? ` · ${plural(e.microsleeps5min, "microssono", "microssonos")}` : ""}`);
    const y = r.yawn;
    let ycls = y.count5min >= 2 ? "s-warn" : "s-ok";
    let yst = y.count5min ? `${y.count5min} em 5 min` : "Nenhum";
    if (!face) { ycls = ""; yst = "Sem rosto"; }
    else if (y.open) { ycls = "s-warn"; yst = `Boca aberta ${br(y.openMs / 1000, 1)} s`; }
    setSig("sig-yawn", ycls, yst, `boca aberta ${br(E.CFG.yawn.minMs / 1000, 1)} s ou mais`);
    const hd = r.head;
    let hcls = hd.count5min >= 2 ? "s-warn" : "s-ok";
    if (hd.state === "baixando") hcls = "s-warn";
    if (hd.state === "caida") hcls = "s-danger";
    if (hd.state === "sem_rosto" || hd.state === "sem_base") hcls = "";
    const hst = hd.state === "caida" ? `Caída ${br(hd.downMs / 1000, 1)} s` : HEAD_TXT[hd.state] || hd.state;
    setSig("sig-head", hcls, hst, `${plural(hd.count5min, "queda", "quedas")} em 5 min`);
  }

  atualizarAlarme(r);
  const agora = performance.now();
  if (agora - UI.ultimoGrafico > 2000) {
    UI.ultimoGrafico = agora;
    atualizarGrafico10();
  }

  // ---- modelos experimentais ----
  if (S.temporal) {
    const p = S.temporal.probs;
    const names = ["alerta", "pouca vigilância", "sonolento"];
    const tag = r.score?.mlCounted ? '<span class="tag">+10 no nível</span>' : r.score?.mlEligible ? '<span class="tag">"sonolento", mas não soma (sinais visíveis &lt; 40)</span>' : "";
    $("ml-lstm").innerHTML = `LSTM de piscadas: <b>${names[S.temporal.cls]}</b> (${names.map((n, i) => `${n} ${Math.round(p[i] * 100)} %`).join(" · ")}; ${S.temporal.nBlinks}/${S.blinkWindow.size} piscadas${S.temporal.partial ? ", janela parcial" : ""})${tag}`;
  } else {
    $("ml-lstm").textContent = S.tmpModel ? `LSTM de piscadas: aguardando piscadas (${S.normalizer.ready ? `${S.blinkWindow.length}/${S.blinkWindow.size}` : `calibração ${S.normalizer.samples.length}/${S.normalizer.nCalib}`})` : "LSTM de piscadas: modelo ausente.";
  }
  $("ml-cnn").textContent = S.eyeModel
    ? `CNN dos olhos, p(aberto): E ${br(S.eyeProb.left, 2)} · D ${br(S.eyeProb.right, 2)}`
    : "CNN dos olhos: modelo ausente.";

  // ---- detalhes técnicos (só com a tela aberta) ----
  if (UI.tela === "tecnico") updateTech(r);
}

// ---- Como funciona: textos montados a partir de E.CFG (os mesmos números que o detector usa) ----
function renderComoFunciona() {
  const C = E.CFG;
  const L = C.levels;
  const P = C.score;
  const seg = (ms) => `${br(ms / 1000, ms % 1000 ? 1 : 0)} s`;
  const min = (ms) => `${br(ms / 60000, 0)} min`;
  const pct = (x) => `${br(x * 100, 0)} %`;
  const linha = (titulo, texto, valor) => `<li class="linha"><span class="texto">${titulo}<small>${texto}</small></span><span class="valor">${valor}</span></li>`;
  const vermelho = (icone, titulo, texto) => `<li class="linha"><span class="quadrado vermelho" aria-hidden="true"><svg><use href="#${icone}"/></svg></span><span class="texto">${titulo}<small>${texto}</small></span></li>`;
  $("como-conteudo").innerHTML = `
    <p class="intro sec">O nível vai de 0 a 100. Cada sinal de sono soma pontos, e os pontos somem sozinhos com o tempo.</p>
    <h2 class="secao">Faixas</h2>
    <ul class="lista">
      <li class="linha"><i class="ponto-cor p-v"></i><span class="texto">Alerta</span><span class="sec num">0 a ${L.attention - 1}</span></li>
      <li class="linha"><i class="ponto-cor p-a"></i><span class="texto">Atenção</span><span class="sec num">${L.attention} a ${L.drowsy - 1}</span></li>
      <li class="linha"><i class="ponto-cor p-r"></i><span class="texto">Sonolento</span><span class="sec num">${L.drowsy} a 100</span></li>
    </ul>
    <p class="rodape">Para a cor não ficar trocando à toa, o amarelo só volta ao verde abaixo de ${L.attentionExit}, e o vermelho só sai abaixo de ${L.drowsyExit}.</p>
    <h2 class="secao">O que soma pontos</h2>
    <ul class="lista">
      ${linha(`Olhos fechados de ${seg(C.eyes.longMs)} a ${seg(C.eyes.microsleepMs)}`, `Cada vez. Some em ${min(P.longClosure.decayMs)}.`, `+${P.longClosure.points}`)}
      ${linha("PERCLOS", `Parte do último minuto com os olhos fechados, só contando fechamentos de ${seg(C.perclos.minEpisodeMs)} ou mais. ${P.perclos.points} pontos em ${pct(P.perclos.at)}.`, `até ${P.perclos.cap}`)}
      ${linha("Bocejo", `Boca bem aberta por ${seg(C.yawn.minMs)} ou mais. Some em ${min(P.yawn.decayMs)}.`, `+${P.yawn.points}`)}
      ${linha("Cabeça caindo ou cabeceio", `Com os olhos fechados. Some em ${min(P.head.decayMs)}.`, `+${P.head.points}`)}
      ${linha("Piscadas mais lentas", `De ${br(P.blinkDur.from, 2)} a ${br(P.blinkDur.at, 0)} vezes a duração da calibração.`, `até ${P.blinkDur.cap}`)}
      ${linha("Modelo LSTM, experimental", `Só soma se o nível já estiver em Atenção (${P.ml.minBase} ou mais).`, `+${P.ml.points}`)}
    </ul>
    <p class="rodape">Tetos: fechamentos ${P.closuresCap}, bocejos ${P.yawn.cap}, cabeça ${P.head.cap}. Nenhum sinal sozinho chega a ${L.drowsy}. Cabeça baixa de olhos abertos é distração e não soma.</p>
    <h2 class="secao">Vermelho na hora</h2>
    <ul class="lista">
      ${vermelho("i-microssono", "Microssono", `Olhos fechados por ${seg(C.eyes.microsleepMs)} ou mais.`)}
      ${vermelho("i-queda", "Cabeça caída", `Com os olhos fechados por ${seg(C.head.redEyesMs)}.`)}
    </ul>
    <p class="rodape">Depois, o app fica pelo menos ${min(P.postForced.holdMs)} em Atenção. "Estou acordado" silencia o som por ${seg(C.alarm.snoozeMs)}, mas nunca num microssono.</p>
    <h2 class="secao">De onde vêm os números</h2>
    <div class="cartao">
      <p>Da literatura (PERCLOS e microssono de 1 s), de testes no DMD (limiar do olho e bocejo de 1,5 s) e de regras de projeto (pontos, tetos, 40 e 70).</p>
      <p>Num teste com 300 combinações, cada peso multiplicado por 0,7 a 1,3 e os limiares mudados em até 10 pontos, o vermelho com motoristas acordados (direção real do DMD) ficou entre 1,7 % e 4,9 % do tempo em 90 % delas. Com os valores do app, 3,3 %.</p>
    </div>
    <ul class="lista espaco">
      <li><button type="button" class="linha toque" data-ir="tecnico"><span class="texto">Detalhes técnicos</span><span class="sec">latência e benchmark</span><svg class="seguir" aria-hidden="true"><use href="#i-seguir"/></svg></button></li>
    </ul>`;
}
window.__ui = { ir, abrirSom, encerrar: () => encerrarViagem(), estado: () => ({ tela: UI.tela, alarme: UI.alarmeVisivel, historico: [...UI.historico] }) }; // inspeção em testes automatizados
function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}
function updateTech(r) {
  const b = S.blend || {};
  $("m-ear").textContent = `${fmt(S.ear.left)} / ${fmt(S.ear.right)}`;
  $("m-blend").textContent = `${fmt(b.eyeBlinkLeft, 2)} / ${fmt(b.eyeBlinkRight, 2)}`;
  $("m-look").textContent = r.eyes ? fmt(r.eyes.lookDown, 2) : "--";
  $("m-mouth").textContent = `${fmt(S.mar, 2)} / ${fmt(b.jawOpen, 2)}`;
  $("m-pose").textContent = S.pose ? `${fmt(S.pose.pitch, 1)}° / ${fmt(S.pose.yaw, 1)}° / ${fmt(S.pose.roll, 1)}°` : "--";
  $("m-dpose").textContent = r.head && Number.isFinite(r.head.dPitch) ? `${signed(r.head.dPitch, 1)}° / ${signed(r.head.dYaw, 1)}° / ${signed(r.head.dRoll, 1)}°` : "--";
  $("m-perclos").textContent = r.eyes ? `${br(r.eyes.perclos * 100, 1)} % (com as piscadas: ${br(r.eyes.perclosAll * 100, 1)} %; observado ${br(r.eyes.perclosObservedMs / 1000, 0)} s)` : "--";
  $("m-blinks").textContent = r.eyes ? `${r.eyes.curBlinkMs ? Math.round(r.eyes.curBlinkMs) : "--"} ms (base ${r.eyes.baseBlinkMs ? Math.round(r.eyes.baseBlinkMs) : "--"} ms)` : "--";
  $("m-closedby").textContent = r.eyes ? `${r.eyes.byBlend ? "blendshape " : ""}${r.eyes.byEar ? "EAR " : ""}${r.eyes.gated ? "(trava: olhar p/ baixo)" : ""}`.trim() || "--" : "--";
  if (S.lastBlink) {
    const f = S.lastBlink.features;
    $("m-feat").textContent = `[${f[0].toFixed(2)}, ${f[1].toFixed(3)}, ${f[2]}, ${f[3].toFixed(3)}]`;
  }
  for (const k of ["lm", "eye", "tmp", "mon", "tot"]) {
    const st = S.lat[k];
    $(`l-${k}-last`).textContent = st.n ? fmtMs(st.last) : "--";
    $(`l-${k}-p50`).textContent = st.n ? fmtMs(st.p50) : "--";
    $(`l-${k}-p95`).textContent = st.n ? fmtMs(st.p95) : "--";
  }
}

// ---------------------------------------------------------------------------
// Benchmark de 60 s (Exp. 4) — também registra o detector v2 quadro a quadro
// ---------------------------------------------------------------------------
function startBenchmark() {
  if (S.bench || !S.running) return;
  for (const k of Object.keys(S.lat)) S.lat[k] = new E.LatencyStats(300);
  S.bench = { t0: performance.now(), rows: [], alarms: 0, warns: 0, dings: 0, events: {} };
  $("btn-bench").classList.add("active");
  $("btn-bench").textContent = "Benchmark...";
  $("bench-summary").classList.add("hidden");
  $("bench-downloads").innerHTML = "";
}
function benchRow(t0, t1, eyeMs, tmpMs, monMs, tot, blinkNow, r) {
  const b = S.blend || {};
  const evTypes = r.events.map((e) => e.type);
  for (const tpe of evTypes) S.bench.events[tpe] = (S.bench.events[tpe] || 0) + 1;
  if (r.alarm.ding) S.bench.dings += 1;
  S.bench.rows.push({
    t_ms: Math.round(t0 - S.bench.t0),
    face: S.faceDetected ? 1 : 0,
    fps: S.fps,
    lat_landmarker_ms: t1 - t0,
    lat_eye_cnn_ms: eyeMs,
    lat_temporal_ms: tmpMs,
    lat_motor_ms: monMs,
    lat_total_ms: tot,
    ear_left: S.ear.left,
    ear_right: S.ear.right,
    mar: S.mar,
    blend_left: b.eyeBlinkLeft ?? null,
    blend_right: b.eyeBlinkRight ?? null,
    jaw_open: b.jawOpen ?? null,
    look_down: r.eyes?.lookDown ?? null,
    pitch: S.pose ? S.pose.pitch : null,
    yaw: S.pose ? S.pose.yaw : null,
    roll: S.pose ? S.pose.roll : null,
    dpitch: r.head?.dPitch ?? null,
    dyaw: r.head?.dYaw ?? null,
    perclos: r.eyes?.perclos ?? null,
    perclos_com_piscadas: r.eyes?.perclosAll ?? null,
    closed: r.eyes?.closed ? 1 : 0,
    closed_blend: r.eyes?.byBlend ? 1 : 0,
    closed_ear: r.eyes?.byEar ? 1 : 0,
    trava_baixo: r.eyes?.gated ? 1 : 0,
    closed_ms: r.eyes?.closedMs ?? 0,
    blink: blinkNow ? 1 : 0,
    eye_open_left: S.eyeProb.left,
    eye_open_right: S.eyeProb.right,
    yawn_open: r.yawn?.open ? 1 : 0,
    head_state: r.head?.state ?? r.phase,
    droll: r.head?.dRoll ?? null,
    nivel: r.score?.level ?? 0,
    estado: r.score?.state ?? 0,
    forcado: r.score?.forced ? 1 : 0,
    pos_evento_ms: r.score?.postForcedMs ?? 0,
    lstm_soma: r.score?.mlCounted ? 1 : 0,
    sirene: r.alarm?.siren ? 1 : 0,
    ding: r.alarm?.ding ? 1 : 0,
    eventos: evTypes.join(";"),
    fase: r.phase,
    cls: S.temporal ? S.temporal.cls : null,
    alert: r.alarm?.siren ? 1 : 0,
  });
  const el = t0 - S.bench.t0;
  $("bench-status").textContent = `Benchmark em andamento: ${Math.round(el / 1000)}/60 s, ${S.bench.rows.length} quadros`;
  if (el >= BENCH_MS) finishBenchmark();
}
function stats(values) {
  const v = values.filter((x) => Number.isFinite(x));
  if (!v.length) return { n: 0 };
  const s = [...v].sort((a, b) => a - b);
  const mean = v.reduce((a, b) => a + b, 0) / v.length;
  return { n: v.length, mean, p50: E.percentile(s, 50), p95: E.percentile(s, 95), max: s[s.length - 1] };
}
function finishBenchmark() {
  const b = S.bench;
  S.bench = null;
  $("btn-bench").classList.remove("active");
  $("btn-bench").textContent = "Benchmark 60 s";
  const rows = b.rows;
  const withFace = rows.filter((r) => r.face);
  const dur = rows.length ? (rows[rows.length - 1].t_ms - rows[0].t_ms) / 1000 : 0;
  const tState = [0, 0, 0];
  for (let i = 1; i < rows.length; i++) if (rows[i].fase === "ativo") tState[rows[i].estado] += (rows[i].t_ms - rows[i - 1].t_ms) / 1000;
  const ev = b.events;
  const summary = {
    tipo: "benchmark_pwa",
    versao_app: APP_VERSION,
    detector: "v2.1: olhos (EAR+blendshape) + bocejo (jawOpen) + cabeça (matriz de pose) -> nível 0-100 (PERCLOS só com fechamentos >= 0,5 s; LSTM só soma com os sinais visíveis >= 40)",
    data: new Date().toISOString(),
    aparelho: showDeviceInfo({
      delegate: S.delegate,
      video: `${video.videoWidth}x${video.videoHeight}`,
      wasmSimd: S.wasmSimd ?? null,
      modelo_ua_ch: (S.uaHints && S.uaHints.model) || "",
      versao_plataforma: (S.uaHints && S.uaHints.platformVersion) || "",
    }),
    delegate_landmarker: S.delegate,
    modelos: { landmarker: "face_landmarker float16 v1 (blendshapes + matriz)", eye_state: S.eyeFile, blink_temporal: S.tmpFile },
    duracao_s: dur,
    quadros: rows.length,
    quadros_com_rosto: withFace.length,
    fps_medio: dur > 0 ? rows.length / dur : 0,
    latencia_ms: {
      landmarker: stats(rows.map((r) => r.lat_landmarker_ms)),
      cnn_olhos_2olhos: stats(withFace.filter((r) => r.lat_eye_cnn_ms > 0).map((r) => r.lat_eye_cnn_ms)),
      temporal: stats(withFace.filter((r) => r.lat_temporal_ms > 0).map((r) => r.lat_temporal_ms)),
      motor_v2: stats(rows.map((r) => r.lat_motor_ms)),
      total_por_quadro: stats(rows.map((r) => r.lat_total_ms)),
    },
    calibracao: S.monitor.baseline,
    som: { volume_alarme_pct: S.volumePct, volume_fonte: S.volumeSource, ganho: Sound.gain, audio: Sound.ctx ? Sound.ctx.state : "não iniciado", audio_session: "audioSession" in navigator ? navigator.audioSession.type : null },
    eventos: { bocejo: ev.bocejo || 0, queda: ev.queda || 0, cabeceio: ev.cabeceio || 0, fechamento: ev.fechamento || 0, microssono: ev.microssono || 0, cabeca_baixa: ev.cabeca_baixa || 0 },
    alarmes: ev.alarme || 0,
    avisos: ev.aviso || 0,
    dings: b.dings,
    tempo_por_estado_s: { ALERTA: tState[0], "ATENÇÃO": tState[1], SONOLENTO: tState[2] },
    nivel_max: rows.reduce((a, r) => Math.max(a, r.nivel), 0),
    nivel_medio: stats(rows.filter((r) => r.fase === "ativo").map((r) => r.nivel)).mean ?? null,
    perclos_final: rows.length ? rows[rows.length - 1].perclos : null,
    piscadas_no_periodo: rows.reduce((a, r) => a + r.blink, 0),
    quadros_em_alerta: rows.reduce((a, r) => a + r.alert, 0),
    ear_medio: stats(withFace.map((r) => (r.ear_left + r.ear_right) / 2)).mean,
  };
  window.__benchSummary = summary; // para automação/testes (Chrome headless via DevTools)
  window.__benchRows = rows;
  const stamp = summary.data.replace(/[:.]/g, "-");
  const jsonBlob = new Blob([JSON.stringify(summary, null, 2)], { type: "application/json" });
  const csvBlob = new Blob([E.toCSV(rows)], { type: "text/csv" });
  const dl = $("bench-downloads");
  dl.innerHTML = "";
  dl.appendChild(downloadButton("Baixar resumo JSON", jsonBlob, `benchmark_pwa_${stamp}.json`));
  dl.appendChild(downloadButton("Baixar CSV por quadro", csvBlob, `benchmark_pwa_${stamp}_frames.csv`));
  const L = summary.latencia_ms;
  const line = (name, s) => `${name.padEnd(18)} p50 ${fmtMs(s.p50)}  p95 ${fmtMs(s.p95)}  média ${fmtMs(s.mean)}  n=${s.n}`;
  const T = summary.tempo_por_estado_s;
  $("bench-summary").textContent = [
    `Delegate: ${S.delegate} | ${summary.quadros} quadros em ${dur.toFixed(1)} s | FPS médio ${summary.fps_medio.toFixed(1)}`,
    line("landmarker", L.landmarker),
    line("CNN olhos (2)", L.cnn_olhos_2olhos),
    line("LSTM 30x4", L.temporal),
    line("motor v2", L.motor_v2),
    line("total/quadro", L.total_por_quadro),
    `Eventos: ${Object.entries(summary.eventos).map(([k, v]) => `${k} ${v}`).join(", ")} | alarmes ${summary.alarmes} | avisos ${summary.avisos} | dings ${summary.dings}`,
    `Tempo: ALERTA ${T.ALERTA.toFixed(1)} s · ATENÇÃO ${T["ATENÇÃO"].toFixed(1)} s · SONOLENTO ${T.SONOLENTO.toFixed(1)} s | nível máx. ${Math.round(summary.nivel_max)}`,
    `Modelos: eye_state=${S.eyeFile || "ausente"} temporal=${S.tmpFile || "ausente"}`,
  ].join("\n");
  $("bench-summary").classList.remove("hidden");
  $("bench-status").textContent = "Benchmark concluído. Enviando para o computador...";
  console.log("[benchmark] " + $("bench-summary").textContent.replace(/\n/g, " | "));
  sendBenchmarkToComputer(summary, E.toCSV(rows), stamp);
}

// Envia o resultado ao serve_https.py (POST /api/benchmark), que grava em results/phone/ no
// repositório. Assim o teste no celular não exige transferir arquivos. Se o app estiver hospedado
// em outro lugar (ex.: GitHub Pages), o envio falha e os botões de download continuam valendo.
async function sendBenchmarkToComputer(summary, csv, stamp) {
  try {
    const r = await fetch("api/benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ stamp, summary, csv }),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const info = await r.json();
    $("bench-status").textContent = `Benchmark concluído e enviado para o computador: ${info.salvo_em.join(", ")}`;
  } catch (e) {
    $("bench-status").textContent =
      "Benchmark concluído. Não foi possível enviar ao computador (" + e.message +
      "); use os botões abaixo e salve os arquivos em results/phone/ no repositório.";
  }
}
function downloadButton(label, blob, filename) {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "botao secundario";
  b.textContent = label;
  b.onclick = () => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  };
  return b;
}

// ---------------------------------------------------------------------------
// Exportação da sessão (calibração, eventos, piscadas, predições, parâmetros)
// ---------------------------------------------------------------------------
function exportSession() {
  const data = {
    tipo: "sessao_pwa",
    versao_app: APP_VERSION,
    data: new Date().toISOString(),
    aparelho: deviceInfo(),
    delegate: S.delegate,
    modelos: { eye_state: S.eyeFile, blink_temporal: S.tmpFile },
    configuracao_detector_v2: E.CFG,
    calibracao: S.monitor.baseline,
    duracao_tipica_piscada_base_ms: S.monitor.blinkBaseMs,
    volume_alarme_pct: S.volumePct,
    estatisticas: S.monitor.stats,
    eventos: S.eventLog.map(({ id, type, wall, text, kind, durMs }) => ({ id, tipo: type, hora: wall, texto: text, tipo_visual: kind, duracao_ms: durMs ?? null })),
    parametros_lstm: {
      limiar_ear: S.detector?.threshold ?? null,
      min_closed_frames: 2,
      max_closed_ms: 1500,
      n_calib_piscadas: S.normalizer.nCalib,
      media_sujeito: S.normalizer.mean,
      desvio_sujeito: S.normalizer.std,
      janela: S.blinkWindow.size,
      ordem_caracteristicas: E.FEATURE_NAMES,
      classes: E.CLASS_NAMES,
    },
    piscadas: S.blinkLog,
    predicoes_lstm: S.predLog,
    latencia_ms: Object.fromEntries(Object.entries(S.lat).map(([k, v]) => [k, { p50: v.p50, p95: v.p95 }])),
  };
  const stamp = data.data.replace(/[:.]/g, "-");
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  downloadButton("x", blob, `sessao_pwa_${stamp}.json`).onclick();
  const csvBlob = new Blob([E.toCSV(S.blinkLog.map((b) => ({ t: b.t, start: b.start, bottom: b.bottom, end: b.end, closedMs: b.closedMs, frequency: b.raw[0], amplitude: b.raw[1], duration: b.raw[2], velocity: b.raw[3], z_frequency: b.z[0], z_amplitude: b.z[1], z_duration: b.z[2], z_velocity: b.z[3], calib: b.calib ? 1 : 0 })))], { type: "text/csv" });
  if (S.blinkLog.length) setTimeout(() => downloadButton("x", csvBlob, `sessao_pwa_${stamp}_piscadas.csv`).onclick(), 600);
}

// ---------------------------------------------------------------------------
// Tela cheia / Wake Lock
// ---------------------------------------------------------------------------
async function toggleFullscreen() {
  try {
    if (!document.fullscreenElement) await document.documentElement.requestFullscreen();
    else await document.exitFullscreen();
  } catch (e) {
    notice(`Tela cheia indisponível: ${e.message}`);
  }
}
function setWakeButton(on) {
  $("btn-wake").setAttribute("aria-checked", String(on));
}
async function requestWakeLock() {
  if (!("wakeLock" in navigator)) return false;
  try {
    S.wakeLock = await navigator.wakeLock.request("screen");
    S.wakeWanted = true;
    setWakeButton(true);
    S.wakeLock.addEventListener("release", () => {
      S.wakeLock = null;
      setWakeButton(false);
    });
    return true;
  } catch (e) {
    console.warn("Wake Lock", e.message);
    return false;
  }
}
async function toggleWakeLock() {
  if (!("wakeLock" in navigator)) return notice("Wake Lock API indisponível neste navegador: ajuste o bloqueio automático da tela nos Ajustes.");
  if (S.wakeLock) {
    S.wakeWanted = false;
    await S.wakeLock.release().catch(() => {});
    S.wakeLock = null;
    setWakeButton(false);
  } else if (!(await requestWakeLock())) notice("Wake Lock falhou (a página precisa estar visível).");
}

// ---------------------------------------------------------------------------
// Inicialização
// ---------------------------------------------------------------------------
function init() {
  // Dependências começam a carregar já; os erros aparecem no "Iniciar viagem" (MediaPipe) ou viram modo
  // degradado (TensorFlow.js), por isso as promessas não ficam sem tratamento aqui.
  S.visionReady = loadVision();
  S.visionReady.catch(() => {});
  S.tfReady = loadTfjs();
  S.tfReady.catch(() => {});
  showDeviceInfo();
  initVolume(); // antes de qualquer som: ?volume= / valor salvo / padrão 20 %
  initTema();
  renderComoFunciona();
  setText("versao-app", APP_VERSION);
  const ultima = carregarResumo();
  renderResumo(ultima);
  renderUltima(ultima);
  ir("inicio", { historico: false });
  // O desbloqueio do áudio precisa acontecer DENTRO do toque, antes de qualquer await.
  $("btn-start").onclick = () => {
    Sound.unlock();
    iniciarViagem();
  };
  for (const id of ["btn-sound", "btn-testar-inicio", "btn-testar-ajustes"]) $(id).onclick = testSound;
  const recalibrar = () => {
    Sound.unlock();
    startCalibration();
  };
  $("btn-calib").onclick = recalibrar;
  $("btn-refazer").onclick = recalibrar;
  $("btn-comecar").onclick = () => ir("viagem");
  $("btn-cancelar").onclick = cancelarViagem;
  $("btn-som").onclick = abrirSom;
  $("btn-volume").onclick = abrirSom;
  $("btn-encerrar").onclick = () => abrirDialogo("dlg-encerrar");
  $("btn-confirmar-encerrar").onclick = encerrarViagem;
  $("btn-encerrar-alarme").onclick = encerrarViagem;
  $("btn-exportar-resumo").onclick = () => {
    if (S.eventLog.length) exportSession();
    else notice("Nada para exportar ainda: a sessão começa no Iniciar viagem.");
  };
  setupDeslizar();
  $("btn-bench").onclick = startBenchmark;
  $("btn-export").onclick = exportSession;
  $("btn-full").onclick = toggleFullscreen;
  $("btn-wake").onclick = toggleWakeLock;
  $("notice").onclick = () => notice("");
  for (const b of document.querySelectorAll("#abas [data-aba]")) b.onclick = () => irAba(b.dataset.aba);
  for (const b of document.querySelectorAll(".segmentado [data-tema]")) b.onclick = () => aplicarTema(b.dataset.tema);
  // Botões com data-ir (abre outra tela), data-voltar e data-fechar (fecha a folha ou o diálogo)
  document.addEventListener("click", (ev) => {
    const alvo = ev.target.closest("[data-ir], [data-voltar], [data-fechar]");
    if (!alvo) return;
    if (alvo.hasAttribute("data-fechar")) fecharDialogo(alvo.closest("dialog"));
    else if (alvo.hasAttribute("data-voltar")) voltar();
    else ir(alvo.dataset.ir);
  });
  // Toque fora da folha ou do diálogo fecha
  for (const d of document.querySelectorAll("dialog")) {
    d.addEventListener("click", (ev) => {
      if (ev.target !== d) return;
      const r = d.getBoundingClientRect();
      if (ev.clientX < r.left || ev.clientX > r.right || ev.clientY < r.top || ev.clientY > r.bottom) fecharDialogo(d);
    });
  }
  for (const t of document.querySelectorAll(".tela")) t.addEventListener("scroll", agendarCamera, { passive: true });
  window.addEventListener("resize", agendarCamera);
  window.addEventListener("orientationchange", agendarCamera);
  try {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", atualizarCorTema);
  } catch {
    /* navegador antigo: a cor da barra do sistema fica a do carregamento */
  }
  // Toque real em qualquer lugar: libera a vibração e, com a câmera ligada, reativa o som se o sistema o
  // pausou (click/touchend/keydown são os eventos que o WebKit aceita como gesto para o áudio).
  const onGesture = (ev) => {
    if (ev.isTrusted) S.userTapped = true;
    if ((S.running || Sound.ctx) && !Sound.ready) Sound.unlock();
  };
  for (const type of ["click", "touchend", "keydown"]) document.addEventListener(type, onGesture, { capture: true, passive: true });
  if (!window.isSecureContext) notice("Página fora de contexto seguro: a câmera só funciona em HTTPS ou localhost. Use serve_https.py.", "error");
  if ("serviceWorker" in navigator && window.isSecureContext) {
    navigator.serviceWorker.register("./sw.js").catch((e) => console.warn("SW não registrado (normal com certificado autoassinado):", e.message));
  }
  document.addEventListener("visibilitychange", async () => {
    if (document.visibilityState === "visible") {
      if (S.wakeWanted && S.wakeLock === null) await requestWakeLock();
      if (Sound.ctx && Sound.ctx.state !== "running") Sound.ctx.resume().catch(() => {});
      agendarCamera();
    }
  });
}
init();
// Parâmetros de URL para testes automatizados: ?autostart=1 inicia a viagem (liga a câmera e calibra, sem
// liberar o som: não é gesto do usuário); &bench=1 inicia o benchmark ao ligar.
const params = new URLSearchParams(location.search);
if (params.get("autostart") === "1") {
  iniciarViagem().then(() => {
    if (params.get("bench") === "1" && S.running) startBenchmark();
  });
}
