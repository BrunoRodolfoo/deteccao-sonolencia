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
 *   Em paralelo, só como informação: recorte dos olhos 48x48 -> eye_state.tflite (CNN, Exp. 3), que roda só com
 *   a tela Detalhes técnicos aberta e no benchmark (onde segue a cada quadro, como nas medições do Exp. 4), e
 *   4 características por piscada -> z-score -> janela 30x4 -> blink_temporal.tflite (LSTM, Exp. 1),
 *   que soma no máximo +10 pontos, e só quando os sinais visíveis já estão em ATENÇÃO (>= 40).
 * Som: sirene/ding pelo Web Audio, liberado no primeiro toque (iOS), com o controle "Volume do alarme"
 * (padrão 20 %, salvo no aparelho; ?volume=0..100 na URL sobrepõe sem gravar — testes usam ?volume=0).
 *
 * Sem etapa de build: vanilla JS + ES modules. Dependências de ./vendor/ ou, se faltarem, da CDN. Todos os
 * textos em pt-BR.
 *
 * Interface "Painel de estrada" (07/11/2026): estados com a forma das placas de trânsito, um número grande e uma
 * frase; o detalhe fica no "Por quê?". Primeiro uso -> Início (checagens) -> Calibração guiada -> Em viagem (sempre
 * escura) -> alarme ("segure: estou acordado") ou "Acorde!" (microssono) -> "segure: encerrar" -> resumo (aba
 * Viagens). Também: Como funciona (interativo), Ajustes, Modo pôster e, no modo pesquisador, Detalhes técnicos.
 * Detector, parâmetros e benchmark iguais aos da 2.1.
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
  somTestado: false, // "Testar" tocou nesta visita (checagem do Início)
  vibrarOn: true, // Ajustes > Vibrar junto com o som (só existe no Android)
  pesquisa: false, // modo pesquisador: Detalhes técnicos, benchmark e exportação
  userTapped: false, // houve toque real (isTrusted) na página: libera a vibração
  // câmera ligada para quê: "viagem" (Iniciar viagem) ou "poster" (Modo pôster); null com a câmera desligada
  modo: null,
  faceBox: null, // caixa do rosto na imagem (0-1): centro e altura, para a calibração guiada
  calibPiscadas: 0, // piscadas vistas na fase 1 da calibração (só para a tela)
  calibFechado: false,
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
        // medidor do teste de som (Primeiro uso): lê o sinal DEPOIS do volume do alarme, o que sai de fato
        this.analyser = ctx.createAnalyser();
        this.analyser.fftSize = 256;
        this.out.connect(this.analyser);
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
/** O volume aparece em vários lugares (folha do Som, Ajustes, Primeiro uso, chip da viagem, checagem do Início). */
function showVolume() {
  for (const el of document.querySelectorAll(".vol-slider")) {
    if (el.value !== String(S.volumePct)) el.value = String(S.volumePct);
    el.setAttribute("aria-valuetext", `${S.volumePct} %`);
  }
  for (const id of ["vol-out", "vol-out2", "pu-vol-out"]) setText(id, `${S.volumePct} %`);
  updateAudioStatus();
  atualizarSomChip();
  if (UI.tela === "inicio") Checagem.atualizar();
}
function initVolume() {
  const params = new URLSearchParams(location.search);
  const r = E.resolveVolume({ urlValue: params.has("volume") ? params.get("volume") : null, storedValue: readStoredVolume() });
  S.volumePct = r.pct;
  S.volumeSource = r.source;
  Sound.setVolume(r.pct);
  for (const el of document.querySelectorAll(".vol-slider")) {
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
  }
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
  else if (S.volumePct < 50) txt = `Volume baixo (${S.volumePct} %), bom para testar sem susto. Na estrada ou na demonstração, suba para 100 % e deixe o volume de mídia do aparelho alto.`;
  else txt = `Som ativo (${S.volumePct} %). O volume final também depende do volume de mídia do aparelho.`;
  if (S.volumeSource === "url") txt += " (Volume definido pela URL ?volume=; não é salvo.)";
  hint.textContent = txt;
  hint.classList.toggle("warn", S.volumePct === 0);
}
function testSound() {
  Sound.unlock();
  S.soundTesting = true;
  S.somTestado = true;
  if (UI.tela === "inicio") Checagem.atualizar();
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
    if (!navigator.vibrate || !S.userTapped || !S.vibrarOn) return;
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
/** Liga a câmera e o detector para uma viagem ("viagem") ou para o Modo pôster ("poster"). */
async function startCamera(modo = "viagem") {
  const btn = $("btn-start");
  if (S.running) return;
  // "Cancelar" (ou sair do modo pôster) durante o carregamento troca a ficha: depois de cada espera, uma ficha
  // velha desiste em silêncio, em vez de ligar a câmera de uma viagem que já foi cancelada.
  const ficha = (S.fichaInicio = (S.fichaInicio || 0) + 1);
  const cancelado = () => ficha !== S.fichaInicio;
  S.modo = modo;
  btn.disabled = true;
  setBadge("carregando modelos", "warn");
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
      return;
    }
    S.running = true;
    $("btn-bench").disabled = modo !== "viagem";
    $("btn-export").disabled = false;
    document.body.classList.add("rodando");
    placeCamera();
    setBadge(`${S.delegate} · FPS --`, "ok");
    console.log(`[app] câmera ${video.videoWidth}x${video.videoHeight}, landmarker delegate=${S.delegate}, eye_state=${S.eyeFile || "ausente"}, temporal=${S.tmpFile || "ausente"}, dependências mediapipe=${S.deps.mediapipe} tfjs=${S.deps.tfjs}`);
    S.wasmSimd = await detectSimd();
    showDeviceInfo({ delegate: S.delegate, video: `${video.videoWidth}x${video.videoHeight}`, wasmSimd: S.wasmSimd });
    requestWakeLock();
    if (modo === "poster") Poster.esperar();
    else startCalibration();
    requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    S.running = false;
    S.modo = null;
    stopStream(); // a câmera não fica ligada se o detector não subiu
    S.viagem = null;
    document.body.classList.remove("rodando");
    placeCamera();
    notice(`Erro ao iniciar: ${e.message}`, "error");
    setBadge("erro", "danger");
    btn.disabled = false;
    ir(modo === "poster" ? "ajustes" : "inicio", { historico: false });
  }
}

// ---------------------------------------------------------------------------
// Calibração: fase 1 (10 s olhando para a frente: limiar do EAR, pose, boca de repouso) ->
// fase 2 em segundo plano (20 piscadas: duração de base e z-score do LSTM). O detector já funciona
// durante a fase 2.
// ---------------------------------------------------------------------------
function startCalibration() {
  S.monitor.startCalibration(performance.now());
  resetarCaminhoLSTM();
  S.faceLostSince = null;
  resetCalibUI();
  $("blink-calib").classList.add("hidden");
  $("m-calib").textContent = "fase 1: 10 s olhando para a frente";
  addLog({ id: `c${Date.now()}`, type: "info", text: "calibração iniciada" });
  if (UI.tela !== "calibracao") ir("calibracao");
}
/** Zera o caminho do LSTM (detector de piscadas, z-score e janela): ele depende da calibração do monitor. */
function resetarCaminhoLSTM() {
  S.normalizer.reset();
  S.blinkWindow.reset();
  S.smoother.reset();
  S.temporal = null;
  S.detector = null;
  S.calibPiscadas = 0;
  S.calibFechado = false;
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
    // caixa do rosto (testa 10, queixo 152, bochechas 234 e 454), para a calibração guiada
    const xs = [lms[234].x, lms[454].x];
    S.faceBox = { cx: (xs[0] + xs[1]) / 2, cy: (lms[10].y + lms[152].y) / 2, h: Math.abs(lms[152].y - lms[10].y) };
    // piscadas vistas na fase 1 da calibração (só para a tela): os dois eyeBlink passam de 0,5
    if (S.monitor.phase === "calibrando" && S.blend) {
      const fechado = Math.min(S.blend.eyeBlinkLeft ?? 0, S.blend.eyeBlinkRight ?? 0) >= E.CFG.eyes.blendClosed;
      if (fechado && !S.calibFechado) S.calibPiscadas += 1;
      S.calibFechado = fechado;
    }

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

    // CNN dos olhos (informativa): só com a tela Detalhes técnicos aberta e no benchmark, onde roda a cada quadro
    // como nas medições do Exp. 4; no uso normal, ela não entra no nível e não precisa gastar bateria
    if (S.eyeModel && (S.bench || UI.tela === "tecnico")) {
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
    aoEvento(e); // ficha da viagem, faixa de Atenção, linha do tempo do modo pôster
    if (e.type === "calibrado") {
      const b = e.baseline;
      $("m-calib").textContent = `fase 2: piscadas (limiar EAR ${fmt(b.earThr)}, pitch ${fmt(b.pitch0, 1)}°, jawOpen ${fmt(b.jaw0, 2)})`;
      $("m-thresh").textContent = `${fmt(b.earThr)} (mediana ${fmt(b.earMedian)})`;
      if (b.warnings.length) addLog({ id: `w${e.id}`, type: "info", kind: "warn", text: `calibração com aviso: ${b.warnings.join(" ")}` });
      if (S.modo === "poster") Poster.calibrado();
      else mostrarCalibPronta(b.warnings);
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
// Interface "Painel de estrada" (07/11/2026): Primeiro uso, Início (checagens reais), Calibração guiada, Em viagem
// (placa, arco por sinal, "Por quê?", alarme com "segure"), Viagens (resumo), Como funciona (interativo, com as
// classes do motor), Ajustes, Detalhes técnicos (modo pesquisador) e Modo pôster. A tela só LÊ o resultado do
// monitor (e, para o "some em", os eventos que ele guarda); nada aqui muda o detector, e a atualização da tela fica
// fora do tempo medido por quadro (benchmark).
// ---------------------------------------------------------------------------
const UI = {
  tela: "inicio",
  historico: [],
  autoAvanco: 0,
  alarmeVisivel: false,
  microVisivel: false,
  cameraAgendada: false,
  calibPronta: false,
  faixaAte: 0, //      até quando mostrar a faixa "Sinais de cansaço" (entrada em Atenção)
  placa: "",
  toastTimer: 0,
  toquesVersao: 0,
  segsViagem: null,
  htmlSinais: "",
  htmlPorque: "",
};
const STATE_TITLES = ["Alerta", "Atenção", "Sonolento"];
const PLACAS = ["p-verde", "p-ambar", "p-vermelha"];
const WHY_SHORT = { perclos: "PERCLOS", fechamentos: "olhos fechados", bocejos: "bocejos", cabeca: "cabeça", piscadas: "piscadas lentas", modelo: "modelo LSTM", pos_evento: "após microssono/cabeça" };
const TEMA_KEY = "sonolencia.tema";
const RESUMO_KEY = "sonolencia.ultimaViagem";
const INTRO_KEY = "sonolencia.introVista";
const PESQUISA_KEY = "sonolencia.pesquisador";
const VIBRAR_KEY = "sonolencia.vibrar";
const SEM_LEITURA_MS = 10000; // sem rosto por 10 s: "Sem leitura" (o estado nunca parece verde sem leitura)
const ARCO = "M68.08 251.92 A130 130 0 1 1 251.92 251.92";
const SVGNS = "http://www.w3.org/2000/svg";
const TELAS_ESCURAS = new Set(["calibracao", "viagem", "poster"]);
const TELAS_COM_CAMERA = new Set(["calibracao", "poster"]);

function cap(s) {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}
function setText(id, txt) {
  const el = $(id);
  if (el && el.textContent !== txt) el.textContent = txt;
}
function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}
function horaCurta(d) {
  return d.toLocaleTimeString("pt-BR", { hour: "2-digit", minute: "2-digit", hour12: false });
}
function diaRelativo(d) {
  const hoje = new Date();
  const ontem = new Date(hoje);
  ontem.setDate(hoje.getDate() - 1);
  if (d.toDateString() === hoje.toDateString()) return "hoje";
  if (d.toDateString() === ontem.toDateString()) return "ontem";
  return d.toLocaleDateString("pt-BR", { day: "2-digit", month: "2-digit" });
}
function duracaoTexto(seg) {
  const min = Math.max(0, Math.round(seg / 60));
  return min < 60 ? `${min} min` : `${Math.floor(min / 60)} h ${String(min % 60).padStart(2, "0")} min`;
}
function plural(n, um, varios) {
  return `${n} ${n === 1 ? um : varios}`;
}
/** 75 s -> "1:15" (contagem regressiva das faixas e do "some em"). */
function mmss(seg) {
  const s = Math.max(0, Math.ceil(seg));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
/** Tempo de viagem em horas e minutos: 42 min -> "0:42". */
function horasMin(ms) {
  const m = Math.max(0, Math.floor(ms / 60000));
  return `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}`;
}
function lerLocal(chave) {
  try {
    return window.localStorage.getItem(chave);
  } catch {
    return null; // modo privado, dados bloqueados etc.
  }
}
function gravarLocal(chave, valor) {
  try {
    window.localStorage.setItem(chave, valor);
  } catch {
    /* vale só nesta visita */
  }
}
function trocarClasse(el, opcoes, nova) {
  if (el.dataset.c === nova) return false;
  el.classList.remove(...opcoes);
  if (nova) el.classList.add(nova);
  el.dataset.c = nova;
  return true;
}
function setHTML(el, html) {
  if (el.dataset.html === html) return;
  el.innerHTML = html;
  el.dataset.html = html;
}
const usar = (id, extra = "") => `<svg aria-hidden="true"${extra}><use href="#${id}"/></svg>`;

// ---- navegação ----
function telaDaViagem() {
  return S.monitor.calibrated && S.last?.phase !== "calibrando" ? "viagem" : "calibracao";
}
function ir(nome, { historico = true } = {}) {
  const nova = $(`tela-${nome}`);
  if (!nova) return;
  const anterior = UI.tela;
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
  const escura = TELAS_ESCURAS.has(nome);
  document.documentElement.classList.toggle("fundo-escuro", escura);
  document.body.classList.toggle("fundo-escuro", escura);
  const aba = nova.dataset.abas || "";
  document.body.classList.toggle("com-abas", !!aba);
  for (const b of document.querySelectorAll("#abas [data-aba]")) b.setAttribute("aria-current", b.dataset.aba === aba ? "page" : "false");
  atualizarCorTema();
  if (anterior === "poster" && nome !== "poster") Poster.sair();
  if (nome === "poster" && anterior !== "poster") Poster.entrar();
  if (nome === "como") Como.iniciar();
  else if (anterior === "como") Como.parar();
  if (nome === "inicio") Checagem.atualizar();
  if (nome === "resumo" && anterior !== "resumo") renderResumo(carregarResumo());
  if (nome === "primeiro" && anterior !== "primeiro") PrimeiroUso.passo(0);
  if (nome === "tecnico" && S.last) updateTech(S.last);
  placeCamera();
}
function voltar() {
  let prev = UI.historico.pop();
  while (prev && prev === UI.tela) prev = UI.historico.pop();
  if ((prev === "calibracao" || prev === "viagem") && !S.running) prev = "inicio";
  ir(prev || (S.running && S.modo === "viagem" ? telaDaViagem() : "inicio"), { historico: false });
}
function irAba(aba) {
  if (aba === "inicio") ir(S.running && S.modo === "viagem" ? telaDaViagem() : "inicio");
  else if (aba === "viagens") ir("resumo");
  else ir(aba);
}

// ---- câmera: o mesmo <video> fica sobre o encaixe da tela atual (painel, círculo ou flutuante) ----
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

// ---- tema (Ajustes > Aparência); a viagem, a calibração e o modo pôster são sempre escuros ----
function temaEscuroAtivo() {
  const t = document.documentElement.dataset.tema;
  return t === "escuro" || (!t && window.matchMedia("(prefers-color-scheme: dark)").matches);
}
function atualizarCorTema() {
  const m = document.querySelector('meta[name="theme-color"]');
  if (m) m.content = TELAS_ESCURAS.has(UI.tela) || temaEscuroAtivo() ? "#000000" : "#F2F3F5";
}
function aplicarTema(t, salvar = true) {
  const valor = t === "claro" || t === "escuro" ? t : "auto";
  if (valor === "auto") delete document.documentElement.dataset.tema;
  else document.documentElement.dataset.tema = valor;
  if (salvar) gravarLocal(TEMA_KEY, valor);
  for (const b of document.querySelectorAll(".segmentado [data-tema]")) b.setAttribute("aria-checked", String(b.dataset.tema === valor));
  atualizarCorTema();
}
function initTema() {
  aplicarTema(lerLocal(TEMA_KEY) || "auto", false);
}

// ---- modo pesquisador (7 toques em Versão, ou ?pesquisa=1): detalhes técnicos, benchmark e exportação ----
function definirPesquisa(on, salvar = true) {
  S.pesquisa = !!on;
  document.body.classList.toggle("pesquisa", S.pesquisa);
  $("btn-pesquisa").setAttribute("aria-checked", String(S.pesquisa));
  if (salvar) gravarLocal(PESQUISA_KEY, S.pesquisa ? "1" : "0");
}
function tocarVersao() {
  if (S.pesquisa) return mostrarToast("O modo pesquisador já está ligado. Para desligar, use a chave logo abaixo.");
  UI.toquesVersao += 1;
  const falta = 7 - UI.toquesVersao;
  if (falta <= 0) {
    UI.toquesVersao = 0;
    definirPesquisa(true);
    mostrarToast("Modo pesquisador ligado: Detalhes técnicos, benchmark e exportação aparecem em Ajustes e na viagem.");
  } else if (falta <= 3) mostrarToast(`Mais ${plural(falta, "toque", "toques")} para o modo pesquisador.`);
}
function mostrarToast(texto) {
  const t = $("toast");
  t.textContent = texto;
  t.hidden = false;
  clearTimeout(UI.toastTimer);
  UI.toastTimer = setTimeout(() => (t.hidden = true), 2600);
}
// vibração (só existe no Android): chave em Ajustes, ligada por padrão
function definirVibrar(on, salvar = true) {
  S.vibrarOn = !!on;
  $("btn-vibrar").setAttribute("aria-checked", String(S.vibrarOn));
  if (salvar) gravarLocal(VIBRAR_KEY, S.vibrarOn ? "1" : "0");
}

// ---- diálogos (folhas) ----
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
function fecharDialogos() {
  for (const d of document.querySelectorAll("dialog")) fecharDialogo(d);
}
function abrirSom() {
  abrirDialogo("folha-som");
}
function abrirPorque() {
  if (UI.alarmeVisivel || UI.microVisivel) return; // o alarme cobre a viagem: a folha não abre por cima dele
  if (S.last) renderPorque(S.last);
  abrirDialogo("folha-porque");
}

/** Botão "segure": a ação só acontece depois de 1,2 s com o dedo em cima (o preenchimento mostra o tempo); soltar ou
 *  arrastar para fora antes cancela. Pelo teclado (Enter ou espaço), a ação é direta. */
function segurar(btn, acao, ms = 1200) {
  let timer = 0;
  let caixa = null;
  const pode = () => !btn.disabled && btn.getAttribute("aria-disabled") !== "true";
  const soltar = () => {
    clearTimeout(timer);
    timer = 0;
    btn.classList.remove("segurando");
  };
  btn.addEventListener("pointerdown", (ev) => {
    if (!pode() || ev.button > 0) return;
    caixa = btn.getBoundingClientRect();
    btn.classList.add("segurando");
    clearTimeout(timer);
    timer = setTimeout(() => {
      soltar();
      acao();
    }, ms);
  });
  btn.addEventListener("pointermove", (ev) => {
    if (!timer || !caixa) return;
    const fora = ev.clientX < caixa.left - 24 || ev.clientX > caixa.right + 24 || ev.clientY < caixa.top - 24 || ev.clientY > caixa.bottom + 24;
    if (fora) soltar();
  });
  for (const tipo of ["pointerup", "pointercancel", "pointerleave"]) btn.addEventListener(tipo, soltar);
  btn.addEventListener("contextmenu", (ev) => ev.preventDefault());
  btn.addEventListener("click", (ev) => {
    if (ev.detail === 0 && pode()) acao(); // clique vindo do teclado
  });
}

// ---- arco do nível: um traço por família de sinais, na cor do estado ----
function criarSegs(g, n) {
  const segs = [];
  for (let i = 0; i < n; i++) {
    const p = document.createElementNS(SVGNS, "path");
    p.setAttribute("d", ARCO);
    p.setAttribute("pathLength", "100");
    p.setAttribute("class", `dial-seg${i === n - 1 ? " cauda" : ""}`);
    p.style.strokeDasharray = "0 200";
    g.appendChild(p);
    segs.push(p);
  }
  return segs;
}
function desenharSegs(segs, valores) {
  let off = 0;
  valores.forEach((v, i) => {
    const len = v > 1.6 ? v - 0.8 : 0; // fresta entre as famílias
    const da = `${len.toFixed(2)} 200`;
    const doff = (-off).toFixed(2);
    if (segs[i].dataset.da !== da) {
      segs[i].style.strokeDasharray = da;
      segs[i].dataset.da = da;
    }
    if (segs[i].dataset.off !== doff) {
      segs[i].style.strokeDashoffset = doff;
      segs[i].dataset.off = doff;
    }
    off += v;
  });
}
/** Marca do limiar v (0-100) no arco de 270°. */
function marcaDial(el, v) {
  const p = (r) => {
    const a = ((135 + (270 * v) / 100) * Math.PI) / 180;
    return [160 + r * Math.cos(a), 160 + r * Math.sin(a)];
  };
  const [x1, y1] = p(146);
  const [x2, y2] = p(158);
  el.setAttribute("x1", x1.toFixed(2));
  el.setAttribute("y1", y1.toFixed(2));
  el.setAttribute("x2", x2.toFixed(2));
  el.setAttribute("y2", y2.toFixed(2));
}
/** Pontos por família: olhos (fechamentos, PERCLOS, piscadas), boca, cabeça, outros (modelo, piso depois do
 *  vermelho) e a "cauda": o que falta até o nível exibido, que desce no máximo 3 pontos por segundo. */
function familias(sc) {
  const f = [0, 0, 0, 0];
  for (const c of sc.contributions) {
    const i = c.key === "fechamentos" || c.key === "perclos" || c.key === "piscadas" ? 0 : c.key === "bocejos" ? 1 : c.key === "cabeca" ? 2 : 3;
    f[i] += c.points;
  }
  const soma = f.reduce((a, b) => a + b, 0);
  const k = soma > 100 ? 100 / soma : 1;
  const raw = Math.min(100, soma);
  return [...f.map((v) => v * k), Math.max(0, Math.min(100, sc.level) - raw)];
}
/** Inteiros que somam `total` (maiores restos), para "Por que N?" fechar com o número do mostrador. */
function inteirosQueSomam(valores, total) {
  const base = valores.map((v) => Math.floor(v));
  let falta = total - base.reduce((a, b) => a + b, 0);
  const ordem = valores.map((v, i) => ({ i, r: v - Math.floor(v) })).sort((a, b) => b.r - a.r);
  for (const o of ordem) {
    if (falta <= 0) break;
    base[o.i] += 1;
    falta -= 1;
  }
  for (const o of [...ordem].reverse()) {
    if (falta >= 0) break;
    if (base[o.i] > 0) {
      base[o.i] -= 1;
      falta += 1;
    }
  }
  return base;
}
/** Quanto falta para um conjunto de eventos do motor (DecayingEvents) sumir de vez, em s. */
function restaS(...grupos) {
  const t = performance.now();
  let fim = 0;
  for (const g of grupos) for (const e of g.items) fim = Math.max(fim, e.t + g.decayMs - t);
  return fim / 1000;
}
/** Linhas de "Por que N?": o que soma no nível agora, em linguagem simples, com o "some em" de cada sinal. */
function partesPorque(r) {
  const sc = r.score;
  const M = S.monitor;
  const P = E.CFG.score;
  const t = performance.now();
  const L = [];
  for (const c of sc.contributions) {
    if (c.key === "fechamentos") {
      const nM = M.evMicro.count(t);
      const nL = M.evLong.count(t);
      const partes = [nM ? plural(nM, "microssono", "microssonos") : "", nL ? plural(nL, "fechamento longo", "fechamentos longos") : ""].filter(Boolean);
      L.push({ icone: "i-olhos", rotulo: `Olhos: ${partes.join(" e ") || "fechamentos longos"}`, sub: `Some em ${mmss(restaS(M.evLong, M.evMicro))}`, v: c.points, teto: P.closuresCap });
    } else if (c.key === "perclos") {
      L.push({ icone: "i-olhos", rotulo: `Olhos fechados devagar: ${br(r.eyes.perclos * 100, 0)} % do último minuto`, sub: "PERCLOS, só com fechamentos de 0,5 s ou mais", v: c.points, teto: P.perclos.cap });
    } else if (c.key === "piscadas") {
      const extra = r.eyes.blinkRatio ? br((r.eyes.blinkRatio - 1) * 100, 0) : null;
      L.push({ icone: "i-olho", rotulo: extra ? `Piscadas ${extra} % mais longas` : "Piscadas mais longas", sub: "Comparadas às da calibração", v: c.points, teto: P.blinkDur.cap });
    } else if (c.key === "bocejos") {
      L.push({ icone: "i-boca", rotulo: `Boca: ${plural(M.evYawn.count(t), "bocejo", "bocejos")}`, sub: `Some em ${mmss(restaS(M.evYawn))}`, v: c.points, teto: P.yawn.cap });
    } else if (c.key === "cabeca") {
      L.push({ icone: "i-cabeca", rotulo: `Cabeça: ${plural(M.evHead.count(t), "queda", "quedas")}`, sub: `Some em ${mmss(restaS(M.evHead))}`, v: c.points, teto: P.head.cap });
    } else if (c.key === "modelo") {
      L.push({ icone: "i-modelo", rotulo: "Modelo LSTM (experimental)", sub: "Só soma com os sinais visíveis já em Atenção", v: c.points, teto: P.ml.points });
    } else if (c.key === "pos_evento") {
      const kind = M.forcedEnd?.kind === "cabeça caída" ? "da cabeça caída" : "do microssono";
      L.push({ icone: "i-relogio", rotulo: `Mínimo depois ${kind}`, sub: `Por mais ${mmss(sc.postForcedMs / 1000)}`, v: c.points, teto: P.postForced.floor, cinza: true });
    }
  }
  const raw = Math.min(100, L.reduce((a, x) => a + x.v, 0));
  const resto = Math.min(100, sc.level) - raw;
  if (resto >= 0.5) {
    if (sc.forced) L.push({ icone: "i-micro", rotulo: "Vermelho na hora", sub: cap(sc.forced), v: resto, teto: 100, forcada: true });
    else L.push({ icone: "i-relogio", rotulo: "Nível ainda descendo", sub: `Cai no máximo ${P.maxFallPerS} pontos por segundo`, v: resto, teto: 100, cinza: true });
  }
  const pts = inteirosQueSomam(L.map((x) => x.v), Math.floor(sc.level));
  return L.map((x, i) => ({ ...x, pts: pts[i] })).filter((x) => x.pts > 0 || x.forcada);
}
function renderPorque(r) {
  const sc = r.score;
  const lista = $("porque-lista");
  if (!sc || r.phase !== "ativo") {
    setText("porque-titulo", "Calibrando");
    setHTML(lista, '<p class="porque-vazio">O nível começa depois da calibração.</p>');
    return;
  }
  const n = Math.floor(sc.level);
  setText("porque-titulo", `Por que ${n}?`);
  const partes = partesPorque(r);
  const cor = ["var(--verde)", "var(--ambar)", "var(--vermelho)"][sc.state];
  const html = partes.length
    ? partes
        .map(
          (p) =>
            `<div class="parte${p.forcada ? " forcada" : ""}"><span class="ic-redondo">${usar(p.icone)}</span><span class="texto"><strong>${esc(p.rotulo)}</strong><span class="barra"><i style="width:${Math.min(100, (100 * p.v) / p.teto).toFixed(1)}%;--cor-parte:${p.cinza ? "var(--cinza)" : cor}"></i></span><small>${esc(p.sub)}</small></span><span class="pts num">+${p.pts}</span></div>`,
        )
        .join("")
    : '<p class="porque-vazio">Nenhum sinal de cansaço agora.</p>';
  setHTML(lista, html);
}

// ---- viagem: começa em "Iniciar viagem" e termina em "Segure: encerrar" ----
function novaViagem() {
  return { inicio: new Date(), ultimoT: null, ativoMs: 0, tempo: [0, 0, 0], serie: [], prox: 0, nivelMax: 0, estado: 0, n: { alarme: 0, aviso: 0, bocejo: 0, fechamento: 0, microssono: 0, queda: 0, cabeceio: 0 }, momentos: [] };
}
function iniciarViagem() {
  if (S.running) {
    ir(S.modo === "viagem" ? telaDaViagem() : "inicio");
    return Promise.resolve();
  }
  if (!S.viagem) S.viagem = novaViagem();
  resetCalibUI();
  ir("calibracao");
  return startCamera("viagem");
}
function pararCamera() {
  if (S.bench) finishBenchmark();
  S.running = false;
  stopStream();
  S.sirenWanted = false;
  Sound.stopSiren();
  document.body.classList.remove("flashing", "rodando");
  for (const id of ["alarme", "alarme-micro"]) $(id).classList.add("hidden");
  $("poster-alarme").hidden = true;
  UI.alarmeVisivel = false;
  UI.microVisivel = false;
  clearTimeout(UI.autoAvanco);
  if (S.wakeLock) {
    S.wakeWanted = false;
    S.wakeLock.release().catch(() => {});
  }
  S.modo = null;
  $("btn-start").disabled = false;
  $("btn-bench").disabled = true;
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
  fecharDialogos();
  const resumo = fecharResumo();
  S.viagem = null;
  pararCamera();
  addLog({ id: `f${Date.now()}`, type: "info", text: "viagem encerrada" });
  if (resumo) {
    salvarResumo(resumo);
    renderUltima(resumo);
  } else notice("Viagem curta demais para um resumo: ele começa 5 s depois da calibração.");
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
  if (sc.state === 2) texto = sc.forced ? cap(sc.forced) : `Sinais somados: ${top || "vários"}`;
  else if (sc.state === 1) texto = v.estado === 2 ? "O nível caiu, mas segue em Atenção" : `Sinais de cansaço: ${top || "vários"}`;
  else texto = "De volta ao verde";
  const partes = sc.contributions.slice(0, 3).map((c) => ({ k: c.key, pts: Math.round(c.points) })).filter((p) => p.pts > 0);
  v.momentos.push({ hora: horaCurta(new Date()), s: Math.round(v.ativoMs / 1000), estado: sc.state, de: v.estado, nivel: Math.round(sc.level), texto, partes, forced: sc.forced || null });
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
  gravarLocal(RESUMO_KEY, JSON.stringify(res));
}
function carregarResumo() {
  try {
    const txt = lerLocal(RESUMO_KEY);
    const r = txt ? JSON.parse(txt) : null;
    return r && r.v === 1 && Array.isArray(r.serie) && Array.isArray(r.tempoS) ? r : null;
  } catch {
    return null;
  }
}

// ---- resumo da viagem (aba Viagens) ----
const COR_ESTADO = ["var(--verde)", "var(--ambar)", "var(--vermelho)"];
function tituloMomento(m) {
  if (m.estado === 2) return "Alarme";
  if (m.estado === 1) return m.de === 2 || /^Saiu do vermelho|segue em Atenção/.test(m.texto) ? "Saiu do vermelho" : "Atenção";
  return "De volta ao verde";
}
function detalheMomento(m) {
  const partes = (m.partes || []).map((p) => `${cap(WHY_SHORT[p.k] || p.k)} +${p.pts}`);
  const t = [];
  if (m.forced) t.push(`${cap(m.forced)}: alarme na hora, sem pausa possível.`);
  if (partes.length) t.push(`${partes.join(", ")}.`);
  t.push(`Nível ${m.nivel} às ${m.hora}.`);
  return t.join(" ");
}
function renderResumo(res) {
  $("resumo-vazio").classList.toggle("hidden", !!res);
  $("resumo-conteudo").classList.toggle("hidden", !res);
  if (!res) {
    setText("res-titulo", "Viagens");
    setText("res-quando", "");
    return;
  }
  const ini = new Date(res.inicio);
  const fim = new Date(res.fim);
  const dia = diaRelativo(ini);
  setText("res-titulo", `Viagem de ${dia}`);
  setText("res-quando", `${ini.toLocaleDateString("pt-BR", { day: "numeric", month: "long" })} · ${horaCurta(ini)} a ${horaCurta(fim)}`);
  const n = res.n;
  const minutos = Math.max(0, Math.round((fim - ini) / 60000));
  setText("res-min", String(minutos));
  setText("res-min-rot", minutos === 1 ? "minuto" : "minutos");
  setText("res-avisos", String(n.aviso));
  setText("res-alarmes", String(n.alarme));
  setText("res-avisos-rot", n.aviso === 1 ? "aviso" : "avisos");
  setText("res-alarmes-rot", n.alarme === 1 ? "alarme" : "alarmes");
  $("res-avisos").classList.toggle("tem", n.aviso > 0);
  $("res-alarmes").classList.toggle("tem", n.alarme > 0);
  setText("res-pico", String(res.nivelMax));
  setText("res-duracao", duracaoTexto(res.ativoS));
  const maxS = Math.max(1, res.ativoS, ...res.serie.map((p) => p[0]));
  const X = (s) => ((s / maxS) * 342).toFixed(1);
  const Y = (v) => (165 - 1.5 * Math.max(0, Math.min(100, v))).toFixed(1);
  const linha = $("res-linha");
  linha.setAttribute("d", res.serie.map(([s, v], i) => `${i ? "L" : "M"}${X(s)} ${Y(v)}`).join(" "));
  linha.style.animation = "none"; // refaz o desenho da linha a cada visita
  void linha.getBoundingClientRect();
  linha.style.animation = "";
  const g = $("res-pontos");
  g.textContent = "";
  res.momentos
    .filter((m) => m.estado > 0 && !(m.estado === 1 && m.de === 2))
    .forEach((m, i) => {
      const c = document.createElementNS(SVGNS, "circle");
      c.setAttribute("cx", X(m.s));
      c.setAttribute("cy", Y(m.nivel ?? 50));
      c.setAttribute("r", "6");
      c.style.fill = COR_ESTADO[m.estado];
      c.style.animationDelay = `${700 + i * 120}ms`;
      g.appendChild(c);
    });
  setText("res-ini", horaCurta(ini));
  setText("res-meio", horaCurta(new Date((ini.getTime() + fim.getTime()) / 2)));
  setText("res-fim", horaCurta(fim));
  const box = $("res-momentos");
  box.textContent = "";
  const momentos = res.momentos.length ? res.momentos : [{ hora: horaCurta(ini), estado: 0, texto: "Sem avisos nem alarmes", nivel: res.nivelMax, partes: [] }];
  momentos.forEach((m, i) => {
    const d = document.createElement("div");
    d.className = "momento";
    d.style.animationDelay = `${300 + i * 60}ms`;
    const b = document.createElement("button");
    b.type = "button";
    b.setAttribute("aria-expanded", "false");
    b.innerHTML = `<span class="cor" style="background:${COR_ESTADO[m.estado]}"></span><span class="texto"><strong>${esc(`${m.hora} · ${tituloMomento(m)}`)}</strong><small>${esc(m.texto)}</small></span>${usar("i-abrir", ' class="seta"')}`;
    const det = document.createElement("div");
    det.className = "detalhe";
    det.hidden = true;
    det.textContent = detalheMomento(m);
    b.onclick = () => {
      const abrir = det.hidden;
      for (const o of box.querySelectorAll(".momento")) {
        o.querySelector(".detalhe").hidden = true;
        o.querySelector("button").setAttribute("aria-expanded", "false");
      }
      det.hidden = !abrir;
      b.setAttribute("aria-expanded", String(abrir));
    };
    d.append(b, det);
    box.appendChild(d);
  });
  setText("res-nota", n.alarme ? "Houve alarme nesta viagem. Com sono, não continue: descanse antes de voltar a dirigir." : n.aviso ? "Houve sinais de cansaço nesta viagem. Em viagens longas, faça pausas." : "Nenhum aviso nesta viagem.");
}
function renderUltima(res) {
  $("ultima").classList.toggle("hidden", !res);
  if (!res) return;
  const ini = new Date(res.inicio);
  const n = res.n;
  const partes = [duracaoTexto((new Date(res.fim) - ini) / 1000), n.aviso ? plural(n.aviso, "aviso", "avisos") : "sem avisos", n.alarme ? plural(n.alarme, "alarme", "alarmes") : "sem alarme"];
  setText("ultima-quando", `Última viagem · ${diaRelativo(ini)}, ${horaCurta(ini)}`);
  setText("ultima-detalhe", partes.join(" · "));
  const maxS = Math.max(1, ...res.serie.map((p) => p[0]));
  const pts = res.serie.map(([s, v]) => [(s / maxS) * 96, 30 - 0.28 * Math.max(0, Math.min(100, v))]);
  $("ultima-linha").setAttribute("d", pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`).join(" "));
  const pico = pts.reduce((m, p) => (p[1] < m[1] ? p : m), [-10, 99]);
  const c = $("ultima-pico");
  c.setAttribute("cx", n.alarme ? pico[0].toFixed(1) : "-10");
  c.setAttribute("cy", n.alarme ? pico[1].toFixed(1) : "-10");
}

// ---- Início: checagens reais antes de sair ----
const Checagem = {
  itens: {},
  definir(chave, estado, sub, tag = "") {
    const el = $(`chk-${chave}`);
    const chaveHtml = `${estado}|${tag}`;
    this.itens[chave] = estado;
    setText(`chk-${chave}-sub`, sub);
    if (el.dataset.k === chaveHtml) return;
    el.dataset.k = chaveHtml;
    if (estado === "espera") el.innerHTML = '<span class="pontos-espera" aria-label="verificando"><i></i><i></i><i></i></span>';
    else if (estado === "ok") el.innerHTML = '<svg class="chk-ok" viewBox="0 0 24 24" role="img" aria-label="ok"><circle cx="12" cy="12" r="12"/><path d="M7 12.5l3.2 3.2L17 9" pathLength="24"/></svg>';
    else el.innerHTML = `<span class="chk-tag${estado === "aviso" ? " aviso" : ""}">${esc(tag)}</span>`;
  },
  async atualizar() {
    // detector de rosto (MediaPipe): carrega ao abrir a página, de ./vendor/ ou da CDN
    if (S.visionErro) this.definir("detector", "aviso", "Não carregou. Veja o aviso no topo.", "Erro");
    else if (S.visionSrc === undefined) this.definir("detector", "espera", "Carregando");
    else this.definir("detector", "ok", S.visionSrc === 0 ? "Pronto, funciona sem internet" : "Pronto, carregado da internet");
    // câmera: permissão (quando o navegador informa)
    if (!navigator.mediaDevices?.getUserMedia) this.definir("camera", "aviso", "Indisponível: abra o app por https", "Ver");
    else {
      let st = null;
      try {
        st = (await navigator.permissions.query({ name: "camera" })).state;
      } catch {
        st = null; // o navegador não informa
      }
      if (st === "granted") this.definir("camera", "ok", "Permitida");
      else if (st === "denied") this.definir("camera", "aviso", "Bloqueada: libere nos ajustes do navegador", "Ver");
      else this.definir("camera", "info", "O navegador pede permissão ao iniciar", "No início");
    }
    // som do alarme
    if (S.volumePct === 0) this.definir("som", "aviso", "Volume do alarme em 0 %: o alarme não toca", "Sem som");
    else if (S.somTestado) this.definir("som", "ok", `Volume em ${S.volumePct} %, testado`);
    else this.definir("som", "info", `Volume em ${S.volumePct} %. Toque para testar`, "Testar");
    // tela ligada (Wake Lock)
    if ("wakeLock" in navigator) this.definir("tela", "ok", "O app pede para a tela não apagar");
    else this.definir("tela", "aviso", "Desligue o bloqueio automático da tela nos ajustes do celular", "Ajustar");
    // carregador (só onde o navegador informa a bateria)
    let carregando = null;
    try {
      if (navigator.getBattery) carregando = (await navigator.getBattery()).charging;
    } catch {
      carregando = null;
    }
    if (carregando) this.definir("carregador", "ok", "Carregando");
    else this.definir("carregador", "info", "Recomendado em viagens longas", "Opcional");
    const v = Object.values(this.itens);
    const avisos = v.filter((x) => x === "aviso").length;
    const st = $("chk-status");
    st.className = v.includes("espera") ? "" : avisos ? "aviso" : "ok";
    st.textContent = v.includes("espera") ? "Verificando…" : avisos ? `${plural(avisos, "item", "itens")} para ver` : "Tudo pronto";
  },
};

// ---- calibração guiada (painel com a câmera, guia e anel do tempo) ----
function setDica(tipo, titulo, texto) {
  const icone = { sem: "i-busca", centro: "i-seta", perto: "i-aproximar", longe: "i-aproximar", ok: "i-olho", feito: "i-check", aviso: "i-aviso", carregando: "i-busca" }[tipo];
  const ic = $("calib-dica-icone");
  trocarClasse(ic, ["ok", "cinza", "empurrar", "aproximar"], tipo === "ok" || tipo === "feito" ? "ok" : tipo === "carregando" ? "cinza" : tipo === "centro" ? "empurrar" : tipo === "perto" ? "aproximar" : "");
  const uso = $("calib-dica-uso");
  if (uso.getAttribute("href") !== `#${icone}`) uso.setAttribute("href", `#${icone}`);
  setText("calib-title", titulo);
  setText("calib-text", texto);
}
/** Onde está o rosto na imagem (como aparece na tela, espelhada): centro, tamanho e o que pedir. */
function guiaRosto() {
  const b = S.faceBox;
  if (!S.faceDetected || !b) return { tipo: "sem" };
  const dx = 1 - b.cx - 0.5;
  const dy = b.cy - 0.5;
  if (Math.abs(dx) > 0.14 || Math.abs(dy) > 0.16) return { tipo: "centro", dx, dy };
  if (b.h < 0.28) return { tipo: "perto" };
  if (b.h > 0.78) return { tipo: "longe" };
  return { tipo: "ok" };
}
/** Anel do tempo de calibração (0 a 100 % do contorno do guia); escondido em zero, senão a ponta arredondada
 *  aparece como um ponto no alto do guia. */
function anelTempo(id, offset) {
  const el = $(id);
  el.style.strokeDashoffset = String(offset);
  el.style.opacity = Number(offset) >= 99.9 ? "0" : "1";
}
function resetCalibUI() {
  clearTimeout(UI.autoAvanco);
  UI.calibPronta = false;
  S.calibPiscadas = 0;
  $("calib-feito").hidden = true;
  $("calib-pronto").hidden = true;
  $("calib-pronto").classList.remove("com-aviso");
  $("calib-nota").hidden = false;
  anelTempo("guia-anel", "100");
  $("guia-elipse").setAttribute("class", "guia-elipse");
  trocarClasse($("calib-chip"), ["ok"], "");
  setText("calib-chip-texto", "Procurando o rosto");
  setText("calib-seg", `${E.CFG.calib.neutralMs / 1000} s`);
  setText("calib-piscadas", "0");
  if (S.running) setDica("sem", "Mostre o rosto", "Olhe para a câmera, com luz de frente.");
  else setDica("carregando", "Preparando a câmera", "Carregando o detector de rosto. Na primeira vez, pode levar alguns segundos.");
}
function atualizarCalib(r) {
  if (UI.calibPronta) return;
  const p = Math.max(0, Math.min(1, r.calib?.progress ?? 0));
  const total = E.CFG.calib.neutralMs / 1000;
  anelTempo("guia-anel", (100 - p * 100).toFixed(1));
  setText("calib-seg", `${Math.max(0, Math.ceil(total * (1 - p)))} s`);
  setText("calib-piscadas", String(S.calibPiscadas));
  const g = guiaRosto();
  const alinhado = g.tipo === "ok";
  $("guia-elipse").setAttribute("class", `guia-elipse${alinhado ? " alinhado" : ""}`);
  trocarClasse($("calib-chip"), ["ok"], alinhado ? "ok" : "");
  setText("calib-chip-texto", g.tipo === "sem" ? "Procurando o rosto" : alinhado ? "Rosto no guia" : "Rosto fora do guia");
  if (g.tipo === "sem") setDica("sem", "Mostre o rosto", "Rosto não encontrado. Mostre o rosto inteiro, com luz de frente.");
  else if (g.tipo === "centro") {
    setDica("centro", "Coloque o rosto no centro", "Mova o celular até o rosto entrar no guia.");
    // a seta aponta para onde o rosto precisa ir (na imagem espelhada da tela)
    const ang = Math.abs(g.dx) >= Math.abs(g.dy) ? (g.dx > 0 ? 180 : 0) : g.dy > 0 ? -90 : 90;
    $("calib-dica-icone").querySelector("svg").style.rotate = `${ang}deg`;
  } else if (g.tipo === "perto") setDica("perto", "Aproxime o celular", "O rosto deve encher o guia.");
  else if (g.tipo === "longe") setDica("longe", "Afaste um pouco o celular", "O rosto inteiro deve caber no guia.");
  else setDica("ok", "Olhe para a frente", `Pisque normalmente por ${total} s.`);
  if (g.tipo !== "centro") $("calib-dica-icone").querySelector("svg").style.rotate = "";
}
function mostrarCalibPronta(avisosMotor) {
  const avisos = avisosMotor.map((a) => a.replace(/toque em Recalibrar e /i, ""));
  const b = S.monitor.baseline;
  // olhos meio fechados em repouso: a regra do olho usa o eyeBlink de repouso + 0,25, e a base sai "sonolenta"
  if (b && Number.isFinite(b.blend0) && b.blend0 >= 0.4) avisos.push("Os olhos pareceram meio fechados na calibração: abra bem os olhos, com luz de frente, e refaça.");
  UI.calibPronta = true;
  anelTempo("guia-anel", "0");
  $("calib-nota").hidden = true;
  $("calib-pronto").hidden = false;
  $("calib-pronto").classList.toggle("com-aviso", avisos.length > 0);
  setText("calib-seg", "Feito");
  clearTimeout(UI.autoAvanco);
  if (avisos.length) {
    $("calib-feito").hidden = true;
    $("guia-elipse").setAttribute("class", "guia-elipse aviso");
    setDica("aviso", "Calibração com aviso", avisos.join(" "));
    setText("btn-comecar", "Seguir assim");
    return; // com aviso, a pessoa decide: refazer ou seguir
  }
  $("calib-feito").hidden = false;
  trocarClasse($("calib-chip"), ["ok"], "ok");
  setText("calib-chip-texto", "Calibrado");
  setDica("feito", "Calibrado", "Abertura dos olhos e posição da cabeça medidas. A viagem começa em 3 s.");
  setText("btn-comecar", "Começar viagem");
  UI.autoAvanco = setTimeout(() => {
    if (UI.tela === "calibracao" && S.running) ir("viagem");
  }, 3000);
}

// ---- em viagem ----
function mostrarFicha(texto, icone) {
  const f = $("ficha");
  setText("ficha-texto", texto);
  $("ficha-uso").setAttribute("href", `#${icone}`);
  f.hidden = true;
  void f.offsetWidth; // reinicia a animação
  f.hidden = false;
  clearTimeout(UI.fichaTimer);
  UI.fichaTimer = setTimeout(() => (f.hidden = true), 1150);
}
/** Eventos do motor que a tela mostra na hora (a ficha que sobe do mostrador e a faixa de Atenção). */
function aoEvento(e) {
  if (S.modo === "poster") return Poster.aoEvento(e);
  if (e.type === "aviso") UI.faixaAte = performance.now() + 4500;
  if (UI.tela !== "viagem") return;
  const P = E.CFG.score;
  const fichas = { fechamento: [`+${P.longClosure.points} Fechamento longo`, "i-olhos"], bocejo: [`+${P.yawn.points} Bocejo`, "i-boca"], queda: [`+${P.head.points} Cabeça caiu`, "i-cabeca"], cabeceio: [`+${P.head.points} Cabeceio`, "i-cabeca"] };
  if (fichas[e.type]) mostrarFicha(...fichas[e.type]);
}
function trocarPlaca(id) {
  if (UI.placa === id) return;
  UI.placa = id;
  const p = $("placa");
  $("placa-uso").setAttribute("href", `#${id}`);
  p.classList.toggle("verde", id === "p-verde");
  p.classList.remove("pulso");
  void p.getBoundingClientRect();
  p.classList.add("pulso");
}
function faixa(tipo, texto, icone = "i-relogio") {
  const f = $("faixa");
  if (!tipo) {
    f.hidden = true;
    f.dataset.tipo = "";
    return;
  }
  if (f.dataset.tipo !== tipo) {
    f.dataset.tipo = tipo;
    f.className = `faixa ${tipo}`;
    f.hidden = false;
  }
  $("faixa-uso").setAttribute("href", `#${icone}`);
  setText("faixa-texto", texto);
}
function atualizarViagem(r) {
  const agora = performance.now();
  const sc = r.score;
  const calibrando = r.phase !== "ativo" || !sc;
  const semRostoMs = S.faceLostSince === null ? 0 : agora - S.faceLostSince;
  const semLeitura = !calibrando && semRostoMs >= SEM_LEITURA_MS && !sc.forced;
  const tela = $("tela-viagem");
  trocarClasse(tela, ["e0", "e1", "e2", "sl"], calibrando || semLeitura ? "sl" : `e${sc.state}`);
  // número, placa e nome do estado (o número é arredondado para baixo: 69,6 aparece como 69, em Atenção)
  setText("level-num", calibrando || semLeitura ? "--" : String(Math.floor(sc.level)));
  trocarPlaca(calibrando || semLeitura ? "p-cinza" : PLACAS[sc.state]);
  const nome = calibrando ? "Calibrando" : semLeitura ? "Sem leitura" : STATE_TITLES[sc.state];
  setText("status-title", nome);
  $("mostrador").setAttribute("aria-label", calibrando ? "Calibrando" : semLeitura ? "Sem leitura. Toque para ver os sinais" : `Nível ${Math.floor(sc.level)}, ${nome}. Toque para ver por quê`);
  desenharSegs(UI.segsViagem, calibrando ? [0, 0, 0, 0, 0] : familias(sc));
  // frase (uma só)
  const kind = S.monitor.forcedEnd?.kind === "cabeça caída" ? "cabeca" : "micro";
  let frase;
  if (calibrando) frase = "Olhe para a frente por alguns segundos.";
  else if (semLeitura) frase = "Não vejo seus olhos. Ajuste o celular.";
  else if (sc.state === 2) frase = r.alarm?.snoozed ? "Som pausado. Pare assim que puder." : "Pare assim que puder.";
  else if (sc.state === 1 && sc.postForcedMs > 0) frase = kind === "cabeca" ? "Sua cabeça caiu. Atenção redobrada." : "Você fechou os olhos. Atenção redobrada.";
  else if (sc.state === 1) frase = "Sinais de cansaço. Planeje uma pausa.";
  else frase = sc.level < 15 ? "Tudo certo. Boa viagem." : "Sinais leves, ainda no verde.";
  setText("frase", frase);
  // faixa do topo: alarme pausado > procurando o rosto > entrada em Atenção > depois do microssono
  if (!calibrando && sc.state === 2 && r.alarm?.snoozed) {
    const resta = (S.monitor.alarm.snoozeUntil - agora) / 1000;
    faixa("pausa", `Alarme pausado. Volta em ${mmss(resta)} se o nível seguir alto.`);
    $("faixa-anel-arco").style.strokeDashoffset = (100 - (100 * Math.max(0, resta)) / (E.CFG.alarm.snoozeMs / 1000)).toFixed(1);
  } else if (!calibrando && !S.faceDetected && semRostoMs > 1500 && !semLeitura) faixa("procurando", "Procurando o rosto…", "i-busca");
  else if (!calibrando && !semLeitura && sc.state === 1 && UI.faixaAte > agora) faixa("atencao", "Sinais de cansaço. Planeje uma pausa.", "i-aviso");
  else if (!calibrando && !semLeitura && sc.state < 2 && sc.postForcedMs > 0) faixa("pos", `Depois ${kind === "cabeca" ? "da cabeça caída" : "do microssono"}, atenção por ${mmss(sc.postForcedMs / 1000)}.`);
  else faixa(null);
  // sinais (fichas que abrem o "Por quê?")
  let htmlSinais = "";
  if (!calibrando && !semLeitura) {
    const f = familias(sc);
    const pts = inteirosQueSomam(f, Math.floor(sc.level));
    const icones = ["i-olhos", "i-boca", "i-cabeca", "i-relogio"];
    htmlSinais = pts.slice(0, 4).map((n, i) => (n > 0 ? `<button type="button">${usar(icones[i])}+${n}</button>` : "")).join("");
  }
  if (UI.htmlSinais !== htmlSinais) {
    UI.htmlSinais = htmlSinais;
    $("sinais").innerHTML = htmlSinais;
  }
  // segunda fase da calibração: as primeiras piscadas da viagem viram a referência de duração
  const fase2 = !calibrando && !!r.calib;
  $("blink-calib").classList.toggle("hidden", !fase2);
  if (fase2) setText("blink-calib-text", `${r.calib.blinks} de ${r.calib.need}`);
  // recalibrar fica bloqueado no vermelho (e enquanto calibra)
  const bloq = calibrando || sc.state === 2;
  const bc = $("btn-calib");
  if (bc.getAttribute("aria-disabled") !== String(bloq)) {
    bc.setAttribute("aria-disabled", String(bloq));
    setText("btn-calib-rot", bloq && !calibrando ? "Bloqueado no vermelho" : "Segure: recalibrar");
  }
  // tempo de viagem e o "Por quê?" aberto
  if (S.viagem) setText("viagem-tempo", horasMin(Date.now() - S.viagem.inicio));
  if ($("folha-porque").open) renderPorque(r);
}
function atualizarSomChip() {
  const chip = $("chip-som");
  let txt;
  if (S.volumePct === 0) txt = "Sem som";
  else if (S.running && !Sound.ready) txt = "Toque: ligar som";
  else txt = `Som ${S.volumePct} %`;
  setText("chip-som-texto", txt);
  chip.classList.toggle("aviso", S.volumePct === 0 || (S.running && !Sound.ready));
}
function mostrarCamada(id, mostrar, foco) {
  const el = $(id);
  if (el.classList.contains("hidden") !== mostrar) return false;
  el.classList.toggle("hidden", !mostrar);
  if (mostrar) {
    fecharDialogos();
    setTimeout(() => $(foco)?.focus({ preventScroll: true }), 0);
  }
  return true;
}
function atualizarAlarme(r) {
  const sc = r.score;
  const vermelho = S.running && S.modo === "viagem" && r.phase === "ativo" && !!sc && sc.state === 2;
  const micro = vermelho && !!sc.forced;
  // depois do "estou acordado", a tela volta para a viagem (ainda vermelha, com o som pausado); se o nível seguir
  // alto quando os 30 s acabarem, o alarme volta. O vermelho forçado (microssono, cabeça caída) sempre mostra.
  const nivel = vermelho && !sc.forced && !r.alarm?.snoozed;
  mostrarCamada("alarme-micro", micro, "micro-titulo");
  mostrarCamada("alarme", nivel, "btn-ack");
  UI.microVisivel = micro;
  UI.alarmeVisivel = nivel;
  if (micro) {
    const cabeca = /cabeça/.test(sc.forced);
    setText("micro-motivo", cabeca ? cap(sc.forced.replace(/ há [\d,]+ s/, "")) : "Olhos fechados por mais de 1 s");
    setText("micro-nota", cabeca ? "O som não pode ser pausado. Ele para quando a cabeça volta e os olhos abrem." : "O som não pode ser pausado. Ele para quando os olhos abrem.");
  }
  if (nivel) {
    const partes = partesPorque(r).filter((p) => !p.cinza).slice(0, 3);
    setHTML($("alarme-motivos"), partes.map((p) => `<li>${usar(p.icone)}${esc(p.rotulo.replace(/^[^:]+: /, ""))}</li>`).join(""));
    setText("alarme-som", S.volumePct === 0 ? "Sem som: volume do alarme em 0 %." : !Sound.ready ? "Toque na tela para ligar o som." : `Nível ${Math.floor(sc.level)} de 100. Procure um lugar seguro para parar.`);
  }
}
function estouAcordado() {
  Sound.unlock();
  S.monitor.acknowledge(performance.now());
  addLog({ id: `a${Date.now()}`, type: "info", text: "“Estou acordado”: som pausado por 30 s" });
}

function updateUI(r) {
  const face = S.faceDetected;
  setBadge(`${S.delegate} · ${S.fps} FPS`, S.fps >= 15 ? "ok" : "warn");
  setText("hud-face-tec", face ? "detectado" : "não detectado");
  // anel da câmera (círculo da viagem): cor do estado; tracejado e cinza sem rosto
  const cam = $("camera");
  cam.classList.toggle("sem-rosto", !face);
  const st = r.score && r.phase === "ativo" ? r.score.state : null;
  cam.style.setProperty("--cor-rosto", st === null ? "var(--cinza)" : ["var(--verde)", "var(--ambar)", "var(--vermelho)"][st]);
  $("slot-viagem").setAttribute("aria-label", face ? "Câmera vendo o rosto" : "Câmera sem rosto");
  atualizarSomChip();
  if (S.modo === "poster") Poster.atualizar(r);
  else {
    if (r.phase === "calibrando") atualizarCalib(r);
    atualizarViagem(r);
    atualizarAlarme(r);
  }

  // ---- modelos experimentais (Detalhes técnicos) ----
  if (S.temporal) {
    const p = S.temporal.probs;
    const names = ["alerta", "pouca vigilância", "sonolento"];
    const tag = r.score?.mlCounted ? '<span class="tag">+10 no nível</span>' : r.score?.mlEligible ? '<span class="tag">"sonolento", mas não soma (sinais visíveis &lt; 40)</span>' : "";
    $("ml-lstm").innerHTML = `LSTM de piscadas: <b>${names[S.temporal.cls]}</b> (${names.map((n, i) => `${n} ${Math.round(p[i] * 100)} %`).join(" · ")}; ${S.temporal.nBlinks}/${S.blinkWindow.size} piscadas${S.temporal.partial ? ", janela parcial" : ""})${tag}`;
  } else {
    $("ml-lstm").textContent = S.tmpModel ? `LSTM de piscadas: aguardando piscadas (${S.normalizer.ready ? `${S.blinkWindow.length}/${S.blinkWindow.size}` : `calibração ${S.normalizer.samples.length}/${S.normalizer.nCalib}`})` : "LSTM de piscadas: modelo ausente.";
  }
  $("ml-cnn").textContent = S.eyeModel ? `CNN dos olhos, p(aberto): E ${br(S.eyeProb.left, 2)} · D ${br(S.eyeProb.right, 2)}` : "CNN dos olhos: modelo ausente.";
  if (UI.tela === "tecnico") updateTech(r);
}

// ---- Como funciona: demonstração com as MESMAS classes e números do motor (E.DecayingEvents, E.LevelController,
// E.CFG), com o relógio 10 vezes mais rápido ----
const Como = {
  timer: 0,
  t: 0,
  ev: null,
  lv: null,
  segs: null,
  montar() {
    const P = E.CFG.score;
    const L = E.CFG.levels;
    const min = (ms) => `${br(ms / 60000, 0)} min`;
    this.segs = criarSegs($("como-segs"), 4);
    const botoes = [
      ["olhos", `Fechar os olhos ${br(0.7, 1)} s`, `+${P.longClosure.points}, some em ${min(P.longClosure.decayMs)}`],
      ["boca", "Bocejar", `+${P.yawn.points}, some em ${min(P.yawn.decayMs)}`],
      ["cabeca", "A cabeça cai", `+${P.head.points}, some em ${min(P.head.decayMs)}`],
    ];
    $("como-botoes").innerHTML = botoes.map(([k, a, b]) => `<button type="button" data-sinal="${k}"><strong>${a}</strong><small>${b}</small></button>`).join("");
    $("como-botoes").onclick = (ev) => {
      const b = ev.target.closest("[data-sinal]");
      if (b) this.somar(b.dataset.sinal);
    };
    setText("como-faixa-0", `Alerta: abaixo de ${L.attention}`);
    setText("como-faixa-1", `Atenção: de ${L.attention} a ${L.drowsy - 1}`);
    setText("como-faixa-2", `Sonolento: ${L.drowsy} ou mais`);
    setText("como-nota", `Olhos fechados por ${br(E.CFG.eyes.microsleepMs / 1000, 0)} s ou mais são um microssono: o alarme toca na hora, sem esperar a soma. Aqui o relógio anda 10 vezes mais rápido.`);
    renderNumerosNivel();
    this.zerar();
  },
  zerar() {
    const P = E.CFG.score;
    this.t = 0;
    this.ev = {
      olhos: new E.DecayingEvents({ points: P.longClosure.points, decayMs: P.longClosure.decayMs, cap: P.closuresCap }),
      boca: new E.DecayingEvents({ points: P.yawn.points, decayMs: P.yawn.decayMs, cap: P.yawn.cap }),
      cabeca: new E.DecayingEvents({ points: P.head.points, decayMs: P.head.decayMs, cap: P.head.cap }),
    };
    this.lv = new E.LevelController();
    this.render();
  },
  somar(k) {
    this.ev[k].add(this.t);
    this.render();
  },
  passo() {
    this.t += 1000; // 100 ms reais = 1 s de viagem
    this.render();
  },
  render() {
    const v = [this.ev.olhos.points(this.t), this.ev.boca.points(this.t), this.ev.cabeca.points(this.t)];
    const raw = Math.min(100, v[0] + v[1] + v[2]);
    const r = this.lv.update(this.t, raw);
    trocarClasse($("tela-como"), ["e0", "e1", "e2"], `e${r.state}`);
    desenharSegs(this.segs, [...v, Math.max(0, r.level - raw)]);
    setText("como-nivel", String(Math.floor(r.level)));
    setText("como-estado", STATE_TITLES[r.state]);
    $("como-leitura").setAttribute("aria-label", `Nível ${Math.floor(r.level)}, ${STATE_TITLES[r.state]}`);
  },
  iniciar() {
    if (!this.timer) this.timer = setInterval(() => this.passo(), 100);
  },
  parar() {
    clearInterval(this.timer);
    this.timer = 0;
  },
};
/** "Os números do nível": montado a partir de E.CFG (os mesmos números que o detector usa). */
function renderNumerosNivel() {
  const C = E.CFG;
  const L = C.levels;
  const P = C.score;
  const seg = (ms) => `${br(ms / 1000, ms % 1000 ? 1 : 0)} s`;
  const min = (ms) => `${br(ms / 60000, 0)} min`;
  const pct = (x) => `${br(x * 100, 0)} %`;
  const li = (a, b) => `<li><span>${a}</span><b>${b}</b></li>`;
  $("como-numeros").innerHTML = `
    <ul>
      ${li(`Olhos fechados de ${seg(C.eyes.longMs)} a ${seg(C.eyes.microsleepMs)}, cada vez; some em ${min(P.longClosure.decayMs)}`, `+${P.longClosure.points}`)}
      ${li(`PERCLOS: parte do último minuto com os olhos fechados, só em fechamentos de ${seg(C.perclos.minEpisodeMs)} ou mais (${P.perclos.points} pontos em ${pct(P.perclos.at)})`, `até ${P.perclos.cap}`)}
      ${li(`Bocejo: boca bem aberta por ${seg(C.yawn.minMs)} ou mais; some em ${min(P.yawn.decayMs)}`, `+${P.yawn.points}`)}
      ${li(`Cabeça caindo ou cabeceio, com os olhos fechados; some em ${min(P.head.decayMs)}`, `+${P.head.points}`)}
      ${li(`Piscadas de ${br(P.blinkDur.from, 2)} a ${br(P.blinkDur.at, 0)} vezes mais longas que na calibração`, `até ${P.blinkDur.cap}`)}
      ${li(`Modelo LSTM, experimental: só soma com o nível já em Atenção (${P.ml.minBase} ou mais)`, `+${P.ml.points}`)}
    </ul>
    <p>Tetos: fechamentos ${P.closuresCap}, bocejos ${P.yawn.cap}, cabeça ${P.head.cap}: nenhum sinal sozinho chega a ${L.drowsy}. Cabeça baixa de olhos abertos é distração e não soma. Microssono (olhos fechados por ${seg(C.eyes.microsleepMs)} ou mais) e cabeça caída de olhos fechados dão vermelho na hora; depois, o app fica pelo menos ${min(P.postForced.holdMs)} em Atenção.</p>
    <p>O nível desce no máximo ${P.maxFallPerS} pontos por segundo. Para a cor não ficar trocando à toa, o amarelo só volta ao verde abaixo de ${L.attentionExit}, e o vermelho só sai abaixo de ${L.drowsyExit}. "Segure: estou acordado" pausa o som por ${seg(C.alarm.snoozeMs)}, nunca num microssono.</p>
    <p>De onde vêm: da literatura (PERCLOS e microssono de 1 s), de testes no DMD (limiar do olho e bocejo de 1,5 s) e de regras de projeto (pontos, tetos, 40 e 70). Num teste com 300 combinações, cada peso multiplicado por 0,7 a 1,3 e os limiares mudados em até 10 pontos, o vermelho com motoristas acordados (direção real do DMD) ficou entre 1,7 % e 4,9 % do tempo em 90 % delas; com os valores do app, 3,3 %.</p>`;
}

// ---- Primeiro uso: 3 passos (o que o app vê, onde prender o celular, teste do alarme) ----
const PrimeiroUso = {
  n: 0,
  tocou: false,
  timers: [],
  medidorRaf: 0,
  montar() {
    const segs = Array.from({ length: 12 }, (_, i) => `<span style="height:${(8 + i * 1.7).toFixed(1)}px"></span>`).join("");
    $("pu-medidor").innerHTML = segs;
    $("pu-continuar").onclick = () => this.passo(this.n + 1);
    $("pu-voltar").onclick = () => this.passo(this.n - 1);
    $("pu-testar").onclick = () => this.testar();
    $("pu-concluir").onclick = () => this.concluir();
    $("pu-pular").onclick = () => this.concluir();
  },
  concluir() {
    gravarLocal(INTRO_KEY, "1");
    this.parar();
    UI.historico = [];
    ir("inicio", { historico: false });
  },
  parar() {
    this.timers.forEach((t) => clearTimeout(t));
    this.timers = [];
    cancelAnimationFrame(this.medidorRaf);
  },
  passo(n) {
    this.n = Math.max(0, Math.min(2, n));
    this.parar();
    for (let i = 0; i < 3; i++) $(`pu-passo-${i}`).hidden = i !== this.n;
    setText("pu-num", String(this.n + 1));
    $("pu-progresso").setAttribute("aria-valuenow", String(this.n + 1));
    $("pu-progresso").setAttribute("aria-label", `Passo ${this.n + 1} de 3`);
    [...$("pu-progresso").children].forEach((s, i) => s.classList.toggle("feito", i <= this.n));
    $("pu-continuar").hidden = this.n === 2;
    $("pu-voltar").hidden = this.n === 0;
    $("pu-testar").hidden = this.n !== 2;
    $("pu-concluir").hidden = !(this.n === 2 && this.tocou);
    $("tela-primeiro").setAttribute("aria-labelledby", `pu-titulo-${this.n}`);
    if (this.n === 0) this.desenharRosto();
  },
  /** Rosto ilustrado: a silhueta some e os pontos (contorno, sobrancelhas, nariz, olhos e boca) aparecem; o contador
   *  sobe até 478, o número de marcos do Face Landmarker. */
  desenharRosto() {
    const elipse = (cx, cy, rx, ry, n, a0, passo, cls = "pu-pt", r = 2.4) =>
      Array.from({ length: n }, (_, i) => {
        const a = (i / n) * Math.PI * 2;
        return `<circle class="${cls}" cx="${(cx + rx * Math.cos(a)).toFixed(1)}" cy="${(cy + ry * Math.sin(a)).toFixed(1)}" r="${r}" style="animation-delay:${a0 + i * passo}ms"/>`;
      }).join("");
    const tracos = [[100, 120], [109, 116], [118, 114], [127, 115], [136, 118], [164, 118], [173, 115], [182, 114], [191, 116], [200, 120], [150, 146], [150, 160], [150, 174], [150, 188], [140, 194], [160, 194]]
      .map(([x, y], i) => `<circle class="pu-pt" cx="${x}" cy="${y}" r="2.4" style="animation-delay:${500 + i * 18}ms"/>`)
      .join("");
    const svg = $("pu-rosto");
    svg.innerHTML = `
      <g class="pu-cantos"><path d="M44 36 V20 H60"/><path d="M240 20 H256 V36"/><path d="M44 300 V316 H60"/><path d="M240 316 H256 V300"/></g>
      <g class="pu-silhueta" id="pu-silhueta">
        <ellipse class="pu-face" cx="150" cy="165" rx="92" ry="118"/>
        <path class="pu-sobrancelha" d="M100 120 Q118 112 136 118M164 118 Q182 112 200 120"/>
        <path class="pu-olho-img" d="M101 141 Q118 131 135 141 Q118 151 101 141 Z M165 141 Q182 131 199 141 Q182 151 165 141 Z"/>
        <path class="pu-boca-traco" d="M126 222 Q150 232 174 222"/>
      </g>
      <g id="pu-pontos-g" hidden>
        ${elipse(150, 165, 92, 118, 32, 250, 14)}
        ${tracos}
        <g class="pu-piscar">${elipse(118, 141, 17, 8, 8, 650, 20, "pu-pt olho", 3)}</g>
        <g class="pu-piscar">${elipse(182, 141, 17, 8, 8, 650, 20, "pu-pt olho", 3)}</g>
        ${elipse(150, 224, 26, 9, 10, 800, 16, "pu-pt", 2.6)}
        <path class="pu-ligacao" d="M8 96 H58 L100 136" pathLength="100" style="animation-delay:1000ms"/>
        <path class="pu-ligacao" d="M292 250 H236 L178 224" pathLength="100" style="animation-delay:1150ms"/>
        <path class="pu-ligacao" d="M292 60 H246 L222 78" pathLength="100" style="animation-delay:1300ms"/>
        <text class="pu-etiqueta" x="8" y="90" style="animation-delay:1200ms">OLHOS</text>
        <text class="pu-etiqueta" x="292" y="244" text-anchor="end" style="animation-delay:1350ms">BOCA</text>
        <text class="pu-etiqueta" x="292" y="54" text-anchor="end" style="animation-delay:1500ms">CABEÇA</text>
      </g>
      <rect class="pu-varredura" id="pu-varredura" x="44" y="20" width="212" height="3" hidden/>`;
    setText("pu-pontos", "0");
    this.timers.push(
      setTimeout(() => {
        $("pu-silhueta").classList.add("some");
        $("pu-pontos-g").removeAttribute("hidden"); // elementos SVG não têm a propriedade .hidden
        $("pu-varredura").removeAttribute("hidden");
      }, 700),
      setTimeout(() => {
        const t0 = performance.now();
        const contar = () => {
          const p = Math.min(1, (performance.now() - t0) / 900);
          setText("pu-pontos", String(Math.round(478 * (1 - Math.pow(1 - p, 3)))));
          if (p < 1) this.medidorRaf = requestAnimationFrame(contar);
        };
        contar();
      }, 900),
    );
  },
  /** Toca o alarme de teste (o mesmo do app, no volume escolhido) e mostra o nível real do som na saída. */
  testar() {
    testSound();
    this.parar();
    $("pu-sino").classList.add("tocando");
    setText("pu-testar", "Tocando…");
    const segs = [...$("pu-medidor").children];
    const buf = new Float32Array(256);
    const t0 = performance.now();
    const medir = () => {
      let n = 0;
      const an = Sound.analyser;
      if (an) {
        an.getFloatTimeDomainData(buf);
        let s = 0;
        for (const x of buf) s += x * x;
        const rms = Math.sqrt(s / buf.length);
        n = rms > 0 ? Math.max(0, Math.min(12, Math.round((20 * Math.log10(rms) + 60) / 5))) : 0; // -60 dB a 0 dB em 12 passos
      }
      segs.forEach((el, i) => (el.className = i < n ? (i >= 10 ? "r" : i >= 7 ? "a" : "v") : ""));
      $("pu-medidor").setAttribute("aria-valuenow", String(n));
      if (performance.now() - t0 < 2700) this.medidorRaf = requestAnimationFrame(medir);
      else {
        segs.forEach((el) => (el.className = ""));
        $("pu-sino").classList.remove("tocando");
        this.tocou = true;
        setText("pu-testar", "Tocar de novo");
        $("pu-concluir").hidden = false;
      }
    };
    medir();
  },
};

// ---- Modo pôster: notebook ou tablet na sessão de pôsteres. Cada visitante senta, o app calibra em 10 s e mostra o
// nível reagindo; quando a pessoa sai (8 s sem rosto), volta a esperar o próximo. Nada é registrado como viagem. ----
const Poster = {
  fase: "parado", // parado | carregando | esperando | calibrando | ativo
  rostoDesde: null,
  semRostoDesde: null,
  eventos: [],
  segs: null,
  entrar() {
    if (S.running && S.modo === "viagem") return;
    this.eventos = [];
    this.renderEventos();
    this.fase = "carregando";
    this.chip("Carregando o detector", false);
    anelTempo("poster-anel", "100");
    startCamera("poster");
  },
  sair() {
    if (S.modo === "poster" || (S.running && S.modo !== "viagem")) pararCamera();
    S.fichaInicio = (S.fichaInicio || 0) + 1; // um carregamento em andamento desiste
    this.fase = "parado";
  },
  /** Volta a esperar um rosto: monitor novo (sem calibração) e caminho do LSTM zerado. */
  esperar() {
    S.monitor = new E.DrowsinessMonitor();
    resetarCaminhoLSTM();
    S.faceLostSince = null;
    this.fase = "esperando";
    this.rostoDesde = null;
    this.semRostoDesde = null;
    anelTempo("poster-anel", "100");
  },
  recomecar() {
    this.eventos = [];
    this.renderEventos();
    if (S.running) this.esperar();
    else this.entrar();
  },
  calibrado() {
    this.fase = "ativo";
    anelTempo("poster-anel", "100");
    this.registrar("Calibrado: pode começar", "var(--verde)");
  },
  aoEvento(e) {
    const P = E.CFG.score;
    const txt = { fechamento: [`Fechamento longo: +${P.longClosure.points}`, "var(--ambar)"], bocejo: [`Bocejo: +${P.yawn.points}`, "var(--ambar)"], queda: [`Cabeça caiu: +${P.head.points}`, "var(--ambar)"], cabeceio: [`Cabeceio: +${P.head.points}`, "var(--ambar)"], microssono: ["Microssono: alarme na hora", "var(--vermelho)"], cabeca_baixa: ["Cabeça baixa de olhos abertos: 0", "var(--cinza)"] }[e.type];
    if (txt) this.registrar(...txt);
  },
  registrar(texto, cor) {
    this.eventos.unshift({ texto, cor, em: Date.now() });
    this.eventos = this.eventos.slice(0, 7);
    this.renderEventos();
  },
  renderEventos() {
    const agora = Date.now();
    $("poster-sem-eventos").hidden = this.eventos.length > 0;
    setHTML($("poster-linha-tempo"), this.eventos.map((e) => `<li><i style="background:${e.cor}"></i><span>${esc(e.texto)}</span><small>há ${Math.max(0, Math.round((agora - e.em) / 1000))} s</small></li>`).join(""));
  },
  chip(texto, ok) {
    setText("poster-chip-texto", texto);
    trocarClasse($("poster-chip"), ["ok"], ok ? "ok" : "");
  },
  atualizar(r) {
    const agora = performance.now();
    const face = S.faceDetected;
    if (this.fase === "esperando") {
      if (face) {
        this.rostoDesde ??= agora;
        if (agora - this.rostoDesde > 1000) {
          S.monitor.startCalibration(agora);
          S.calibPiscadas = 0;
          this.fase = "calibrando";
        }
      } else this.rostoDesde = null;
    } else if (this.fase === "calibrando" || this.fase === "ativo") {
      if (!face) {
        this.semRostoDesde ??= agora;
        if (agora - this.semRostoDesde > 8000) {
          this.esperar();
          this.registrar("A pessoa saiu: esperando o próximo", "var(--cinza)");
        }
      } else this.semRostoDesde = null;
    }
    const sc = r.score;
    const ativo = this.fase === "ativo" && r.phase === "ativo" && !!sc;
    if (this.fase === "carregando") this.chip("Carregando o detector", false);
    else if (this.fase === "esperando") this.chip(face ? "Rosto encontrado: começando" : "Sente-se de frente para a câmera", face);
    else if (this.fase === "calibrando") {
      const p = Math.max(0, Math.min(1, r.calib?.progress ?? 0));
      anelTempo("poster-anel", (100 - p * 100).toFixed(1));
      this.chip(`Calibrando: olhe para a câmera (${Math.max(0, Math.ceil(10 * (1 - p)))} s)`, face);
    } else this.chip(face ? "Rosto encontrado · 478 pontos" : "Procurando o rosto", face);
    trocarClasse($("tela-poster"), ["e0", "e1", "e2"], ativo ? `e${sc.state}` : "");
    setText("poster-nivel", ativo ? String(Math.floor(sc.level)) : "--");
    setText("poster-estado", ativo ? STATE_TITLES[sc.state] : this.fase === "calibrando" ? "Calibrando" : "Aguardando");
    $("poster-leitura").setAttribute("aria-label", ativo ? `Nível ${Math.floor(sc.level)}, ${STATE_TITLES[sc.state]}` : "Aguardando um rosto");
    desenharSegs(this.segs, ativo ? familias(sc) : [0, 0, 0, 0, 0]);
    $("poster-guia").classList.toggle("oculto", this.fase === "ativo" || this.fase === "carregando");
    setText("poster-porque-titulo", ativo ? `Por que ${Math.floor(sc.level)}?` : "Por quê?");
    const partes = ativo ? partesPorque(r) : [];
    const vazio = ativo ? "Nenhum sinal de cansaço agora." : "O nível começa depois da calibração.";
    setHTML($("poster-porque"), partes.length ? partes.map((p) => `<div class="linha-pt"><span>${esc(p.rotulo)}</span><b>+${p.pts}</b></div>`).join("") : `<p class="sec">${vazio}</p>`);
    const vermelho = ativo && sc.state === 2;
    $("poster-alarme").hidden = !vermelho;
    if (vermelho) setText("poster-alarme-texto", sc.forced ? (/cabeça/.test(sc.forced) ? "ACORDE! A cabeça caiu" : "ACORDE! Olhos fechados por 1 s") : "PARE E DESCANSE");
    if (this.eventos.length) this.renderEventos();
  },
};

// ---- Detalhes técnicos ----
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
window.__ui = { ir, abrirSom, abrirPorque, encerrar: () => encerrarViagem(), estado: () => ({ tela: UI.tela, alarme: UI.alarmeVisivel || UI.microVisivel, historico: [...UI.historico] }) }; // inspeção em testes automatizados

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
/** Pede para a tela não apagar enquanto a câmera estiver ligada (o pedido cai quando a página some; volta ao
 *  reaparecer, em "visibilitychange"). */
async function requestWakeLock() {
  if (!("wakeLock" in navigator)) return false;
  try {
    S.wakeLock = await navigator.wakeLock.request("screen");
    S.wakeWanted = true;
    S.wakeLock.addEventListener("release", () => {
      S.wakeLock = null;
    });
    return true;
  } catch (e) {
    console.warn("Wake Lock", e.message);
    return false;
  }
}

// ---------------------------------------------------------------------------
// Inicialização
// ---------------------------------------------------------------------------
function init() {
  // Dependências começam a carregar já; os erros aparecem no "Iniciar viagem" (MediaPipe) ou viram modo
  // degradado (TensorFlow.js), por isso as promessas não ficam sem tratamento aqui.
  S.visionReady = loadVision();
  S.visionReady
    .then(
      (i) => {
        S.visionSrc = i; // 0 = ./vendor/ (funciona sem internet), 1 = CDN
      },
      () => {
        S.visionErro = true;
      },
    )
    .finally(() => {
      if (UI.tela === "inicio") Checagem.atualizar();
    });
  S.tfReady = loadTfjs();
  S.tfReady.catch(() => {});
  showDeviceInfo();
  initVolume(); // antes de qualquer som: ?volume= / valor salvo / padrão 20 %
  initTema();
  const params = new URLSearchParams(location.search);
  definirPesquisa(params.get("pesquisa") === "1" || lerLocal(PESQUISA_KEY) === "1", false);
  definirVibrar(lerLocal(VIBRAR_KEY) !== "0", false);
  $("linha-vibrar").hidden = !("vibrate" in navigator);
  for (const n of TELAS_COM_CAMERA) $(`tela-${n}`).classList.add("com-camera");
  UI.segsViagem = criarSegs($("dial-segs"), 5);
  marcaDial($("marca-40"), E.CFG.levels.attention);
  marcaDial($("marca-70"), E.CFG.levels.drowsy);
  Poster.segs = criarSegs($("poster-segs"), 5);
  Como.montar();
  PrimeiroUso.montar();
  setText("versao-app", APP_VERSION);
  renderUltima(carregarResumo());
  // Primeiro uso só na primeira visita (ou com ?intro=1); os testes automáticos (?autostart=1) pulam
  const intro = params.get("intro") === "1" || (lerLocal(INTRO_KEY) !== "1" && params.get("autostart") !== "1");
  ir(intro ? "primeiro" : "inicio", { historico: false });

  // O desbloqueio do áudio precisa acontecer DENTRO do toque, antes de qualquer await.
  $("btn-start").onclick = () => {
    Sound.unlock();
    iniciarViagem();
  };
  for (const id of ["btn-sound", "btn-testar-ajustes", "chk-som-btn"]) $(id).onclick = testSound;
  const recalibrar = () => {
    Sound.unlock();
    startCalibration();
  };
  segurar($("btn-calib"), recalibrar);
  $("btn-refazer").onclick = recalibrar;
  $("btn-comecar").onclick = () => {
    clearTimeout(UI.autoAvanco);
    ir("viagem");
  };
  $("btn-cancelar").onclick = cancelarViagem;
  $("btn-som").onclick = abrirSom;
  $("chip-som").onclick = () => {
    Sound.unlock();
    abrirSom();
  };
  $("mostrador").onclick = abrirPorque;
  $("sinais").onclick = (ev) => {
    if (ev.target.closest("button")) abrirPorque();
  };
  segurar($("btn-encerrar"), encerrarViagem);
  segurar($("btn-ack"), estouAcordado);
  $("btn-encerrar-alarme").onclick = encerrarViagem;
  $("btn-nova").onclick = () => ir("inicio");
  $("btn-exportar-resumo").onclick = () => {
    if (S.eventLog.length) exportSession();
    else notice("Nada para exportar ainda: a sessão começa no Iniciar viagem.");
  };
  $("btn-bench").onclick = startBenchmark;
  $("btn-export").onclick = exportSession;
  $("btn-full").onclick = toggleFullscreen;
  $("btn-poster").onclick = () => {
    Sound.unlock();
    ir("poster");
  };
  $("poster-sair").onclick = () => ir("ajustes", { historico: false });
  $("poster-recomecar").onclick = () => {
    Sound.unlock();
    Poster.recomecar();
  };
  $("btn-versao").onclick = tocarVersao;
  $("btn-intro").onclick = () => ir("primeiro");
  $("btn-pesquisa").onclick = () => definirPesquisa(!S.pesquisa);
  $("btn-vibrar").onclick = () => definirVibrar(!S.vibrarOn);
  $("como-zerar").onclick = () => Como.zerar();
  $("notice").onclick = () => notice("");
  for (const b of document.querySelectorAll("#abas [data-aba]")) b.onclick = () => irAba(b.dataset.aba);
  for (const b of document.querySelectorAll(".segmentado [data-tema]")) b.onclick = () => aplicarTema(b.dataset.tema);
  // Botões com data-ir (abre outra tela), data-voltar e data-fechar (fecha a folha)
  document.addEventListener("click", (ev) => {
    const alvo = ev.target.closest("[data-ir], [data-voltar], [data-fechar]");
    if (!alvo) return;
    if (alvo.hasAttribute("data-fechar")) fecharDialogo(alvo.closest("dialog"));
    else if (alvo.hasAttribute("data-voltar")) voltar();
    else {
      fecharDialogo(alvo.closest("dialog"));
      ir(alvo.dataset.ir);
    }
  });
  // Toque fora da folha fecha
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
// liberar o som: não é gesto do usuário); &bench=1 inicia o benchmark ao ligar; ?pesquisa=1 liga o modo
// pesquisador nesta visita; ?intro=1 mostra o Primeiro uso.
const params = new URLSearchParams(location.search);
if (params.get("autostart") === "1") {
  iniciarViagem().then(() => {
    if (params.get("bench") === "1" && S.running) startBenchmark();
  });
}
