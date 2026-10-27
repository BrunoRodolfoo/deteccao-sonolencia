/**
 * engine.js — Motor de sonolência (módulo puro, sem DOM, testável com Node).
 *
 * Reproduz as fórmulas do protótipo Python (src/realtime/metrics.py) e as
 * características por piscada de Ghoddoosian et al. (CVPRW 2019, eq. 2-5):
 *
 *   EAR       = (|p2-p6| + |p3-p5|) / (2 |p1-p4|)
 *   MAR       = |13-14| / |61-291|
 *   Duration  = end - start + 1                       (quadros)
 *   Amplitude = (EAR[start] - 2 EAR[bottom] + EAR[end]) / 2
 *   Velocity  = (EAR[end] - EAR[bottom]) / (end - bottom)
 *   Frequency = 100 * (piscadas até agora) / (quadros até end)
 *
 * Ordem das 4 características (igual aos .npy do UTA-RLDD):
 *   [0] Frequency, [1] Amplitude, [2] Duration, [3] Velocity
 *
 * Todas as classes deste arquivo são independentes do navegador para que
 * `node test_engine.mjs` valide as fórmulas sem câmera.
 */

// ---------------------------------------------------------------------------
// Índices de landmarks do MediaPipe Face Mesh / Face Landmarker (478 pontos).
// Ordem p1..p6: canto externo, 2 pontos da pálpebra superior, canto interno,
// 2 pontos da pálpebra inferior (mesmos índices de src/realtime/metrics.py).
// ---------------------------------------------------------------------------
export const LEFT_EYE = [362, 385, 387, 263, 373, 380];
export const RIGHT_EYE = [33, 160, 158, 133, 153, 144];
export const MOUTH_CORNERS = [61, 291];
export const MOUTH_VERTICAL = [13, 14];

export const FEATURE_NAMES = ["Frequency", "Amplitude", "Duration", "Velocity"];
export const CLASS_NAMES = ["alerta", "pouca vigilância", "sonolento"];

export function dist(a, b) {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return Math.sqrt(dx * dx + dy * dy);
}

/** EAR de um olho a partir dos 6 pontos [[x,y] p1..p6]. */
export function eyeAspectRatio(pts) {
  const [p1, p2, p3, p4, p5, p6] = pts;
  const horiz = dist(p1, p4);
  if (horiz < 1e-6) return 0;
  return (dist(p2, p6) + dist(p3, p5)) / (2 * horiz);
}

/** Converte landmarks normalizados (x,y em [0,1]) em pixels para os índices dados. */
export function pickPoints(landmarks, idxs, w, h) {
  return idxs.map((i) => [landmarks[i].x * w, landmarks[i].y * h]);
}

/** EAR esquerdo/direito/médio a partir da lista de landmarks. */
export function earFromLandmarks(landmarks, w, h) {
  const left = eyeAspectRatio(pickPoints(landmarks, LEFT_EYE, w, h));
  const right = eyeAspectRatio(pickPoints(landmarks, RIGHT_EYE, w, h));
  return { left, right, mean: (left + right) / 2 };
}

/** MAR: abertura vertical da boca normalizada pela largura. */
export function marFromLandmarks(landmarks, w, h) {
  const [c0, c1] = pickPoints(landmarks, MOUTH_CORNERS, w, h);
  const [v0, v1] = pickPoints(landmarks, MOUTH_VERTICAL, w, h);
  const horiz = dist(c0, c1);
  if (horiz < 1e-6) return 0;
  return dist(v0, v1) / horiz;
}

/**
 * Caixa quadrada de recorte do olho: centrada no centroide dos 6 pontos,
 * lado = scale * |p1-p4| (padrão 1.6). Retorna {x, y, side} em pixels
 * (canto superior esquerdo), já limitada ao quadro w x h.
 */
export function eyeCropBox(pts, w, h, scale = 1.6, minSide = 8) {
  let cx = 0;
  let cy = 0;
  for (const p of pts) {
    cx += p[0];
    cy += p[1];
  }
  cx /= pts.length;
  cy /= pts.length;
  let side = Math.max(minSide, scale * dist(pts[0], pts[3]));
  side = Math.min(side, w, h);
  let x = cx - side / 2;
  let y = cy - side / 2;
  x = Math.max(0, Math.min(x, w - side));
  y = Math.max(0, Math.min(y, h - side));
  return { x, y, side, cx, cy };
}

/**
 * Normaliza um recorte em escala de cinza (valores 0..255) conforme os
 * metadados de models/eye_state.json.
 *   meta.normalize: "0-1" (x/255), "-1-1" (x/127.5-1), "mean_std" ((x/255-mean)/std)
 *                   ou "none" (x cru, 0..255).
 * Retorna Float32Array.
 */
export function normalizeEyeInput(gray, meta = {}) {
  const mode = meta.normalize || "0-1";
  const out = new Float32Array(gray.length);
  if (mode === "mean_std") {
    const mean = meta.mean ?? 0.5;
    const std = meta.std ?? 0.5;
    for (let i = 0; i < gray.length; i++) out[i] = (gray[i] / 255 - mean) / (std || 1e-6);
  } else if (mode === "-1-1") {
    for (let i = 0; i < gray.length; i++) out[i] = gray[i] / 127.5 - 1;
  } else if (mode === "none") {
    for (let i = 0; i < gray.length; i++) out[i] = gray[i];
  } else {
    for (let i = 0; i < gray.length; i++) out[i] = gray[i] / 255;
  }
  return out;
}

/** Quantiza valores float para int8/uint8 dado {scale, zero_point}. */
export function quantize(values, quant, dtype = "int8") {
  const scale = quant?.scale ?? 1;
  const zp = quant?.zero_point ?? 0;
  const lo = dtype === "uint8" ? 0 : -128;
  const hi = dtype === "uint8" ? 255 : 127;
  const out = new Int32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    let q = Math.round(values[i] / scale) + zp;
    if (q < lo) q = lo;
    if (q > hi) q = hi;
    out[i] = q;
  }
  return out;
}

/** Desquantiza inteiros para float dado {scale, zero_point}. */
export function dequantize(values, quant) {
  const scale = quant?.scale ?? 1;
  const zp = quant?.zero_point ?? 0;
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = (values[i] - zp) * scale;
  return out;
}

export function softmax(logits) {
  let m = -Infinity;
  for (const v of logits) if (v > m) m = v;
  let s = 0;
  const out = new Float32Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] - m);
    s += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= s || 1;
  return out;
}

export function argmax(arr) {
  let bi = 0;
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[bi]) bi = i;
  return bi;
}

export function median(values) {
  if (!values.length) return 0;
  const s = Array.from(values).sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

export function percentile(values, p) {
  if (!values.length) return 0;
  const s = Array.from(values).sort((a, b) => a - b);
  const idx = Math.min(s.length - 1, Math.max(0, Math.round((p / 100) * (s.length - 1))));
  return s[idx];
}

// ---------------------------------------------------------------------------
// PERCLOS em janela deslizante (ms).
// ---------------------------------------------------------------------------
export class Perclos {
  constructor(windowMs = 60000) {
    this.windowMs = windowMs;
    this.events = []; // [t, closed]
    this.closedCount = 0;
  }
  update(t, closed) {
    const c = closed ? 1 : 0;
    this.events.push([t, c]);
    this.closedCount += c;
    while (this.events.length && t - this.events[0][0] > this.windowMs) {
      this.closedCount -= this.events[0][1];
      this.events.shift();
    }
    return this.value;
  }
  get value() {
    return this.events.length ? this.closedCount / this.events.length : 0;
  }
  reset() {
    this.events = [];
    this.closedCount = 0;
  }
}

// ---------------------------------------------------------------------------
// Calibração do limiar de EAR: 0.75 x mediana do EAR de olho aberto (3 s).
// ---------------------------------------------------------------------------
export class ThresholdCalibrator {
  constructor({ durationMs = 3000, factor = 0.75 } = {}) {
    this.durationMs = durationMs;
    this.factor = factor;
    this.values = [];
    this.t0 = null;
    this.threshold = null;
    this.openMedian = null;
  }
  get active() {
    return this.t0 !== null && this.threshold === null;
  }
  start(t) {
    this.values = [];
    this.t0 = t;
    this.threshold = null;
    this.openMedian = null;
  }
  /** Retorna progresso 0..1; ao completar define this.threshold. */
  update(t, ear) {
    if (!this.active) return this.threshold === null ? 0 : 1;
    this.values.push(ear);
    const prog = Math.min(1, (t - this.t0) / this.durationMs);
    if (prog >= 1 && this.values.length >= 5) {
      this.openMedian = median(this.values);
      this.threshold = this.factor * this.openMedian;
    }
    return prog;
  }
}

// ---------------------------------------------------------------------------
// Detector de piscadas por limiar com start/bottom/end e as 4 características.
//   start  = último quadro com olho aberto antes de cruzar o limiar
//   bottom = quadro de EAR mínimo durante o fechamento
//   end    = primeiro quadro com olho aberto após o fechamento
// Regras: mínimo `minClosedFrames` quadros abaixo do limiar; fechamento com
// duração acima de `maxClosedMs` é descartado como piscada (é "olho fechado",
// não piscada) — mas continua contando para PERCLOS/alertas na camada acima.
// ---------------------------------------------------------------------------
export class BlinkDetector {
  constructor({ threshold = 0.2, minClosedFrames = 2, maxClosedMs = 1500 } = {}) {
    this.threshold = threshold;
    this.minClosedFrames = minClosedFrames;
    this.maxClosedMs = maxClosedMs;
    this.reset();
  }
  reset() {
    this.frameIdx = -1; // índice do quadro atual (0-based)
    this.blinkCount = 0;
    this.closed = false;
    this.prevOpen = null; // {idx, ear}
    this.startIdx = null;
    this.startEar = null;
    this.startT = null;
    this.bottomIdx = null;
    this.bottomEar = Infinity;
    this.closedFrames = 0;
    this.blinks = [];
  }
  get closedSinceMs() {
    return this.closed ? this.startT : null;
  }
  /**
   * Processa um quadro. `t` em ms. Retorna a piscada concluída
   * ({start,bottom,end,features:[F,A,D,V], t}) ou null.
   */
  update(ear, t) {
    this.frameIdx += 1;
    const idx = this.frameIdx;
    const below = ear < this.threshold;
    let result = null;

    if (!this.closed) {
      if (below) {
        // Início de fechamento: start = último quadro aberto (ou este, se for o primeiro)
        this.closed = true;
        if (this.prevOpen) {
          this.startIdx = this.prevOpen.idx;
          this.startEar = this.prevOpen.ear;
        } else {
          this.startIdx = idx;
          this.startEar = ear;
        }
        this.startT = t;
        this.bottomIdx = idx;
        this.bottomEar = ear;
        this.closedFrames = 1;
      } else {
        this.prevOpen = { idx, ear };
      }
      return null;
    }

    // Estamos em fechamento
    if (below) {
      this.closedFrames += 1;
      if (ear < this.bottomEar) {
        this.bottomEar = ear;
        this.bottomIdx = idx;
      }
      return null;
    }

    // Reabriu: end = este quadro
    const endIdx = idx;
    const endEar = ear;
    const closedMs = t - this.startT;
    const valid = this.closedFrames >= this.minClosedFrames && closedMs <= this.maxClosedMs && endIdx > this.bottomIdx;
    if (valid) {
      this.blinkCount += 1;
      const duration = endIdx - this.startIdx + 1;
      const amplitude = (this.startEar - 2 * this.bottomEar + endEar) / 2;
      const velocity = (endEar - this.bottomEar) / (endIdx - this.bottomIdx);
      const frequency = (100 * this.blinkCount) / (endIdx + 1);
      result = {
        start: this.startIdx,
        bottom: this.bottomIdx,
        end: endIdx,
        closedMs,
        t,
        features: [frequency, amplitude, duration, velocity],
      };
      this.blinks.push(result);
    }
    this.closed = false;
    this.prevOpen = { idx, ear };
    this.startIdx = null;
    this.bottomIdx = null;
    this.bottomEar = Infinity;
    this.closedFrames = 0;
    return result;
  }
}

// ---------------------------------------------------------------------------
// Normalização por sujeito: média/desvio das primeiras N piscadas em estado
// alerta -> z-score (paper: um terço do vídeo de alerta; aqui N=20 piscadas).
// ---------------------------------------------------------------------------
export class SubjectNormalizer {
  constructor({ nCalib = 20, nFeatures = 4, eps = 1e-6 } = {}) {
    this.nCalib = nCalib;
    this.nFeatures = nFeatures;
    this.eps = eps;
    this.reset();
  }
  reset() {
    this.samples = [];
    this.mean = null;
    this.std = null;
    this.collecting = false;
  }
  start() {
    this.reset();
    this.collecting = true;
  }
  get ready() {
    return this.mean !== null;
  }
  get progress() {
    return this.ready ? 1 : Math.min(1, this.samples.length / this.nCalib);
  }
  /** Adiciona piscada em calibração; retorna true quando completar. */
  add(features) {
    if (!this.collecting || this.ready) return this.ready;
    this.samples.push(Array.from(features));
    if (this.samples.length >= this.nCalib) this.finish();
    return this.ready;
  }
  finish() {
    const n = this.samples.length;
    if (n < 2) return false;
    const mean = new Array(this.nFeatures).fill(0);
    const std = new Array(this.nFeatures).fill(0);
    for (const s of this.samples) for (let j = 0; j < this.nFeatures; j++) mean[j] += s[j] / n;
    for (const s of this.samples) for (let j = 0; j < this.nFeatures; j++) std[j] += (s[j] - mean[j]) ** 2 / n;
    for (let j = 0; j < this.nFeatures; j++) std[j] = Math.sqrt(std[j]);
    this.mean = mean;
    this.std = std;
    this.collecting = false;
    return true;
  }
  /** Define parâmetros manualmente (ex.: carregados de JSON). */
  setParams(mean, std) {
    this.mean = Array.from(mean);
    this.std = Array.from(std);
    this.collecting = false;
  }
  zscore(features) {
    if (!this.ready) return Array.from(features);
    return Array.from(features, (v, j) => (v - this.mean[j]) / (this.std[j] > this.eps ? this.std[j] : this.eps));
  }
}

// ---------------------------------------------------------------------------
// Buffer das últimas `size` piscadas -> tensor [1, size, nFeatures].
// Com `padPartial`, permite predição parcial a partir de `minBlinks` piscadas,
// preenchendo o início com zeros (= piscada "típica" após z-score).
// ---------------------------------------------------------------------------
export class BlinkWindow {
  constructor({ size = 30, nFeatures = 4, minBlinks = 30, padPartial = false } = {}) {
    this.size = size;
    this.nFeatures = nFeatures;
    this.minBlinks = minBlinks;
    this.padPartial = padPartial;
    this.items = [];
  }
  push(z) {
    this.items.push(Array.from(z));
    if (this.items.length > this.size) this.items.shift();
  }
  get length() {
    return this.items.length;
  }
  get full() {
    return this.items.length >= this.size;
  }
  get ready() {
    return this.full || (this.padPartial && this.items.length >= this.minBlinks);
  }
  /** Float32Array achatada [size*nFeatures] ou null. */
  tensor() {
    if (!this.ready) return null;
    const out = new Float32Array(this.size * this.nFeatures);
    const off = this.size - this.items.length;
    for (let i = 0; i < this.items.length; i++)
      for (let j = 0; j < this.nFeatures; j++) out[(off + i) * this.nFeatures + j] = this.items[i][j];
    return out;
  }
  reset() {
    this.items = [];
  }
}

// ---------------------------------------------------------------------------
// Suavização: média móvel exponencial das probabilidades + histórico para voto.
// ---------------------------------------------------------------------------
export class ClassSmoother {
  constructor({ alpha = 0.3, nClasses = 3, history = 10 } = {}) {
    this.alpha = alpha;
    this.nClasses = nClasses;
    this.historySize = history;
    this.reset();
  }
  reset() {
    this.probs = null;
    this.classes = [];
    this.scores = [];
  }
  /** probs: array de tamanho nClasses (soma 1). Retorna {probs, cls, vote}. */
  update(probs) {
    if (!this.probs) this.probs = Array.from(probs);
    else for (let i = 0; i < this.nClasses; i++) this.probs[i] = this.alpha * probs[i] + (1 - this.alpha) * this.probs[i];
    const cls = argmax(probs);
    this.classes.push(cls);
    this.scores.push(expectedScore(probs));
    if (this.classes.length > this.historySize) {
      this.classes.shift();
      this.scores.shift();
    }
    return { probs: Array.from(this.probs), cls: argmax(this.probs), vote: majorityVote(this.classes, this.scores) };
  }
}

/** Escore contínuo 0..10 equivalente à regressão do paper: E[classe]*5. */
export function expectedScore(probs) {
  let s = 0;
  for (let i = 0; i < probs.length; i++) s += i * probs[i];
  return s * 5;
}

/** Discretização do escore 0..10 como no paper: <3.3 alerta, 3.3-6.6 pouca vigilância, >6.6 sonolento. */
export function scoreToClass(score) {
  if (score < 3.3) return 0;
  if (score <= 6.6) return 1;
  return 2;
}

/**
 * Voto majoritário das classes preditas, com desempate pela média dos
 * escores contínuos (métrica VA de Ghoddoosian et al.).
 */
export function majorityVote(classes, scores = null, nClasses = 3) {
  if (!classes.length) return null;
  const counts = new Array(nClasses).fill(0);
  for (const c of classes) counts[c] += 1;
  const best = Math.max(...counts);
  const winners = [];
  for (let i = 0; i < nClasses; i++) if (counts[i] === best) winners.push(i);
  if (winners.length === 1) return winners[0];
  if (!scores || !scores.length) return winners[0];
  const meanScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  const byScore = scoreToClass(meanScore);
  return winners.includes(byScore) ? byScore : winners.reduce((a, b) => (Math.abs(a - byScore) <= Math.abs(b - byScore) ? a : b));
}

// ---------------------------------------------------------------------------
// Estatísticas de latência (p50/p95 das últimas N medições).
// ---------------------------------------------------------------------------
export class LatencyStats {
  constructor(size = 300) {
    this.size = size;
    this.values = [];
  }
  push(ms) {
    this.values.push(ms);
    if (this.values.length > this.size) this.values.shift();
  }
  get last() {
    return this.values.length ? this.values[this.values.length - 1] : 0;
  }
  get p50() {
    return percentile(this.values, 50);
  }
  get p95() {
    return percentile(this.values, 95);
  }
  get mean() {
    return this.values.length ? this.values.reduce((a, b) => a + b, 0) / this.values.length : 0;
  }
  get n() {
    return this.values.length;
  }
}

// ---------------------------------------------------------------------------
// Lógica de alertas com amortecimento temporal.
// ---------------------------------------------------------------------------
export class AlertLogic {
  constructor({ closedMs = 1000, perclosThresh = 0.15, drowsyMs = 10000, yawnMs = 2000, marThresh = 0.6 } = {}) {
    this.closedMs = closedMs;
    this.perclosThresh = perclosThresh;
    this.drowsyMs = drowsyMs;
    this.yawnMs = yawnMs;
    this.marThresh = marThresh;
    this.closedSince = null;
    this.drowsySince = null;
    this.yawnSince = null;
  }
  /**
   * @param t ms; closed bool; perclos 0..1; cls classe suavizada (ou null); mar
   * @returns {alerts: string[], warnings: string[]}
   */
  update(t, { closed, perclos, cls, mar }) {
    const alerts = [];
    const warnings = [];
    if (closed) {
      this.closedSince = this.closedSince ?? t;
      if (t - this.closedSince >= this.closedMs) alerts.push("OLHOS FECHADOS");
    } else this.closedSince = null;

    if (perclos > this.perclosThresh) alerts.push(`PERCLOS ${(perclos * 100).toFixed(0)}%`);

    if (cls === 2) {
      this.drowsySince = this.drowsySince ?? t;
      if (t - this.drowsySince >= this.drowsyMs) alerts.push("SONOLÊNCIA SUSTENTADA");
    } else this.drowsySince = null;

    if (mar !== undefined && mar > this.marThresh) {
      this.yawnSince = this.yawnSince ?? t;
      if (t - this.yawnSince >= this.yawnMs) warnings.push("BOCEJO");
    } else this.yawnSince = null;

    return { alerts, warnings };
  }
}

/** Converte tabela de objetos em CSV (cabeçalho = chaves do primeiro item). */
export function toCSV(rows, sep = ",") {
  if (!rows.length) return "";
  const keys = Object.keys(rows[0]);
  const esc = (v) => {
    if (v === null || v === undefined) return "";
    const s = typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(4)) : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const lines = [keys.join(sep)];
  for (const r of rows) lines.push(keys.map((k) => esc(r[k])).join(sep));
  return lines.join("\n");
}

// ===========================================================================
// DETECTOR v2 (27/09/2026) — sinais interpretáveis + nível de sonolência 0-100
// ===========================================================================
//
// Por que existe: a v1 (AlertLogic acima) disparava VERMELHO com um único sinal (olho fechado
// >= 1 s, PERCLOS > 15 % ou classe "sonolento" do LSTM por 10 s). No iPhone (27/09/2026) isso deu:
//  - PERCLOS preso acima de 15 % por até 60 s depois de UM fechamento longo: alarme contínuo com a
//    pessoa já de olhos abertos (results/phone/benchmark_pwa_2026-09-27T22-41-35-365Z_frames.csv);
//  - fechamento decidido só pelo EAR: olhar para baixo abaixa a pálpebra e conta como "fechado";
//  - bocejo só como aviso por MAR (revocação de 35 % no DMD) e nenhuma pose de cabeça;
//  - o LSTM (VA ~57 % no Exp. 1) decidindo alarme sozinho, com características fora da distribuição.
// A v2 junta sinais que a banca consegue ver e conferir — olhos, bocejo, cabeça — num nível 0-100
// em que cada ponto tem origem explícita ("Por quê?"). Nenhum sinal isolado leva ao vermelho, exceto
// os de perigo imediato (microssono e cabeça caída sustentada), que forçam o vermelho na hora.
// Todos os limiares e pesos estão em CFG, abaixo; a justificativa de cada um está no README.md.

function deepFreeze(o) {
  for (const v of Object.values(o)) if (v && typeof v === "object") deepFreeze(v);
  return Object.freeze(o);
}

/** Limiares, janelas e pesos do detector v2 — ÚNICO lugar onde esses números aparecem. */
export const CFG = deepFreeze({
  calib: {
    neutralMs: 10000, //   fase 1: 10 s de rosto visível, olhando para a frente, boca fechada
    earFactor: 0.75, //    limiar do EAR = 0,75 x mediana (maior kappa por quadro no DMD: results/exp5_dmd_sensibilidade_limiar.csv)
    nBlinks: 20, //        fase 2: piscadas espontâneas -> duração de base (e z-score do LSTM, como na v1)
    maxClosedFrac: 0.1, // aviso se os olhos (os dois eyeBlink >= 0,5) ficarem fechados em > 10 % da fase 1: acordado,
    //                     15-20 piscadas/min de 0,1-0,3 s dão 2-10 % (a base sairia com a pessoa já sonolenta)
  },
  eyes: {
    // Regra escolhida entre 11 variantes no DMD (results/app_validacao_dmd_variantes_olhos.csv):
    blendClosed: 0.5, //   fechado se eyeBlinkLeft E eyeBlinkRight >= 0,5 ...
    blendRel: 0.25, //     ... e >= (eyeBlink de repouso da calibração + 0,25): pálpebra pesada/sol não vira "fechado"
    earPath: true, //      o EAR abaixo do limiar também marca fechado se TODAS as condições abaixo valerem:
    blendSupport: 0.4, //  - o MENOR eyeBlink dos dois olhos >= 0,4 (os dois olhos parecem fechar)
    supportUseMin: true,
    lookDownGate: 0.6, //  - NÃO olhando para baixo (média dos eyeLookDown < 0,6)
    headDownGateDeg: 10, // - NÃO com a cabeça >= 10° abaixo da base (olhar para baixo abaixa a pálpebra)
    squintGate: 0.45, //   - NÃO apertando os olhos (média dos eyeSquint < 0,45: sorriso, risada, sol)
    asideGateDeg: 20, //   - NÃO olhando para o lado (|Δyaw| < 20°)
    reopenHyst: 0, //      histerese do caminho A (reabrir só abaixo do limiar - reopenHyst): AVALIADA E DESLIGADA. Com
    //                     0,1, olhar para baixo depois de uma piscada deixa o eyeBlink em 0,4-0,5 e o "fechado" se
    //                     estende: na direção real (DMD s1) os fechamentos de 0,5-1 s foram de 15 para 46 (README)
    longMs: 500, //        fechamento longo (0,5-1 s)
    microsleepMs: 1000, // microssono: olhos fechados >= 1 s -> VERMELHO imediato
    blinkMinMs: 50, //     episódios mais curtos são ruído (1 quadro a 30-60 FPS)
    blinkMaxMs: 500, //    piscada = fechamento < 0,5 s (entra na duração típica)
    lostGraceMs: 300, //   rosto perdido por mais que isso encerra o fechamento em andamento
  },
  perclos: {
    windowMs: 60000, //    PERCLOS em 60 s (Wierwille et al., 1994; Dinges e Grace, 1998)
    minWindowMs: 60000, // denominador mínimo de 60 s: antes de 1 min de observação, o tempo não visto conta como aberto
    maxDtMs: 100, //       um quadro nunca "vale" mais que 100 ms (pausas da câmera)
    minEpisodeMs: 500, //  só entra o tempo de fechamentos >= 0,5 s (o mesmo limite de "fechamento longo", longMs):
    //                     o PERCLOS mede fechamentos LENTOS da pálpebra, não piscadas (Wierwille et al., 1994;
    //                     Dinges e Grace, 1998). Com as piscadas dentro, quem pisca 75 vezes/min acordado passava
    //                     de 12 % de PERCLOS (UTA-RLDD Fold3/31, vídeo alerta: results/app_validacao.md)
  },
  yawn: {
    jawOpen: 0.4, //       blendshape jawOpen >= 0,4 = boca bem aberta (grade no DMD: results/app_validacao_dmd_bocejo_grade.csv)
    jawRel: 0.3, //        ... e pelo menos 0,3 acima do jawOpen de repouso da calibração
    marRel: 0.4, //        apoio: MAR >= MAR de repouso + 0,4 também conta como boca aberta
    minMs: 1500, //        aberta por >= 1,5 s = 1 bocejo (fala abre e fecha em frações de segundo)
    gapMs: 150, //         quedas curtas (< 0,15 s) no meio do bocejo não o encerram (a fala fecha a boca por mais tempo)
    lostMaxMs: 1000, //    mão na frente da boca / rosto perdido por até 1 s não encerra o bocejo
    windowMs: 300000, //   contagem em 5 min
  },
  head: {
    pitchDownSign: 1, //   pitch da matriz > 0 = cabeça para baixo (conferido em dados reais; README)
    neutralDeg: 5, //      |Δpitch| < 5° = perto do neutro (referência para medir a velocidade da queda)
    nodDeg: 15, //         Δpitch >= 15° abre um episódio de cabeça baixando
    dropDeg: 18, //        Δpitch >= 18° ...
    dropMs: 800, //        ... por >= 0,8 s ...
    eyesMs: 400, //        ... com os olhos fechados por >= 0,4 s CONTÍNUOS no episódio (ou rosto perdido) = "cabeça caindo"
    releaseDeg: 10, //     episódio termina quando Δpitch volta abaixo de 10° (histerese)
    rollDeg: 30, //        cabeça tombando para o lado (|Δroll| >= 30°) conta como queda
    rollReleaseDeg: 20,
    redEyesMs: 800, //     caída com olhos fechados por >= 0,8 s -> VERMELHO imediato
    redLostMs: 2000, //    ... ou rosto perdido por >= 2 s depois de começar a cair
    downOpenMs: 2000, //   baixa >= 2 s com olhos ABERTOS = "cabeça baixa" (distração): só registro, 0 ponto
    yawAsideDeg: 25, //    |Δyaw| > 25° = olhando para o lado (retrovisor): episódio ignorado
    nodOnsetMs: 500, //    cabeceio: desce do neutro a >= 15° em <= 0,5 s e volta em <= 1,5 s,
    nodMaxMs: 1500, //     com olhos fechados >= 0,4 s (uma piscada comum dura 0,1-0,3 s)
    lostWindowMs: 300, //  rosto perdido até 0,3 s depois de Δpitch >= 10°: cabeça caindo fora do quadro ...
    lostDownDeg: 10,
    lostEyesMs: 400, //    ... SÓ se os olhos estavam fechados >= 0,4 s contínuos quando o rosto sumiu (a mesma evidência
    //                     de sono de eyesMs). Quem se abaixa para pegar algo sai do quadro de olhos ABERTOS: não é queda
    lostMaxMs: 4000, //    sem rosto por > 4 s encerra o episódio
    driftBandDeg: 8, //    perto do neutro a base acompanha a postura devagar ...
    driftTauMs: 120000, // ... com constante de tempo de 2 min
    windowMs: 300000, //   contagem em 5 min
  },
  score: {
    perclos: { from: 0.075, at: 0.15, points: 40, cap: 60 }, // 0 pt até 7,5 %; 40 pt em 15 % (referência do app: mais conservadora que os 8 a 12 % usuais); teto 60
    longClosure: { points: 25, decayMs: 120000, countMs: 60000 }, // cada fechamento 0,5-1 s: 25 pt, somem em 2 min
    // microssono: o efeito principal é o VERMELHO imediato (forçado) enquanto os olhos estão fechados; depois de
    // reabrir, ele conta no nível como mais um fechamento longo (a especificação E pede "fechamentos longos por
    // minuto"): 25 pt que somem em 2 min, e o estado fica pelo menos em ATENÇÃO por 1 min (postForced, abaixo)
    microsleep: { points: 25, decayMs: 120000 },
    closuresCap: 60, //    teto somado de fechamentos longos + microssonos
    yawn: { points: 25, decayMs: 300000, cap: 50 }, // cada bocejo: 25 pt, somem em 5 min; teto 50 (2 bocejos em até 2 min = ATENÇÃO)
    head: { points: 30, decayMs: 300000, cap: 60 }, // cada queda/cabeceio: 30 pt, somem em 5 min; teto 60
    blinkDur: { from: 1.25, at: 2.0, points: 20, cap: 20, minBlinks: 5, windowMs: 120000 }, // piscadas 25-100 % mais longas que na calibração
    // LSTM: +10 só com "sonolento" (p >= 0,6) em 5 predições seguidas E com os sinais visíveis já em ATENÇÃO
    // (>= 40 sem o modelo): ele reforça um quadro de cansaço, mas nunca tira ninguém do verde sozinho
    ml: { points: 10, minProb: 0.6, consecutive: 5, minBase: 40 },
    // Depois de um vermelho forçado (microssono / cabeça caída), o nível fica em pelo menos 40 (ATENÇÃO) por 1 min:
    // quem acabou de cochilar não volta para o verde segundos depois de abrir os olhos
    postForced: { floor: 40, holdMs: 60000 },
    forcedFloor: 75, //    microssono / cabeça caída sustentada: nível exibido vai a pelo menos 75
    maxFallPerS: 3, //     o nível exibido desce no máximo 3 pontos por segundo
  },
  levels: { attention: 40, attentionExit: 35, drowsy: 70, drowsyExit: 60 }, // histerese entre os 3 estados
  alarm: {
    sirenMinMs: 2000, //   sirene toca pelo menos 2 s
    dingGapMs: 20000, //   no máximo um "ding" a cada 20 s
    vibrateEveryMs: 1500, // vibração repetida durante a sirene (Android)
    snoozeMs: 30000, //    "Estou acordado" silencia o vermelho por nível durante 30 s (nunca o microssono)
  },
});

export const STATE_NAMES = ["ALERTA", "ATENÇÃO", "SONOLENTO"];

/** Formata número com vírgula decimal (pt-BR). */
export function fmtBR(v, d = 1) {
  return Number.isFinite(v) ? v.toFixed(d).replace(".", ",") : "--";
}

/** Rampa linear: 0 até `from`, `points` em `at`, mesma inclinação depois, limitada a `cap`. */
export function ramp(x, from, at, points, cap = points) {
  if (!Number.isFinite(x) || x <= from) return 0;
  return Math.min(cap, ((x - from) / (at - from)) * points);
}

// ---------------------------------------------------------------------------
// Pose da cabeça a partir da matriz de transformação facial do Face Landmarker.
//
// O Face Landmarker (outputFacialTransformationMatrixes) devolve, por rosto, {rows:4, columns:4,
// data:number[16]}: a pose do modelo canônico de rosto no espaço métrico da câmera (cm; câmera na
// origem olhando para -Z, Y para cima). No tasks-vision (web) `data` é o packed_data do proto
// MatrixData sem reordenar; o C++ grava a Eigen::Matrix4f em ordem de COLUNAS, logo a translação fica
// em data[12..14] (e data[3], data[7], data[11] = 0). matrixLayout() confere isso pelos próprios
// valores (a translação em z é sempre ~-30 a -60 cm), para não depender da ordem.
//
// Ângulos (graus), com R = Rz(roll) · Ry(yaw) · Rx(pitch):
//   pitch = atan2(R21, R22)  (rotação em X: cabeça para cima/baixo; > 0 = para baixo)
//   yaw   = asin(-R20)       (rotação em Y: virar para os lados)
//   roll  = atan2(R10, R00)  (rotação em Z: tombar para o lado)
// O sinal "pitch > 0 = para baixo" foi conferido em 27/09/2026 nos vídeos do DMD (README).
// ---------------------------------------------------------------------------
const RAD2DEG = 180 / Math.PI;

export function matrixLayout(data) {
  const col = Math.abs(data[12]) + Math.abs(data[13]) + Math.abs(data[14]);
  const row = Math.abs(data[3]) + Math.abs(data[7]) + Math.abs(data[11]);
  return row > col ? "row" : "col";
}

/** {pitch, yaw, roll, tx, ty, tz, layout} a partir da matriz 4x4 achatada; null se inválida. */
export function poseFromMatrix(data, layout = "auto") {
  if (!data || data.length < 16) return null;
  for (let i = 0; i < 16; i++) if (!Number.isFinite(data[i])) return null;
  const L = layout === "auto" ? matrixLayout(data) : layout;
  const m = L === "row" ? (r, c) => data[r * 4 + c] : (r, c) => data[c * 4 + r];
  const s = [0, 1, 2].map((c) => Math.hypot(m(0, c), m(1, c), m(2, c)) || 1); // remove escala, se houver
  const R = (r, c) => m(r, c) / s[c];
  return {
    pitch: Math.atan2(R(2, 1), R(2, 2)) * RAD2DEG,
    yaw: Math.asin(Math.max(-1, Math.min(1, -R(2, 0)))) * RAD2DEG,
    roll: Math.atan2(R(1, 0), R(0, 0)) * RAD2DEG,
    tx: m(0, 3),
    ty: m(1, 3),
    tz: m(2, 3),
    layout: L,
  };
}

/** Inverso de poseFromMatrix (para testes): matriz 4x4 achatada de R = Rz·Ry·Rx e translação. */
export function matrixFromPose({ pitch = 0, yaw = 0, roll = 0, tx = 0, ty = 0, tz = -40 } = {}, layout = "col") {
  const a = pitch / RAD2DEG;
  const b = yaw / RAD2DEG;
  const g = roll / RAD2DEG;
  const mul = (A, B) => A.map((row) => [0, 1, 2].map((j) => row[0] * B[0][j] + row[1] * B[1][j] + row[2] * B[2][j]));
  const Rx = [[1, 0, 0], [0, Math.cos(a), -Math.sin(a)], [0, Math.sin(a), Math.cos(a)]];
  const Ry = [[Math.cos(b), 0, Math.sin(b)], [0, 1, 0], [-Math.sin(b), 0, Math.cos(b)]];
  const Rz = [[Math.cos(g), -Math.sin(g), 0], [Math.sin(g), Math.cos(g), 0], [0, 0, 1]];
  const R = mul(Rz, mul(Ry, Rx));
  const M = [[...R[0], tx], [...R[1], ty], [...R[2], tz], [0, 0, 0, 1]];
  const out = new Array(16);
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out[layout === "row" ? r * 4 + c : c * 4 + r] = M[r][c];
  return out;
}

/** Lista de categorias do Face Landmarker -> {categoryName: score}. */
export function blendshapeMap(categories) {
  const out = {};
  if (categories) for (const c of categories) out[c.categoryName] = c.score;
  return out;
}

// ---------------------------------------------------------------------------
// Olhos: fusão do EAR calibrado com os blendshapes eyeBlinkLeft/Right (valores em CFG.eyes).
//   fechado = CAMINHO A: min(eyeBlinkL, eyeBlinkR) >= max(0,5; eyeBlink de repouso + 0,25)
//                        (CFG.eyes.reopenHyst permite histerese na reabertura; está DESLIGADA: ver o comentário lá)
//          OU CAMINHO B: EAR < limiar calibrado (0,75 x mediana)
//                        E min(eyeBlinkL, eyeBlinkR) >= 0,4 (os dois olhos parecem fechar)
//                        E nenhuma trava ativa:
//                          - cabeça >= 10° abaixo da base (pitch da matriz),
//                          - olhar para baixo (média eyeLookDown >= 0,6),
//                          - olhos apertados (média eyeSquint >= 0,45: risada, sol),
//                          - olhando para o lado (|Δyaw| >= 20°).
// Quando a pessoa olha para baixo a pálpebra superior desce e o EAR cai sem o olho fechar; nesse
// caso só o blendshape decide (caminho A). Sem blendshapes (null), vale só o EAR, com as travas que
// tiverem dado (ex.: a da cabeça, se houver pose). Regra escolhida entre 11 variantes no DMD
// (tools/app_validacao/validar_motor_dmd.mjs -> results/app_validacao_dmd_variantes_olhos.csv).
// ---------------------------------------------------------------------------
export function fuseEyeClosure({ ear, earThr, blinkL, blinkR, lookDown, dPitchDown, squint, dYaw, blend0, prevClosed = false } = {}, cfg = CFG.eyes) {
  const hasBlend = Number.isFinite(blinkL) && Number.isFinite(blinkR);
  const blendMean = hasBlend ? (blinkL + blinkR) / 2 : null;
  const blendMin = hasBlend ? Math.min(blinkL, blinkR) : null;
  const thrClose = cfg.blendRel > 0 && Number.isFinite(blend0) ? Math.max(cfg.blendClosed, blend0 + cfg.blendRel) : cfg.blendClosed;
  // Histerese (caminho A): já fechado, continua fechado até o menor eyeBlink cair abaixo de (limiar - reopenHyst).
  const thrBlend = prevClosed && cfg.reopenHyst > 0 ? thrClose - cfg.reopenHyst : thrClose;
  const byBlend = hasBlend && blendMin >= thrBlend;
  const byEar = Number.isFinite(ear) && Number.isFinite(earThr) && ear < earThr;
  const headDown = Number.isFinite(dPitchDown) && dPitchDown >= cfg.headDownGateDeg;
  const gazeDown = Number.isFinite(lookDown) && lookDown >= cfg.lookDownGate;
  const squinting = Number.isFinite(squint) && squint >= cfg.squintGate;
  const aside = Number.isFinite(dYaw) && Math.abs(dYaw) >= cfg.asideGateDeg;
  const gated = headDown || gazeDown || squinting || aside;
  const support = !hasBlend || (cfg.supportUseMin ? blendMin : blendMean) >= cfg.blendSupport;
  const closed = byBlend || (cfg.earPath !== false && byEar && !gated && support);
  return { closed, byBlend, byEar, gated, headDown, gazeDown, squinting, aside, blendMean, blendMin };
}

/**
 * Episódios de olho fechado (a partir do booleano fundido). Duração = do primeiro quadro fechado ao
 * primeiro quadro aberto. Rosto perdido por > lostGraceMs encerra o episódio no último quadro visto.
 */
export class ClosureTracker {
  constructor(cfg = CFG.eyes) {
    this.cfg = cfg;
    this.reset();
  }
  reset() {
    this.active = false;
    this.start = null;
    this.lastSeen = null;
    this.msFired = false;
  }
  /** @returns {{closedMs:number, ended:null|{start,end,durMs,lost}, microsleep:boolean}} */
  update(t, closed, face = true) {
    const out = { closedMs: 0, ended: null, microsleep: false };
    if (!face) {
      if (this.active && t - this.lastSeen > this.cfg.lostGraceMs) out.ended = this._end(this.lastSeen, true);
      else if (this.active) out.closedMs = this.lastSeen - this.start;
      return out;
    }
    this.lastSeen = t;
    if (closed) {
      if (!this.active) {
        this.active = true;
        this.start = t;
        this.msFired = false;
      }
      out.closedMs = t - this.start;
      if (!this.msFired && out.closedMs >= this.cfg.microsleepMs) {
        this.msFired = true;
        out.microsleep = true;
      }
    } else if (this.active) {
      out.ended = this._end(t, false);
    }
    return out;
  }
  _end(tEnd, lost) {
    const ep = { start: this.start, end: tEnd, durMs: tEnd - this.start, lost };
    this.active = false;
    this.start = null;
    return ep;
  }
}

/**
 * PERCLOS por TEMPO numa janela de 60 s: tempo de olho fechado em fechamentos LENTOS (>= minEpisodeMs,
 * 0,5 s) / max(tempo observado, 60 s).
 *  - Só fechamentos lentos: o PERCLOS foi proposto para medir fechamentos lentos da pálpebra, não piscadas
 *    (Wierwille et al., 1994; Dinges e Grace, 1998). Um fechamento só entra quando completa 0,5 s; aí entra
 *    inteiro (desde o 1º quadro). Piscadas (< 0,5 s) ficam de fora — mas continuam em `valueAll`, informativo.
 *  - Denominador mínimo de 60 s: um único fechamento logo após a calibração não infla o PERCLOS (na v1, 3 s
 *    de olho fechado com 20 s de janela davam 15 % e alarme por quase 1 min).
 * Com minEpisodeMs = 0 o comportamento é o antigo (todo quadro fechado conta).
 */
export class PerclosTime {
  constructor({ windowMs = 60000, minWindowMs = 60000, maxDtMs = 100, minEpisodeMs = 0 } = {}) {
    this.windowMs = windowMs;
    this.minWindowMs = minWindowMs;
    this.maxDtMs = maxDtMs;
    this.minEpisodeMs = minEpisodeMs;
    this.reset();
  }
  reset() {
    this.buf = []; // [t, dt, conta(0/1), fechado(0/1), saiu da janela]
    this.obs = 0;
    this.cl = 0; //    tempo fechado em fechamentos >= minEpisodeMs
    this.clAll = 0; // tempo fechado em qualquer quadro (inclui piscadas)
    this.lastT = null;
    this.epStart = null;
    this.pending = [];
    this.epCounted = false;
  }
  _commit() {
    for (const e of this.pending) if (!e[4] && !e[2]) { e[2] = 1; this.cl += e[1]; }
    this.pending = [];
    this.epCounted = true;
  }
  _endEpisode(tEnd) {
    if (this.epStart !== null && !this.epCounted && tEnd - this.epStart >= this.minEpisodeMs) this._commit();
    this.epStart = null;
    this.pending = [];
    this.epCounted = false;
  }
  /** Rosto perdido: encerra o fechamento em andamento no último quadro visto. */
  faceLost() {
    if (this.epStart !== null) this._endEpisode(this.lastT);
  }
  /** Chamar só em quadros com rosto. */
  update(t, closed) {
    const dt = this.lastT === null ? 0 : Math.min(this.maxDtMs, Math.max(0, t - this.lastT));
    this.lastT = t;
    const entry = [t, dt, 0, closed ? 1 : 0, false];
    this.buf.push(entry);
    this.obs += dt;
    if (closed) {
      this.clAll += dt;
      if (this.epStart === null) {
        this.epStart = t;
        this.pending = [];
        this.epCounted = this.minEpisodeMs <= 0;
      }
      if (this.epCounted) {
        entry[2] = 1;
        this.cl += dt;
      } else {
        this.pending.push(entry);
        if (t - this.epStart >= this.minEpisodeMs) this._commit();
      }
    } else if (this.epStart !== null) {
      this._endEpisode(t); // duração = do 1º quadro fechado ao 1º aberto (como no ClosureTracker)
    }
    while (this.buf.length && t - this.buf[0][0] > this.windowMs) {
      const e = this.buf.shift();
      e[4] = true;
      this.obs -= e[1];
      this.cl -= e[2] * e[1];
      this.clAll -= e[3] * e[1];
    }
    if (this.obs < 0) this.obs = 0;
    if (this.cl < 0) this.cl = 0;
    if (this.clAll < 0) this.clAll = 0;
    return this.value;
  }
  get observedMs() {
    return this.obs;
  }
  get value() {
    return this.cl / Math.max(this.obs, this.minWindowMs);
  }
  /** PERCLOS com as piscadas (informativo: é o que a v2 inicial usava). */
  get valueAll() {
    return this.clAll / Math.max(this.obs, this.minWindowMs);
  }
  get rawValue() {
    return this.obs > 0 ? this.cl / this.obs : 0;
  }
}

/**
 * Bocejo: boca aberta (jawOpen do blendshape, com o MAR relativo como apoio) sustentada por >= 1,5 s.
 * Emite o evento no instante em que completa 1,5 s (a interface reage durante o bocejo) e informa a
 * duração total quando a boca fecha.
 */
export class YawnDetector {
  constructor(cfg = CFG.yawn) {
    this.cfg = cfg;
    this.jaw0 = 0;
    this.mar0 = null;
    this.reset();
  }
  setBaseline({ jaw0 = 0, mar0 = null } = {}) {
    this.jaw0 = Number.isFinite(jaw0) ? jaw0 : 0;
    this.mar0 = Number.isFinite(mar0) ? mar0 : null;
    this.reset();
  }
  reset() {
    this.active = false;
    this.start = null;
    this.lastOpen = null;
    this.openMs = 0;
    this.emitted = false;
    this.peak = 0;
  }
  isOpen({ jawOpen, mar } = {}) {
    const c = this.cfg;
    const byJaw = Number.isFinite(jawOpen) && jawOpen >= Math.max(c.jawOpen, this.jaw0 + c.jawRel);
    const byMar = Number.isFinite(mar) && this.mar0 !== null && mar - this.mar0 >= c.marRel;
    return byJaw || byMar;
  }
  /** @returns {{open:boolean, openMs:number, event:null|object, ended:null|object}} */
  update(t, { face = true, jawOpen, mar } = {}) {
    const c = this.cfg;
    const out = { open: false, openMs: 0, event: null, ended: null };
    if (face && this.isOpen({ jawOpen, mar })) {
      out.open = true;
      if (!this.active) {
        this.active = true;
        this.start = t;
        this.emitted = false;
        this.peak = 0;
      }
      this.lastOpen = t;
      if (Number.isFinite(jawOpen)) this.peak = Math.max(this.peak, jawOpen);
    } else if (this.active) {
      const limit = face ? c.gapMs : c.lostMaxMs;
      if (t - this.lastOpen > limit) {
        if (this.emitted) out.ended = { start: this.start, end: this.lastOpen, durMs: this.lastOpen - this.start, peak: this.peak };
        this.active = false;
      }
    }
    if (this.active) {
      out.openMs = this.lastOpen - this.start;
      if (!this.emitted && out.openMs >= c.minMs) {
        this.emitted = true;
        out.event = { type: "bocejo", t, peak: this.peak };
      }
    }
    return out;
  }
}

/**
 * Cabeça caindo, a partir da pose (graus) relativa à linha de base da calibração.
 *   Δpitch = sinal x (pitch - pitch_base)  (> 0 = mais para baixo que o neutro)
 *   - "queda" (cabeça caindo): Δpitch >= 18° (ou |Δroll| >= 30°) por >= 0,8 s COM olhos fechados por
 *     >= 0,4 s CONTÍNUOS (ou rosto perdido depois de olhos fechados >= 0,4 s) -> evento, uma vez por episódio
 *     (a soma de piscadas comuns não conta: olhar o celular piscando normalmente não é queda);
 *   - "cabeceio": desce do neutro a >= 15° em <= 0,5 s e volta a < 10° em <= 1,5 s, com olhos fechados
 *     por >= 0,4 s -> evento ao terminar;
 *   - "cabeça baixa": >= 2 s abaixo com os olhos ABERTOS (olhando o painel/celular): só registro, sem
 *     pontos — é distração, não sono;
 *   - sustentada (força VERMELHO): caída com olhos fechados por >= 0,8 s contínuos, ou rosto perdido por >= 2 s
 *     depois de olhos fechados >= 0,4 s contínuos;
 *   - |Δyaw| > 25° (olhando para o lado/retrovisor) anula o episódio;
 *   - rosto perdido logo depois de começar a baixar, com os olhos fechados >= 0,4 s = cabeça caindo fora do
 *     quadro (até 4 s). Com os olhos abertos (pegar algo no banco, abaixar-se) não é queda.
 * "Olhos fechados contínuos" = maior fechamento contínuo DENTRO do episódio (uma piscada no início de um
 * olhar para baixo — comum em movimentos da cabeça — não conta); na perda do rosto sem episódio aberto, vale o
 * fechamento em andamento no último quadro visto.
 * O evento informa o pico para baixo (peakDeg) e o pico de inclinação lateral (peakRollDeg; byRoll quando a
 * queda foi pela inclinação lateral).
 * Perto do neutro, a base acompanha a postura devagar (constante de tempo de 2 min).
 */
export class HeadDropDetector {
  constructor(cfg = CFG.head) {
    this.cfg = cfg;
    this.base = null;
    this.reset();
  }
  setBaseline({ pitch, yaw, roll } = {}) {
    this.base = Number.isFinite(pitch) ? { pitch, yaw: Number.isFinite(yaw) ? yaw : 0, roll: Number.isFinite(roll) ? roll : 0 } : null;
    this.reset();
  }
  reset() {
    this.ep = null;
    this.lastT = null;
    this.lastNeutralT = null;
    this.lastVisible = null;
    this.eyeRunStart = null; // início do fechamento contínuo em andamento (quadros com rosto)
  }
  delta(pose) {
    if (!this.base || !pose || !Number.isFinite(pose.pitch)) return null;
    const yaw = Number.isFinite(pose.yaw) ? pose.yaw : this.base.yaw;
    const roll = Number.isFinite(pose.roll) ? pose.roll : this.base.roll;
    return {
      dPitch: this.cfg.pitchDownSign * (pose.pitch - this.base.pitch),
      dYaw: yaw - this.base.yaw,
      dRoll: roll - this.base.roll,
    };
  }
  _isDown(d) {
    return d.dPitch >= this.cfg.dropDeg || Math.abs(d.dRoll) >= this.cfg.rollDeg;
  }
  _released(d) {
    return d.dPitch < this.cfg.releaseDeg && Math.abs(d.dRoll) < this.cfg.rollReleaseDeg;
  }
  /** @returns {{state:string, dPitch, dYaw, dRoll, event:null|object, sustained:boolean, downMs:number, eyesMs:number}} */
  update(t, { face = true, pose = null, eyesClosed = false } = {}) {
    const c = this.cfg;
    const out = { state: "sem_base", dPitch: null, dYaw: null, dRoll: null, event: null, sustained: false, downMs: 0, eyesMs: 0 };
    const dt = this.lastT === null ? 0 : Math.min(200, Math.max(0, t - this.lastT));
    this.lastT = t;
    if (!this.base) return out;
    // Fechamento contínuo dos olhos com o rosto visível. Sem rosto o estado do olho é desconhecido: o valor do
    // último quadro visto fica em lastVisible.eyeRunMs (evidência para a perda do rosto) e a contagem recomeça.
    if (face && eyesClosed) this.eyeRunStart = this.eyeRunStart ?? t;
    else this.eyeRunStart = null;
    const eyeRunMs = face && this.eyeRunStart !== null ? t - this.eyeRunStart : 0;
    const d = face ? this.delta(pose) : null;
    if (d) {
      Object.assign(out, d);
      const aside = Math.abs(d.dYaw) > c.yawAsideDeg;
      this.lastVisible = { t, ...d, eyeRunMs };
      if (d.dPitch < c.neutralDeg) this.lastNeutralT = t;
      if (!this.ep) {
        const starts = !aside && (d.dPitch >= c.nodDeg || Math.abs(d.dRoll) >= c.rollDeg);
        if (!starts) {
          if (!aside && dt > 0) {
            const k = Math.min(1, dt / c.driftTauMs);
            if (Math.abs(d.dPitch) < c.driftBandDeg) this.base.pitch += k * c.pitchDownSign * d.dPitch;
            if (Math.abs(d.dYaw) < c.driftBandDeg) this.base.yaw += k * d.dYaw;
            if (Math.abs(d.dRoll) < c.driftBandDeg) this.base.roll += k * d.dRoll;
          }
          out.state = aside ? "lado" : "neutra";
          return out;
        }
        this.ep = this._newEpisode(t, this.lastNeutralT === null ? Infinity : t - this.lastNeutralT, d.dPitch, false);
      }
      const ep = this.ep;
      ep.lostSince = null;
      if (aside) ep.aside = true;
      ep.peak = Math.max(ep.peak, d.dPitch);
      ep.peakRoll = Math.max(ep.peakRoll, Math.abs(d.dRoll));
      if (eyesClosed) {
        ep.closedMs += dt;
        if (ep.runStart === null) ep.runStart = t;
        ep.maxRunMs = Math.max(ep.maxRunMs, t - ep.runStart);
      } else ep.runStart = null;
      if (this._isDown(d)) {
        if (ep.dropSince === null) {
          ep.dropSince = t;
          ep.byRoll = d.dPitch < c.dropDeg; // caiu pela inclinação lateral, não para a frente
        }
      } else if (d.dPitch < c.dropDeg - 3 && Math.abs(d.dRoll) < c.rollDeg - 3) {
        ep.dropSince = null; // subiu de novo sem terminar o episódio: recomeça a contar
      }
      if (this._released(d)) return this._finish(t, out, false);
    } else {
      if (!this.ep) {
        // Rosto sumiu logo depois de a cabeça começar a baixar E com os olhos fechados >= 0,4 s naquele instante.
        const lv = this.lastVisible;
        const wasGoingDown = lv && t - lv.t <= c.lostWindowMs && lv.dPitch >= c.lostDownDeg && Math.abs(lv.dYaw) <= c.yawAsideDeg && lv.eyeRunMs >= c.lostEyesMs;
        if (!wasGoingDown) {
          out.state = "sem_rosto";
          return out;
        }
        this.ep = this._newEpisode(t, Infinity, lv.dPitch, true);
        this.ep.dropSince = t;
        this.ep.peakRoll = Math.abs(lv.dRoll);
        this.ep.maxRunMs = lv.eyeRunMs;
      }
      const ep = this.ep;
      if (ep.lostSince === null) ep.lostSince = t;
      ep.lost = true;
      ep.runStart = null;
      ep.lostMs += dt;
      if (ep.dropSince === null && this.lastVisible && this.lastVisible.dPitch >= c.lostDownDeg) ep.dropSince = t;
      if (t - ep.lostSince > c.lostMaxMs) return this._finish(t, out, true);
    }
    const ep = this.ep;
    out.downMs = ep.dropSince === null ? 0 : t - ep.dropSince;
    out.eyesMs = ep.maxRunMs;
    out.byRoll = !!ep.byRoll;
    out.peakRollDeg = ep.peakRoll;
    if (ep.aside) {
      out.state = "lado";
      return out;
    }
    // Olhos fechados = MAIOR fechamento CONTÍNUO no episódio (não a soma): olhando o celular com a cabeça
    // baixa, piscadas comuns (0,1-0,3 s cada) somariam 0,4-0,8 s e virariam "queda" e vermelho.
    // Rosto perdido NÃO é mais evidência de sono por si só (quem se abaixa sai do quadro de olhos abertos).
    const sleepy = ep.maxRunMs >= c.eyesMs;
    out.state = ep.dropSince === null ? "baixando" : sleepy ? "caida" : "baixa";
    if (ep.dropSince !== null && !ep.dropEmitted && out.downMs >= c.dropMs && sleepy) {
      ep.dropEmitted = true;
      out.event = { type: "queda", t, peakDeg: ep.peak, peakRollDeg: ep.peakRoll, byRoll: !!ep.byRoll, lost: ep.lost };
    } else if (ep.dropSince !== null && !ep.dropEmitted && !ep.lowEmitted && out.downMs >= c.downOpenMs && !sleepy) {
      ep.lowEmitted = true;
      out.event = { type: "cabeca_baixa", t, peakDeg: ep.peak, peakRollDeg: ep.peakRoll, byRoll: !!ep.byRoll };
    }
    out.sustained = ep.dropSince !== null && (ep.maxRunMs >= c.redEyesMs || (sleepy && ep.lostMs >= c.redLostMs));
    return out;
  }
  _newEpisode(t, onsetMs, peak, lost) {
    return { start: t, onsetMs, peak, peakRoll: 0, byRoll: false, aside: false, dropSince: null, dropEmitted: false, lowEmitted: false, lost, lostSince: lost ? t : null, lostMs: 0, closedMs: 0, runStart: null, maxRunMs: 0 };
  }
  _finish(t, out, timedOut) {
    const ep = this.ep;
    const c = this.cfg;
    this.ep = null;
    out.state = timedOut ? "sem_rosto" : "neutra";
    const dur = t - ep.start;
    if (!ep.aside && !ep.dropEmitted && !timedOut && dur <= c.nodMaxMs && ep.onsetMs <= c.nodOnsetMs && ep.peak >= c.nodDeg && ep.maxRunMs >= c.eyesMs) {
      out.event = { type: "cabeceio", t, peakDeg: ep.peak, peakRollDeg: ep.peakRoll, durMs: dur };
    }
    return out;
  }
}

/** Eventos que valem `points` ao acontecer e perdem valor linearmente até zerar em `decayMs`. */
export class DecayingEvents {
  constructor({ points, decayMs, cap = Infinity }) {
    this.pts = points;
    this.decayMs = decayMs;
    this.cap = cap;
    this.items = [];
  }
  add(t, meta = {}) {
    this.items.push({ ...meta, t });
  }
  prune(t) {
    while (this.items.length && t - this.items[0].t >= this.decayMs) this.items.shift();
  }
  points(t) {
    this.prune(t);
    let s = 0;
    for (const e of this.items) s += this.pts * Math.max(0, 1 - (t - e.t) / this.decayMs);
    return Math.min(this.cap, s);
  }
  count(t, windowMs = this.decayMs) {
    let n = 0;
    for (const e of this.items) if (t - e.t < windowMs) n++;
    return n;
  }
  last() {
    return this.items.length ? this.items[this.items.length - 1] : null;
  }
  reset() {
    this.items = [];
  }
}

/**
 * Modelo de aprendizado de máquina (LSTM do Exp. 1) como CONSELHEIRO: no máximo +10 pontos, e só
 * quando as últimas 5 predições foram "sonolento" com probabilidade >= 0,6. Nunca dispara alarme
 * sozinho (o Exp. 5b mostrou que a generalização para uma pessoa nova é limitada).
 */
export class MLAdvisory {
  constructor(cfg = CFG.score.ml) {
    this.cfg = cfg;
    this.reset();
  }
  reset() {
    this.hist = [];
    this.last = null;
  }
  update(probs) {
    const p = Array.from(probs);
    const cls = argmax(p);
    this.last = { probs: p, cls };
    this.hist.push(cls === 2 && p[2] >= this.cfg.minProb);
    if (this.hist.length > this.cfg.consecutive) this.hist.shift();
  }
  get eligible() {
    return this.hist.length >= this.cfg.consecutive && this.hist.every(Boolean);
  }
  get points() {
    return this.eligible ? this.cfg.points : 0;
  }
}

/**
 * Nível exibido (0-100) e estado com histerese. Sobe na hora; desce no máximo `maxFallPerS` por
 * segundo. Condição forçada (microssono/cabeça caída) coloca o nível em pelo menos `forcedFloor`.
 *   ALERTA -> ATENÇÃO em >= 40 (volta em < 35); ATENÇÃO -> SONOLENTO em >= 70 (volta em < 60).
 * A margem de saída do vermelho (60) vale só para o vermelho que ENTROU pelo nível: ela existe para o
 * estado não piscar quando o nível oscila perto de 70. Quando o vermelho foi FORÇADO, o nível não estava
 * perto do limiar; ao terminar a condição, o estado segue o nível com o limiar de entrada (70).
 */
export class LevelController {
  constructor(score = CFG.score, levels = CFG.levels) {
    this.score = score;
    this.levels = levels;
    this.reset();
  }
  reset() {
    this.level = 0;
    this.state = 0;
    this.lastT = null;
    this.redByLevel = false; // o vermelho atual entrou pelo nível (>= 70), e não por condição forçada
  }
  update(t, target, forced = false) {
    const dt = this.lastT === null ? 0 : Math.max(0, (t - this.lastT) / 1000);
    this.lastT = t;
    let goal = Math.max(0, Math.min(100, Number.isFinite(target) ? target : 0));
    if (forced) goal = Math.max(goal, this.score.forcedFloor);
    this.level = goal >= this.level ? goal : Math.max(goal, this.level - this.score.maxFallPerS * dt);
    const L = this.level;
    const c = this.levels;
    const prev = this.state;
    let s;
    if (forced) {
      s = 2;
      this.redByLevel = false;
    } else if (prev === 2 && this.redByLevel ? L >= c.drowsyExit : L >= c.drowsy) {
      s = 2;
      if (prev !== 2) this.redByLevel = true;
    } else if (L >= c.attention || (prev >= 1 && L >= c.attentionExit)) s = 1;
    else s = 0;
    if (s !== 2) this.redByLevel = false;
    this.state = s;
    return { level: L, state: s, changed: s !== prev };
  }
}

/**
 * O que tocar/mostrar a cada quadro (sem áudio: a interface executa).
 *   VERMELHO: sirene enquanto durar (mínimo 2 s), flash e vibração; "Estou acordado" corta a sirene na
 *             hora e silencia o vermelho por nível por 30 s (o forçado — microssono/cabeça — nunca é
 *             silenciado).
 *   AMARELO : um "ding" ao entrar em ATENÇÃO e a cada novo sinal, no máximo um a cada 20 s.
 */
export class AlarmController {
  constructor(cfg = CFG.alarm) {
    this.cfg = cfg;
    this.reset();
  }
  reset() {
    this.prevState = 0;
    this.sirenSince = null;
    this.lastDing = -Infinity;
    this.lastVib = -Infinity;
    this.snoozeUntil = -Infinity;
    this.alarms = 0;
    this.dings = 0;
  }
  snooze(t) {
    this.snoozeUntil = t + this.cfg.snoozeMs;
  }
  update(t, { state = 0, forced = false, newEvents = 0 } = {}) {
    const c = this.cfg;
    const out = { siren: false, ding: false, vibrate: null, flash: false, alarmStart: false, warnStart: false, snoozed: false };
    if (state === 2 && this.prevState !== 2) {
      out.alarmStart = true;
      this.alarms += 1;
    }
    if (state === 1 && this.prevState === 0) out.warnStart = true;
    const snoozed = state === 2 && !forced && t < this.snoozeUntil;
    out.snoozed = snoozed;
    if (state === 2 && !snoozed) {
      if (this.sirenSince === null) this.sirenSince = t;
      out.siren = true;
    } else if (snoozed) {
      this.sirenSince = null; // "Estou acordado" corta a sirene na hora (o mínimo de 2 s não se aplica)
    } else if (this.sirenSince !== null) {
      if (t - this.sirenSince < c.sirenMinMs) out.siren = true;
      else this.sirenSince = null;
    }
    out.flash = out.siren;
    if (out.siren && t - this.lastVib >= c.vibrateEveryMs) {
      out.vibrate = "alarme";
      this.lastVib = t;
    }
    if (state === 1 && (out.warnStart || newEvents > 0) && t - this.lastDing >= c.dingGapMs) {
      out.ding = true;
      this.lastDing = t;
      this.dings += 1;
      if (!out.vibrate) out.vibrate = "aviso";
    }
    this.prevState = state;
    return out;
  }
}

// ---------------------------------------------------------------------------
// Volume do alarme. A interface aplica o ganho no Web Audio (app.js); aqui fica só a regra, testável
// em Node.
//   - padrão 20 %: baixo de propósito, para testar perto de outras pessoas sem susto; na demonstração
//     o usuário sobe o controle "Volume do alarme" (o valor fica salvo no aparelho, localStorage);
//   - ?volume=0..100 na URL sobrepõe o valor salvo SEM gravar (os testes automáticos usam ?volume=0);
//   - curva quadrática, ganho = (v/100)^2, porque a percepção de volume é aproximadamente
//     logarítmica: 20 % -> ganho 0,04 (-28 dB em relação a 100 %), 50 % -> 0,25 (-12 dB), 100 % -> 1.
// O som final continua limitado pelo volume de mídia do aparelho (o app não consegue aumentá-lo).
// ---------------------------------------------------------------------------
export const SOUND = deepFreeze({ defaultVolumePct: 20, storageKey: "sonolencia.volumeAlarme", curveExp: 2 });

/** Texto/número -> volume inteiro 0-100 (limitado); null se vazio ou inválido. Aceita "35", "35,5", "35%". */
export function parseVolumePct(v) {
  if (v === null || v === undefined || typeof v === "boolean") return null;
  const s = String(v).trim().replace(",", ".").replace(/%$/, "");
  if (s === "") return null;
  const n = Number(s);
  if (!Number.isFinite(n)) return null;
  return Math.round(Math.min(100, Math.max(0, n)));
}

/** Volume efetivo: ?volume= da URL (não é gravado) > valor salvo > padrão (20 %). */
export function resolveVolume({ urlValue = null, storedValue = null } = {}, cfg = SOUND) {
  const u = parseVolumePct(urlValue);
  if (u !== null) return { pct: u, source: "url" };
  const st = parseVolumePct(storedValue);
  if (st !== null) return { pct: st, source: "salvo" };
  return { pct: cfg.defaultVolumePct, source: "padrão" };
}

/** Ganho do Web Audio para um volume 0-100 (curva quadrática; 0 = silêncio absoluto). */
export function volumeGain(pct, cfg = SOUND) {
  const v = parseVolumePct(pct);
  if (v === null || v <= 0) return 0;
  return Math.pow(v / 100, cfg.curveExp);
}

const EVENT_LABEL = {
  bocejo: "bocejo",
  queda: "cabeça caindo",
  cabeceio: "cabeceio",
  fechamento: "olhos fechados",
  microssono: "microssono",
  cabeca_baixa: "cabeça baixa com olhos abertos",
};

/**
 * Orquestra calibração, sinais, nível e alarmes. Puro: recebe os sinais de um quadro e devolve
 * tudo o que a interface precisa. O mesmo código roda no app e na validação offline
 * (tools/app_validacao/replay_motor.mjs), por isso nada aqui depende do navegador.
 *
 * Quadro de entrada `f`:
 *   { face, ear: {left,right,mean}, mar, blend: {eyeBlinkLeft, eyeBlinkRight, jawOpen,
 *     eyeLookDownLeft, eyeLookDownRight} | null, pose: {pitch,yaw,roll} | null,
 *     mlProbs: [3] | null (só nos quadros em que o LSTM fez uma predição nova) }
 */
export class DrowsinessMonitor {
  constructor(cfg = CFG) {
    this.cfg = cfg;
    const S = cfg.score;
    this.closure = new ClosureTracker(cfg.eyes);
    this.perclos = new PerclosTime(cfg.perclos); // só fechamentos >= cfg.perclos.minEpisodeMs
    this.yawn = new YawnDetector(cfg.yawn);
    this.head = new HeadDropDetector(cfg.head);
    this.ml = new MLAdvisory(S.ml);
    this.levelCtl = new LevelController(S, cfg.levels);
    this.alarm = new AlarmController(cfg.alarm);
    this.evLong = new DecayingEvents({ points: S.longClosure.points, decayMs: S.longClosure.decayMs, cap: S.closuresCap });
    this.evMicro = new DecayingEvents({ points: S.microsleep.points, decayMs: S.microsleep.decayMs, cap: S.closuresCap });
    this.evYawn = new DecayingEvents({ points: S.yawn.points, decayMs: S.yawn.decayMs, cap: S.yawn.cap });
    this.evHead = new DecayingEvents({ points: S.head.points, decayMs: S.head.decayMs, cap: S.head.cap });
    this.phase = "inativo";
    this.baseline = null;
    this.nextId = 1;
    this._resetRun();
  }
  _resetRun() {
    for (const k of ["closure", "perclos", "yawn", "head", "ml", "levelCtl", "alarm", "evLong", "evMicro", "evYawn", "evHead"]) this[k].reset();
    this.blinks = [];
    this.baseBlinks = [];
    this.blinkBaseMs = null;
    this.openMicro = null;
    this.openYawn = null;
    this.lastT = null;
    this.lastFaceT = null;
    this.prevClosed = false; // estado fundido do quadro anterior (histerese do caminho A)
    this.wasForced = null; //   motivo da condição forçada no quadro anterior (ou null)
    this.forcedEnd = null; //   {t, kind} do fim da última condição forçada (piso de ATENÇÃO por 1 min)
    this.stats = { eventos: { bocejo: 0, queda: 0, cabeceio: 0, fechamento: 0, microssono: 0, cabeca_baixa: 0 }, tempoEstadoMs: [0, 0, 0], alarmes: 0, avisos: 0, dings: 0 };
  }
  /** Reinicia tudo e começa a fase 1 (10 s olhando para a frente). */
  startCalibration(t) {
    this._resetRun();
    this.baseline = null;
    this.phase = "calibrando";
    this.cal = { ear: [], blend: [], jaw: [], mar: [], pitch: [], yaw: [], roll: [], closedN: 0, blendN: 0, faceMs: 0, lastT: null, t0: t };
  }
  get calibrated() {
    return this.baseline !== null;
  }
  get blinkBaselineReady() {
    return this.blinkBaseMs !== null;
  }
  _calibUpdate(t, f) {
    const c = this.cal;
    const dt = c.lastT === null ? 0 : Math.min(100, Math.max(0, t - c.lastT));
    c.lastT = t;
    if (!f.face) return;
    c.faceMs += dt;
    if (Number.isFinite(f.ear?.mean)) c.ear.push(f.ear.mean);
    const b = f.blend || {};
    if (Number.isFinite(b.eyeBlinkLeft) && Number.isFinite(b.eyeBlinkRight)) {
      c.blend.push((b.eyeBlinkLeft + b.eyeBlinkRight) / 2);
      c.blendN += 1;
      if (Math.min(b.eyeBlinkLeft, b.eyeBlinkRight) >= this.cfg.eyes.blendClosed) c.closedN += 1; // olho fechado na calibração
    }
    if (Number.isFinite(b.jawOpen)) c.jaw.push(b.jawOpen);
    if (Number.isFinite(f.mar)) c.mar.push(f.mar);
    if (f.pose && Number.isFinite(f.pose.pitch)) {
      c.pitch.push(f.pose.pitch);
      c.yaw.push(f.pose.yaw);
      c.roll.push(f.pose.roll);
    }
  }
  _finishCalibration(t) {
    const c = this.cal;
    const earMed = median(c.ear);
    const bl = {
      earMedian: earMed,
      earThr: this.cfg.calib.earFactor * earMed,
      blend0: c.blend.length ? median(c.blend) : null,
      jaw0: c.jaw.length ? median(c.jaw) : 0,
      mar0: c.mar.length ? median(c.mar) : null,
      pitch0: c.pitch.length ? median(c.pitch) : null,
      yaw0: c.yaw.length ? median(c.yaw) : null,
      roll0: c.roll.length ? median(c.roll) : null,
      frames: c.ear.length,
      closedFrac: c.blendN ? c.closedN / c.blendN : null,
      warnings: [],
    };
    if (earMed < 0.15) bl.warnings.push("EAR de olho aberto muito baixo: recalibre com os olhos bem abertos e o rosto iluminado.");
    // Acordado, a pessoa passa ~2-5 % do tempo de olho fechado (piscadas); acima de 10 % a base sai "sonolenta".
    if (bl.closedFrac !== null && bl.closedFrac > this.cfg.calib.maxClosedFrac)
      bl.warnings.push(`Olhos fechados em ${fmtBR(100 * bl.closedFrac, 0)} % da calibração (esperado: menos de ${fmtBR(100 * this.cfg.calib.maxClosedFrac, 0)} %): toque em Recalibrar e olhe para a frente com os olhos abertos.`);
    if (bl.yaw0 !== null && Math.abs(bl.yaw0) > 35) bl.warnings.push("Rosto muito de lado para a câmera: centralize o celular.");
    if (bl.pitch0 === null) bl.warnings.push("Sem pose da cabeça (matriz indisponível): detector de cabeça desligado.");
    this.baseline = bl;
    this.yawn.setBaseline({ jaw0: bl.jaw0, mar0: bl.mar0 });
    this.head.setBaseline({ pitch: bl.pitch0, yaw: bl.yaw0, roll: bl.roll0 });
    this.phase = "ativo";
    this.cal = null;
    return bl;
  }
  _event(type, t, extra = {}) {
    const ev = { id: this.nextId++, type, t, label: EVENT_LABEL[type] || type, ...extra };
    if (type in this.stats.eventos) this.stats.eventos[type] += 1;
    return ev;
  }
  /** Processa um quadro. `t` em ms (monotônico). */
  update(t, f) {
    const cfg = this.cfg;
    const S = cfg.score;
    const dt = this.lastT === null ? 0 : Math.max(0, t - this.lastT);
    this.lastT = t;
    const face = !!f.face;
    if (face) this.lastFaceT = t;
    const events = [];
    const updates = [];
    const res = { phase: this.phase, events, updates, calib: null, eyes: null, yawn: null, head: null, score: null, alarm: null };

    if (this.phase === "calibrando") {
      this._calibUpdate(t, f);
      const prog = Math.min(1, this.cal.faceMs / cfg.calib.neutralMs);
      res.calib = { phase: 1, progress: prog, face };
      if (prog >= 1 && this.cal.ear.length >= 10) {
        const bl = this._finishCalibration(t);
        events.push(this._event("calibrado", t, { baseline: bl }));
        res.phase = this.phase;
        res.calib = { phase: 2, progress: 0, face, baseline: bl };
      }
      res.score = { raw: 0, level: 0, state: 0, forced: null, contributions: [] };
      res.alarm = this.alarm.update(t, { state: 0 });
      return res;
    }
    if (this.phase !== "ativo") {
      res.score = { raw: 0, level: 0, state: 0, forced: null, contributions: [] };
      res.alarm = this.alarm.update(t, { state: 0 });
      return res;
    }

    const bl = this.baseline;
    const b = f.blend || {};
    const hd = face ? this.head.delta(f.pose) : null;
    const lookDown = Number.isFinite(b.eyeLookDownLeft) && Number.isFinite(b.eyeLookDownRight) ? (b.eyeLookDownLeft + b.eyeLookDownRight) / 2 : null;

    // ---- olhos ----
    const squint = Number.isFinite(b.eyeSquintLeft) && Number.isFinite(b.eyeSquintRight) ? (b.eyeSquintLeft + b.eyeSquintRight) / 2 : null;
    const fu = face
      ? fuseEyeClosure({ ear: f.ear?.mean, earThr: bl.earThr, blinkL: b.eyeBlinkLeft, blinkR: b.eyeBlinkRight, lookDown, dPitchDown: hd ? hd.dPitch : null, squint, dYaw: hd ? hd.dYaw : null, blend0: bl.blend0, prevClosed: this.prevClosed }, cfg.eyes)
      : { closed: false, byBlend: false, byEar: false, gated: false, headDown: false, gazeDown: false, squinting: false, aside: false, blendMean: null, blendMin: null };
    this.prevClosed = face && fu.closed;
    const cl = this.closure.update(t, fu.closed, face);
    if (face) this.perclos.update(t, fu.closed);
    else this.perclos.faceLost();
    if (cl.microsleep) {
      const ev = this._event("microssono", t, { durMs: null });
      this.evMicro.add(t, { id: ev.id });
      this.openMicro = ev.id;
      events.push(ev);
    }
    if (cl.ended) {
      const d = cl.ended.durMs;
      if (d >= cfg.eyes.microsleepMs && this.openMicro !== null) {
        updates.push({ id: this.openMicro, durMs: d, lost: cl.ended.lost });
        this.openMicro = null;
      } else if (d >= cfg.eyes.longMs && d < cfg.eyes.microsleepMs) {
        const ev = this._event("fechamento", t, { durMs: d });
        this.evLong.add(t, { id: ev.id });
        events.push(ev);
      } else if (d >= cfg.eyes.blinkMinMs && d < cfg.eyes.blinkMaxMs && !cl.ended.lost) {
        this.blinks.push({ t, durMs: d });
        if (this.blinkBaseMs === null) {
          this.baseBlinks.push(d);
          if (this.baseBlinks.length >= cfg.calib.nBlinks) {
            this.blinkBaseMs = median(this.baseBlinks);
            events.push(this._event("base_piscadas", t, { baseMs: this.blinkBaseMs }));
          }
        }
      }
    }
    while (this.blinks.length && t - this.blinks[0].t > S.blinkDur.windowMs) this.blinks.shift();

    // ---- bocejo ----
    const ya = this.yawn.update(t, { face, jawOpen: b.jawOpen, mar: f.mar });
    if (ya.event) {
      const ev = this._event("bocejo", t, { durMs: null, peak: ya.event.peak });
      this.evYawn.add(t, { id: ev.id });
      this.openYawn = ev.id;
      events.push(ev);
    }
    if (ya.ended && this.openYawn !== null) {
      updates.push({ id: this.openYawn, durMs: ya.ended.durMs, peak: ya.ended.peak });
      this.openYawn = null;
    }

    // ---- cabeça ----
    const he = this.head.update(t, { face, pose: f.pose, eyesClosed: fu.closed });
    if (he.event) {
      const ev = this._event(he.event.type, t, { peakDeg: he.event.peakDeg, peakRollDeg: he.event.peakRollDeg ?? null, byRoll: !!he.event.byRoll, lost: !!he.event.lost, durMs: he.event.durMs ?? null });
      if (he.event.type !== "cabeca_baixa") this.evHead.add(t, { id: ev.id }); // distração: registra, sem pontos
      events.push(ev);
    }

    // ---- modelo (conselheiro) ----
    if (f.mlProbs) this.ml.update(f.mlProbs);

    // ---- pontos ----
    const contrib = [];
    const P = this.perclos.value;
    const pPerclos = ramp(P, S.perclos.from, S.perclos.at, S.perclos.points, S.perclos.cap);
    contrib.push({ key: "perclos", points: pPerclos, label: `PERCLOS ${fmtBR(P * 100, 1)} % em 60 s, só fechamentos >= 0,5 s (referência do app: 15 %)` });
    const nLong1m = this.evLong.count(t, S.longClosure.countMs) + this.evMicro.count(t, S.longClosure.countMs);
    const pClos = Math.min(S.closuresCap, this.evLong.points(t) + this.evMicro.points(t));
    const nMicro5 = this.evMicro.count(t);
    const nLong2 = this.evLong.count(t);
    const microMin = Math.round(S.microsleep.decayMs / 60000);
    contrib.push({
      key: "fechamentos",
      points: pClos,
      label: `${nMicro5 ? `${nMicro5} microssono(s) em ${microMin} min` : ""}${nMicro5 && nLong2 ? " e " : ""}${nLong2 ? `${nLong2} fechamento(s) de 0,5-1 s em 2 min` : ""}` || "fechamentos longos",
    });
    const nYawn = this.evYawn.count(t);
    contrib.push({ key: "bocejos", points: this.evYawn.points(t), label: `${nYawn} bocejo(s) nos últimos 5 min` });
    const nHead = this.evHead.count(t);
    contrib.push({ key: "cabeca", points: this.evHead.points(t), label: `${nHead} queda(s)/cabeceio(s) da cabeça nos últimos 5 min` });
    let blinkRatio = null;
    let curBlinkMs = null;
    if (this.blinkBaseMs && this.blinks.length >= S.blinkDur.minBlinks) {
      curBlinkMs = median(this.blinks.map((x) => x.durMs));
      blinkRatio = curBlinkMs / this.blinkBaseMs;
    }
    const pBlink = ramp(blinkRatio, S.blinkDur.from, S.blinkDur.at, S.blinkDur.points, S.blinkDur.cap);
    contrib.push({ key: "piscadas", points: pBlink, label: blinkRatio ? `piscadas ${fmtBR((blinkRatio - 1) * 100, 0)} % mais longas que na calibração (${Math.round(curBlinkMs)} x ${Math.round(this.blinkBaseMs)} ms)` : "duração das piscadas" });
    // Modelo (conselheiro): só soma se os sinais visíveis, sozinhos, já dão ATENÇÃO (>= 40).
    const baseRaw = contrib.reduce((a, c) => a + c.points, 0);
    const mlGate = this.ml.eligible && baseRaw >= S.ml.minBase;
    contrib.push({ key: "modelo", points: mlGate ? this.ml.points : 0, label: "modelo LSTM (experimental): \"sonolento\" em 5 predições seguidas, com os sinais visíveis já em ATENÇÃO" });

    // ---- condição forçada ----
    let forced = null;
    let forcedKind = null;
    if (cl.closedMs >= cfg.eyes.microsleepMs && this.closure.active) {
      forced = `olhos fechados há ${fmtBR(cl.closedMs / 1000, 1)} s (microssono)`;
      forcedKind = "microssono";
    } else if (he.sustained) {
      forced = `cabeça ${he.byRoll ? "tombada para o lado" : "caída"} há ${fmtBR(he.downMs / 1000, 1)} s${he.eyesMs >= cfg.head.redEyesMs ? " com os olhos fechados" : " (rosto fora do quadro)"}`;
      forcedKind = "cabeça caída";
    }
    if (this.wasForced && !forced) this.forcedEnd = { t, kind: this.wasForced };
    this.wasForced = forcedKind;
    // Piso de ATENÇÃO depois de um vermelho forçado (1 min): aparece em "Por quê?" com o que falta até 40.
    let postFloor = 0;
    if (!forced && this.forcedEnd && t - this.forcedEnd.t < S.postForced.holdMs) {
      const sum = contrib.reduce((a, c) => a + c.points, 0);
      postFloor = Math.max(0, S.postForced.floor - sum);
      const left = Math.ceil((S.postForced.holdMs - (t - this.forcedEnd.t)) / 1000);
      contrib.push({ key: "pos_evento", points: postFloor, label: `${this.forcedEnd.kind} há pouco: ATENÇÃO mantida por 1 min (faltam ${left} s)` });
    }
    const raw = Math.min(100, contrib.reduce((a, c) => a + c.points, 0));

    const lv = this.levelCtl.update(t, raw, !!forced);
    const nNew = events.filter((e) => e.type in this.stats.eventos && e.type !== "cabeca_baixa").length;
    const al = this.alarm.update(t, { state: lv.state, forced: !!forced, newEvents: nNew });
    if (al.alarmStart) {
      this.stats.alarmes += 1;
      events.push(this._event("alarme", t, { motivo: forced || "nível de sonolência >= 70" }));
    }
    if (al.warnStart) {
      this.stats.avisos += 1;
      events.push(this._event("aviso", t, {}));
    }
    if (al.ding) this.stats.dings += 1;
    this.stats.tempoEstadoMs[lv.state] += dt;

    res.phase = this.phase;
    res.calib = this.blinkBaseMs === null ? { phase: 2, progress: Math.min(1, this.baseBlinks.length / cfg.calib.nBlinks), blinks: this.baseBlinks.length, need: cfg.calib.nBlinks } : null;
    res.eyes = {
      closed: fu.closed,
      byBlend: fu.byBlend,
      byEar: fu.byEar,
      gated: fu.gated,
      headDown: fu.headDown,
      gazeDown: fu.gazeDown,
      squinting: fu.squinting,
      aside: fu.aside,
      squint,
      blendMean: fu.blendMean,
      lookDown,
      closedMs: cl.closedMs,
      perclos: P,
      perclosAll: this.perclos.valueAll,
      perclosObservedMs: this.perclos.observedMs,
      longPerMin: nLong1m,
      microsleeps5min: nMicro5,
      blinkRatio,
      curBlinkMs,
      baseBlinkMs: this.blinkBaseMs,
    };
    res.yawn = { open: ya.open, openMs: ya.openMs, jawOpen: b.jawOpen ?? null, count5min: nYawn };
    res.head = { state: he.state, dPitch: he.dPitch, dYaw: he.dYaw, dRoll: he.dRoll, downMs: he.downMs, eyesMs: he.eyesMs, sustained: he.sustained, byRoll: !!he.byRoll, count5min: nHead };
    res.score = {
      raw,
      level: lv.level,
      state: lv.state,
      stateName: STATE_NAMES[lv.state],
      forced,
      contributions: contrib.filter((c) => c.points > 0.05).sort((a, b) => b.points - a.points),
      all: contrib,
      mlEligible: this.ml.eligible,
      mlCounted: mlGate,
      postForcedMs: this.forcedEnd && !forced ? Math.max(0, S.postForced.holdMs - (t - this.forcedEnd.t)) : 0,
    };
    res.alarm = al;
    return res;
  }
  /** "Estou acordado": silencia o vermelho por nível durante 30 s. */
  acknowledge(t) {
    this.alarm.snooze(t);
  }
}
