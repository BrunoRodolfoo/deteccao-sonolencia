/**
 * Teste INDEPENDENTE do motor do app (app/pwa/engine.js, detector v2.1) em sono real: UTA-RLDD.
 *
 *   node tools/app_validacao/validar_motor_uta.mjs [--res <pasta>] [--parcial 1]
 *
 * Por que: o app foi ajustado no DMD (sonolência encenada e direção real de pessoas acordadas,
 * results/app_validacao_dmd.md) e testado no navegador com 3 pessoas da UTA-RLDD (Fold1/09, Fold3/31 e
 * Fold5/51, results/app_validacao.md). Aqui ele roda em pessoas que não entraram em nenhuma dessas etapas,
 * com a sonolência declarada pela própria pessoa (KSS) em vídeos de cerca de 10 min.
 *
 * Amostra (regra fixa, sem escolha por vídeo): todos os sujeitos dos folds oficiais 2 e 4 da UTA-RLDD, os dois
 * folds sem nenhum sujeito do teste no navegador; de cada um, o vídeo 0 (acordado, KSS 1 a 3) e o vídeo 10
 * (sonolento, KSS 8 e 9). Folds, sujeitos e rótulos vêm de data/features/final/uta/windows_10s.csv.
 *
 * Entrada: sinais por quadro do MESMO Face Landmarker da PWA, com blendshapes e matriz de pose
 * (tools/app_validacao/extrair_sinais_app.py --uta-folds 2 4 --uta-rotulos 0 10 ->
 * data/signals/app_tasks/uta/<id>.csv.gz + .json). O código avaliado é o do app, importado sem cópia, com a
 * configuração final (E.CFG) e sem o LSTM (mlProbs = null), como em validar_motor_dmd.mjs.
 *
 * Calibração:
 *  - principal: a do app, nos 10 s iniciais com rosto de CADA vídeo (o que o app faz no início de cada viagem);
 *  - variante (só vídeos sonolentos): o monitor calibra nos 10 s iniciais com rosto do vídeo ACORDADO da mesma
 *    pessoa e depois recebe o vídeo sonolento (quem calibra acordado e fica com sono depois). A única coisa que
 *    passa de um vídeo para o outro é a base da calibração (limiar do EAR, eyeBlink, jawOpen e MAR de repouso,
 *    pose da cabeça). Os quadros avaliados são os mesmos da principal: do fim dos 10 s iniciais com rosto do
 *    próprio vídeo sonolento até o fim.
 *
 * Métricas por vídeo (depois da calibração): fração de quadros com rosto (vídeo inteiro), % do tempo em verde,
 * amarelo e vermelho, alarmes (entradas no vermelho), avisos (entradas no amarelo), nível exibido médio (ponderado
 * pelo tempo) e máximo, eventos (fechamentos de 0,5 a 1 s, microssonos, bocejos, quedas e cabeceios da cabeça,
 * cabeça baixa com olhos abertos) e, para a leitura, piscadas por minuto e % dos quadros com olho fechado.
 * Conjunto: falso alarme nos acordados (% de vídeos com algum alarme, % do tempo em vermelho), % dos sonolentos que
 * chegam ao amarelo e ao vermelho, comparação pareada por pessoa (% do tempo fora do verde) e AUC (sonolento contra
 * acordado) do % fora do verde e do nível médio, com IC de 95 % por bootstrap de sujeitos (1.000 reamostras,
 * semente 42, gerador mulberry32, percentis 2,5 e 97,5; cada sujeito sorteado entra com os seus dois vídeos).
 *
 * Saídas: results/app_validacao_uta.csv (uma linha por vídeo; as colunas cal_acordado_* são da variante),
 * results/app_validacao_uta.md e results/logs/app_validacao_uta_<data>.log (com o log da extração,
 * data/signals/app_tasks/uta/extracao.log). Com --res <pasta>, tudo vai para essa pasta (ensaios).
 * Nenhum quadro de vídeo é lido ou gravado aqui: só os sinais numéricos.
 */
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";
import * as E from "../../app/pwa/engine.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => (a.startsWith("--") && acc.push([a.slice(2), arr[i + 1]]), acc), []));
const RES = args.res ? path.resolve(args.res) : path.join(ROOT, "results");
const PARCIAL = !!args.parcial; // ensaio com a extração ainda em andamento: usa só os sujeitos com os 2 vídeos prontos
const SIG = path.join(ROOT, "data/signals/app_tasks/uta");
const WINDOWS = path.join(ROOT, "data/features/final/uta/windows_10s.csv");
const EXTRACAO_LOG = path.join(SIG, "extracao.log");
const FOLDS = [2, 4];
const NAVEGADOR = ["F1_09", "F3_31", "F5_51"]; // sujeitos do teste no navegador (results/app_validacao.md)
const ROTULO = { 0: "acordado (KSS 1 a 3)", 10: "sonolento (KSS 8 e 9)" };
const N_BOOT = 1000;
const SEED = 42;

const _d = new Date();
const _p = (v) => String(v).padStart(2, "0");
const stamp = `${_d.getFullYear()}${_p(_d.getMonth() + 1)}${_p(_d.getDate())}-${_p(_d.getHours())}${_p(_d.getMinutes())}${_p(_d.getSeconds())}`;
const logRel = `results/logs/app_validacao_uta_${stamp}.log`;
const logPath = path.join(RES, "logs", `app_validacao_uta_${stamp}.log`);
const logLines = [];
const log = (...a) => {
  const s = a.join(" ");
  logLines.push(s);
  console.log(s);
};
const md5 = (p) => crypto.createHash("md5").update(fs.readFileSync(p)).digest("hex");

// ---------------------------------------------------------------------------
// Leitura (igual a validar_motor_dmd.mjs)
// ---------------------------------------------------------------------------
function readSignals(file) {
  const txt = zlib.gunzipSync(fs.readFileSync(file)).toString("utf8").trim().split("\n");
  const head = txt[0].split(",");
  const idx = Object.fromEntries(head.map((h, i) => [h, i]));
  const rows = new Array(txt.length - 1);
  for (let k = 1; k < txt.length; k++) {
    const c = txt[k].split(",");
    const num = (name) => {
      const v = c[idx[name]];
      return v === "" || v === undefined ? NaN : Number(v);
    };
    const face = num("face") === 1;
    const m = [];
    for (let i = 0; i < 16; i++) m.push(num(`m${i}`));
    rows[k - 1] = {
      frame: num("frame"),
      t: num("t_ms"),
      face,
      earL: num("ear_l"),
      earR: num("ear_r"),
      mar: num("mar"),
      jaw: num("jaw_open"),
      bl: num("blink_l"),
      br: num("blink_r"),
      ldl: num("look_down_l"),
      ldr: num("look_down_r"),
      sql: num("squint_l"),
      sqr: num("squint_r"),
      pose: face ? E.poseFromMatrix(m) : null,
    };
  }
  return rows;
}

function frameInput(r) {
  return {
    face: r.face,
    ear: { left: r.earL, right: r.earR, mean: (r.earL + r.earR) / 2 },
    mar: r.mar,
    blend: r.face ? { eyeBlinkLeft: r.bl, eyeBlinkRight: r.br, jawOpen: r.jaw, eyeLookDownLeft: r.ldl, eyeLookDownRight: r.ldr, eyeSquintLeft: r.sql, eyeSquintRight: r.sqr } : null,
    pose: r.pose,
    mlProbs: null,
  };
}

/** Sujeito, fold e rótulo de cada vídeo da UTA-RLDD (colunas sujeito, fold, rotulo_kss, arquivo, video). */
function readWindowsMeta(file) {
  const lines = fs.readFileSync(file, "utf8").trim().split("\n");
  const head = lines[0].split(",");
  const ix = Object.fromEntries(["sujeito", "fold", "rotulo_kss", "arquivo", "video"].map((k) => [k, head.indexOf(k)]));
  if (Object.values(ix).some((v) => v < 0)) throw new Error(`colunas ausentes em ${path.relative(ROOT, file)}`);
  const out = new Map();
  for (let k = 1; k < lines.length; k++) {
    const c = lines[k].split(",");
    const v = c[ix.video];
    const o = { sujeito: c[ix.sujeito], fold: Number(c[ix.fold]), rotulo_kss: Number(c[ix.rotulo_kss]), arquivo: c[ix.arquivo] };
    const prev = out.get(v);
    if (prev && (prev.sujeito !== o.sujeito || prev.fold !== o.fold || prev.rotulo_kss !== o.rotulo_kss)) throw new Error(`metadados inconsistentes para ${v}`);
    out.set(v, o);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Monitor do app
// ---------------------------------------------------------------------------
/**
 * Roda o DrowsinessMonitor do app quadro a quadro.
 *  - sem calRows: calibra nos 10 s iniciais com rosto do próprio vídeo (como o app);
 *  - com calRows: calibra nos 10 s iniciais com rosto de calRows (vídeo acordado) e depois recebe os quadros de
 *    `rows` com t > fromT, deslocados no tempo para continuar logo depois do fim dessa calibração.
 * Devolve, por quadro avaliado, o estado (0 verde, 1 amarelo, 2 vermelho; -1 = não avaliado), o nível exibido e
 * o olho fechado da fusão do app.
 */
function runMonitor(rows, { calRows = null, fromT = null } = {}) {
  const mon = new E.DrowsinessMonitor(E.CFG);
  const n = rows.length;
  let i0 = 0, offset = 0, calT = null, baseline = null;
  if (calRows) {
    mon.startCalibration(calRows[0].t);
    let tEnd = null;
    for (const r of calRows) {
      const res = mon.update(r.t, frameInput(r));
      if (res.events.some((e) => e.type === "calibrado")) { tEnd = r.t; break; }
    }
    if (tEnd === null) throw new Error("a calibração não terminou no vídeo acordado");
    baseline = mon.baseline;
    i0 = rows.findIndex((r) => r.t > fromT);
    const gap = i0 > 0 ? rows[i0].t - rows[i0 - 1].t : 0;
    offset = tEnd + gap - rows[i0].t;
    calT = fromT;
  } else {
    mon.startCalibration(rows[0].t);
  }
  const state = new Int8Array(n).fill(-1);
  const level = new Float32Array(n).fill(NaN);
  const closed = new Uint8Array(n);
  let perclosMax = 0;
  const motivos = [];
  for (let i = i0; i < n; i++) {
    const r = rows[i];
    const res = mon.update(r.t + offset, frameInput(r));
    for (const e of res.events) if (e.type === "alarme") motivos.push(e.motivo);
    if (!calRows && calT === null) {
      if (res.events.some((e) => e.type === "calibrado")) { calT = r.t; baseline = mon.baseline; }
      continue;
    }
    if (r.t <= calT) continue;
    state[i] = res.score.state;
    level[i] = res.score.level;
    if (res.eyes) {
      closed[i] = res.eyes.closed ? 1 : 0;
      if (Number.isFinite(res.eyes.perclos)) perclosMax = Math.max(perclosMax, res.eyes.perclos);
    }
  }
  if (calT === null) throw new Error("a calibração não terminou");
  return { mon, calT, baseline, state, level, closed, perclosMax, motivos };
}

/** Métricas de um vídeo a partir da execução do monitor (só quadros avaliados, tempo do próprio vídeo). */
function metrics(rows, run) {
  const n = rows.length;
  const tS = [0, 0, 0];
  let tt = 0, lvSum = 0, lvMax = 0, firstY = null, firstR = null, nFace = 0, nClosed = 0;
  for (let i = 1; i < n; i++) {
    const s = run.state[i];
    if (s < 0) continue;
    const d = rows[i].t - rows[i - 1].t;
    tt += d;
    tS[s] += d;
    lvSum += run.level[i] * d;
    lvMax = Math.max(lvMax, run.level[i]);
    if (s >= 1 && firstY === null) firstY = rows[i].t - run.calT;
    if (s === 2 && firstR === null) firstR = rows[i].t - run.calT;
    if (rows[i].face) { nFace++; nClosed += run.closed[i]; }
  }
  // piscadas: episódios de olho fechado de 50 a 500 ms (CFG.eyes.blinkMinMs/blinkMaxMs), como o app as conta
  const ct = new E.ClosureTracker(E.CFG.eyes);
  let blinks = 0;
  for (let i = 0; i < n; i++) {
    if (run.state[i] < 0) continue;
    const o = ct.update(rows[i].t, !!run.closed[i], rows[i].face);
    if (o.ended && !o.ended.lost && o.ended.durMs >= E.CFG.eyes.blinkMinMs && o.ended.durMs < E.CFG.eyes.blinkMaxMs) blinks++;
  }
  const st = run.mon.stats;
  const bl = run.baseline;
  // conferência: o tempo por estado somado aqui é o mesmo que o monitor acumula (stats.tempoEstadoMs)
  const dif = Math.abs(tt - st.tempoEstadoMs.reduce((a, b) => a + b, 0)) + st.tempoEstadoMs.reduce((a, v, k) => a + Math.abs(v - tS[k]), 0);
  if (dif > 1) throw new Error(`tempo por estado diverge do monitor (${dif.toFixed(1)} ms)`);
  const pct = (v) => (tt > 0 ? (100 * v) / tt : NaN);
  return {
    duracao_avaliada_s: tt / 1000,
    verde_pct: pct(tS[0]),
    amarelo_pct: pct(tS[1]),
    vermelho_pct: pct(tS[2]),
    fora_do_verde_pct: pct(tS[1] + tS[2]),
    alarmes: st.alarmes,
    alarmes_por_microssono: run.motivos.filter((m) => /microssono/.test(m)).length,
    alarmes_por_cabeca: run.motivos.filter((m) => /cabeça/.test(m)).length,
    alarmes_por_nivel: run.motivos.filter((m) => /nível/.test(m)).length,
    avisos: st.avisos,
    nivel_medio: tt > 0 ? lvSum / tt : NaN,
    nivel_max: lvMax,
    primeiro_amarelo_s: firstY === null ? null : firstY / 1000,
    primeiro_vermelho_s: firstR === null ? null : firstR / 1000,
    fechamentos_05_1s: st.eventos.fechamento,
    microssonos: st.eventos.microssono,
    bocejos: st.eventos.bocejo,
    quedas: st.eventos.queda,
    cabeceios: st.eventos.cabeceio,
    cabeca_baixa: st.eventos.cabeca_baixa,
    piscadas_min: tt > 0 ? blinks / (tt / 60000) : NaN,
    olho_fechado_pct: nFace ? (100 * nClosed) / nFace : NaN,
    perclos_max_pct: 100 * run.perclosMax,
    calib_limiar_ear: bl.earThr,
    calib_blink_repouso: bl.blend0,
    calib_olhos_fechados_pct: bl.closedFrac === null ? null : 100 * bl.closedFrac,
    calib_aviso: bl.warnings.length ? bl.warnings.join(" | ") : "",
    calib_pitch: bl.pitch0,
    calib_yaw: bl.yaw0,
  };
}

// ---------------------------------------------------------------------------
// Estatística
// ---------------------------------------------------------------------------
/** AUC de Mann-Whitney: P(sonolento > acordado), empates valem 1/2. */
function auc(pos, neg) {
  let s = 0;
  for (const a of pos) for (const b of neg) s += a > b ? 1 : a === b ? 0.5 : 0;
  return s / (pos.length * neg.length);
}
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** Percentil com interpolação linear (o mesmo critério padrão do numpy). */
function quantile(sorted, p) {
  const h = (sorted.length - 1) * p;
  const lo = Math.floor(h), hi = Math.ceil(h);
  return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
}
/** Teste do sinal bilateral exato (empates fora). */
function signTest(k, m) {
  if (m === 0) return NaN;
  const c = (nn, kk) => { let r = 1; for (let i = 1; i <= kk; i++) r = (r * (nn - kk + i)) / i; return r; };
  const lo = Math.min(k, m - k);
  let p = 0;
  for (let i = 0; i <= lo; i++) p += c(m, i);
  return Math.min(1, (2 * p) / 2 ** m);
}
const median = (v) => { const f = v.filter(Number.isFinite); return f.length ? E.median(f) : NaN; };

// ---------------------------------------------------------------------------
// Execução
// ---------------------------------------------------------------------------
const engineFile = path.join(ROOT, "app/pwa/engine.js");
const appVersion = (fs.readFileSync(path.join(ROOT, "app/pwa/app.js"), "utf8").match(/APP_VERSION\s*=\s*["']([^"']+)["']/) || [])[1] || "?";
log(`# Teste independente do motor do app na UTA-RLDD (sono real), ${new Date().toISOString()}`);
log(`# node ${process.version}; app ${appVersion}; engine.js md5 ${md5(engineFile)}; LSTM fora (mlProbs = null)`);
log(`# CFG = ${JSON.stringify(E.CFG)}`);
log(`# sinais: ${path.relative(ROOT, SIG)}; folds: ${FOLDS.join(", ")}; sujeitos do teste no navegador (excluídos): ${NAVEGADOR.join(", ")}`);

const meta = readWindowsMeta(WINDOWS);
const sujeitos = [...new Set([...meta.values()].filter((o) => FOLDS.includes(o.fold)).map((o) => o.sujeito))].sort();
if (sujeitos.some((s) => NAVEGADOR.includes(s))) throw new Error("um sujeito do teste no navegador entrou na amostra");
log(`# regra: todos os sujeitos dos folds ${FOLDS.join(" e ")} em ${path.relative(ROOT, WINDOWS)} (${sujeitos.length}): ${sujeitos.join(" ")}`);

const pronto = (id) => fs.existsSync(path.join(SIG, `${id}.csv.gz`)) && fs.existsSync(path.join(SIG, `${id}.json`));
const amostra = [];
for (const s of sujeitos) {
  const v0 = `${s}_0`, v10 = `${s}_10`;
  for (const [v, k] of [[v0, 0], [v10, 10]]) {
    const o = meta.get(v);
    if (!o || o.rotulo_kss !== k) throw new Error(`${v}: vídeo ausente ou com rótulo inesperado em windows_10s.csv`);
  }
  if (!pronto(v0) || !pronto(v10)) {
    if (PARCIAL) { log(`# (ensaio) ${s}: sinais ainda não extraídos, fica de fora`); continue; }
    throw new Error(`${s}: faltam os sinais de ${[v0, v10].filter((v) => !pronto(v)).join(" e ")} (rode extrair_sinais_app.py --uta-folds ${FOLDS.join(" ")})`);
  }
  amostra.push(s);
}
if (!amostra.length) throw new Error("nenhum sujeito com os dois vídeos extraídos");

const linhas = [];
const porSujeito = [];
for (const s of amostra) {
  const ids = { 0: `${s}_0`, 10: `${s}_10` };
  const rows = {}, metas = {}, runs = {}, mets = {};
  for (const k of [0, 10]) {
    metas[k] = JSON.parse(fs.readFileSync(path.join(SIG, `${ids[k]}.json`), "utf8"));
    rows[k] = readSignals(path.join(SIG, `${ids[k]}.csv.gz`));
    if (rows[k].length !== metas[k].quadros) throw new Error(`${ids[k]}: ${rows[k].length} linhas, ${metas[k].quadros} quadros no .json`);
    runs[k] = runMonitor(rows[k]);
    mets[k] = { rosto_frac: metas[k].quadros_com_rosto / metas[k].quadros, ...metrics(rows[k], runs[k]) };
  }
  // variante: base da calibração do vídeo acordado aplicada ao sonolento (mesmos quadros avaliados)
  const runX = runMonitor(rows[10], { calRows: rows[0], fromT: runs[10].calT });
  const mX = { rosto_frac: mets[10].rosto_frac, ...metrics(rows[10], runX) };
  porSujeito.push({ s, fold: meta.get(ids[0]).fold, a: mets[0], d: mets[10], x: mX });
  for (const k of [0, 10]) {
    const m = mets[k];
    const me = metas[k];
    const o = meta.get(ids[k]);
    const row = {
      sujeito: s,
      fold: o.fold,
      video: ids[k],
      rotulo: k,
      estado: ROTULO[k],
      arquivo: me.video,
      fps: me.fps,
      largura_proc: me.largura,
      altura_proc: me.altura,
      rotacao: me.rotacao_meta,
      quadros: me.quadros,
      ...m,
    };
    const cx = {};
    for (const key of ["verde_pct", "amarelo_pct", "vermelho_pct", "fora_do_verde_pct", "alarmes", "avisos", "nivel_medio", "nivel_max", "primeiro_amarelo_s", "primeiro_vermelho_s", "fechamentos_05_1s", "microssonos", "bocejos", "quedas", "cabeceios", "cabeca_baixa", "olho_fechado_pct", "calib_limiar_ear"])
      cx[`cal_acordado_${key}`] = k === 10 ? mX[key] : null;
    cx.cal_acordado_dif_pitch_base = k === 10 ? m.calib_pitch - mX.calib_pitch : null;
    cx.cal_acordado_dif_yaw_base = k === 10 ? m.calib_yaw - mX.calib_yaw : null;
    linhas.push({ ...row, ...cx });
  }
  const g1 = (v) => (Number.isFinite(v) ? v.toFixed(1) : "--");
  log(`${s}: rosto ${g1(100 * mets[0].rosto_frac)}/${g1(100 * mets[10].rosto_frac)} %; limiar EAR ${mets[0].calib_limiar_ear.toFixed(3)}/${mets[10].calib_limiar_ear.toFixed(3)}; ` +
    `olhos fechados na calibração ${g1(mets[0].calib_olhos_fechados_pct)}/${g1(mets[10].calib_olhos_fechados_pct)} %; pitch base ${g1(mets[0].calib_pitch)}/${g1(mets[10].calib_pitch)}°, yaw base ${g1(mets[0].calib_yaw)}/${g1(mets[10].calib_yaw)}°`);
  log(`  acordado verde/amarelo/vermelho ${g1(mets[0].verde_pct)}/${g1(mets[0].amarelo_pct)}/${g1(mets[0].vermelho_pct)} % alarmes ${mets[0].alarmes} nível ${g1(mets[0].nivel_medio)} (máx ${g1(mets[0].nivel_max)}) | ` +
    `sonolento ${g1(mets[10].verde_pct)}/${g1(mets[10].amarelo_pct)}/${g1(mets[10].vermelho_pct)} % alarmes ${mets[10].alarmes} nível ${g1(mets[10].nivel_medio)} (máx ${g1(mets[10].nivel_max)}) | ` +
    `sonolento c/ calibração acordada ${g1(mX.verde_pct)}/${g1(mX.amarelo_pct)}/${g1(mX.vermelho_pct)} % alarmes ${mX.alarmes} nível ${g1(mX.nivel_medio)} (máx ${g1(mX.nivel_max)}) |`);
  log(`  ` +
    `eventos sonolento: fech ${mets[10].fechamentos_05_1s}, micro ${mets[10].microssonos}, boc ${mets[10].bocejos}, queda ${mets[10].quedas}, cabeceio ${mets[10].cabeceios}; ` +
    `acordado: fech ${mets[0].fechamentos_05_1s}, micro ${mets[0].microssonos}, boc ${mets[0].bocejos}, queda ${mets[0].quedas}, cabeceio ${mets[0].cabeceios}`);
}

// ---------------------------------------------------------------------------
// Conjunto
// ---------------------------------------------------------------------------
const N = porSujeito.length;
const A = porSujeito.map((p) => p.a), D = porSujeito.map((p) => p.d), X = porSujeito.map((p) => p.x);
const sum = (arr, k) => arr.reduce((acc, m) => acc + m[k], 0);
const conjunto = (arr) => {
  const tt = sum(arr, "duracao_avaliada_s");
  const tempo = (k) => arr.reduce((acc, m) => acc + (m[k] / 100) * m.duracao_avaliada_s, 0);
  return {
    videos: arr.length,
    horas: tt / 3600,
    com_alarme: arr.filter((m) => m.alarmes > 0).length,
    chegam_amarelo: arr.filter((m) => m.amarelo_pct > 0 || m.vermelho_pct > 0).length,
    chegam_vermelho: arr.filter((m) => m.vermelho_pct > 0).length,
    vermelho_pct: (100 * tempo("vermelho_pct")) / tt,
    amarelo_pct: (100 * tempo("amarelo_pct")) / tt,
    fora_pct: (100 * tempo("fora_do_verde_pct")) / tt,
    alarmes: sum(arr, "alarmes"),
    alarmes_h: sum(arr, "alarmes") / (tt / 3600),
    avisos_h: sum(arr, "avisos") / (tt / 3600),
    fora_mediana: median(arr.map((m) => m.fora_do_verde_pct)),
    nivel_medio_mediana: median(arr.map((m) => m.nivel_medio)),
    eventos: Object.fromEntries(["fechamentos_05_1s", "microssonos", "bocejos", "quedas", "cabeceios", "cabeca_baixa"].map((k) => [k, sum(arr, k)])),
    videos_com_evento_sono: arr.filter((m) => m.fechamentos_05_1s + m.microssonos + m.bocejos + m.quedas + m.cabeceios > 0).length,
    rosto_min: Math.min(...arr.map((m) => m.rosto_frac ?? 1)),
    piscadas_mediana: median(arr.map((m) => m.piscadas_min)),
    olho_fechado_mediana: median(arr.map((m) => m.olho_fechado_pct)),
    calib_aviso: arr.filter((m) => m.calib_aviso).length,
    t_amarelo_mediana_min: median(arr.filter((m) => m.primeiro_amarelo_s !== null).map((m) => m.primeiro_amarelo_s / 60)),
    t_vermelho_mediana_min: median(arr.filter((m) => m.primeiro_vermelho_s !== null).map((m) => m.primeiro_vermelho_s / 60)),
  };
};
const cA = conjunto(A), cD = conjunto(D), cX = conjunto(X);
const pareado = (dd) => {
  let mais = 0, igual = 0, menos = 0;
  porSujeito.forEach((p, i) => {
    const a = p.a.fora_do_verde_pct, b = dd[i].fora_do_verde_pct;
    if (b > a) mais++;
    else if (b === a) igual++;
    else menos++;
  });
  return { mais, igual, menos, p_sinal: signTest(mais, mais + menos) };
};
const pD = pareado(D), pX = pareado(X);
const nivelPareado = (dd) => porSujeito.filter((p, i) => dd[i].nivel_medio > p.a.nivel_medio).length;

// AUC e bootstrap de sujeitos (as mesmas reamostras para todas as medidas e variantes)
const SCORES = { fora_do_verde_pct: "% do tempo fora do verde", nivel_medio: "nível médio" };
const rng = mulberry32(SEED);
const boot = {};
for (const v of ["propria", "acordado"]) for (const k of Object.keys(SCORES)) boot[`${v}:${k}`] = [];
const draws = [];
for (let b = 0; b < N_BOOT; b++) {
  const idx = Array.from({ length: N }, () => Math.floor(rng() * N));
  draws.push(idx);
  for (const k of Object.keys(SCORES)) {
    const neg = idx.map((i) => A[i][k]);
    boot[`propria:${k}`].push(auc(idx.map((i) => D[i][k]), neg));
    boot[`acordado:${k}`].push(auc(idx.map((i) => X[i][k]), neg));
  }
}
const aucs = {};
for (const [v, dd] of [["propria", D], ["acordado", X]]) {
  for (const k of Object.keys(SCORES)) {
    const bs = [...boot[`${v}:${k}`]].sort((a, b) => a - b);
    aucs[`${v}:${k}`] = { auc: auc(dd.map((m) => m[k]), A.map((m) => m[k])), lo: quantile(bs, 0.025), hi: quantile(bs, 0.975) };
  }
}

const f1 = (v) => (Number.isFinite(v) ? v.toFixed(1).replace(".", ",") : "--");
const f2 = (v) => (Number.isFinite(v) ? v.toFixed(2).replace(".", ",") : "--");
const f3 = (v) => (Number.isFinite(v) ? v.toFixed(3).replace(".", ",") : "--");
const pc = (k, n) => `${k}/${n} (${f1((100 * k) / n)} %)`;
const aucTxt = (o) => `${f3(o.auc)} [${f3(o.lo)}; ${f3(o.hi)}]`;
const pTxt = (p) => (Number.isFinite(p) ? (p < 0.001 ? "p < 0,001" : `p = ${f3(p)}`) : "p indefinido");

log("");
log(`# Conjunto (${N} sujeitos, ${2 * N} vídeos; ${f2(cA.horas + cD.horas)} h avaliadas)`);
for (const [nome, c] of [["acordados", cA], ["sonolentos", cD], ["sonolentos com a calibração do acordado", cX]])
  log(`${nome}: ${c.videos} vídeos, ${f2(c.horas)} h; com alarme ${c.com_alarme}; chegam ao amarelo ${c.chegam_amarelo}; ao vermelho ${c.chegam_vermelho}; ` +
    `tempo em vermelho ${f2(c.vermelho_pct)} %, amarelo ${f2(c.amarelo_pct)} %, fora do verde ${f2(c.fora_pct)} %; alarmes ${c.alarmes} (${f2(c.alarmes_h)}/h), avisos ${f2(c.avisos_h)}/h; ` +
    `fora do verde mediano ${f2(c.fora_mediana)} %, nível médio mediano ${f2(c.nivel_medio_mediana)}; eventos ${JSON.stringify(c.eventos)}; vídeos com evento de sono ${c.videos_com_evento_sono}; ` +
    `piscadas/min mediana ${f1(c.piscadas_mediana)}, olho fechado mediano ${f1(c.olho_fechado_mediana)} %; aviso de calibração ${c.calib_aviso}; ` +
    `1º amarelo mediano ${f1(c.t_amarelo_mediana_min)} min e 1º vermelho mediano ${f1(c.t_vermelho_mediana_min)} min depois de calibrar (entre os que chegam)`);
log(`pareado (fora do verde, sonolento contra acordado): mais ${pD.mais}, igual ${pD.igual}, menos ${pD.menos}, teste do sinal ${pTxt(pD.p_sinal)} (p = ${pD.p_sinal.toExponential(2)}); ` +
  `com a calibração do acordado: mais ${pX.mais}, igual ${pX.igual}, menos ${pX.menos}, ${pTxt(pX.p_sinal)} (p = ${pX.p_sinal.toExponential(2)})`);
log(`pareado (nível médio maior no sonolento): ${nivelPareado(D)}/${N}; com a calibração do acordado: ${nivelPareado(X)}/${N}`);
for (const [key, o] of Object.entries(aucs)) log(`AUC ${key}: ${aucTxt(o)} (bootstrap de ${N_BOOT} reamostras de sujeitos, semente ${SEED})`);
log(`bootstrap: primeira reamostra = ${draws[0].map((i) => porSujeito[i].s).join(" ")}`);

// ---------------------------------------------------------------------------
// Saídas
// ---------------------------------------------------------------------------
fs.mkdirSync(path.join(RES, "logs"), { recursive: true });
fs.writeFileSync(path.join(RES, "app_validacao_uta.csv"), E.toCSV(linhas) + "\n");
log(`  -> results/app_validacao_uta.csv (${linhas.length} linhas)`);

// Log da extração: cada execução começa com "extração de sinais do app"; a última é a dos vídeos restantes.
const ext = { linhas: [], execucoes: [] };
if (fs.existsSync(EXTRACAO_LOG)) {
  ext.linhas = fs.readFileSync(EXTRACAO_LOG, "utf8").trim().split("\n");
  for (const l of ext.linhas) {
    if (/extração de sinais do app/.test(l)) ext.execucoes.push({ novos: 0, existentes: 0, procs: Number((l.match(/(\d+) processos/) || [])[1]), min: null });
    const cur = ext.execucoes[ext.execucoes.length - 1];
    if (!cur) continue;
    if (/: \d+ quadros, rosto/.test(l)) cur.novos++;
    // nota acrescentada à mão quando o computador foi suspenso no meio do lote (registro do pmset)
    const z = l.match(/nota: .*suspenso de (\d\d):(\d\d):(\d\d) a (\d\d):(\d\d):(\d\d)/);
    if (z) cur.suspensoMin = ((+z[4] * 3600 + +z[5] * 60 + +z[6]) - (+z[1] * 3600 + +z[2] * 60 + +z[3])) / 60, cur.suspensoTxt = `${z[1]}h${z[2]} às ${z[4]}h${z[5]}`;
    if (/já existia/.test(l)) cur.existentes++;
    const m = l.match(/fim: \d+ quadros novos em ([\d.]+) min/);
    if (m) cur.min = Number(m[1]);
  }
}
const segs = amostra.flatMap((s) => [0, 10].map((k) => JSON.parse(fs.readFileSync(path.join(SIG, `${s}_${k}.json`), "utf8")).segundos_proc));
const PILOTO = "F4_43_10"; // primeiro vídeo extraído, sozinho, para medir o tempo antes de escolher a amostra
const piloto = JSON.parse(fs.readFileSync(path.join(SIG, `${PILOTO}.json`), "utf8"));
const lote = ext.execucoes.length > 1 ? ext.execucoes[ext.execucoes.length - 1] : null;
log(`# extração: piloto ${PILOTO} em ${piloto.segundos_proc} s (1 processo); lote ${lote ? `${lote.novos} vídeos novos em ${lote.min} min com ${lote.procs} processos${lote.suspensoMin ? ` (${lote.suspensoMin.toFixed(1)} min suspenso)` : ""}` : "(sem registro)"}; ` +
  `por vídeo de ${Math.min(...segs)} a ${Math.max(...segs)} s; soma ${(segs.reduce((a, b) => a + b, 0) / 60).toFixed(1)} min`);

/** Leitura em poucas linhas, com os números desta execução. */
function leitura() {
  const nome = (s) => s.replace(/^F(\d)_/, "Fold$1/");
  const lista = (v) => (v.length <= 1 ? v.join("") : `${v.slice(0, -1).join(", ")} e ${v[v.length - 1]}`);
  const eventosSono = (m) => m.fechamentos_05_1s + m.microssonos + m.bocejos + m.quedas + m.cabeceios;
  const verdes = porSujeito.filter((p) => p.d.fora_do_verde_pct === 0);
  const verdesSemEvento = verdes.filter((p) => eventosSono(p.d) === 0).length;
  const comAlarmeA = porSujeito.filter((p) => p.a.alarmes > 0);
  const alarmesMicroA = A.reduce((acc, m) => acc + m.alarmes_por_microssono, 0);
  const menos = porSujeito.filter((p) => p.d.fora_do_verde_pct < p.a.fora_do_verde_pct);
  const empatesVerdes = porSujeito.filter((p) => p.d.fora_do_verde_pct === p.a.fora_do_verde_pct).every((p) => p.a.fora_do_verde_pct === 0);
  const aF = aucs["propria:fora_do_verde_pct"], aN = aucs["propria:nivel_medio"];
  const xF = aucs["acordado:fora_do_verde_pct"], xN = aucs["acordado:nivel_medio"];
  const acaso = aF.lo > 0.5 && aN.lo > 0.5 ? "acima do acaso (0,5) com o IC inteiro, mas longe de 1" : "sem separação clara do acaso (o IC inclui 0,5)";
  return [
    `1. **Acordados: ${cA.com_alarme} de ${cA.videos} vídeos com alarme.** Foram ${cA.alarmes} alarmes em ${f1(cA.horas)} h (${f2(cA.alarmes_h)} por hora) e ${f2(cA.vermelho_pct)} % do tempo em vermelho; ${cA.chegam_amarelo} vídeos chegaram ao amarelo. ` +
      (comAlarmeA.length ? `Os alarmes foram de ${lista(comAlarmeA.map((p) => nome(p.s)))}, e ${alarmesMicroA === cA.alarmes ? "todos" : `${alarmesMicroA} deles`} começaram por microssono (olho fechado por 1 s ou mais, segundo o app); sem anotação de olho na UTA-RLDD, não dá para saber se foram olhos fechados de verdade ou olhar para baixo.` : ""),
    `2. **Sonolentos: ${cD.chegam_amarelo} de ${cD.videos} chegaram ao amarelo e ${cD.chegam_vermelho} ao vermelho**, o primeiro vermelho em mediana ${f1(cD.t_vermelho_mediana_min)} min depois de calibrar. ${verdes.length} ficaram verdes o vídeo inteiro com KSS 8 ou 9; em ${verdesSemEvento} deles o app não viu nenhum fechamento de 0,5 s ou mais, bocejo ou queda da cabeça.`,
    `3. **Por pessoa:** em ${pD.mais} de ${N} o vídeo sonolento ficou mais tempo fora do verde que o acordado, ${pD.igual} empataram${empatesVerdes && pD.igual ? " (verdes nos dois)" : ""} e ${pD.menos} teve o contrário` +
      (menos.length ? ` (${lista(menos.map((p) => `${nome(p.s)}, que calibrou o vídeo sonolento com os olhos fechados em ${f1(p.d.calib_olhos_fechados_pct)} % do tempo`))})` : "") +
      `. AUC de ${aucTxt(aF)} pelo % do tempo fora do verde e de ${aucTxt(aN)} pelo nível médio: ${acaso}.`,
    `4. **Calibrar já com sono atrapalha.** Em ${cD.calib_aviso} dos ${cD.videos} vídeos sonolentos, a calibração viu os olhos fechados em mais de 10 % do tempo (o app avisa e pede para recalibrar). Com a calibração do vídeo acordado, ${cX.chegam_amarelo} sonolentos chegaram ao amarelo e ${cX.chegam_vermelho} ao vermelho, o vídeo sonolento ficou mais tempo fora do verde em ${pX.mais} pessoas e menos em ${pX.menos === 0 ? "nenhuma" : pX.menos}, e a AUC passou a ${aucTxt(xF)} e ${aucTxt(xN)}.`,
    `5. **Para a banca:** com sono real, declarado pela própria pessoa, o app saiu do verde em ${cD.chegam_amarelo} de ${cD.videos} sonolentos e em ${cA.chegam_amarelo} de ${cA.videos} acordados, e alarmou ${cD.com_alarme} sonolentos contra ${cA.com_alarme} acordados. Ele reage a sinais visíveis (fechamentos longos e microssonos), não ao sono declarado: ${verdes.length} sonolentos ficaram verdes. O teste foi com pessoas sentadas, não dirigindo.`,
  ];
}

const md = [];
md.push("# Teste independente do app em sono real (UTA-RLDD)", "");
md.push(`Gerado em ${_p(_d.getDate())}/${_p(_d.getMonth() + 1)}/${_d.getFullYear()}, ${_p(_d.getHours())}h${_p(_d.getMinutes())}, por \`tools/app_validacao/validar_motor_uta.mjs\` (log: \`${logRel}\`). Uma linha por vídeo em \`results/app_validacao_uta.csv\`.`, "");
md.push("## Como foi feito", "");
const engineMd5 = md5(engineFile);
const versaoTxt = {
  c4cc34e59a6a51b8076276d2e41f9a58: "É o mesmo arquivo da validação no navegador (md5 c4cc34e5…).",
  // conferido com diff em 26/10/2026: muda só um comentário e o texto do PERCLOS em "Por quê?"; CFG e lógica iguais
  ef2c9d3e85dc51bcdb582c9b5a34d354: "O arquivo de hoje (md5 ef2c9d3e…) difere do usado na validação no navegador (md5 c4cc34e5…) só num comentário e no texto do PERCLOS em \"Por quê?\"; a configuração e a lógica são as mesmas.",
}[engineMd5] || `O arquivo de hoje (md5 ${engineMd5.slice(0, 8)}…) não é nenhum dos conferidos: compare com o da validação no navegador (md5 c4cc34e5…).`;
md.push(`- **O que foi testado:** o motor do app ${appVersion}, \`app/pwa/engine.js\`, importado sem cópia, com a configuração final (\`CFG\`), pelo mesmo caminho da validação offline do DMD (\`results/app_validacao_dmd.md\`). ${versaoTxt} O LSTM não entra: no app ele soma no máximo 10 pontos e só quando os sinais visíveis já deram amarelo, então não muda a entrada no amarelo; o tempo em vermelho do app completo pode ser um pouco maior.`);
md.push(`- **Sinais:** o mesmo Face Landmarker da PWA, com blendshapes e pose (\`tools/app_validacao/extrair_sinais_app.py\`), em todos os quadros de cada vídeo. A rotação do celular foi aplicada e o quadro foi reduzido para no máximo 1280 px no lado maior, como na cadeia offline e no DMD. O tempo de cada quadro é o carimbo do arquivo (há vídeos com fps variável).`);
md.push("- **Calibração:** a do app, nos 10 s iniciais com rosto de cada vídeo, como no início de cada viagem. Nos vídeos sonolentos, isso quer dizer calibrar já com sono.");
md.push("- **Variante:** o monitor calibra no começo do vídeo acordado da mesma pessoa e depois recebe o vídeo sonolento, como quem calibra acordado e fica com sono depois. Só a base da calibração (limiar do olho, boca e pose de repouso) passa de um vídeo para o outro; a duração de base das piscadas sai, nas duas formas, das 20 primeiras piscadas do próprio vídeo. Os quadros avaliados são os mesmos nas duas formas.");
md.push(`- **Métricas:** depois da calibração, até o fim do vídeo (cerca de 10 min). Verde, amarelo e vermelho são os estados do app (nível abaixo de 40, de 40 a 69, 70 ou mais, com histerese, ou vermelho forçado por microssono e cabeça caída). Alarme = entrada no vermelho. "Chega ao amarelo" = algum tempo em amarelo ou vermelho. AUC = chance de um vídeo sonolento ter valor maior que um acordado (empate vale meio). IC de 95 % por bootstrap de sujeitos (${N_BOOT.toLocaleString("pt-BR")} reamostras, semente ${SEED}; cada pessoa sorteada entra com os dois vídeos).`, "");

md.push("## Amostra e regra de escolha", "");
md.push(`- **Regra, fixada antes de rodar:** todos os sujeitos dos folds oficiais ${FOLDS.join(" e ")} da UTA-RLDD (\`data/features/final/uta/windows_10s.csv\`). São os dois folds sem nenhum dos 3 sujeitos do teste no navegador (Fold1/09, Fold3/31 e Fold5/51 estão nos folds 1, 3 e 5). Nenhum sujeito ou vídeo foi tirado depois.`);
md.push(`- **Amostra:** ${N} sujeitos (${amostra.join(", ").replace(/_/g, "/").replace(/F(\d)\//g, "Fold$1/")}), com o vídeo 0 (acordado, KSS 1 a 3) e o vídeo 10 (sonolento, KSS 8 e 9) de cada um: ${2 * N} vídeos e ${f1(cA.horas + cD.horas)} h avaliadas.`);
md.push(`- **Tempo:** antes de escolher, um vídeo (Fold4/43, vídeo 10: ${piloto.largura}x${piloto.altura}, ${f1(piloto.duracao_s / 60)} min) foi extraído sozinho em ${f1(piloto.segundos_proc)} s. Com esse tempo e os da cadeia offline nos mesmos vídeos, a estimativa para os ${2 * N} vídeos foi de cerca de 40 a 45 min com 6 processos. ` +
  (lote && lote.min !== null
    ? `A extração dos ${lote.novos} restantes levou ${f1(lote.min)} min de relógio com ${lote.procs} processos` +
      (lote.suspensoMin ? `, mas o computador ficou suspenso (tampa fechada) das ${lote.suspensoTxt}, por ${f1(lote.suspensoMin)} min; descontada a suspensão, foram cerca de ${f1(lote.min - lote.suspensoMin)} min de processamento.` : ".")
    : "A extração do lote não tem registro de fim.") +
  " Os vídeos 1080p e 1440p são os mais lentos. O motor do app roda em segundos.", "");

md.push("## Resultado principal", "");
md.push(`| Medida | Calibração no próprio vídeo | Calibração do vídeo acordado |`, "|---|---|---|");
md.push(`| Acordados com algum alarme vermelho | ${pc(cA.com_alarme, cA.videos)} | igual (mesma calibração) |`);
md.push(`| Tempo em vermelho nos acordados | ${f2(cA.vermelho_pct)} %, com ${cA.alarmes} alarmes em ${f1(cA.horas)} h (${f2(cA.alarmes_h)} por hora) | igual |`);
md.push(`| Acordados que chegam ao amarelo | ${pc(cA.chegam_amarelo, cA.videos)}, com ${f2(cA.amarelo_pct)} % do tempo em amarelo | igual |`);
md.push(`| Sonolentos que chegam ao amarelo | ${pc(cD.chegam_amarelo, cD.videos)} | ${pc(cX.chegam_amarelo, cX.videos)} |`);
md.push(`| Sonolentos que chegam ao vermelho | ${pc(cD.chegam_vermelho, cD.videos)} | ${pc(cX.chegam_vermelho, cX.videos)} |`);
md.push(`| Tempo fora do verde: acordados / sonolentos (soma dos vídeos) | ${f2(cA.fora_pct)} % / ${f2(cD.fora_pct)} % | ${f2(cA.fora_pct)} % / ${f2(cX.fora_pct)} % |`);
md.push(`| Tempo em vermelho nos sonolentos | ${f2(cD.vermelho_pct)} % | ${f2(cX.vermelho_pct)} % |`);
md.push(`| Minutos até o 1º amarelo / 1º vermelho nos sonolentos (mediana entre os que chegam) | ${f1(cD.t_amarelo_mediana_min)} / ${f1(cD.t_vermelho_mediana_min)} | ${f1(cX.t_amarelo_mediana_min)} / ${f1(cX.t_vermelho_mediana_min)} |`);
md.push(`| Pessoas com mais tempo fora do verde no vídeo sonolento (mais / igual / menos) | ${pD.mais} / ${pD.igual} / ${pD.menos} | ${pX.mais} / ${pX.igual} / ${pX.menos} |`);
md.push(`| AUC do % do tempo fora do verde [IC 95 %] | ${aucTxt(aucs["propria:fora_do_verde_pct"])} | ${aucTxt(aucs["acordado:fora_do_verde_pct"])} |`);
md.push(`| AUC do nível médio [IC 95 %] | ${aucTxt(aucs["propria:nivel_medio"])} | ${aucTxt(aucs["acordado:nivel_medio"])} |`, "");
md.push(`Teste do sinal (bilateral, sem os empates) na comparação pareada: ${pTxt(pD.p_sinal)} com a calibração no próprio vídeo e ${pTxt(pX.p_sinal)} com a do acordado. Nível médio maior no vídeo sonolento: ${nivelPareado(D)} de ${N} pessoas (${nivelPareado(X)} com a calibração do acordado).`, "");
md.push("| Eventos somados | Fechamentos de 0,5 a 1 s | Microssonos | Bocejos | Quedas da cabeça | Cabeceios | Vídeos com algum desses |", "|---|---:|---:|---:|---:|---:|---:|");
for (const [nome, c] of [[`${N} acordados`, cA], [`${N} sonolentos`, cD], [`${N} sonolentos, calibração do acordado`, cX]])
  md.push(`| ${nome} | ${c.eventos.fechamentos_05_1s} | ${c.eventos.microssonos} | ${c.eventos.bocejos} | ${c.eventos.quedas} | ${c.eventos.cabeceios} | ${c.videos_com_evento_sono} |`);
md.push("");

md.push("## Leitura honesta", "");
for (const l of leitura()) md.push(l);
md.push("");

const baixoFps = linhas.filter((l) => l.fps < 20);
const rostoMin = linhas.reduce((m, l) => (l.rosto_frac < m.rosto_frac ? l : m), linhas[0]);
md.push("## Limitações", "");
md.push("- **Sentado, não dirigindo.** Os vídeos da UTA-RLDD foram gravados pelas próprias pessoas, com o celular ou a webcam, sentadas diante da câmera e sem dirigir. Na direção há movimento, luz que muda e olhares para os espelhos; no DMD, só o movimento do carro já derrubou a qualidade do sinal dos olhos (Exp. 6). O falso alarme em movimento continua medido pela direção real do DMD (`results/app_validacao_dmd.md`).");
md.push("- **O rótulo é o que a pessoa declarou (KSS) para o vídeo inteiro**, não sinais anotados quadro a quadro. O app só vê sinais visíveis (olhos, boca, cabeça): quem disse estar com sono mas não fechou os olhos nem bocejou fica verde, e aqui não dá para medir revocação por evento.");
md.push("- **Calibração.** No uso normal, cada vídeo sonolento é calibrado já com sono (o app calibra no início de cada viagem). A variante usa a calibração do vídeo acordado, mas cruza duas gravações feitas em momentos diferentes: a posição da câmera e a luz podem mudar, e a pose de base da cabeça pode ficar deslocada.");
md.push(`- **Offline, no Mac.** É o mesmo código do motor do app, mas não o app no celular: o LSTM ficou de fora, o quadro foi reduzido para no máximo 1280 px (no celular o app recebe 640x480) e ${baixoFps.length} dos ${linhas.length} vídeos, de ${new Set(baixoFps.map((l) => l.sujeito)).size} pessoas (${[...new Set(baixoFps.map((l) => l.sujeito.replace(/^F(\d)_/, "Fold$1/")))].join(", ")}), têm menos de 20 quadros por segundo, o que mede as piscadas com menos quadros. O rosto foi achado em pelo menos ${f1(100 * rostoMin.rosto_frac)} % dos quadros de cada vídeo.`);
md.push(`- **Amostra.** ${N} pessoas de 2 dos 5 folds, escolhidas pela regra acima e não pelos resultados. Com esse tamanho, os intervalos de confiança são largos; as outras pessoas da base não foram testadas aqui.`);
md.push("");

md.push("## Por pessoa", "");
md.push("A = vídeo acordado (0); S = vídeo sonolento (10), calibrado no próprio começo; S\\* = vídeo sonolento com a calibração do vídeo acordado. Rosto = % dos quadros com rosto. Fora do verde e vermelho em % do tempo avaliado. Eventos = fechamentos de 0,5 a 1 s / microssonos / bocejos / quedas e cabeceios da cabeça. Piscadas/min = fechamentos de 50 a 500 ms que o app marcou como olho fechado: o app não foi feito para contar piscadas, e valores perto de 0 querem dizer que, nas piscadas rápidas daquela pessoa, a pálpebra não chegou ao limiar de \"fechado\". Olho fechado = % dos quadros com rosto marcados como olho fechado.", "");
md.push("| Pessoa | Rosto A / S | Fora do verde A / S / S\\* | Vermelho A / S / S\\* | Alarmes A / S / S\\* | Nível médio A / S / S\\* | Nível máx. A / S / S\\* | Eventos A | Eventos S | Eventos S\\* | Piscadas/min A / S | Olho fechado A / S |",
  "|---|---|---|---|---|---|---|---|---|---|---|---|");
const ev = (m) => `${m.fechamentos_05_1s} / ${m.microssonos} / ${m.bocejos} / ${m.quedas + m.cabeceios}`;
const tri = (k, fmt = f1) => (p) => `${fmt(p.a[k])} / ${fmt(p.d[k])} / ${fmt(p.x[k])}`;
const int = (v) => String(v);
for (const p of porSujeito) {
  const nome = p.s.replace(/^F(\d)_/, "Fold$1/");
  md.push(`| ${nome} | ${f1(100 * p.a.rosto_frac)} / ${f1(100 * p.d.rosto_frac)} | ${tri("fora_do_verde_pct")(p)} | ${tri("vermelho_pct")(p)} | ${tri("alarmes", int)(p)} | ${tri("nivel_medio")(p)} | ${tri("nivel_max", (v) => f1(v).replace(",0", ""))(p)} | ${ev(p.a)} | ${ev(p.d)} | ${ev(p.x)} | ${f1(p.a.piscadas_min)} / ${f1(p.d.piscadas_min)} | ${f1(p.a.olho_fechado_pct)} / ${f1(p.d.olho_fechado_pct)} |`);
}
md.push("");

fs.writeFileSync(path.join(RES, "app_validacao_uta.md"), md.join("\n") + "\n");
log(`  -> results/app_validacao_uta.md`);
log("");
log(`# Log da extração dos sinais (${path.relative(ROOT, EXTRACAO_LOG)}):`);
for (const l of ext.linhas) log(l);
fs.writeFileSync(logPath, logLines.join("\n") + "\n");
console.log(`log: ${path.relative(ROOT, logPath)}`);
