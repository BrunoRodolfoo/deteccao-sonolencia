/**
 * Validação OFFLINE do motor do app (app/pwa/engine.js, detector v2) contra a anotação do DMD.
 *
 *   node tools/app_validacao/validar_motor_dmd.mjs
 *
 * Entrada: sinais por quadro extraídos com o MESMO Face Landmarker da PWA
 * (tools/app_validacao/extrair_sinais_app.py -> data/signals/app_tasks/dmd/<id>.csv.gz) e o
 * OpenLABEL oficial de cada sessão (data/dmd/.../*_rgb_ann_*.json).
 * O código avaliado é o do app, importado sem cópia: ../../app/pwa/engine.js.
 *
 * O que mede:
 *  s5 (sonolência ENCENADA, carro parado; anotação de olho e bocejo):
 *   - olhos por quadro: fechado (anotação "close") x aberto ("open"), transições excluídas, para
 *     EAR sozinho (limiar 0,75 x mediana dos 10 s iniciais), blendshape sozinho e a fusão do app;
 *   - fechamentos >= 0,5 s e >= 1 s por evento (casamento um-para-um pela sobreposição);
 *   - bocejos por evento (com e sem mão) para o detector do app, uma grade de limiares e a regra da v1
 *     (MAR > 0,6 por 2 s);
 *  s1 (direção REAL, carro em movimento, motorista acordado; anotação de ações de distração):
 *   - eventos e alarmes por hora com o monitor completo (calibração nos 10 s iniciais), no total e
 *     por ação anotada (safe_drive, radio, drinking, reach_side, talking_to_passenger...).
 *  s5 com o monitor completo: tempo em cada estado e alarmes por sessão (comparação com s1).
 *  s1 + s5, AJUSTE FINAL (27/09/2026): alternativas avaliadas a partir da configuração final do app (histerese
 *   nos olhos, PERCLOS com as piscadas, microssono em 1,2 s, bocejo com 1,0 s, microssono com 40 pt por 5 min),
 *   no monitor completo -> results/app_validacao_dmd_ajuste_final.csv e seção própria no .md;
 *  s1 + s5: SINAL do pitch da matriz de pose (convenção "pitch > 0 = cabeça para baixo"): correlação,
 *   quadro a quadro, entre o pitch e a posição vertical relativa da ponta do nariz,
 *   (nariz_y - olhos_y) / (queixo_y - olhos_y) na imagem, que cresce quando a cabeça baixa.
 * O LSTM não entra (mlProbs = null): no app ele vale no máximo +10 pontos e não dispara alarme.
 *
 * Saídas: results/app_validacao_dmd_olhos.csv, _fechamentos.csv, _bocejo.csv, _bocejo_grade.csv,
 * _sessoes.csv, _acoes_s1.csv, _variantes_olhos.csv, _pose_sinal.csv, _ajuste_final.csv, results/app_validacao_dmd.md e
 * results/logs/app_validacao_dmd_<data>.log.
 */
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";
import * as E from "../../app/pwa/engine.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const SIG = path.join(ROOT, "data/signals/app_tasks/dmd");
const RES = path.join(ROOT, "results");
const _d = new Date();
const _p = (v) => String(v).padStart(2, "0");
const stamp = `${_d.getFullYear()}${_p(_d.getMonth() + 1)}${_p(_d.getDate())}-${_p(_d.getHours())}${_p(_d.getMinutes())}${_p(_d.getSeconds())}`;
const logPath = path.join(RES, "logs", `app_validacao_dmd_${stamp}.log`);
const logLines = [];
const log = (...a) => {
  const s = a.join(" ");
  logLines.push(s);
  console.log(s);
};

// ---------------------------------------------------------------------------
// Leitura
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
      noseRel: face ? (num("nose_y") - num("eye_y")) / (num("chin_y") - num("eye_y")) : NaN,
    };
  }
  return rows;
}

function readAnnotation(file, n) {
  const d = JSON.parse(fs.readFileSync(file, "utf8")).openlabel;
  const acts = {};
  for (const a of Object.values(d.actions || {})) {
    for (const iv of a.frame_intervals || []) (acts[a.type] ||= []).push([Number(iv.frame_start), Number(iv.frame_end)]);
  }
  for (const k of Object.keys(acts)) acts[k].sort((x, y) => x[0] - y[0]);
  const eye = new Int8Array(n).fill(-1);
  const code = { "eyes_state/open": 0, "eyes_state/close": 1, "eyes_state/opening": 2, "eyes_state/closing": 3 };
  for (const [k, v] of Object.entries(code)) for (const [s, e] of acts[k] || []) for (let i = s; i <= Math.min(e, n - 1); i++) eye[i] = v;
  return { acts, eye };
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

/** Calibração igual à do app: primeiros 10 s de rosto visível (medianas). */
function calibrate(rows) {
  const ear = [], jaw = [], mar = [], pitch = [], yaw = [], roll = [];
  let faceMs = 0;
  let last = null;
  for (const r of rows) {
    const dt = last === null ? 0 : Math.min(100, r.t - last);
    last = r.t;
    if (!r.face) continue;
    faceMs += dt;
    ear.push((r.earL + r.earR) / 2);
    if (Number.isFinite(r.jaw)) jaw.push(r.jaw);
    if (Number.isFinite(r.mar)) mar.push(r.mar);
    if (r.pose) { pitch.push(r.pose.pitch); yaw.push(r.pose.yaw); roll.push(r.pose.roll); }
    if (faceMs >= E.CFG.calib.neutralMs) break;
  }
  const em = E.median(ear);
  return { earThr: E.CFG.calib.earFactor * em, earMedian: em, jaw0: E.median(jaw), mar0: E.median(mar), pitch0: E.median(pitch), yaw0: E.median(yaw), roll0: E.median(roll) };
}

// ---------------------------------------------------------------------------
// Métricas
// ---------------------------------------------------------------------------
function confusion(pred, truth) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (let i = 0; i < pred.length; i++) {
    if (truth[i] === null) continue;
    if (pred[i] && truth[i]) tp++;
    else if (!pred[i] && !truth[i]) tn++;
    else if (pred[i]) fp++;
    else fn++;
  }
  const n = tp + tn + fp + fn;
  const sens = tp / (tp + fn || 1);
  const spec = tn / (tn + fp || 1);
  const po = (tp + tn) / (n || 1);
  const pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / ((n || 1) ** 2);
  return { tp, tn, fp, fn, sens, spec, bal: (sens + spec) / 2, kappa: (po - pe) / (1 - pe || 1), prec: tp / (tp + fp || 1) };
}

/** Casamento um-para-um guloso pela maior sobreposição (intervalos em ms, fechados). */
function matchEvents(det, ann) {
  const pairs = [];
  det.forEach((d, i) => ann.forEach((a, j) => {
    const ov = Math.min(d[1], a[1]) - Math.max(d[0], a[0]);
    if (ov >= 0) pairs.push([ov, i, j]);
  }));
  pairs.sort((x, y) => y[0] - x[0]);
  const ud = new Set(), ua = new Set();
  for (const [, i, j] of pairs) if (!ud.has(i) && !ua.has(j)) { ud.add(i); ua.add(j); }
  return { tpDet: ud.size, tpAnn: ua.size, matchedAnn: ua, matchedDet: ud };
}

const fps = (rows) => 1000 / ((rows[rows.length - 1].t - rows[0].t) / (rows.length - 1));
const f3 = (v) => (Number.isFinite(v) ? v.toFixed(3) : "");
function writeCSV(name, rows) {
  const p = path.join(RES, name);
  fs.writeFileSync(p, E.toCSV(rows) + "\n");
  log(`  -> ${path.relative(ROOT, p)} (${rows.length} linhas)`);
}

// ---------------------------------------------------------------------------
// Execução
// ---------------------------------------------------------------------------
const files = fs.readdirSync(SIG).filter((f) => f.endsWith(".csv.gz")).sort();
const sess = (f) => (f.match(/_(s\d)_/) || [])[1];
log(`# Validação offline do motor do app (engine.js v2) no DMD — ${new Date().toISOString()}`);
log(`# node ${process.version}; sinais: ${path.relative(ROOT, SIG)} (${files.length} vídeos); CFG = ${JSON.stringify(E.CFG)}`);

const olhosRows = [], fechRows = [], bocejoRows = [], gradeAcc = new Map(), sessRows = [], acaoAcc = new Map(), poseRows = [];
const durDiffs = []; // duração app - anotação 'close' (fechamentos >= 0,3 s casados um-para-um), por par
const diagOlhos = { fechado: { ld: [], bm: [] }, aberto: { ld: [], bm: [] } };
/** Correlação de Pearson. */
function pearson(x, y) {
  const n = x.length;
  if (n < 3) return NaN;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += x[i] / n; my += y[i] / n; }
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { const a = x[i] - mx, b = y[i] - my; sxy += a * b; sxx += a * a; syy += b * b; }
  return sxy / Math.sqrt(sxx * syy);
}
const Y = E.CFG.yawn;
const YAWN_APP_LABEL = `app (jawOpen>=${Y.jawOpen} e >= repouso+${Y.jawRel}, ou MAR>=repouso+${Y.marRel}; por ${Y.minMs / 1000} s)`.replace(/\./g, ",");
const YAWN_GRID = [];
for (const jaw of [0.3, 0.35, 0.4, 0.45, 0.5]) for (const minMs of [1000, 1500, 2000]) YAWN_GRID.push({ jawOpen: jaw, minMs });
// Variantes da fusão dos olhos avaliadas no monitor COMPLETO (s5: anotação de olho; s1: direção real).
const G = E.CFG.eyes;
const NO_GATES = { lookDownGate: Infinity, headDownGateDeg: Infinity, squintGate: Infinity, asideGateDeg: Infinity };
const EYE_VARIANTS = {
  ear_v1: { ...G, ...NO_GATES, blendClosed: Infinity, blendSupport: 0 }, // só EAR (como a v1)
  blend050: { ...G, earPath: false, blendClosed: 0.5 },
  blend045: { ...G, earPath: false, blendClosed: 0.45 },
  blend055: { ...G, earPath: false, blendClosed: 0.55 },
  blend050_rel030: { ...G, earPath: false, blendClosed: 0.5, blendRel: 0.3 },
  fusao_media035_2travas: { ...G, earPath: true, blendClosed: 0.5, blendSupport: 0.35, supportUseMin: false, lookDownGate: 0.6, headDownGateDeg: 10, squintGate: Infinity, asideGateDeg: Infinity },
  fusao_min040_4travas: { ...G, earPath: true, blendClosed: 0.5, blendSupport: 0.4, supportUseMin: true, lookDownGate: 0.6, headDownGateDeg: 10, squintGate: 0.45, asideGateDeg: 20 },
  fusao_min045_4travas: { ...G, earPath: true, blendClosed: 0.5, blendSupport: 0.45, supportUseMin: true, lookDownGate: 0.6, headDownGateDeg: 10, squintGate: 0.45, asideGateDeg: 20 },
  blend050_rel025: { ...G, earPath: false, blendClosed: 0.5, blendRel: 0.25 },
  fusao_min040_4travas_rel025: { ...G, earPath: true, blendClosed: 0.5, blendRel: 0.25, blendSupport: 0.4, supportUseMin: true, lookDownGate: 0.6, headDownGateDeg: 10, squintGate: 0.45, asideGateDeg: 20 },
  fusao_min040_4travas_rel030: { ...G, earPath: true, blendClosed: 0.5, blendRel: 0.3, blendSupport: 0.4, supportUseMin: true, lookDownGate: 0.6, headDownGateDeg: 10, squintGate: 0.45, asideGateDeg: 20 },
  app: G,
};

// Ajuste final: cada alternativa = a configuração FINAL do app com UMA mudança (sobreposição por seção de CFG).
const AJUSTES = {
  final: { desc: "configuração final do app (v2.1)", over: {} },
  perclos_com_piscadas: { desc: "PERCLOS com as piscadas (como na v2.0: todo quadro fechado conta)", over: { perclos: { minEpisodeMs: 0 } } },
  histerese_010: { desc: "histerese nos olhos: reabre só com o menor eyeBlink < limiar - 0,1", over: { eyes: { reopenHyst: 0.1 } } },
  histerese_005: { desc: "histerese nos olhos: reabre só com o menor eyeBlink < limiar - 0,05", over: { eyes: { reopenHyst: 0.05 } } },
  microssono_12s: { desc: "microssono (vermelho imediato) a partir de 1,2 s em vez de 1,0 s", over: { eyes: { microsleepMs: 1200 } } },
  bocejo_10s: { desc: "bocejo com boca aberta por 1,0 s em vez de 1,5 s", over: { yawn: { minMs: 1000 } } },
  microssono_40pt_5min: { desc: "microssono com 40 pt por 5 min e sem o piso de 1 min em ATENÇÃO (como na v2.0)", over: { score: { microsleep: { points: 40, decayMs: 300000 }, postForced: { floor: 0 } } } },
};
function mergeCfg(over) {
  const out = {};
  for (const [k, v] of Object.entries(E.CFG)) out[k] = v && typeof v === "object" ? { ...v } : v;
  for (const [sec, o] of Object.entries(over || {})) for (const [k, v] of Object.entries(o)) out[sec][k] = v && typeof v === "object" && !Array.isArray(v) ? { ...E.CFG[sec][k], ...v } : v;
  return out;
}
const ajAcc = new Map();
function accAj(name, key, v) {
  const o = ajAcc.get(name) || { alternativa: name, descricao: AJUSTES[name].desc, perclos_s1: [] };
  o[key] = (o[key] || 0) + v;
  ajAcc.set(name, o);
}

/** Roda o monitor completo com uma configuração dos olhos (ou uma configuração inteira) e devolve por quadro o essencial. */
function runMonitor(rows, eyesCfg, fullCfg = null) {
  const cfgUse = fullCfg || { ...E.CFG, eyes: eyesCfg };
  eyesCfg = cfgUse.eyes;
  const mon = new E.DrowsinessMonitor(cfgUse);
  mon.startCalibration(rows[0].t);
  const n = rows.length;
  const closed = new Uint8Array(n);
  const byBlend = new Uint8Array(n);
  const state = new Int8Array(n);
  const evs = [];
  const perclos = new Float32Array(n).fill(NaN); // PERCLOS do app depois de 60 s de observação
  let calT = null;
  for (let i = 0; i < n; i++) {
    const r = rows[i];
    const res = mon.update(r.t, frameInput(r));
    if (calT === null && res.events.some((e) => e.type === "calibrado")) calT = r.t;
    if (res.eyes) {
      closed[i] = res.eyes.closed ? 1 : 0;
      byBlend[i] = res.eyes.byBlend ? 1 : 0;
      if (calT !== null && r.t - calT >= 60000) perclos[i] = res.eyes.perclos;
    }
    state[i] = res.score.state;
    for (const e of res.events) evs.push({ ...e, frame: i });
  }
  const episodes = [];
  const ct = new E.ClosureTracker(eyesCfg);
  for (let i = 0; i < n; i++) {
    if (calT === null || rows[i].t <= calT) continue;
    const o = ct.update(rows[i].t, !!closed[i], rows[i].face);
    if (o.ended) episodes.push([o.ended.start, o.ended.end]);
  }
  return { mon, closed, byBlend, state, evs, calT, episodes, perclos };
}
/** Replay da lógica do app ANTERIOR (v1): limiar de EAR em 3 s, olho fechado = EAR < limiar,
 * PERCLOS por quadros em 60 s e AlertLogic (vermelho com olho fechado >= 1 s OU PERCLOS > 15 %).
 * O LSTM da v1 não é simulado (ele só podia ACRESCENTAR alarmes: o tempo em vermelho é um limite inferior). */
function runV1(rows) {
  const cal = new E.ThresholdCalibrator({ durationMs: 3000, factor: 0.75 });
  cal.start(rows[0].t);
  const perclos = new E.Perclos(60000);
  const al = new E.AlertLogic({ closedMs: 1000, perclosThresh: 0.15, drowsyMs: 10000 });
  let thr = null, calT = null, redMs = 0, alarms = 0, prevRed = false, lastT = null;
  for (const r of rows) {
    const dt = lastT === null ? 0 : r.t - lastT;
    lastT = r.t;
    if (thr === null) {
      if (r.face) { cal.update(r.t, (r.earL + r.earR) / 2); if (cal.threshold !== null) { thr = cal.threshold; calT = r.t; } }
      continue;
    }
    let a;
    if (r.face) {
      const closed = (r.earL + r.earR) / 2 < thr;
      perclos.update(r.t, closed);
      a = al.update(r.t, { closed, perclos: perclos.value, cls: null, mar: r.mar });
    } else a = al.update(r.t, { closed: false, perclos: perclos.value, cls: null, mar: 0 });
    const red = a.alerts.length > 0;
    if (red) redMs += dt;
    if (red && !prevRed) alarms++;
    prevRed = red;
  }
  return { redMs, alarms, activeMs: calT === null ? 0 : rows[rows.length - 1].t - calT };
}
const v1Acc = { s1: { red: 0, alarms: 0, ms: 0 }, s5: { red: 0, alarms: 0, ms: 0 } };
const varAcc = new Map();
function accVar(name, key, v) {
  const o = varAcc.get(name) || { variante: name };
  o[key] = (o[key] || 0) + v;
  varAcc.set(name, o);
}

for (const file of files) {
  const id = file.replace(".csv.gz", "");
  const s = sess(id);
  const meta = JSON.parse(fs.readFileSync(path.join(SIG, id + ".json"), "utf8"));
  const rows = readSignals(path.join(SIG, file));
  const n = rows.length;
  const hz = meta.fps || fps(rows);
  const f2ms = (fr) => (fr * 1000) / hz;
  const { acts, eye } = readAnnotation(path.join(ROOT, meta.anotacao), n);
  const cal = calibrate(rows);
  const sub = id.split("_").slice(0, 2).join("_");

  // ---- sinal do pitch (s1 e s5) ----
  {
    const P = [], R = [];
    for (const r of rows) if (r.face && r.pose && Number.isFinite(r.noseRel)) { P.push(r.pose.pitch); R.push(r.noseRel); }
    const sp = [...P].sort((a, b) => a - b);
    poseRows.push({ id, sessao: s, sujeito: sub, quadros: P.length, pitch_p05: sp[Math.floor(0.05 * (sp.length - 1))], pitch_mediana: sp[sp.length >> 1], pitch_p95: sp[Math.floor(0.95 * (sp.length - 1))], r_pitch_nariz: pearson(P, R) });
  }

  // ---- monitor completo (o app) ----
  const run = runMonitor(rows, E.CFG.eyes);
  const { mon, state, evs, calT } = run;
  const closedApp = run.closed;
  const closures = run.episodes;

  // ---- app anterior (v1) ----
  if (v1Acc[s]) { const v = runV1(rows); v1Acc[s].red += v.redMs; v1Acc[s].alarms += v.alarms; v1Acc[s].ms += v.activeMs; }

  // ---- variantes da fusão dos olhos (monitor completo) ----
  for (const [vname, vcfg] of Object.entries(EYE_VARIANTS)) {
    const vr = vname === "app" ? run : runMonitor(rows, vcfg);
    const mins = (rows[n - 1].t - (vr.calT ?? rows[0].t)) / 60000;
    let red = 0, yel = 0, tt = 0;
    for (let i = 1; i < n; i++) if (vr.calT !== null && rows[i].t > vr.calT) { const d = rows[i].t - rows[i - 1].t; tt += d; if (vr.state[i] === 2) red += d; if (vr.state[i] === 1) yel += d; }
    accVar(vname, `${s}_min`, mins);
    accVar(vname, `${s}_min_vermelho`, red / 60000);
    accVar(vname, `${s}_min_amarelo`, yel / 60000);
    accVar(vname, `${s}_alarmes`, vr.mon.stats.alarmes);
    accVar(vname, `${s}_microssonos`, vr.mon.stats.eventos.microssono);
    accVar(vname, `${s}_fech05`, vr.mon.stats.eventos.fechamento);
    if (s === "s5") {
      const truthV = Array.from(eye, (v, i) => (!rows[i].face || vr.calT === null || rows[i].t <= vr.calT ? null : v === 1 ? true : v === 0 ? false : null));
      const c = confusion(Array.from(vr.closed, Boolean), truthV);
      accVar(vname, "s5_tp", c.tp); accVar(vname, "s5_tn", c.tn); accVar(vname, "s5_fp", c.fp); accVar(vname, "s5_fn", c.fn);
      accVar(vname, "s5_kappa_soma", c.kappa); accVar(vname, "s5_n", 1);
      const notOpen = [];
      for (const k of ["eyes_state/close", "eyes_state/closing", "eyes_state/opening"]) for (const [a, b] of acts[k] || []) notOpen.push([f2ms(a), f2ms(b + 1)]);
      const annC = (acts["eyes_state/close"] || []).map(([a, b]) => [f2ms(a), f2ms(b + 1)]).filter(([a]) => vr.calT !== null && a > vr.calT);
      for (const minMs of [500, 1000]) {
        const A = annC.filter(([a, b]) => b - a >= minMs);
        const D = vr.episodes.filter(([a, b]) => b - a >= minMs);
        const m = matchEvents(D, A);
        const lenient = D.filter((d) => notOpen.some((a) => Math.min(d[1], a[1]) - Math.max(d[0], a[0]) >= 0)).length;
        accVar(vname, `s5_anot_${minMs}`, A.length); accVar(vname, `s5_det_${minMs}`, D.length);
        accVar(vname, `s5_tp_anot_${minMs}`, m.tpAnn); accVar(vname, `s5_det_sobrepoe_${minMs}`, lenient);
      }
    }
  }
  // ---- ajuste final: alternativas no monitor completo ----
  for (const name of Object.keys(AJUSTES)) {
    const vr = name === "final" ? run : runMonitor(rows, null, mergeCfg(AJUSTES[name].over));
    let red = 0, yel = 0, tt = 0;
    for (let i = 1; i < n; i++) if (vr.calT !== null && rows[i].t > vr.calT) { const d = rows[i].t - rows[i - 1].t; tt += d; if (vr.state[i] === 2) red += d; if (vr.state[i] === 1) yel += d; }
    const st = vr.mon.stats;
    for (const [k, v] of Object.entries({ min: tt / 60000, min_vermelho: red / 60000, min_amarelo: yel / 60000, alarmes: st.alarmes, avisos: st.avisos, microssonos: st.eventos.microssono, fech05: st.eventos.fechamento, bocejos: st.eventos.bocejo, quedas: st.eventos.queda, cabeceios: st.eventos.cabeceio })) accAj(name, `${s}_${k}`, v);
    if (s === "s1") { const o = ajAcc.get(name); for (const v of vr.perclos) if (Number.isFinite(v)) o.perclos_s1.push(v); }
    if (s === "s5") {
      const truthV = Array.from(eye, (v, i) => (!rows[i].face || vr.calT === null || rows[i].t <= vr.calT ? null : v === 1 ? true : v === 0 ? false : null));
      const c = confusion(Array.from(vr.closed, Boolean), truthV);
      accAj(name, "s5_kappa_soma", c.kappa); accAj(name, "s5_n", 1);
      const annC = (acts["eyes_state/close"] || []).map(([a, b]) => [f2ms(a), f2ms(b + 1)]).filter(([a]) => vr.calT !== null && a > vr.calT);
      for (const minMs of [500, 1000]) {
        const A = annC.filter(([a, b]) => b - a >= minMs);
        const D = vr.episodes.filter(([a, b]) => b - a >= minMs);
        const m = matchEvents(D, A);
        accAj(name, `s5_anot_${minMs}`, A.length); accAj(name, `s5_det_${minMs}`, D.length); accAj(name, `s5_tp_anot_${minMs}`, m.tpAnn); accAj(name, `s5_tp_det_${minMs}`, m.tpDet);
      }
      // microssonos do app (vermelho imediato) que caem sobre um fechamento anotado >= 1 s
      const A1 = annC.filter(([a, b]) => b - a >= 1000);
      const thr = mergeCfg(AJUSTES[name].over).eyes.microsleepMs;
      const Dm = vr.episodes.filter(([a, b]) => b - a >= thr);
      const mm = matchEvents(Dm, A1);
      accAj(name, "s5_micro_anot1s_revocados", mm.tpAnn); accAj(name, "s5_micro_casados", mm.tpDet); accAj(name, "s5_micro_episodios", Dm.length);
    }
  }
  const activeMs = rows[n - 1].t - (calT ?? rows[0].t);
  const counts = { ...mon.stats.eventos };
  const hours = activeMs / 3600000;
  const st = mon.stats.tempoEstadoMs;
  const stTot = st[0] + st[1] + st[2] || 1;
  sessRows.push({
    id, sessao: s, sujeito: sub, minutos_ativos: activeMs / 60000, fps: hz, taxa_rosto: meta.quadros_com_rosto / meta.quadros,
    limiar_ear: cal.earThr, pitch_base: cal.pitch0, jaw_base: cal.jaw0,
    bocejos: counts.bocejo, quedas: counts.queda, cabeceios: counts.cabeceio, fechamentos_05_1s: counts.fechamento, microssonos: counts.microssono,
    alarmes: mon.stats.alarmes, avisos: mon.stats.avisos, dings: mon.stats.dings,
    alarmes_por_hora: mon.stats.alarmes / hours, avisos_por_hora: mon.stats.avisos / hours,
    frac_alerta: st[0] / stTot, frac_atencao: st[1] / stTot, frac_sonolento: st[2] / stTot,
    ...(() => {
      let nA = 0, nC = 0, nB = 0;
      const ld = [];
      for (let i = 0; i < n; i++) {
        if (!rows[i].face || calT === null || rows[i].t <= calT) continue;
        nA++;
        nC += run.closed[i];
        nB += run.byBlend[i];
        const v = (rows[i].ldl + rows[i].ldr) / 2;
        if (Number.isFinite(v)) ld.push(v);
      }
      return { frac_fechado: nC / (nA || 1), frac_fechado_caminho_A: nB / (nA || 1), eyelookdown_mediana: E.median(ld) };
    })(),
  });

  // ---- s1: eventos por ação anotada ----
  if (s === "s1") {
    const actionOf = new Array(n).fill("sem_anotacao");
    for (const [k, ivs] of Object.entries(acts)) if (k.startsWith("driver_actions/")) for (const [a, b] of ivs) for (let i = a; i <= Math.min(b, n - 1); i++) actionOf[i] = k.split("/")[1];
    const addMs = (act, key, v) => {
      const o = acaoAcc.get(act) || { acao: act, minutos: 0, bocejo: 0, queda: 0, cabeceio: 0, fechamento: 0, microssono: 0, alarme: 0, aviso: 0, min_sonolento: 0, min_atencao: 0 };
      o[key] += v;
      acaoAcc.set(act, o);
    };
    for (let i = 1; i < n; i++) {
      if (calT === null || rows[i].t <= calT) continue;
      const dtm = (rows[i].t - rows[i - 1].t) / 60000;
      addMs(actionOf[i], "minutos", dtm);
      if (state[i] === 2) addMs(actionOf[i], "min_sonolento", dtm);
      if (state[i] === 1) addMs(actionOf[i], "min_atencao", dtm);
    }
    for (const e of evs) if (["bocejo", "queda", "cabeceio", "fechamento", "microssono", "alarme", "aviso"].includes(e.type)) addMs(actionOf[e.frame], e.type, 1);
    for (const act of ["TOTAL"]) {
      addMs(act, "minutos", activeMs / 60000);
      for (const e of evs) if (["bocejo", "queda", "cabeceio", "fechamento", "microssono", "alarme", "aviso"].includes(e.type)) addMs(act, e.type, 1);
      let sonol = 0, aten = 0;
      for (let i = 1; i < n; i++) if (calT !== null && rows[i].t > calT) { const d = (rows[i].t - rows[i - 1].t) / 60000; if (state[i] === 2) sonol += d; if (state[i] === 1) aten += d; }
      addMs(act, "min_sonolento", sonol);
      addMs(act, "min_atencao", aten);
    }
  }

  if (s !== "s5") continue;

  // ---- s5: diagnóstico da regra dos olhos (quadros anotados, após a calibração) ----
  for (let i = 0; i < n; i++) {
    if (!rows[i].face || calT === null || rows[i].t <= calT || (eye[i] !== 0 && eye[i] !== 1)) continue;
    const g = eye[i] === 1 ? diagOlhos.fechado : diagOlhos.aberto;
    g.ld.push((rows[i].ldl + rows[i].ldr) / 2);
    g.bm.push(Math.min(rows[i].bl, rows[i].br));
  }

  // ---- s5: olhos por quadro (por vídeo, algumas variantes) ----
  const truth = Array.from(eye, (v, i) => (!rows[i].face || calT === null || rows[i].t <= calT ? null : v === 1 ? true : v === 0 ? false : null));
  const perFrame = {
    ear_v1: rows.map((r) => r.face && (r.earL + r.earR) / 2 < cal.earThr),
    blend050: rows.map((r) => r.face && Math.min(r.bl, r.br) >= 0.5),
    app: Array.from(closedApp, Boolean),
  };
  for (const [name, pred] of Object.entries(perFrame)) {
    const c = confusion(pred, truth);
    olhosRows.push({ id, sujeito: sub, variante: name, quadros_fechado: c.tp + c.fn, quadros_aberto: c.tn + c.fp, sensibilidade: c.sens, especificidade: c.spec, acuracia_bal: c.bal, kappa: c.kappa, precisao: c.prec });
  }

  // ---- s5: fechamentos por evento ----
  const annClose = (acts["eyes_state/close"] || []).map(([a, b]) => [f2ms(a), f2ms(b + 1)]).filter(([a]) => calT === null || a > calT);
  for (const minMs of [500, 1000]) {
    const A = annClose.filter(([a, b]) => b - a >= minMs);
    const D = closures.filter(([a, b]) => b - a >= minMs);
    const m = matchEvents(D, A);
    fechRows.push({ id, sujeito: sub, duracao_min_ms: minMs, n_anotados: A.length, n_detectados: D.length, tp_anot: m.tpAnn, tp_det: m.tpDet, revocacao: A.length ? m.tpAnn / A.length : NaN, precisao: D.length ? m.tpDet / D.length : NaN });
  }
  {
    // Viés de duração: o app mede do meio do fechamento (eyeBlink >= 0,5) ao meio da reabertura; a anotação
    // 'close' exclui o 'closing' e o 'opening'. Pares um-para-um pela maior sobreposição, ambos >= 0,3 s.
    const D = closures.filter(([a, b]) => b - a >= 300), A = annClose.filter(([a, b]) => b - a >= 300);
    const pairs = [];
    D.forEach((d, i) => A.forEach((a, j) => { const ov = Math.min(d[1], a[1]) - Math.max(d[0], a[0]); if (ov > 0) pairs.push([ov, i, j]); }));
    pairs.sort((x, y) => y[0] - x[0]);
    const ud = new Set(), ua = new Set();
    for (const [, i, j] of pairs) if (!ud.has(i) && !ua.has(j)) { ud.add(i); ua.add(j); durDiffs.push({ id, sujeito: sub, dif_ms: (D[i][1] - D[i][0]) - (A[j][1] - A[j][0]) }); }
  }

  // ---- s5: bocejos (app, grade, v1) ----
  const annY = [];
  for (const [k, ivs] of Object.entries(acts)) if (k.startsWith("yawning/")) for (const [a, b] of ivs) if (f2ms(a) > (calT ?? 0)) annY.push({ iv: [f2ms(a), f2ms(b + 1)], hand: /with hand/.test(k) && !/without/.test(k) });
  const runYawn = (cfgY) => {
    const y = new E.YawnDetector(cfgY);
    y.setBaseline({ jaw0: cal.jaw0, mar0: cal.mar0 });
    const out = [];
    let start = null;
    for (let i = 0; i < n; i++) {
      if (calT === null || rows[i].t <= calT) continue;
      const r = rows[i];
      const o = y.update(r.t, { face: r.face, jawOpen: r.jaw, mar: r.mar });
      if (o.event) start = r.t - o.openMs;
      if (o.ended) { out.push([o.ended.start, o.ended.end]); start = null; }
    }
    if (start !== null) out.push([start, rows[n - 1].t]);
    return out;
  };
  const runYawnV1 = () => {
    const out = [];
    let since = null, emitted = false;
    for (let i = 0; i < n; i++) {
      if (calT === null || rows[i].t <= calT) continue;
      const r = rows[i];
      if (r.face && r.mar > 0.6) {
        since = since ?? r.t;
        if (!emitted && r.t - since >= 2000) { emitted = true; out.push([since, r.t]); }
        if (emitted) out[out.length - 1][1] = r.t;
      } else { since = null; emitted = false; }
    }
    return out;
  };
  const evalY = (D) => {
    const m = matchEvents(D, annY.map((a) => a.iv));
    const hand = annY.filter((a) => a.hand).length;
    let tpHand = 0, tpNo = 0;
    for (const j of m.matchedAnn) (annY[j].hand ? tpHand++ : tpNo++);
    return { n_anotados: annY.length, n_com_mao: hand, n_sem_mao: annY.length - hand, n_detectados: D.length, tp_det: m.tpDet, tp_com_mao: tpHand, tp_sem_mao: tpNo, fp: D.length - m.tpDet };
  };
  bocejoRows.push({ id, sujeito: sub, detector: YAWN_APP_LABEL, ...evalY(runYawn(E.CFG.yawn)) });
  bocejoRows.push({ id, sujeito: sub, detector: "v1 (MAR>0,6 por 2 s)", ...evalY(runYawnV1()) });
  for (const g of YAWN_GRID) {
    const r = evalY(runYawn({ ...E.CFG.yawn, jawOpen: g.jawOpen, minMs: g.minMs }));
    const k = `jaw>=${g.jawOpen} ${g.minMs} ms`;
    const acc = gradeAcc.get(k) || { limiar_jaw: g.jawOpen, min_ms: g.minMs, n_anotados: 0, n_com_mao: 0, n_sem_mao: 0, n_detectados: 0, tp_det: 0, tp_com_mao: 0, tp_sem_mao: 0, fp_s5: 0 };
    for (const key of ["n_anotados", "n_com_mao", "n_sem_mao", "n_detectados", "tp_det", "tp_com_mao", "tp_sem_mao"]) acc[key] += r[key];
    acc.fp_s5 += r.fp;
    gradeAcc.set(k, acc);
  }
  log(`${id}: ${n} quadros, limiar EAR ${f3(cal.earThr)}, pitch base ${cal.pitch0.toFixed(1)}°, eventos ${JSON.stringify(counts)}, alarmes ${mon.stats.alarmes}`);
}

// s1: falsos bocejos por hora da grade (fala e direção real)
{
  const s1files = files.filter((f) => sess(f) === "s1");
  let s1Hours = 0;
  const fpPerGrid = new Map();
  for (const file of s1files) {
    const id = file.replace(".csv.gz", "");
    const rows = readSignals(path.join(SIG, file));
    const cal = calibrate(rows);
    let calT = null;
    { let fm = 0, last = null; for (const r of rows) { const dt = last === null ? 0 : Math.min(100, r.t - last); last = r.t; if (r.face) fm += dt; if (fm >= E.CFG.calib.neutralMs) { calT = r.t; break; } } }
    s1Hours += (rows[rows.length - 1].t - calT) / 3600000;
    for (const g of YAWN_GRID) {
      const y = new E.YawnDetector({ ...E.CFG.yawn, jawOpen: g.jawOpen, minMs: g.minMs });
      y.setBaseline({ jaw0: cal.jaw0, mar0: cal.mar0 });
      let c = 0;
      for (const r of rows) if (r.t > calT && y.update(r.t, { face: r.face, jawOpen: r.jaw, mar: r.mar }).event) c++;
      const k = `jaw>=${g.jawOpen} ${g.minMs} ms`;
      fpPerGrid.set(k, (fpPerGrid.get(k) || 0) + c);
    }
    log(`${id}: s1 ${((rows[rows.length - 1].t - calT) / 60000).toFixed(1)} min ativos`);
  }
  for (const [k, acc] of gradeAcc) {
    acc.revocacao_total = acc.tp_com_mao + acc.tp_sem_mao ? (acc.tp_com_mao + acc.tp_sem_mao) / acc.n_anotados : 0;
    acc.revocacao_sem_mao = acc.tp_sem_mao / (acc.n_sem_mao || 1);
    acc.revocacao_com_mao = acc.tp_com_mao / (acc.n_com_mao || 1);
    acc.precisao_s5 = acc.tp_det / (acc.n_detectados || 1);
    acc.bocejos_falsos_s1 = fpPerGrid.get(k) || 0;
    acc.horas_s1 = s1Hours;
    acc.bocejos_falsos_por_hora_s1 = (fpPerGrid.get(k) || 0) / (s1Hours || 1);
  }
}

// ---------------------------------------------------------------------------
// Agregados + relatório
// ---------------------------------------------------------------------------
log("\n## Arquivos");
writeCSV("app_validacao_dmd_olhos.csv", olhosRows);
writeCSV("app_validacao_dmd_fechamentos.csv", fechRows);
writeCSV("app_validacao_dmd_bocejo.csv", bocejoRows);
writeCSV("app_validacao_dmd_bocejo_grade.csv", [...gradeAcc.values()]);
writeCSV("app_validacao_dmd_sessoes.csv", sessRows);
const acoes = [...acaoAcc.values()].map((o) => ({ ...o, horas: o.minutos / 60, eventos_sono_por_hora: (o.bocejo + o.queda + o.cabeceio + o.fechamento + o.microssono) / (o.minutos / 60 || 1), alarmes_por_hora: o.alarme / (o.minutos / 60 || 1), frac_sonolento: o.min_sonolento / (o.minutos || 1), frac_atencao: o.min_atencao / (o.minutos || 1) }));
writeCSV("app_validacao_dmd_acoes_s1.csv", acoes);
writeCSV("app_validacao_dmd_pose_sinal.csv", poseRows);
const q = (a, pp) => { const v = a.filter(Number.isFinite).sort((x, y) => x - y); return v.length ? v[Math.floor(pp * (v.length - 1))] : NaN; };
// Ajuste final: tabela das alternativas
const ajRows = [...ajAcc.values()].map((o) => ({
  alternativa: o.alternativa,
  descricao: o.descricao,
  s1_min: o.s1_min,
  s1_pct_vermelho: (100 * o.s1_min_vermelho) / (o.s1_min || 1),
  s1_pct_amarelo: (100 * o.s1_min_amarelo) / (o.s1_min || 1),
  s1_alarmes: o.s1_alarmes,
  s1_alarmes_h: o.s1_alarmes / (o.s1_min / 60 || 1),
  s1_avisos_h: o.s1_avisos / (o.s1_min / 60 || 1),
  s1_microssonos: o.s1_microssonos,
  s1_fech05_1s: o.s1_fech05,
  s1_bocejos: o.s1_bocejos,
  s1_quedas_cabeceios: o.s1_quedas + o.s1_cabeceios,
  s1_perclos_p50_pct: 100 * q(o.perclos_s1, 0.5),
  s1_perclos_p95_pct: 100 * q(o.perclos_s1, 0.95),
  s1_perclos_max_pct: 100 * q(o.perclos_s1, 1),
  s5_min: o.s5_min,
  s5_kappa: o.s5_kappa_soma / (o.s5_n || 1),
  s5_revoc_fech05: o.s5_tp_anot_500 / (o.s5_anot_500 || 1),
  s5_prec_fech05: o.s5_tp_det_500 / (o.s5_det_500 || 1),
  s5_revoc_fech1: o.s5_tp_anot_1000 / (o.s5_anot_1000 || 1),
  s5_prec_fech1: o.s5_tp_det_1000 / (o.s5_det_1000 || 1),
  s5_micro_revoc_anot1s: `${o.s5_micro_anot1s_revocados}/${o.s5_anot_1000}`,
  s5_microssonos: o.s5_microssonos,
  s5_bocejos: o.s5_bocejos,
  s5_pct_vermelho: (100 * o.s5_min_vermelho) / (o.s5_min || 1),
  s5_pct_amarelo: (100 * o.s5_min_amarelo) / (o.s5_min || 1),
  s5_alarmes: o.s5_alarmes,
}));
writeCSV("app_validacao_dmd_ajuste_final.csv", ajRows);
const diagRows = Object.entries(diagOlhos).map(([grupo, g]) => ({ grupo: `${grupo}_anotado_s5`, quadros: g.ld.length, eyelookdown_p10: q(g.ld, 0.1), eyelookdown_p50: q(g.ld, 0.5), eyelookdown_p90: q(g.ld, 0.9), eyeblink_min_p10: q(g.bm, 0.1), eyeblink_min_p50: q(g.bm, 0.5), eyeblink_min_p90: q(g.bm, 0.9) }));
writeCSV("app_validacao_dmd_diag_olhos.csv", diagRows);

const mean = (a) => a.reduce((x, y) => x + y, 0) / (a.length || 1);
const md = [];
md.push("# Validação offline do motor do app (detector v2) no DMD", "");
md.push(`Gerado em ${new Date().toISOString()} por \`tools/app_validacao/validar_motor_dmd.mjs\` (log: \`${path.relative(ROOT, logPath)}\`). ` +
  "Sinais por quadro extraídos com o MESMO `face_landmarker.task` da PWA (`tools/app_validacao/extrair_sinais_app.py`, MediaPipe 0.10.21 Tasks, CPU); " +
  "o código avaliado é o `app/pwa/engine.js` importado sem cópia, com a calibração do app (10 s iniciais de cada vídeo). O LSTM não entra (no app vale no máximo +10 pontos).", "");
md.push("## Olhos por quadro (DMD s5, sonolência encenada; quadros anotados 'open' x 'close', transições excluídas)", "");
md.push("| Regra | Sensibilidade | Especificidade | Acurácia bal. | Kappa |", "|---|---:|---:|---:|---:|");
for (const v of ["ear_v1", "blend050", "app"]) {
  const r = olhosRows.filter((x) => x.variante === v);
  md.push(`| ${v} | ${f3(mean(r.map((x) => x.sensibilidade)))} | ${f3(mean(r.map((x) => x.especificidade)))} | ${f3(mean(r.map((x) => x.acuracia_bal)))} | ${f3(mean(r.map((x) => x.kappa)))} |`);
}
md.push("", "Médias entre as 16 sessões. `ear_v1` = EAR < 0,75 x mediana dos 10 s iniciais (regra da v1); `blend050` = os dois eyeBlink >= 0,5; `app` = regra final do app (CFG.eyes).", "");
md.push("## Variantes da regra dos olhos no monitor completo (escolha da regra final)", "");
md.push("Cada variante roda o monitor inteiro (nível, alarmes) em s5 (sonolência encenada, com anotação de olho) e em s1 (direção real, motoristas acordados). " +
  "Revocação = fechamentos anotados >= X casados um-para-um com episódios detectados >= X; precisão = episódios >= X que se sobrepõem a olho não aberto anotado (fechado, fechando ou abrindo). " +
  "Kappa por quadro = média das 16 sessões s5.", "");
md.push("| Variante | Kappa s5 | Sens. s5 | Espec. s5 | Revoc. >=0,5 s | Precisão >=0,5 s | Revoc. >=1 s | Precisão >=1 s | % tempo vermelho s5 | % vermelho s1 | Alarmes/h s1 | Microssonos/h s1 | Fech. 0,5-1 s/h s1 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
const varRows = [];
for (const [name, o] of varAcc) {
  const row = {
    variante: name,
    config: JSON.stringify(EYE_VARIANTS[name]),
    kappa_s5: o.s5_kappa_soma / (o.s5_n || 1),
    sens_s5: o.s5_tp / (o.s5_tp + o.s5_fn || 1),
    espec_s5: o.s5_tn / (o.s5_tn + o.s5_fp || 1),
    revoc_05_s5: o.s5_tp_anot_500 / (o.s5_anot_500 || 1),
    precisao_05_s5: o.s5_det_sobrepoe_500 / (o.s5_det_500 || 1),
    revoc_1_s5: o.s5_tp_anot_1000 / (o.s5_anot_1000 || 1),
    precisao_1_s5: o.s5_det_sobrepoe_1000 / (o.s5_det_1000 || 1),
    pct_vermelho_s5: (100 * o.s5_min_vermelho) / (o.s5_min || 1),
    pct_vermelho_s1: (100 * o.s1_min_vermelho) / (o.s1_min || 1),
    pct_amarelo_s1: (100 * o.s1_min_amarelo) / (o.s1_min || 1),
    alarmes_h_s1: o.s1_alarmes / ((o.s1_min || 1) / 60),
    microssonos_h_s1: o.s1_microssonos / ((o.s1_min || 1) / 60),
    fech05_h_s1: o.s1_fech05 / ((o.s1_min || 1) / 60),
    alarmes_h_s5: o.s5_alarmes / ((o.s5_min || 1) / 60),
    anotados_05_s5: o.s5_anot_500, anotados_1_s5: o.s5_anot_1000, minutos_s1: o.s1_min, minutos_s5: o.s5_min,
  };
  varRows.push(row);
  md.push(`| ${name} | ${f3(row.kappa_s5)} | ${f3(row.sens_s5)} | ${f3(row.espec_s5)} | ${f3(row.revoc_05_s5)} | ${f3(row.precisao_05_s5)} | ${f3(row.revoc_1_s5)} | ${f3(row.precisao_1_s5)} | ${row.pct_vermelho_s5.toFixed(1)} | ${row.pct_vermelho_s1.toFixed(1)} | ${row.alarmes_h_s1.toFixed(1)} | ${row.microssonos_h_s1.toFixed(1)} | ${row.fech05_h_s1.toFixed(1)} |`);
}
const v1Row = { variante: "v1_app_anterior (EAR + PERCLOS>15% + olho fechado>=1 s; sem LSTM)", pct_vermelho_s5: (100 * v1Acc.s5.red) / (v1Acc.s5.ms || 1), pct_vermelho_s1: (100 * v1Acc.s1.red) / (v1Acc.s1.ms || 1), alarmes_h_s1: v1Acc.s1.alarms / (v1Acc.s1.ms / 3600000 || 1), alarmes_h_s5: v1Acc.s5.alarms / (v1Acc.s5.ms / 3600000 || 1), minutos_s1: v1Acc.s1.ms / 60000, minutos_s5: v1Acc.s5.ms / 60000 };
varRows.push(v1Row);
md.push(`| ${v1Row.variante} | | | | | | | | ${v1Row.pct_vermelho_s5.toFixed(1)} | ${v1Row.pct_vermelho_s1.toFixed(1)} | ${v1Row.alarmes_h_s1.toFixed(1)} | | |`);
writeCSV("app_validacao_dmd_variantes_olhos.csv", varRows);
md.push("", "`app` = configuração final (CFG.eyes em app/pwa/engine.js).", "");
md.push("## Ajuste final (27/09/2026): alternativas avaliadas no monitor completo", "",
  "Cada linha é a configuração FINAL do app com UMA mudança. s1 = direção real, motoristas acordados (falso alarme); s5 = sonolência encenada (detecção). " +
  "Revocação/precisão de fechamentos: casamento um-para-um estrito com a anotação 'close'. \"Microssonos sobre fechamento anotado >= 1 s\" = fechamentos anotados >= 1 s que dispararam o vermelho imediato. " +
  "A 'final' é a do app; as outras foram avaliadas e NÃO aplicadas (motivo no app/pwa/README.md). Alarmes = entradas no vermelho; com a v2.1 o estado volta ao amarelo entre microssonos, então alarmes/h não é comparável com a v2.0 — compare também o % do tempo em vermelho (sirene).", "",
  "| Alternativa | s1 % vermelho | s1 % amarelo | s1 alarmes/h | s1 avisos/h | s1 fech. 0,5-1 s | s1 microssonos | s1 bocejos | s1 PERCLOS p95 | s5 kappa | s5 revoc. >=1 s | s5 microssonos sobre fech. >=1 s | s5 bocejos | s5 % vermelho | s5 alarmes |",
  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
for (const r of ajRows) md.push(`| ${r.alternativa} | ${r.s1_pct_vermelho.toFixed(1)} | ${r.s1_pct_amarelo.toFixed(1)} | ${r.s1_alarmes_h.toFixed(2)} | ${r.s1_avisos_h.toFixed(2)} | ${r.s1_fech05_1s} | ${r.s1_microssonos} | ${r.s1_bocejos} | ${r.s1_perclos_p95_pct.toFixed(1)} % | ${f3(r.s5_kappa)} | ${f3(r.s5_revoc_fech1)} | ${r.s5_micro_revoc_anot1s} | ${r.s5_bocejos} | ${r.s5_pct_vermelho.toFixed(1)} | ${r.s5_alarmes} |`);
md.push("", "Tabela completa: `results/app_validacao_dmd_ajuste_final.csv`.", "");
{
  const dd = durDiffs.map((x) => x.dif_ms);
  const bySub = [...new Set(durDiffs.map((x) => x.id))].map((id) => ({ id, med: E.median(durDiffs.filter((x) => x.id === id).map((x) => x.dif_ms)), n: durDiffs.filter((x) => x.id === id).length }));
  md.push("## Duração dos fechamentos: app - anotação (DMD s5)", "",
    `Pares um-para-um (fechamentos >= 0,3 s do app x 'close' anotado >= 0,3 s): n = ${dd.length}; diferença p25 / mediana / p75 = ${q(dd, 0.25).toFixed(0)} / ${q(dd, 0.5).toFixed(0)} / ${q(dd, 0.75).toFixed(0)} ms. ` +
    "O app mede do meio do fechamento (os dois eyeBlink >= 0,5) ao meio da reabertura; a anotação 'close' exclui as fases 'closing' e 'opening'. " +
    `Por sessão (mediana, ms): ${bySub.map((b) => `${b.id.split("_").slice(0, 3).join("_")} ${b.med.toFixed(0)} (n=${b.n})`).join("; ")}.`, "");
  log(`duração app - anotação (s5, pares >= 0,3 s): n=${dd.length}, p25/p50/p75 = ${q(dd, 0.25).toFixed(0)}/${q(dd, 0.5).toFixed(0)}/${q(dd, 0.75).toFixed(0)} ms`);
}
md.push("## Fechamentos por evento (app, DMD s5)", "", "| Duração mínima | Anotados | Detectados | Revocação | Precisão |", "|---|---:|---:|---:|---:|");
for (const minMs of [500, 1000]) {
  const r = fechRows.filter((x) => x.duracao_min_ms === minMs);
  const A = r.reduce((a, x) => a + x.n_anotados, 0), D = r.reduce((a, x) => a + x.n_detectados, 0);
  const ta = r.reduce((a, x) => a + x.tp_anot, 0), td = r.reduce((a, x) => a + x.tp_det, 0);
  md.push(`| >= ${minMs} ms | ${A} | ${D} | ${f3(ta / (A || 1))} | ${f3(td / (D || 1))} |`);
}
md.push("", "## Bocejos por evento (DMD s5: bocejos anotados com e sem a mão na frente da boca)", "", "| Detector | Anotados (sem/com mão) | Detectados | Revocação sem mão | Revocação com mão | Precisão |", "|---|---:|---:|---:|---:|---:|");
for (const det of [...new Set(bocejoRows.map((x) => x.detector))]) {
  const r = bocejoRows.filter((x) => x.detector === det);
  const S = (k) => r.reduce((a, x) => a + x[k], 0);
  md.push(`| ${det} | ${S("n_anotados")} (${S("n_sem_mao")}/${S("n_com_mao")}) | ${S("n_detectados")} | ${f3(S("tp_sem_mao") / (S("n_sem_mao") || 1))} | ${f3(S("tp_com_mao") / (S("n_com_mao") || 1))} | ${f3(S("tp_det") / (S("n_detectados") || 1))} |`);
}
md.push("", "Grade (jawOpen x duração mínima) com os falsos bocejos por hora na direção real (s1): `results/app_validacao_dmd_bocejo_grade.csv`.", "");
md.push("| jawOpen >= | Duração mín. | Revocação sem mão | Revocação com mão | Precisão s5 | Falsos/h em s1 |", "|---:|---:|---:|---:|---:|---:|");
for (const g of gradeAcc.values()) md.push(`| ${g.limiar_jaw} | ${g.min_ms} ms | ${f3(g.revocacao_sem_mao)} | ${f3(g.revocacao_com_mao)} | ${f3(g.precisao_s5)} | ${f3(g.bocejos_falsos_por_hora_s1)} |`);
md.push("", "## Monitor completo por tipo de sessão", "", "| Sessões | Minutos ativos | Bocejos | Quedas | Cabeceios | Fech. 0,5-1 s | Microssonos | Alarmes (vermelho) | Alarmes/h | Avisos/h | % tempo SONOLENTO | % ATENÇÃO |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
for (const s of ["s5", "s1"]) {
  const r = sessRows.filter((x) => x.sessao === s);
  if (!r.length) continue;
  const S = (k) => r.reduce((a, x) => a + x[k], 0);
  const min = S("minutos_ativos");
  const wS = r.reduce((a, x) => a + x.frac_sonolento * x.minutos_ativos, 0) / (min || 1);
  const wA = r.reduce((a, x) => a + x.frac_atencao * x.minutos_ativos, 0) / (min || 1);
  md.push(`| ${s} (${r.length}) | ${min.toFixed(1)} | ${S("bocejos")} | ${S("quedas")} | ${S("cabeceios")} | ${S("fechamentos_05_1s")} | ${S("microssonos")} | ${S("alarmes")} | ${(S("alarmes") / (min / 60)).toFixed(2)} | ${(S("avisos") / (min / 60)).toFixed(2)} | ${(100 * wS).toFixed(1)} | ${(100 * wA).toFixed(1)} |`);
}
md.push("", "## Direção real (s1) por ação anotada", "", "| Ação | Minutos | Eventos de sono/h | Alarmes/h | % SONOLENTO | Bocejos | Quedas | Cabeceios | Fech. | Microssonos |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
for (const o of acoes.sort((a, b) => b.minutos - a.minutos)) md.push(`| ${o.acao} | ${o.minutos.toFixed(1)} | ${o.eventos_sono_por_hora.toFixed(1)} | ${o.alarmes_por_hora.toFixed(1)} | ${(100 * o.frac_sonolento).toFixed(1)} | ${o.bocejo} | ${o.queda} | ${o.cabeceio} | ${o.fechamento} | ${o.microssono} |`);
{
  md.push("", "## Diagnóstico da regra dos olhos", "",
    "Quadros de s5 anotados como olho fechado x aberto (após a calibração): distribuição da média de eyeLookDown e do menor eyeBlink. " +
    "Por sessão (`results/app_validacao_dmd_sessoes.csv`): fração dos quadros ativos com rosto marcados como fechados, fração pelo caminho A (blendshape) e eyeLookDown mediano.", "",
    "| Grupo | Quadros | eyeLookDown p10 / p50 / p90 | menor eyeBlink p10 / p50 / p90 |", "|---|---:|---|---|");
  for (const r of diagRows) md.push(`| ${r.grupo} | ${r.quadros} | ${f3(r.eyelookdown_p10)} / ${f3(r.eyelookdown_p50)} / ${f3(r.eyelookdown_p90)} | ${f3(r.eyeblink_min_p10)} / ${f3(r.eyeblink_min_p50)} / ${f3(r.eyeblink_min_p90)} |`);
  const s1s = sessRows.filter((r) => r.sessao === "s1").sort((a, b) => b.frac_sonolento - a.frac_sonolento);
  md.push("", "Direção real (s1), sessões com mais tempo em SONOLENTO:", "", "| Sessão | % SONOLENTO | Alarmes | Microssonos | % quadros fechados | % fechados pelo caminho A | eyeLookDown mediano |", "|---|---:|---:|---:|---:|---:|---:|");
  for (const r of s1s.slice(0, 3)) md.push(`| ${r.id.split("_").slice(0, 3).join("_")} | ${(100 * r.frac_sonolento).toFixed(1)} | ${r.alarmes} | ${r.microssonos} | ${(100 * r.frac_fechado).toFixed(1)} | ${(100 * r.frac_fechado_caminho_A).toFixed(1)} | ${f3(r.eyelookdown_mediana)} |`);
  const outros = s1s.slice(1);
  md.push("", `Demais ${outros.length} sessões s1: % quadros fechados mediano ${(100 * E.median(outros.map((r) => r.frac_fechado))).toFixed(1)}, eyeLookDown mediano ${f3(E.median(outros.map((r) => r.eyelookdown_mediana)))}.`);
}
{
  const rs = poseRows.map((r) => r.r_pitch_nariz).filter(Number.isFinite).sort((a, b) => a - b);
  const pos = rs.filter((v) => v > 0).length;
  const meds = poseRows.map((r) => r.pitch_mediana).filter(Number.isFinite).sort((a, b) => a - b);
  md.push("", "## Sinal do pitch da matriz de pose (convenção do app: pitch > 0 = cabeça para baixo)", "",
    "Quadro a quadro, correlação entre o pitch (`poseFromMatrix`) e a posição vertical relativa da ponta do nariz na imagem, " +
    "(nariz_y - olhos_y) / (queixo_y - olhos_y) (marcos 1, 33/263 e 152), que aumenta quando a cabeça baixa (o nariz se projeta para baixo). " +
    "Por vídeo em `results/app_validacao_dmd_pose_sinal.csv`.", "",
    "| Vídeos | Correlação > 0 | Correlação mín. | Mediana | Máx. | Pitch mediano por vídeo (mín. / mediana / máx.) |", "|---:|---:|---:|---:|---:|---|",
    `| ${rs.length} | ${pos} | ${f3(rs[0])} | ${f3(rs[rs.length >> 1])} | ${f3(rs[rs.length - 1])} | ${meds[0].toFixed(1)}° / ${meds[meds.length >> 1].toFixed(1)}° / ${meds[meds.length - 1].toFixed(1)}° |`, "",
    "Pitch mediano negativo é o esperado no DMD: a câmera fica abaixo do rosto (no painel), o que equivale a ver a cabeça levemente para cima. " +
    "O app usa sempre o pitch RELATIVO à calibração (Δpitch), então essa inclinação fixa da câmera não conta como queda.");
}
fs.writeFileSync(path.join(RES, "app_validacao_dmd.md"), md.join("\n") + "\n");
log(`  -> results/app_validacao_dmd.md`);
log("\n" + md.join("\n"));
fs.mkdirSync(path.dirname(logPath), { recursive: true });
fs.writeFileSync(logPath, logLines.join("\n") + "\n");
