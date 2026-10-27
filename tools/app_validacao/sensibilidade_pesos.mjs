/**
 * Sensibilidade do NÍVEL do app aos números de projeto (CFG.score e CFG.levels de app/pwa/engine.js), no DMD.
 *
 *   node tools/app_validacao/sensibilidade_pesos.mjs              (na raiz do repositório; até 6 threads em paralelo)
 *   node tools/app_validacao/sensibilidade_pesos.mjs --workers=1  (tudo numa thread só; mesmo resultado)
 *   --saida=<pasta> grava o .csv, o .md e o log nessa pasta em vez de results/ (para testar sem sobrescrever).
 *
 * Contexto: os limiares de DETECÇÃO de cada sinal (olho fechado, bocejo, cabeça) foram escolhidos comparando
 * variantes contra a anotação do DMD (validar_motor_dmd.mjs). Os números do NÍVEL de 0 a 100 (pontos por evento,
 * tetos, decaimentos e os limiares 40 e 70) são regras de projeto; aqui se mede quanto o comportamento muda com eles.
 *
 * Mesmos sinais e mesma calibração da validação existente (validar_motor_dmd.mjs, seção "Ajuste final"): sinais por
 * quadro do Face Landmarker (data/signals/app_tasks/dmd/<id>.csv.gz), monitor COMPLETO do app (DrowsinessMonitor de
 * ../../app/pwa/engine.js, importado sem cópia) com a calibração dele (10 s iniciais), LSTM fora (mlProbs = null).
 * Copiados de lá (aquele script roda tudo ao ser importado): leitura dos sinais e da anotação, entrada do quadro,
 * contabilidade do tempo por estado e o casamento um-para-um de eventos.
 *
 * Configurações (só CFG.score e CFG.levels mudam):
 *  - final: a do app, sem mudança. Precisa reproduzir a linha "final" de results/app_validacao_dmd_ajuste_final.csv;
 *    se não reproduzir, o script para sem gravar o .csv e o .md (o log explica o que divergiu).
 *  - uma mudança por vez: cada grupo de pontos (os pontos e o teto próprio do grupo) x 0,7, 0,85, 1,15 e 1,3, e o teto
 *    de fechamentos e microssonos somados (que os dois grupos dividem) com os mesmos fatores; cada decaimento x 0,5 e
 *    x 2 (os quatro decayMs, a descida máxima do nível e o tempo de ATENÇÃO mantida depois de um vermelho forçado);
 *    limiar de ATENÇÃO em 30 ou 50 e de SONOLENTO em 60 ou 80 (saídas na mesma distância).
 *  - 300 sorteios (gerador mulberry32, semente 42): cada peso, teto e decaimento x um fator uniforme de 0,7 a 1,3
 *    (4 casas) e os limiares 40 e 70 deslocados por um inteiro uniforme de -10 a +10.
 * Ao mover um limiar, acompanham os números definidos a partir dele: o piso de ATENÇÃO depois de um vermelho forçado
 * (postForced.floor = 40), o portão do LSTM (ml.minBase = 40) e o nível exibido no vermelho forçado (forcedFloor = 75,
 * 5 acima de 70). As linhas "só o limiar" movem só os quatro números de CFG.levels.
 * Fixos: CFG.calib, eyes, perclos, yawn, head e alarm, as rampas do PERCLOS (7,5 % e 15 %) e das piscadas (1,25 e 2 vezes
 * a base, mínimo de piscadas e janela), o tempo de contagem de fechamentos (só exibição) e o resto do portão do LSTM
 * (probabilidade e predições seguidas): são definições dos sinais, inclusive o microssono de 1 s e a cabeça caída com
 * olhos fechados (os vermelhos forçados). O script confere que os eventos detectados e o tempo de vermelho forçado
 * saem idênticos em todas as configurações.
 *
 * Saídas: results/app_sensibilidade_pesos.csv (uma linha por configuração, com todos os fatores),
 * results/app_sensibilidade_pesos.md e results/logs/app_sensibilidade_pesos_<data-hora>.log.
 */
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";
import { Worker, isMainThread, parentPort } from "node:worker_threads";
import * as E from "../../app/pwa/engine.js";

const SCRIPT = fileURLToPath(import.meta.url);
const ROOT = path.resolve(path.dirname(SCRIPT), "../..");
const SIG = path.join(ROOT, "data/signals/app_tasks/dmd");
const REF_CSV = path.join(ROOT, "results/app_validacao_dmd_ajuste_final.csv");
const ENGINE = path.join(ROOT, "app/pwa/engine.js");
/** md5 do motor: o log registra a versão usada e o script para se ela mudar durante a execução. */
const md5Motor = () => crypto.createHash("md5").update(fs.readFileSync(ENGINE)).digest("hex");
const MAX_WORKERS = 6;
const SEMENTE = 42;
const N_SORTEIOS = 300;
const FATORES_PONTOS = [0.7, 0.85, 1.15, 1.3];
const FATORES_DECAIMENTO = [0.5, 2];
const BOM_S1_MAX = 5; //  "boa" = s1 em vermelho até 5 % do tempo ...
const BOM_S5_MIN = 50; // ... e s5 em vermelho acima de 50 %

// ---------------------------------------------------------------------------
// Números do nível que variam: chave da coluna no CSV (f_<chave>), caminho em CFG, tipo e nome no relatório.
// ---------------------------------------------------------------------------
const PARAMS = [
  ["pts_perclos", "score.perclos.points", "peso", "pontos do PERCLOS em 15 %"],
  ["pts_fechamento", "score.longClosure.points", "peso", "pontos por fechamento de 0,5 a 1 s"],
  ["pts_microssono", "score.microsleep.points", "peso", "pontos por microssono"],
  ["pts_bocejo", "score.yawn.points", "peso", "pontos por bocejo"],
  ["pts_cabeca", "score.head.points", "peso", "pontos por queda ou cabeceio"],
  ["pts_piscadas", "score.blinkDur.points", "peso", "pontos das piscadas mais longas"],
  ["pts_lstm", "score.ml.points", "peso", "pontos do LSTM"],
  ["teto_perclos", "score.perclos.cap", "teto", "teto do PERCLOS"],
  ["teto_fechamentos", "score.closuresCap", "teto", "teto de fechamentos e microssonos somados"],
  ["teto_bocejo", "score.yawn.cap", "teto", "teto dos bocejos"],
  ["teto_cabeca", "score.head.cap", "teto", "teto da cabeça"],
  ["teto_piscadas", "score.blinkDur.cap", "teto", "teto das piscadas"],
  ["dec_fechamento", "score.longClosure.decayMs", "decaimento", "decaimento do fechamento"],
  ["dec_microssono", "score.microsleep.decayMs", "decaimento", "decaimento do microssono"],
  ["dec_bocejo", "score.yawn.decayMs", "decaimento", "decaimento do bocejo"],
  ["dec_cabeca", "score.head.decayMs", "decaimento", "decaimento da cabeça"],
  ["dec_descida", "score.maxFallPerS", "decaimento", "descida máxima do nível"],
  ["dec_pos_forcado", "score.postForced.holdMs", "decaimento", "ATENÇÃO mantida depois do vermelho forçado"],
].map(([key, cam, tipo, nome]) => ({ key, path: cam.split("."), tipo, nome }));
const getPath = (o, p) => p.reduce((a, k) => a[k], o);
const setPath = (o, p, v) => { getPath(o, p.slice(0, -1))[p[p.length - 1]] = v; };
const BASE = Object.fromEntries(PARAMS.map((p) => [p.key, getPath(E.CFG, p.path)]));
const NOME = Object.fromEntries(PARAMS.map((p) => [p.key, p.nome]));
const br = (v, d = 1) => E.fmtBR(v, d);
/** Valor de CFG com unidade, para o relatório (pontos, minutos, segundos ou pontos por segundo). */
function comUnidade(key, v) {
  if (key === "dec_descida") return `${br(v, Number.isInteger(v) ? 0 : 1)} pontos por segundo`;
  if (key.startsWith("dec_")) return v >= 60000 ? `${br(v / 60000, Number.isInteger(v / 60000) ? 0 : 1)} min` : `${br(v / 1000, 0)} s`;
  return `${br(v, Number.isInteger(v) ? 0 : 1)} pt`;
}
// Uma mudança por vez: grupos de pontos (pontos e o teto próprio). O teto de fechamentos e microssonos é
// compartilhado pelos dois grupos: fica fixo quando um deles muda e tem a sua própria linha.
const GRUPOS = [
  ["perclos", "PERCLOS", ["pts_perclos", "teto_perclos"]],
  ["fechamento", "fechamento de 0,5 a 1 s", ["pts_fechamento"]],
  ["microssono", "microssono", ["pts_microssono"]],
  ["teto_fechamentos", "teto de fechamentos e microssonos somados", ["teto_fechamentos"]],
  ["bocejo", "bocejo", ["pts_bocejo", "teto_bocejo"]],
  ["cabeca", "queda ou cabeceio", ["pts_cabeca", "teto_cabeca"]],
  ["piscadas", "piscadas mais longas", ["pts_piscadas", "teto_piscadas"]],
  ["lstm", "LSTM", ["pts_lstm"]],
];
const DECAIMENTOS = ["dec_fechamento", "dec_microssono", "dec_bocejo", "dec_cabeca", "dec_descida", "dec_pos_forcado"];
const L0 = E.CFG.levels;
const S0 = E.CFG.score;

/** CFG completo de uma configuração: cópia da final com os fatores e os deslocamentos dos limiares. */
function montarCfg(spec) {
  const cfg = structuredClone(E.CFG);
  for (const p of PARAMS) setPath(cfg, p.path, BASE[p.key] * spec.f[p.key]);
  const L = cfg.levels;
  L.attention = L0.attention + spec.dA;
  L.attentionExit = L0.attentionExit + spec.dA;
  L.drowsy = L0.drowsy + spec.dD;
  L.drowsyExit = L0.drowsyExit + spec.dD;
  if (spec.acompanha) {
    cfg.score.postForced.floor = S0.postForced.floor + spec.dA;
    cfg.score.ml.minBase = S0.ml.minBase + spec.dA;
    cfg.score.forcedFloor = S0.forcedFloor + spec.dD;
  }
  if (!(L.attentionExit < L.attention && L.attention < L.drowsy && L.drowsyExit < L.drowsy && L.attention <= L.drowsyExit)) throw new Error(`limiares fora de ordem: ${JSON.stringify(L)}`);
  return cfg;
}

/** Gerador pseudoaleatório mulberry32 (uniforme em [0, 1)). */
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

function montarConfiguracoes() {
  const um = Object.fromEntries(PARAMS.map((p) => [p.key, 1]));
  const specs = [{ id: "final", tipo: "final", grupo: "", nomeGrupo: "", descricao: "configuração final do app (sem mudança)", f: { ...um }, dA: 0, dD: 0, acompanha: true }];
  for (const [g, nome, keys] of GRUPOS) {
    const base = keys.map((k) => (k.startsWith("teto") && keys.length > 1 ? `teto ${comUnidade(k, BASE[k])}` : comUnidade(k, BASE[k]))).join(", ");
    const nomeGrupo = `${nome} (${base})`;
    for (const k of FATORES_PONTOS) specs.push({ id: `${g}_x${k}`, tipo: "uma_mudanca", grupo: g, nomeGrupo, curto: nome, valor: comUnidade(keys[0], BASE[keys[0]] * k), descricao: `${nomeGrupo} × ${br(k, 2).replace(/0$/, "")}`, f: { ...um, ...Object.fromEntries(keys.map((key) => [key, k])) }, dA: 0, dD: 0, acompanha: true });
  }
  for (const key of DECAIMENTOS) {
    const nomeGrupo = `${NOME[key]} (${comUnidade(key, BASE[key])})`;
    for (const k of FATORES_DECAIMENTO) specs.push({ id: `${key}_x${k}`, tipo: "uma_mudanca", grupo: key, nomeGrupo, curto: NOME[key], valor: comUnidade(key, BASE[key] * k), descricao: `${nomeGrupo} × ${br(k, k < 1 ? 1 : 0)} = ${comUnidade(key, BASE[key] * k)}`, f: { ...um, [key]: k }, dA: 0, dD: 0, acompanha: true });
  }
  for (const [dA, dD] of [[-10, 0], [10, 0], [0, -10], [0, 10]]) {
    for (const acompanha of [true, false]) {
      const ent = dA ? L0.attention + dA : L0.drowsy + dD;
      const sai = dA ? L0.attentionExit + dA : L0.drowsyExit + dD;
      const estado = dA ? "ATENÇÃO" : "SONOLENTO";
      const nomeGrupo = `limiar de ${estado} (${dA ? L0.attention : L0.drowsy})`;
      const junto = dA ? `piso de ATENÇÃO depois do forçado em ${S0.postForced.floor + (acompanha ? dA : 0)}` : `nível exibido no forçado em ${S0.forcedFloor + (acompanha ? dD : 0)}`;
      specs.push({ id: `${dA ? "atencao" : "sonolento"}_${ent}${acompanha ? "" : "_so_limiar"}`, tipo: "uma_mudanca", grupo: dA ? "limiar_atencao" : "limiar_sonolento", nomeGrupo, curto: `limiar de ${estado}`, valor: `limiar em ${ent}`, descricao: `${estado} em ${ent} (sai em ${sai})${acompanha ? "" : ", só o limiar"}; ${junto}`, f: { ...um }, dA, dD, acompanha });
    }
  }
  const rng = mulberry32(SEMENTE);
  for (let s = 1; s <= N_SORTEIOS; s++) {
    const f = {};
    for (const p of PARAMS) f[p.key] = Math.round((0.7 + 0.6 * rng()) * 10000) / 10000;
    const dA = Math.floor(rng() * 21) - 10;
    const dD = Math.floor(rng() * 21) - 10;
    specs.push({ id: `sorteio_${String(s).padStart(3, "0")}`, tipo: "sorteio", grupo: "", nomeGrupo: "", descricao: "", f, dA, dD, acompanha: true });
  }
  return specs;
}

// ---------------------------------------------------------------------------
// Leitura (copiado de validar_motor_dmd.mjs, sem os campos que o monitor não usa)
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
      t: num("t_ms"), face, earL: num("ear_l"), earR: num("ear_r"), mar: num("mar"), jaw: num("jaw_open"), bl: num("blink_l"), br: num("blink_r"),
      ldl: num("look_down_l"), ldr: num("look_down_r"), sql: num("squint_l"), sqr: num("squint_r"), pose: face ? E.poseFromMatrix(m) : null,
    };
  }
  return rows;
}
/** Intervalos (quadros) da anotação "eyes_state/close" do OpenLABEL, na ordem de validar_motor_dmd.mjs. */
function readCloseIntervals(file) {
  const d = JSON.parse(fs.readFileSync(file, "utf8")).openlabel;
  const out = [];
  for (const a of Object.values(d.actions || {})) if (a.type === "eyes_state/close") for (const iv of a.frame_intervals || []) out.push([Number(iv.frame_start), Number(iv.frame_end)]);
  return out.sort((x, y) => x[0] - y[0]);
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
const fps = (rows) => 1000 / ((rows[rows.length - 1].t - rows[0].t) / (rows.length - 1));
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
  return { tpDet: ud.size, tpAnn: ua.size };
}
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
/** Postos (empates com o posto médio) e correlação de Spearman. */
function postos(a) {
  const ord = a.map((v, i) => [v, i]).sort((x, y) => x[0] - y[0]);
  const r = new Array(a.length);
  for (let i = 0; i < ord.length;) {
    let j = i;
    while (j + 1 < ord.length && ord[j + 1][0] === ord[i][0]) j++;
    for (let k = i; k <= j; k++) r[ord[k][1]] = (i + j) / 2 + 1;
    i = j + 1;
  }
  return r;
}
const spearman = (x, y) => pearson(postos(x), postos(y));

/** Rótulo da sessão como em validar_motor_dmd.mjs (gA_5_s1); com dia e hora quando dois vídeos têm o mesmo rótulo. */
function rotuloSessao(id, ids) {
  const curto = id.split("_").slice(0, 3).join("_");
  if (ids.filter((x) => x.split("_").slice(0, 3).join("_") === curto).length < 2) return curto;
  const m = id.match(/_(\d{4})-(\d{2})-(\d{2})T(\d{2});(\d{2})/);
  return m ? `${curto} de ${m[3]}/${m[2]} às ${m[4]}h${m[5]}` : id;
}

/** Os 33 vídeos (s1 e s5): tempo, rosto, quadros de entrada do monitor e, em s5, os fechamentos anotados (ms). */
function carregarVideos() {
  const files = fs.readdirSync(SIG).filter((f) => f.endsWith(".csv.gz")).sort();
  const ids = files.map((f) => f.replace(".csv.gz", ""));
  const out = [];
  for (const file of files) {
    const id = file.replace(".csv.gz", "");
    const sessao = (id.match(/_(s\d)_/) || [])[1];
    if (sessao !== "s1" && sessao !== "s5") continue;
    const meta = JSON.parse(fs.readFileSync(path.join(SIG, id + ".json"), "utf8"));
    const rows = readSignals(path.join(SIG, file));
    const v = { id, rotulo: rotuloSessao(id, ids), sessao, n: rows.length, t: Float64Array.from(rows, (r) => r.t), face: Uint8Array.from(rows, (r) => (r.face ? 1 : 0)), frames: rows.map(frameInput) };
    if (sessao === "s5") {
      const hz = meta.fps || fps(rows);
      const f2ms = (fr) => (fr * 1000) / hz;
      v.fechAnot = readCloseIntervals(path.join(ROOT, meta.anotacao)).map(([a, b]) => [f2ms(a), f2ms(b + 1)]);
    }
    out.push(v);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Um vídeo com uma configuração: o monitor completo do app, quadro a quadro
// ---------------------------------------------------------------------------
function rodarVideo(v, cfg) {
  const { n, t, face, frames } = v;
  const mon = new E.DrowsinessMonitor(cfg);
  mon.startCalibration(t[0]);
  const closed = v.sessao === "s5" ? new Uint8Array(n) : null;
  let calT = null, ms = 0, verm = 0, amar = 0, vForc = 0, vPos = 0, vNivel = 0, epForcado = false, piscMax = 0;
  for (let i = 0; i < n; i++) {
    const res = mon.update(t[i], frames[i]);
    if (calT === null && res.events.some((e) => e.type === "calibrado")) calT = t[i];
    if (closed && res.eyes) closed[i] = res.eyes.closed ? 1 : 0;
    if (res.score.all) piscMax = Math.max(piscMax, res.score.all.find((c) => c.key === "piscadas").points); // diagnóstico
    const st = res.score.state;
    const forcado = !!res.score.forced;
    // vermelho em 3 partes: forçado (olhos fechados >= 1 s ou cabeça caída), nível >= SONOLENTO logo depois de um
    // forçado no MESMO episódio de vermelho, e nível >= SONOLENTO sem forçado antes no episódio
    epForcado = st === 2 && (epForcado || forcado);
    // mesma contabilidade de validar_motor_dmd.mjs: quadro i (i >= 1) depois da calibração vale t[i] - t[i-1]
    if (i > 0 && calT !== null && t[i] > calT) {
      const d = t[i] - t[i - 1];
      ms += d;
      if (st === 2) {
        verm += d;
        if (forcado) vForc += d;
        else if (epForcado) vPos += d;
        else vNivel += d;
      }
      if (st === 1) amar += d;
    }
  }
  const out = { ms, verm, amar, vForc, vPos, vNivel, piscMax, alarmes: mon.stats.alarmes, avisos: mon.stats.avisos, eventos: { ...mon.stats.eventos }, anot1s: 0, microRev: 0 };
  if (closed) {
    // fechamentos anotados >= 1 s que dispararam o vermelho imediato (como em validar_motor_dmd.mjs)
    const ct = new E.ClosureTracker(cfg.eyes);
    const ep = [];
    for (let i = 0; i < n; i++) {
      if (calT === null || t[i] <= calT) continue;
      const o = ct.update(t[i], !!closed[i], face[i] === 1);
      if (o.ended) ep.push([o.ended.start, o.ended.end]);
    }
    const A1 = v.fechAnot.filter(([a, b]) => calT !== null && a > calT && b - a >= 1000);
    const D = ep.filter(([a, b]) => b - a >= cfg.eyes.microsleepMs);
    out.anot1s = A1.length;
    out.microRev = matchEvents(D, A1).tpAnn;
  }
  return out;
}

/** Uma configuração nos 33 vídeos -> métricas (somas por tipo de sessão, como no "Ajuste final"). */
function avaliar(spec, videos) {
  const cfg = montarCfg(spec);
  const z = () => ({ sessoes: 0, min: 0, min_verm: 0, min_amar: 0, min_forc: 0, min_pos: 0, min_nivel: 0, alarmes: 0, avisos: 0, sem_alarme: 0, eventos: {}, anot1s: 0, microRev: 0, com_vermelho: [], piscMax: 0, piscSessoes: 0 });
  const a = { s1: z(), s5: z() };
  for (const v of videos) {
    const r = rodarVideo(v, cfg);
    const o = a[v.sessao];
    o.sessoes += 1;
    if (r.verm > 0) o.com_vermelho.push(v.rotulo);
    o.piscMax = Math.max(o.piscMax, r.piscMax);
    if (r.piscMax > 0) o.piscSessoes += 1;
    o.min += r.ms / 60000;
    o.min_verm += r.verm / 60000;
    o.min_amar += r.amar / 60000;
    o.min_forc += r.vForc / 60000;
    o.min_pos += r.vPos / 60000;
    o.min_nivel += r.vNivel / 60000;
    o.alarmes += r.alarmes;
    o.avisos += r.avisos;
    if (r.alarmes === 0) o.sem_alarme += 1;
    for (const [k, c] of Object.entries(r.eventos)) o.eventos[k] = (o.eventos[k] || 0) + c;
    o.anot1s += r.anot1s;
    o.microRev += r.microRev;
  }
  const pct = (o, k) => (100 * o[k]) / (o.min || 1);
  const { s1, s5 } = a;
  return {
    s1_minutos: s1.min,
    s1_pct_vermelho: pct(s1, "min_verm"),
    s1_pct_amarelo: pct(s1, "min_amar"),
    s1_alarmes: s1.alarmes,
    s1_alarmes_h: s1.alarmes / (s1.min / 60 || 1),
    s1_avisos_h: s1.avisos / (s1.min / 60 || 1),
    s1_sessoes: s1.sessoes,
    s1_sessoes_sem_alarme: s1.sem_alarme,
    s1_pct_vermelho_forcado: pct(s1, "min_forc"),
    s1_pct_vermelho_pos_forcado: pct(s1, "min_pos"),
    s1_pct_vermelho_so_nivel: pct(s1, "min_nivel"),
    s1_sessoes_com_vermelho: s1.com_vermelho.join("; "),
    s5_minutos: s5.min,
    s5_pct_vermelho: pct(s5, "min_verm"),
    s5_pct_amarelo: pct(s5, "min_amar"),
    s5_alarmes: s5.alarmes,
    s5_alarmes_h: s5.alarmes / (s5.min / 60 || 1),
    s5_sessoes: s5.sessoes,
    s5_sessoes_com_alarme: s5.sessoes - s5.sem_alarme,
    s5_pct_vermelho_forcado: pct(s5, "min_forc"),
    s5_pct_vermelho_pos_forcado: pct(s5, "min_pos"),
    s5_pct_vermelho_so_nivel: pct(s5, "min_nivel"),
    s5_microssonos_sobre_fech_anot_1s: `${s5.microRev}/${s5.anot1s}`,
    s5_revoc_fech_anot_1s: s5.microRev / (s5.anot1s || 1),
    piscadas_pts_max: Math.max(s1.piscMax, s5.piscMax),
    piscadas_sessoes_com_pontos: s1.piscSessoes + s5.piscSessoes,
    // assinatura do que NÃO pode mudar com os pesos: eventos detectados e vermelho forçado (conferida no fim)
    fixo: JSON.stringify({ s1: s1.eventos, s5: s5.eventos, s1_forc: s1.min_forc, s5_forc: s5.min_forc, micro: [s5.microRev, s5.anot1s] }),
  };
}

// ---------------------------------------------------------------------------
// Execução em paralelo: cada thread carrega os vídeos uma vez e avalia as configurações que receber
// ---------------------------------------------------------------------------
async function avaliarEmParalelo(specs, idxs, nW, onDone) {
  const results = new Map();
  await new Promise((resolve, reject) => {
    let next = 0;
    let done = 0;
    if (!idxs.length) return resolve();
    for (let w = 0; w < Math.min(nW, idxs.length); w++) {
      const wk = new Worker(SCRIPT);
      const send = () => (next < idxs.length ? wk.postMessage({ idx: idxs[next], spec: specs[idxs[next++]] }) : wk.postMessage(null));
      wk.on("message", (m) => {
        if (m.pronto) return send();
        results.set(m.idx, m.out);
        onDone(++done, idxs.length);
        if (done === idxs.length) resolve();
        send();
      });
      wk.on("error", reject);
    }
  });
  return results;
}

/** CSV simples (campos entre aspas com vírgulas), para ler a referência. */
function lerCSV(file) {
  const campos = (l) => {
    const out = [];
    let cur = "", q = false;
    for (let i = 0; i < l.length; i++) {
      const ch = l[i];
      if (q) {
        if (ch === '"' && l[i + 1] === '"') { cur += '"'; i++; } else if (ch === '"') q = false;
        else cur += ch;
      } else if (ch === '"') q = true;
      else if (ch === ",") { out.push(cur); cur = ""; } else cur += ch;
    }
    out.push(cur);
    return out;
  };
  const linhas = fs.readFileSync(file, "utf8").trim().split("\n");
  const head = campos(linhas[0]);
  return linhas.slice(1).map((l) => { const c = campos(l); return Object.fromEntries(head.map((h, i) => [h, c[i]])); });
}
/** Mesmo formato numérico de E.toCSV (inteiro sem casas; senão 4 casas): a conferência compara o texto. */
const comoCSV = (v) => (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(4)) : String(v));

async function principal() {
  const t0 = performance.now();
  const args = process.argv.slice(2);
  const arg = (nome, def) => { const a = args.find((x) => x.startsWith(`--${nome}=`)); return a ? a.slice(nome.length + 3) : def; };
  const SAIDA = path.resolve(arg("saida", path.join(ROOT, "results")));
  const nW = Math.max(1, Math.min(MAX_WORKERS, Number(arg("workers", MAX_WORKERS)) || 1, os.availableParallelism?.() ?? os.cpus().length));
  const _d = new Date();
  const _p = (v) => String(v).padStart(2, "0");
  const stamp = `${_d.getFullYear()}${_p(_d.getMonth() + 1)}${_p(_d.getDate())}-${_p(_d.getHours())}${_p(_d.getMinutes())}${_p(_d.getSeconds())}`;
  const logPath = path.join(SAIDA, "logs", `app_sensibilidade_pesos_${stamp}.log`);
  const rel = (p) => path.relative(ROOT, p);
  const logLines = [];
  const log = (...a) => { const s = a.join(" "); logLines.push(s); console.log(s); };
  const gravarLog = () => { fs.mkdirSync(path.dirname(logPath), { recursive: true }); fs.writeFileSync(logPath, logLines.join("\n") + "\n"); };
  const comando = ["node", rel(SCRIPT), ...args.map((a) => (a.startsWith("--saida=") ? `--saida=${rel(SAIDA)}` : a))].join(" ");

  log(`# Sensibilidade dos pesos do nível do app no DMD, ${_d.toISOString()}`);
  log(`# comando (na raiz do repositório): ${comando}`);
  log(`# node ${process.version}; ${nW} thread(s) em paralelo; ${os.cpus().length} núcleos`);
  const md5Inicio = md5Motor();
  const _m = fs.statSync(ENGINE).mtime;
  log(`# motor: ${rel(ENGINE)} (md5 ${md5Inicio}, modificado em ${_m.getFullYear()}-${_p(_m.getMonth() + 1)}-${_p(_m.getDate())} ${_p(_m.getHours())}:${_p(_m.getMinutes())}:${_p(_m.getSeconds())})`);
  const specs = montarConfiguracoes();
  log(`# configurações: ${specs.length} (1 final, ${specs.filter((s) => s.tipo === "uma_mudanca").length} com uma mudança, ${N_SORTEIOS} sorteadas com semente ${SEMENTE})`);
  log(`# CFG.score final = ${JSON.stringify(E.CFG.score)}; CFG.levels final = ${JSON.stringify(E.CFG.levels)}`);

  // ---- 1) configuração final, primeiro (para parar logo se não reproduzir) ----
  const videos = carregarVideos();
  log(`# sinais: ${rel(SIG)} (${videos.length} vídeos: ${videos.filter((v) => v.sessao === "s1").length} s1, ${videos.filter((v) => v.sessao === "s5").length} s5), carregados em ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  const tF = performance.now();
  const final = avaliar(specs[0], videos);
  log(`final avaliada em ${((performance.now() - tF) / 1000).toFixed(1)} s: s1 ${final.s1_pct_vermelho.toFixed(4)} % vermelho, ${final.s1_alarmes} alarmes; s5 ${final.s5_pct_vermelho.toFixed(4)} % vermelho, ${final.s5_alarmes} alarmes`);
  const ref = lerCSV(REF_CSV).find((r) => r.alternativa === "final");
  const ev = JSON.parse(final.fixo);
  const pares = [
    ["s1_min", final.s1_minutos], ["s1_pct_vermelho", final.s1_pct_vermelho], ["s1_pct_amarelo", final.s1_pct_amarelo], ["s1_alarmes", final.s1_alarmes],
    ["s1_alarmes_h", final.s1_alarmes_h], ["s1_avisos_h", final.s1_avisos_h], ["s1_microssonos", ev.s1.microssono], ["s1_fech05_1s", ev.s1.fechamento],
    ["s1_bocejos", ev.s1.bocejo], ["s1_quedas_cabeceios", ev.s1.queda + ev.s1.cabeceio],
    ["s5_min", final.s5_minutos], ["s5_pct_vermelho", final.s5_pct_vermelho], ["s5_pct_amarelo", final.s5_pct_amarelo], ["s5_alarmes", final.s5_alarmes],
    ["s5_microssonos", ev.s5.microssono], ["s5_bocejos", ev.s5.bocejo],
    ["s5_revoc_fech1", final.s5_revoc_fech_anot_1s], ["s5_micro_revoc_anot1s", final.s5_microssonos_sobre_fech_anot_1s],
  ];
  const diverge = pares.filter(([k, v]) => !ref || ref[k] !== comoCSV(v));
  log(`\n## Conferência com a linha "final" de ${rel(REF_CSV)}`);
  for (const [k, v] of pares) log(`  ${k}: referência ${ref ? ref[k] : "(sem linha final)"}, agora ${comoCSV(v)}${ref && ref[k] === comoCSV(v) ? "" : "  <-- DIFERENTE"}`);
  if (diverge.length) {
    log(`\nPAROU: a configuração sem mudança NÃO reproduz ${diverge.length} número(s) da validação existente (${diverge.map(([k]) => k).join(", ")}).`);
    log("O .csv e o .md não foram gravados. Causas possíveis: app/pwa/engine.js, os sinais em data/signals/app_tasks/dmd ou a anotação mudaram depois de 27/09/2026");
    log("(rode de novo tools/app_validacao/validar_motor_dmd.mjs para atualizar a referência e compare as duas versões antes de seguir).");
    log(`# tempo de execução: ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    gravarLog();
    process.exitCode = 1;
    return;
  }
  log(`OK: os ${pares.length} números conferidos são idênticos (mesma precisão do CSV de referência).`);

  // ---- 2) as outras configurações ----
  const resto = specs.map((_, i) => i).slice(1);
  const tR = performance.now();
  let results;
  const progresso = (d, tot) => { if (d % 25 === 0 || d === tot) log(`  ${d}/${tot} configurações (${((performance.now() - tR) / 1000).toFixed(0)} s)`); };
  if (nW === 1) {
    results = new Map();
    resto.forEach((i, k) => { results.set(i, avaliar(specs[i], videos)); progresso(k + 1, resto.length); });
  } else {
    results = await avaliarEmParalelo(specs, resto, nW, progresso);
  }
  results.set(0, final);
  const linhas = specs.map((spec, i) => ({ spec, m: results.get(i) }));

  // ---- 3) o que não pode mudar não mudou ----
  if (md5Motor() !== md5Inicio) {
    log(`\nPAROU: ${rel(ENGINE)} mudou durante a execução (md5 ${md5Inicio} no início, ${md5Motor()} agora). Nada foi gravado além deste log; rode de novo.`);
    log(`# tempo de execução: ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    gravarLog();
    process.exitCode = 1;
    return;
  }
  log(`\nmotor sem mudança durante a execução (md5 ${md5Inicio})`);
  const difFixo = linhas.filter((l) => l.m.fixo !== final.fixo).map((l) => l.spec.id);
  const F = JSON.parse(final.fixo);
  log(`\n## Definições fixas: eventos detectados e vermelho forçado iguais aos da final em ${linhas.length - difFixo.length} de ${linhas.length} configurações${difFixo.length ? ` (DIFERENTES: ${difFixo.join(", ")})` : ""}`);
  log(`  s1: ${JSON.stringify(F.s1)}; s5: ${JSON.stringify(F.s5)}`);
  if (difFixo.length) throw new Error("eventos ou vermelho forçado mudaram com os pesos: a separação entre definições e pesos não vale");

  // ---- 4) CSV ----
  const csvRows = linhas.map(({ spec, m }) => {
    const cfg = montarCfg(spec);
    const { fixo, ...met } = m;
    return {
      id: spec.id, tipo: spec.tipo, grupo: spec.grupo, descricao: spec.descricao,
      ...Object.fromEntries(PARAMS.map((p) => [`f_${p.key}`, spec.f[p.key]])),
      d_atencao: spec.dA, d_sonolento: spec.dD, limiares_acompanham: spec.acompanha ? "sim" : "nao",
      atencao: cfg.levels.attention, atencao_saida: cfg.levels.attentionExit, sonolento: cfg.levels.drowsy, sonolento_saida: cfg.levels.drowsyExit,
      piso_pos_forcado: cfg.score.postForced.floor, nivel_no_forcado: cfg.score.forcedFloor,
      ...met,
      boa: met.s1_pct_vermelho <= BOM_S1_MAX && met.s5_pct_vermelho > BOM_S5_MIN ? "sim" : "nao",
    };
  });
  fs.mkdirSync(SAIDA, { recursive: true });
  const csvPath = path.join(SAIDA, "app_sensibilidade_pesos.csv");
  fs.writeFileSync(csvPath, E.toCSV(csvRows) + "\n");
  log(`\n  -> ${rel(csvPath)} (${csvRows.length} linhas)`);

  // ---- 5) relatório ----
  const segundos = (performance.now() - t0) / 1000;
  const md = relatorio({ linhas, final, F, csvPath, logPath, rel, segundos, nW, pares, rotulos: new Map(videos.map((v) => [v.id, v.rotulo])) });
  const mdTxt = md.join("\n") + "\n";
  const tracos = new RegExp(`[${String.fromCharCode(0x2013, 0x2014)}]`); // meia-risca e travessão
  if (tracos.test(mdTxt)) throw new Error("o relatório tem travessão ou meia-risca");
  const mdPath = path.join(SAIDA, "app_sensibilidade_pesos.md");
  fs.writeFileSync(mdPath, mdTxt);
  log(`  -> ${rel(mdPath)}`);
  log("\n" + mdTxt);
  log(`# tempo de execução: ${segundos.toFixed(1)} s (${nW} thread(s))`);
  gravarLog();
}

// ---------------------------------------------------------------------------
// Relatório em português simples (todos os números saem da execução)
// ---------------------------------------------------------------------------
function relatorio({ linhas, final, F, csvPath, logPath, rel, segundos, nW, pares, rotulos }) {
  const um = linhas.filter((l) => l.spec.tipo === "uma_mudanca");
  const sort = linhas.filter((l) => l.spec.tipo === "sorteio");
  const N = linhas.length;
  const nS = final.s1_sessoes + final.s5_sessoes;
  const pc = (v, d = 1) => `${br(v, d)} %`;
  const num = (v, d = 1) => (Number.isInteger(v) ? br(v, 0) : br(v, d));
  const pl = (n, s, p) => `${n} ${n === 1 ? s : p}`;
  const lista = (a) => (a.length <= 1 ? a.join("") : `${a.slice(0, -1).join(", ")} e ${a[a.length - 1]}`);
  const sessoes = (txt) => txt.split("; ").filter(Boolean);
  const evTxt = (e) => `${pl(e.microssono, "microssono", "microssonos")}, ${pl(e.fechamento, "fechamento", "fechamentos")} de 0,5 a 1 s, ${pl(e.bocejo, "bocejo", "bocejos")}, ${pl(e.queda, "queda", "quedas")} e ${pl(e.cabeceio, "cabeceio", "cabeceios")} da cabeça`;
  const nivel = (m, s) => m[`${s}_pct_vermelho_pos_forcado`] + m[`${s}_pct_vermelho_so_nivel`];
  const md = [];

  md.push("# Sensibilidade dos pesos do nível do app (DMD)", "");
  md.push(`Gerado em ${new Date().toISOString()} por \`${rel(SCRIPT)}\` em ${br(segundos, 0)} s (${pl(nW, "thread", "threads")}; log: \`${rel(logPath)}\`). ` +
    `Tabela completa, uma linha por configuração e com todos os fatores: \`${rel(csvPath)}\`.`, "");
  md.push("**Pergunta.** Os limiares de detecção de cada sinal (olho fechado, bocejo, cabeça) foram escolhidos comparando variantes contra a anotação do DMD (`results/app_validacao_dmd.md`). " +
    "Já os números do nível de 0 a 100 (pontos por evento, tetos, decaimentos e os limiares 40 e 70) são regras de projeto. Aqui se mede quanto o comportamento do app muda quando esses números mudam.", "");
  md.push(`**Como.** Os mesmos sinais e a mesma calibração da validação offline (\`tools/app_validacao/validar_motor_dmd.mjs\`): o monitor completo do app (\`app/pwa/engine.js\`, importado sem cópia) roda quadro a quadro nos ${nS} vídeos do DMD, ` +
    `${final.s1_sessoes} de direção real com motoristas acordados (s1, ${br(final.s1_minutos, 1)} min depois da calibração; todo vermelho ali é falso alarme) e ${final.s5_sessoes} de sonolência encenada com o carro parado (s5, ${br(final.s5_minutos, 1)} min). ` +
    `Só mudam os números do nível (\`CFG.score\` e \`CFG.levels\`), em ${N} configurações. Ficam fixos as definições dos sinais (olho fechado, PERCLOS e as suas rampas de 7,5 % e 15 %, bocejo, cabeça), o microssono de 1 s e a cabeça caída com olhos fechados, que são os dois vermelhos forçados, e o alarme. ` +
    "O LSTM não roda offline (como na validação existente), então o peso dele não tem efeito aqui.", "");
  md.push(`**Conferência.** A configuração sem mudança reproduz a linha \`final\` de \`results/app_validacao_dmd_ajuste_final.csv\` nos ${pares.length} números comparados, na precisão do arquivo (minutos, % do tempo em vermelho e em amarelo, alarmes, avisos por hora, eventos detectados e fechamentos anotados com vermelho imediato): ` +
    `s1 com ${pc(final.s1_pct_vermelho)} do tempo em vermelho (${br(final.s1_pct_vermelho, 4)}), ${pc(final.s1_pct_amarelo)} em amarelo e ${final.s1_alarmes} alarmes (${br(final.s1_alarmes_h, 2)} por hora); ` +
    `s5 com ${pc(final.s5_pct_vermelho)} em vermelho (${br(final.s5_pct_vermelho, 4)}) e ${final.s5_alarmes} alarmes.`, "");

  // ---- de onde vem o vermelho ----
  const s1verm = sessoes(final.s1_sessoes_com_vermelho);
  let topoSessoes = "";
  try {
    const ses = lerCSV(path.join(ROOT, "results/app_validacao_dmd_sessoes.csv")).filter((r) => r.sessao === "s1").sort((a, b) => Number(b.frac_sonolento) - Number(a.frac_sonolento));
    if (s1verm.length === 1 && ses.length && rotulos.get(ses[0].id) === s1verm[0]) topoSessoes = ", a mesma que lidera as sessões s1 com mais tempo em SONOLENTO em `results/app_validacao_dmd.md`";
  } catch { /* sem o arquivo da validação: só não cita */ }
  const fracPos = (s) => (100 * final[`${s}_pct_vermelho_pos_forcado`]) / (nivel(final, s) || 1);
  const fracComForc = (s) => (100 * (final[`${s}_pct_vermelho_forcado`] + final[`${s}_pct_vermelho_pos_forcado`])) / (final[`${s}_pct_vermelho`] || 1);
  md.push("## De onde vem o vermelho", "");
  md.push("Configuração final. \"Pelo nível\" quer dizer o nível de 0 a 100 acima do limiar de SONOLENTO (entra em 70; quando entrou pelo nível, sai abaixo de 60).", "");
  md.push("| Parte do tempo | s1, direção real | s5, sonolência encenada |", "|---|---:|---:|");
  md.push(`| Vermelho forçado: olhos fechados há 1 s ou mais, ou cabeça caída com olhos fechados | ${pc(final.s1_pct_vermelho_forcado, 2)} | ${pc(final.s5_pct_vermelho_forcado, 2)} |`);
  md.push(`| Vermelho pelo nível, logo depois de um vermelho forçado (no mesmo episódio de vermelho) | ${pc(final.s1_pct_vermelho_pos_forcado, 2)} | ${pc(final.s5_pct_vermelho_pos_forcado, 2)} |`);
  md.push(`| Vermelho pelo nível, sem vermelho forçado antes no mesmo episódio | ${pc(final.s1_pct_vermelho_so_nivel, 2)} | ${pc(final.s5_pct_vermelho_so_nivel, 2)} |`);
  md.push(`| **Total em vermelho** | **${pc(final.s1_pct_vermelho, 2)}** | **${pc(final.s5_pct_vermelho, 2)}** |`, "");
  md.push(`Em s5, só ${pc(final.s5_pct_vermelho_forcado)} do tempo é vermelho forçado, porque ele dura apenas enquanto os olhos estão fechados (ou a cabeça caída). Os outros ${pc(nivel(final, "s5"))} vêm do nível, ${pc(fracPos("s5"), 0)} deles logo depois de um forçado: os pontos do próprio microssono, dos fechamentos e do PERCLOS seguram o vermelho depois que os olhos abrem. ` +
    `Em s1, ${pc(final.s1_pct_vermelho_forcado, 2)} é forçado e ${pc(nivel(final, "s1"), 2)} vem do nível; ${pc(fracComForc("s1"), 1)} desse vermelho está em episódios que começaram com um vermelho forçado, todos em ${pl(s1verm.length, "sessão", "sessões")} de ${final.s1_sessoes} (${lista(s1verm)})${topoSessoes}.`, "");
  md.push(`O vermelho forçado e todos os eventos detectados saíram idênticos nas ${N} configurações, porque dependem só das definições fixas: em s1, ${evTxt(F.s1)}; em s5, ${evTxt(F.s5)}. ` +
    `Pelo mesmo motivo, a revocação de eventos anotados que a validação existente mede no app também não muda com os pesos: ${final.s5_microssonos_sobre_fech_anot_1s.replace("/", " de ")} fechamentos anotados de 1 s ou mais em s5 deram vermelho imediato em todas as configurações.`, "");

  // ---- uma mudança por vez ----
  md.push("## Uma mudança por vez", "");
  md.push("Cada linha é a configuração final com UMA mudança. Grupos de pontos: os pontos e o teto do próprio grupo vezes o fator; o teto de fechamentos e microssonos somados, que os dois grupos dividem, tem linha própria; no microssono mudam só os pontos que ele soma no nível depois que os olhos reabrem (o vermelho imediato não muda). " +
    "Decaimentos: o tempo até os pontos de um evento sumirem, vezes 0,5 ou 2; a descida máxima do nível é uma velocidade (vezes 0,5 = desce mais devagar). " +
    "Limiares: a saída fica na mesma distância (ATENÇÃO sai 5 abaixo da entrada; SONOLENTO, 10 abaixo). Dois números são definidos a partir dos limiares e acompanham: o piso de ATENÇÃO por 1 min depois de um vermelho forçado (40) e o nível exibido durante o forçado (75, de onde o nível desce depois que os olhos abrem); o portão do LSTM (40) também acompanha, sem efeito aqui. Nas linhas \"só o limiar\" eles ficam como estão. " +
    "Alarme = entrada no vermelho; menos alarmes pode querer dizer vermelho mais longo e contínuo, por isso a medida principal é o % do tempo em vermelho.", "");
  md.push("| Mudança | s1 % vermelho | s1 % amarelo | s1 alarmes por hora | s1 sessões sem alarme | s5 % vermelho | s5 alarmes |", "|---|---:|---:|---:|---:|---:|---:|");
  for (const l of [linhas[0], ...um]) {
    const d = l.spec.tipo === "final" ? `**${l.spec.descricao}**` : l.spec.descricao;
    md.push(`| ${d} | ${br(l.m.s1_pct_vermelho, 1)} | ${br(l.m.s1_pct_amarelo, 1)} | ${br(l.m.s1_alarmes_h, 2)} | ${l.m.s1_sessoes_sem_alarme} de ${l.m.s1_sessoes} | ${br(l.m.s5_pct_vermelho, 1)} | ${l.m.s5_alarmes} |`);
  }
  md.push("");
  // efeito de cada item (faixas só com as linhas em que os números ligados aos limiares acompanham)
  const chave = (m) => ["s1_pct_vermelho", "s1_pct_amarelo", "s1_alarmes", "s1_avisos_h", "s1_sessoes_sem_alarme", "s5_pct_vermelho", "s5_pct_amarelo", "s5_alarmes"].map((c) => m[c]).join("|");
  const efeito = [...new Set(um.map((l) => l.spec.grupo))].map((g) => {
    const todas = um.filter((l) => l.spec.grupo === g);
    const ls = todas.filter((l) => l.spec.acompanha);
    const r = { g, nome: ls[0].spec.nomeGrupo, curto: ls[0].spec.curto };
    for (const s of ["s1", "s5"]) {
      const k = `${s}_pct_vermelho`;
      r[`${s}lo`] = ls.reduce((a, b) => (b.m[k] < a.m[k] ? b : a));
      r[`${s}hi`] = ls.reduce((a, b) => (b.m[k] > a.m[k] ? b : a));
      r[`${s}amp`] = Math.max(...ls.map((l) => Math.abs(l.m[k] - final[k])));
    }
    r.nada = todas.every((l) => chave(l.m) === chave(final)); // tempo em cada estado e alarmes exatamente iguais
    r.nadaS1 = r.s1amp < 0.05;
    return r;
  });
  const fx = (e, s) => `de ${pc(e[`${s}lo`].m[`${s}_pct_vermelho`])} com ${e[`${s}lo`].spec.valor} a ${pc(e[`${s}hi`].m[`${s}_pct_vermelho`])} com ${e[`${s}hi`].spec.valor}`;
  const top = (s) => [...efeito].sort((a, b) => b[`${s}amp`] - a[`${s}amp`]).slice(0, 3);
  const t1 = top("s1"), t5 = top("s5");
  const nada = efeito.filter((e) => e.nada);
  const mexemS1 = efeito.filter((e) => !e.nadaS1);
  md.push("**O que mais mexe, uma mudança por vez** (do menor ao maior vermelho, com o valor que deu cada ponta):", "");
  md.push(`- no falso alarme (vermelho de s1; ${pc(final.s1_pct_vermelho)} na final): ${t1.map((e) => `${e.curto}, ${fx(e, "s1")}`).join("; ")};`);
  md.push(`- no vermelho de s5 (${pc(final.s5_pct_vermelho)} na final): ${t5.map((e) => `${e.curto}, ${fx(e, "s5")}`).join("; ")}.`, "");
  md.push(`**No falso alarme, só ${mexemS1.length} dos ${efeito.length} itens testados mexeram mais que 0,05 ponto percentual:** ${lista(mexemS1.map((e) => e.curto))}. ` +
    `Em s1 o app detectou ${pl(F.s1.bocejo, "bocejo", "bocejos")} e ${pl(F.s1.queda + F.s1.cabeceio, "queda ou cabeceio", "quedas ou cabeceios")} da cabeça, por isso os pesos desses dois sinais quase não têm como mexer no falso alarme medido aqui. ` +
    `Não mudaram nada (tempo em cada estado e alarmes exatamente iguais aos da final): ${lista(nada.map((e) => e.curto))}; o LSTM não roda offline, e as piscadas mais longas somaram no máximo ${br(final.piscadas_pts_max, 1)} pontos, em ${final.piscadas_sessoes_com_pontos} das ${nS} sessões.`, "");

  // ---- sorteio ----
  const MET = [
    ["s1_pct_vermelho", "s1: % do tempo em vermelho", 1],
    ["s1_pct_amarelo", "s1: % do tempo em amarelo", 1],
    ["s1_alarmes_h", "s1: alarmes por hora", 2],
    ["s1_sessoes_sem_alarme", `s1: sessões sem alarme (de ${final.s1_sessoes})`, 1],
    ["s5_pct_vermelho", "s5: % do tempo em vermelho", 1],
    ["s5_nivel", "s5: % do tempo em vermelho pelo nível", 1, (m) => nivel(m, "s5")],
    ["s5_alarmes", "s5: alarmes", 1],
  ];
  md.push(`## Todos juntos, ao acaso (${sort.length} configurações)`, "");
  md.push(`Cada configuração sorteada (gerador mulberry32, semente ${SEMENTE}) multiplica cada um dos ${PARAMS.length} pesos, tetos e decaimentos por um fator próprio, uniforme de 0,7 a 1,3, e desloca os limiares 40 e 70 por um inteiro uniforme de -10 a +10 ` +
    `(ATENÇÃO de ${L0.attention - 10} a ${L0.attention + 10} e SONOLENTO de ${L0.drowsy - 10} a ${L0.drowsy + 10}, com as saídas e os números que acompanham na mesma distância). Percentis pelo posto mais próximo: 5 % das configurações ficam abaixo do percentil 5 e 5 % acima do percentil 95.`, "");
  md.push("| Métrica | Final | Mediana | Percentil 5 | Percentil 95 | Menor | Maior |", "|---|---:|---:|---:|---:|---:|---:|");
  for (const [k, nome, d, f] of MET) {
    const v = sort.map((l) => (f ? f(l.m) : l.m[k]));
    md.push(`| ${nome} | ${num(f ? f(final) : final[k], d)} | ${num(E.median(v), d)} | ${num(E.percentile(v, 5), d)} | ${num(E.percentile(v, 95), d)} | ${num(Math.min(...v), d)} | ${num(Math.max(...v), d)} |`);
  }
  const ok1 = sort.filter((l) => l.m.s1_pct_vermelho <= BOM_S1_MAX), ok5 = sort.filter((l) => l.m.s5_pct_vermelho > BOM_S5_MIN);
  const boas = sort.filter((l) => l.m.s1_pct_vermelho <= BOM_S1_MAX && l.m.s5_pct_vermelho > BOM_S5_MIN);
  const ruins = sort.filter((l) => l.m.s1_pct_vermelho > BOM_S1_MAX);
  const dRuins = ruins.map((l) => L0.drowsy + l.spec.dD);
  md.push("", `- s1 em vermelho até ${BOM_S1_MAX} % do tempo: ${ok1.length} de ${sort.length} (${pc((100 * ok1.length) / sort.length, 0)});`,
    `- s5 em vermelho acima de ${BOM_S5_MIN} % do tempo: ${ok5.length} de ${sort.length} (${pc((100 * ok5.length) / sort.length, 0)});`,
    `- **as duas coisas juntas: ${boas.length} de ${sort.length} (${pc((100 * boas.length) / sort.length, 0)})**.`, "");
  if (ruins.length) {
    const cT = ruins.map((l) => l.spec.f.teto_fechamentos);
    const dTodas = sort.map((l) => L0.drowsy + l.spec.dD);
    const baixo = Math.max(...dRuins);
    const nBaixo = sort.filter((l) => L0.drowsy + l.spec.dD <= baixo).length;
    md.push(`As ${ruins.length} que passaram de ${BOM_S1_MAX} % em s1 (até ${pc(Math.max(...ruins.map((l) => l.m.s1_pct_vermelho)))}) têm o limiar de SONOLENTO entre ${Math.min(...dRuins)} e ${baixo} ` +
      `(${ruins.length} das ${nBaixo} sorteadas com o limiar até ${baixo}; o sorteio cobriu de ${Math.min(...dTodas)} a ${Math.max(...dTodas)}) e o teto de fechamentos e microssonos vezes ${br(Math.min(...cT), 2)} a ${br(Math.max(...cT), 2)}.`, "");
  }
  const cont = new Map();
  for (const l of sort) for (const s of sessoes(l.m.s1_sessoes_com_vermelho)) cont.set(s, (cont.get(s) || 0) + 1);
  const fora = sort.filter((l) => sessoes(l.m.s1_sessoes_com_vermelho).some((s) => !s1verm.includes(s))).length;
  md.push(`Sessões de s1 com algum vermelho, nas ${sort.length} sorteadas: ${lista([...cont].sort((a, b) => b[1] - a[1]).map(([s, c]) => `${s} em ${c}`))}; ` +
    `só ${fora} das ${sort.length} criaram vermelho numa sessão em que a final não tem nenhum.`, "");

  // ---- o que mais pesa quando tudo varia junto ----
  const alvos = [["s1_pct_vermelho", "s1 % vermelho"], ["s1_alarmes_h", "s1 alarmes por hora"], ["s5_pct_vermelho", "s5 % vermelho"]];
  const fat = [
    ...PARAMS.map((p) => ({ key: p.key, nome: `${p.nome} (${comUnidade(p.key, BASE[p.key])})`, curto: p.nome, val: (s) => s.f[p.key] })),
    { key: "d_atencao", nome: `limiar de ATENÇÃO (${L0.attention})`, curto: "limiar de ATENÇÃO", val: (s) => s.dA },
    { key: "d_sonolento", nome: `limiar de SONOLENTO (${L0.drowsy})`, curto: "limiar de SONOLENTO", val: (s) => s.dD },
  ].map((x) => ({ ...x, r: alvos.map(([k]) => spearman(sort.map((l) => x.val(l.spec)), sort.map((l) => l.m[k]))) }));
  const maxAbs = (x) => Math.max(...x.r.map(Math.abs));
  // tamanho do acaso: números que não podem ter efeito (o LSTM em tudo; o bocejo nas métricas de s1, que não tem bocejo)
  const nulos = [...fat.find((x) => x.key === "pts_lstm").r.map(Math.abs)];
  if (F.s1.bocejo === 0) for (const k of ["pts_bocejo", "teto_bocejo", "dec_bocejo"]) nulos.push(...fat.find((x) => x.key === k).r.slice(0, 2).map(Math.abs));
  const acaso = Math.max(...nulos);
  const fortes = fat.filter((x) => maxAbs(x) > acaso).sort((a, b) => maxAbs(b) - maxAbs(a));
  md.push("## O que mais pesa quando tudo varia junto", "");
  md.push(`Correlação de postos (Spearman) entre cada número sorteado e cada resultado, nas ${sort.length} sorteadas: perto de 0, o número quase não mexe; perto de +1 ou -1, mexe muito (+ = mais daquele resultado quando o número cresce; nos limiares, "cresce" = limiar mais alto). ` +
    `Para saber o tamanho do acaso, valem os números que não podem ter efeito: o peso do LSTM (não roda offline) e, nas métricas de s1, os do bocejo (s1 não tem bocejo). Eles chegaram a ${br(acaso, 2)}; a tabela mostra só os números que passaram disso em pelo menos um resultado.`, "");
  md.push(`| Número | ${alvos.map(([, n]) => n).join(" | ")} |`, `|---|${alvos.map(() => "---:").join("|")}|`);
  for (const x of fortes) md.push(`| ${x.nome} | ${x.r.map((v) => br(v, 2)).join(" | ")} |`);
  md.push("", `Os outros ${fat.length - fortes.length} números ficaram com correlação entre ${br(-acaso, 2)} e ${br(acaso, 2)} nos três resultados, o tamanho do acaso.`, "");

  // ---- parágrafo final ----
  const r1 = [...fat].sort((a, b) => Math.abs(b.r[0]) - Math.abs(a.r[0])).slice(0, 2);
  const sem = um.map((l) => l.m.s1_sessoes_sem_alarme);
  const semTxt = Math.min(...sem) === Math.max(...sem) ? `sempre ${sem[0]}` : `de ${Math.min(...sem)} a ${Math.max(...sem)}`;
  const v1 = um.map((l) => l.m.s1_pct_vermelho), v5 = um.map((l) => l.m.s5_pct_vermelho);
  const boc = efeito.find((e) => e.g === "bocejo");
  const limS = efeito.find((e) => e.g === "limiar_sonolento");
  const limNoTopo = [t1[0].g, t1[1].g].includes("limiar_sonolento");
  // número sorteado -> item da tabela de uma mudança por vez (para comparar os dois rankings)
  const item = (k) => ({ d_atencao: "limiar_atencao", d_sonolento: "limiar_sonolento", teto_fechamentos: "teto_fechamentos" })[k] || (k.startsWith("dec_") ? k : k.replace(/^(pts|teto)_/, ""));
  const mesmos = new Set(r1.map((x) => item(x.key))).size === 2 && r1.every((x) => [t1[0].g, t1[1].g].includes(item(x.key)));
  md.push("## Onde é estável e onde é sensível", "");
  md.push(`**Estável:** ${efeito.length - mexemS1.length} dos ${efeito.length} itens testados não mexem no falso alarme (vermelho em s1) nem 0,05 ponto percentual quando mudam sozinhos. ` +
    `Com uma mudança isolada, inclusive os limiares em 10 pontos, o falso alarme ficou sempre entre ${pc(Math.min(...v1))} e ${pc(Math.max(...v1))} (era ${pc(final.s1_pct_vermelho)}), o vermelho de s5 entre ${pc(Math.min(...v5))} e ${pc(Math.max(...v5))} (era ${pc(final.s5_pct_vermelho)}), e ${semTxt} das ${final.s1_sessoes} sessões de s1 ficaram sem alarme. ` +
    `Com tudo sorteado junto, ${boas.length} de ${sort.length} configurações (${pc((100 * boas.length) / sort.length, 0)}) continuaram com s1 até ${BOM_S1_MAX} % e s5 acima de ${BOM_S5_MIN} %. ` +
    `O peso do bocejo não mexe no falso alarme (s1 tem ${pl(F.s1.bocejo, "bocejo detectado", "bocejos detectados")}) e muda o vermelho de s5 ${fx(boc, "s5")}. ` +
    `**Sensível:** os dois itens que mais mexem no falso alarme, uma mudança por vez, são ${t1[0].curto}, ${fx(t1[0], "s1")}, e ${t1[1].curto}, ${fx(t1[1], "s1")}` +
    (limNoTopo ? "" : `; o limiar de 70 leva o falso alarme ${fx(limS, "s1")}`) +
    `. Com tudo junto, os que mais pesam são ${mesmos ? "os mesmos dois" : "outros"}: ${lista(r1.map((x) => `${x.curto} (correlação ${br(x.r[0], 2)})`))}` +
    (ruins.length ? `; as ${ruins.length} configurações que passaram de ${BOM_S1_MAX} % têm o limiar de SONOLENTO entre ${Math.min(...dRuins)} e ${Math.max(...dRuins)}` : "") + ". " +
    `No vermelho de s5, o item que mais mexe é ${t5[0].curto}, ${fx(t5[0], "s5")}. ` +
    `**Limite:** todo o falso alarme medido está em ${pl(s1verm.length, "sessão", "sessões")} de ${final.s1_sessoes} (${lista(s1verm)}), e ${pc(fracComForc("s1"), 1)} dele começa com um vermelho forçado (microssono ou cabeça caída com olhos fechados); ` +
    `os pesos mudam sobretudo quanto tempo esse vermelho dura (só ${fora} das ${sort.length} sorteadas criaram vermelho em outra sessão). ` +
    `São ${br(final.s1_minutos, 0)} min de direção real e ${br(final.s5_minutos, 0)} min de sonolência encenada: nestes dados o comportamento não depende de um ajuste fino desses números, mas isso não os transforma em números ajustados com sonolência real.`, "");
  return md;
}

// ---------------------------------------------------------------------------
// Ponto de entrada (no fim do arquivo: tudo acima já está definido)
// ---------------------------------------------------------------------------
if (!isMainThread) {
  const videos = carregarVideos();
  parentPort.on("message", (msg) => {
    if (msg === null) return parentPort.close();
    parentPort.postMessage({ idx: msg.idx, out: avaliar(msg.spec, videos) });
  });
  parentPort.postMessage({ pronto: true });
} else {
  await principal();
}
