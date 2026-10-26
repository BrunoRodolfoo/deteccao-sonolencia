/**
 * Analisa as execuções da PWA no Chrome headless com vídeos reais (tools/app_validacao/rodar_app_video.mjs)
 * e compara com a anotação quando existe (DMD s5). Gera:
 *   results/app_validacao.csv  (uma linha por vídeo x métrica)
 *   results/app_validacao.md   (tabelas + leitura)
 *   results/logs/app_validacao_<data>.log
 * e copia os dados por quadro (só números) para data/signals/app_headless/ (fora do Git).
 *
 *   node tools/app_validacao/analisar_app_videos.mjs --runs <pasta com <id>.json e <id>.clip.json> \
 *        [--versao "2.1 (ajuste final)"] [--anterior results_antigo.csv --versao-anterior "2.0 (antes do ajuste)"]
 *
 * Com --versao, o CSV ganha a coluna versao_app; com --anterior, as linhas do CSV anterior (mesmo formato, sem a
 * coluna) entram no começo com a versão --versao-anterior, e a tabela "antes x depois" vai para _tabelas.md.
 *
 * Sincronização vídeo x anotação (DMD): o app registra video.currentTime (vt) em cada quadro. O deslocamento
 * entre vt e o tempo do arquivo é estimado pela correlação cruzada do eyeBlink médio do app com o eyeBlink
 * extraído offline do MESMO vídeo (data/signals/app_tasks/dmd, mesmo Face Landmarker), em defasagens de
 * -3 a +3 s (passo de 1/30 s), no trecho inteiro e em cada metade (para ver deriva).
 * Casamento de eventos: um-para-um guloso pela maior sobreposição, com tolerância de 0,5 s em cada ponta
 * do intervalo detectado (a sincronização tem incerteza de ~1 quadro e os eventos do app têm início estimado).
 */
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, arr) => (a.startsWith("--") && acc.push([a.slice(2), arr[i + 1]]), acc), []));
const RUNS = path.resolve(args.runs);
const RES = args.res ? path.resolve(args.res) : path.join(ROOT, "results"); // --res <pasta> para ensaios (não grava em results/)
const _d = new Date();
const _p = (v) => String(v).padStart(2, "0");
const stamp = `${_d.getFullYear()}${_p(_d.getMonth() + 1)}${_p(_d.getDate())}-${_p(_d.getHours())}${_p(_d.getMinutes())}${_p(_d.getSeconds())}`;
const logLines = [];
const log = (...a) => { const s = a.join(" "); logLines.push(s); console.log(s); };
const TOL = 0.5;
const WIN_UTA_S = 180;
const DMD_FPS = 29.76;

const SPECS = {
  dmd_gE29_s5: { base: "DMD s5 (encenada)", ann: "data/dmd/gE/29/s5/gE_29_s5_2019-03-15T13;51;09+01;00_rgb_ann_drowsiness.json", sig: "gE_29_s5_2019-03-15T13;51;09+01;00", desc: "gE_29 (óculos)" },
  dmd_gZ33_s5: { base: "DMD s5 (encenada)", ann: "data/dmd/gZ/33/s5/gZ_33_s5_2019-04-04T09;29;07+02;00_rgb_ann_drowsiness.json", sig: "gZ_33_s5_2019-04-04T09;29;07+02;00", desc: "gZ_33" },
  dmd_gC14_s5: { base: "DMD s5 (encenada)", ann: "data/dmd/gC/14/s5/gC_14_s5_2019-03-12T09;18;58+01;00_rgb_ann_drowsiness.json", sig: "gC_14_s5_2019-03-12T09;18;58+01;00", desc: "gC_14" },
  dmd_gB7_s5: { base: "DMD s5 (encenada)", ann: "data/dmd/gB/7/s5/gB_7_s5_2019-03-13T13;55;52+01;00_rgb_ann_drowsiness.json", sig: "gB_7_s5_2019-03-13T13;55;52+01;00", desc: "gB_7 (extra: quedas de cabeça reais no motor offline)" },
  uta_F1_09_v0: { base: "UTA-RLDD 0 (alerta)", desc: "Fold1/09, vídeo 0" },
  uta_F3_31_v0: { base: "UTA-RLDD 0 (alerta)", desc: "Fold3/31, vídeo 0" },
  uta_F5_51_v0: { base: "UTA-RLDD 0 (alerta)", desc: "Fold5/51, vídeo 0" },
  uta_F1_09_v10: { base: "UTA-RLDD 10 (sonolento)", desc: "Fold1/09, vídeo 10" },
  uta_F3_31_v10: { base: "UTA-RLDD 10 (sonolento)", desc: "Fold3/31, vídeo 10" },
  uta_F5_51_v10: { base: "UTA-RLDD 10 (sonolento)", desc: "Fold5/51, vídeo 10" },
  nity_bocejo_F_P1042778: { base: "NITYMED bocejo (noite)", desc: "Yawning/Female P1042778" },
  nity_bocejo_M_P1042760: { base: "NITYMED bocejo (noite)", desc: "Yawning/Male P1042760" },
  nity_microssono_M_P1042751: { base: "NITYMED microssono (noite)", desc: "Microsleep/Male P1042751" },
  sint_cabeca_gC14: { base: "Sintético (cabeça)", ann: "data/dmd/gC/14/s5/gC_14_s5_2019-03-12T09;18;58+01;00_rgb_ann_drowsiness.json", sig: "gC_14_s5_2019-03-12T09;18;58+01;00", desc: "gC_14 s5, 0-80 s, com transformações" },
};

// ---------------------------------------------------------------------------
function loadRun(id) {
  const out = JSON.parse(fs.readFileSync(path.join(RUNS, `${id}.json`), "utf8"));
  const clip = JSON.parse(fs.readFileSync(path.join(RUNS, `${id}.clip.json`), "utf8"));
  const ix = Object.fromEntries(out.colunas.map((c, i) => [c, i]));
  const rows = out.rows.map((r) => Object.fromEntries(out.colunas.map((c) => [c, r[ix[c]]])));
  return { out, clip, rows };
}
function readSig(name) {
  const f = path.join(ROOT, "data/signals/app_tasks/dmd", `${name}.csv.gz`);
  if (!fs.existsSync(f)) return null;
  const txt = zlib.gunzipSync(fs.readFileSync(f)).toString("utf8").trim().split("\n");
  const h = txt[0].split(",");
  const I = Object.fromEntries(h.map((x, i) => [x, i]));
  return txt.slice(1).map((l) => { const c = l.split(","); return { t: Number(c[I.t_ms]) / 1000, face: c[I.face] === "1", b: (Number(c[I.blink_l]) + Number(c[I.blink_r])) / 2 }; });
}
function readAnn(file) {
  const d = JSON.parse(fs.readFileSync(path.join(ROOT, file), "utf8")).openlabel;
  const acts = {};
  for (const a of Object.values(d.actions || {})) for (const iv of a.frame_intervals || []) (acts[a.type] ||= []).push([Number(iv.frame_start), Number(iv.frame_end)]);
  let glasses = null;
  (function walk(o) { if (o && typeof o === "object") { if (o.name === "glasses" && "val" in o) glasses = o.val; for (const v of Object.values(o)) walk(v); } })(d);
  return { acts, glasses };
}
function pearson(x, y) {
  const n = x.length;
  if (n < 30) return NaN;
  let mx = 0, my = 0;
  for (let i = 0; i < n; i++) { mx += x[i]; my += y[i]; }
  mx /= n; my /= n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { const a = x[i] - mx, b = y[i] - my; sxy += a * b; sxx += a * a; syy += b * b; }
  return sxy / Math.sqrt(sxx * syy || 1);
}
/** Deslocamento L (s) tal que vt = t_arquivo + L, pela correlação do eyeBlink médio. */
function alignOffset(rows, sig, lead, vt0, vt1) {
  const sigAt = (tOrig) => { const k = Math.round(tOrig * DMD_FPS); return k >= 0 && k < sig.length && sig[k].face ? sig[k].b : null; };
  const sel = rows.filter((r) => r.face && r.vt >= vt0 && r.vt <= vt1 && Number.isFinite(r.blink_l) && Number.isFinite(r.blink_r));
  let best = { L: NaN, r: -2 };
  for (let k = -90; k <= 90; k++) {
    const L = k / 30;
    const x = [], y = [];
    for (const r of sel) { const s = sigAt(r.vt - L - lead); if (s !== null) { x.push((r.blink_l + r.blink_r) / 2); y.push(s); } }
    const c = pearson(x, y);
    if (c > best.r) best = { L, r: c, n: x.length };
  }
  return best;
}
function matchEvents(det, ann, tol = TOL) {
  const pairs = [];
  det.forEach((d, i) => ann.forEach((a, j) => { const ov = Math.min(d[1] + tol, a[1]) - Math.max(d[0] - tol, a[0]); if (ov >= 0) pairs.push([ov, i, j]); }));
  pairs.sort((p, q) => q[0] - p[0]);
  const ud = new Set(), ua = new Set();
  for (const [, i, j] of pairs) if (!ud.has(i) && !ua.has(j)) { ud.add(i); ua.add(j); }
  return { tpDet: ud.size, tpAnn: ua.size, matchedAnn: ua, matchedDet: ud };
}
const median = (a) => { const v = a.filter(Number.isFinite).sort((x, y) => x - y); return v.length ? (v.length % 2 ? v[v.length >> 1] : (v[v.length / 2 - 1] + v[v.length / 2]) / 2) : NaN; };
const pct = (a, p) => { const v = a.filter(Number.isFinite).sort((x, y) => x - y); return v.length ? v[Math.min(v.length - 1, Math.floor(p * (v.length - 1)))] : NaN; };
const f = (v, d = 2) => (Number.isFinite(v) ? v.toFixed(d) : "—");
const br = (v, d = 2) => f(v, d).replace(".", ",");

/** Episódios do app (em vt): fechamentos (>= 0,5 s), microssonos, bocejos, eventos de cabeça, alarmes. */
function appEpisodes(run) {
  const { out, rows } = run;
  const byT = new Map(rows.map((r) => [r.t, r]));
  const upd = new Map(out.upd.map((u) => [u.id, u]));
  const lastVt = rows[rows.length - 1].vt;
  const closures = [], yawns = [], head = [], alarms = [], warns = [];
  for (const e of out.ev) {
    if (e.type === "fechamento") closures.push({ a: e.vt - e.durMs / 1000, b: e.vt, dur: e.durMs / 1000, micro: false });
    else if (e.type === "microssono") {
      const cm = byT.get(e.t)?.fechado_ms ?? 1000;
      const a = e.vt - cm / 1000;
      const u = upd.get(e.id);
      const dur = u && Number.isFinite(u.durMs) ? u.durMs / 1000 : lastVt - a;
      closures.push({ a, b: a + dur, dur, micro: true, aberto: !u });
    } else if (e.type === "bocejo") {
      const a = e.vt - 1.5;
      const u = upd.get(e.id);
      const dur = u && Number.isFinite(u.durMs) ? u.durMs / 1000 : lastVt - a;
      yawns.push({ a, b: a + dur, dur });
    } else if (["queda", "cabeceio", "cabeca_baixa"].includes(e.type)) head.push({ type: e.type, vt: e.vt, peak: e.peakDeg });
    else if (e.type === "alarme") alarms.push({ vt: e.vt, motivo: e.motivo });
    else if (e.type === "aviso") warns.push({ vt: e.vt });
  }
  return { closures, yawns, head, alarms, warns };
}
/** Tempo em cada estado e outros resumos no intervalo [v0, v1] de vt (só fase ativa). */
function stateSummary(rows, v0, v1) {
  const T = [0, 0, 0];
  let tot = 0, faceN = 0, n = 0, maxLv = 0, firstRed = null, forcedS = 0;
  const lv = [];
  for (let i = 1; i < rows.length; i++) {
    const r = rows[i];
    if (r.fase !== 1 || r.vt < v0 || r.vt > v1) continue;
    const dt = Math.min(0.2, Math.max(0, r.vt - rows[i - 1].vt));
    T[r.estado] += dt;
    tot += dt;
    if (r.forcado) forcedS += dt;
    n++;
    faceN += r.face;
    maxLv = Math.max(maxLv, r.nivel);
    lv.push(r.nivel);
    if (r.estado === 2 && firstRed === null) firstRed = r.vt;
  }
  return { T, tot, faceRate: faceN / (n || 1), maxLv, meanLv: lv.reduce((a, b) => a + b, 0) / (lv.length || 1), firstRed, forcedS };
}
/** Piscadas (episódios fechados < 0,5 s) por minuto, duração mediana e fração do tempo com olho fechado. */
function blinkStats(rows, v0, v1) {
  let tot = 0, closedT = 0, cur = null;
  const runs = [];
  for (let i = 1; i < rows.length; i++) {
    const r = rows[i];
    if (r.fase !== 1 || r.vt < v0 || r.vt > v1) continue;
    const dt = Math.min(0.1, Math.max(0, r.vt - rows[i - 1].vt));
    tot += dt;
    if (r.face && r.fechado) { closedT += dt; if (cur === null) cur = r.vt; } else if (cur !== null) { runs.push(r.vt - cur); cur = null; }
  }
  const bl = runs.filter((x) => x < 0.5);
  return { perMin: bl.length / (tot / 60 || 1), medMs: 1000 * median(bl), closedFrac: closedT / (tot || 1) };
}
function fpsStats(rows) {
  const d = [];
  for (let i = 1; i < rows.length; i++) d.push(rows[i].t - rows[i - 1].t);
  const dur = (rows[rows.length - 1].t - rows[0].t) / 1000;
  return { fps: rows.length / dur, gapP95ms: pct(d, 0.95), gapMaxMs: Math.max(...d) };
}

// ---------------------------------------------------------------------------
const csv = [];
const add = (id, metrica, valor, obs = "") => csv.push({ video: id, base: SPECS[id].base, descricao: SPECS[id].desc, metrica, valor: Number.isFinite(valor) ? Number(valor.toFixed(4)) : valor, observacao: obs });
const ids = Object.keys(SPECS).filter((id) => fs.existsSync(path.join(RUNS, `${id}.json`)));
log(`# Validação da PWA com vídeos reais pela câmera falsa do Chrome headless — ${new Date().toISOString()}`);
log(`# Execuções em ${RUNS}; analisador tools/app_validacao/analisar_app_videos.mjs; tolerância de casamento ${TOL} s; janela UTA ${WIN_UTA_S} s após a calibração.`);
log("# Chrome headless: --headless=new --mute-audio --use-fake-ui-for-media-stream --use-fake-device-for-media-stream --use-file-for-fake-video-capture=<clipe .y4m local, fora do repositório> (tools/app_validacao/rodar_app_video.mjs); app aberto com ?autostart=1&volume=0; volume 0 % e ganho 0 conferidos a cada 1 s (aborta se não).");
log("# Clipes: tools/app_validacao/preparar_clipe.py (ffmpeg do imageio-ffmpeg; 30 fps; 640x480 ou 480x640 com faixas pretas; 5 s de preto antes e 3 s depois). Nenhum quadro ou rosto foi salvo; os .y4m foram apagados após cada execução.");
if (args.app) log(`# Versão do app servida (cópia congelada de app/pwa): ${args.app}`);
const summ = {};
for (const id of ids) {
  const S = SPECS[id];
  const run = loadRun(id);
  const { out, clip, rows } = run;
  const lead = clip.preto_inicio_s;
  const dur = clip.duracao_s;
  const cal = out.ev.find((e) => e.type === "calibrado");
  const baseBl = out.ev.find((e) => e.type === "base_piscadas");
  const firstFace = rows.find((r) => r.face);
  const fp = fpsStats(rows);
  const exc = out.console.filter((c) => c.kind === "EXCECAO").length;
  const silent = out.silencio.every((s) => s.pct === 0 && s.ganho === 0 && (s.ganhoNo === null || s.ganhoNo === 0));
  log("", `## ${id} — ${S.base}: ${S.desc}`);
  log(`clipe: ${clip.largura}x${clip.altura} a 30 fps (origem ${clip.origem.join("x")}), trecho ${clip.inicio_s}-${clip.inicio_s + dur} s, preto ${lead} s antes e ${clip.preto_fim_s} s depois; chrome ${out.chrome}; delegate ${out.final?.delegate}; vídeo no app ${out.final?.video?.join("x")}`);
  log(`silêncio: ${silent ? "OK" : "FALHOU"} (${out.silencio.length} checagens registradas; ex.: ${JSON.stringify(out.silencio[out.silencio.length - 1])}); exceções JS: ${exc}; quadros processados: ${rows.length}; FPS médio ${f(fp.fps, 1)}; intervalo p95 ${f(fp.gapP95ms, 0)} ms, máx ${f(fp.gapMaxMs, 0)} ms`);
  if (!cal) { log("SEM CALIBRAÇÃO — execução descartada"); add(id, "calibrou", 0); continue; }
  const bl = cal.baseline;
  log(`primeiro rosto em vt=${f(firstFace?.vt)} s (preto até ${lead} s); calibração concluída em vt=${f(cal.vt)} s: limiar EAR ${f(bl.earThr, 3)}, pitch0 ${f(bl.pitch0, 1)}°, jaw0 ${f(bl.jaw0, 3)}, avisos: ${JSON.stringify(bl.warnings)}; linha de base das piscadas: ${baseBl ? `${f(baseBl.baseMs, 0)} ms em vt=${f(baseBl.vt, 1)} s` : "não completou 20 piscadas"}`);
  add(id, "fps_medio", fp.fps);
  add(id, "excecoes_js", exc);
  add(id, "silencio_ok", silent ? 1 : 0, "--mute-audio + ?volume=0 + ganho 0 conferido");
  add(id, "calibracao_fim_s_video", cal.vt - lead, "segundos do trecho");
  add(id, "base_piscadas_ms", baseBl ? baseBl.baseMs : NaN, baseBl ? "" : "não completou 20 piscadas");
  const ep = appEpisodes(run);
  const vEnd = lead + dur; // fim do trecho útil no relógio do vídeo
  let off = 0;
  if (S.sig) {
    const sig = readSig(S.sig);
    const mid = (cal.vt + vEnd) / 2;
    const all = alignOffset(rows, sig, lead, cal.vt, vEnd);
    const h1 = alignOffset(rows, sig, lead, cal.vt, mid);
    const h2 = alignOffset(rows, sig, lead, mid, vEnd);
    off = all.L;
    log(`sincronização (eyeBlink app x offline): vt = t_arquivo + ${f(all.L, 3)} s (r=${f(all.r, 3)}, n=${all.n}); 1ª metade ${f(h1.L, 3)} s (r=${f(h1.r, 3)}), 2ª metade ${f(h2.L, 3)} s (r=${f(h2.r, 3)})`);
    add(id, "sinc_deslocamento_s", all.L, `r=${f(all.r, 3)}; metades ${f(h1.L, 3)} e ${f(h2.L, 3)} s`);
    add(id, "sinc_correlacao", all.r);
  }
  const toOrig = (vt) => vt - off - lead; // segundos no vídeo original
  const calO = toOrig(cal.vt);
  const endO = dur;
  // estados
  const winEnd = S.base.startsWith("UTA") ? Math.min(vEnd, cal.vt + WIN_UTA_S) : vEnd;
  const st = stateSummary(rows, cal.vt, winEnd);
  const inWin = (vt) => vt >= cal.vt && vt <= winEnd;
  const cnt = (type) => out.ev.filter((e) => e.type === type && inWin(e.vt)).length;
  const counts = { bocejo: cnt("bocejo"), fechamento: cnt("fechamento"), microssono: cnt("microssono"), queda: cnt("queda"), cabeceio: cnt("cabeceio"), cabeca_baixa: cnt("cabeca_baixa"), alarme: cnt("alarme"), aviso: cnt("aviso") };
  const minutes = st.tot / 60;
  log(`janela avaliada: vt ${f(cal.vt, 1)}-${f(winEnd, 1)} s (${f(st.tot, 1)} s ativos); rosto em ${f(100 * st.faceRate, 1)} % dos quadros`);
  log(`tempo: ALERTA ${f(st.T[0], 1)} s, ATENÇÃO ${f(st.T[1], 1)} s, SONOLENTO ${f(st.T[2], 1)} s (forçado ${f(st.forcedS, 1)} s); nível máx ${f(st.maxLv, 0)}, médio ${f(st.meanLv, 1)}; 1º vermelho ${st.firstRed === null ? "nunca" : `${f(st.firstRed - cal.vt, 1)} s após a calibração`}`);
  log(`eventos: ${JSON.stringify(counts)}`);
  const bs = blinkStats(rows, cal.vt, winEnd);
  log(`piscadas (< 0,5 s): ${f(bs.perMin, 1)}/min, duração mediana ${f(bs.medMs, 0)} ms; olho fechado em ${f(100 * bs.closedFrac, 1)} % do tempo`);
  add(id, "piscadas_por_min", bs.perMin);
  add(id, "piscada_duracao_mediana_ms", bs.medMs);
  add(id, "pct_tempo_olho_fechado", 100 * bs.closedFrac);
  st.bs = bs;
  {
    // PERCLOS do app na janela (após 60 s de observação) e, se a coluna existir, o PERCLOS com as piscadas.
    const w = rows.filter((r) => r.fase === 1 && r.vt >= cal.vt + 60 && r.vt <= winEnd);
    const pc = w.map((r) => r.perclos).filter(Number.isFinite);
    const pa = w.map((r) => r.perclos_piscadas).filter(Number.isFinite);
    st.perclosMax = pc.length ? Math.max(...pc) : NaN;
    st.perclosAllMax = pa.length ? Math.max(...pa) : NaN;
    log(`PERCLOS do app (após 60 s): p95 ${f(100 * pct(pc, 0.95), 1)} %, máx ${f(100 * st.perclosMax, 1)} %${pa.length ? `; com as piscadas (informativo): máx ${f(100 * st.perclosAllMax, 1)} %` : ""}`);
    add(id, "perclos_p95_pct", 100 * pct(pc, 0.95), "PERCLOS do app após 60 s de observação");
    add(id, "perclos_max_pct", 100 * st.perclosMax);
    if (pa.length) add(id, "perclos_com_piscadas_max_pct", 100 * st.perclosAllMax, "informativo: o PERCLOS antigo, com as piscadas");
  }
  {
    // quanto o LSTM (conselheiro, +10) pesou e qual seria o nível sem ele (soma dos outros sinais, sem teto)
    let mlN = 0, n = 0, maxNoML = 0;
    for (const r of rows) {
      if (r.fase !== 1 || r.vt < cal.vt || r.vt > winEnd || !r.pontos) continue;
      const p = Object.fromEntries(r.pontos.split("|").map((x) => x.split(":")).map(([k, v]) => [k, Number(v)]));
      n++;
      if (p.modelo > 0) mlN++;
      maxNoML = Math.max(maxNoML, Object.entries(p).filter(([k]) => k !== "modelo").reduce((a, [, v]) => a + v, 0));
    }
    log(`LSTM (+10) ativo em ${f((100 * mlN) / (n || 1), 1)} % dos quadros; soma máxima dos outros sinais: ${f(maxNoML, 0)} pontos`);
    add(id, "pct_quadros_lstm_mais10", (100 * mlN) / (n || 1));
    add(id, "nivel_max_sem_lstm", Math.min(100, maxNoML), "soma dos outros sinais (limitada a 100)");
    st.mlPct = (100 * mlN) / (n || 1);
    st.maxNoML = Math.min(100, maxNoML);
  }
  add(id, "segundos_avaliados", st.tot);
  add(id, "taxa_rosto", st.faceRate);
  add(id, "pct_tempo_alerta", (100 * st.T[0]) / (st.tot || 1));
  add(id, "pct_tempo_atencao", (100 * st.T[1]) / (st.tot || 1));
  add(id, "pct_tempo_sonolento", (100 * st.T[2]) / (st.tot || 1));
  add(id, "s_tempo_sonolento", st.T[2]);
  add(id, "s_tempo_atencao", st.T[1]);
  add(id, "nivel_max", st.maxLv);
  add(id, "nivel_medio", st.meanLv);
  add(id, "primeiro_vermelho_s_apos_calib", st.firstRed === null ? NaN : st.firstRed - cal.vt, st.firstRed === null ? "nunca ficou vermelho" : "");
  for (const [k, v] of Object.entries(counts)) add(id, `n_${k}`, v);
  add(id, "alarmes_por_hora", counts.alarme / (minutes / 60 || 1), "extrapolação linear da janela");
  // lista de eventos (log)
  const evList = out.ev.filter((e) => !["calibrado", "base_piscadas"].includes(e.type)).map((e) => {
    const u = out.upd.find((x) => x.id === e.id);
    const d = e.durMs ?? u?.durMs;
    return `${f(toOrig(e.vt), 1)}s ${e.type}${Number.isFinite(d) ? ` ${f(d / 1000, 1)}s` : ""}${Number.isFinite(e.peakDeg) ? ` pico ${f(e.peakDeg, 0)}°` : ""}${e.motivo ? ` (${e.motivo})` : ""}`;
  });
  log(`linha do tempo (s do vídeo original): ${evList.join("; ") || "nenhum evento"}`);
  summ[id] = { S, st, counts, cal, calO, baseBl, fp, off, ep, lead, dur, exc, silent };

  // ---- DMD: comparação com a anotação ----
  if (S.ann && id.startsWith("dmd_")) {
    const { acts, glasses } = readAnn(S.ann);
    const fr2s = (k) => k / DMD_FPS;
    const inEval = ([a, b]) => a > calO && b <= endO;
    const annY = [];
    for (const [k, ivs] of Object.entries(acts)) if (k.startsWith("yawning/")) for (const [a, b] of ivs) { const iv = [fr2s(a), fr2s(b + 1)]; if (inEval(iv)) annY.push({ iv, hand: /with hand/.test(k) && !/without/.test(k) }); }
    const annC = (acts["eyes_state/close"] || []).map(([a, b]) => [fr2s(a), fr2s(b + 1)]).filter(inEval);
    const detY = ep.yawns.map((y) => [toOrig(y.a), toOrig(y.b)]);
    // diagnóstico: jawOpen do app dentro de cada bocejo anotado (máximo e maior trecho contínuo >= 0,4)
    for (const a of annY) {
      const w = rows.filter((r) => r.face && toOrig(r.vt) >= a.iv[0] && toOrig(r.vt) <= a.iv[1]);
      let run = 0, best = 0, prev = null;
      for (const r of w) { const dt = prev === null ? 0 : r.vt - prev; prev = r.vt; run = r.jaw >= 0.4 ? run + dt : 0; best = Math.max(best, run); }
      const mx = Math.max(...w.map((r) => r.jaw).filter(Number.isFinite));
      log(`  bocejo anotado ${f(a.iv[0], 1)}-${f(a.iv[1], 1)} s (${a.hand ? "com mão" : "sem mão"}): jawOpen máx ${f(mx, 2)}, maior trecho >= 0,4: ${f(best, 2)} s`);
    }
    const mY = matchEvents(detY, annY.map((a) => a.iv));
    let tpH = 0, tpN = 0;
    for (const j of mY.matchedAnn) (annY[j].hand ? tpH++ : tpN++);
    const nH = annY.filter((a) => a.hand).length, nN = annY.length - nH;
    log(`óculos (anotação): ${glasses}; bocejos anotados após a calibração: ${annY.length} (${nN} sem mão, ${nH} com mão) em ${annY.map((a) => `${f(a.iv[0], 1)}-${f(a.iv[1], 1)}${a.hand ? "m" : ""}`).join(", ")}`);
    log(`bocejos detectados: ${detY.length} em ${detY.map((d) => `${f(d[0], 1)}-${f(d[1], 1)}`).join(", ")}; VP ${mY.tpDet}; revocação ${mY.tpAnn}/${annY.length} (sem mão ${tpN}/${nN}, com mão ${tpH}/${nH}); precisão ${mY.tpDet}/${detY.length}`);
    add(id, "oculos", glasses ? 1 : 0);
    add(id, "bocejos_anotados", annY.length);
    add(id, "bocejos_anotados_sem_mao", nN);
    add(id, "bocejos_anotados_com_mao", nH);
    add(id, "bocejos_detectados", detY.length);
    add(id, "bocejos_vp_anotados", mY.tpAnn);
    add(id, "bocejos_vp_sem_mao", tpN);
    add(id, "bocejos_vp_com_mao", tpH);
    add(id, "bocejos_vp_detectados", mY.tpDet);
    add(id, "bocejos_revocacao", annY.length ? mY.tpAnn / annY.length : NaN);
    add(id, "bocejos_precisao", detY.length ? mY.tpDet / detY.length : NaN);
    for (const minS of [0.5, 1.0]) {
      const A = annC.filter(([a, b]) => b - a >= minS);
      const D = ep.closures.filter((c) => c.dur >= minS).map((c) => [toOrig(c.a), toOrig(c.b)]);
      const m = matchEvents(D, A);
      const tag = minS === 1 ? "1s" : "05s";
      log(`fechamentos >= ${minS} s: anotados ${A.length}, detectados ${D.length}, revocação ${m.tpAnn}/${A.length}, precisão ${m.tpDet}/${D.length}`);
      add(id, `fech_${tag}_anotados`, A.length);
      add(id, `fech_${tag}_detectados`, D.length);
      add(id, `fech_${tag}_vp_anotados`, m.tpAnn);
      add(id, `fech_${tag}_vp_detectados`, m.tpDet);
      add(id, `fech_${tag}_revocacao`, A.length ? m.tpAnn / A.length : NaN);
      add(id, `fech_${tag}_precisao`, D.length ? m.tpDet / D.length : NaN);
      summ[id][`fech_${tag}`] = { A: A.length, D: D.length, tpA: m.tpAnn, tpD: m.tpDet };
    }
    // duração: app x anotação (pares casados entre TODOS os fechamentos anotados e os episódios >= 0,5 s do app)
    {
      const D = ep.closures.map((c) => [toOrig(c.a), toOrig(c.b)]);
      const pairs = [];
      D.forEach((d, i) => annC.forEach((a, j) => { const ov = Math.min(d[1], a[1]) - Math.max(d[0], a[0]); if (ov > 0) pairs.push([ov, i, j]); }));
      pairs.sort((p, q) => q[0] - p[0]);
      const ud = new Set(), ua = new Set(), diffs = [];
      for (const [, i, j] of pairs) if (!ud.has(i) && !ua.has(j)) { ud.add(i); ua.add(j); diffs.push((D[i][1] - D[i][0]) - (annC[j][1] - annC[j][0])); }
      const micro = ep.closures.filter((c) => c.dur >= 1).map((c) => [toOrig(c.a), toOrig(c.b)]);
      const microOnShort = micro.filter((d) => { const best = annC.filter((a) => Math.min(d[1], a[1]) - Math.max(d[0], a[0]) > 0); return best.length && best.every((a) => a[1] - a[0] < 1); }).length;
      log(`duração dos fechamentos (app - anotação, ${diffs.length} pares): mediana ${f(median(diffs), 2)} s, p25 ${f(pct(diffs, 0.25), 2)} s, p75 ${f(pct(diffs, 0.75), 2)} s; microssonos do app sobre fechamentos anotados de 0,5-1 s: ${microOnShort} de ${micro.length}`);
      add(id, "fech_duracao_app_menos_anot_mediana_s", median(diffs), `${diffs.length} pares`);
      add(id, "microssonos_sobre_fech_anotado_menor_1s", microOnShort, `de ${micro.length} microssonos do app`);
      summ[id].durDiff = { n: diffs.length, med: median(diffs), microOnShort, micro: micro.length, diffs };
    }
    summ[id].yawn = { A: annY.length, nN, nH, D: detY.length, tpA: mY.tpAnn, tpN, tpH, tpD: mY.tpDet };
    // olhos por quadro (fechado anotado x fechado do app), após a calibração
    const eye = new Int8Array(Math.ceil((endO + 5) * DMD_FPS)).fill(-1);
    const code = { "eyes_state/open": 0, "eyes_state/close": 1 };
    for (const [k, v] of Object.entries(code)) for (const [a, b] of acts[k] || []) for (let i = a; i <= b && i < eye.length; i++) eye[i] = v;
    let tp = 0, tn = 0, fpp = 0, fn = 0;
    for (const r of rows) {
      if (r.fase !== 1 || !r.face) continue;
      const k = Math.round(toOrig(r.vt) * DMD_FPS);
      if (k < 0 || k >= eye.length || eye[k] < 0 || toOrig(r.vt) <= calO) continue;
      if (r.fechado && eye[k] === 1) tp++; else if (!r.fechado && eye[k] === 0) tn++; else if (r.fechado) fpp++; else fn++;
    }
    const n = tp + tn + fpp + fn;
    const po = (tp + tn) / n, pe = ((tp + fpp) * (tp + fn) + (fn + tn) * (fpp + tn)) / (n * n);
    const kappa = (po - pe) / (1 - pe);
    log(`olhos por quadro (app no navegador x anotação): sensibilidade ${f(tp / (tp + fn), 3)}, especificidade ${f(tn / (tn + fpp), 3)}, kappa ${f(kappa, 3)} (n=${n})`);
    add(id, "olhos_quadro_sensibilidade", tp / (tp + fn));
    add(id, "olhos_quadro_especificidade", tn / (tn + fpp));
    add(id, "olhos_quadro_kappa", kappa);
    add(id, "olhos_quadro_n", n);
    summ[id].eyes = { tp, tn, fp: fpp, fn, kappa };
    summ[id].glasses = glasses;
  }

  // ---- sintético: janelas das transformações ----
  if (id.startsWith("sint_")) {
    const eps = [];
    for (const [k, lab] of [["pitch", "pitch (topo perto) "], ["pitch_longe", "pitch (topo longe) "], ["achatar", "achatamento "], ["roll", "roll "]]) for (const e of clip[k] || []) eps.push({ kind: k, lab: `${lab}${e[4]}°`, t0: e[0], t3: e[3], e });
    const { acts } = readAnn(S.ann);
    const closeIv = (acts["eyes_state/close"] || []).map(([a, b]) => [a / DMD_FPS, (b + 1) / DMD_FPS]);
    summ[id].sint = [];
    for (const e of eps) {
      const v0 = e.t0 + lead + off, v3 = e.t3 + lead + off;
      const w = rows.filter((r) => r.face && r.vt >= v0 && r.vt <= v3 && r.fase === 1);
      const mx = (k, fn = (x) => x) => Math.max(...w.map((r) => fn(r[k])).filter(Number.isFinite));
      const evs = out.ev.filter((x) => ["queda", "cabeceio", "cabeca_baixa"].includes(x.type) && x.vt >= v0 - 0.2 && x.vt <= v3 + 2.5).map((x) => x.type);
      const forced = w.some((r) => r.forcado && (r.cabeca === "caida"));
      const eyesClosedAnn = closeIv.some(([a, b]) => Math.min(b, e.t3) - Math.max(a, e.t0) > 0.5);
      const res = { ...e, maxDp: mx("dpitch"), minDp: Math.min(...w.map((r) => r.dpitch).filter(Number.isFinite)), maxAbsDr: mx("droll", Math.abs), maxAbsDy: mx("dyaw", Math.abs), evs, forced, eyesClosedAnn };
      summ[id].sint.push(res);
      log(`transformação ${e.lab} em ${e.t0}-${e.t3} s do trecho (olhos fechados na anotação: ${eyesClosedAnn ? "sim" : "não"}): Δpitch ${f(res.minDp, 1)} a ${f(res.maxDp, 1)}°, |Δroll| máx ${f(res.maxAbsDr, 1)}°, |Δyaw| máx ${f(res.maxAbsDy, 1)}°; eventos de cabeça: ${evs.join(", ") || "nenhum"}; vermelho forçado por cabeça caída: ${forced ? "sim" : "não"}`);
      const key = `${e.kind}_${e.t0}s`;
      add(id, `${key}_dpitch_max`, res.maxDp, e.lab);
      add(id, `${key}_absdroll_max`, res.maxAbsDr, e.lab);
      add(id, `${key}_absdyaw_max`, res.maxAbsDy, e.lab);
      add(id, `${key}_eventos_cabeca`, evs.join("+") || "nenhum", `${e.lab}; olhos fechados na anotação: ${eyesClosedAnn ? "sim" : "não"}`);
      add(id, `${key}_vermelho_forcado`, forced ? 1 : 0, e.lab);
    }
    const winsV = eps.map((e) => [e.t0 + lead + off - 0.2, e.t3 + lead + off + 2.5]);
    const outside = out.ev.filter((x) => ["queda", "cabeceio", "cabeca_baixa"].includes(x.type) && !winsV.some(([a, b]) => x.vt >= a && x.vt <= b));
    log(`eventos de cabeça FORA das janelas transformadas: ${outside.length}`);
    add(id, "eventos_cabeca_fora_das_janelas", outside.length, "cabeça parada (sem transformação) no mesmo clipe");
    summ[id].outside = outside.length;
  }
}
// controle do sintético: o mesmo trecho sem transformação (execução dmd_gC14_s5)
if (summ.sint_cabeca_gC14 && summ.dmd_gC14_s5) {
  const c = loadRun("dmd_gC14_s5");
  const lead = c.clip.preto_inicio_s, off = summ.dmd_gC14_s5.off;
  const evs = c.out.ev.filter((x) => ["queda", "cabeceio", "cabeca_baixa"].includes(x.type) && x.vt - lead - off <= 80.5);
  log("", `## Controle do teste sintético: dmd_gC14_s5 (mesmo vídeo, SEM transformação) nos primeiros 80 s: eventos de cabeça ${evs.length}${evs.length ? ` (${evs.map((x) => `${f(x.vt - lead - off, 1)}s ${x.type}`).join(", ")})` : ""}`);
  add("sint_cabeca_gC14", "controle_sem_transformacao_eventos_cabeca_0_80s", evs.length, "execução dmd_gC14_s5");
  summ.sint_cabeca_gC14.control = evs.length;
}

// ---------------------------------------------------------------------------
// CSV: com --versao, coluna versao_app; com --anterior, as linhas antigas entram antes (sem recalcular nada).
function parseCSV(txt) {
  const out = [];
  let row = [], cur = "", q = false;
  for (let i = 0; i < txt.length; i++) {
    const ch = txt[i];
    if (q) { if (ch === '"' && txt[i + 1] === '"') { cur += '"'; i++; } else if (ch === '"') q = false; else cur += ch; }
    else if (ch === '"') q = true;
    else if (ch === ",") { row.push(cur); cur = ""; }
    else if (ch === "\n") { row.push(cur); out.push(row); row = []; cur = ""; }
    else cur += ch;
  }
  if (cur || row.length) { row.push(cur); out.push(row); }
  return out;
}
const csvCell = (v) => (typeof v === "string" && /[",;\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
const csvHead = Object.keys(csv[0]);
const csvLines = [];
let oldRows = [];
if (args.anterior) {
  const parsed = parseCSV(fs.readFileSync(path.resolve(args.anterior), "utf8").trim() + "\n");
  const h = parsed[0];
  oldRows = parsed.slice(1).filter((r) => r.length === h.length).map((r) => Object.fromEntries(h.map((k, i) => [k, r[i]])));
}
if (args.versao) {
  csvLines.push(["versao_app", ...csvHead].join(","));
  for (const r of oldRows) csvLines.push([csvCell(args["versao-anterior"] || "anterior"), ...csvHead.map((k) => csvCell(r[k] ?? ""))].join(","));
  for (const r of csv) csvLines.push([csvCell(args.versao), ...Object.values(r).map(csvCell)].join(","));
} else {
  csvLines.push(csvHead.join(","));
  for (const r of csv) csvLines.push(Object.values(r).map(csvCell).join(","));
}
fs.writeFileSync(path.join(RES, "app_validacao.csv"), csvLines.join("\n") + "\n");
fs.writeFileSync(path.join(RUNS, "_resumo.json"), JSON.stringify(summ, null, 1));
const logPath = path.join(RES, "logs", `app_validacao_${stamp}.log`);
fs.writeFileSync(logPath, logLines.join("\n") + "\n");
// dados por quadro (só números) para reanálise: data/ é ignorado pelo Git
const keep = args.res ? path.join(RES, "app_headless") : path.join(ROOT, "data/signals/app_headless");
fs.mkdirSync(keep, { recursive: true });
for (const id of ids) fs.writeFileSync(path.join(keep, `${id}.json.gz`), zlib.gzipSync(fs.readFileSync(path.join(RUNS, `${id}.json`))));
console.log(`\nCSV: results/app_validacao.csv (${csv.length} linhas); log: ${path.relative(ROOT, logPath)}; resumo: ${path.join(RUNS, "_resumo.json")}`);

// ---------------------------------------------------------------------------
// Tabelas em Markdown (a leitura/interpretação é escrita à parte em results/app_validacao.md)
const md = [];
const P = (a, b) => (b ? `${a}/${b} (${br((100 * a) / b, 0)} %)` : "—");
const dmdIds = ids.filter((id) => id.startsWith("dmd_") && summ[id]?.yawn);
if (dmdIds.length) {
  md.push("### DMD s5 (sonolência encenada, anotação quadro a quadro)", "", "| Sessão | Óculos | FPS | Olho fechado por quadro: sens. / espec. / kappa | Fechamentos >= 1 s: revocação | Precisão | Fechamentos >= 0,5 s: revocação | Precisão | Bocejos: revocação (sem mão / com mão) | Precisão | 1º vermelho após calibrar | Tempo em vermelho | Alarmes |", "|---|---|---:|---|---|---|---|---|---|---|---:|---:|---:|");
  const tot = { e: [0, 0, 0, 0], c1: [0, 0, 0, 0], c05: [0, 0, 0, 0], y: [0, 0, 0, 0, 0, 0, 0] };
  for (const id of dmdIds) {
    const s = summ[id];
    const e = s.eyes;
    const c1 = s.fech_1s, c05 = s.fech_05s, y = s.yawn;
    tot.e[0] += e.tp; tot.e[1] += e.tn; tot.e[2] += e.fp; tot.e[3] += e.fn;
    tot.c1[0] += c1.A; tot.c1[1] += c1.D; tot.c1[2] += c1.tpA; tot.c1[3] += c1.tpD;
    tot.c05[0] += c05.A; tot.c05[1] += c05.D; tot.c05[2] += c05.tpA; tot.c05[3] += c05.tpD;
    tot.y[0] += y.A; tot.y[1] += y.D; tot.y[2] += y.tpA; tot.y[3] += y.tpD; tot.y[4] += y.nN; tot.y[5] += y.tpN; tot.y[6] += y.tpH;
    md.push(`| ${s.S.desc} | ${s.glasses ? "sim" : "não"} | ${br(s.fp.fps, 0)} | ${br(e.tp / (e.tp + e.fn), 3)} / ${br(e.tn / (e.tn + e.fp), 3)} / ${br(e.kappa, 3)} | ${P(c1.tpA, c1.A)} | ${P(c1.tpD, c1.D)} | ${P(c05.tpA, c05.A)} | ${P(c05.tpD, c05.D)} | ${P(y.tpA, y.A)} (${y.tpN}/${y.nN} · ${y.tpH}/${y.A - y.nN}) | ${P(y.tpD, y.D)} | ${s.st.firstRed === null ? "nunca" : `${br(s.st.firstRed - s.cal.vt, 1)} s`} | ${br((100 * s.st.T[2]) / s.st.tot, 1)} % | ${s.counts.alarme} |`);
  }
  const [tp, tn, fp, fn] = tot.e;
  const n = tp + tn + fp + fn, po = (tp + tn) / n, pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n);
  md.push(`| **Total** | | | ${br(tp / (tp + fn), 3)} / ${br(tn / (tn + fp), 3)} / ${br((po - pe) / (1 - pe), 3)} | ${P(tot.c1[2], tot.c1[0])} | ${P(tot.c1[3], tot.c1[1])} | ${P(tot.c05[2], tot.c05[0])} | ${P(tot.c05[3], tot.c05[1])} | ${P(tot.y[2], tot.y[0])} (${tot.y[5]}/${tot.y[4]} · ${tot.y[6]}/${tot.y[0] - tot.y[4]}) | ${P(tot.y[3], tot.y[1])} | | | |`, "");
}
const stIds = ids.filter((id) => !id.startsWith("sint_") && summ[id]);
md.push("### Tempo em cada nível e eventos (após a calibração)", "", "| Vídeo | Base | Janela | Rosto | Verde | Amarelo | Vermelho | Alarmes | Avisos | Bocejos | Fech. 0,5-1 s | Microssonos | Quedas | Cabeceios | Cabeça baixa | Nível máx. | Piscadas/min | Olho fechado |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
for (const id of stIds) {
  const s = summ[id], T = s.st.T, t = s.st.tot || 1, c = s.counts;
  md.push(`| ${s.S.desc} | ${s.S.base} | ${br(t, 0)} s | ${br(100 * s.st.faceRate, 0)} % | ${br((100 * T[0]) / t, 1)} % | ${br((100 * T[1]) / t, 1)} % | ${br((100 * T[2]) / t, 1)} % | ${c.alarme} | ${c.aviso} | ${c.bocejo} | ${c.fechamento} | ${c.microssono} | ${c.queda} | ${c.cabeceio} | ${c.cabeca_baixa} | ${br(s.st.maxLv, 0)} | ${br(s.st.bs.perMin, 0)} | ${br(100 * s.st.bs.closedFrac, 1)} % |`);
}
md.push("");
if (summ.sint_cabeca_gC14?.sint) {
  const s = summ.sint_cabeca_gC14;
  md.push("### Cabeça: teste sintético (transformação do vídeo com ffmpeg)", "", "| Transformação (trecho) | Olhos (anotação) | Δpitch medido | \\|Δroll\\| máx. | \\|Δyaw\\| máx. | Evento de cabeça | Vermelho forçado |", "|---|---|---:|---:|---:|---|---|");
  for (const e of s.sint) md.push(`| ${e.lab} (${br(e.t0, 1)}-${br(e.t3, 1)} s) | ${e.eyesClosedAnn ? "fechados" : "abertos"} | ${br(e.minDp, 1)} a ${br(e.maxDp, 1)}° | ${br(e.maxAbsDr, 1)}° | ${br(e.maxAbsDy, 1)}° | ${e.evs.join(", ") || "nenhum"} | ${e.forced ? "sim" : "não"} |`);
  md.push("", `Eventos de cabeça fora das janelas transformadas (cabeça parada): ${s.outside}. Controle (mesmo trecho sem transformação, execução dmd_gC14_s5): ${s.control ?? "—"} evento(s) de cabeça.`, "");
}
// Antes x depois (com --anterior): mesmo vídeo, versão anterior do CSV x esta execução.
if (oldRows.length) {
  const val = (rows, id, m) => { const r = rows.find((x) => x.video === id && x.metrica === m); return r ? Number(r.valor) : NaN; };
  const cur = csv.map((r) => ({ ...r, valor: String(r.valor) }));
  md.push("### Antes x depois do ajuste (mesmos clipes)", "", `| Vídeo | Base | Verde (antes → depois) | Amarelo | Vermelho | Alarmes | Avisos | Nível máx. | Microssonos | Bocejos |`, "|---|---|---|---|---|---|---|---|---|---|");
  for (const id of stIds) {
    const s = summ[id];
    const c = (m, d = 1) => `${br(val(oldRows, id, m), d)} → ${br(val(cur, id, m), d)}`;
    md.push(`| ${s.S.desc} | ${s.S.base} | ${c("pct_tempo_alerta")} % | ${c("pct_tempo_atencao")} % | ${c("pct_tempo_sonolento")} % | ${c("n_alarme", 0)} | ${c("n_aviso", 0)} | ${c("nivel_max", 0)} | ${c("n_microssono", 0)} | ${c("n_bocejo", 0)} |`);
  }
  md.push("");
}
fs.writeFileSync(path.join(RUNS, "_tabelas.md"), md.join("\n") + "\n");
console.log(`tabelas: ${path.join(RUNS, "_tabelas.md")}`);
