/**
 * Testes do motor (sem dependências): node test_engine.mjs
 * Valida EAR, MAR, recorte, detector de piscadas (4 características), z-score,
 * janela, voto majoritário, PERCLOS, suavização, calibração e alertas (v1, blocos 1-10)
 * e o detector v2 (blocos 11-21): pose a partir da matriz, fusão dos olhos, fechamentos,
 * PERCLOS por tempo, bocejo, cabeça caindo, pontos com decaimento, nível com histerese,
 * alarme (sem áudio), conselheiro do LSTM e o monitor completo com quadros sintéticos;
 * bloco 22: regra do volume do alarme (padrão 20 %, ?volume= da URL, curva do ganho);
 * bloco 23: o roteiro da demonstração (2 bocejos + microssono, "Estou acordado", Recalibrar);
 * bloco 24: casos difíceis (sinais negativos da pose, cabeça baixa com piscadas comuns,
 * nível sem oscilar nem ficar preso no vermelho);
 * bloco 25: ajuste final a partir da validação com vídeos reais (PERCLOS só com fechamentos >= 0,5 s, quem
 * pisca muito continua verde, LSTM só soma com os sinais visíveis em ATENÇÃO, saída do vermelho forçado,
 * 1 min de ATENÇÃO depois do microssono, rosto perdido exige olhos fechados, inclinação lateral no evento,
 * aviso de olhos fechados na calibração, histerese opcional dos olhos).
 * Regras mudadas de propósito no ajuste final (27/09/2026) e as asserções antigas trocadas: [16] rosto perdido
 * com olhos abertos não é mais queda; [21] S2 (1 min de ATENÇÃO após o microssono) e S7 (LSTM só soma com os
 * sinais visíveis >= 40); [23] roteiro da banca (ao reabrir os olhos, volta ao amarelo em segundos).
 */
import {
  LEFT_EYE, RIGHT_EYE, eyeAspectRatio, earFromLandmarks, marFromLandmarks, eyeCropBox,
  normalizeEyeInput, quantize, dequantize, softmax, argmax, median, percentile,
  Perclos, ThresholdCalibrator, BlinkDetector, SubjectNormalizer, BlinkWindow,
  ClassSmoother, majorityVote, scoreToClass, expectedScore, LatencyStats, AlertLogic, toCSV,
  CFG, ramp, matrixLayout, poseFromMatrix, matrixFromPose, blendshapeMap, fuseEyeClosure, ClosureTracker,
  PerclosTime, YawnDetector, HeadDropDetector, DecayingEvents, MLAdvisory, LevelController, AlarmController,
  DrowsinessMonitor, SOUND, parseVolumePct, resolveVolume, volumeGain,
} from "./engine.js";
import { adaptEyeMeta } from "./tflite_io.js";

let passed = 0;
let failed = 0;
function ok(cond, msg) {
  if (cond) { passed++; console.log("  ok  -", msg); }
  else { failed++; console.log("  FALHOU -", msg); }
}
function near(a, b, tol = 1e-6) { return Math.abs(a - b) <= tol; }

console.log("[1] EAR de pontos sintéticos");
{
  // p1=(0,0) p4=(4,0); p2=(1,-1) p6=(1,1); p3=(3,-1) p5=(3,1) -> (2+2)/(2*4)=0.5
  const pts = [[0, 0], [1, -1], [3, -1], [4, 0], [3, 1], [1, 1]];
  ok(near(eyeAspectRatio(pts), 0.5), "EAR = 0.5 para olho sintético aberto");
  const closed = [[0, 0], [1, 0], [3, 0], [4, 0], [3, 0], [1, 0]];
  ok(near(eyeAspectRatio(closed), 0), "EAR = 0 para olho fechado");
  ok(eyeAspectRatio([[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]) === 0, "EAR degenerado -> 0 sem divisão por zero");
}

console.log("[2] EAR/MAR a partir de landmarks normalizados");
{
  const lm = Array.from({ length: 478 }, () => ({ x: 0, y: 0 }));
  const W = 100, H = 100;
  const set = (idxs, pts) => idxs.forEach((i, k) => { lm[i] = { x: pts[k][0] / W, y: pts[k][1] / H }; });
  set(LEFT_EYE, [[0, 0], [1, -1], [3, -1], [4, 0], [3, 1], [1, 1]]);       // 0.5
  set(RIGHT_EYE, [[0, 0], [1, -0.5], [3, -0.5], [4, 0], [3, 0.5], [1, 0.5]]); // 0.25
  const e = earFromLandmarks(lm, W, H);
  ok(near(e.left, 0.5) && near(e.right, 0.25) && near(e.mean, 0.375), `EAR L/R/médio = ${e.left}/${e.right}/${e.mean}`);
  lm[61] = { x: 0.1, y: 0.5 }; lm[291] = { x: 0.5, y: 0.5 }; // largura 40 px
  lm[13] = { x: 0.3, y: 0.4 }; lm[14] = { x: 0.3, y: 0.6 };  // altura 20 px
  ok(near(marFromLandmarks(lm, W, H), 0.5), "MAR = 0.5");
}

console.log("[3] Caixa de recorte do olho");
{
  const pts = [[10, 10], [12, 8], [16, 8], [20, 10], [16, 12], [12, 12]];
  const box = eyeCropBox(pts, 640, 480, 1.6);
  ok(near(box.side, 16), `lado = 1.6 x |p1-p4| = ${box.side}`);
  ok(near(box.cx, 14.333333, 1e-4) && near(box.cy, 10), "centro no centroide dos 6 pontos");
  ok(box.x >= 0 && box.y >= 0, "caixa limitada ao quadro");
  const box2 = eyeCropBox([[0, 0], [1, 0], [2, 0], [3, 0], [2, 0], [1, 0]], 20, 20, 1.6);
  ok(box2.side === 8 && box2.x === 0 && box2.y === 0, "lado mínimo e limite superior esquerdo respeitados");
}

console.log("[4] Normalização e quantização do recorte");
{
  const g = new Float32Array([0, 127.5, 255]);
  const a = normalizeEyeInput(g, { normalize: "0-1" });
  ok(near(a[0], 0) && near(a[1], 0.5) && near(a[2], 1), "0-1");
  const b = normalizeEyeInput(g, { normalize: "-1-1" });
  ok(near(b[0], -1) && near(b[1], 0) && near(b[2], 1), "-1-1");
  const c = normalizeEyeInput(g, { normalize: "mean_std", mean: 0.5, std: 0.25 });
  ok(near(c[0], -2) && near(c[1], 0) && near(c[2], 2), "mean_std");
  const q = quantize(new Float32Array([0, 0.5, 1]), { scale: 1 / 255, zero_point: -128 }, "int8");
  ok(q[0] === -128 && q[1] === 0 && q[2] === 127, `int8 quantizado = [${Array.from(q)}]`);
  const d = dequantize(new Int32Array([-128, 0, 127]), { scale: 1 / 255, zero_point: -128 });
  ok(near(d[0], 0) && near(d[1], 0.50196, 1e-4) && near(d[2], 1), "desquantização inversa");
  const s = softmax([1, 2, 3]);
  ok(near(s[0] + s[1] + s[2], 1) && argmax(s) === 2, "softmax soma 1 e argmax=2");
  ok(median([5, 1, 3]) === 3 && median([4, 1, 3, 2]) === 2.5, "mediana ímpar/par");
  ok(percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) === 6 || percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) === 5, "percentil 50");
}

console.log("[5] Detecção de uma piscada sintética -> 4 características");
{
  // Sequência de EAR a 30 fps: 10 quadros abertos (0.30), depois 0.25, 0.10 (bottom), 0.05?? não: bottom único
  // quadros: 0..9 abertos 0.30; 10: 0.15; 11: 0.05 (bottom); 12: 0.15; 13: 0.30 (end); depois abertos.
  const seq = [];
  for (let i = 0; i < 10; i++) seq.push(0.30);
  seq.push(0.15, 0.05, 0.15, 0.30, 0.30, 0.30);
  const det = new BlinkDetector({ threshold: 0.75 * 0.30, minClosedFrames: 2, maxClosedMs: 1500 });
  let blink = null;
  seq.forEach((e, i) => { const r = det.update(e, i * (1000 / 30)); if (r) blink = r; });
  ok(blink !== null, "uma piscada detectada");
  ok(blink.start === 9 && blink.bottom === 11 && blink.end === 13, `start/bottom/end = ${blink.start}/${blink.bottom}/${blink.end}`);
  const [F, A, D, V] = blink.features;
  // Duration = 13-9+1 = 5; Amplitude = (0.30 - 2*0.05 + 0.30)/2 = 0.25; Velocity = (0.30-0.05)/(13-11) = 0.125;
  // Frequency = 100 * 1 / (13+1) = 7.142857
  ok(D === 5, `Duration = ${D} (esperado 5)`);
  ok(near(A, 0.25), `Amplitude = ${A} (esperado 0.25)`);
  ok(near(V, 0.125), `Velocity = ${V} (esperado 0.125)`);
  ok(near(F, 100 / 14), `Frequency = ${F} (esperado ${100 / 14})`);
  ok(det.blinkCount === 1 && det.blinks.length === 1, "contador = 1");

  // Fechamento de 1 quadro só -> ignorado (mínimo 2 quadros)
  const det2 = new BlinkDetector({ threshold: 0.2, minClosedFrames: 2 });
  let n = 0;
  [0.3, 0.3, 0.1, 0.3, 0.3].forEach((e, i) => { if (det2.update(e, i * 33)) n++; });
  ok(n === 0 && det2.blinkCount === 0, "fechamento de 1 quadro não conta como piscada");

  // Fechamento longo (> 1.5 s) -> não é piscada
  const det3 = new BlinkDetector({ threshold: 0.2, minClosedFrames: 2, maxClosedMs: 1500 });
  let n3 = 0;
  const seq3 = [0.3, 0.3];
  for (let i = 0; i < 60; i++) seq3.push(0.05); // 2 s fechado a 30 fps
  seq3.push(0.3, 0.3);
  seq3.forEach((e, i) => { if (det3.update(e, i * (1000 / 30))) n3++; });
  ok(n3 === 0, "fechamento de 2 s não conta como piscada (max 1.5 s)");

  // Duas piscadas: frequência = 100*2/(end2+1)
  const det4 = new BlinkDetector({ threshold: 0.2, minClosedFrames: 2 });
  const seq4 = [0.3, 0.3, 0.1, 0.05, 0.3, 0.3, 0.3, 0.3, 0.1, 0.05, 0.1, 0.3];
  const bl = [];
  seq4.forEach((e, i) => { const r = det4.update(e, i * 33); if (r) bl.push(r); });
  ok(bl.length === 2, "duas piscadas detectadas");
  ok(near(bl[1].features[0], 100 * 2 / 12), `Frequency 2ª piscada = ${bl[1].features[0]} (esperado ${100 * 2 / 12})`);
  ok(bl[1].start === 7 && bl[1].bottom === 9 && bl[1].end === 11 && bl[1].features[2] === 5, "2ª piscada: start/bottom/end = 7/9/11, duration 5");
}

console.log("[6] Normalização por sujeito (z-score)");
{
  const norm = new SubjectNormalizer({ nCalib: 4 });
  norm.start();
  [[1, 10, 3, 0.1], [3, 20, 5, 0.3], [1, 10, 3, 0.1], [3, 20, 5, 0.3]].forEach((f) => norm.add(f));
  ok(norm.ready, "pronto após 4 piscadas");
  ok(near(norm.mean[0], 2) && near(norm.mean[1], 15) && near(norm.std[0], 1) && near(norm.std[1], 5), `média=${norm.mean} desvio=${norm.std}`);
  const z = norm.zscore([3, 20, 5, 0.3]);
  ok(near(z[0], 1) && near(z[1], 1) && near(z[2], 1) && near(z[3], 1), `z-score = ${z.map((v) => v.toFixed(3))}`);
  const norm2 = new SubjectNormalizer({ nCalib: 2 });
  norm2.start(); norm2.add([1, 1, 1, 1]); norm2.add([1, 1, 1, 1]);
  const z2 = norm2.zscore([2, 2, 2, 2]);
  ok(Number.isFinite(z2[0]), "desvio zero não gera NaN/Infinity");
}

console.log("[7] Janela de 30 piscadas");
{
  const w = new BlinkWindow({ size: 30, nFeatures: 4, minBlinks: 10, padPartial: true });
  ok(w.tensor() === null, "vazia -> null");
  for (let i = 0; i < 10; i++) w.push([i, i, i, i]);
  const t = w.tensor();
  ok(t && t.length === 120 && t[0] === 0 && t[(20) * 4] === 0 && t[29 * 4] === 9, "parcial preenchida com zeros à esquerda");
  for (let i = 10; i < 40; i++) w.push([i, i, i, i]);
  const t2 = w.tensor();
  ok(w.full && t2[0] === 10 && t2[29 * 4 + 3] === 39, "mantém somente as últimas 30");
  const strict = new BlinkWindow({ size: 30 });
  for (let i = 0; i < 29; i++) strict.push([0, 0, 0, 0]);
  ok(strict.tensor() === null, "modo estrito exige 30 piscadas");
}

console.log("[8] Voto majoritário com desempate por escore");
{
  ok(majorityVote([0, 0, 1, 2, 0]) === 0, "maioria simples");
  ok(majorityVote([0, 2, 0, 2], [1, 9, 1, 9]) === 0, "empate 0/2 com escore médio 5 (classe 1 não empatada) -> equidistante, fica a classe menor (0)");
  ok(majorityVote([1, 1, 0, 0], [2, 2, 2, 2]) === 0, "empate 0/1 com escore médio 2 -> alerta (0)");
  ok(majorityVote([0, 2, 0, 2], [8, 9, 8, 9]) === 2, "empate 0/2 com escore médio 8.5 -> sonolento (2)");
  ok(majorityVote([0, 2, 0, 2], [1, 2, 1, 2]) === 0, "empate 0/2 com escore médio 1.5 -> alerta (0)");
  ok(majorityVote([1, 2, 1, 2], [5, 5, 5, 5]) === 1, "empate 1/2 com escore médio 5 -> pouca vigilância (1)");
  ok(scoreToClass(2) === 0 && scoreToClass(5) === 1 && scoreToClass(8) === 2, "discretização <3.3 / 3.3-6.6 / >6.6");
  ok(near(expectedScore([0, 0, 1]), 10) && near(expectedScore([1, 0, 0]), 0) && near(expectedScore([0, 1, 0]), 5), "escore esperado 0..10");
  ok(majorityVote([]) === null, "lista vazia -> null");
}

console.log("[9] PERCLOS, calibração, suavização, latência, alertas, CSV");
{
  const p = new Perclos(1000);
  for (let t = 0; t < 10; t++) p.update(t * 100, t < 5);
  ok(near(p.value, 0.5), `PERCLOS = ${p.value} (5 de 10 quadros)`);
  for (let t = 10; t < 30; t++) p.update(t * 100, false);
  ok(p.value === 0, "janela deslizante esquece quadros antigos");

  const cal = new ThresholdCalibrator({ durationMs: 3000, factor: 0.75 });
  cal.start(0);
  for (let i = 0; i <= 90; i++) cal.update(i * (3000 / 90), 0.3 + (i % 3) * 0.01);
  ok(cal.threshold !== null && near(cal.threshold, 0.75 * 0.31), `limiar = ${cal.threshold} (0.75 x mediana 0.31)`);

  const sm = new ClassSmoother({ alpha: 0.5, history: 5 });
  let r;
  for (let i = 0; i < 5; i++) r = sm.update([0.1, 0.1, 0.8]);
  ok(r.cls === 2 && r.vote === 2 && r.probs[2] > 0.7, "suavização converge para a classe 2");

  const ls = new LatencyStats(300);
  for (let i = 1; i <= 100; i++) ls.push(i);
  ok(ls.p50 === 50 || ls.p50 === 51, `p50 = ${ls.p50}`);
  ok(ls.p95 >= 94 && ls.p95 <= 96, `p95 = ${ls.p95}`);
  for (let i = 0; i < 400; i++) ls.push(1);
  ok(ls.n === 300, "mantém as últimas 300");

  const al = new AlertLogic({ closedMs: 1000, perclosThresh: 0.15, drowsyMs: 2000 });
  let a = al.update(0, { closed: true, perclos: 0, cls: 0, mar: 0 });
  ok(a.alerts.length === 0, "olho fechado < 1 s não alerta");
  a = al.update(1100, { closed: true, perclos: 0.2, cls: 2, mar: 0.7 });
  ok(a.alerts.includes("OLHOS FECHADOS") && a.alerts.some((s) => s.startsWith("PERCLOS")), "olhos fechados >= 1 s e PERCLOS > 0.15 alertam");
  a = al.update(3200, { closed: false, perclos: 0.1, cls: 2, mar: 0.7 });
  ok(a.alerts.includes("SONOLÊNCIA SUSTENTADA") && a.warnings.includes("BOCEJO"), "sonolento sustentado e bocejo");

  const csv = toCSV([{ a: 1, b: "x,y" }, { a: 2.5, b: "z" }]);
  ok(csv.split("\n").length === 3 && csv.includes('"x,y"'), "CSV com escape");
}

console.log("[10] Adaptação do models/eye_state.json (frente de exportação) ao contrato da PWA");
{
  const real = {
    entrada: { forma: [1, 48, 48, 1], descricao: "recorte ... valores em [0,1] = pixel/255" },
    saida: { forma: [1, 1], descricao: "P(olho ABERTO); classe 1 = aberto" },
    fp32: { arquivo: "eye_state_fp32.tflite" },
    int8: { arquivo: "eye_state_int8.tflite", io: "uint8", entrada_dtype: "uint8", entrada_quantizacao: { scale: 1 / 255, zero_point: 0 }, saida_dtype: "uint8", saida_quantizacao: { scale: 1 / 256, zero_point: 0 } },
  };
  const m = adaptEyeMeta(real);
  ok(m.files.fp32 === "eye_state_fp32.tflite" && m.files.int8 === "eye_state_int8.tflite", "arquivos fp32/int8 mapeados");
  ok(m.input_size === 48 && m.channels === 1, "input_size/channels a partir de entrada.forma");
  ok(m.int8.input.dtype === "uint8" && near(m.int8.input.scale, 1 / 255) && m.int8.input.zero_point === 0, "quantização de entrada mapeada");
  ok(m.int8.output.dtype === "uint8" && near(m.int8.output.scale, 1 / 256), "quantização de saída mapeada");
  ok(m.output === "prob_open" && m.normalize === "0-1", "saída prob_open e normalização 0-1 inferidas");
  const pwa = { input_size: 24, files: { fp32: "x.tflite" }, int8: { input: { dtype: "int8", scale: 0.01, zero_point: -128 } } };
  const m2 = adaptEyeMeta(pwa);
  ok(m2.input_size === 24 && m2.files.fp32 === "x.tflite" && m2.int8.input.zero_point === -128, "JSON já no contrato da PWA passa inalterado");
  ok(adaptEyeMeta(null) === null, "null -> null");
}

// ===========================================================================
// Detector v2
// ===========================================================================
const DT = 1000 / 30; // 30 FPS

console.log("[11] Pose da cabeça a partir de matrizes 4x4 sintéticas (coluna a coluna e linha a linha)");
{
  const I = matrixFromPose({ tz: -40 });
  const p0 = poseFromMatrix(I);
  ok(near(p0.pitch, 0) && near(p0.yaw, 0) && near(p0.roll, 0) && p0.layout === "col" && near(p0.tz, -40), "identidade: ângulos 0, layout col, tz = -40");
  ok(I[14] === -40 && I[11] === 0, "coluna a coluna: translação em data[12..14] (como o packed_data do MediaPipe)");
  for (const layout of ["col", "row"]) {
    const m = matrixFromPose({ pitch: 20, tz: -45 }, layout);
    const p = poseFromMatrix(m);
    ok(p.layout === layout && near(p.pitch, 20, 1e-9) && near(p.yaw, 0, 1e-9) && near(p.roll, 0, 1e-9), `só pitch 20° (${layout}) -> pitch ${p.pitch.toFixed(6)}`);
    const q = poseFromMatrix(matrixFromPose({ pitch: -15, yaw: 30, roll: 10, tx: 3, ty: -2, tz: -50 }, layout));
    ok(near(q.pitch, -15, 1e-9) && near(q.yaw, 30, 1e-9) && near(q.roll, 10, 1e-9), `pitch -15 / yaw 30 / roll 10 (${layout}) recuperados`);
    ok(near(q.tx, 3) && near(q.ty, -2) && near(q.tz, -50), `translação recuperada (${layout})`);
  }
  // Com R = Rx(pitch), a frente do rosto (eixo +Z do modelo) vai para (0, -sen, cos): pitch > 0 aponta para -Y (baixo).
  const m20 = matrixFromPose({ pitch: 20 });
  ok(m20[9] < 0 && m20[10] > 0, "pitch > 0 gira a frente do rosto para baixo (-Y da câmera)");
  const scaled = matrixFromPose({ pitch: 12, yaw: -8, roll: 4 }).map((v, i) => ([0, 1, 2, 4, 5, 6, 8, 9, 10].includes(i) ? v * 1.3 : v));
  const ps = poseFromMatrix(scaled);
  ok(near(ps.pitch, 12, 1e-9) && near(ps.yaw, -8, 1e-9) && near(ps.roll, 4, 1e-9), "escala uniforme na parte 3x3 não altera os ângulos");
  ok(poseFromMatrix(null) === null && poseFromMatrix([1, 2, 3]) === null && poseFromMatrix(new Array(16).fill(NaN)) === null, "entrada inválida -> null");
  // Matriz REAL devolvida pelo Face Landmarker (DMD gA_1 s5, quadro 0; tools/app_validacao/extrair_sinais_app.py)
  const real = [0.98384, 0.04366, -0.17364, 0, -0.10647, 0.92237, -0.37134, 0, 0.14395, 0.38383, 0.91212, 0, -3.98343, -6.756, -42.29756, 1];
  const pr = poseFromMatrix(real);
  ok(pr.layout === "col" && near(pr.tz, -42.29756) && near(pr.pitch, -22.15, 0.05), `matriz real do MediaPipe: layout col, tz ${pr.tz.toFixed(1)} cm, pitch ${pr.pitch.toFixed(1)}° (câmera abaixo do rosto)`);
  ok(matrixLayout(matrixFromPose({ yaw: 10 }, "row")) === "row", "detecção automática da ordem linha a linha");
  ok(blendshapeMap([{ categoryName: "jawOpen", score: 0.7 }, { categoryName: "eyeBlinkLeft", score: 0.1 }]).jawOpen === 0.7, "blendshapeMap");
}

console.log("[12] Fusão EAR + blendshapes (olho fechado) e trava de olhar para baixo");
{
  const thr = 0.2;
  ok(fuseEyeClosure({ ear: 0.3, earThr: thr, blinkL: 0.7, blinkR: 0.6 }).closed, "os dois eyeBlink >= 0,5 -> fechado mesmo com EAR alto");
  ok(!fuseEyeClosure({ ear: 0.3, earThr: thr, blinkL: 0.7, blinkR: 0.3 }).byBlend, "só um olho >= 0,5 não basta pelo blendshape");
  ok(fuseEyeClosure({ ear: 0.1, earThr: thr, blinkL: 0.6, blinkR: 0.45 }).closed, "EAR abaixo do limiar + menor eyeBlink 0,45 (>= 0,4) -> fechado");
  ok(!fuseEyeClosure({ ear: 0.1, earThr: thr, blinkL: 0.6, blinkR: 0.3 }).closed, "EAR baixo com um olho só (eyeBlink 0,6 / 0,3: piscadela, sorriso torto) -> aberto");
  ok(!fuseEyeClosure({ ear: 0.1, earThr: thr, blinkL: 0.2, blinkR: 0.2 }).closed, "EAR baixo com eyeBlink 0,2 (olho aberto para o blendshape) -> aberto");
  const sq = fuseEyeClosure({ ear: 0.1, earThr: thr, blinkL: 0.45, blinkR: 0.45, squint: 0.55 });
  ok(!sq.closed && sq.squinting, "apertando os olhos (eyeSquint 0,55: risada, sol): EAR sozinho não conta");
  const as = fuseEyeClosure({ ear: 0.1, earThr: thr, blinkL: 0.45, blinkR: 0.45, dYaw: -28 });
  ok(!as.closed && as.aside, "olhando para o lado (Δyaw -28°): EAR sozinho não conta");
  ok(!fuseEyeClosure({ ear: 0.3, earThr: thr, blinkL: 0.55, blinkR: 0.55, blend0: 0.34 }).byBlend, "pálpebra pesada (eyeBlink de repouso 0,34): 0,55 não basta (limiar = 0,34 + 0,25 = 0,59)");
  ok(fuseEyeClosure({ ear: 0.3, earThr: thr, blinkL: 0.65, blinkR: 0.62, blend0: 0.34 }).byBlend, "... e 0,62-0,65 conta como fechado");
  ok(fuseEyeClosure({ ear: 0.3, earThr: thr, blinkL: 0.52, blinkR: 0.51, blend0: 0.1 }).byBlend, "repouso baixo (0,1): vale o piso de 0,5");
  const down = fuseEyeClosure({ ear: 0.1, earThr: thr, blinkL: 0.45, blinkR: 0.45, dPitchDown: 15 });
  ok(!down.closed && down.gated && down.headDown, "cabeça 15° abaixo: EAR baixo sozinho não conta");
  ok(fuseEyeClosure({ ear: 0.1, earThr: thr, blinkL: 0.8, blinkR: 0.8, dPitchDown: 15 }).closed, "cabeça abaixo mas blendshape fechado -> fechado");
  ok(!fuseEyeClosure({ ear: 0.1, earThr: thr, blinkL: 0.45, blinkR: 0.45, lookDown: 0.7 }).closed, "olhar para baixo (eyeLookDown 0,7): EAR sozinho não conta");
  ok(fuseEyeClosure({ ear: 0.1, earThr: thr }).closed && !fuseEyeClosure({ ear: 0.1, earThr: thr, dPitchDown: 20 }).closed, "sem blendshapes: vale o EAR, com a trava da cabeça");
}

console.log("[13] Episódios de olho fechado (piscada, microssono, rosto perdido)");
{
  const ct = new ClosureTracker();
  let ended = null;
  let micro = 0;
  let t = 0;
  for (let i = 0; i < 10; i++, t += DT) ct.update(t, false);
  for (let i = 0; i < 6; i++, t += DT) ct.update(t, true); // 6 quadros = 200 ms
  ended = ct.update(t, false).ended;
  ok(ended && near(ended.durMs, 200, 1e-6) && !ended.lost, `piscada de 6 quadros -> ${ended && ended.durMs.toFixed(0)} ms`);
  for (let i = 0; i < 45; i++, t += DT) if (ct.update(t, true).microsleep) micro++; // 1,5 s
  const e2 = ct.update(t, false).ended;
  ok(micro === 1 && e2 && near(e2.durMs, 1500, 1e-6), `fechamento de 1,5 s: microssono sinalizado 1 vez, duração ${e2 && e2.durMs.toFixed(0)} ms`);
  for (let i = 0; i < 5; i++, t += DT) ct.update(t, true);
  const lastSeen = t - DT;
  let e3 = null;
  for (let i = 0; i < 15 && !e3; i++, t += DT) e3 = ct.update(t, false, false).ended;
  ok(e3 && e3.lost && near(e3.end, lastSeen), "rosto perdido por > 0,3 s encerra o fechamento no último quadro visto");
}

console.log("[14] PERCLOS por tempo com denominador mínimo de 60 s");
{
  const p = new PerclosTime();
  let t = 0;
  for (; t <= 17000; t += DT) p.update(t, false);
  for (let k = 0; k < 90; k++, t += DT) p.update(t, true); // 3 s fechado
  ok(near(p.value, 3000 / 60000, 0.002) && p.rawValue > 0.14, `20 s observados, 3 s fechado: PERCLOS ${(p.value * 100).toFixed(1)} % (bruto ${(p.rawValue * 100).toFixed(1)} %)`);
  for (let k = 0; k < 45 * 30; k++, t += DT) p.update(t, false);
  ok(near(p.value, 3000 / 60000, 0.002) && near(p.observedMs, 60000, 100), `janela cheia (60 s) com os mesmos 3 s: ${(p.value * 100).toFixed(1)} %`);
  for (let k = 0; k < 20 * 30; k++, t += DT) p.update(t, false);
  ok(p.value === 0, "o fechamento sai da janela depois de 60 s");
  const q = new PerclosTime();
  q.update(0, false);
  q.update(5000, true);
  ok(near(q.observedMs, 100), "pausa longa entre quadros conta no máximo 100 ms");
}

console.log("[15] Bocejo: jawOpen sustentado >= 1,5 s; fala não conta");
{
  const run = (det, seq) => {
    const evs = [];
    const ends = [];
    let t = 0;
    for (const [jaw, n, extra] of seq) for (let i = 0; i < n; i++, t += DT) {
      const r = det.update(t, { face: true, jawOpen: jaw, mar: 0.05, ...(extra || {}) });
      if (r.event) evs.push({ ...r.event });
      if (r.ended) ends.push(r.ended);
    }
    return { evs, ends };
  };
  const y = new YawnDetector();
  y.setBaseline({ jaw0: 0.02, mar0: 0.03 });
  const a = run(y, [[0.02, 30], [0.8, 75], [0.02, 30]]); // 2,5 s aberta
  ok(a.evs.length === 1 && a.evs[0].t >= 1500 + 1000 - 1 && a.evs[0].t < 1000 + 1600, `1 bocejo emitido ao completar 1,5 s (t = ${a.evs[0] && a.evs[0].t.toFixed(0)} ms)`);
  ok(a.ends.length === 1 && near(a.ends[0].durMs, 74 * DT, 1e-6) && near(a.ends[0].peak, 0.8), `duração informada ao fechar a boca: ${a.ends[0] && a.ends[0].durMs.toFixed(0)} ms`);
  y.reset();
  ok(run(y, [[0.8, 30], [0.02, 30]]).evs.length === 0, "boca aberta 1 s não é bocejo");
  y.reset();
  const talk = [];
  for (let k = 0; k < 12; k++) talk.push([0.6, 8], [0.1, 8]); // abre/fecha a cada ~0,27 s por 6 s
  ok(run(y, talk).evs.length === 0, "fala (abre e fecha a cada 0,27 s por 6 s) não vira bocejo");
  y.reset();
  ok(run(y, [[0.8, 20], [0.1, 3], [0.8, 40]]).evs.length === 1, "fechamento de 0,1 s no meio não quebra o bocejo");
  y.reset();
  const lost = run(y, [[0.8, 20], [0.8, 20, { face: false }], [0.8, 30]]);
  ok(lost.evs.length === 1, "mão na frente (rosto perdido 0,67 s) não encerra o bocejo");
  const m = new YawnDetector();
  m.setBaseline({ jaw0: 0.02, mar0: 0.03 });
  let t = 0;
  let nm = 0;
  for (let i = 0; i < 60; i++, t += DT) if (m.update(t, { face: true, jawOpen: null, mar: 0.55 }).event) nm++;
  ok(nm === 1, "sem blendshape: MAR >= base + 0,4 por 1,5 s também conta");
  const hb = new YawnDetector();
  hb.setBaseline({ jaw0: 0.35 });
  ok(!hb.isOpen({ jawOpen: 0.55 }) && hb.isOpen({ jawOpen: 0.7 }), "jawOpen de repouso alto (0,35) exige >= 0,65");
}

console.log("[16] Cabeça caindo: queda sustentada, cabeceio, olhar para o lado, rosto perdido");
{
  const sim = (seq, { base = { pitch: 0, yaw: 0, roll: 0 }, cfg } = {}) => {
    const h = new HeadDropDetector(cfg || CFG.head);
    h.setBaseline(base);
    const evs = [];
    let sustainedAt = null;
    let t = 0;
    let last = null;
    for (const [n, pose, eyes, face] of seq) for (let i = 0; i < n; i++, t += DT) {
      last = h.update(t, { face: face !== false, pose: face === false ? null : pose, eyesClosed: !!eyes });
      if (last.event) evs.push({ ...last.event });
      if (last.sustained && sustainedAt === null) sustainedAt = t;
    }
    return { evs, sustainedAt, last, h };
  };
  const drop = sim([[30, { pitch: 0, yaw: 0, roll: 0 }], [90, { pitch: 25, yaw: 0, roll: 0 }, true], [30, { pitch: 0, yaw: 0, roll: 0 }]]);
  ok(drop.evs.length === 1 && drop.evs[0].type === "queda" && near(drop.evs[0].t - 1000, 800, DT + 1e-6), `cabeça 25° abaixo com olhos fechados: 1 evento "queda" ${drop.evs[0] && (drop.evs[0].t - 1000).toFixed(0)} ms após cair`);
  ok(drop.sustainedAt !== null && near(drop.sustainedAt - 1000, 800, DT + 1e-6), "com olhos fechados >= 0,8 s: sustentada (vermelho)");
  const low = sim([[30, { pitch: 0 }], [90, { pitch: 25 }], [30, { pitch: 0 }]]);
  ok(low.evs.length === 1 && low.evs[0].type === "cabeca_baixa" && near(low.evs[0].t - 1000, 2000, DT + 1e-6) && low.sustainedAt === null, "cabeça 25° abaixo 3 s com olhos ABERTOS: só 'cabeça baixa' (distração) aos 2 s, sem vermelho");
  const short = sim([[30, { pitch: 0 }], [21, { pitch: 25 }], [30, { pitch: 0 }]]);
  ok(short.evs.length === 0, "cabeça baixa 0,7 s com olhos abertos: nenhum evento");
  const nod = sim([[30, { pitch: 0 }], [3, { pitch: 8 }], [3, { pitch: 16 }], [15, { pitch: 22 }, true], [6, { pitch: 5 }]]);
  ok(nod.evs.length === 1 && nod.evs[0].type === "cabeceio", `cabeceio (queda rápida e volta, olhos fechados): ${nod.evs.map((e) => e.type).join(",") || "nenhum"}`);
  const glance = sim([[30, { pitch: 0 }], [3, { pitch: 8 }], [3, { pitch: 16 }], [15, { pitch: 22 }, false], [6, { pitch: 5 }]]);
  ok(glance.evs.length === 0, "mesma curva com olhos abertos (olhar rápido para o painel): nenhum evento");
  const blinkDip = sim([[30, { pitch: 0 }], [3, { pitch: 8 }], [3, { pitch: 16 }], [4, { pitch: 22 }, true], [11, { pitch: 22 }], [6, { pitch: 5 }]]);
  ok(blinkDip.evs.length === 0, "mesma curva com uma piscada comum (0,13 s) no meio: nenhum evento");
  const eyes = sim([[30, { pitch: 0 }], [40, { pitch: 25 }, true]]);
  ok(eyes.sustainedAt !== null && near(eyes.sustainedAt - 1000, 800, DT + 1e-6), "caída com olhos fechados: vermelho já em 0,8 s");
  const aside = sim([[30, { pitch: 0, yaw: 0 }], [90, { pitch: 25, yaw: 40 }]]);
  ok(aside.evs.length === 0 && aside.sustainedAt === null && aside.last.state === "lado", "olhando para o lado (yaw 40°): ignorado");
  // Regra do ajuste final: o rosto sumir só vale como queda se os olhos estavam fechados >= 0,4 s naquele instante.
  const lostOpen = sim([[30, { pitch: 0 }], [6, { pitch: 12 }], [90, null, false, false]]);
  ok(lostOpen.evs.length === 0 && lostOpen.sustainedAt === null, "rosto some logo após baixar 12° com os olhos ABERTOS (pegar algo no banco): nenhum evento, sem vermelho");
  const lost = sim([[30, { pitch: 0 }], [15, { pitch: 12 }, true], [90, null, false, false]]);
  ok(lost.evs.length === 1 && lost.evs[0].lost && lost.sustainedAt !== null, "rosto some logo após baixar 12° com os olhos fechados há 0,5 s: queda fora do quadro e vermelho em 2 s");
  const gone = sim([[30, { pitch: 0 }], [200, null, false, false]]);
  ok(gone.evs.length === 0 && gone.last.state === "sem_rosto", "rosto some com a cabeça no neutro: só 'sem rosto'");
  const roll = sim([[30, { pitch: 0, roll: 0 }], [40, { pitch: 5, roll: 35 }, true]]);
  ok(roll.evs.length === 1 && roll.evs[0].type === "queda", "tombando para o lado (roll 35°) por > 0,8 s: queda");
  const neg = sim([[30, { pitch: 0 }], [90, { pitch: 25 }, true]], { cfg: { ...CFG.head, pitchDownSign: -1 } });
  ok(neg.evs.length === 0, "com pitchDownSign = -1, pitch positivo é para cima: nenhum evento");
  const drift = sim([[30 * 240, { pitch: 6 }]]);
  ok(near(drift.h.base.pitch, 6 * (1 - Math.exp(-2)), 0.05) && drift.evs.length === 0, `postura 6° abaixo por 4 min: base acompanha (${drift.h.base.pitch.toFixed(2)}° = 6·(1-e^-2)), sem evento`);
}

console.log("[17] Pontos com decaimento linear e teto");
{
  const d = new DecayingEvents({ points: 20, decayMs: 300000, cap: 50 });
  d.add(0);
  ok(near(d.points(0), 20) && near(d.points(150000), 10) && d.points(300000) === 0, "20 pt no evento, 10 pt na metade, 0 no fim");
  d.add(300000); d.add(300000); d.add(300000);
  ok(d.points(300000) === 50 && d.count(300000) === 3, "teto de 50 pt com 3 eventos simultâneos; contagem 3");
  ok(near(ramp(0.15, 0.075, 0.15, 40, 60), 40) && ramp(0.05, 0.075, 0.15, 40, 60) === 0 && ramp(0.5, 0.075, 0.15, 40, 60) === 60 && ramp(null, 0, 1, 1) === 0, "rampa do PERCLOS: 0 até 7,5 %, 40 em 15 %, teto 60");
}

console.log("[18] Nível com subida imediata, descida limitada e histerese");
{
  const L = new LevelController();
  let r = L.update(0, 45);
  ok(r.level === 45 && r.state === 1 && r.changed, "45 -> ATENÇÃO na hora");
  r = L.update(1000, 0);
  ok(near(r.level, 42) && r.state === 1, "desce no máximo 3 pt/s (45 -> 42)");
  r = L.update(3000, 0);
  ok(near(r.level, 36) && r.state === 1, "36: continua em ATENÇÃO (sai só abaixo de 35)");
  r = L.update(4000, 0);
  ok(near(r.level, 33) && r.state === 0, "33: volta a ALERTA");
  r = L.update(5000, 72);
  ok(r.state === 2, "72 -> SONOLENTO");
  r = L.update(8000, 0);
  ok(near(r.level, 63) && r.state === 2, "63: continua SONOLENTO (sai só abaixo de 60)");
  r = L.update(10000, 0);
  ok(near(r.level, 57) && r.state === 1, "57: volta a ATENÇÃO");
  r = L.update(10100, 10, true);
  ok(r.state === 2 && r.level >= 75, "condição forçada: VERMELHO e nível >= 75");
}

console.log("[19] Lógica de alarme (sem áudio): sirene, ding, vibração, silenciar");
{
  const A = new AlarmController();
  let a = A.update(0, { state: 2 });
  ok(a.siren && a.alarmStart && a.flash && a.vibrate === "alarme", "entrou no vermelho: sirene, flash, vibração, alarme contado");
  a = A.update(500, { state: 2 });
  ok(a.siren && !a.alarmStart && a.vibrate === null, "sirene contínua; vibração só a cada 1,5 s");
  a = A.update(1000, { state: 1 });
  ok(a.siren, "saiu do vermelho em 1 s: sirene continua até completar 2 s");
  a = A.update(2100, { state: 1 });
  ok(!a.siren && A.alarms === 1, "após 2 s a sirene para");
  const B = new AlarmController();
  let b = B.update(0, { state: 1 });
  ok(b.ding && b.warnStart && b.vibrate === "aviso", "entrou em ATENÇÃO: ding + vibração curta");
  b = B.update(5000, { state: 1, newEvents: 1 });
  ok(!b.ding, "novo sinal 5 s depois: sem ding (mínimo 20 s)");
  b = B.update(21000, { state: 1, newEvents: 1 });
  ok(b.ding, "novo sinal 21 s depois: ding");
  b = B.update(22000, { state: 1 });
  ok(!b.ding, "sem sinal novo: sem ding");
  const C = new AlarmController();
  C.update(0, { state: 2 });
  C.snooze(2500);
  let c = C.update(3000, { state: 2 });
  ok(!c.siren && c.snoozed, "Estou acordado: vermelho por nível silenciado por 30 s");
  c = C.update(3100, { state: 2, forced: true });
  ok(c.siren, "microssono/cabeça caída nunca é silenciado");
  c = C.update(34000, { state: 2 });
  ok(c.siren, "depois de 30 s a sirene volta se o nível continuar vermelho");
  const Dd = new AlarmController();
  Dd.update(0, { state: 2 });
  Dd.snooze(500);
  const d1 = Dd.update(600, { state: 2 });
  ok(!d1.siren && d1.snoozed, "Estou acordado corta a sirene na hora, mesmo antes do mínimo de 2 s");
}

console.log("[20] Modelo LSTM como conselheiro (+10 no máximo)");
{
  const m = new MLAdvisory();
  for (let i = 0; i < 4; i++) m.update([0.1, 0.1, 0.8]);
  ok(m.points === 0, "4 predições 'sonolento': 0 pt");
  m.update([0.1, 0.1, 0.8]);
  ok(m.points === 10, "5 seguidas com p >= 0,6: +10 pt");
  m.update([0.7, 0.2, 0.1]);
  ok(m.points === 0, "uma predição 'alerta' zera");
  const w = new MLAdvisory();
  for (let i = 0; i < 6; i++) w.update([0.2, 0.25, 0.55]);
  ok(w.points === 0, "'sonolento' com p 0,55 (< 0,6) não conta");
}

console.log("[21] Monitor completo com quadros sintéticos (30 FPS)");
{
  const F = ({ face = true, ear = 0.3, bl = 0.1, jaw = 0.02, mar = 0.03, look = 0.2, pitch = -5, yaw = 2, roll = 1, ml = null } = {}) => ({
    face, ear: { left: ear, right: ear, mean: ear }, mar,
    blend: face ? { eyeBlinkLeft: bl, eyeBlinkRight: bl, jawOpen: jaw, eyeLookDownLeft: look, eyeLookDownRight: look } : null,
    pose: face ? { pitch, yaw, roll } : null, mlProbs: ml,
  });
  const OPEN = {};
  const CLOSED = { ear: 0.05, bl: 0.8 };
  const YAWN = { jaw: 0.8, mar: 0.6 };
  /** Monitor calibrado (10 s) + helpers; `rec` acumula eventos, estado máximo, sirene e dings. */
  const session = () => {
    const S = { mon: new DrowsinessMonitor(), t: 0, log: [], upd: [], maxState: 0, maxLevel: 0, siren: 0, dings: 0, last: null };
    S.step = (n, fr = OPEN, cb) => {
      for (let i = 0; i < n; i++, S.t += DT) {
        const r = S.mon.update(S.t, F(fr));
        S.last = r;
        S.log.push(...r.events);
        S.upd.push(...r.updates);
        if (S.mon.phase === "ativo" && !r.calib?.baseline) {
          S.maxState = Math.max(S.maxState, r.score.state);
          S.maxLevel = Math.max(S.maxLevel, r.score.level);
          if (r.alarm.siren) S.siren++;
          if (r.alarm.ding) S.dings++;
        }
        if (cb) cb(r, S.t);
      }
    };
    S.awake = (seconds) => { for (let s = 0; s < seconds; s += 4) { S.step(115); S.step(5, CLOSED); } S.step(30); };
    S.reset = () => { S.maxState = 0; S.maxLevel = 0; S.siren = 0; S.dings = 0; };
    S.mon.startCalibration(S.t);
    S.step(310);
    return S;
  };
  const sleepEvents = (log) => log.filter((e) => ["bocejo", "queda", "cabeceio", "fechamento", "microssono"].includes(e.type));

  // S1: acordado
  const A = session();
  ok(A.log.some((e) => e.type === "calibrado") && A.mon.phase === "ativo", "calibração de 10 s concluída");
  ok(near(A.mon.baseline.earThr, 0.225, 1e-9) && near(A.mon.baseline.pitch0, -5) && near(A.mon.baseline.jaw0, 0.02), `base: limiar EAR ${A.mon.baseline.earThr.toFixed(3)}, pitch ${A.mon.baseline.pitch0}°, jawOpen ${A.mon.baseline.jaw0}`);
  A.awake(120);
  ok(A.maxState === 0 && A.siren === 0 && A.dings === 0 && sleepEvents(A.log).length === 0, "2 min acordado piscando normalmente: sempre ALERTA, nenhum som, nenhum evento");
  ok(A.mon.blinkBaselineReady && near(A.mon.blinkBaseMs, 5 * DT, 1e-6), `linha de base das piscadas: ${A.mon.blinkBaseMs && A.mon.blinkBaseMs.toFixed(0)} ms`);

  // S2: microssono de 1,5 s
  const M = session();
  M.awake(20);
  M.reset();
  let firstRed = null;
  const t0 = M.t;
  M.step(45, CLOSED, (r, t) => { if (r.score.state === 2 && firstRed === null) firstRed = t; });
  ok(firstRed !== null && near(firstRed - t0, 1000, DT + 1e-6), `microssono: VERMELHO forçado ao completar 1 s (${(firstRed - t0).toFixed(0)} ms)`);
  ok(M.siren > 0 && M.log.some((e) => e.type === "microssono") && M.log.some((e) => e.type === "alarme"), "sirene + eventos 'microssono' e 'alarme' registrados");
  let backTo = null;
  const t1 = M.t;
  M.step(30 * 60, OPEN, (r, t) => { if (backTo === null && r.score.state < 2) backTo = t; });
  ok(M.upd.some((u) => near(u.durMs, 1500, 1e-6)), "duração do microssono (1,5 s) informada ao reabrir os olhos");
  ok(backTo !== null && backTo - t1 <= 7000, `depois de reabrir os olhos o vermelho termina em ${((backTo - t1) / 1000).toFixed(1)} s`);
  // Ajuste final: depois de um vermelho forçado, pelo menos 1 min em ATENÇÃO (antes: verde ~40 s depois).
  ok(M.last.score.state === 1 && M.last.score.all.some((c) => c.key === "pos_evento" && c.points > 0), `~60 s depois de UM microssono: ainda ATENÇÃO pelo piso de 1 min (nível ${M.last.score.level.toFixed(0)})`);
  M.step(30 * 8, OPEN);
  ok(M.last.score.state === 0, `68 s depois de UM microssono: ALERTA de novo (nível ${M.last.score.level.toFixed(0)})`);

  // S3: dois bocejos a 20 s um do outro -> ATENÇÃO + um ding
  const Y = session();
  Y.awake(8);
  Y.reset();
  Y.step(90, YAWN);
  Y.awake(20);
  Y.step(90, YAWN);
  Y.step(10);
  const nY = Y.log.filter((e) => e.type === "bocejo").length;
  ok(nY === 2 && Y.last.score.state === 1 && Y.dings === 1, `2 bocejos -> ATENÇÃO com 1 ding (nível ${Y.last.score.level.toFixed(0)})`);
  ok(Y.last.score.contributions[0].key === "bocejos", "'Por quê?' lista os bocejos em primeiro");
  // S4: bocejos sozinhos nunca levam ao vermelho
  const Y4 = session();
  Y4.reset();
  for (let k = 0; k < 4; k++) { Y4.step(90, YAWN); Y4.awake(28); }
  ok(Y4.maxState === 1 && Y4.maxLevel <= 50 + 1e-9, `4 bocejos em 2 min: no máximo ATENÇÃO (nível máximo ${Y4.maxLevel.toFixed(0)})`);

  // S5: cabeça caída com os olhos fechados: 'queda' (30 pt) e vermelho FORÇADO em 0,8 s
  const H = session();
  H.awake(8);
  H.reset();
  const tH = H.t;
  let headRedAt = null;
  let headPts = null;
  H.step(60, { pitch: 20, ...CLOSED }, (r, t) => {
    if (r.events.some((e) => e.type === "queda")) headPts = r.score.all.find((c) => c.key === "cabeca").points;
    if (r.score.state === 2 && headRedAt === null) headRedAt = t;
  });
  ok(headPts !== null && near(headPts, 30, 0.01), `'queda' vale 30 pt (${headPts && headPts.toFixed(1)})`);
  ok(headRedAt !== null && near(headRedAt - tH, 800, DT + 1e-6), `cabeça 25° abaixo com olhos fechados: vermelho forçado em ${((headRedAt - tH) / 1000).toFixed(2)} s`);
  // S5b: cabeça baixa 3 s com olhos ABERTOS (olhando o painel): registro, sem alarme
  const HB = session();
  HB.awake(8);
  HB.reset();
  HB.step(90, { pitch: 20 });
  HB.step(30);
  ok(HB.log.some((e) => e.type === "cabeca_baixa") && HB.maxState === 0 && HB.siren === 0, "cabeça baixa 3 s com olhos abertos: 'cabeça baixa' no registro, continua ALERTA, sem som");

  // S6: sinais combinados levam ao vermelho pelo nível (sem condição forçada)
  const C = session();
  C.reset();
  C.step(90, YAWN);
  C.awake(8);
  C.step(90, YAWN);
  C.awake(8);
  let redByLevel = false;
  const nod = [[3, { pitch: 3 }], [3, { pitch: 11 }], [3, { pitch: 16, ...CLOSED }], [15, { pitch: 20, ...CLOSED }], [6, { pitch: 0 }]];
  for (const [n, fr] of nod) C.step(n, fr, (r) => { if (r.score.state === 2 && !r.score.forced) redByLevel = true; });
  C.step(60, OPEN, (r) => { if (r.score.state === 2 && !r.score.forced) redByLevel = true; });
  ok(redByLevel && C.log.some((e) => e.type === "cabeceio"), "2 bocejos + 1 cabeceio (sem condição forçada): SONOLENTO pelo nível");

  // S7: o LSTM sozinho não muda nada: com os sinais visíveis abaixo de 40 ele vale 0 (ajuste final; antes, +10)
  const L = session();
  let mlPts = 0;
  L.reset();
  for (let i = 0; i < 30 * 60; i++) L.step(1, { ml: i % 90 === 0 ? [0.05, 0.05, 0.9] : null }, (r) => { mlPts = Math.max(mlPts, r.score.all.find((c) => c.key === "modelo").points); });
  ok(mlPts === 0 && L.maxState === 0 && L.mon.ml.eligible, `LSTM 'sonolento' com 90 % por 1 min, sinais visíveis em 0: +${mlPts} pt (elegível, mas não conta), estado máximo ALERTA`);
  // S7b: com os sinais visíveis já em ATENÇÃO (2 bocejos), o mesmo LSTM soma +10
  const L2 = session();
  L2.reset();
  L2.step(90, YAWN);
  L2.awake(8);
  L2.step(90, YAWN);
  let ml2 = 0;
  for (let i = 0; i < 30 * 20; i++) L2.step(1, { ml: i % 90 === 0 ? [0.05, 0.05, 0.9] : null }, (r) => { ml2 = Math.max(ml2, r.score.all.find((c) => c.key === "modelo").points); });
  ok(ml2 === 10 && L2.last.score.mlCounted, "com 2 bocejos (sinais visíveis >= 40), o LSTM 'sonolento' soma +10");

  // S8: sem rosto com a cabeça no neutro
  const N = session();
  N.reset();
  N.step(30 * 20, { face: false });
  ok(N.maxState === 0 && N.siren === 0, "20 s sem rosto (cabeça no neutro antes): nenhum alarme");

  // S9: estatísticas para o benchmark
  const st = C.mon.stats;
  ok(st.eventos.bocejo === 2 && st.eventos.cabeceio === 1 && st.alarmes >= 1 && st.avisos >= 1, `estatísticas: ${JSON.stringify(st.eventos)}, alarmes ${st.alarmes}, avisos ${st.avisos}, dings ${st.dings}`);
  const active = C.t - DT - C.log.find((e) => e.type === "calibrado").t;
  ok(Math.abs(st.tempoEstadoMs.reduce((a, b) => a + b, 0) - active) < 50, "tempo em cada estado soma a duração da sessão ativa");
}

console.log("[22] Volume do alarme: padrão 20 %, ?volume= da URL sem gravar, curva do ganho");
{
  ok(SOUND.defaultVolumePct === 20, "volume padrão = 20 % (baixo, para testes)");
  let v = resolveVolume({});
  ok(v.pct === 20 && v.source === "padrão", "sem URL e sem valor salvo: 20 % (padrão)");
  v = resolveVolume({ storedValue: "65" });
  ok(v.pct === 65 && v.source === "salvo", "valor salvo 65 -> 65 %");
  v = resolveVolume({ urlValue: "0", storedValue: "80" });
  ok(v.pct === 0 && v.source === "url", "?volume=0 sobrepõe o salvo (80) -> 0 %, fonte url");
  ok(resolveVolume({ urlValue: "150" }).pct === 100 && resolveVolume({ urlValue: "-5" }).pct === 0, "?volume fora de 0-100 é limitado (150 -> 100, -5 -> 0)");
  v = resolveVolume({ urlValue: "abc", storedValue: "40" });
  ok(v.pct === 40 && v.source === "salvo", "?volume inválido é ignorado (usa o salvo)");
  ok(resolveVolume({ urlValue: "", storedValue: null }).pct === 20, "?volume vazio é ignorado (usa o padrão)");
  ok(parseVolumePct("35,5") === 36 && parseVolumePct("35%") === 35 && parseVolumePct(true) === null && parseVolumePct(null) === null, "aceita vírgula decimal e %; rejeita booleano/nulo");
  ok(volumeGain(0) === 0 && volumeGain(null) === 0 && volumeGain("x") === 0, "volume 0 (ou inválido) -> ganho 0: silêncio absoluto");
  ok(near(volumeGain(100), 1) && near(volumeGain(50), 0.25) && near(volumeGain(20), 0.04), "curva quadrática: 100 % -> 1; 50 % -> 0,25; 20 % -> 0,04 (-28 dB)");
  let mono = true;
  for (let p = 1; p <= 100; p++) if (!(volumeGain(p) > volumeGain(p - 1))) mono = false;
  ok(mono, "ganho estritamente crescente de 0 a 100 %");
}

console.log("[23] Roteiro da demonstração (2 min): verde, 2 bocejos -> amarelo, cabeça baixa, microssono -> vermelho, reabrir -> volta");
{
  const D = 1000 / 30;
  const mon = new DrowsinessMonitor();
  let t = 0;
  const fr = ({ closed = false, jaw = 0.02 } = {}) => ({
    face: true,
    ear: { left: closed ? 0.08 : 0.3, right: closed ? 0.08 : 0.3, mean: closed ? 0.08 : 0.3 },
    mar: jaw > 0.3 ? 0.7 : 0.05,
    blend: { eyeBlinkLeft: closed ? 0.8 : 0.1, eyeBlinkRight: closed ? 0.8 : 0.1, jawOpen: jaw, eyeLookDownLeft: 0.2, eyeLookDownRight: 0.2, eyeSquintLeft: 0.1, eyeSquintRight: 0.1 },
    pose: { pitch: -5, yaw: 0, roll: 0 },
    mlProbs: null,
  });
  const trace = [];
  const run = (ms, f) => { let r = null; for (let i = 0, n = Math.round(ms / D); i < n; i++, t += D) { r = mon.update(t, f); trace.push([t, r]); } return r; };
  const blinks = (s, extra = {}) => { let r = null; for (let k = 0; k < s / 3; k++) { run(2800, fr(extra)); r = run(200, fr({ ...extra, closed: true })); } return r; };
  const since = (t0, pred) => trace.filter(([tt]) => tt >= t0).some(([, x]) => pred(x));
  mon.startCalibration(0);
  run(10000, fr());
  let tS = t;
  let r = blinks(15); // 1) piscar normal
  ok(!since(tS, (x) => x.score.state !== 0 || x.alarm.siren || x.alarm.ding), "1) 15 s piscando normalmente: verde o tempo todo, nenhum som");
  tS = t;
  run(2500, fr({ jaw: 0.8 })); r = blinks(12); // 2) 1º bocejo
  ok(r.yawn.count5min === 1 && r.score.state === 0, `2a) 1º bocejo: contado (1 em 5 min), ainda verde (nível ${Math.floor(r.score.level)})`);
  run(2500, fr({ jaw: 0.8 }));
  r = run(300, fr());
  ok(r.yawn.count5min === 2 && r.score.state === 1 && since(tS, (x) => x.alarm.ding), `2b) 2º bocejo: ATENÇÃO com "ding" (nível ${Math.floor(r.score.level)})`);
  blinks(9);
  tS = t;
  const fr3 = { pitch: 25 }; // 3) cabeça 30° abaixo da base (-5°), olhos abertos, piscando
  const blinks3 = (s) => { for (let k = 0; k < s / 3; k++) { run(2800, { ...fr(), pose: { pitch: fr3.pitch, yaw: 0, roll: 0 } }); run(200, { ...fr({ closed: true }), pose: { pitch: fr3.pitch, yaw: 0, roll: 0 } }); } };
  blinks3(6);
  ok(since(tS, (x) => x.events.some((e) => e.type === "cabeca_baixa")) && !since(tS, (x) => x.score.state === 2 || x.alarm.siren), "3) cabeça baixa 6 s com os olhos abertos: evento 'cabeça baixa' no registro, sem vermelho");
  blinks(6);
  tS = t;
  r = run(2000, fr({ closed: true })); // 4) fecha os olhos 2 s
  ok(r.score.state === 2 && r.score.forced && r.alarm.siren && since(tS, (x) => x.events.some((e) => e.type === "microssono")), "4) olhos fechados 2 s: microssono, VERMELHO forçado e sirene");
  const tOpen = t;
  r = blinks(21); // 5) reabre
  const back = trace.find(([tt, x]) => tt > tOpen && x.score.state < 2 && !x.alarm.siren);
  const backS = back ? (back[0] - tOpen) / 1000 : Infinity;
  ok(backS <= 5, `5) reabriu: a sirene para e o estado volta ao amarelo em ${backS.toFixed(1)} s`);
  ok(!since(tOpen + 5000, (x) => x.score.state === 2 || x.alarm.siren) && r.score.state === 1, `5b) e fica em ATENÇÃO (sem vermelho nem sirene) enquanto os sinais decaem (nível ${Math.floor(r.score.level)} após 21 s)`);
  const keys = r.score.contributions.map((c) => c.key);
  ok(keys.includes("bocejos") && keys.includes("fechamentos"), `6) "Por quê?": ${r.score.contributions.map((c) => `${c.key} +${Math.round(c.points)}`).join(", ")}`);
  const ms = mon.stats;
  ok(ms.eventos.bocejo === 2 && ms.eventos.microssono === 1 && ms.eventos.cabeca_baixa === 1 && ms.alarmes === 1, `registro: ${JSON.stringify(ms.eventos)}, alarmes ${ms.alarmes}`);
  mon.startCalibration(t);
  r = run(500, fr());
  ok(r.phase === "calibrando" && r.score.level === 0 && r.score.state === 0 && !r.alarm.siren && mon.stats.eventos.bocejo === 0, "Recalibrar zera nível, estado, sirene e contagens");
}

console.log("[24] Casos difíceis: sinais negativos da pose, cabeça baixa com piscadas, nível sem travar nem oscilar");
{
  for (const layout of ["col", "row"]) {
    const q = poseFromMatrix(matrixFromPose({ pitch: 25, yaw: -35, roll: -20, tz: -40 }, layout));
    ok(near(q.pitch, 25, 1e-9) && near(q.yaw, -35, 1e-9) && near(q.roll, -20, 1e-9), `pitch +25 / yaw -35 / roll -20 (${layout}) recuperados`);
  }
  const mYaw = matrixFromPose({ yaw: 30 });
  ok(mYaw[8] > 0, "yaw > 0 gira a frente do rosto (+Z) para +X");
  const mRoll = matrixFromPose({ roll: 30 });
  ok(mRoll[1] > 0, "roll > 0 gira o eixo X do rosto para +Y");
  // Cabeça 25° abaixo (olhando o celular) por 8 s com piscadas comuns de 0,2 s a cada 1,5 s:
  // a SOMA das piscadas passa de 0,8 s, mas nenhum fechamento contínuo chega a 0,4 s -> só "cabeça baixa".
  const h = new HeadDropDetector();
  h.setBaseline({ pitch: 0, yaw: 0, roll: 0 });
  let t = 0;
  const evs = [];
  let sustained = false;
  for (let i = 0; i < 30; i++, t += DT) h.update(t, { pose: { pitch: 0, yaw: 0, roll: 0 } });
  for (let k = 0; k < 6; k++) {
    for (let i = 0; i < 39; i++, t += DT) { const r = h.update(t, { pose: { pitch: 25, yaw: 0, roll: 0 }, eyesClosed: false }); if (r.event) evs.push(r.event.type); sustained ||= r.sustained; }
    for (let i = 0; i < 6; i++, t += DT) { const r = h.update(t, { pose: { pitch: 25, yaw: 0, roll: 0 }, eyesClosed: true }); if (r.event) evs.push(r.event.type); sustained ||= r.sustained; }
  }
  ok(evs.join(",") === "cabeca_baixa" && !sustained, `cabeça baixa 8 s com 6 piscadas comuns: só 'cabeça baixa', sem vermelho (${evs.join(",") || "nenhum"})`);
  // Nível: alvo oscilando em torno de 70 não faz o estado piscar; depois de um vermelho forçado volta ao verde.
  const L = new LevelController();
  let changes = 0;
  let r = null;
  for (let i = 0; i < 300; i++) { r = L.update(i * DT, i % 2 ? 72 : 66); if (r.changed) changes++; }
  ok(r.state === 2 && changes <= 2, `alvo alternando 66/72 por 10 s: ${changes} troca(s) de estado`);
  const L2 = new LevelController();
  L2.update(0, 0, true);
  let tGreen = null;
  for (let tt = 0; tt <= 60000; tt += DT) { const x = L2.update(tt, 0, false); if (x.state === 0 && tGreen === null) tGreen = tt; }
  ok(tGreen !== null && tGreen < 20000 && L2.level === 0, `vermelho forçado que acaba (alvo 0): volta ao ALERTA em ${(tGreen / 1000).toFixed(1)} s e nível 0`);
}

console.log("[25] Ajuste final: PERCLOS sem piscadas, LSTM só em ATENÇÃO, saída do vermelho forçado, rosto perdido, calibração");
{
  // PERCLOS: só fechamentos >= 0,5 s entram (inteiros); piscadas ficam só no valueAll informativo.
  const p = new PerclosTime(CFG.perclos);
  let t = 0;
  for (let s = 0; s < 70; s++) for (let i = 0; i < 30; i++, t += DT) p.update(t, i < 3); // 1 piscada de 0,1 s por segundo (60/min)
  ok(p.value === 0 && near(p.valueAll, 0.1, 0.01), `60 piscadas/min de 0,1 s: PERCLOS ${(100 * p.value).toFixed(1)} % (com as piscadas seria ${(100 * p.valueAll).toFixed(1)} %)`);
  const q = new PerclosTime(CFG.perclos);
  t = 0;
  for (let i = 0; i < 30 * 70; i++, t += DT) q.update(t, false);
  for (let i = 0; i < 18; i++, t += DT) q.update(t, true); // 0,6 s fechado
  q.update(t, false);
  ok(near(q.value, 0.6 / 60, 0.001), `fechamento de 0,6 s conta inteiro, desde o 1º quadro: ${(100 * q.value).toFixed(2)} % (0,6 s / 60 s)`);
  for (let i = 0; i < 13; i++, t += DT) q.update(t, true); // 0,43 s fechado
  q.update(t, false);
  ok(near(q.value, 0.6 / 60, 0.001), "fechamento de 0,43 s (piscada longa) não entra");
  for (let i = 0; i < 12; i++, t += DT) q.update(t, true);
  q.faceLost(); // rosto some no meio: 0,37 s < 0,5 s, descartado
  for (let i = 0; i < 20; i++, t += DT) q.update(t, true);
  q.faceLost(); // 0,63 s antes de o rosto sumir: entra
  ok(near(q.value, (0.6 + 19 * DT / 1000) / 60, 0.001), `rosto perdido encerra o fechamento: 0,37 s descartado, 0,63 s contado (${(100 * q.value).toFixed(2)} %)`);
  const old = new PerclosTime({ ...CFG.perclos, minEpisodeMs: 0 });
  t = 0;
  for (let i = 0; i < 30 * 60; i++, t += DT) old.update(t, i % 30 < 3);
  ok(near(old.value, 0.1, 0.01), "com minEpisodeMs = 0 volta ao PERCLOS antigo (todo quadro fechado conta)");

  // Monitor: quem pisca 75 vezes por minuto (piscadas de 0,1 s) continua VERDE (UTA-RLDD Fold3/31, vídeo alerta).
  const mon = new DrowsinessMonitor();
  const fr = (closed) => ({ face: true, ear: { left: closed ? 0.08 : 0.3, right: closed ? 0.08 : 0.3, mean: closed ? 0.08 : 0.3 }, mar: 0.05,
    blend: { eyeBlinkLeft: closed ? 0.7 : 0.1, eyeBlinkRight: closed ? 0.7 : 0.1, jawOpen: 0.02, eyeLookDownLeft: 0.2, eyeLookDownRight: 0.2, eyeSquintLeft: 0.1, eyeSquintRight: 0.1 },
    pose: { pitch: -5, yaw: 0, roll: 0 }, mlProbs: null });
  t = 0;
  mon.startCalibration(0);
  for (let i = 0; i < 310; i++, t += DT) mon.update(t, fr(false));
  let maxLv = 0, maxSt = 0, lastR = null;
  for (let i = 0; i < 30 * 180; i++, t += DT) { lastR = mon.update(t, fr(i % 24 < 3)); maxLv = Math.max(maxLv, lastR.score.level); maxSt = Math.max(maxSt, lastR.score.state); }
  ok(maxSt === 0 && maxLv < 1 && lastR.eyes.perclosAll > 0.1, `75 piscadas/min por 3 min: sempre ALERTA, nível máx. ${maxLv.toFixed(1)} (PERCLOS com piscadas seria ${(100 * lastR.eyes.perclosAll).toFixed(1)} %)`);
  // ... e mesmo com o LSTM dizendo "sonolento" o tempo todo, continua verde (o +10 só vale com >= 40 dos sinais visíveis).
  for (let i = 0; i < 30 * 60; i++, t += DT) { const f = fr(i % 24 < 3); if (i % 60 === 0) f.mlProbs = [0.05, 0.05, 0.9]; lastR = mon.update(t, f); maxSt = Math.max(maxSt, lastR.score.state); }
  ok(maxSt === 0 && lastR.score.mlCounted === false && mon.ml.eligible, "piscando muito + LSTM 'sonolento' por 1 min: continua ALERTA (modelo elegível, mas não soma)");

  // Nível: a margem de saída do vermelho (60) vale para o vermelho que entrou pelo nível, não para o forçado.
  const Lf = new LevelController();
  Lf.update(0, 65, true);
  let x = null;
  let tYellow = null;
  for (let tt = DT; tt < 5000; tt += DT) { x = Lf.update(tt, 65, false); if (x.state === 1 && tYellow === null) tYellow = tt; }
  ok(x.state === 1 && near(x.level, 65, 1e-6) && tYellow < 2000, `forçado termina com o alvo em 65: ATENÇÃO em ${(tYellow / 1000).toFixed(1)} s (o nível desce de 75 a 3 pt/s e o limiar de saída é o de entrada, 70)`);
  const Ll = new LevelController();
  Ll.update(0, 72);
  x = Ll.update(1000, 65);
  ok(x.state === 2 && Ll.redByLevel, "vermelho que entrou pelo nível (72) e cai para 65: continua vermelho (sai só abaixo de 60)");
  const Lx = new LevelController();
  Lx.update(0, 50, true);
  x = Lx.update(100, 75, false);
  const stillRed = x.state === 2 && !Lx.redByLevel;
  x = Lx.update(3100, 68, false);
  ok(stillRed && x.state === 1 && near(x.level, 68, 1e-6), "forçado termina com o alvo em 75: continua vermelho (>= 70); o alvo cai para 68: ATENÇÃO sem esperar os 60");

  // Cabeça: queda pela inclinação lateral informa o ângulo lateral; a queda para a frente, o ângulo para baixo.
  const h = new HeadDropDetector();
  h.setBaseline({ pitch: 0, yaw: 0, roll: 0 });
  t = 0;
  let ev = null;
  for (let i = 0; i < 30; i++, t += DT) h.update(t, { pose: { pitch: 0, yaw: 0, roll: 0 } });
  for (let i = 0; i < 45; i++, t += DT) { const o = h.update(t, { pose: { pitch: 2, yaw: 0, roll: 40 }, eyesClosed: true }); if (o.event) ev = o.event; }
  ok(ev && ev.type === "queda" && ev.byRoll && near(ev.peakRollDeg, 40, 1e-6) && ev.peakDeg < 5, `cabeça tombando 40° para o lado com olhos fechados: queda com byRoll e pico lateral ${ev && ev.peakRollDeg}°`);
  const h2 = new HeadDropDetector();
  h2.setBaseline({ pitch: 0, yaw: 0, roll: 0 });
  ev = null;
  t = 0;
  for (let i = 0; i < 30; i++, t += DT) h2.update(t, { pose: { pitch: 0, yaw: 0, roll: 0 } });
  for (let i = 0; i < 45; i++, t += DT) { const o = h2.update(t, { pose: { pitch: 28, yaw: 0, roll: 3 }, eyesClosed: true }); if (o.event) ev = o.event; }
  ok(ev && ev.type === "queda" && !ev.byRoll && near(ev.peakDeg, 28, 1e-6), "cabeça caindo 28° para a frente: queda sem byRoll, pico 28° para baixo");
  // Com um episódio aberto (cabeça 20° abaixo, olhos abertos), o rosto sumir não vira queda nem vermelho.
  const h3 = new HeadDropDetector();
  h3.setBaseline({ pitch: 0, yaw: 0, roll: 0 });
  t = 0;
  let evs3 = [];
  let sus3 = false;
  for (let i = 0; i < 30; i++, t += DT) h3.update(t, { pose: { pitch: 0, yaw: 0, roll: 0 } });
  for (let i = 0; i < 20; i++, t += DT) { const o = h3.update(t, { pose: { pitch: 20, yaw: 0, roll: 0 } }); if (o.event) evs3.push(o.event.type); }
  for (let i = 0; i < 90; i++, t += DT) { const o = h3.update(t, { face: false }); if (o.event) evs3.push(o.event.type); sus3 ||= o.sustained; }
  ok(!evs3.includes("queda") && !sus3, `cabeça 20° abaixo com olhos abertos e depois o rosto some por 3 s (abaixou-se): ${evs3.join(",") || "nenhum evento"}, sem vermelho`);

  // Calibração: aviso quando os olhos ficam fechados em mais de 10 % dos quadros.
  const mc = new DrowsinessMonitor();
  t = 0;
  mc.startCalibration(0);
  let calEv = null;
  for (let i = 0; i < 320 && !calEv; i++, t += DT) { const r = mc.update(t, fr(i % 6 === 0)); calEv = r.events.find((e) => e.type === "calibrado"); }
  ok(calEv && calEv.baseline.closedFrac > 0.1 && calEv.baseline.warnings.some((w) => /Olhos fechados em/.test(w)), `calibração com olhos fechados em ${(100 * calEv.baseline.closedFrac).toFixed(0)} % dos quadros: aviso para recalibrar`);
  const mo = new DrowsinessMonitor();
  t = 0;
  mo.startCalibration(0);
  calEv = null;
  for (let i = 0; i < 320 && !calEv; i++, t += DT) { const r = mo.update(t, fr(i % 90 < 5)); calEv = r.events.find((e) => e.type === "calibrado"); }
  ok(calEv && calEv.baseline.closedFrac < 0.1 && !calEv.baseline.warnings.some((w) => /Olhos fechados em/.test(w)), `calibração normal (olho fechado em ${(100 * calEv.baseline.closedFrac).toFixed(0)} %): sem aviso`);

  // Roteiro da banca no pior caso: bocejos a 5 s um do outro e o 2º só 3 s antes de fechar os olhos por 2 s.
  {
    const mr = new DrowsinessMonitor();
    let tt = 0, rr = null;
    const frd = ({ closed = false, jaw = 0.02 } = {}) => ({ ...fr(closed), mar: jaw > 0.3 ? 0.7 : 0.05, blend: { ...fr(closed).blend, jawOpen: jaw } });
    const run = (ms, f) => { for (let i = 0, n = Math.round(ms / DT); i < n; i++, tt += DT) rr = mr.update(tt, f); };
    const blinks = (s) => { for (let k = 0; k < s / 3; k++) { run(2800, frd()); run(200, frd({ closed: true })); } };
    mr.startCalibration(0);
    run(10000, frd());
    blinks(15);
    run(2500, frd({ jaw: 0.8 })); blinks(5); run(2500, frd({ jaw: 0.8 })); blinks(3);
    run(2000, frd({ closed: true }));
    const tOpen = tt;
    let back = null;
    for (let i = 0; i < 30 * 60 && back === null; i++, tt += DT) { rr = mr.update(tt, frd({ closed: i % 90 >= 84 })); if (rr.score.state < 2 && !rr.alarm.siren) back = tt; }
    ok(back !== null && back - tOpen <= 10000, `roteiro da banca, pior caso (2º bocejo 3 s antes de fechar os olhos): o vermelho termina ${((back - tOpen) / 1000).toFixed(1)} s depois de reabrir (<= 10 s)`);
  }

  // Histerese dos olhos: opção existe (reopenHyst), mas fica DESLIGADA no app (piorou a direção real no DMD s1).
  ok(CFG.eyes.reopenHyst === 0 && !fuseEyeClosure({ ear: 0.3, earThr: 0.2, blinkL: 0.45, blinkR: 0.45, prevClosed: true }).closed, "histerese desligada no app: com o olho 'fechado' antes, eyeBlink 0,45 já reabre");
  ok(fuseEyeClosure({ ear: 0.3, earThr: 0.2, blinkL: 0.45, blinkR: 0.45, prevClosed: true }, { ...CFG.eyes, reopenHyst: 0.1 }).closed, "com reopenHyst = 0,1 (só para avaliação), 0,45 manteria fechado");
}

console.log(`\n${passed} passaram, ${failed} falharam`);
process.exit(failed ? 1 : 0);
