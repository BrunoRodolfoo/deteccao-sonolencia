#!/usr/bin/env python3
"""Extrai, quadro a quadro, os MESMOS sinais que a PWA (app/pwa) usa no detector v2, para validar o
motor (app/pwa/engine.js) contra a anotação do DMD sem precisar de câmera.

Rodar no .venv-mp (mediapipe 0.10.21; o 1.0.1 aborta neste Mac):

    .venv-mp/bin/python tools/app_validacao/extrair_sinais_app.py --sessoes s5 s1 --proc 2

Modelo: face_landmarker.task float16 v1 — o MESMO arquivo que a PWA baixa de
storage.googleapis.com (md5 b0e7274907a1644404fef66b28dd6d85, conferido em 27/09/2026). Opções iguais às
da PWA: 1 rosto, blendshapes e matrizes de transformação ligados, confianças padrão (0,5), modo VIDEO.

Saída por vídeo (fora do Git: data/** é ignorado): data/signals/app_tasks/dmd/<id>.csv.gz com as colunas
  frame, t_ms, face, ear_l, ear_r, mar, jaw_open, blink_l, blink_r, look_down_l, look_down_r,
  squint_l, squint_r, m0..m15 (matriz 4x4 na MESMA ordem que a PWA recebe: packed_data do proto,
  coluna a coluna), nose_y, chin_y, eye_y, forehead_z, chin_z (marcos 1, 152, média de 33/263, 10;
  usados só para conferir o sinal do ângulo de inclinação da cabeça).
Os índices de EAR/MAR são os de app/pwa/engine.js (LEFT_EYE, RIGHT_EYE, MOUTH_*), em pixels.
Nenhum quadro de vídeo é gravado.

UTA-RLDD (teste independente do app, tools/app_validacao/validar_motor_uta.mjs): os mesmos sinais, com as
colunas acima, em data/signals/app_tasks/uta/<F<fold>_<sujeito>_<rótulo>>.csv.gz (+ .json):

    .venv-mp/bin/python tools/app_validacao/extrair_sinais_app.py --uta-folds 2 4 --uta-rotulos 0 10 --proc 6

Só nesse modo, como na cadeia offline (src/features/extract_signals.py): a rotação do metadado do celular
é aplicada (CAP_PROP_ORIENTATION_AUTO), o quadro é reduzido para no máximo 1280 px no lado maior (a mesma
resolução dos vídeos do DMD) e t_ms é o carimbo do contêiner, forçado a ser não decrescente e a começar
em 0 (há vídeos de fps variável na UTA-RLDD). Os vídeos maiores são processados primeiro. O modo DMD não muda.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import gzip
import json
import logging
import multiprocessing as mp_
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DMD = ROOT / "data" / "dmd"
OUT = ROOT / "data" / "signals" / "app_tasks" / "dmd"
UTA = ROOT / "data" / "uta_rldd_videos"
OUT_UTA = ROOT / "data" / "signals" / "app_tasks" / "uta"
UTA_MAX_SIDE = 1280  # lado maior do quadro processado na UTA-RLDD (igual a MAX_SIDE de src/features/extract_signals.py)
LOGS = ROOT / "results" / "logs"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_MD5 = "b0e7274907a1644404fef66b28dd6d85"

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_CORNERS = [61, 291]
MOUTH_VERTICAL = [13, 14]
BLENDS = {"eyeBlinkLeft": "blink_l", "eyeBlinkRight": "blink_r", "jawOpen": "jaw_open",
          "eyeLookDownLeft": "look_down_l", "eyeLookDownRight": "look_down_r",
          "eyeSquintLeft": "squint_l", "eyeSquintRight": "squint_r"}
COLS = (["frame", "t_ms", "face", "ear_l", "ear_r", "mar"] + list(BLENDS.values())
        + [f"m{i}" for i in range(16)] + ["nose_y", "chin_y", "eye_y", "forehead_z", "chin_z"])


def model_path() -> str:
    """Baixa (uma vez) o face_landmarker.task para data/cache e confere o md5."""
    import hashlib
    import urllib.request
    dst = ROOT / "data" / "cache" / "face_landmarker_float16_v1.task"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, dst)
    md5 = hashlib.md5(dst.read_bytes()).hexdigest()
    if md5 != MODEL_MD5:
        raise SystemExit(f"md5 inesperado para {dst}: {md5}")
    return str(dst)


def ear(pts):
    import math
    d = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    h = d(pts[0], pts[3])
    return 0.0 if h < 1e-6 else (d(pts[1], pts[5]) + d(pts[2], pts[4])) / (2 * h)


def videos(sessoes):
    out = []
    for s in sessoes:
        for ann in sorted(glob.glob(str(DMD / "*" / "*" / s / "*_rgb_ann_*.json"))):
            prefix = ann.split("_rgb_ann_")[0]
            mp4 = prefix + "_rgb_face.mp4"
            if os.path.exists(mp4):
                out.append((Path(prefix).name, mp4, ann))
    return out


def videos_uta(folds, rotulos, ids=None):
    """Vídeos da UTA-RLDD dos folds e rótulos pedidos, com o id da cadeia offline: F<fold>_<sujeito>_<rótulo>.
    Os vídeos divididos em partes (10_1, 10_2: Fold3/32 e Fold5/49) não são aceitos aqui."""
    out = []
    for f in folds:
        for d in sorted(glob.glob(str(UTA / f"Fold{f}" / "*"))):
            if not os.path.isdir(d):
                continue
            for lab in rotulos:
                vid = f"F{f}_{Path(d).name}_{lab}"
                if ids and vid not in ids:
                    continue
                cands = [p for p in glob.glob(os.path.join(d, f"{lab}.*")) if Path(p).suffix.lower() in (".mp4", ".mov", ".m4v", ".avi")]
                if len(cands) != 1:
                    raise SystemExit(f"{vid}: esperado 1 vídeo '{lab}.*' em {os.path.relpath(d, ROOT)}, achei {len(cands)}")
                out.append((vid, cands[0], ""))
    return out


def processar(args):
    vid, mp4, ann, model = args[:4]
    # Opções do modo UTA-RLDD; sem elas (DMD), o comportamento é o original.
    uta = args[4] if len(args) > 4 else None
    out_dir = OUT_UTA if uta else OUT
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision

    cv2.setNumThreads(1)
    dst = out_dir / f"{vid}.csv.gz"
    if dst.exists() and (out_dir / f"{vid}.json").exists():
        return vid, "já existia", 0, 0.0
    opts = vision.FaceLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=model, delegate=mpt.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True)
    lm = vision.FaceLandmarker.create_from_options(opts)
    cap = cv2.VideoCapture(mp4)
    if uta:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # rotação do metadado (celular), como na cadeia offline
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w0, h0 = w, h
    rot = float(cap.get(cv2.CAP_PROP_ORIENTATION_META)) if uta else 0.0
    scale = 1.0
    pts0 = None
    t_prev = 0.0
    ts_prev = -1
    sem_carimbo = 0
    t_dec = t_mod = 0.0
    t0 = time.time()
    n = faces = 0
    tmp = dst.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(COLS)
        while True:
            ta = time.time()
            ok, fr = cap.read()
            if not ok:
                break
            if uta:
                if n == 0:  # dimensões depois da rotação; redução para no máximo UTA_MAX_SIDE no lado maior
                    hh, ww = fr.shape[:2]
                    scale = min(1.0, UTA_MAX_SIDE / max(hh, ww))
                    w, h = int(round(ww * scale)), int(round(hh * scale))
                if scale < 1.0:
                    fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
                pts = cap.get(cv2.CAP_PROP_POS_MSEC)
                if pts0 is None and pts == pts and pts >= 0:
                    pts0 = pts
                if pts0 is not None and pts == pts and pts >= 0:
                    t_ms = max(t_prev, pts - pts0)  # carimbo do contêiner, não decrescente, a partir de 0
                else:
                    t_ms = max(t_prev, n * 1000.0 / fps)
                    sem_carimbo += 1
                t_prev = t_ms
            else:
                t_ms = n * 1000.0 / fps
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            tb = time.time()
            t_dec += tb - ta
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(round(t_ms))
            if uta:
                ts = max(ts, ts_prev + 1)  # o modo VIDEO exige carimbos estritamente crescentes
                ts_prev = ts
            r = lm.detect_for_video(img, ts)
            t_mod += time.time() - tb
            row = [n, f"{t_ms:.1f}"]
            if r.face_landmarks:
                faces += 1
                L = r.face_landmarks[0]
                P = lambda idx: [(L[i].x * w, L[i].y * h) for i in idx]
                el, er = ear(P(LEFT_EYE)), ear(P(RIGHT_EYE))
                c0, c1 = P(MOUTH_CORNERS)
                v0, v1 = P(MOUTH_VERTICAL)
                hz = ((c0[0] - c1[0]) ** 2 + (c0[1] - c1[1]) ** 2) ** 0.5
                mar = 0.0 if hz < 1e-6 else ((v0[0] - v1[0]) ** 2 + (v0[1] - v1[1]) ** 2) ** 0.5 / hz
                bs = {c.category_name: c.score for c in r.face_blendshapes[0]} if r.face_blendshapes else {}
                M = r.facial_transformation_matrixes[0] if r.facial_transformation_matrixes else None
                # A API Python devolve a matriz "de verdade" (numpy 4x4, translação na última coluna);
                # a PWA recebe o packed_data do proto, que é coluna a coluna -> M.T.flatten().
                packed = [f"{v:.5f}" for v in M.T.flatten()] if M is not None else [""] * 16
                eye_y = (L[33].y + L[263].y) / 2 * h
                row += [1, f"{el:.4f}", f"{er:.4f}", f"{mar:.4f}"]
                row += [f"{bs[k]:.4f}" if k in bs else "" for k in BLENDS]
                row += packed
                row += [f"{L[1].y * h:.2f}", f"{L[152].y * h:.2f}", f"{eye_y:.2f}",
                        f"{L[10].z * w:.2f}", f"{L[152].z * w:.2f}"]
            else:
                row += [0] + [""] * (len(COLS) - 3)
            wr.writerow(row)
            n += 1
    cap.release()
    lm.close()
    os.replace(tmp, dst)
    secs = time.time() - t0
    meta = {"id": vid, "video": os.path.relpath(mp4, ROOT), "anotacao": os.path.relpath(ann, ROOT) if ann else "", "fps": fps,
            "largura": w, "altura": h, "quadros": n, "quadros_com_rosto": faces, "segundos_proc": round(secs, 1),
            "modelo": "face_landmarker.task float16 v1 (md5 %s)" % MODEL_MD5, "mediapipe": mp.__version__}
    if uta:
        meta.update({"largura_original": w0, "altura_original": h0, "rotacao_meta": rot, "escala": round(scale, 4),
                     "tempo": "carimbo do contêiner (CAP_PROP_POS_MSEC), não decrescente, a partir de 0",
                     "quadros_sem_carimbo": sem_carimbo, "duracao_s": round(t_prev / 1000.0, 3),
                     "segundos_leitura": round(t_dec, 1), "segundos_modelo": round(t_mod, 1)})
    (out_dir / f"{vid}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    return vid, f"{n} quadros, rosto {faces / max(n, 1):.4f}", n, secs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessoes", nargs="+", default=["s5", "s1"])
    ap.add_argument("--proc", type=int, default=2, help="processos em paralelo (máx. 2 enquanto outros experimentos rodam)")
    ap.add_argument("--limite", type=int, default=0, help="processa só os N primeiros vídeos (teste)")
    ap.add_argument("--uta-folds", nargs="+", type=int, default=None, help="modo UTA-RLDD: folds oficiais a processar")
    ap.add_argument("--uta-rotulos", nargs="+", default=["0", "10"], help="modo UTA-RLDD: vídeos (0, 5, 10)")
    ap.add_argument("--uta-ids", nargs="+", default=None, help="modo UTA-RLDD: só estes ids (ex.: F2_13_0)")
    ap.add_argument("--log", default=None, help="arquivo de log (acrescenta); padrão results/logs/app_extrair_sinais_<data>.log")
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log = Path(args.log) if args.log else LOGS / f"app_extrair_sinais_{stamp}.log"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(log), logging.StreamHandler(sys.stdout)])
    model = model_path()
    uta = None
    if args.uta_folds:
        import cv2
        OUT_UTA.mkdir(parents=True, exist_ok=True)
        uta = {"max_side": UTA_MAX_SIDE}
        todos = videos_uta(args.uta_folds, args.uta_rotulos, set(args.uta_ids) if args.uta_ids else None)

        def custo(item):  # pixels x quadros: os vídeos mais caros entram primeiro no pool
            c = cv2.VideoCapture(item[1])
            v = c.get(cv2.CAP_PROP_FRAME_WIDTH) * c.get(cv2.CAP_PROP_FRAME_HEIGHT) * c.get(cv2.CAP_PROP_FRAME_COUNT)
            c.release()
            return v
        todos.sort(key=custo, reverse=True)
        origem = f"UTA-RLDD folds {args.uta_folds}, vídeos {args.uta_rotulos}"
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        todos = videos(args.sessoes)
        origem = f"sessões {args.sessoes}"
    if args.limite:
        todos = todos[: args.limite]
    logging.info("extração de sinais do app (Face Landmarker Tasks, blendshapes + matrizes): %d vídeos, %s, %d processos",
                 len(todos), origem, args.proc)
    t0 = time.time()
    tot = 0
    with mp_.get_context("spawn").Pool(args.proc) as pool:
        tarefas = [(v, m, a, model, uta) if uta else (v, m, a, model) for v, m, a in todos]
        for vid, msg, n, secs in pool.imap_unordered(processar, tarefas):
            tot += n
            logging.info("%s: %s (%.0f s, %.0f q/s)", vid, msg, secs, n / secs if secs else 0)
    logging.info("fim: %d quadros novos em %.1f min", tot, (time.time() - t0) / 60)


if __name__ == "__main__":
    main()
