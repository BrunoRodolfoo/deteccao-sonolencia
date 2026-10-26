"""Extração de sinais por quadro (vídeo -> EAR/MAR/pose) com MediaPipe Face Mesh.

Roda SOMENTE no ambiente .venv-mp (mediapipe 0.10.21 + opencv 4.11), pois o
mediapipe 1.x aborta neste Mac e o 0.10.21 é incompatível com o TensorFlow do .venv.

Uso:
    python src/features/extract_signals.py --dataset {dmd,nitymed,uta} [--workers 6] [--limit N]
                                           [--only SUBSTR] [--force]

Saídas:
    data/signals/<dataset>/<id>.npz   (um por vídeo, comprimido)
    data/signals/<dataset>/index.csv  (uma linha por vídeo; reconstruído a partir de todos os .npz)

Arrays por quadro em cada .npz (comprimento = nº de quadros lidos):
    frame (int32)      índice do quadro (0..n-1), na ordem de decodificação
    t_s (float64)      frame / fps (segundos)
    t_pts_s (float64)  carimbo de tempo do contêiner (CAP_PROP_POS_MSEC/1000); útil em vídeos de fps variável
    face (bool)        Face Mesh encontrou rosto no quadro
    ear_l, ear_r, ear  EAR do olho esquerdo, direito e média (NaN sem rosto)
    mar                MAR = |13-14| / |61-291| (NaN sem rosto)
    roll_deg           inclinação lateral da cabeça (graus), ângulo da reta canto externo 33 -> 263
    yaw_proxy          desvio horizontal da ponta do nariz (1) em relação ao ponto médio dos cantos
                       externos, dividido pela distância interocular (≈0 de frente; sinal = lado)
    pitch_proxy        distância vertical (olhos->nariz) / (olhos->queixo 152); cresce ao abaixar a cabeça
    lm (float32, n x 21 x 2)  coordenadas em pixels (quadro processado) dos pontos em LM_IDX
Metadados escalares em chaves "meta_*" (ver META_FIELDS) e "lm_idx".

Reinicialização: um .npz existente é pulado se abrir, tiver todas as chaves e
meta_complete == True. A escrita é atômica (arquivo temporário + rename).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import multiprocessing as mp_
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT_BASE = DATA / "signals"

# Índices MediaPipe (iguais a src/realtime/metrics.py), ordem p1..p6.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH_CORNERS = (61, 291)
MOUTH_VERTICAL = (13, 14)
NOSE_TIP, CHIN, FOREHEAD = 1, 152, 10
FACE_SIDES = (234, 454)
LM_IDX = LEFT_EYE + RIGHT_EYE + list(MOUTH_CORNERS) + list(MOUTH_VERTICAL) + [NOSE_TIP, CHIN, FOREHEAD] + list(FACE_SIDES)
assert len(LM_IDX) == 21

MAX_SIDE = 1280  # lado maior do quadro processado (reduz vídeos 1080p/1440p)
ARRAY_KEYS = ["frame", "t_s", "t_pts_s", "face", "ear_l", "ear_r", "ear", "mar",
              "roll_deg", "yaw_proxy", "pitch_proxy", "lm"]
META_FIELDS = ["id", "dataset", "subject", "session", "label", "path", "fps", "n_frames_meta",
               "n_frames", "width_orig", "height_orig", "rotation_meta", "width_proc",
               "height_proc", "scale", "face_rate", "proc_fps", "proc_seconds", "extra_json",
               "complete"]


# --------------------------------------------------------------------------------------
# Descoberta de vídeos por base
# --------------------------------------------------------------------------------------
def list_dmd():
    """Vídeos da câmera do rosto do DMD: sessão s5 (sonolência) e s1-s4 (distração).

    Ids: s5 mantém "<grupo>_<sujeito>_<timestamp>" (compatível com os sinais já extraídos);
    as sessões de distração ganham a sessão no id: "<grupo>_<sujeito>_<sN>_<timestamp>".
    extra["sessao"] e extra["pacote"] ("drowsiness"/"distraction") identificam a origem.
    """
    items = []
    for v in sorted(glob.glob(str(DATA / "dmd" / "*" / "*" / "s[1-5]" / "*_rgb_face.mp4"))):
        p = Path(v)
        grupo, sujeito, sess = p.parts[-4], p.parts[-3], p.parts[-2]
        ts = p.name.split(f"_{sess}_")[1].replace("_rgb_face.mp4", "")
        anns = sorted(glob.glob(v.replace("_rgb_face.mp4", "_rgb_ann_*.json")))
        ann = anns[0] if anns else ""
        pacote = "drowsiness" if sess == "s5" else "distraction"
        extra = {"annotation": os.path.relpath(ann, ROOT) if ann else "", "sessao": sess, "pacote": pacote}
        if ann:
            try:
                d = json.load(open(ann))["openlabel"]
                obj = next(iter(d.get("objects", {}).values()), {})
                od = obj.get("object_data", {})
                for kind in ("num", "text", "boolean"):
                    for e in od.get(kind, []):
                        extra[e["name"]] = e["val"]
                ctx = next(iter(d.get("contexts", {}).values()), {})
                for e in ctx.get("context_data", {}).get("text", []):
                    extra[e["name"]] = e["val"]
                fc = d["streams"]["face_camera"]["stream_properties"]
                extra["ann_face_total_frames"] = fc.get("total_frames")
                extra["ann_face_frame_shift"] = fc.get("sync", {}).get("frame_shift")
            except Exception as exc:  # anotação ilegível não impede a extração
                extra["annotation_error"] = repr(exc)
        vid = f"{grupo}_{sujeito}_{ts}" if sess == "s5" else f"{grupo}_{sujeito}_{sess}_{ts}"
        items.append(dict(id=vid, dataset="dmd", subject=f"{grupo}_{sujeito}",
                          session=ts, label="", path=v, extra=extra))
    return items


def list_nitymed():
    items = []
    for v in sorted(glob.glob(str(DATA / "nitymed" / "DSM_Dataset-HDTV720" / "*" / "*" / "HDTV720" / "*.mp4"))):
        p = Path(v)
        cat, gen = p.parts[-4], p.parts[-3]
        vid = f"{cat}_{gen}_{p.stem}"
        items.append(dict(id=vid, dataset="nitymed", subject=vid, session=p.stem, label=cat, path=v,
                          extra={"gender": gen, "category": cat}))
    return items


def uta_folds():
    """Folds da UTA-RLDD prontos (possuem FoldN/.ok)."""
    folds = []
    for ok in sorted(glob.glob(str(DATA / "uta_rldd_videos" / "Fold*" / ".ok"))):
        name = Path(ok).parent.name
        try:
            folds.append(int(name.replace("Fold", "")))
        except ValueError:
            pass
    return sorted(folds)


def list_uta():
    items = []
    for n in uta_folds():
        for v in sorted(glob.glob(str(DATA / "uta_rldd_videos" / f"Fold{n}" / "*" / "*"))):
            p = Path(v)
            if p.suffix.lower() not in (".mp4", ".mov", ".m4v", ".avi"):
                continue
            # Nomes: "0", "5", "10" e, em dois sujeitos (Fold3/32 e Fold5/49), o vídeo sonolento
            # dividido em "10_1" e "10_2". Cada parte vira um arquivo de sinais próprio (nenhuma
            # janela atravessa o corte), mas as duas compartilham o mesmo vídeo lógico ("video").
            label, _, part = p.stem.partition("_")
            if label not in ("0", "5", "10"):
                logging.warning("UTA: arquivo ignorado (nome fora do padrão): %s", v)
                continue
            suj = p.parent.name
            video = f"F{n}_{suj}_{label}"
            items.append(dict(id=video + (f"_p{part}" if part else ""), dataset="uta", subject=f"F{n}_{suj}",
                              session=label, label=label, path=v,
                              extra={"fold": n, "folder_subject": suj, "file": p.name,
                                     "video": video, "part": part}))
    return items


LISTERS = {"dmd": list_dmd, "nitymed": list_nitymed, "uta": list_uta}


# --------------------------------------------------------------------------------------
# Medidas
# --------------------------------------------------------------------------------------
def _ear(P, idx):
    p1, p2, p3, p4, p5, p6 = (P[i] for i in idx)
    h = np.linalg.norm(p1 - p4)
    if h < 1e-6:
        return np.nan
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * h)


def measures_from_points(P):
    """P: dict índice -> np.array([x_px, y_px]). Retorna (ear_l, ear_r, mar, roll, yaw, pitch)."""
    ear_l = _ear(P, LEFT_EYE)
    ear_r = _ear(P, RIGHT_EYE)
    mh = np.linalg.norm(P[MOUTH_CORNERS[0]] - P[MOUTH_CORNERS[1]])
    mar = np.linalg.norm(P[MOUTH_VERTICAL[0]] - P[MOUTH_VERTICAL[1]]) / mh if mh > 1e-6 else np.nan
    a, b = P[33], P[263]  # canto externo do olho direito e esquerdo (da pessoa)
    d = b - a
    roll = float(np.degrees(np.arctan2(d[1], d[0])))
    io = np.linalg.norm(d)
    mid = (a + b) / 2.0
    if io > 1e-6:
        u = d / io                       # eixo "horizontal" do rosto
        n = np.array([-u[1], u[0]])      # eixo "vertical" do rosto (para baixo)
        nose = P[NOSE_TIP] - mid
        yaw = float(np.dot(nose, u) / io)
        chin = float(np.dot(P[CHIN] - mid, n))
        pitch = float(np.dot(nose, n) / chin) if abs(chin) > 1e-6 else np.nan
    else:
        yaw = pitch = np.nan
    return ear_l, ear_r, mar, roll, yaw, pitch


# --------------------------------------------------------------------------------------
# Processamento de um vídeo
# --------------------------------------------------------------------------------------
def npz_valid(path):
    try:
        with np.load(path, allow_pickle=False) as z:
            if not all(k in z.files for k in ARRAY_KEYS):
                return False
            if "meta_complete" not in z.files or not bool(z["meta_complete"]):
                return False
            return len(z["frame"]) == int(z["meta_n_frames"])
    except Exception:
        return False


def process_video(item, out_dir, limit=None):
    import cv2
    import mediapipe as mp

    cv2.setNumThreads(1)
    out = Path(out_dir) / f"{item['id']}.npz"
    t0 = time.time()
    cap = cv2.VideoCapture(item["path"])
    if not cap.isOpened():
        return {"id": item["id"], "error": "não abriu"}
    # Aplica a rotação do metadado (iPhone/Android). Nesta build o padrão reportado é 0,
    # por isso é ligado explicitamente (verificado visualmente nos 36 vídeos do Fold 1).
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_meta = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rot = float(cap.get(cv2.CAP_PROP_ORIENTATION_META))

    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                         refine_landmarks=False, min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
    rows_pts, face, pts_ms = [], [], []
    meas = []
    wp = hp = None
    scale = 1.0
    i = 0
    while True:
        if limit is not None and i >= limit:
            break
        ok, frame = cap.read()
        if not ok:
            break
        pts_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        h, w = frame.shape[:2]
        if wp is None:
            scale = min(1.0, MAX_SIDE / max(h, w))
            wp, hp = int(round(w * scale)), int(round(h * scale))
        if scale < 1.0:
            frame = cv2.resize(frame, (wp, hp), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = fm.process(rgb)
        if res.multi_face_landmarks:
            lms = res.multi_face_landmarks[0].landmark
            P = {k: np.array([lms[k].x * wp, lms[k].y * hp]) for k in set(LM_IDX + [33, 263])}
            meas.append(measures_from_points(P))
            rows_pts.append(np.array([P[k] for k in LM_IDX], dtype=np.float32))
            face.append(True)
        else:
            meas.append((np.nan,) * 6)
            rows_pts.append(np.full((len(LM_IDX), 2), np.nan, np.float32))
            face.append(False)
        i += 1
    cap.release()
    fm.close()

    n = i
    M = np.array(meas, dtype=np.float64).reshape(n, 6)
    face = np.array(face, dtype=bool)
    dt = time.time() - t0
    frame_idx = np.arange(n, dtype=np.int32)
    ear = np.nanmean(M[:, :2], axis=1) if n else np.zeros(0)
    extra = dict(item.get("extra", {}))
    if limit is not None:
        extra["limit"] = limit
    meta = dict(id=item["id"], dataset=item["dataset"], subject=item["subject"],
                session=item["session"], label=str(item["label"]),
                path=os.path.relpath(item["path"], ROOT), fps=fps, n_frames_meta=n_meta,
                n_frames=n, width_orig=w0, height_orig=h0, rotation_meta=rot,
                width_proc=wp or 0, height_proc=hp or 0, scale=scale,
                face_rate=float(face.mean()) if n else 0.0, proc_fps=n / dt if dt > 0 else 0.0,
                proc_seconds=dt, extra_json=json.dumps(extra, ensure_ascii=False), complete=True)
    arrays = dict(frame=frame_idx, t_s=frame_idx / fps if fps > 0 else np.full(n, np.nan),
                  t_pts_s=np.array(pts_ms, dtype=np.float64) / 1000.0, face=face,
                  ear_l=M[:, 0], ear_r=M[:, 1], ear=ear, mar=M[:, 2], roll_deg=M[:, 3],
                  yaw_proxy=M[:, 4], pitch_proxy=M[:, 5],
                  lm=np.stack(rows_pts) if n else np.zeros((0, len(LM_IDX), 2), np.float32),
                  lm_idx=np.array(LM_IDX, dtype=np.int32))
    for k, v in meta.items():
        arrays[f"meta_{k}"] = np.array(v)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, out)
    return {k: meta[k] for k in ("id", "n_frames", "face_rate", "proc_fps", "proc_seconds")}


def _worker(args):
    item, out_dir, limit = args
    try:
        return process_video(item, out_dir, limit)
    except Exception as exc:  # não derruba o lote
        return {"id": item["id"], "error": repr(exc)}


# --------------------------------------------------------------------------------------
# Índice
# --------------------------------------------------------------------------------------
def build_index(out_dir):
    import pandas as pd
    rows = []
    for f in sorted(Path(out_dir).glob("*.npz")):
        if f.name.endswith(".tmp.npz"):
            continue
        try:
            with np.load(f, allow_pickle=False) as z:
                r = {k: z[f"meta_{k}"].item() for k in META_FIELDS if f"meta_{k}" in z.files}
                ear = z["ear"]
                r["ear_median"] = float(np.nanmedian(ear)) if np.isfinite(ear).any() else np.nan
                r["mar_median"] = float(np.nanmedian(z["mar"])) if np.isfinite(z["mar"]).any() else np.nan
        except Exception as exc:
            r = {"id": f.stem, "error": repr(exc)}
        r["npz"] = os.path.relpath(f, ROOT)
        rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["duration_s"] = df["n_frames"] / df["fps"]
    df.to_csv(Path(out_dir) / "index.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description="Extrai EAR/MAR/pose por quadro com MediaPipe Face Mesh.")
    ap.add_argument("--dataset", required=True, choices=sorted(LISTERS))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None,
                    help="processa só os N primeiros quadros (teste); grava em data/signals/_teste/")
    ap.add_argument("--max-videos", type=int, default=None, help="processa só os K primeiros vídeos")
    ap.add_argument("--only", default=None, help="filtra vídeos cujo id contém este texto")
    ap.add_argument("--force", action="store_true", help="reprocessa mesmo se o .npz for válido")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    items = LISTERS[args.dataset]()
    if args.dataset == "uta":
        logging.info("Folds da UTA-RLDD com .ok: %s", uta_folds())
    if args.only:
        items = [it for it in items if args.only in it["id"]]
    if args.max_videos:
        items = items[: args.max_videos]
    out_dir = (OUT_BASE / "_teste" / args.dataset) if args.limit else (OUT_BASE / args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [it for it in items if args.force or not npz_valid(out_dir / f"{it['id']}.npz")]
    logging.info("%s: %d vídeos encontrados, %d já prontos, %d a processar (workers=%d, limit=%s)",
                 args.dataset, len(items), len(items) - len(todo), len(todo), args.workers, args.limit)
    t0 = time.time()
    done_frames = 0
    errors = []
    if todo:
        ctx = mp_.get_context("spawn")
        with ctx.Pool(min(args.workers, len(todo)), maxtasksperchild=1) as pool:
            for k, r in enumerate(pool.imap_unordered(_worker, [(it, str(out_dir), args.limit) for it in todo]), 1):
                if "error" in r:
                    errors.append(r)
                    logging.error("[%d/%d] %s ERRO %s", k, len(todo), r["id"], r["error"])
                    continue
                done_frames += r["n_frames"]
                el = time.time() - t0
                logging.info("[%d/%d] %s quadros=%d rosto=%.3f %.0f q/s (vídeo) | total %.0f q/s agregado, %.1f min",
                             k, len(todo), r["id"], r["n_frames"], r["face_rate"], r["proc_fps"],
                             done_frames / el, el / 60)
    df = build_index(out_dir)
    logging.info("Índice: %s (%d linhas). Erros: %d. Tempo: %.1f min", out_dir / "index.csv", len(df),
                 len(errors), (time.time() - t0) / 60)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
