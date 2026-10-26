#!/usr/bin/env python3
"""Exp. 5a, parte 1d: o detector de rosto do dlib, com o pré-processamento dos autores, acha o rosto em cada vídeo?

Rodar no .venv-mp (OpenCV 4.11 + dlib-bin 20.0.1; o .venv não tem dlib):
    python src/experiments/exp5a_dlib_rosto.py [--folds 1 5] [--workers 4] [--trechos 12] [--quadros-por-trecho 10] [--force]

Motivo: as pastas F1_12 e F5_49 não estão nas características oficiais (rodada final: 5 folds). O código original (blink_video.py de Ghoddoosian et al.)
reduz cada quadro a 450 px de largura (imutils.resize, INTER_AREA), converte para cinza, aplica gama 1,5 e roda
dlib.get_frontal_face_detector() SEM ampliação (detector(gray, 0)). Esse detector HOG só acha rostos a partir de cerca
de 80 x 80 px. Se ele quase não acha o rosto no vídeo ALERTA, não há piscadas para a referência do z-score, e o sujeito
inteiro fica de fora. Este script repete exatamente esse pré-processamento em quadros amostrados de todos os vídeos
dos folds prontos (data/uta_rldd_videos/Fold*/.ok). Mede a taxa de detecção do dlib sem ampliação, a taxa com uma
ampliação (detector(gray, 1), só nos quadros em que a primeira falhou) e a largura da caixa detectada.

Amostragem: T trechos de Q quadros consecutivos, com inícios igualmente espaçados no vídeo (padrão 12 x 10 = 120
quadros por vídeo). Orientação: quadro exibido (auto-rotação ligada, como em src/features/extract_signals.py). Nos
vídeos sem metadado de rotação, como os três de F1_12, é o mesmo quadro que o código original lia.
Saída incremental: results/exp5a_dlib_rosto.csv, regravado a cada vídeo concluído (vídeos já medidos com os mesmos
parâmetros são pulados; uma execução interrompida não perde o que já terminou). --folds restringe aos folds dados.
Log: results/logs/exp5a_dlib_rosto.log.
"""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
SAIDA = RES / "exp5a_dlib_rosto.csv"
LOG = RES / "logs" / "exp5a_dlib_rosto.log"
LARGURA = 450      # imutils.resize(frame, width=450) no blink_video.py
GAMA = 1.5         # adjust_gamma(gray, gamma=1.5) no blink_video.py


def listar_videos():
    out = []
    for ok in sorted(glob.glob(str(ROOT / "data" / "uta_rldd_videos" / "Fold*" / ".ok"))):
        fdir = Path(ok).parent
        k = int(fdir.name.replace("Fold", ""))
        for sdir in sorted(p for p in fdir.iterdir() if p.is_dir()):
            for f in sorted(sdir.iterdir()):
                if f.suffix.lower() not in (".mp4", ".mov", ".avi", ".m4v"):
                    continue
                # mesmos ids de src/features/extract_signals.py (list_uta): "0", "5", "10" e, em Fold3/32 e Fold5/49,
                # o vídeo sonolento dividido em "10_1"/"10_2" -> ids "F<k>_<suj>_10_p1"/"_p2" (rodada final)
                rot, _, parte = f.stem.partition("_")
                if rot in ("0", "5", "10"):
                    out.append((f"F{k}_{sdir.name}_{rot}" + (f"_p{parte}" if parte else ""), str(f), k))
    return out


def medir(args):
    vid, path, fold, T, Q = args
    import cv2
    import dlib
    cv2.setNumThreads(1)
    det = dlib.get_frontal_face_detector()
    tabela = np.array([((i / 255.0) ** (1.0 / GAMA)) * 255 for i in range(256)]).astype("uint8")
    t0 = time.time()
    cap = cv2.VideoCapture(path)
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    inicios = np.linspace(0, max(n - Q - 1, 0), T).astype(int)
    hit0, hit1, larg, alt_q = [], [], [], None
    for s in inicios:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s))
        for _ in range(Q):
            ok, fr = cap.read()
            if not ok:
                break
            h, w = fr.shape[:2]
            fr = cv2.resize(fr, (LARGURA, int(h * LARGURA / float(w))), interpolation=cv2.INTER_AREA)
            alt_q = fr.shape[0]
            g = cv2.LUT(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), tabela)
            r0 = det(g, 0)
            hit0.append(len(r0) > 0)
            if len(r0):
                larg.append(max(rr.width() for rr in r0))
                hit1.append(True)
            else:
                r1 = det(g, 1)
                hit1.append(len(r1) > 0)
                if len(r1):
                    larg.append(max(rr.width() for rr in r1))
    cap.release()
    return dict(id=vid, fold=fold, arquivo=str(Path(path).relative_to(ROOT)), quadros_no_video=n,
                trechos=T, quadros_por_trecho=Q, quadros_testados=len(hit0), quadro_processado=f"{LARGURA}x{alt_q}",
                taxa_dlib_sem_ampliacao=float(np.mean(hit0)) if hit0 else np.nan,
                taxa_dlib_com_1_ampliacao=float(np.mean(hit1)) if hit1 else np.nan,
                largura_caixa_mediana_px=float(np.median(larg)) if larg else np.nan,
                largura_caixa_p10_px=float(np.percentile(larg, 10)) if larg else np.nan,
                segundos=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description="Detector de rosto do dlib com o pré-processamento dos autores (Exp. 5a).")
    ap.add_argument("--folds", type=int, nargs="*", default=None, help="só estes folds (padrão: todos com .ok)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--trechos", type=int, default=12)
    ap.add_argument("--quadros-por-trecho", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    LOG.parent.mkdir(parents=True, exist_ok=True)
    flog = open(LOG, "a", encoding="utf-8")

    def log(msg):
        print(msg, flush=True)
        flog.write(msg + "\n")
        flog.flush()

    import dlib
    import cv2
    log(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} exp5a_dlib_rosto: dlib {dlib.__version__}, OpenCV {cv2.__version__}, "
        f"{a.trechos} trechos x {a.quadros_por_trecho} quadros por vídeo, {a.workers} processos")
    videos = [v for v in listar_videos() if a.folds is None or v[2] in a.folds]
    antigo = pd.read_csv(SAIDA) if (SAIDA.exists() and not a.force) else pd.DataFrame(columns=["id"])
    if len(antigo):
        antigo = antigo[(antigo["trechos"] == a.trechos) & (antigo["quadros_por_trecho"] == a.quadros_por_trecho)]
    feitos = set(antigo["id"].astype(str))
    todo = [(v, p, k, a.trechos, a.quadros_por_trecho) for v, p, k in videos if v not in feitos]
    log(f"{len(videos)} vídeos nos folds pedidos ({sorted({v[2] for v in videos})}); {len(feitos & {v[0] for v in videos})} "
        f"já medidos; {len(todo)} a medir")
    completo = pd.read_csv(SAIDA) if SAIDA.exists() else pd.DataFrame(columns=["id"])

    def gravar(novos):
        base = completo[~completo["id"].astype(str).isin([r["id"] for r in novos])] if len(completo) else completo
        out = pd.concat([base, pd.DataFrame(novos)], ignore_index=True).sort_values("id")
        tmp = SAIDA.with_suffix(".tmp.csv")
        out.to_csv(tmp, index=False)
        tmp.replace(SAIDA)
        return out

    novos = []
    if todo:
        with mp.get_context("spawn").Pool(min(a.workers, 4), maxtasksperchild=1) as pool:   # regra: <= 4 workers de vídeo
            for r in pool.imap_unordered(medir, todo):
                novos.append(r)
                gravar(novos)
                log(f"  {r['id']}: dlib sem ampliação {r['taxa_dlib_sem_ampliacao']:.3f}, com 1 ampliação "
                    f"{r['taxa_dlib_com_1_ampliacao']:.3f}, caixa mediana {r['largura_caixa_mediana_px']:.0f} px "
                    f"({r['quadros_testados']} quadros, {r['segundos']:.0f} s)")
    out = gravar(novos)
    log(f"-> {SAIDA.relative_to(ROOT)} ({len(out)} vídeos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
