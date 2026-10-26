"""Exp. 6 (auxiliar) — iluminação medida nos vídeos: luminância do quadro, do rosto e dos olhos.

Roda SÓ no ambiente .venv-mp (precisa do OpenCV; não precisa do MediaPipe, porque reaproveita os
landmarks já gravados em data/signals/<base>/<id>.npz):

    source .venv-mp/bin/activate
    python src/experiments/exp6_luminancia.py --workers 4          # incremental
    python src/experiments/exp6_luminancia.py --force               # recalcula tudo

Por que: a PP3 pergunta se a queda de desempenho se concentra à NOITE. "Noite" aqui é um rótulo da base
(NITYMED); esta etapa mede quanta luz de fato chega ao rosto em cada base, para que a comparação
noite × dia não dependa só da descrição dos autores. Nenhuma imagem é gravada — só números.

Método (por ARQUIVO de vídeo; as partes _p1/_p2 dos vídeos divididos da UTA são medidas em separado e juntadas no
vídeo lógico por src/experiments/exp6_noite.py, que usa as amostras individuais gravadas aqui):
  * N_AMOSTRAS = 24 quadros COM ROSTO, igualmente espaçados entre os quadros com rosto do .npz.
  * Leitura SEQUENCIAL (grab/retrieve, sem seek), com a mesma rotação automática da extração
    (CAP_PROP_ORIENTATION_AUTO = 1), para que o quadro lido seja exatamente o do índice do .npz.
  * Cinza = cv2.COLOR_BGR2GRAY (luma BT.601, 0–255), no quadro ORIGINAL (landmarks ÷ escala da extração).
  * lum_quadro: média do cinza no quadro inteiro.
    lum_rosto: média no retângulo envolvente dos 21 landmarks gravados (olhos, boca, nariz, queixo, testa, laterais).
    contraste_rosto: desvio-padrão do cinza nesse retângulo (contraste RMS).
    lum_olhos: média no retângulo dos 12 pontos dos olhos, com margem de 15% da distância entre os cantos externos.
    frac_escuro_rosto: fração dos pixels do rosto com cinza < 40 (sombra profunda).
  * Também registra o codec (FOURCC), fps, nº de quadros e resolução informados pelo contêiner.
Versões do cache: 1.0 (rodada preliminar, data/features/<base>/luminancia/) e 1.1 (= 1.0 + os valores de cada
amostra, campo "amostras"). As medidas são idênticas nas duas; um cache 1.0 da rodada preliminar é reaproveitado
(copiado para a pasta final, com "reaproveitado_de") quando o nº de quadros do .npz atual é o mesmo da medição.
Saídas: data/features/final/<base>/luminancia/<id>.json (cache por arquivo) e results/exp6_luminancia_videos.csv
(uma linha por arquivo, com a coluna `video` = vídeo lógico). Log: results/logs/exp6_final_luminancia.log
(o run_exp6.sh redireciona a saída para ele). UTA-RLDD: só vídeos com sinal extraído e fold com .ok (mesma
descoberta de src/features/blinks.py).
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "features"))
import blinks as B  # noqa: E402  (listar_videos: descoberta de vídeos/folds igual à das outras etapas)

N_AMOSTRAS = 24
LIMIAR_ESCURO = 40
VERSAO = "1.1"                    # 1.1 = 1.0 + valores por amostra ("amostras")
VERSOES_VALIDAS = ("1.0", "1.1")  # mesmas medidas; 1.0 = rodada preliminar
BASES = ("dmd", "nitymed", "uta")
CACHE_FINAL = ROOT / "data" / "features" / "final"
CACHE_PRELIM = ROOT / "data" / "features"      # rodada preliminar (só leitura)
METRICAS = ("lum_quadro", "lum_rosto", "contraste_rosto", "lum_olhos", "frac_escuro_rosto", "area_rosto_frac")


def _fourcc(cap):
    import cv2
    v = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((v >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ")


def n_quadros_npz(npz_path):
    with np.load(npz_path, allow_pickle=False) as z:
        return int(len(z["face"]))


def medir_video(args):
    ds, vid, video_path, npz_path, out_json = args
    import cv2
    cv2.setNumThreads(1)
    t0 = time.time()
    with np.load(npz_path, allow_pickle=False) as z:
        face = z["face"].astype(bool) & np.isfinite(z["ear"])
        lm = z["lm"].astype(float)
        escala = float(z["meta_scale"].item()) if "meta_scale" in z.files else 1.0
        n_npz = len(face)
    alvo = np.flatnonzero(face)
    if len(alvo) == 0:
        res = dict(dataset=ds, id=vid, versao=VERSAO, erro="sem quadros com rosto")
        json.dump(res, open(out_json, "w"))
        return res
    sel = np.unique(alvo[np.linspace(0, len(alvo) - 1, min(N_AMOSTRAS, len(alvo))).round().astype(int)])
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    meta = dict(codec=_fourcc(cap), fps_cont=float(cap.get(cv2.CAP_PROP_FPS)),
                n_quadros_cont=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    medidas = []
    i, j = 0, 0
    largura = altura = None
    while j < len(sel):
        ok = cap.grab()
        if not ok:
            break
        if i == sel[j]:
            ok, fr = cap.retrieve()
            if ok:
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                altura, largura = g.shape
                p = lm[i] / escala                        # landmarks no quadro original
                x0, y0 = np.nanmin(p, 0); x1, y1 = np.nanmax(p, 0)
                xa, xb = int(max(0, np.floor(x0))), int(min(largura, np.ceil(x1)))
                ya, yb = int(max(0, np.floor(y0))), int(min(altura, np.ceil(y1)))
                rosto = g[ya:yb, xa:xb]
                olhos = p[0:12]
                d_io = float(np.linalg.norm(p[3] - p[6]))  # cantos externos: 263 (olho esq., p4) e 33 (olho dir., p1)
                m = 0.15 * d_io
                ex0, ey0 = np.nanmin(olhos, 0) - m; ex1, ey1 = np.nanmax(olhos, 0) + m
                oa = g[int(max(0, ey0)):int(min(altura, ey1)), int(max(0, ex0)):int(min(largura, ex1))]
                if rosto.size >= 100 and oa.size >= 20:
                    medidas.append(dict(q=int(i), lum_quadro=float(g.mean()), lum_rosto=float(rosto.mean()),
                                        contraste_rosto=float(rosto.std()), lum_olhos=float(oa.mean()),
                                        frac_escuro_rosto=float((rosto < LIMIAR_ESCURO).mean()),
                                        area_rosto_frac=float(rosto.size / g.size)))
            j += 1
        i += 1
    cap.release()
    res = dict(dataset=ds, id=vid, versao=VERSAO, n_amostras=len(medidas), n_quadros_npz=n_npz,
               largura=largura, altura=altura, tempo_s=round(time.time() - t0, 2), **meta)
    if medidas:
        for k in METRICAS:
            v = np.array([m_[k] for m_ in medidas])
            res[k] = float(np.median(v))
            res[k + "_p10"] = float(np.percentile(v, 10))
            res[k + "_p90"] = float(np.percentile(v, 90))
        res["amostras"] = medidas
    else:
        res["erro"] = "nenhum quadro amostrado lido"
    tmp = str(out_json) + ".tmp"
    json.dump(res, open(tmp, "w"), ensure_ascii=False)
    os.replace(tmp, out_json)
    return res


def cache_valido(p, n_npz):
    """Cache legível, de versão válida e medido sobre um .npz com o mesmo nº de quadros do atual."""
    try:
        c = json.load(open(p))
    except (json.JSONDecodeError, OSError):
        return None
    if c.get("versao") not in VERSOES_VALIDAS or "erro" in c:
        return None
    if int(c.get("n_quadros_npz", -1)) != n_npz:
        return None
    return c


def main():
    ap = argparse.ArgumentParser(description="Luminância do quadro/rosto/olhos por vídeo (Exp. 6).")
    ap.add_argument("--workers", type=int, default=4, help="processos (no máximo 4 de vídeo neste Mac)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dataset", default="todos", choices=list(BASES) + ["todos"])
    args = ap.parse_args()
    args.workers = max(1, min(4, args.workers))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    log = logging.getLogger("exp6_luminancia")
    import pandas as pd
    bases = BASES if args.dataset == "todos" else (args.dataset,)
    log.info("Folds da UTA-RLDD com .ok: %s", B.uta_folds_ok())
    tarefas, prontos, reaproveitados = [], 0, 0
    video_de = {}
    for ds in BASES:
        idx = B.listar_videos(ds)
        for _, r in idx.iterrows():
            video_de[(ds, r["id"])] = r["video"]
        if ds not in bases:
            continue
        cache = CACHE_FINAL / ds / "luminancia"
        cache.mkdir(parents=True, exist_ok=True)
        for _, r in idx.iterrows():
            out = cache / f"{r['id']}.json"
            n_npz = n_quadros_npz(ROOT / r["npz"])
            if not args.force:
                if out.exists() and cache_valido(out, n_npz) is not None:
                    prontos += 1
                    continue
                antigo = CACHE_PRELIM / ds / "luminancia" / f"{r['id']}.json"
                c = cache_valido(antigo, n_npz) if antigo.exists() else None
                if c is not None:
                    c["reaproveitado_de"] = str(antigo.relative_to(ROOT))
                    tmp = str(out) + ".tmp"
                    json.dump(c, open(tmp, "w"), ensure_ascii=False)
                    os.replace(tmp, out)
                    reaproveitados += 1
                    continue
            tarefas.append((ds, r["id"], ROOT / r["path"], ROOT / r["npz"], out))
    log.info("%d arquivos já prontos, %d reaproveitados da rodada preliminar (mesmo nº de quadros no .npz), "
             "%d a processar (N_AMOSTRAS=%d, %d processos)", prontos, reaproveitados, len(tarefas), N_AMOSTRAS,
             args.workers)
    t0 = time.time()
    if tarefas:
        ctx = mp.get_context("spawn")
        with ctx.Pool(args.workers, maxtasksperchild=4) as pool:
            for k, res in enumerate(pool.imap_unordered(medir_video, tarefas), 1):
                if "erro" in res:
                    log.warning("%s/%s: %s", res["dataset"], res["id"], res["erro"])
                if k % 10 == 0 or k == len(tarefas):
                    log.info("%d/%d arquivos (%.0f s)", k, len(tarefas), time.time() - t0)
    linhas = []
    for ds in BASES:
        for f in sorted((CACHE_FINAL / ds / "luminancia").glob("*.json")):
            c = json.load(open(f))
            if (ds, c.get("id")) not in video_de:      # arquivo que saiu da lista (não deve ocorrer)
                continue
            c.pop("amostras", None)
            c["video"] = video_de[(ds, c["id"])]
            linhas.append(c)
    df = pd.DataFrame(linhas)
    out = ROOT / "results" / "exp6_luminancia_videos.csv"
    df.to_csv(out, index=False)
    for ds, g in df.groupby("dataset"):
        log.info("%s: %d arquivos (%d vídeos lógicos); lum_rosto mediana %.1f (mín %.1f, máx %.1f); "
                 "lum_quadro mediana %.1f; versões %s", ds, len(g), g["video"].nunique(), g["lum_rosto"].median(),
                 g["lum_rosto"].min(), g["lum_rosto"].max(), g["lum_quadro"].median(),
                 g["versao"].value_counts().to_dict())
    faltam = [k for k in video_de if k[0] in bases and not (CACHE_FINAL / k[0] / "luminancia" / f"{k[1]}.json").exists()]
    if faltam:
        log.warning("Arquivos sem medida: %s", faltam)
    log.info("Tabela: %s (%.0f s)", out.relative_to(ROOT), time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
