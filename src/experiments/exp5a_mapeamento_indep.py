#!/usr/bin/env python3
"""Exp. 5a — verificação INDEPENDENTE do mapeamento sujeito oficial <-> pasta (implementação independente, para conferir a principal).

Não reutiliza o detector de vales nem o escore de src/features/uta_oficial.py. Usa só os instantes NF das piscadas
oficiais já recuperados (data/features/final/uta/oficial_piscadas_fold<k>.csv) e o EAR bruto de data/signals/uta/*.npz:
  * vales próprios: EAR / mediana móvel de 91 quadros, scipy.signal.find_peaks(-r, prominence=0.10, distance=4);
  * defasagem LOCAL por piscada: moda de (NF - vale) nas 15 piscadas oficiais vizinhas de cada lado (a própria
    piscada fica de fora), tolerância ±2 quadros, busca em [-1500, +100];
  * escore do vídeo = fração de piscadas oficiais com vale na defasagem local, menos a média com os instantes
    deslocados circularmente 1/3 e 2/3 do vídeo (acaso); escore do sujeito = soma dos 3 vídeos;
  * pasta inviável num vídeo (NF final > nº de quadros) não recebe escore naquele vídeo.

Rodada final (27/09/2026): 5 folds. Pastas candidatas = sujeitos de data/signals/uta/index.csv (coluna `subject`), e
não o nome do arquivo, porque os vídeos divididos (Fold3/32 e Fold5/49, vídeo 10 em duas partes "_p1"/"_p2") têm um
.npz por parte: o EAR do vídeo lógico é a concatenação das partes em ordem (a contagem de quadros do original segue
na parte 2; results/exp5_caracteristicas.md, seção 5.4). Sujeitos oficiais sem pasta adotada (indeterminados do Fold 3)
também recebem escore contra todas as pastas, mas `pasta_adotada` fica vazia. Otimização sem efeito no resultado: só os
vales dentro de [min(NF vizinhos) - 100, max(NF vizinhos) + 1500] entram na diferença (os demais caem fora de
[LMIN, LMAX] de qualquer forma).
Saída: results/exp5a_mapeamento_indep.csv (um escore por sujeito oficial x pasta) e log em results/logs/.
Rodar no .venv: python src/experiments/exp5a_mapeamento_indep.py
"""
from __future__ import annotations

import glob
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[2]
TOL, VIZ, LMIN, LMAX = 2, 15, -1500, 100
ROT = {0: "0", 5: "5", 10: "10"}
OFICIAL = ROOT / "data" / "features" / "final" / "uta"
SINAIS = ROOT / "data" / "signals" / "uta"


def partes_por_video():
    """vídeo lógico -> lista ordenada de ids de arquivo (partes _p1, _p2 em ordem); e pastas por fold."""
    idx = pd.read_csv(SINAIS / "index.csv")
    idx = idx[idx["complete"].astype(bool)]
    ex = idx["extra_json"].map(json.loads)
    idx = idx.assign(fold=ex.map(lambda e: int(e["fold"])),
                     video=[e.get("video") or i for e, i in zip(ex, idx["id"])],
                     parte=ex.map(lambda e: int(e.get("part") or 0)))
    partes = {v: g.sort_values("parte")["id"].tolist() for v, g in idx.groupby("video")}
    pastas = {k: sorted(g["subject"].unique()) for k, g in idx.groupby("fold")}
    return partes, pastas


def vales(video, partes, cache={}):
    if video not in cache:
        es = []
        for cid in partes[video]:
            z = np.load(SINAIS / f"{cid}.npz")
            e = z["ear"].astype(float)
            f = z["face"] & np.isfinite(e)
            es.append(np.where(f, e, np.nan))
        e = pd.Series(np.concatenate(es)).interpolate(limit_direction="both").values
        r = e / median_filter(e, size=91, mode="nearest")
        cache[video] = (find_peaks(-r, prominence=0.10, distance=4)[0], len(e))
    return cache[video]


def escore(w, p):
    if len(p) == 0:
        return 0.0
    hits = []
    for i in range(len(w)):
        nb = np.r_[max(0, i - VIZ):i, i + 1:min(len(w), i + VIZ + 1)]
        if len(nb) == 0:
            hits.append(False)
            continue
        # só vales com w[nb] - p em [LMIN, LMAX] para algum vizinho (p ordenado): mesmo resultado, bem mais rápido
        a, b = np.searchsorted(p, [w[nb].min() - LMAX, w[nb].max() - LMIN + 1])
        pp = p[a:b]
        dif = (w[nb][:, None] - pp[None, :]).ravel()
        dif = dif[(dif >= LMIN) & (dif <= LMAX)]
        if len(dif) == 0:
            hits.append(False)
            continue
        h = np.bincount((dif - LMIN).astype(int), minlength=LMAX - LMIN + 1)
        lag = int(np.argmax(np.convolve(h, np.ones(2 * TOL + 1), mode="same"))) + LMIN
        hits.append(bool(np.any(np.abs(w[i] - p - lag) <= TOL)))
    return float(np.mean(hits))


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    logp = ROOT / "results" / "logs" / f"exp5a_mapeamento_indep_{ts}.log"
    lf = open(logp, "w", encoding="utf-8")
    t0 = time.time()

    def log(m):
        print(m, flush=True)
        lf.write(m + "\n")
        lf.flush()
    partes, pastas_fold = partes_por_video()
    log(f"Verificação independente do mapeamento (rodada final): {len(partes)} vídeos lógicos em data/signals/uta; "
        f"TOL={TOL}, VIZ={VIZ}, busca [{LMIN}, {LMAX}]")
    linhas = []
    for arq in sorted(glob.glob(str(OFICIAL / "oficial_piscadas_fold*.csv"))):
        k = int(re.search(r"fold(\d+)", arq).group(1))
        of = pd.read_csv(arq)
        pastas = pastas_fold[k]
        for s in sorted(of["sujeito_oficial"].unique()):
            g_s = of[of["sujeito_oficial"] == s]
            adot = g_s["pasta"].iloc[0]
            adot = str(adot) if isinstance(adot, str) and adot else ""
            classe = str(g_s["classe"].iloc[0])
            for pa in pastas:
                tot, inviaveis = 0.0, 0
                for r_, g in g_s.groupby("rotulo"):
                    w = np.round(g["NF"].values).astype(int)
                    p, n = vales(f"{pa}_{ROT[int(r_)]}", partes)
                    if w[-1] > n:
                        inviaveis += 1
                        continue
                    real = escore(w, p)
                    nulo = np.mean([escore((w + int(n * f)) % n, p) for f in (1 / 3, 2 / 3)])
                    tot += real - nulo
                linhas.append(dict(fold=k, sujeito_oficial=int(s), classe_adotada=classe, pasta=pa, pasta_adotada=adot,
                                   escore=tot, videos_inviaveis_duracao=inviaveis))
            d = pd.DataFrame([x for x in linhas if x["fold"] == k and x["sujeito_oficial"] == s])
            d = d.sort_values("escore", ascending=False)
            log(f"fold {k} sujeito {s} ({classe}): adotada {adot or '—'}; melhor {d.pasta.iloc[0]} ({d.escore.iloc[0]:.3f}); "
                f"2ª {d.pasta.iloc[1]} ({d.escore.iloc[1]:.3f}) [{time.time() - t0:.0f} s]")
    df = pd.DataFrame(linhas)
    df["melhor_do_sujeito"] = df.groupby(["fold", "sujeito_oficial"])["escore"].transform("max") == df["escore"]
    out = ROOT / "results" / "exp5a_mapeamento_indep.csv"
    tmp = out.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(out)
    res = df[df["melhor_do_sujeito"] & (df["pasta_adotada"] != "")]
    for k, g in res.groupby("fold"):
        for cl, gg in g.groupby("classe_adotada"):
            log(f"fold {k} ({cl}): melhor pasta = adotada em {(gg.pasta == gg.pasta_adotada).sum()}/{len(gg)} sujeitos")
    log(f"-> {out.relative_to(ROOT)} ({len(df)} linhas; {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
