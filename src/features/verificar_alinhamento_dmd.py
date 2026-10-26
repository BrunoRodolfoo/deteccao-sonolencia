"""Verificação INDEPENDENTE do alinhamento anotação <-> *_rgb_face.mp4 do DMD (verificação do Exp. 5).

Não reutiliza o código de dmd_labels.py: lê o OpenLABEL e o .npz de sinais diretamente.
  1. Para as 16 sessões: correlação de Pearson entre o indicador de "eyes_state/close" (e de "blinks/blinking")
     e -EAR em lags de -300 a +300 quadros (~ +-10 s; a busca de dmd_labels.py vai só até +-30).
  2. Para 3 sessões (gA_1, gZ_37, gC_14): grava um mosaico com quadros no MEIO de 2 fechamentos anotados
     (>= 15 quadros), no PRIMEIRO quadro de um terceiro fechamento e no meio de 2 trechos "open" longos,
     lidos SEQUENCIALMENTE do vídeo (sem seek), para conferência visual.

Rodar no .venv-mp (precisa de OpenCV):  python src/features/verificar_alinhamento_dmd.py
Saídas: results/exp5_dmd_alinhamento_verificacao.csv, results/figures/exp5_dmd_verificacao_visual.jpg
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MAXLAG = 300


def acoes(ann):
    d = json.load(open(ann))["openlabel"]
    iv = {}
    for a in d["actions"].values():
        iv.setdefault(a["type"], []).extend((int(x["frame_start"]), int(x["frame_end"])) for x in a["frame_intervals"])
    return iv


def ind(ints, n):
    x = np.zeros(n)
    for s, e in ints:
        x[max(s, 0):min(e, n - 1) + 1] = 1
    return x


def corr(a, x, lag):
    n = len(a)
    aa, xx = (a[:n - lag], x[lag:]) if lag >= 0 else (a[-lag:], x[:n + lag])
    m = np.isfinite(xx)
    return float(np.corrcoef(aa[m], xx[m])[0, 1])


def main():
    idx = pd.read_csv(ROOT / "data/signals/dmd/index.csv")
    rows = []
    lags = np.arange(-MAXLAG, MAXLAG + 1)
    for _, r in idx.iterrows():
        ann = json.loads(r["extra_json"])["annotation"]
        iv = acoes(ROOT / ann)
        z = np.load(ROOT / r["npz"])
        ear = np.where(z["face"].astype(bool), z["ear"], np.nan)
        n = len(ear)
        out = dict(id=r["id"], n_quadros=n)
        for nome, chave in (("close", "eyes_state/close"), ("blinking", "blinks/blinking")):
            a = ind(iv.get(chave, []), n)
            c = np.array([corr(a, -ear, int(L)) for L in lags])
            i = int(np.nanargmax(c))
            longe = np.abs(lags) > 10
            j = int(np.nanargmax(np.where(longe, c, -np.inf)))
            out.update({f"{nome}_melhor_lag": int(lags[i]), f"{nome}_r_melhor": c[i], f"{nome}_r_lag0": c[MAXLAG],
                        f"{nome}_melhor_lag_fora_10": int(lags[j]), f"{nome}_r_fora_10": c[j]})
        e = z["ear"]; f = z["face"].astype(bool)
        out["ear_mediana_close"] = float(np.nanmedian(e[f & (ind(iv["eyes_state/close"], n) > 0)]))
        out["ear_mediana_open"] = float(np.nanmedian(e[f & (ind(iv["eyes_state/open"], n) > 0)]))
        rows.append(out)
        print(f"{r['id'][:12]:12s} close: lag*={out['close_melhor_lag']:+d} r*={out['close_r_melhor']:.3f} "
              f"r0={out['close_r_lag0']:.3f} (melhor fora de +-10: {out['close_r_fora_10']:.3f}) | blinking: "
              f"lag*={out['blinking_melhor_lag']:+d} r*={out['blinking_r_melhor']:.3f} r0={out['blinking_r_lag0']:.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results/exp5_dmd_alinhamento_verificacao.csv", index=False)
    print(f"melhor lag (close) em [{df.close_melhor_lag.min()}, {df.close_melhor_lag.max()}]; "
          f"(blinking) em [{df.blinking_melhor_lag.min()}, {df.blinking_melhor_lag.max()}]; busca +-{MAXLAG} quadros; "
          f"ganho máx. de r sobre o lag 0 (close) = {(df.close_r_melhor - df.close_r_lag0).max():.3f}")

    import cv2
    tiles_rows = []
    for pat in ("gA/1", "gZ/37", "gC/14"):
        ann = sorted(glob.glob(str(ROOT / f"data/dmd/{pat}/s5/*_rgb_ann_drowsiness.json")))[0]
        vid = ann.replace("_rgb_ann_drowsiness.json", "_rgb_face.mp4")
        iv = acoes(ann)
        cl = sorted(x for x in iv["eyes_state/close"] if x[1] - x[0] >= 15)
        op = sorted(x for x in iv["eyes_state/open"] if x[1] - x[0] >= 60)
        sel = [("FECHADO", (cl[1][0] + cl[1][1]) // 2), ("FECHADO", (cl[-1][0] + cl[-1][1]) // 2),
               ("FECHADO 1o quadro", cl[2][0]), ("ABERTO", (op[3][0] + op[3][1]) // 2),
               ("ABERTO", (op[-2][0] + op[-2][1]) // 2)]
        want = {q for _, q in sel}
        cap = cv2.VideoCapture(vid)
        got, i = {}, 0
        while len(got) < len(want) and cap.grab():
            if i in want:
                got[i] = cap.retrieve()[1]
            i += 1
        tiles = []
        for lab, q in sel:
            im = cv2.resize(got[q][100:520, 340:940], (360, 252))
            cv2.putText(im, f"{pat} {lab} q{q}", (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            tiles.append(im)
        print(pat, "quadros:", sel)
        tiles_rows.append(np.hstack(tiles))
    (ROOT / "results/figures").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ROOT / "results/figures/exp5_dmd_verificacao_visual.jpg"), np.vstack(tiles_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
