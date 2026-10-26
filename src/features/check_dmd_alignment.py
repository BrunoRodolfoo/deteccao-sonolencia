"""Verifica o alinhamento entre as anotações do DMD (OpenLABEL) e os quadros do *_rgb_face.mp4.

Para cada sessão s5:
  1. Confere nº de quadros do vídeo (lido na extração) x face_camera.total_frames x fim das ações
     x streams.face_camera.sync.frame_shift.
  2. Constrói o indicador por quadro de "eyes_state/close" e procura o deslocamento (lag, em
     quadros, de -30 a +30) que maximiza a correlação de Pearson entre esse indicador e (-EAR).
     lag > 0 significa que o sinal do vídeo está ATRASADO em relação à anotação
     (ear[t + lag] casa com ann[t]).
  3. Mede a separação: mediana do EAR dentro e fora de "eyes_state/close" (lag 0).
Gera results/sinais_alinhamento_dmd.csv e a figura results/figures/diag_sinal_dmd.png (20 s de
gA_1 com os intervalos anotados sobrepostos).

Rodar no .venv (precisa de matplotlib): python src/features/check_dmd_alignment.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SIG = ROOT / "data" / "signals" / "dmd"
MAXLAG = 30


def load_actions(ann_path):
    d = json.load(open(ann_path))["openlabel"]
    acts = {}
    for a in d["actions"].values():
        acts.setdefault(a["type"], []).extend(a.get("frame_intervals", []))
    fc = d["streams"]["face_camera"]["stream_properties"]
    return acts, fc.get("total_frames"), fc.get("sync", {}).get("frame_shift"), d["frame_intervals"]


def indicator(intervals, n):
    x = np.zeros(n, bool)
    for iv in intervals:
        x[iv["frame_start"]: min(iv["frame_end"], n - 1) + 1] = True  # frame_end inclusivo
    return x


def corr_at_lag(ann, ear, lag):
    n = len(ann)
    if lag >= 0:
        a, e = ann[: n - lag], ear[lag:]
    else:
        a, e = ann[-lag:], ear[: n + lag]
    m = np.isfinite(e)
    if m.sum() < 100 or a[m].std() == 0:
        return np.nan
    return float(np.corrcoef(a[m].astype(float), -e[m])[0, 1])


def main():
    idx = pd.read_csv(SIG / "index.csv")
    rows = []
    for _, r in idx.iterrows():
        z = np.load(ROOT / r["npz"])
        ear = z["ear"]
        n = len(ear)
        extra = json.loads(r["extra_json"])
        acts, tot, shift, fi = load_actions(ROOT / extra["annotation"])
        close = indicator(acts.get("eyes_state/close", []), n)
        max_end = max(iv["frame_end"] for v in acts.values() for iv in v)
        lags = list(range(-MAXLAG, MAXLAG + 1))
        cs = [corr_at_lag(close, ear, L) for L in lags]
        best = lags[int(np.nanargmax(cs))]
        fin = np.isfinite(ear)
        rows.append(dict(id=r["id"], subject=r["subject"], n_frames_video=n,
                         face_camera_total_frames=tot, face_frame_shift=shift,
                         file_frame_interval_end=fi[0]["frame_end"], max_action_frame_end=max_end,
                         n_close_intervals=len(acts.get("eyes_state/close", [])),
                         frac_close=float(close.mean()),
                         corr_lag0=cs[lags.index(0)], best_lag=best, corr_best=float(np.nanmax(cs)),
                         ear_med_close=float(np.nanmedian(ear[close & fin])),
                         ear_med_open=float(np.nanmedian(ear[~close & fin])),
                         glasses=extra.get("glasses"), face_rate=r["face_rate"]))
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "sinais_alinhamento_dmd.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))

    # ---- figura: 20 s de gA_1 ----
    r = idx[idx["id"].str.startswith("gA_1_")].iloc[0]
    z = np.load(ROOT / r["npz"])
    extra = json.loads(r["extra_json"])
    acts, *_ = load_actions(ROOT / extra["annotation"])
    fps = float(r["fps"])
    t0, t1 = 60.0, 80.0  # trecho com fechamentos longos em gA_1 (ver contexto do projeto)
    f0, f1 = int(t0 * fps), int(t1 * fps)
    t = z["t_s"][f0:f1]
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 8, "axes.edgecolor": "#c3c2b7",
                         "axes.linewidth": 0.6, "xtick.color": "#c3c2b7", "ytick.color": "#c3c2b7",
                         "xtick.labelcolor": "#0b0b0b", "ytick.labelcolor": "#0b0b0b",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.formatter.use_locale": False})
    fig, ax = plt.subplots(figsize=(17 / 2.54, 5.5 / 2.54), dpi=300)
    for k, iv in enumerate(acts.get("eyes_state/close", [])):
        a, b = iv["frame_start"] / fps, (iv["frame_end"] + 1) / fps
        if b < t0 or a > t1:
            continue
        ax.axvspan(max(a, t0), min(b, t1), color="#eb6834", alpha=0.25, lw=0,
                   label="anotação: eyes_state/close" if k >= 0 and not ax.get_legend_handles_labels()[1] else None)
    ax.plot(t, z["ear"][f0:f1], color="#2a78d6", lw=1.0, label="EAR (média dos olhos)")
    ax.set_xlim(t0, t1)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("EAR")
    ax.grid(True, axis="y", color="#e1e0d9", lw=0.5)
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
    ax.yaxis.set_major_formatter(fmt)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}".replace(".", ",")))
    ax.set_ylim(0, max(0.35, float(np.nanmax(z["ear"][f0:f1])) * 1.05))
    row = df[df["id"] == r["id"]].iloc[0]
    fig.suptitle(f"DMD {r['subject']} (s5): EAR do Face Mesh x anotação de olho fechado — "
                 f"melhor lag = {int(row['best_lag'])} quadros", fontsize=8, x=0.01, ha="left", color="#0b0b0b")
    ax.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
              borderaxespad=0.2)
    fig.tight_layout()
    figp = ROOT / "results" / "figures" / "diag_sinal_dmd.png"
    fig.savefig(figp)
    print("figura:", figp, "| csv:", out)


if __name__ == "__main__":
    main()
