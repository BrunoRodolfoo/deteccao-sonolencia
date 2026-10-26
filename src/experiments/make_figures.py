#!/usr/bin/env python3
"""Figuras de resultados (Entrega 02 / artigo IEEE / pôster) — matplotlib, vetoriais.

Lê SOMENTE os CSV/JSON de `results/` e `models/` (nenhum número de resultado é
digitado aqui; as únicas constantes são os valores de referência publicados por
Ghoddoosian et al. (2019), Tabelas 2 e 3, citados também em results/exp1_intra.md).

Saída em results/figures/: <nome>.pdf (vetorial, para o artigo) e <nome>.png (300 dpi,
para o pôster). Largura = 8,5 cm (uma coluna IEEE); fonte Helvetica/Arial/DejaVu Sans
8 pt no tamanho final; rótulos com vírgula decimal (pt-BR).

Figuras:
  fig_exp1_va_por_fold   VA por fold: melhor clássico e melhor temporal (por VA média,
                         3 classes) vs. HM-LSTM e humanos (Tabela 3 do artigo), com chance.
  fig_exp1_bsa_vs_va     BSA e VA (média ± dp dos 5 folds) por modelo + referências.
  fig_exp2_vazamento     Protocolos janela / vídeo / sujeito, RF e LSTM, BSA e VA, com Δ em pp.
  fig_exp3_cross         Matriz 2x2 treino x teste (acur. balanceada) + barras por condição.
  fig_exp4_latencia      Latência p50/p95 TFLite no desktop e tamanho; (c) smartphone POR APARELHO
                         (results/exp4_phone_latency.csv: mediana entre as execuções de cada aparelho).

Figuras do artigo (Fig. 2 e 3), em tons de cinza e já na largura final do IEEEtran (coluna = 252 pt,
página = 516 pt), com texto de 7 pt ou mais; o PDF também é copiado para paper/figures/:
  fig_exp5b_transferencia  Fig. 2: Exp. 5b, AUC e acurácia por janela e por vídeo nos cenários A, B e C
                           (results/exp5b_transferencia.csv).
  fig_exp6_noite           Fig. 3: Exp. 6, (A) SNR do EAR por vídeo e (B) taxa de disparo no limiar 0,5
                           (results/exp6_noite.csv e results/exp6_noite_videos.csv).
  exp5_transfer.py e exp6_noite.py gravam em results/figures/ uma versão de trabalho com o mesmo nome;
  rodar este script depois deles (como faz o run_all.sh) deixa a versão do artigo nos dois lugares.

Uso:  python src/experiments/make_figures.py [--cew-variant patch|crop] [--only fig_exp3_cross]
      python src/experiments/make_figures.py --only fig_exp5b_transferencia fig_exp6_noite
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
MODELS = ROOT / "models"
OUT = RES / "figures"

# ----------------------------------------------------------------------------
# Estilo (paleta validada com scripts/validate_palette.js da skill dataviz:
# azul #2a78d6 + laranja #eb6834 passam nos testes de CVD e contraste; cinzas
# só para referências/chrome, nunca como série categórica).
# ----------------------------------------------------------------------------
C1 = "#2a78d6"      # destaque 1 (azul)      — série principal / BSA / fp32 / MRL->MRL
C2 = "#eb6834"      # destaque 2 (laranja)   — série secundária / VA / int8 / CEW->MRL
INK = "#0b0b0b"
INK2 = "#52514e"    # referência escura (HM-LSTM)
MUTED = "#898781"   # referência clara (humanos) / rótulos de eixo
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

COL_W_CM = 8.5
COL_W_IN = COL_W_CM / 2.54
FS = 8.0     # fonte base (eixos, legendas)
FS_S = 7.5   # anotações

# Referências publicadas (Ghoddoosian et al., 2019 — CVPRW; Tabelas 2 e 3) — em %.
REF_HMLSTM_VA_FOLD = [64.0, 61.0, 70.0, 64.0, 67.0]
REF_HUMAN_VA_FOLD = [62.0, 59.0, 60.0, 53.0, 55.0]
REF_HMLSTM = {"bsa": 54.0, "va": 65.2}
REF_HUMAN = {"bsa": None, "va": 57.8}
REF_SRC = "Ghoddoosian et al. (2019)"

# Desvio-padrão: Exp. 1/2 usam ddof=0 (como results/exp1_intra.md e exp2_leakage.md);
# Exp. 3 usa ddof=1 (pandas padrão, como results/exp3_eye_state.md).
EXP3_DDOF = 1


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS_S,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": AXIS, "ytick.color": AXIS,
        "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
        "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
        "legend.frameon": False, "legend.handlelength": 1.6, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.0, "legend.labelspacing": 0.3,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 100, "savefig.dpi": 300,
    })


def br(x: float, nd: int = 1) -> str:
    """Número com vírgula decimal (pt-BR)."""
    s = f"{x:.{nd}f}".replace("-", "−")
    return s.replace(".", ",")


def pct_formatter(nd: int = 0):
    return FuncFormatter(lambda v, _: br(v, nd))


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    w, h = fig.get_size_inches()
    print(f"{name}: {w * 2.54:.2f} x {h * 2.54:.2f} cm -> {OUT / (name + '.pdf')} (+ .png 300 dpi)")
    plt.close(fig)


def errbar_kw(color=INK2):
    return dict(ecolor=color, elinewidth=0.6, capsize=1.6, capthick=0.6)


def light_grid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, linewidth=0.5)
    ax.grid(False, axis="x" if axis == "y" else "y")


# ----------------------------------------------------------------------------
# Carregamento
# ----------------------------------------------------------------------------
def load_exp1() -> pd.DataFrame:
    df = pd.read_csv(RES / "exp1_intra.csv")
    df = df[(df.scheme == "3c")].copy()
    df["value"] *= 100.0
    return df


def exp1_summary(df: pd.DataFrame) -> pd.DataFrame:
    """média ± dp populacional (ddof=0) nos 5 folds, por modelo x métrica — igual ao .md."""
    g = df.groupby(["model", "family", "metric"])["value"]
    out = g.agg(mean="mean", std=lambda s: s.std(ddof=0), n="count").reset_index()
    return out


def best_by_va(summary: pd.DataFrame, family: str) -> str:
    s = summary[(summary.family == family) & (summary.metric == "va")]
    return s.sort_values("mean", ascending=False).iloc[0]["model"]


def load_exp2() -> pd.DataFrame:
    df = pd.read_csv(RES / "exp2_leakage_plot.csv")
    df.loc[df.metric.isin(["bsa", "va", "bal_acc", "f1_macro", "auc_macro"]), ["mean", "std"]] *= 100.0
    return df


def load_exp3(cew_variant: str):
    d = pd.read_csv(RES / "exp3_eye_state.csv")
    s = pd.read_csv(RES / "exp3_strat_conditions.csv")
    d["value"] = d["value"].astype(float)
    s["value"] = s["value"].astype(float)
    return d, s


def load_exp4():
    lat = pd.read_csv(RES / "exp4_desktop_latency.csv")
    blink = json.loads((MODELS / "blink_temporal.json").read_text(encoding="utf-8"))
    phone_csv = RES / "exp4_phone_latency.csv"
    phone = pd.read_csv(phone_csv) if phone_csv.exists() else None
    return lat, blink, phone


# ----------------------------------------------------------------------------
# Fig. 1 — VA por fold
# ----------------------------------------------------------------------------
def fig_exp1_va_por_fold(df: pd.DataFrame) -> None:
    summ = exp1_summary(df)
    best_c, best_t = best_by_va(summ, "classico"), best_by_va(summ, "temporal")
    folds = [1, 2, 3, 4, 5]
    short = {"RandomForest": "RF"}

    def va_fold(model):
        s = df[(df.model == model) & (df.metric == "va")].set_index("fold")["value"]
        return [s.loc[f] for f in folds]

    chance = df[(df.model.str.startswith("Chance")) & (df.metric == "va")]["value"].mean()
    fig, ax = plt.subplots(figsize=(COL_W_IN, 2.6), layout="constrained")
    ax.axhline(chance, color=AXIS, lw=0.8, zorder=1)
    ax.text(5.3, chance + 0.8, f"chance por vídeo ({br(chance)} %)", color=MUTED, fontsize=FS_S - 0.5,
            va="bottom", ha="right")
    series = [
        (f"{short.get(best_c, best_c)} (nosso, clássico)", va_fold(best_c), C1, "o", "-"),
        (f"{short.get(best_t, best_t)} (nosso, temporal)", va_fold(best_t), C2, "s", "-"),
        (f"HM-LSTM, {REF_SRC}", REF_HMLSTM_VA_FOLD, INK2, "^", "--"),
        (f"Humanos, {REF_SRC}", REF_HUMAN_VA_FOLD, MUTED, "D", ":"),
    ]
    for label, y, c, m, ls in series:
        ax.plot(folds, y, color=c, marker=m, ms=4.2, lw=1.4, ls=ls, label=label,
                markeredgecolor=SURFACE, markeredgewidth=0.7, zorder=3)
    ax.set_xticks(folds)
    ax.set_xlim(0.7, 5.35)
    ax.set_ylim(28, 74)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_formatter(pct_formatter(0))
    ax.set_xlabel("Fold oficial do UTA-RLDD (partição por sujeito), 3 classes")
    ax.set_ylabel("Acurácia por vídeo, VA (%)")
    fig.legend(loc="outside lower center", ncol=2, fontsize=FS_S - 0.5, handlelength=2.0)
    save(fig, "fig_exp1_va_por_fold")


# ----------------------------------------------------------------------------
# Fig. 2 — BSA vs VA por modelo
# ----------------------------------------------------------------------------
def fig_exp1_bsa_vs_va(df: pd.DataFrame) -> None:
    summ = exp1_summary(df)
    models = ["LogReg", "SVM-RBF", "RandomForest", "LSTM", "CNN1D"]
    short = {"RandomForest": "RF"}
    chance = {m: summ[(summ.model.str.startswith("Chance")) & (summ.metric == m)]["mean"].item() for m in ("bsa", "va")}

    def stat(model, metric):
        r = summ[(summ.model == model) & (summ.metric == metric)].iloc[0]
        return r["mean"], r["std"]

    fig, ax = plt.subplots(figsize=(COL_W_IN, 2.55), layout="constrained")
    x = np.arange(len(models))
    w = 0.36
    handles = []
    for i, (metric, color, name) in enumerate([("bsa", C1, "BSA (por sequência)"), ("va", C2, "VA (por vídeo)")]):
        means = [stat(m, metric)[0] for m in models]
        stds = [stat(m, metric)[1] for m in models]
        ax.bar(x + (i - 0.5) * w, means, w - 0.04, color=color, zorder=3)
        ax.errorbar(x + (i - 0.5) * w, means, yerr=stds, fmt="none", zorder=4, **errbar_kw())
        handles.append(Patch(color=color, label=name))
    # referências publicadas: barras hachuradas (textura = canal secundário), sem dp
    xr = len(models) + np.array([0.35, 1.35])
    refs = [("HM-LSTM", REF_HMLSTM), ("Humanos", REF_HUMAN)]
    for j, (name, vals) in enumerate(refs):
        for i, (metric, color) in enumerate([("bsa", C1), ("va", C2)]):
            v = vals[metric]
            if v is None:
                ax.text(xr[j] + (i - 0.5) * w, 2.5, "n/d", ha="center", va="bottom", fontsize=FS_S - 1,
                        color=MUTED, rotation=90)
                continue
            ax.bar(xr[j] + (i - 0.5) * w, v, w - 0.04, facecolor=SURFACE, edgecolor=color, hatch="////",
                   linewidth=0.6, zorder=3)
    ax.axvline(len(models) - 0.15, color=AXIS, lw=0.6, zorder=2)
    for metric, color, name in (("bsa", C1, "chance BSA"), ("va", C2, "chance VA")):
        ax.axhline(chance[metric], color=color, lw=0.7, ls=(0, (2, 2)), zorder=2)
        handles.append(Line2D([0], [0], color=color, lw=0.7, ls=(0, (2, 2)),
                              label=f"{name} ({br(chance[metric])} %)"))
    ax.set_xticks(list(x) + list(xr))
    ax.set_xticklabels([short.get(m, m) for m in models] + [f"{n}*" for n, _ in refs],
                       rotation=30, ha="right", rotation_mode="anchor", fontsize=FS_S)
    ax.set_xlim(-0.65, xr[-1] + 0.6)
    ax.set_ylim(0, 80)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_formatter(pct_formatter(0))
    ax.set_ylabel("Acurácia, 3 classes (%)")
    ax.set_xlabel(f"Barras: média ± dp em 5 folds (por sujeito).\n"
                  f"* {REF_SRC}: valores publicados, sem dp.", fontsize=FS_S - 1, color=INK2)
    ax.legend(handles=handles, loc="upper left", ncol=2, bbox_to_anchor=(-0.02, 1.04), fontsize=FS_S - 0.5,
              handlelength=1.6)
    save(fig, "fig_exp1_bsa_vs_va")


# ----------------------------------------------------------------------------
# Fig. 3 — vazamento de identidade
# ----------------------------------------------------------------------------
def fig_exp2_vazamento(df: pd.DataFrame) -> None:
    protos = [("janela", "(i)\njanela"), ("video", "(ii)\nvídeo"), ("sujeito", "(iii)\nsujeito")]
    models = [("RandomForest", "RF (24 estat.)", C1), ("LSTM", "LSTM (30×4)", C2)]

    def stat(proto, model, metric):
        r = df[(df.protocol == proto) & (df.model == model) & (df.metric == metric)]
        if r.empty or pd.isna(r["mean"].item()):
            return None, None
        return r["mean"].item(), r["std"].item()

    fig, axes = plt.subplots(1, 2, figsize=(COL_W_IN, 2.6), layout="constrained", sharey=True)
    x = np.arange(len(protos))
    w = 0.36
    for ax, (metric, title) in zip(axes, [("bsa", "BSA (por sequência)"), ("va", "VA (por vídeo)")]):
        for i, (model, label, color) in enumerate(models):
            xs = x + (i - 0.5) * w
            base = stat("sujeito", model, metric)[0]
            for k, (proto, _) in enumerate(protos):
                m, s = stat(proto, model, metric)
                if m is None:
                    ax.text(xs[k], 1.5, "n/a", rotation=90, ha="center", va="bottom", fontsize=FS_S - 1, color=MUTED)
                    continue
                ax.bar(xs[k], m, w - 0.04, color=color, label=label if k == 0 else None, zorder=3)
                ax.errorbar(xs[k], m, yerr=s, fmt="none", zorder=4, **errbar_kw())
                if proto != "sujeito" and base is not None:
                    d = m - base
                    ax.text(xs[k], m + (s or 0) + 1.5, f"{'+' if d >= 0 else ''}{br(d)} pp", ha="center",
                            va="bottom", fontsize=FS_S - 1, color=INK2, rotation=90)
        ax.set_title(title, fontsize=FS, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([p for _, p in protos], fontsize=FS_S, linespacing=1.1)
        ax.set_ylim(0, 132)
        ax.set_yticks(range(0, 101, 20))
        ax.yaxis.set_major_formatter(pct_formatter(0))
        ax.set_xlim(-0.6, len(protos) - 0.4)
    axes[0].set_ylabel("Acurácia, 3 classes (%)")
    axes[1].legend(handles=[Patch(color=c, label=l) for _, l, c in models], loc="upper left",
                   bbox_to_anchor=(-0.02, 1.0), fontsize=FS_S - 0.5)
    fig.supxlabel("Rótulos: diferença em pp para o protocolo por sujeito (iii).\n"
                  "Média ± dp em 3 seeds (i, ii) ou nos 5 folds oficiais (iii).",
                  fontsize=FS_S - 1, color=INK2)
    save(fig, "fig_exp2_vazamento")


# ----------------------------------------------------------------------------
# Fig. 4 — cross-dataset MRL <-> CEW
# ----------------------------------------------------------------------------
def fig_exp3_cross(d: pd.DataFrame, s: pd.DataFrame, cew_variant: str) -> None:
    cew = f"CEW_{cew_variant}"
    folds5 = ["1", "2", "3", "4", "5"]

    def fold_stats(protocol, test_on, metric="bal_acc"):
        r = d[(d.protocol == protocol) & (d.test_on == test_on) & (d.stratum == "todos")
              & (d.metric == metric) & (d.fold.astype(str).isin(folds5))]["value"] * 100
        return r.mean(), r.std(ddof=EXP3_DDOF)

    def one(protocol, test_on, fold, metric="bal_acc"):
        r = d[(d.protocol == protocol) & (d.test_on == test_on) & (d.stratum == "todos")
              & (d.metric == metric) & (d.fold.astype(str) == fold)]["value"]
        return r.item() * 100

    cells = {
        ("MRL", "MRL"): fold_stats("A_intra_mrl", "MRL"),
        ("MRL", "CEW"): fold_stats("B_folds_mrl_to_cew", cew),
        ("CEW", "MRL"): fold_stats(f"C_cew_{cew_variant}_to_mrl", "MRL"),
        ("CEW", "CEW"): fold_stats(f"D_intra_cew_{cew_variant}", cew),
    }
    final_b = one("B_final_mrl_to_cew", cew, "final")

    fig = plt.figure(figsize=(COL_W_IN, 2.85), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.5])
    ax = fig.add_subplot(gs[0, 0])
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
    M = np.array([[cells[("MRL", "MRL")][0], cells[("MRL", "CEW")][0]],
                  [cells[("CEW", "MRL")][0], cells[("CEW", "CEW")][0]]])
    ax.imshow(M, cmap=cmap, vmin=50, vmax=100, aspect="auto")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, tr in enumerate(["MRL", "CEW"]):
        for j, te in enumerate(["MRL", "CEW"]):
            m, sd = cells[(tr, te)]
            txt = f"{br(m)}\n± {br(sd)}"
            if (tr, te) == ("MRL", "CEW"):
                txt += f"\n({br(final_b)}*)"
            ink = SURFACE if m > 80 else INK
            ax.text(j, i, txt, ha="center", va="center", fontsize=FS_S, color=ink, linespacing=1.15)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["MRL", "CEW"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["MRL", "CEW"], rotation=90, va="center")
    ax.tick_params(length=0, pad=2)
    ax.set_ylabel("Treino")
    ax.set_xlabel(f"Teste\nmédia ± dp, 5 folds;\n* modelo final (37 suj.);\nCEW = recorte {cew_variant}",
                  fontsize=FS_S - 1, color=INK2, linespacing=1.15)
    ax.set_title("(a) Treino × teste", fontsize=FS, pad=3)

    ax2 = fig.add_subplot(gs[0, 1])
    conds = [("glasses", "sem óculos", "sem óculos"), ("glasses", "com óculos", "com óculos"),
             ("lighting", "iluminação boa", "ilum. boa"), ("lighting", "iluminação ruim", "ilum. ruim"),
             ("reflections", "sem reflexo", "sem reflexo"), ("reflections", "reflexo pequeno", "reflexo peq."),
             ("reflections", "reflexo grande", "reflexo grande")]

    def strat(protocol, fold_sel, stratum, level, metric="bal_acc"):
        r = s[(s.protocol == protocol) & (s.test_on == "MRL") & (s.stratum == stratum) & (s.level == level)
              & (s.metric == metric) & (s.fold.astype(str).isin(fold_sel))]["value"] * 100
        return r.mean(), (r.std(ddof=EXP3_DDOF) if len(r) > 1 else 0.0)

    y = np.arange(len(conds))[::-1]
    h = 0.38
    a_vals = [strat("A_intra_mrl", ["oof"], st, lv) for st, lv, _ in conds]
    c_vals = [strat(f"C_cew_{cew_variant}_to_mrl", folds5, st, lv) for st, lv, _ in conds]
    ax2.barh(y + h / 2, [v[0] for v in a_vals], h - 0.04, color=C1, label="MRL → MRL", zorder=3)
    ax2.barh(y - h / 2, [v[0] for v in c_vals], h - 0.04, color=C2, label="CEW → MRL", zorder=3)
    ax2.errorbar([v[0] for v in c_vals], y - h / 2, xerr=[v[1] for v in c_vals], fmt="none", zorder=4, **errbar_kw())
    ax2.axvline(50, color=AXIS, lw=0.8, zorder=2)
    ax2.set_yticks(y); ax2.set_yticklabels([lab for _, _, lab in conds], fontsize=FS_S)
    ax2.set_xlim(40, 100); ax2.set_ylim(y[-1] - 0.6, y[0] + 0.6)
    ax2.xaxis.set_major_locator(MultipleLocator(20))
    ax2.xaxis.set_major_formatter(pct_formatter(0))
    light_grid(ax2, "x")
    ax2.set_xlabel("Acur. balanceada no MRL (%)\nMRL→MRL: out-of-fold;\n"
                   "CEW→MRL: média ± dp,\n5 modelos; linha = chance",
                   fontsize=FS_S - 1, color=INK2, linespacing=1.15)
    ax2.set_title("(b) Por condição do MRL Eye", fontsize=FS, pad=3)
    fig.legend(handles=[Patch(color=C1, label="MRL → MRL (sujeitos não vistos)"),
                        Patch(color=C2, label="CEW → MRL (cross-dataset)")],
               loc="outside lower center", ncol=2, fontsize=FS_S - 0.5, handlelength=1.2)
    save(fig, "fig_exp3_cross")


# ----------------------------------------------------------------------------
# Fig. 5 — latência e tamanho TFLite
# ----------------------------------------------------------------------------
def fig_exp4_latencia(lat: pd.DataFrame, blink: dict, phone: pd.DataFrame | None) -> None:
    """(a) desktop + (b) tamanho; se results/exp4_phone_latency.csv existir, (c) smartphone numa
    segunda linha de largura inteira (nenhum valor de celular é inserido à mão)."""
    has_phone = phone is not None and len(phone) > 0
    if has_phone:
        fig = plt.figure(figsize=(COL_W_IN, 4.6), layout="constrained")
        gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 0.9], height_ratios=[1.0, 0.95])
    else:
        fig = plt.figure(figsize=(COL_W_IN, 2.45), layout="constrained")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 0.9])

    # (a) desktop: p50 barra, p95 = whisker
    ax = fig.add_subplot(gs[0, 0])
    rows = []
    for _, r in lat.sort_values(["modelo", "threads"]).iterrows():
        rows.append((f"{r.modelo}\n{int(r.threads)} thr.", r.p50_ms, r.p95_ms, C1 if r.modelo == "fp32" else C2))
    bl = blink["latencia_mac_cpu_ms"]
    rows.append(("LSTM\nfp32", bl["p50"], bl["p95"], C1))
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], 0.6, color=[r[3] for r in rows], zorder=3)
    ax.errorbar(x, [r[1] for r in rows], yerr=[[0] * len(rows), [r[2] - r[1] for r in rows]],
                fmt="none", zorder=4, **errbar_kw())
    for xi, r in zip(x, rows):
        ax.text(xi, r[2] + 0.004, br(r[1], 3), ha="center", va="bottom", fontsize=FS_S - 1, color=INK2, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], fontsize=FS_S - 1)
    ax.set_ylim(0, max(r[2] for r in rows) * 1.6)
    ax.yaxis.set_major_locator(MultipleLocator(0.02))
    ax.yaxis.set_major_formatter(pct_formatter(2))
    ax.set_ylabel("Latência, lote 1 (ms)")
    maq = str(lat.maquina.iloc[0]).split("/")[0].strip()
    ax.set_title(f"(a) Desktop ({maq}, TF {lat.tf_version.iloc[0]})", fontsize=FS_S, pad=3)
    ax.legend(handles=[Patch(color=C1, label="float32"), Patch(color=C2, label="int8"),
                       Line2D([0], [0], color=INK2, lw=0.6, label="p50 → p95")],
              loc="upper right", fontsize=FS_S - 1, handlelength=1.2, borderaxespad=0.2)
    ax.set_xlabel("CNN olho (4 barras) e LSTM de piscadas", fontsize=FS_S - 1, color=INK2)

    # (b) tamanho dos .tflite
    axs = fig.add_subplot(gs[0, 1])
    sizes = [("CNN\nfp32", float(lat[lat.modelo == "fp32"].tamanho_kb.iloc[0]), C1),
             ("CNN\nint8", float(lat[lat.modelo == "int8"].tamanho_kb.iloc[0]), C2),
             ("LSTM\nfp32", float(blink["tamanho_kb"]), C1)]
    xs2 = np.arange(len(sizes))
    axs.bar(xs2, [v for _, v, _ in sizes], 0.6, color=[c for _, _, c in sizes], zorder=3)
    for xi, (_, v, _) in zip(xs2, sizes):
        axs.text(xi, v + 5, br(v, 1), ha="center", va="bottom", fontsize=FS_S - 1, color=INK2)
    axs.set_xticks(xs2); axs.set_xticklabels([n for n, _, _ in sizes], fontsize=FS_S - 1)
    axs.set_ylim(0, max(v for _, v, _ in sizes) * 1.25)
    axs.yaxis.set_major_locator(MultipleLocator(100))
    axs.set_ylabel("Tamanho do .tflite (KB)")
    axs.set_title("(b) Tamanho", fontsize=FS_S, pad=3)

    # (c) smartphone (só se results/exp4_phone_latency.csv existir): latência por quadro e estágio,
    # AGRUPADA POR APARELHO. Barra = mediana, entre as execuções do aparelho, do p50 de cada execução;
    # bigode = maior p95 entre as execuções (pior caso); pontos = p50 de cada execução (dispersão).
    if has_phone:
        axp = fig.add_subplot(gs[1, :])
        devs = phone_devices(phone)
        stages = [("landmarker_ms", "Face\nLandmarker"), ("cnn_ms", "CNN olho\n(2 recortes)"),
                  ("engine_ms", "LSTM\n(piscadas)")]
        if "motor_ms_p50" in phone.columns and phone["motor_ms_p50"].notna().any():
            stages.append(("motor_ms", "Motor de\nsinais v2"))
        stages.append(("total_ms", "Total\npor quadro"))
        n_d = len(devs)
        wid = min(0.78 / max(n_d, 1), 0.36)
        xs = np.arange(len(stages))
        ymax = 0.0
        for di, (dev, g, color) in enumerate(devs):
            p50 = [float(g[f"{st}_p50"].median()) for st, _ in stages]
            p95 = [float(g[f"{st}_p95"].max()) for st, _ in stages]
            pos = xs + (di - (n_d - 1) / 2) * wid
            ok = [np.isfinite(v) for v in p50]
            versoes = ", ".join(sorted({str(v) for v in g.get("versao_app", pd.Series(["n/d"])).dropna()}))
            label = f"{dev}: {len(g)} execuç{'ão' if len(g) == 1 else 'ões'} (app {versoes})"
            axp.bar(pos[ok], np.array(p50)[ok], wid - 0.05, color=color, zorder=3, label=label)
            lo = [0] * sum(ok)
            hi = [max(b - a, 0) for a, b, k in zip(p50, p95, ok) if k]
            axp.errorbar(pos[ok], np.array(p50)[ok], yerr=[lo, hi], fmt="none", zorder=4, **errbar_kw())
            for si, (st, _) in enumerate(stages):   # p50 de cada execução
                vals = g[f"{st}_p50"].dropna().to_numpy(float)
                if len(g) > 1 and len(vals):
                    axp.scatter(np.full(len(vals), pos[si]), vals, s=9, color=SURFACE, edgecolor=INK,
                                linewidth=0.6, zorder=5)
            for (st, _), xi, a, b, k in zip(stages, pos, p50, p95, ok):
                if k:
                    top = max(b, float(g[f"{st}_p50"].max()))
                    axp.text(xi, top + 0.8, br(a, 0) if a >= 10 else br(a, 1), ha="center", va="bottom",
                             fontsize=FS_S - 1, color=INK2)
            ymax = max([ymax] + [v for v, k in zip(p95, ok) if k] +
                       [float(g[f"{st}_p50"].max()) for st, _ in stages if g[f"{st}_p50"].notna().any()])
        axp.set_xticks(xs); axp.set_xticklabels([n for _, n in stages], fontsize=FS_S - 0.5)
        axp.set_ylim(0, ymax * 1.3)
        axp.yaxis.set_major_formatter(pct_formatter(0))
        axp.set_ylabel("Latência por quadro (ms)")
        n_exec = len(phone)
        axp.set_title(f"(c) Smartphone (PWA): {n_d} aparelho{'s' if n_d > 1 else ''}, "
                      f"{n_exec} execuç{'ão' if n_exec == 1 else 'ões'} de 60 s", fontsize=FS_S, pad=3)
        fps = ", ".join(f"{br(float(g.fps_medio.min()), 0)}–{br(float(g.fps_medio.max()), 0)}"
                        if len(g) > 1 else br(float(g.fps_medio.iloc[0]), 0) for _, g, _ in devs)
        axp.set_xlabel(f"barra = mediana dos p50 das execuções; bigode = maior p95\n"
                       f"FPS médio por execução: {fps}", fontsize=FS_S - 1.5, color=INK2)
        handles, labels = axp.get_legend_handles_labels()
        if len(devs) > 1 or any(len(g) > 1 for _, g, _ in devs):
            handles.append(Line2D([0], [0], marker="o", ls="none", markersize=3.2, markerfacecolor=SURFACE,
                                  markeredgecolor=INK, markeredgewidth=0.6))
            labels.append("p50 de cada execução")
        axp.legend(handles, labels, loc="upper left", fontsize=FS_S - 1.5, handlelength=1.2, borderaxespad=0.2)
    save(fig, "fig_exp4_latencia")


# Cores por aparelho (painel c): fixas pela ORDEM DA PRIMEIRA MEDIÇÃO de cada aparelho, para que um
# aparelho novo (ex.: Galaxy S23 Ultra na PUC) não repinte os anteriores. Violeta e aqua da paleta
# categórica da skill dataviz (validadas: ΔE CVD 31,1; aqua < 3:1 com o fundo -> valores rotulados
# nas barras). Não usam o azul/laranja dos painéis (a)/(b), que ali significam float32/int8.
DEVICE_COLORS = ["#4a3aa7", "#1baf7a"]


def phone_devices(phone: pd.DataFrame):
    """[(aparelho, linhas, cor)] na ordem da primeira execução de cada aparelho."""
    ph = phone.copy()
    ph["_ordem"] = ph["data"].fillna("") if "data" in ph.columns else ""
    ph["_ordem"] = ph["_ordem"].where(ph["_ordem"] != "", ph["arquivo"])
    first = ph.groupby("aparelho")["_ordem"].min().sort_values()
    out = []
    for i, dev in enumerate(first.index):
        color = DEVICE_COLORS[i] if i < len(DEVICE_COLORS) else MUTED
        out.append((str(dev), ph[ph.aparelho == dev].drop(columns="_ordem"), color))
    return out


# ----------------------------------------------------------------------------
# Figuras do artigo IEEE (Fig. 2 e Fig. 3): tons de cinza, sem cor
# ----------------------------------------------------------------------------
# Geradas na largura com que entram no artigo (sem redução na inclusão): o menor texto tem 7 pt no PDF final.
# Leem só os CSV de results/ que o make_paper_numbers.py também lê para o texto. A leitura não depende de cor:
# forma do marcador (RF = círculo, RL = quadrado), marcador cheio ou vazado (limiar 0,5 ou calibrado no treino)
# e preenchimento das barras (cinza escuro, cinza claro e branco, sempre com contorno preto).
ART_DIR = ROOT / "paper" / "figures"
ART_COL_IN = 252 / 72.27     # \columnwidth do IEEEtran (conference): 252 pt do TeX = 3,49 in
ART_PAG_IN = 516 / 72.27     # \textwidth: 516 pt do TeX = 7,14 in
ART_FS = 8.0                 # títulos dos painéis e rótulos dos eixos
ART_FS_MIN = 7.0             # marcas dos eixos, legendas e valores: o mínimo no tamanho final
ART_PRETO = "#000000"
ART_CINZA_ESC = "#4d4d4d"
ART_CINZA = "#808080"
ART_CINZA_CLA = "#c8c8c8"
ART_GRADE = "#e6e6e6"
ART_RC = {
    "font.family": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": ART_FS, "axes.titlesize": ART_FS, "axes.labelsize": ART_FS,
    "xtick.labelsize": ART_FS_MIN, "ytick.labelsize": ART_FS_MIN, "legend.fontsize": ART_FS_MIN,
    "text.color": ART_PRETO, "axes.labelcolor": ART_PRETO, "axes.edgecolor": ART_PRETO,
    "xtick.color": ART_PRETO, "ytick.color": ART_PRETO, "xtick.labelcolor": ART_PRETO, "ytick.labelcolor": ART_PRETO,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": ART_GRADE, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.axisbelow": True, "axes.titlepad": 3.0, "axes.labelpad": 2.0,
    "legend.frameon": False, "legend.handlelength": 1.2, "legend.handletextpad": 0.4, "legend.columnspacing": 1.2,
    "legend.borderaxespad": 0.2, "legend.borderpad": 0.2,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def conferir_fontes(fig: plt.Figure, minimo: float = ART_FS_MIN) -> None:
    """Interrompe se algum texto visível da figura ficar abaixo de `minimo` pt no tamanho final."""
    from matplotlib.text import Text
    fig.canvas.draw()
    pequenos = sorted({(round(t.get_fontsize(), 2), t.get_text()) for t in fig.findobj(Text)
                       if t.get_visible() and t.get_text().strip() and t.get_fontsize() < minimo - 1e-9})
    if pequenos:
        raise ValueError(f"texto abaixo de {br(minimo)} pt: {pequenos[:6]}")


def save_artigo(fig: plt.Figure, name: str) -> None:
    """Grava em results/figures/ (PDF e PNG) e copia o PDF para paper/figures/, que é o que o artigo inclui."""
    conferir_fontes(fig)
    save(fig, name)
    ART_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT / f"{name}.pdf", ART_DIR / f"{name}.pdf")
    print(f"  cópia para o artigo: {ART_DIR / (name + '.pdf')}")


def fig_exp5b_transferencia(t: pd.DataFrame) -> None:
    """Fig. 2 do artigo (Exp. 5b, variante principal, B3 na UTA-RLDD): AUC e acurácia por janela e por vídeo nos
    cenários A (DMD→DMD, deixando um sujeito de fora), B (UTA→UTA, 5 folds oficiais) e C (DMD→UTA), com IC 95% por
    bootstrap de sujeitos. Ponto + IC, não barra: a linha de base da AUC e da acurácia balanceada é o acaso (50).
    O DMD (A) não tem nível de vídeo, por isso os painéis por vídeo só têm B e C. O CSV chama a RL de "LR"."""
    def valor(exp, mod, nivel, met, lim):
        q = t[(t.experimento == exp) & (t.variante == "principal") & (t.modelo == mod)
              & (t.esquema == ("DMD" if exp == "A_dmd_intra" else "B3")) & (t.nivel == nivel)
              & (t.estrato == "todos") & (t.limiar_tipo == lim) & (t.metrica == met)]
        if len(q) != 1:
            raise ValueError(f"exp5b_transferencia.csv: {len(q)} linhas para {exp}, {mod}, {nivel}, {met}, {lim}")
        r = q.iloc[0]
        return 100 * r.valor, 100 * r.ic95_inf, 100 * r.ic95_sup

    cenarios = {"A_dmd_intra": "A\nDMD→DMD", "B_uta_intra": "B\nUTA→UTA", "C_dmd_para_uta": "C\nDMD→UTA"}
    fixo, cal = "fixo_0.5", "calibrado_treino"
    paineis = [(0, 0, "auc", "janela", "AUC por janela", (fixo,)),
               (0, 1, "auc", "video", "AUC por vídeo", (fixo,)),
               (1, 0, "acuracia_balanceada", "janela", "Acurácia balanceada por janela", (fixo, cal)),
               (1, 1, "acuracia", "video", "Acurácia por vídeo", (fixo, cal))]
    modelos = {"RF": ("o", 4.6, ART_PRETO), "LR": ("s", 4.0, ART_CINZA)}   # marcador, tamanho (pt), cor

    def exps_do(nivel):
        return [e for e in cenarios if not (nivel == "video" and e == "A_dmd_intra")]

    pontos = []   # (painel, x, valor, IC inferior, IC superior, modelo, limiar)
    for i, j, met, nivel, _, lims in paineis:
        n = len(modelos) * len(lims)
        passo = 0.2 if n == 4 else 0.26
        for k, exp in enumerate(exps_do(nivel)):
            pos = 0
            for mod in modelos:
                for lim in lims:
                    pontos.append(((i, j), k + (pos - (n - 1) / 2) * passo, *valor(exp, mod, nivel, met, lim), mod, lim))
                    pos += 1
    ymin = max(0.0, min(40.0, np.floor(min(p[3] for p in pontos) / 10.0) * 10.0))

    with plt.rc_context(ART_RC):
        fig = plt.figure(figsize=(ART_COL_IN, 3.4), layout="constrained")
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 2])
        axes = {}
        for i, j, met, nivel, titulo, lims in paineis:
            ax = fig.add_subplot(gs[i, j], sharey=axes.get((0, 0)))
            axes[(i, j)] = ax
            exps = exps_do(nivel)
            ax.axhline(50, color=ART_CINZA, lw=0.7, ls=(0, (3, 2)), zorder=1)   # acaso
            ax.set_xticks(range(len(exps)))
            ax.set_xticklabels([cenarios[e] for e in exps] if i == 1 else [""] * len(exps), linespacing=1.15)
            ax.tick_params(axis="x", length=0)
            ax.set_xlim(-0.5, len(exps) - 0.5)
            ax.set_ylim(ymin, 100)
            ax.yaxis.set_major_locator(MultipleLocator(10))
            ax.yaxis.set_major_formatter(pct_formatter(0))
            ax.set_title(titulo)
            if j == 0:
                ax.set_ylabel("AUC × 100" if i == 0 else "Acurácia (%)")
            else:
                ax.tick_params(axis="y", labelleft=False)
        for painel, x, v, lo, hi, mod, lim in pontos:
            marcador, ms, cor = modelos[mod]
            ax = axes[painel]
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=cor, elinewidth=0.8, capsize=1.8,
                        capthick=0.8, zorder=3)
            ax.plot(x, v, marcador, ms=ms, mfc=cor if lim == fixo else "white", mec=cor, mew=0.9, zorder=4)
        cheio = {m: Line2D([], [], marker=s[0], ms=s[1], ls="none", mfc=s[2], mec=s[2], mew=0.9)
                 for m, s in modelos.items()}
        vazado = {m: Line2D([], [], marker=s[0], ms=s[1], ls="none", mfc="white", mec=s[2], mew=0.9)
                  for m, s in modelos.items()}
        # legenda em duas colunas: modelos (forma do marcador) e limiares (cheio ou vazado)
        fig.legend([cheio["RF"], cheio["LR"], (cheio["RF"], cheio["LR"]), (vazado["RF"], vazado["LR"])],
                   ["RF", "RL", "limiar 0,5", "limiar calibrado no treino"], loc="outside upper center", ncol=2,
                   handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)}, handlelength=1.7, columnspacing=2.5)
        save_artigo(fig, "fig_exp5b_transferencia")


def fig_exp6_noite(L: pd.DataFrame, V: pd.DataFrame) -> None:
    """Fig. 3 do artigo (Exp. 6; características brutas, que são a variante principal do Exp. 6).
    (A) SNR do EAR por vídeo (amplitude das piscadas ÷ ruído quadro a quadro): ponto = vídeo, traço = mediana;
        NITYMED (noite), DMD de dia com o carro em movimento e parado (trechos de direção segura) e UTA-RLDD.
    (B) Taxa de disparo: fração das janelas de 10 s marcadas como sonolentas no limiar 0,5, com IC 95%, pelo RF
        treinado no DMD, pelo RF treinado na UTA-RLDD (B3) e pela regra do rótulo do DMD (fechamento ≥ 0,5 s ou
        bocejo). Grupos, filtros e nomes dos modelos: os de exp6_noite.py (LINHAS, FONTE_NOME, VAR_PRINCIPAL).
    O painel de quadros sem rosto saiu da figura; a taxa de detecção de rosto continua no texto e no relatório."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import exp6_noite as E6

    def linha(**filtros):
        q = L
        for k, v in filtros.items():
            q = q[q[k].isna()] if v is None else q[q[k] == v]
        if len(q) != 1:
            raise ValueError(f"exp6_noite.csv: {len(q)} linhas para {filtros}")
        return q.iloc[0]

    grupos_snr = [("nitymed|todos", "NITYMED\n(noite)"), ("dmd_mov|safe", "DMD em\nmovimento"),
                  ("dmd_par|safe", "DMD\nparado"), ("uta|todos", "UTA-RLDD")]
    grupos_disp = [("uta", "KSS 0", "UTA\nalerta"), ("dmd", "alerta_parado", "DMD alerta\nparado"),
                   ("dmd", "alerta_movimento", "DMD alerta\nem movimento"), ("uta", "KSS 10", "UTA\nsonolento"),
                   ("nitymed", "Yawning", "NITYMED\nbocejo"), ("nitymed", "Microsleep", "NITYMED\nmicrossono")]
    series = [(E6.FONTE_NOME["dmd"], E6.VAR_PRINCIPAL, "RF do DMD", ART_CINZA_ESC),
              (E6.FONTE_NOME["uta"], E6.VAR_PRINCIPAL, "RF da UTA-RLDD", ART_CINZA_CLA),
              ("regra_dmd", None, "Regra (fechamento ≥ 0,5 s ou bocejo)", "white")]

    with plt.rc_context(ART_RC):
        fig = plt.figure(figsize=(ART_PAG_IN, 2.35), layout="constrained")
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.8])
        a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

        # (A) SNR do EAR por vídeo; o traço é a mediana registrada em exp6_noite.csv (conferida com os pontos)
        rng = np.random.default_rng(42)   # só o espalhamento horizontal dos pontos (semente dos experimentos)
        for i, (chave, _) in enumerate(grupos_snr):
            v = E6.sub(V, chave)["snr_rapido"].to_numpy(float)
            v = v[np.isfinite(v)]
            med = float(linha(secao="b", grupo=chave, metrica="snr_rapido_mediana_por_video")["valor"])
            if not np.isclose(med, np.median(v)):
                raise ValueError(f"{chave}: mediana {med} em exp6_noite.csv e {np.median(v)} em exp6_noite_videos.csv")
            a1.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=5, color=ART_CINZA, alpha=0.6, linewidths=0,
                       zorder=3)
            a1.plot([i - 0.3, i + 0.3], [med, med], color=ART_PRETO, lw=1.6, zorder=4, solid_capstyle="butt")
        a1.set_xticks(range(len(grupos_snr)), [r for _, r in grupos_snr], linespacing=1.15)
        a1.tick_params(axis="x", length=0)
        a1.set_xlim(-0.55, len(grupos_snr) - 0.45)
        a1.set_ylim(bottom=0)
        a1.yaxis.set_major_locator(MultipleLocator(20))
        a1.set_ylabel("SNR do EAR (amplitude ÷\nruído quadro a quadro)")
        a1.set_title("(A) Qualidade do sinal", loc="left")

        # (B) taxa de disparo no limiar 0,5, com IC 95% (bootstrap por agrupamento, como no relatório)
        larg = 0.26
        for j, (modelo, variante, rotulo, cor) in enumerate(series):
            for i, (base, grupo, _) in enumerate(grupos_disp):
                r = linha(secao="c", base=base, grupo=grupo, modelo=modelo, variante=variante,
                          metrica="taxa_disparo_janela")
                v, lo, hi = 100 * r.valor, 100 * r.ic95_inf, 100 * r.ic95_sup
                x = i + (j - 1) * larg
                a2.bar(x, v, larg, color=cor, edgecolor=ART_PRETO, linewidth=0.6, zorder=3,
                       label=rotulo if i == 0 else None)
                a2.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", ecolor=ART_PRETO, elinewidth=0.7,
                            capsize=1.5, capthick=0.7, zorder=4)
                a2.text(x, hi + 1.5, br(v, 0), ha="center", va="bottom", fontsize=ART_FS_MIN, zorder=5)
        a2.set_xticks(range(len(grupos_disp)), [r for _, _, r in grupos_disp], linespacing=1.15)
        a2.tick_params(axis="x", length=0)
        a2.set_xlim(-0.55, len(grupos_disp) - 0.45)
        a2.set_ylim(0, 105)
        a2.yaxis.set_major_locator(MultipleLocator(20))
        a2.axvline(2.5, color=ART_CINZA, lw=0.6, zorder=1)   # pessoas acordadas à esquerda
        a2.set_ylabel("Janelas marcadas como\nsonolentas (%)")
        a2.set_title("(B) Taxa de disparo (limiar 0,5; IC 95%)", loc="left")
        fig.legend(*a2.get_legend_handles_labels(), loc="outside upper right", ncol=3, handlelength=1.4)
        save_artigo(fig, "fig_exp6_noite")


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cew-variant", choices=["patch", "crop"], default="patch",
                    help="recorte do CEW usado na fig_exp3_cross (padrão: patch = 24x24 oficial)")
    ap.add_argument("--only", nargs="*", default=None, help="gerar só estas figuras (nomes sem extensão)")
    args = ap.parse_args()
    setup_style()
    todo = set(args.only) if args.only else None

    def want(name):
        return todo is None or name in todo

    if want("fig_exp1_va_por_fold") or want("fig_exp1_bsa_vs_va"):
        e1 = load_exp1()
        if want("fig_exp1_va_por_fold"):
            fig_exp1_va_por_fold(e1)
        if want("fig_exp1_bsa_vs_va"):
            fig_exp1_bsa_vs_va(e1)
    if want("fig_exp2_vazamento"):
        fig_exp2_vazamento(load_exp2())
    if want("fig_exp3_cross"):
        d, s = load_exp3(args.cew_variant)
        fig_exp3_cross(d, s, args.cew_variant)
    if want("fig_exp4_latencia"):
        lat, blink, phone = load_exp4()
        if phone is None:
            print("aviso: results/exp4_phone_latency.csv não existe — fig_exp4_latencia só com o desktop "
                  "(rode tools/import_phone_benchmark.py results/phone ou bash tools/atualizar_celular.sh).")
        fig_exp4_latencia(lat, blink, phone)
    if want("fig_exp5b_transferencia"):
        fig_exp5b_transferencia(pd.read_csv(RES / "exp5b_transferencia.csv"))
    if want("fig_exp6_noite"):
        fig_exp6_noite(pd.read_csv(RES / "exp6_noite.csv"),
                       pd.read_csv(RES / "exp6_noite_videos.csv", low_memory=False))


if __name__ == "__main__":
    main()
