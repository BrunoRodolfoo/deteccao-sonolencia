"""Figura de amostras (só recortes de olhos, sem rostos inteiros): MRL Eye, aberto/fechado x óculos x
iluminação. Só o MRL Eye: a licença do CEW proíbe redistribuição e o repositório é público, então
nenhum recorte do CEW vai para figuras. Saída: results/figures/amostras_mrl.{png,pdf}, 300 dpi."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pipeline_b import datasets as D  # noqa: E402

OUT = ROOT / "results" / "figures"
SEED = 42
N_EX = 3  # exemplos por condição


def main():
    rng = np.random.default_rng(SEED)
    x_mrl, mrl = D.load_mrl()

    conds = [(0, 1, "sem óculos\nilum. boa"), (0, 0, "sem óculos\nilum. ruim"),
             (1, 1, "com óculos\nilum. boa"), (1, 0, "com óculos\nilum. ruim")]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
    fig = plt.figure(figsize=(7.2, 3.9))
    gs = fig.add_gridspec(1, 2, wspace=0.12, top=0.80)

    # ---- 2 painéis (aberto / fechado), cada um 4 condições x N_EX (mesmo sorteio da versão com o CEW) ----
    for col, (state, title) in enumerate([(1, "MRL Eye: olho ABERTO (1)"), (0, "MRL Eye: olho FECHADO (0)")]):
        sub = fig.add_subplot(gs[0, col])
        sub.set_title(title, fontsize=8, fontweight="bold")
        sub.axis("off")
        inner = gs[0, col].subgridspec(N_EX, 4, wspace=0.05, hspace=0.05)
        for c, (gl, li, name) in enumerate(conds):
            idx = np.where((mrl.state.values == state) & (mrl.glasses.values == gl) & (mrl.lighting.values == li))[0]
            pick = rng.choice(idx, N_EX, replace=False)
            for r in range(N_EX):
                ax = fig.add_subplot(inner[r, c])
                ax.imshow(x_mrl[pick[r]], cmap="gray", vmin=0, vmax=255)
                ax.set_xticks([]); ax.set_yticks([])
                if r == N_EX - 1:
                    ax.set_xlabel(name, fontsize=6.5)
    fig.suptitle("Amostras de recortes de olhos do MRL Eye usados no Exp. 3\n"
                 "(48x48, escala de cinza, infravermelho, 37 sujeitos; 3 exemplos por condição)", fontsize=8)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "amostras_mrl.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "amostras_mrl.pdf", bbox_inches="tight")
    print("figura salva em", OUT / "amostras_mrl.png")


if __name__ == "__main__":
    main()
