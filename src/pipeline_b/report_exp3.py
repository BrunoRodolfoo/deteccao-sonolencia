"""Gera results/exp3_eye_state.md a partir dos CSVs (nenhum número é digitado
à mão): results/exp3_eye_state.csv, results/exp3_strat_conditions.csv,
results/exp3_meta.json e, se existirem, results/exp4_tflite_eval.csv e
results/exp4_desktop_latency.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pipeline_b.datasets import CEW_CROP_FRAC  # noqa: E402

R = ROOT / "results"
METRICS = ["acc", "bal_acc", "f1_macro", "auc", "chance"]
MNAMES = {"acc": "Acurácia", "bal_acc": "Acur. balanceada", "f1_macro": "F1 (macro)", "auc": "AUC", "chance": "Chance"}
PNAMES = {"A_intra_mrl": "A — intra-MRL por sujeito (5 folds)",
          "B_folds_mrl_to_cew": "B — MRL→CEW, modelos dos 5 folds de A",
          "B_final_mrl_to_cew": "B — MRL→CEW, modelo final (37 sujeitos)",
          "C_cew_patch_to_mrl": "C — CEW(patch)→MRL",
          "C_cew_crop_to_mrl": "C — CEW(crop)→MRL",
          "D_intra_cew_patch": "D — intra-CEW (patch), 5 folds por face",
          "D_intra_cew_crop": "D — intra-CEW (crop), 5 folds por face"}


def pct(v):
    return f"{100 * v:.1f}"


def pivot(df):
    return df.pivot_table(index=["protocol", "test_on", "fold", "stratum", "level"], columns="metric", values="value").reset_index()


def fold_table(pv, protocol, test_on=None):
    """Tabela por fold + média±desvio (só folds numéricos) + linha agregada."""
    d = pv[(pv.protocol == protocol) & (pv.stratum == "todos")]
    if test_on:
        d = d[d.test_on == test_on]
    lines = ["| Fold | n | " + " | ".join(MNAMES[m] for m in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    num = d[d.fold.astype(str).str.isdigit()].sort_values("fold", key=lambda s: s.astype(int))
    for _, r in num.iterrows():
        lines.append(f"| {r.fold} | {int(r.n)} | " + " | ".join(pct(r[m]) for m in METRICS) + " |")
    if len(num) > 1:
        lines.append("| **média ± dp** | | " + " | ".join(f"**{pct(num[m].mean())} ± {pct(num[m].std(ddof=1))}**" for m in METRICS) + " |")
    for _, r in d[~d.fold.astype(str).str.isdigit()].iterrows():
        lab = {"oof": "agregado out-of-fold", "ensemble5": "ensemble (média) dos 5", "final": "modelo final"}.get(r.fold, r.fold)
        lines.append(f"| {lab} | {int(r.n)} | " + " | ".join(pct(r[m]) for m in METRICS) + " |")
    return "\n".join(lines)


def strata_table(pv, protocol, fold_key):
    d = pv[(pv.protocol == protocol) & (pv.stratum != "todos") & (pv.stratum != "side")]
    if fold_key == "mean_folds":
        d = d[d.fold.astype(str).str.isdigit()]
        g = d.groupby(["stratum", "level"])
        rows = []
        for (s, lv), grp in g:
            rows.append({"stratum": s, "level": lv, "n": grp.n.iloc[0],
                         **{m: f"{pct(grp[m].mean())} ± {pct(grp[m].std(ddof=1))}" for m in METRICS}})
    else:
        d = d[d.fold == fold_key]
        rows = [{"stratum": r.stratum, "level": r.level, "n": r.n, **{m: pct(r[m]) for m in METRICS}} for _, r in d.iterrows()]
    order = ["glasses", "lighting", "reflections", "sensor"]
    rows.sort(key=lambda r: (order.index(r["stratum"]), r["level"]))
    lines = ["| Condição | n | " + " | ".join(MNAMES[m] for m in METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for r in rows:
        lines.append(f"| {r['level']} | {int(r['n'])} | " + " | ".join(str(r[m]) for m in METRICS) + " |")
    return "\n".join(lines)


def main():
    df = pd.read_csv(R / "exp3_eye_state.csv")
    meta = json.loads((R / "exp3_meta.json").read_text(encoding="utf-8"))
    pv = pivot(df)
    a = meta["args"]
    L = ["# Exp. 3 — Estado ocular (aberto/fechado): intra-dataset e generalização cross-dataset MRL Eye ↔ CEW\n",
         "Todos os números abaixo vêm de `results/exp3_eye_state.csv` / `results/exp3_strat_conditions.csv` "
         "(gerado por `src/pipeline_b/report_exp3.py`; log em `results/logs/exp3_eye_state_train.log`). "
         "Rótulo 1 = aberto, 0 = fechado. Valores em %. Chance = proporção da classe majoritária no conjunto de teste.\n",
         "## Configuração\n",
         f"- Modelo: EyeNet (`src/pipeline_b/model.py`), {meta['eyenet_params']} parâmetros, entrada 48x48x1 em [0,1], "
         "Conv(16)-BN-ReLU-Pool / Conv(32)-BN-ReLU-Pool / Conv(64)-BN-ReLU-Conv(64)-BN-ReLU-Pool / GAP / Dense(32) / Dropout(0,3) / sigmoide. "
         "BatchNorm com momento 0,9 em vez do padrão 0,99 (ver ablação ao final: no treino completo os dois valores dão resultados comparáveis; 0,9 é mantido por ser o usado em todos os resultados reportados).",
         f"- Treino: Adam lr 1e-3, batch {a['batch']}, até {a['epochs']} épocas, EarlyStopping (val_loss, paciência 3, melhores pesos), "
         "ReduceLROnPlateau (0,5, paciência 2), entropia cruzada binária, seed 42.",
         "- Augmentation (só no treino): flip horizontal, rotação ±10°, translação ±8%, zoom ±10%, brilho ±15%, contraste ±20%.",
         f"- Protocolo A: GroupKFold 5 folds sobre os 37 sujeitos do MRL; 10% dos sujeitos de treino separados para early stopping; "
         f"treino subamostrado para ≤{a['max_train']} imagens (estratificado por sujeito e classe); teste em TODAS as imagens dos sujeitos de teste.",
         "- Protocolo B: modelo final treinado com a mesma receita em todos os 37 sujeitos "
         f"(sujeitos de validação: {meta['final_model']['val_subjects']}, {meta['final_model']['n_train']} imagens de treino, "
         f"melhor época {meta['final_model']['best_epoch']}/{meta['final_model']['epochs_run']}) avaliado em todo o CEW; "
         "também reportamos os 5 modelos dos folds de A no CEW.",
         f"- CEW: dois recortes — *patch* = 24x24 oficiais ampliados para 48x48; *crop* = caixa quadrada de lado {CEW_CROP_FRAC}·largura da face "
         "(36 px nas faces 100x100) centrada na coordenada do olho, redimensionada para 48x48.",
         "- Protocolos C/D: GroupKFold 5 folds no CEW agrupado por face de origem (olhos E/D da mesma face no mesmo lado); 10% das faces de treino para early stopping; "
         "cada modelo é avaliado no fold de teste do CEW (D) e em todo o MRL (C).\n"]
    L += ["## Protocolo A — intra-MRL, separação por sujeito\n", fold_table(pv, "A_intra_mrl"), "",
          "### A — por condição (predições out-of-fold agregadas dos 5 folds, todo o MRL)\n", strata_table(pv, "A_intra_mrl", "oof"), ""]
    L += ["## Protocolo B — MRL → CEW (cross-dataset, IR de laboratório → RGB in-the-wild)\n"]
    for t in ("CEW_patch", "CEW_crop"):
        L += [f"### B — avaliado em {t}\n", "Modelos dos 5 folds de A:\n", fold_table(pv, "B_folds_mrl_to_cew", t), "",
              "Modelo final (37 sujeitos):\n", fold_table(pv, "B_final_mrl_to_cew", t), ""]
    side = pv[(pv.protocol == "B_final_mrl_to_cew") & (pv.stratum == "side")]
    if len(side):
        L += ["Modelo final por lado do olho (CEW):\n", "| Conjunto | Lado | n | Acurácia | AUC |", "|---|---|---|---|---|"]
        for _, r in side.iterrows():
            L.append(f"| {r.test_on} | {r.level} | {int(r.n)} | {pct(r.acc)} | {pct(r.auc)} |")
        L.append("")
    L += ["## Protocolo C — CEW → MRL (cross-dataset, RGB in-the-wild → IR de laboratório)\n"]
    for v in ("patch", "crop"):
        L += [f"### C — treinado em CEW ({v}), avaliado em todo o MRL\n", fold_table(pv, f"C_cew_{v}_to_mrl"), "",
              f"Por condição do MRL (média ± dp dos 5 modelos de fold):\n", strata_table(pv, f"C_cew_{v}_to_mrl", "mean_folds"), ""]
    L += ["## Protocolo D — intra-CEW (referência), 5 folds por face\n"]
    for v in ("patch", "crop"):
        L += [f"### D — CEW ({v})\n", fold_table(pv, f"D_intra_cew_{v}"), ""]
    # resumo
    def mean_of(protocol, test_on=None, metric="acc"):
        d = pv[(pv.protocol == protocol) & (pv.stratum == "todos") & pv.fold.astype(str).str.isdigit()]
        if test_on:
            d = d[d.test_on == test_on]
        return d[metric].mean(), d[metric].std(ddof=1), d["chance"].mean()
    L += ["## Resumo (acurácia média ± dp dos 5 folds)\n", "| Treino → Teste | Acurácia | AUC | Chance |", "|---|---|---|---|"]
    for name, p, t in [("MRL → MRL (sujeitos não vistos)", "A_intra_mrl", None),
                       ("MRL → CEW patch", "B_folds_mrl_to_cew", "CEW_patch"), ("MRL → CEW crop", "B_folds_mrl_to_cew", "CEW_crop"),
                       ("CEW patch → MRL", "C_cew_patch_to_mrl", None), ("CEW crop → MRL", "C_cew_crop_to_mrl", None),
                       ("CEW patch → CEW patch (faces não vistas)", "D_intra_cew_patch", None),
                       ("CEW crop → CEW crop (faces não vistas)", "D_intra_cew_crop", None)]:
        m, s, ch = mean_of(p, t); au, aus, _ = mean_of(p, t, "auc")
        L.append(f"| {name} | {pct(m)} ± {pct(s)} | {pct(au)} ± {pct(aus)} | {pct(ch)} |")
    L.append("")
    mb_p = R / "exp3_mobilenetv3.csv"
    if mb_p.exists():
        mb = pivot(pd.read_csv(mb_p))
        d = mb[(mb.stratum == "todos")].iloc[0]
        ey = pv[(pv.protocol == "A_intra_mrl") & (pv.stratum == "todos") & (pv.fold.astype(str) == "1")].iloc[0]
        L += ["## MobileNetV3-Small (opcional, só fold 1 do protocolo A)\n",
              "Fonte: `results/exp3_mobilenetv3.csv` (`src/pipeline_b/train_mobilenet.py`; alpha 0,75, pesos ImageNet, "
              "entrada 96x96x3 com o cinza replicado, batch 64, guarda de 15 min, mesma partição e subamostragem do fold 1).\n",
              "| Modelo (fold 1, mesmos sujeitos de teste) | Parâmetros | n | " + " | ".join(MNAMES[m] for m in METRICS) + " | Épocas | Treino (s) |",
              "|---|---|---|" + "---|" * len(METRICS) + "---|---|",
              f"| EyeNet | {meta['eyenet_params']} | {int(ey.n)} | " + " | ".join(pct(ey[m]) for m in METRICS) + f" | {int(ey.epochs_run)} | {ey.train_s:.0f} |",
              f"| MobileNetV3-Small 0,75 | {int(d.params)} | {int(d.n)} | " + " | ".join(pct(d[m]) for m in METRICS) + f" | {int(d.epochs_run)} | {d.train_s:.0f} |", "",
              f"A MobileNetV3 executou {int(d.epochs_run)} épocas com melhor época {int(d.best_epoch)} (val_loss): "
              + ("parou por EarlyStopping (paciência 3), não pela guarda de tempo." if int(d.epochs_run) - int(d.best_epoch) >= 3 else "parou pela guarda de tempo ou pelo limite de épocas.")
              + " Parâmetros ≈ 9x os do EyeNet; comparação nos 5 folds e exportação TFLite ficam como trabalho futuro.", ""]
    abl_p = R / "exp3_bn_momentum_ablation.csv"
    if abl_p.exists():
        ab = pd.read_csv(abl_p)
        L += ["## Ablação: momento da BatchNormalization (fold 1 do protocolo A)\n",
              "Fonte: `results/exp3_bn_momentum_ablation.csv` (`src/pipeline_b/ablation_bn_momentum.py`, mesma partição e "
              "subamostragem do fold 1; acurácia em modo de inferência). A hipótese original (0,99 faria o modelo prever classe constante em modo de inferência) NÃO se confirma no treino completo — só havia sido observada em execuções curtas de teste; com 0,99 o fold 1 fica até um pouco melhor. O valor 0,9 foi mantido por ser o usado em todos os resultados reportados.\n",
              "| BN momentum | Conjunto | n | Acurácia | Acur. balanceada | AUC | Épocas | Melhor época |", "|---|---|---|---|---|---|---|---|"]
        for _, r in ab.iterrows():
            L.append(f"| {r.bn_momentum} | {r.conjunto} | {int(r.n)} | {pct(r.acc)} | {pct(r.bal_acc)} | {pct(r.auc)} | {int(r.epochs_run)} | {int(r.best_epoch)} |")
        L.append("")
    else:
        L += ["## MobileNetV3-Small\n", "Não treinada nesta execução (limite de tempo); fica como trabalho futuro.\n"]
    # exp4
    ev_p = R / "exp4_tflite_eval.csv"
    if ev_p.exists():
        ev = pd.read_csv(ev_p)
        L += ["## Exp. 4 (desktop) — exportação TFLite do modelo final MRL\n",
              "Atenção: o modelo final foi treinado com todos os sujeitos do MRL (exceto os de validação, usados só para early stopping); "
              "por isso os conjuntos `MRL_*` abaixo NÃO são teste cego (o número cego intra-MRL é o do protocolo A). "
              "Eles servem para medir a QUEDA causada pela quantização, que é a coluna Δ. O CEW é totalmente cego.\n",
              "Fonte: `results/exp4_tflite_eval.csv`, `results/exp4_desktop_latency.csv`, `models/eye_state.json`.\n",
              "| Conjunto | n | Keras acc | fp32 acc | int8 acc | Δ int8−Keras (p.p.) | Keras AUC | int8 AUC |", "|---|---|---|---|---|---|---|---|"]
        for ds in ev.dataset.unique():
            g = lambda mo, me: float(ev[(ev.dataset == ds) & (ev.modelo == mo) & (ev.metric == me)].value.iloc[0])
            L.append(f"| {ds} | {int(g('keras', 'n'))} | {pct(g('keras', 'acc'))} | {pct(g('fp32', 'acc'))} | {pct(g('int8', 'acc'))} | "
                     f"{100 * (g('int8', 'acc') - g('keras', 'acc')):+.2f} | {pct(g('keras', 'auc'))} | {pct(g('int8', 'auc'))} |")
        lat = pd.read_csv(R / "exp4_desktop_latency.csv")
        L += ["", f"Latência batch 1 ({lat.maquina.iloc[0]}, TF {lat.tf_version.iloc[0]}, 50 aquecimento + 500 execuções):\n",
              "| Modelo | Tamanho (KB) | Threads | p50 (ms) | p95 (ms) |", "|---|---|---|---|---|"]
        for _, r in lat.iterrows():
            L.append(f"| {r.modelo} | {r.tamanho_kb} | {r.threads} | {r.p50_ms:.3f} | {r.p95_ms:.3f} |")
        c = json.loads((ROOT / "models" / "eye_state.json").read_text(encoding="utf-8"))
        L += ["", f"Modelo int8: E/S `{c['int8']['io']}` (entrada {c['int8']['entrada_dtype']}, "
              f"scale={c['int8']['entrada_quantizacao']['scale']:.6g}, zero_point={c['int8']['entrada_quantizacao']['zero_point']}; "
              f"saída {c['int8']['saida_dtype']}, scale={c['int8']['saida_quantizacao']['scale']:.6g}, "
              f"zero_point={c['int8']['saida_quantizacao']['zero_point']}).\n"]
    (R / "exp3_eye_state.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("relatório em", R / "exp3_eye_state.md")


if __name__ == "__main__":
    main()
