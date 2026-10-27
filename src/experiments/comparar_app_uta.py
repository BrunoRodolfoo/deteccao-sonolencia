#!/usr/bin/env python3
"""Compara o app com os modelos do Exp. 5b nas MESMAS pessoas do teste do app em sono real (UTA-RLDD).

Rodar no .venv, depois de `tools/app_validacao/validar_motor_uta.mjs` (resultado do app por vídeo) e do Exp. 5b
(escores por vídeo) e do Exp. 6 (regra do rótulo do DMD por janela):

    .venv/bin/python src/experiments/comparar_app_uta.py

Para cada detector, a AUC por vídeo entre o vídeo sonolento (KSS 8 e 9) e o acordado (KSS 1 a 3) dos sujeitos
testados no app, com IC 95% por bootstrap de sujeitos (1.000 reamostragens, semente 42; cada sujeito sorteado entra
com os dois vídeos). Detectores:
  - app 2.1: % do tempo fora do verde e nível médio (results/app_validacao_uta.csv);
  - Exp. 5b, C (treino no DMD) e B (treino na própria UTA-RLDD, 5 folds oficiais), RF e RL, com e sem centragem:
    escore médio das janelas do vídeo (results/exp5b_predicoes_video.csv, esquema B3);
  - regra do rótulo do DMD aplicada aos nossos sinais (fechamento de 0,5 s ou mais ou bocejo): fração de janelas
    marcadas no vídeo (results/exp6_noite_janelas.csv).
O app não foi treinado nem ajustado nessas pessoas; B foi treinado na UTA-RLDD (nos outros folds), e a sua versão
centrada usa a referência tirada do próprio vídeo alerta (ver results/exp5b_robustez.md).

Saídas: results/app_comparacao_uta.csv e results/app_comparacao_uta.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
CSV_OUT = RES / "app_comparacao_uta.csv"
MD_OUT = RES / "app_comparacao_uta.md"
SEED = 42
N_BOOT = 1000


def v3(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"])


def main() -> int:
    app = pd.read_csv(RES / "app_validacao_uta.csv")
    sujeitos = sorted(app["sujeito"].unique())
    app = app.assign(y=(app["rotulo"] == 10).astype(int))
    detectores = {
        ("app_2.1", "tempo_fora_do_verde"): app.set_index("video")[["sujeito", "y", "fora_do_verde_pct"]]
        .rename(columns={"fora_do_verde_pct": "escore"}),
        ("app_2.1", "nivel_medio"): app.set_index("video")[["sujeito", "y", "nivel_medio"]]
        .rename(columns={"nivel_medio": "escore"}),
    }
    pv = pd.read_csv(RES / "exp5b_predicoes_video.csv")
    pv = pv[(pv["esquema"] == "B3") & pv["sujeito"].isin(sujeitos)]
    for exp, et in [("C_dmd_para_uta", "C_treino_DMD"), ("B_uta_intra", "B_treino_UTA")]:
        for var in ("principal", "sem_normalizacao"):
            for mod in ("RF", "LR"):
                q = pv[(pv["experimento"] == exp) & (pv["variante"] == var) & (pv["modelo"] == mod)]
                detectores[(f"{et}_{mod}", var)] = q.set_index("video")[["sujeito", "y", "escore_medio"]] \
                    .rename(columns={"escore_medio": "escore"})
    j = pd.read_csv(RES / "exp6_noite_janelas.csv", low_memory=False)
    j = j[(j["dataset"] == "uta") & j["sujeito"].isin(sujeitos) & _bool(j["avaliavel"])]
    j = j.assign(kss=j["grupo"].str.replace("KSS ", "", regex=False).astype(int), regra=_bool(j["regra_dmd"]))
    j = j[j["kss"] != 5]
    regra = j.groupby("video").agg(sujeito=("sujeito", "first"), y=("kss", lambda s: int(s.iloc[0] == 10)),
                                   escore=("regra", "mean"))
    detectores[("regra_rotulo_DMD", "fracao_janelas")] = regra

    rng = np.random.default_rng(SEED)
    sorteios = [rng.choice(sujeitos, len(sujeitos), replace=True) for _ in range(N_BOOT)]
    linhas = []
    for (det, medida), d in detectores.items():
        assert len(d) == 2 * len(sujeitos) and set(d["sujeito"]) == set(sujeitos), (det, medida, len(d))
        auc = roc_auc_score(d["y"], d["escore"])
        por_suj = {s: g for s, g in d.groupby("sujeito")}
        boot = []
        for sorteio in sorteios:
            g = pd.concat([por_suj[s] for s in sorteio])
            boot.append(roc_auc_score(g["y"], g["escore"]))
        lo, hi = np.percentile(boot, [2.5, 97.5])
        linhas.append(dict(detector=det, medida=medida, auc=auc, ic95_inf=lo, ic95_sup=hi,
                           n_videos=len(d), n_sujeitos=len(sujeitos)))
    df = pd.DataFrame(linhas)
    df.to_csv(CSV_OUT, index=False)

    nomes = {"app_2.1": "app 2.1 (sem treino nesta base)", "C_treino_DMD_RF": "RF treinado no DMD (C)",
             "C_treino_DMD_LR": "RL treinada no DMD (C)", "B_treino_UTA_RF": "RF treinado na UTA-RLDD (B)",
             "B_treino_UTA_LR": "RL treinada na UTA-RLDD (B)", "regra_rotulo_DMD": "regra do rótulo do DMD (sem treino)"}
    medidas = {"tempo_fora_do_verde": "% do tempo fora do verde", "nivel_medio": "nível médio",
               "principal": "escore médio, com centragem", "sem_normalizacao": "escore médio, sem centragem",
               "fracao_janelas": "fração de janelas marcadas"}
    L = ["# App × modelos nas mesmas pessoas da UTA-RLDD", "",
         f"Gerado por `src/experiments/comparar_app_uta.py`. {len(sujeitos)} sujeitos (os do teste do app em "
         "`results/app_validacao_uta.md`), um vídeo acordado e um sonolento de cada. AUC por vídeo (sonolento × "
         "acordado); IC 95% por bootstrap de sujeitos (1.000 reamostragens, semente 42).", "",
         "| Detector | Medida | AUC por vídeo [IC 95%] |", "|---|---|---|"]
    for r in df.itertuples():
        L.append(f"| {nomes[r.detector]} | {medidas[r.medida]} | {v3(r.auc)} [{v3(r.ic95_inf)}; {v3(r.ic95_sup)}] |")
    L += ["", "Leitura: o app, sem treino nem ajuste nesta base, fica no nível do RF treinado no DMD e perto da regra "
          "do rótulo, que é a base dos seus sinais de olho. O RF treinado na própria UTA-RLDD só fica bem acima com a "
          "centragem pelo trecho final do vídeo alerta, o atalho discutido em `results/exp5b_robustez.md`; sem "
          "centragem, ele fica no mesmo nível do app."]
    MD_OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(df.to_string(index=False))
    print(f"-> {CSV_OUT.relative_to(ROOT)}, {MD_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
