"""Baseline do projeto: classificadores clássicos sobre as blink features do
UTA-RLDD, avaliados nas 5 partições oficiais (divisão por sujeito).

Referências de comparação (Ghoddoosian et al., CVPRW 2019, 3 classes):
  - HM-LSTM (modelo temporal dos autores): 65.2% de acurácia média
  - julgamento humano: 57.8%

Uso:
    python src/baseline/train.py

Saídas:
  - results/logs/baseline_<timestamp>.log  (log completo da execução)
  - results/baseline_uta_rldd.md           (tabela de resultados)
  - models/<modelo>_fold<k>.pkl            (classificadores treinados)
"""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from load_folds import N_FOLDS, load_fold, sequences_flat, sequences_to_features

ROOT = Path(__file__).resolve().parents[2]
SEED = 42


def build_models():
    return {
        "LogReg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, random_state=SEED)),
        "SVM-RBF": make_pipeline(StandardScaler(),
                                 SVC(kernel="rbf", C=10, gamma="scale", random_state=SEED)),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=SEED,
                                               n_jobs=-1),
    }


REPRESENTATIONS = {
    "estatisticas(24d)": sequences_to_features,
    "achatado(120d)": sequences_flat,
}


def main():
    ts = time.strftime("%Y%m%d-%H%M%S")
    log_path = ROOT / "results" / "logs" / f"baseline_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w")

    def log(msg=""):
        print(msg)
        log_f.write(msg + "\n")

    log(f"Baseline UTA-RLDD (blink features, 5 folds oficiais) — seed={SEED}")
    log(f"Execução: {ts}\n")

    # results[repr][modelo] = lista de acurácias por fold
    results = {r: {m: [] for m in build_models()} for r in REPRESENTATIONS}
    bal = {r: {m: [] for m in build_models()} for r in REPRESENTATIONS}
    f1m = {r: {m: [] for m in build_models()} for r in REPRESENTATIONS}

    for k in range(1, N_FOLDS + 1):
        xtr_seq, ytr, xte_seq, yte = load_fold(k)
        log(f"Fold {k}: treino {xtr_seq.shape} | teste {xte_seq.shape} | "
            f"classes teste {np.bincount(yte)}")
        for rname, transform in REPRESENTATIONS.items():
            xtr, xte = transform(xtr_seq), transform(xte_seq)
            for mname, model in build_models().items():
                model.fit(xtr, ytr)
                pred = model.predict(xte)
                acc = accuracy_score(yte, pred)
                results[rname][mname].append(acc)
                bal[rname][mname].append(balanced_accuracy_score(yte, pred))
                f1m[rname][mname].append(f1_score(yte, pred, average="macro"))
                log(f"  [{rname}] {mname:13s} acc={acc:.3f}")
                if rname == "estatisticas(24d)":
                    joblib.dump(model, ROOT / "models" / f"{mname}_fold{k}.pkl",
                                compress=3)
        log()

    # Tabela final
    lines = [
        "# Baseline — UTA-RLDD (3 classes, 5 folds oficiais, divisão por sujeito)",
        "",
        f"Classificadores clássicos sobre blink features (30 piscadas x 4 caract.). Seed {SEED}.",
        "Referências (Ghoddoosian et al., 2019): HM-LSTM 65.2% | humanos 57.8%.",
        "",
        "| Representação | Modelo | Acurácia média | Acur. balanceada | F1 macro |",
        "|---|---|---|---|---|",
    ]
    best = (None, None, -1.0)
    for rname in REPRESENTATIONS:
        for mname in build_models():
            a = np.mean(results[rname][mname])
            lines.append(f"| {rname} | {mname} | {a:.1%} ± {np.std(results[rname][mname]):.1%} "
                         f"| {np.mean(bal[rname][mname]):.1%} | {np.mean(f1m[rname][mname]):.1%} |")
            if a > best[2]:
                best = (rname, mname, a)
    lines += ["", f"**Melhor:** {best[1]} com {best[0]} — acurácia média {best[2]:.1%} "
                  f"(vs. 65.2% do HM-LSTM e 57.8% de humanos)."]

    out_md = ROOT / "results" / "baseline_uta_rldd.md"
    out_md.write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    log(f"\nTabela salva em {out_md}\nLog salvo em {log_path}")
    log_f.close()


if __name__ == "__main__":
    sys.exit(main())
