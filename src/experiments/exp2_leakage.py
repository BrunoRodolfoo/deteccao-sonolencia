"""Experimento 2 — vazamento de identidade no UTA-RLDD (blink features).

Pool: as 8661 sequências de TESTE dos 5 folds oficiais (174 vídeos, 58 sujeitos), com
``video_id`` e ``subject_id`` reconstruídos (``uta_common.load_pool``). O mesmo pool é
particionado em validação cruzada 5-fold por três protocolos:

  (i)   janela  — StratifiedKFold aleatório nas sequências: janelas sobrepostas do
                  mesmo vídeo (passo 2 piscadas, tamanho 30) caem dos dois lados;
  (ii)  video   — GroupKFold com grupos = video_id: o mesmo sujeito pode estar em
                  treino e teste (nos outros dois vídeos), mas nunca o mesmo vídeo;
  (iii) sujeito — folds oficiais (divisão por sujeito): teste = sequências do fold k,
                  treino = sequências dos outros 4 folds do pool.

(i) e (ii) são repetidos com 3 seeds (42, 43, 44); (iii) é determinístico.
Modelos: RandomForest sobre estatísticas 24d e o melhor modelo temporal do Exp. 1
(``models/exp1_best_temporal.txt``), com os mesmos hiperparâmetros. Métricas: BSA,
acurácia balanceada, F1 macro, AUC macro (3 classes) e VA quando o protocolo permite
((ii) e (iii); em (i) o voto por vídeo não faz sentido, pois cada vídeo tem janelas
dos dois lados). "Inflação" = diferença em pontos percentuais entre (i) e (iii) e
entre (ii) e (iii).

Saídas: results/exp2_leakage.csv (bruto), results/exp2_leakage_plot.csv (agregado,
pronto para gráfico), results/exp2_leakage.md, results/logs/exp2_<timestamp>.log.

Uso: python src/experiments/exp2_leakage.py [--seeds 42,43,44] [--models RandomForest,LSTM]
     python src/experiments/exp2_leakage.py --regen-md   (só reescreve o .md)
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uta_common import (N_FOLDS, ROOT, SEED, chance_levels, fmt_pct, load_pool,  # noqa: E402
                        mean_std, sequence_metrics, sequences_to_features, video_accuracy)
from blink_models import (EPOCHS, HPARAMS, TEMPORAL, build_classical, build_temporal,  # noqa: E402
                          fit_temporal, predict_proba_any)

warnings.filterwarnings("ignore", category=FutureWarning)
RESULTS = ROOT / "results"
PROTOCOLS = {"janela": "(i) aleatória por janela", "video": "(ii) por vídeo",
             "sujeito": "(iii) por sujeito (folds oficiais)"}
METRICS = ["bsa", "bal_acc", "f1_macro", "auc_macro", "va"]
METRIC_LABEL = {"bsa": "BSA", "bal_acc": "Acur. bal.", "f1_macro": "F1 macro",
                "auc_macro": "AUC macro", "va": "VA"}


class Logger:
    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.f.write(msg + "\n"); self.f.flush()


def splits(protocol, pool, seed):
    y, vid, fold = pool["y"], pool["video_id"], pool["fold"]
    idx = np.arange(len(y))
    if protocol == "janela":
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
        return list(cv.split(idx, y))
    if protocol == "video":
        # GroupKFold embaralha grupos a partir do scikit-learn 1.6; senão, permuta os ids.
        if "shuffle" in inspect.signature(GroupKFold.__init__).parameters:
            cv = GroupKFold(N_FOLDS, shuffle=True, random_state=seed)
            return list(cv.split(idx, y, groups=vid))
        rng = np.random.default_rng(seed)
        perm = rng.permutation(vid.max() + 1)
        return list(GroupKFold(N_FOLDS).split(idx, y, groups=perm[vid]))
    if protocol == "sujeito":
        return [(idx[fold != k], idx[fold == k]) for k in range(1, N_FOLDS + 1)]
    raise ValueError(protocol)


def evaluate(model_name, pool, tr, te, seed, epochs, feats):
    X, y, vid = pool["X"], pool["y"], pool["video_id"]
    if model_name in TEMPORAL:
        model = build_temporal(model_name, 3, X[tr], seed=seed)
        fit_temporal(model, X[tr], y[tr], epochs=epochs, seed=seed)
        proba = predict_proba_any(model, X[te], True)
        pred = proba.argmax(1)
    else:
        model = build_classical(model_name, seed=seed)
        model.fit(feats[tr], y[tr])
        proba = predict_proba_any(model, feats[te], False)
        pred = model.predict(feats[te])
    m = sequence_metrics(y[te], pred, proba)
    m["va"], _, _ = video_accuracy(y[te], pred, proba, vid[te])
    m["leak_videos"] = float(np.isin(np.unique(vid[te]), np.unique(vid[tr])).mean())
    m["leak_subjects"] = float(np.isin(np.unique(pool["subject_id"][te]),
                                       np.unique(pool["subject_id"][tr])).mean())
    return m


def aggregate(df):
    """Média nos folds por seed; depois média ± desvio entre seeds (ou entre folds se 1 seed)."""
    rows = []
    for (prot, model, met), d in df.groupby(["protocol", "model", "metric"]):
        per_seed = d.groupby("seed").value.mean()
        if len(per_seed) > 1:
            m, s = mean_std(per_seed.values); how = f"{len(per_seed)} seeds"
        else:
            m, s = mean_std(d.value.values); how = f"{d.fold.nunique()} folds"
        rows.append(dict(protocol=prot, model=model, metric=met, mean=m, std=s, dispersao=how))
    return pd.DataFrame(rows)


def write_md(df, agg, ts, log_path, models, epochs, seeds):
    L = ["# Experimento 2 — vazamento de identidade no UTA-RLDD (blink features)", "",
         f"Execução {ts}. Pool: {df.attrs['n_seq']} sequências de teste dos 5 folds oficiais, "
         f"{df.attrs['n_vid']} vídeos, {df.attrs['n_sub']} sujeitos; 3 classes. "
         f"Seeds {seeds} para (i) e (ii); (iii) usa os folds oficiais. Modelos temporais: {epochs} épocas. "
         f"Dados brutos: `results/exp2_leakage.csv`; agregado: `results/exp2_leakage_plot.csv`; "
         f"log: `results/logs/{log_path.name}`.", "",
         "Valores em % (média ± desvio populacional, ddof=0, entre seeds para (i)/(ii); entre folds para (iii)). "
         "Chance (classe majoritária) por sequência no pool: " +
         fmt_pct(df.attrs["chance_seq"]) + "; por vídeo: " + fmt_pct(df.attrs["chance_vid"]) + ".", ""]
    for model in models:
        L += [f"## {model}" + (f" — {HPARAMS[model]}" if model in HPARAMS else ""), "",
              "| Protocolo | " + " | ".join(METRIC_LABEL[m] for m in METRICS) + " | vídeos do teste vistos no treino | sujeitos do teste vistos no treino |",
              "|---|" + "---|" * (len(METRICS) + 2)]
        for prot, desc in PROTOCOLS.items():
            a = agg[(agg.model == model) & (agg.protocol == prot)].set_index("metric")
            cells = []
            for met in METRICS + ["leak_videos", "leak_subjects"]:
                if met in a.index and not np.isnan(a.loc[met, "mean"]):
                    cells.append(fmt_pct(a.loc[met, "mean"], a.loc[met, "std"]))
                else:
                    cells.append("n/a")
            L.append(f"| {desc} | " + " | ".join(cells) + " |")
        L += ["", "### Inflação (pontos percentuais em relação ao protocolo por sujeito)", "",
              "| Comparação | " + " | ".join(METRIC_LABEL[m] for m in METRICS) + " |",
              "|---|" + "---|" * len(METRICS)]
        base = agg[(agg.model == model) & (agg.protocol == "sujeito")].set_index("metric")["mean"]
        for prot, desc in [("janela", "(i) janela − (iii) sujeito"), ("video", "(ii) vídeo − (iii) sujeito")]:
            a = agg[(agg.model == model) & (agg.protocol == prot)].set_index("metric")["mean"]
            cells = []
            for met in METRICS:
                if met in a.index and met in base.index and not (np.isnan(a[met]) or np.isnan(base[met])):
                    cells.append(f"{100 * (a[met] - base[met]):+.1f} pp")
                else:
                    cells.append("n/a")
            L.append(f"| {desc} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Leitura", "",
          "- (i) mede o que se obtém quando janelas sobrepostas do mesmo vídeo aparecem em treino e teste "
          "(vazamento de vídeo e de identidade); (ii) elimina o vazamento de vídeo, mas mantém o de "
          "identidade (os outros dois vídeos do mesmo sujeito ficam no treino); (iii) é o protocolo "
          "oficial, sem nenhum vazamento.",
          "- A diferença (i)−(iii) e (ii)−(iii) quantifica quanto uma partição aleatória (comum na "
          "literatura de sonolência atuada) infla os resultados em relação a sujeitos nunca vistos.",
          "- Em (ii) o resultado fica ABAIXO de (iii), e não acima: no UTA-RLDD cada sujeito contribui com "
          "exatamente um vídeo por classe, logo, quando um vídeo está no teste, o mesmo sujeito aparece no "
          "treino apenas com os OUTROS dois rótulos. A identidade vira uma pista contrária (transferência "
          "negativa), o que mostra que o vazamento de identidade só infla resultados quando o rótulo do "
          "sujeito é o mesmo dos dois lados — exatamente o caso das partições aleatórias por quadro/janela "
          "de bases de sonolência atuada.",
          "- Em (iii) o treino de cada fold é o pool menos o fold de teste (ex.: 7392 sequências no fold 1), "
          "e não o arquivo oficial de treino (7379 no fold 1): as 13 sequências de diferença são as do "
          "último sujeito do fold 3 (três vídeos com 1, 6 e 6 sequências), ausente de todos os arquivos "
          "oficiais de treino. Além disso, a ordem das sequências de treino é outra. Para o RandomForest "
          "a diferença em relação ao Exp. 1 é pequena (BSA 54,2 vs. 54,5; VA 55,7 vs. 57,3), mas o LSTM é "
          "sensível a essa perturbação (BSA 49,2 vs. 50,8; VA 50,8 vs. 57,2), o que reforça a instabilidade "
          "dos modelos temporais pequenos com 58 sujeitos.", ""]
    (RESULTS / "exp2_leakage.md").write_text("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--models", default=None, help="ex.: RandomForest,LSTM (padrão: RF + melhor temporal do Exp. 1)")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--protocols", default="janela,video,sujeito")
    ap.add_argument("--regen-md", action="store_true",
                    help="só reescreve results/exp2_leakage.md a partir de results/exp2_leakage.csv")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.regen_md:
        logs = sorted((RESULTS / "logs").glob("exp2_*.log"))
        if not logs:
            raise SystemExit("nenhum log exp2 encontrado para regenerar o .md")
        log_path = logs[-1]
        df = pd.read_csv(RESULTS / "exp2_leakage.csv")
        pool = load_pool()
        ch_seq, ch_vid = chance_levels(pool["y"], pool["video_id"])
        df.attrs.update(n_seq=len(pool["y"]), n_vid=int(pool["video_id"].max() + 1),
                        n_sub=int(pool["subject_id"].max() + 1), chance_seq=ch_seq, chance_vid=ch_vid)
        agg = aggregate(df)  # o CSV agregado existente não é reescrito (evita ruído de arredondamento)
        write_md(df, agg, log_path.stem.split("_")[-1], log_path, list(dict.fromkeys(df.model)), args.epochs,
                 sorted(df[df.protocol != "sujeito"].seed.unique().tolist()))
        print(f"{RESULTS / 'exp2_leakage.md'} regenerado a partir de {RESULTS / 'exp2_leakage.csv'} (log {log_path.name})")
        return
    if args.models:
        models = args.models.split(",")
    else:
        best = (ROOT / "models" / "exp1_best_temporal.txt").read_text().strip()
        models = ["RandomForest", best]
    ts = time.strftime("%Y%m%d-%H%M%S")
    (RESULTS / "logs").mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "logs" / f"exp2_{ts}.log"
    log = Logger(log_path)
    pool = load_pool()
    feats = sequences_to_features(pool["X"])
    ch_seq, ch_vid = chance_levels(pool["y"], pool["video_id"])
    log(f"Exp. 2 vazamento de identidade — pool {pool['X'].shape}, vídeos {pool['video_id'].max() + 1}, "
        f"sujeitos {pool['subject_id'].max() + 1}, classes {np.bincount(pool['y'])}, "
        f"chance seq {ch_seq:.3f} vídeo {ch_vid:.3f}; modelos {models}; seeds {seeds}; épocas {args.epochs}")
    rows = []
    for prot in args.protocols.split(","):
        prot_seeds = seeds if prot != "sujeito" else [SEED]
        for seed in prot_seeds:
            for model_name in models:
                for k, (tr, te) in enumerate(splits(prot, pool, seed), start=1):
                    t0 = time.time()
                    m = evaluate(model_name, pool, tr, te, seed, args.epochs, feats)
                    if prot == "janela":
                        m["va"] = np.nan  # voto por vídeo não se aplica
                    for met in METRICS + ["leak_videos", "leak_subjects"]:
                        rows.append(dict(protocol=prot, seed=seed, model=model_name, fold=k,
                                         metric=met, value=float(m[met]),
                                         n_train=len(tr), n_test=len(te)))
                    log(f"[{prot}] seed {seed} {model_name:13s} fold {k}: BSA={m['bsa']:.3f} "
                        f"bal={m['bal_acc']:.3f} F1={m['f1_macro']:.3f} AUC={m['auc_macro']:.3f} "
                        f"VA={m['va']:.3f} | vídeos vistos={m['leak_videos']:.2f} "
                        f"sujeitos vistos={m['leak_subjects']:.2f} ({time.time() - t0:.0f}s)")
                df = pd.DataFrame(rows)
                df.to_csv(RESULTS / "exp2_leakage.csv", index=False)
    df = pd.DataFrame(rows)
    df.attrs.update(n_seq=len(pool["y"]), n_vid=int(pool["video_id"].max() + 1),
                    n_sub=int(pool["subject_id"].max() + 1), chance_seq=ch_seq, chance_vid=ch_vid)
    agg = aggregate(df)
    agg.to_csv(RESULTS / "exp2_leakage_plot.csv", index=False)
    write_md(df, agg, ts, log_path, models, args.epochs, seeds)
    log("\n" + agg[agg.metric.isin(["bsa", "va"])].to_string(index=False))
    log(f"\nResultados em {RESULTS / 'exp2_leakage.csv'}, {RESULTS / 'exp2_leakage_plot.csv'}, "
        f"{RESULTS / 'exp2_leakage.md'}; log {log_path}")


if __name__ == "__main__":
    main()
