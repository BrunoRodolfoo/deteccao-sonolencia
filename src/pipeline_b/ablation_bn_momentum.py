"""Ablação do momento da BatchNormalization (verificação da afirmação feita em
model.py / results/exp3_eye_state.md): treina o EyeNet no fold 1 do protocolo A
com BN momentum = 0,99 (padrão do Keras) e com 0,9 (usado no pipeline), na
MESMA partição e subamostragem do treino principal, e mede a acurácia em modo
de inferência (i) no próprio subconjunto de treino e (ii) nos sujeitos de
teste do fold 1. Saída: results/exp3_bn_momentum_ablation.csv e log em
results/logs/exp3_bn_momentum_ablation.log."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pipeline_b import datasets as D  # noqa: E402
from pipeline_b import model as M  # noqa: E402
from pipeline_b import train_eye_state as T  # noqa: E402


def main():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(8)
    from sklearn.model_selection import GroupKFold
    x, mrl = D.load_mrl()
    y, subj = mrl["state"].values.astype(int), mrl["subject"].values
    rows = []
    for mom in (0.99, 0.9):
        rng = np.random.default_rng(T.SEED)  # mesma partição do treino principal
        tr, te = next(iter(GroupKFold(n_splits=5).split(x, y, subj)))
        tr_fit, tr_val = T.split_val_by_group(tr, subj, rng)
        tr_fit = T.stratified_subsample(tr_fit, subj, y, 25000, rng)
        tr_val = T.stratified_subsample(tr_val, subj, y, 2500, rng)
        M.BN_MOMENTUM = mom
        model, info = T.train_model(x[tr_fit], y[tr_fit], x[tr_val], y[tr_val], 12, 128,
                                    tag=f"BN momentum {mom} fold 1")
        for name, ii in [("treino (mesmas imagens, modo inferência)", tr_fit), ("teste fold 1", te)]:
            m = T.metrics(y[ii], T.predict(model, x[ii]))
            T.log(f"BN momentum {mom} — {name}: {T.fmt(m)}")
            rows.append({"bn_momentum": mom, "conjunto": name, **m, **info})
    out = ROOT / "results" / "exp3_bn_momentum_ablation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    T.log(f"salvo em {out}")


if __name__ == "__main__":
    main()
