"""Opcional: MobileNetV3-Small (alpha 0,75, pesos ImageNet, entrada 96x96x3
com o cinza replicado) no fold 1 do protocolo A, mesma partição/subamostragem
do EyeNet, com guarda de tempo (--max-minutes). Saída:
results/exp3_mobilenetv3.csv + log results/logs/exp3_mobilenetv3.log."""

from __future__ import annotations

import argparse
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
from pipeline_b import train_eye_state as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-minutes", type=float, default=15)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=25000)
    args = ap.parse_args()
    import keras
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    from sklearn.model_selection import GroupKFold
    rng = np.random.default_rng(T.SEED)
    x, mrl = D.load_mrl()
    y, subj = mrl["state"].values.astype(int), mrl["subject"].values
    tr, te = next(iter(GroupKFold(n_splits=5).split(x, y, subj)))
    tr_fit, tr_val = T.split_val_by_group(tr, subj, rng)
    tr_fit = T.stratified_subsample(tr_fit, subj, y, args.max_train, rng)
    tr_val = T.stratified_subsample(tr_val, subj, y, max(1000, args.max_train // 10), rng)
    t_start = time.time()

    class Guard(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (time.time() - t_start) / 60 > args.max_minutes:
                T.log(f"guarda de tempo: parando após a época {epoch + 1}")
                self.model.stop_training = True

    orig_fit = keras.Model.fit

    def fit_with_guard(self, *a, **k):
        k["callbacks"] = list(k.get("callbacks", [])) + [Guard()]
        return orig_fit(self, *a, **k)

    keras.Model.fit = fit_with_guard
    model, info = T.train_model(x[tr_fit], y[tr_fit], x[tr_val], y[tr_val], args.epochs, 64,
                                arch="mobilenet", tag="A fold 1 MobileNetV3-Small")
    p = T.predict(model, x[te], batch=256)
    rec = T.Recorder()
    m = rec.add("A_intra_mrl_mobilenetv3", "MRL", "MRL", 1, "todos", "todos", y[te], p,
                {**info, "params": model.count_params(), "total_min": (time.time() - t_start) / 60})
    T.log(f"MobileNetV3 fold 1 teste: {T.fmt(m)} params={model.count_params()}")
    rec.add_strata("A_intra_mrl_mobilenetv3", "MRL", "MRL", 1, mrl.iloc[te], y[te], p)
    rec.df().to_csv(ROOT / "results" / "exp3_mobilenetv3.csv", index=False)
    model.save(ROOT / "models" / "exp3" / "mobilenetv3_mrl_fold1.keras")


if __name__ == "__main__":
    main()
