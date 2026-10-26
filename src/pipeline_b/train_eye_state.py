"""Exp. 3 — Treino e avaliação do classificador de estado ocular (EyeNet).

Protocolos:
  A  INTRA-MRL por sujeito: GroupKFold (5 folds) sobre os 37 sujeitos do MRL;
     treino subamostrado (máx. --max-train imagens, estratificado por sujeito
     e classe), early stopping em 10% dos sujeitos de treino, avaliação em
     TODAS as imagens dos sujeitos de teste. Métricas por fold e estratificadas
     por óculos / iluminação / reflexos / sensor nas predições agregadas
     (out-of-fold) dos 5 folds.
  B  CROSS MRL -> CEW: (B-final) modelo treinado em todos os 37 sujeitos
     (mesma subamostragem, 10% dos sujeitos para early stopping) avaliado
     em todo o CEW (patch 24x24 -> 48 e crop 36x36 da face -> 48);
     (B-folds) os 5 modelos do protocolo A avaliados no CEW (média ± desvio).
  C  CROSS CEW -> MRL: 5 folds no CEW agrupados por face de origem (olhos
     esquerdo/direito da mesma face ficam juntos); cada modelo é avaliado em
     TODO o MRL, com as mesmas estratificações. Duas variantes de treino:
     patch (24x24 oficiais) e crop (recorte da face 100x100).
  D  INTRA-CEW: os mesmos 5 modelos de C avaliados no fold de teste do CEW.

Hiperparâmetros (fixos, seed 42): Adam lr 1e-3, ReduceLROnPlateau (fator
0,5, paciência 2), EarlyStopping (paciência 3, restaura melhores pesos),
batch 128, --epochs épocas (padrão 12), entropia cruzada binária;
augmentation em model.build_augmentation().

Saídas: results/exp3_eye_state.csv, results/exp3_strat_conditions.csv,
results/exp3_eye_state.md, results/exp3_predictions.npz, models/exp3/*.keras,
log em results/logs/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from pipeline_b import datasets as D  # noqa: E402
from pipeline_b.model import build_augmentation, build_eyenet, build_mobilenetv3  # noqa: E402

SEED = 42
RESULTS = ROOT / "results"
MODELS = ROOT / "models" / "exp3"
STRATA = {"glasses": {0: "sem óculos", 1: "com óculos"},
          "lighting": {0: "iluminação ruim", 1: "iluminação boa"},
          "reflections": {0: "sem reflexo", 1: "reflexo pequeno", 2: "reflexo grande"},
          "sensor": {1: "RealSense", 2: "IDS", 3: "Aptina"}}


def log(msg: str):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    """acc, bal_acc, f1 (macro), auc, chance (classe majoritária), n."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
    y = np.asarray(y).astype(int)
    yhat = (p >= 0.5).astype(int)
    out = {"n": int(len(y)), "acc": accuracy_score(y, yhat),
           "bal_acc": balanced_accuracy_score(y, yhat),
           "f1_macro": f1_score(y, yhat, average="macro"),
           "chance": max(y.mean(), 1 - y.mean())}
    out["auc"] = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    return out


# --------------------------------------------------------------------------- #
# Subamostragem e treino
# --------------------------------------------------------------------------- #
def stratified_subsample(idx: np.ndarray, groups: np.ndarray, y: np.ndarray, max_n: int, rng) -> np.ndarray:
    """Subamostra idx mantendo a proporção de cada (grupo, classe)."""
    if len(idx) <= max_n:
        return idx
    frac = max_n / len(idx)
    keep = []
    keys = groups[idx] * 2 + y[idx]
    for k in np.unique(keys):
        members = idx[keys == k]
        n = max(1, int(round(len(members) * frac)))
        keep.append(rng.choice(members, n, replace=False))
    return np.sort(np.concatenate(keep))


def make_ds(x_u8, y, batch, training, aug=None):
    import tensorflow as tf
    x = D.to_float(x_u8)
    ds = tf.data.Dataset.from_tensor_slices((x, y.astype(np.float32)))
    if training:
        ds = ds.shuffle(min(len(x), 20000), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(batch)
    if training and aug is not None:
        ds = ds.map(lambda a, b: (aug(a, training=True), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def train_model(x_tr, y_tr, x_val, y_val, epochs, batch, arch="eyenet", tag=""):
    import keras
    keras.utils.set_random_seed(SEED)
    model = build_mobilenetv3() if arch == "mobilenet" else build_eyenet()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                  metrics=[keras.metrics.BinaryAccuracy(name="acc")])
    aug = build_augmentation()
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5)]
    t0 = time.time()
    h = model.fit(make_ds(x_tr, y_tr, batch, True, aug), validation_data=make_ds(x_val, y_val, batch, False),
                  epochs=epochs, callbacks=cbs, verbose=2, shuffle=False)
    ep = len(h.history["loss"])
    best = int(np.argmin(h.history["val_loss"])) + 1
    log(f"{tag}: {ep} épocas ({time.time() - t0:.0f}s), melhor época {best}, "
        f"val_acc={h.history['val_acc'][best - 1]:.4f}, n_treino={len(x_tr)}, n_val={len(x_val)}")
    return model, {"epochs_run": ep, "best_epoch": best, "train_s": time.time() - t0,
                   "val_acc": float(h.history["val_acc"][best - 1]), "n_train": int(len(x_tr)), "n_val": int(len(x_val))}


def predict(model, x_u8, batch=1024) -> np.ndarray:
    return model.predict(D.to_float(x_u8), batch_size=batch, verbose=0).reshape(-1)


def split_val_by_group(idx, groups, rng, frac=0.1):
    """Separa ~frac dos GRUPOS (sujeitos/faces) de idx para validação."""
    g = np.unique(groups[idx])
    rng.shuffle(g)
    n_val = max(1, int(round(len(g) * frac)))
    val_g = set(g[:n_val].tolist())
    m = np.array([gg in val_g for gg in groups[idx]])
    return idx[~m], idx[m]


# --------------------------------------------------------------------------- #
# Registro de resultados
# --------------------------------------------------------------------------- #
class Recorder:
    def __init__(self):
        self.rows = []

    def add(self, protocol, train_on, test_on, fold, stratum, level, y, p, extra=None):
        m = metrics(y, p)
        for k, v in m.items():
            self.rows.append({"protocol": protocol, "train_on": train_on, "test_on": test_on, "fold": fold,
                              "stratum": stratum, "level": level, "metric": k, "value": float(v)})
        if extra:
            for k, v in extra.items():
                self.rows.append({"protocol": protocol, "train_on": train_on, "test_on": test_on, "fold": fold,
                                  "stratum": stratum, "level": level, "metric": k, "value": float(v)})
        return m

    def add_strata(self, protocol, train_on, test_on, fold, df, y, p):
        for col, names in STRATA.items():
            for lv, name in names.items():
                m = (df[col].values == lv)
                if m.sum() == 0 or len(np.unique(y[m])) < 1:
                    continue
                self.add(protocol, train_on, test_on, fold, col, name, y[m], p[m])

    def df(self):
        return pd.DataFrame(self.rows)


def fmt(m):
    return (f"acc={m['acc']:.4f} bal={m['bal_acc']:.4f} f1={m['f1_macro']:.4f} "
            f"auc={m['auc']:.4f} chance={m['chance']:.4f} n={m['n']}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-train", type=int, default=25000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--protocols", default="A,B,C,D")
    ap.add_argument("--mobilenet", action="store_true", help="também treina MobileNetV3-Small (só no protocolo A, fold 1)")
    ap.add_argument("--smoke", action="store_true", help="execução rápida de teste (2 épocas, 3 mil imagens)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    import keras
    keras.utils.set_random_seed(SEED)
    rng = np.random.default_rng(SEED)
    if args.smoke:
        args.epochs, args.max_train = 2, 3000
    tag = args.tag or ("smoke" if args.smoke else "")
    suffix = f"_{tag}" if tag else ""
    MODELS.mkdir(parents=True, exist_ok=True)
    protocols = set(args.protocols.split(","))

    log(f"args={vars(args)}")
    x_mrl, mrl = D.load_mrl()
    y_mrl = mrl["state"].values.astype(int)
    subj = mrl["subject"].values
    x_cp, cew_p = D.load_cew("patch")
    x_cc, cew_c = D.load_cew("crop")
    y_cp, y_cc = cew_p["state"].values.astype(int), cew_c["state"].values.astype(int)
    log(f"MRL {x_mrl.shape} | CEW patch {x_cp.shape} | CEW crop {x_cc.shape}")

    rec = Recorder()
    meta = {}
    preds = {}
    from sklearn.model_selection import GroupKFold

    # ---------------- Protocolo A (+ B-folds) ----------------
    if "A" in protocols:
        oof = np.full(len(y_mrl), np.nan, dtype=np.float32)
        fold_models = []
        gkf = GroupKFold(n_splits=5)
        for k, (tr, te) in enumerate(gkf.split(x_mrl, y_mrl, subj), start=1):
            tr_fit, tr_val = split_val_by_group(tr, subj, rng)
            tr_fit = stratified_subsample(tr_fit, subj, y_mrl, args.max_train, rng)
            tr_val = stratified_subsample(tr_val, subj, y_mrl, max(1000, args.max_train // 10), rng)
            log(f"A fold {k}: sujeitos teste={sorted(np.unique(subj[te]).tolist())} "
                f"n_test={len(te)} n_fit={len(tr_fit)} n_val={len(tr_val)}")
            model, info = train_model(x_mrl[tr_fit], y_mrl[tr_fit], x_mrl[tr_val], y_mrl[tr_val],
                                      args.epochs, args.batch, tag=f"A fold {k}")
            p = predict(model, x_mrl[te])
            oof[te] = p
            m = rec.add("A_intra_mrl", "MRL", "MRL", k, "todos", "todos", y_mrl[te], p, info)
            log(f"A fold {k} teste: {fmt(m)}")
            rec.add_strata("A_intra_mrl", "MRL", "MRL", k, mrl.iloc[te], y_mrl[te], p)
            model.save(MODELS / f"eyenet_mrl_fold{k}{suffix}.keras")
            fold_models.append(model)
            if "B" in protocols:
                for name, xx, yy in [("CEW_patch", x_cp, y_cp), ("CEW_crop", x_cc, y_cc)]:
                    pb = predict(model, xx)
                    m = rec.add("B_folds_mrl_to_cew", "MRL", name, k, "todos", "todos", yy, pb)
                    log(f"B (fold {k}) -> {name}: {fmt(m)}")
            if args.mobilenet and k == 1:
                model_m, info_m = train_model(x_mrl[tr_fit], y_mrl[tr_fit], x_mrl[tr_val], y_mrl[tr_val],
                                              args.epochs, args.batch, arch="mobilenet", tag="A fold 1 MobileNetV3")
                pm = predict(model_m, x_mrl[te])
                m = rec.add("A_intra_mrl_mobilenetv3", "MRL", "MRL", k, "todos", "todos", y_mrl[te], pm, info_m)
                log(f"A fold 1 MobileNetV3 teste: {fmt(m)}")
                meta["mobilenetv3_params"] = int(model_m.count_params())
        m = rec.add("A_intra_mrl", "MRL", "MRL", "oof", "todos", "todos", y_mrl, oof)
        log(f"A agregado (out-of-fold): {fmt(m)}")
        rec.add_strata("A_intra_mrl", "MRL", "MRL", "oof", mrl, y_mrl, oof)
        preds["A_oof_mrl"] = oof

    # ---------------- Protocolo B-final ----------------
    if "B" in protocols:
        all_idx = np.arange(len(y_mrl))
        tr_fit, tr_val = split_val_by_group(all_idx, subj, rng)
        tr_fit = stratified_subsample(tr_fit, subj, y_mrl, args.max_train, rng)
        tr_val = stratified_subsample(tr_val, subj, y_mrl, max(1000, args.max_train // 10), rng)
        log(f"B final: sujeitos val={sorted(np.unique(subj[tr_val]).tolist())} n_fit={len(tr_fit)} n_val={len(tr_val)}")
        model, info = train_model(x_mrl[tr_fit], y_mrl[tr_fit], x_mrl[tr_val], y_mrl[tr_val],
                                  args.epochs, args.batch, tag="B final (MRL todos)")
        model.save(MODELS / f"eyenet_mrl_final{suffix}.keras")
        meta["final_model"] = {"path": (MODELS / f"eyenet_mrl_final{suffix}.keras").relative_to(ROOT).as_posix(),
                               "params": int(model.count_params()),
                               "val_subjects": sorted(np.unique(subj[tr_val]).tolist()), **info}
        for name, xx, yy in [("CEW_patch", x_cp, y_cp), ("CEW_crop", x_cc, y_cc)]:
            pb = predict(model, xx)
            preds[f"B_final_{name}"] = pb
            m = rec.add("B_final_mrl_to_cew", "MRL", name, "final", "todos", "todos", yy, pb, info)
            log(f"B final -> {name}: {fmt(m)}")
            for side in ("L", "R"):
                df_side = cew_p if name == "CEW_patch" else cew_c
                mm = df_side["side"].values == side
                rec.add("B_final_mrl_to_cew", "MRL", name, "final", "side", side, yy[mm], pb[mm])

    # ---------------- Protocolos C e D ----------------
    if "C" in protocols or "D" in protocols:
        for variant, xx, df_c, yy in [("patch", x_cp, cew_p, y_cp), ("crop", x_cc, cew_c, y_cc)]:
            faces = pd.factorize(df_c["face"])[0]
            gkf = GroupKFold(n_splits=5)
            oof = np.full(len(yy), np.nan, dtype=np.float32)
            mrl_probs = []
            for k, (tr, te) in enumerate(gkf.split(xx, yy, faces), start=1):
                tr_fit, tr_val = split_val_by_group(tr, faces, rng)
                model, info = train_model(xx[tr_fit], yy[tr_fit], xx[tr_val], yy[tr_val],
                                          args.epochs, args.batch, tag=f"C/D CEW-{variant} fold {k}")
                model.save(MODELS / f"eyenet_cew_{variant}_fold{k}{suffix}.keras")
                if "D" in protocols:
                    p = predict(model, xx[te])
                    oof[te] = p
                    m = rec.add(f"D_intra_cew_{variant}", f"CEW_{variant}", f"CEW_{variant}", k, "todos", "todos", yy[te], p, info)
                    log(f"D CEW-{variant} fold {k}: {fmt(m)}")
                if "C" in protocols:
                    pm = predict(model, x_mrl)
                    mrl_probs.append(pm)
                    m = rec.add(f"C_cew_{variant}_to_mrl", f"CEW_{variant}", "MRL", k, "todos", "todos", y_mrl, pm, info)
                    log(f"C CEW-{variant} fold {k} -> MRL: {fmt(m)}")
                    rec.add_strata(f"C_cew_{variant}_to_mrl", f"CEW_{variant}", "MRL", k, mrl, y_mrl, pm)
            if "D" in protocols:
                m = rec.add(f"D_intra_cew_{variant}", f"CEW_{variant}", f"CEW_{variant}", "oof", "todos", "todos", yy, oof)
                log(f"D CEW-{variant} agregado: {fmt(m)}")
                preds[f"D_oof_cew_{variant}"] = oof
            if "C" in protocols:
                pm = np.mean(mrl_probs, axis=0)  # ensemble (média) dos 5 modelos
                m = rec.add(f"C_cew_{variant}_to_mrl", f"CEW_{variant}", "MRL", "ensemble5", "todos", "todos", y_mrl, pm)
                log(f"C CEW-{variant} ensemble dos 5 folds -> MRL: {fmt(m)}")
                rec.add_strata(f"C_cew_{variant}_to_mrl", f"CEW_{variant}", "MRL", "ensemble5", mrl, y_mrl, pm)
                preds[f"C_ens_cew_{variant}_mrl"] = pm

    # ---------------- Saídas ----------------
    df = rec.df()
    df.to_csv(RESULTS / f"exp3_eye_state{suffix}.csv", index=False)
    strat = df[df["stratum"] != "todos"]
    strat.to_csv(RESULTS / f"exp3_strat_conditions{suffix}.csv", index=False)
    np.savez_compressed(RESULTS / f"exp3_predictions{suffix}.npz", **preds)
    meta["args"] = vars(args)
    meta["eyenet_params"] = int(build_eyenet().count_params())
    (RESULTS / f"exp3_meta{suffix}.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"CSV salvo em {RESULTS / f'exp3_eye_state{suffix}.csv'} ({len(df)} linhas)")


if __name__ == "__main__":
    main()
