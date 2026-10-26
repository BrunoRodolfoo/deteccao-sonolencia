"""Modelos usados nos experimentos sobre as blink features do UTA-RLDD.

  * clássicos (scikit-learn) sobre estatísticas 24d — mesmos hiperparâmetros do
    baseline da Entrega 01 (``src/baseline/train.py``);
  * temporais (Keras 3 / TensorFlow) sobre a sequência bruta (30, 4):
      - "LSTM":  Normalization -> LSTM(32, unroll) -> Dropout(0.2) -> Dense(16, relu)
                 -> Dropout(0.2) -> Dense(K, softmax)
      - "CNN1D": Normalization -> Conv1D(32,3) -> Conv1D(64,3) -> GlobalAveragePooling1D
                 -> Dropout(0.2) -> Dense(16, relu) -> Dense(K, softmax)
    Otimizador Adam(1e-3), batch 64, EPOCHS épocas fixas, sem early stopping
    (o conjunto de teste nunca é consultado durante o treino). A camada
    Normalization (z-score por característica, ajustada no TREINO) fica embutida
    no modelo, de modo que a exportação para TFLite recebe as features cruas.
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSICAL = ["LogReg", "SVM-RBF", "RandomForest"]
TEMPORAL = ["LSTM", "CNN1D"]
EPOCHS = 50
BATCH = 64
LR = 1e-3
HPARAMS = {
    "LogReg": {"scaler": "StandardScaler", "max_iter": 2000},
    "SVM-RBF": {"scaler": "StandardScaler", "kernel": "rbf", "C": 10, "gamma": "scale",
                "probability": True},
    "RandomForest": {"n_estimators": 300},
    "LSTM": {"units": 32, "dense": 16, "dropout": 0.2, "unroll": True, "optimizer": "Adam",
             "lr": LR, "batch": BATCH, "epochs": EPOCHS, "input_norm": "z-score (treino)"},
    "CNN1D": {"conv": [32, 64], "kernel": 3, "pool": "GAP", "dense": 16, "dropout": 0.2,
              "optimizer": "Adam", "lr": LR, "batch": BATCH, "epochs": EPOCHS,
              "input_norm": "z-score (treino)"},
}


def build_classical(name, seed=42):
    if name == "LogReg":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, random_state=seed))
    if name == "SVM-RBF":
        # probability=True (Platt) apenas para obter escores contínuos (AUC e desempate
        # da VA); a classe predita continua sendo a de ``decision_function``.
        return make_pipeline(StandardScaler(),
                             SVC(kernel="rbf", C=10, gamma="scale", probability=True,
                                 random_state=seed))
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    raise ValueError(name)


def build_temporal(name, n_classes, x_train, seed=42):
    """Constrói e compila o modelo Keras; ``x_train`` é usado para adaptar a normalização."""
    import keras

    keras.utils.set_random_seed(seed)
    norm = keras.layers.Normalization(axis=-1)
    norm.adapt(np.asarray(x_train, dtype=np.float32))
    inp = keras.Input(shape=(x_train.shape[1], x_train.shape[2]), name="blinks")
    h = norm(inp)
    if name == "LSTM":
        h = keras.layers.LSTM(32, unroll=True)(h)
        h = keras.layers.Dropout(0.2)(h)
        h = keras.layers.Dense(16, activation="relu")(h)
        h = keras.layers.Dropout(0.2)(h)
    elif name == "CNN1D":
        h = keras.layers.Conv1D(32, 3, padding="same", activation="relu")(h)
        h = keras.layers.Conv1D(64, 3, padding="same", activation="relu")(h)
        h = keras.layers.GlobalAveragePooling1D()(h)
        h = keras.layers.Dropout(0.2)(h)
        h = keras.layers.Dense(16, activation="relu")(h)
    else:
        raise ValueError(name)
    out = keras.layers.Dense(n_classes, activation="softmax", name="probs")(h)
    model = keras.Model(inp, out, name=name)
    model.compile(optimizer=keras.optimizers.Adam(LR),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def fit_temporal(model, x, y, epochs=EPOCHS, seed=42):
    import keras
    keras.utils.set_random_seed(seed)
    hist = model.fit(np.asarray(x, dtype=np.float32), np.asarray(y), epochs=epochs,
                     batch_size=BATCH, shuffle=True, verbose=0)
    return hist.history


def predict_proba_any(model, x, is_temporal):
    if is_temporal:
        p = model.predict(np.asarray(x, dtype=np.float32), batch_size=512, verbose=0)
    else:
        p = model.predict_proba(x)
    return np.asarray(p)


def predict_classes(model, x, proba, is_temporal, name):
    """Classe predita. Para SVM usa ``predict`` (decision_function), que pode diferir de argmax(Platt)."""
    if is_temporal or name == "SVM-RBF":
        return model.predict(x) if not is_temporal else np.argmax(proba, axis=1)
    return np.argmax(proba, axis=1)
