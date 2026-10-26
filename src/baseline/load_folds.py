"""Carrega as 5 partições oficiais (por sujeito) das blink features do UTA-RLDD.

Cada amostra é uma sequência de 30 piscadas x 4 características, na ordem
[0] frequência, [1] amplitude, [2] duração, [3] velocidade de abertura do olho
(ordem confirmada em Preprocessing.py, função normalize_blinks, do repositório
original de Ghoddoosian et al.), com rótulo em 3 classes
(0 = alerta, 1 = pouca vigilância, 2 = sonolento; originais 0/5/10).

Para os classificadores clássicos do baseline, cada sequência é resumida em
um vetor fixo de características (estatísticas por dimensão + concatenação).
"""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "uta_rldd_blinks"
N_FOLDS = 5


def _normalize_labels(y):
    """Mapeia os rótulos originais (ex.: 0/5/10 da escala KSS) para 0..K-1."""
    classes = np.unique(y)
    lut = {c: i for i, c in enumerate(sorted(classes))}
    return np.vectorize(lut.get)(y).astype(np.int64)


def load_fold(k):
    """Retorna (X_train, y_train, X_test, y_test) do fold k (1..5).

    X tem forma (N, 30, 4); y tem forma (N,).
    """
    if not DATA_DIR.exists():
        raise SystemExit("Dados não encontrados. Rode: python src/baseline/download_data.py")
    xtr = np.load(DATA_DIR / f"Blinks_30_Fold{k}.npy", allow_pickle=True)
    ytr = np.load(DATA_DIR / f"Labels_30_Fold{k}.npy", allow_pickle=True)
    xte = np.load(DATA_DIR / f"BlinksTest_30_Fold{k}.npy", allow_pickle=True)
    yte = np.load(DATA_DIR / f"LabelsTest_30_Fold{k}.npy", allow_pickle=True)
    xtr, xte = np.asarray(xtr, dtype=np.float32), np.asarray(xte, dtype=np.float32)
    ytr = _normalize_labels(np.asarray(ytr).reshape(-1))
    yte = _normalize_labels(np.asarray(yte).reshape(-1))
    return xtr, ytr, xte, yte


def sequences_to_features(x):
    """(N, 30, 4) -> (N, D): estatísticas por característica ao longo da sequência.

    Para cada uma das 4 características: média, desvio, mínimo, máximo,
    mediana e amplitude interquartil -> 24 dimensões, invariantes à ordem,
    adequadas a SVM/RF/regressão logística.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:  # já achatado
        return x
    feats = [x.mean(axis=1), x.std(axis=1), x.min(axis=1), x.max(axis=1),
             np.median(x, axis=1),
             np.percentile(x, 75, axis=1) - np.percentile(x, 25, axis=1)]
    return np.concatenate(feats, axis=1)


def sequences_flat(x):
    """(N, 30, 4) -> (N, 120): concatenação bruta (mantém a ordem temporal)."""
    x = np.asarray(x, dtype=np.float32)
    return x.reshape(len(x), -1)
