"""Utilitários compartilhados pelos experimentos sobre as blink features do UTA-RLDD.

Conteúdo:
  * carregamento dos 5 folds oficiais (reutiliza ``src/baseline/load_folds.py``);
  * reconstrução de ``video_id`` e ``subject_id`` das sequências de TESTE de cada
    fold a partir das mudanças de rótulo (padrão alerta -> pouca vigilância ->
    sonolento repetido por sujeito), validada contra os ``start_indices``
    hard-coded em ``Training.py`` do repositório original de Ghoddoosian et al.;
  * métricas: BSA (acurácia por sequência), acurácia balanceada, F1 macro,
    AUC macro OvR, matriz de confusão e VA (acurácia por vídeo, voto majoritário
    com desempate pela média das probabilidades);
  * harmonizações binárias B1/B2/B3;
  * níveis de chance (classe majoritária) por sequência e por vídeo.

Ordem das 4 características em cada piscada (confirmada em ``Preprocessing.py``,
função ``normalize_blinks``): [0] frequência, [1] amplitude, [2] duração,
[3] velocidade de abertura do olho. Rótulos: 0 = alerta, 1 = pouca vigilância,
2 = sonolento (originais 0/5/10).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "baseline"))

from load_folds import N_FOLDS, load_fold, sequences_to_features  # noqa: E402

FEATURE_NAMES = ["frequencia", "amplitude", "duracao", "velocidade_abertura"]
CLASS_NAMES = ["alerta", "pouca_vigilancia", "sonolento"]
SEED = 42

# start_indices oficiais (Training.py do repositório rezaghoddoosian/Early-Drowsiness-Detection):
# índice da primeira sequência de cada vídeo no arquivo de TESTE de cada fold.
OFFICIAL_START_INDICES = {
    1: [0, 52, 106, 156, 164, 182, 219, 224, 303, 318, 330, 398, 443, 471, 521, 615, 657,
        693, 718, 753, 879, 948, 949, 965, 1068, 1091, 1126, 1177, 1178, 1185, 1200, 1221, 1226],
    2: [0, 1, 104, 126, 155, 168, 268, 339, 491, 695, 696, 697, 712, 715, 761, 818, 886, 973,
        1012, 1087, 1189, 1270, 1298, 1320, 1380, 1397, 1406, 1429, 1439, 1463, 1464, 1465,
        1549, 1592, 1753, 1945],
    3: [0, 1, 2, 7, 11, 19, 36, 71, 89, 97, 263, 505, 740, 779, 888, 995, 1129, 1232, 1435,
        1441, 1496, 1502, 1503, 1524, 1617, 1618, 1662, 1755, 1790, 1881, 2039, 2065, 2138,
        2183, 2184, 2190],
    4: [0, 8, 38, 108, 113, 114, 135, 137, 285, 380, 411, 473, 568, 569, 581, 590, 610, 699,
        819, 827, 865, 884, 885, 923, 949, 990, 1051, 1156, 1157, 1185, 1265, 1266, 1284,
        1299, 1391, 1450],
    5: [0, 1, 27, 45, 55, 153, 205, 256, 382, 515, 521, 522, 527, 536, 556, 575, 598, 730,
        794, 795, 904, 955, 999, 1027, 1059, 1060, 1088, 1143, 1275, 1480, 1605, 1622, 1659],
}

# Resultados de referência (Ghoddoosian et al., CVPRW 2019, Tabelas 2 e 3)
PAPER_TABLE2 = {  # (BSA, VA) médias dos 5 folds, 3 classes
    "HM-LSTM": (0.540, 0.652),
    "LSTM": (0.528, 0.614),
    "Camadas densas": (0.520, 0.570),
    "Humanos": (None, 0.578),
}
PAPER_TABLE3_VA = {  # VA por fold
    "HM-LSTM": [0.64, 0.61, 0.70, 0.64, 0.67],
    "Humanos": [0.62, 0.59, 0.60, 0.53, 0.55],
}


# ----------------------------------------------------------------------------
# Reconstrução de identidades
# ----------------------------------------------------------------------------
def video_starts_from_labels(y):
    """Índices onde começa cada vídeo: posição 0 e cada mudança de rótulo.

    Os arquivos de teste não são embaralhados e vêm ordenados por sujeito e, dentro
    do sujeito, na ordem alerta(0) -> pouca vigilância(1) -> sonolento(2); portanto
    toda mudança de rótulo é uma fronteira de vídeo.
    """
    y = np.asarray(y).reshape(-1)
    return [0] + [i for i in range(1, len(y)) if y[i] != y[i - 1]]


def reconstruct_ids(y, fold=None):
    """Retorna (video_id, subject_id) locais ao fold para cada sequência de teste.

    Se ``fold`` for informado, verifica a igualdade com os start_indices oficiais
    e o padrão de rótulos 0,1,2 repetido (3 vídeos por sujeito).
    """
    y = np.asarray(y).reshape(-1)
    starts = video_starts_from_labels(y)
    if fold is not None:
        assert starts == OFFICIAL_START_INDICES[fold], (
            f"Fold {fold}: fronteiras de vídeo divergem dos start_indices oficiais")
    labels_at_starts = [int(y[s]) for s in starts]
    assert len(starts) % 3 == 0 and labels_at_starts == [0, 1, 2] * (len(starts) // 3), (
        "Padrão de rótulos 0,1,2 por sujeito não confirmado")
    video_id = np.zeros(len(y), dtype=np.int64)
    bounds = starts + [len(y)]
    for v, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        video_id[a:b] = v
    subject_id = video_id // 3
    return video_id, subject_id


def load_fold_with_ids(k):
    """(X_train, y_train, X_test, y_test, video_id, subject_id) do fold k (ids locais)."""
    xtr, ytr, xte, yte = load_fold(k)
    vid, sid = reconstruct_ids(yte, fold=k)
    return xtr, ytr, xte, yte, vid, sid


def load_pool():
    """Pool das sequências de TESTE dos 5 folds (8661 sequências, 174 vídeos, 58 sujeitos)
    com ids GLOBAIS. Retorna dict com X, y, video_id, subject_id, fold."""
    xs, ys, vids, sids, folds = [], [], [], [], []
    v_off = s_off = 0
    for k in range(1, N_FOLDS + 1):
        _, _, xte, yte, vid, sid = load_fold_with_ids(k)
        xs.append(xte); ys.append(yte)
        vids.append(vid + v_off); sids.append(sid + s_off)
        folds.append(np.full(len(yte), k, dtype=np.int64))
        v_off += vid.max() + 1
        s_off += sid.max() + 1
    return {"X": np.concatenate(xs), "y": np.concatenate(ys),
            "video_id": np.concatenate(vids), "subject_id": np.concatenate(sids),
            "fold": np.concatenate(folds)}


# ----------------------------------------------------------------------------
# Harmonizações binárias
# ----------------------------------------------------------------------------
SCHEMES = {
    "3c": "3 classes: alerta / pouca vigilância / sonolento",
    "B1": "alerta {0} vs. não alerta {1,2}",
    "B2": "não sonolento {0,1} vs. sonolento {2}",
    "B3": "alerta {0} vs. sonolento {2}, descartando a classe intermediária",
}


def apply_scheme(y, scheme):
    """Retorna (y_harmonizado, máscara de amostras mantidas) para o esquema."""
    y = np.asarray(y).reshape(-1)
    keep = np.ones(len(y), dtype=bool)
    if scheme == "3c":
        return y.copy(), keep
    if scheme == "B1":
        return (y >= 1).astype(np.int64), keep
    if scheme == "B2":
        return (y == 2).astype(np.int64), keep
    if scheme == "B3":
        keep = y != 1
        return (y[keep] == 2).astype(np.int64), keep
    raise ValueError(scheme)


def n_classes(scheme):
    return 3 if scheme == "3c" else 2


# ----------------------------------------------------------------------------
# Métricas
# ----------------------------------------------------------------------------
def sequence_metrics(y_true, y_pred, proba=None):
    """BSA, acurácia balanceada, F1 macro, AUC macro OvR (se houver proba) e matriz de confusão."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    k = int(max(y_true.max(), y_pred.max())) + 1
    out = {
        "bsa": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "auc_macro": np.nan,
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(k))),
    }
    if proba is not None:
        proba = np.asarray(proba)
        try:
            if proba.shape[1] == 2:
                out["auc_macro"] = roc_auc_score(y_true, proba[:, 1])
            else:
                out["auc_macro"] = roc_auc_score(y_true, proba, multi_class="ovr",
                                                 average="macro")
        except ValueError:
            out["auc_macro"] = np.nan
    return out


def video_vote(y_pred, proba, video_id):
    """Decisão por vídeo: voto majoritário das classes preditas das sequências.

    Regra de desempate (documentada): se duas ou mais classes empatam no número de
    votos, escolhe-se, entre as empatadas, a de maior probabilidade média ao longo
    do vídeo (análogo à média da saída contínua no código original). Sem
    probabilidades, o empate é resolvido pela classe de menor índice.
    Retorna dict video -> classe predita.
    """
    y_pred = np.asarray(y_pred); video_id = np.asarray(video_id)
    k = int(y_pred.max()) + 1 if proba is None else np.asarray(proba).shape[1]
    decisions = {}
    for v in np.unique(video_id):
        m = video_id == v
        votes = np.bincount(y_pred[m], minlength=k)
        tied = np.flatnonzero(votes == votes.max())
        if len(tied) == 1 or proba is None:
            decisions[int(v)] = int(tied[0])
        else:
            mean_p = np.asarray(proba)[m].mean(axis=0)
            decisions[int(v)] = int(tied[np.argmax(mean_p[tied])])
    return decisions


def video_accuracy(y_true, y_pred, proba, video_id):
    """VA: fração de vídeos cuja decisão por voto coincide com o rótulo do vídeo."""
    y_true = np.asarray(y_true); video_id = np.asarray(video_id)
    dec = video_vote(y_pred, proba, video_id)
    truth = {int(v): int(y_true[video_id == v][0]) for v in np.unique(video_id)}
    hits = [dec[v] == truth[v] for v in truth]
    return float(np.mean(hits)), dec, truth


def chance_levels(y_true, video_id=None):
    """Nível de chance da classe majoritária: por sequência e (opcional) por vídeo."""
    y_true = np.asarray(y_true)
    seq = np.bincount(y_true).max() / len(y_true)
    vid = None
    if video_id is not None:
        video_id = np.asarray(video_id)
        vl = np.array([y_true[video_id == v][0] for v in np.unique(video_id)])
        vid = np.bincount(vl).max() / len(vl)
    return seq, vid


def mean_std(values):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v))


def fmt_pct(m, s=None):
    if s is None:
        return f"{100 * m:.1f}%"
    return f"{100 * m:.1f} ± {100 * s:.1f}"


__all__ = ["ROOT", "SEED", "N_FOLDS", "FEATURE_NAMES", "CLASS_NAMES", "SCHEMES",
           "OFFICIAL_START_INDICES", "PAPER_TABLE2", "PAPER_TABLE3_VA",
           "load_fold", "load_fold_with_ids", "load_pool", "sequences_to_features",
           "reconstruct_ids", "video_starts_from_labels", "apply_scheme", "n_classes",
           "sequence_metrics", "video_vote", "video_accuracy", "chance_levels",
           "mean_std", "fmt_pct"]
