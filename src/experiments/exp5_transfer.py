#!/usr/bin/env python3
"""Exp. 5b (RODADA FINAL) — Transferência de sonolência ATUADA (DMD) para REAL (UTA-RLDD): a resposta direta da PP1.

Rodar no .venv (não precisa de MediaPipe), DEPOIS de `bash src/features/run_features.sh`:

    export PATH="$HOME/.local/bin:$PATH"
    source .venv/bin/activate                               # na raiz do repositório
    python src/experiments/exp5_transfer.py                 # análises + CSV + relatório + figura (~20 min)
    python src/experiments/exp5_transfer.py --so-relatorio  # refaz só o .md e a figura a partir dos CSV
    (atalho com log: bash src/experiments/run_exp5b.sh)

A rodada PRELIMINAR (janelas de 30 s, só o Fold 1 da UTA-RLDD, DMD só com a s5) foi descartada; seus arquivos
foram removidos em 06/10/2026.

Entradas (rodada final da etapa de características: src/features/, relatório results/exp5_caracteristicas.md):
    data/features/final/dmd/windows_{10,5,15}s.csv  DMD: s5 (sonolência) e s1/s2/s3 (distração); coluna `uso`
    data/features/final/uta/windows_{10,5,15}s.csv  UTA-RLDD: 5 folds, 60 sujeitos, 180 vídeos lógicos (182 arquivos)
    data/features/final/<base>/windows_referencia_10s.csv  estatísticas da referência (estimabilidade do z-score)
    data/features/final/uta/blinks_index.csv         piscadas por arquivo (vídeos sem nenhuma piscada)
    results/exp5_dmd_validacao_detector.csv         detector de piscadas × anotação do DMD, por sessão s5 (PP3)
    results/exp5_uta_mapeamento_oficial.csv         sujeito oficial <-> pasta (pastas ausentes do arquivo oficial)
    results/exp1_intra.csv                          Exp. 1 (NÃO recalculado; só lido para a comparação)

Desenho (justificativas em docs/ENTREGA02_relatorio.md):
  Janela de 10 s, passo 5 s, nas duas bases. DMD: alerta = janelas inteiramente dentro de um trecho
  driver_actions/safe_drive da sessão s2 (carro parado); sonolento = janelas da s5 com fechamento anotado >= 0,5 s
  ou bocejo anotado; referência de normalização = safe_drive da s3 (nunca avaliada); gE_28 (sem s5) fica fora.
  UTA: alerta = janelas nos 2/3 iniciais do vídeo 0; referência = último terço do vídeo 0 (nunca avaliada);
  janelas que cruzam o corte (fronteira) ficam fora; vídeos KSS 5 e 10 inteiros.
  (A) DMD intra: deixa-um-sujeito-de-fora (LOSO) nos 13 sujeitos (as duas sessões de gB_10, gF_23, gZ_33 juntas).
  (B) UTA intra (teto em dado real com as MESMAS características): os 5 folds oficiais (mesma partição do Exp. 1);
      esquemas B3 (0 x 10, principal), B1 (0 x 5+10), B2 (0+5 x 10).
  (C) Transferência: treino em TODO o DMD, teste em TODAS as janelas avaliáveis da UTA.
  Características: as 11 da janela CENTRADAS na referência do sujeito (<f>_c = x - média da referência); a escala é
  aprendida no treino (padronização da LR ajustada só no treino). O z-score (<f>_z) continua sem estimativa em várias
  características (desvio 0 ou < 5 janelas na referência) — ver a seção de decisões do relatório.
  Modelos: RandomForest (300 árvores, class_weight="balanced", seed 42) e Regressão Logística (padronizada,
  class_weight="balanced"); os dois com imputação pela mediana do TREINO. Hiperparâmetros fixos.
  Limiares: (i) 0,5, fixo a priori; (ii) "calibrado no treino" = o que maximiza a acurácia balanceada nos escores
  FORA DA AMOSTRA do próprio treino do modelo: C -> LOSO do DMD (os escores de A; nunca olha a UTA — a única
  calibração honesta da transferência); A -> LOSO aninhado (o sujeito de teste não participa da escolha);
  B -> validação interna nos 4 folds de treino de cada fold. Diagnóstico, NÃO resultado: limiar oráculo de C na UTA.
  Métricas por janela (acurácia balanceada, F1 macro, AUC, AUC média intra-sujeito, matriz de confusão) e por VÍDEO
  LÓGICO (escore = média dos escores das janelas; acurácia, acurácia balanceada, F1 macro, AUC; acurácia por voto
  só para comparar com o Exp. 1). IC 95%: bootstrap percentil sobre SUJEITOS (1000 reamostragens, seed 42).
  Queda da PP1 = B - C (pareada por sujeito da UTA); choque de realidade = A - C (reamostragens independentes).
  Sensibilidades: sem normalização por sujeito; alerta da s1 (carro em movimento); s5 sem sinal como alerta;
  janelas de 5 s e 15 s; referência do DMD na própria s2 (espelha a estrutura da UTA); sem gE_29 (falha do detector).
Saídas:
  results/exp5b_transferencia.csv   uma linha por experimento x variante x modelo x esquema x nível x estrato x limiar x métrica
  results/exp5b_limiares.csv        limiares calibrados (C: um por variante/modelo; A: um por sujeito; B: um por fold)
  results/exp5b_importancia.csv     importância por permutação dentro do sujeito (A, B3, C), AUC univariada, coeficientes da LR
  results/exp5b_por_sujeito.csv     UTA (60): AUC intra-sujeito B x C; DMD (13): A por motorista + óculos + detector
  results/exp5b_comparacao_exp1.csv B por fold x Exp. 1 por fold (mesmos folds oficiais)
  results/exp5b_predicoes.csv       escore de cada janela (variantes principal e sem_normalizacao; A em todas)
  results/exp5b_predicoes_video.csv escore de cada vídeo lógico (todas as variantes)
  results/exp5b_amostra.csv         fatos das bases usados no relatório (nenhum número do .md é digitado à mão)
  results/exp5b_transferencia.md    relatório gerado a partir dos CSV acima
  results/figures/fig_exp5b_transferencia.{pdf,png}
  models/exp5/exp5b_final_C_<variante>_<modelo>.joblib + models/exp5/exp5b_final_modelos.json
  results/logs/exp5b_transferencia.log
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data" / "features" / "final"
SIG_UTA_INDEX = ROOT / "data" / "signals" / "uta" / "index.csv"
RES = ROOT / "results"
LOGS = RES / "logs"
FIGS = RES / "figures"
MODELS = ROOT / "models" / "exp5"
sys.path.insert(0, str(ROOT / "src" / "features"))   # windows.py (referência espelhada do DMD, sensibilidade)

CSV_RES = RES / "exp5b_transferencia.csv"
CSV_LIM = RES / "exp5b_limiares.csv"
CSV_IMP = RES / "exp5b_importancia.csv"
CSV_SUJ = RES / "exp5b_por_sujeito.csv"
CSV_EXP1 = RES / "exp5b_comparacao_exp1.csv"
CSV_PRED = RES / "exp5b_predicoes.csv"
CSV_PRED_VID = RES / "exp5b_predicoes_video.csv"
CSV_AMOSTRA = RES / "exp5b_amostra.csv"
CSV_EAR = RES / "exp5b_oculos_ear_dmd.csv"
MD = RES / "exp5b_transferencia.md"
LOG = LOGS / "exp5b_transferencia.log"
FIG = "fig_exp5b_transferencia"
DET_DMD = RES / "exp5_dmd_validacao_detector.csv"
MAPA_UTA = RES / "exp5_uta_mapeamento_oficial.csv"
EXP1 = RES / "exp1_intra.csv"
PREFIXO_MODELO = "exp5b_final_C_"
JSON_MODELOS = MODELS / "exp5b_final_modelos.json"

SEED = 42
N_BOOT = 1000
BLOCO_BOOT = 200      # reamostragens por bloco (limita a memória com ~20 mil janelas)
N_REP_PERM = 20
N_JOBS = 6            # Mac de 10 núcleos com outros processos: no máximo 6 processos/threads
LIMIAR = 0.5
JANELAS = (10, 5, 15)

FEATURES = ["perclos", "taxa_piscadas_min", "dur_media_piscada_s", "dur_max_fechamento_s", "n_fechamentos_05s",
            "ear_media", "ear_dp", "amp_media", "vel_media", "frac_bocejo", "taxa_rosto"]
MODELOS = ("RF", "LR")
NOME_MODELO = {"RF": "Random Forest", "LR": "Regressão logística"}
ART = {"RF": ("o", "do", "no"), "LR": ("a", "da", "na")}   # artigo, "de" + artigo, "em" + artigo (RF masc.; LR fem.)
ESQUEMAS = ("B3", "B1", "B2")
# Diagnóstico do confundimento "vídeo de referência": os modelos de B3 (e o de C) aplicados às janelas KSS 5.
# K10v5 = KSS 10 (positivo) x KSS 5 — os dois FORA do vídeo de referência (vídeo 0); K5v0 = KSS 5 (positivo) x KSS 0.
ESQUEMAS_DIAG = ("K10v5", "K5v0")
DESC_ESQUEMA = {"B3": "alerta (0) x sonolento (10); KSS 5 descartado", "B1": "alerta (0) x não alerta (5 + 10)",
                "B2": "não sonolento (0 + 5) x sonolento (10)", "DMD": "alerta (safe_drive da s2) x sonolento (s5 com sinal)",
                "K10v5": "diagnóstico: KSS 10 x KSS 5 com o modelo de B3/C", "K5v0": "diagnóstico: KSS 5 x KSS 0 com o modelo de B3/C"}
EFEITO_MIN_UNIV = 0.05   # AUC univariada "discrimina" se o IC exclui 0,5 E |AUC - 0,5| >= 0,05 (regra descritiva)

LIM_FIXO, LIM_CAL, LIM_ORACULO = "fixo_0.5", "calibrado_treino", "oraculo_uta"
DESC_LIMIAR = {LIM_FIXO: "0,5 fixo a priori",
               LIM_CAL: "calibrado só com o treino (máxima acurácia balanceada nos escores fora da amostra do treino)",
               LIM_ORACULO: "DIAGNÓSTICO: limiar que maximiza a acurácia balanceada na própria UTA (olha o rótulo do teste)"}

# Variantes de A e C (a de B que serve de comparação em `b`)
VARIANTES = {
    "principal": dict(janela=10, rotulo="y", norm="c", ref="s3", b="principal", excluir=()),
    "sem_normalizacao": dict(janela=10, rotulo="y", norm="bruta", ref="s3", b="sem_normalizacao", excluir=()),
    "alerta_s1": dict(janela=10, rotulo="y_sens_s1", norm="c", ref="s3", b="principal", excluir=()),
    "s5_sem_sinal_alerta": dict(janela=10, rotulo="y_sens_s5_sem_sinal", norm="c", ref="s3", b="principal", excluir=()),
    "janela_5s": dict(janela=5, rotulo="y", norm="c", ref="s3", b="janela_5s", excluir=()),
    "janela_15s": dict(janela=15, rotulo="y", norm="c", ref="s3", b="janela_15s", excluir=()),
    "ref_s2_espelho": dict(janela=10, rotulo="y", norm="c", ref="s2", b="principal", excluir=()),
    "sem_gE_29": dict(janela=10, rotulo="y", norm="c", ref="s3", b="principal", excluir=("gE_29",)),
}
# Variantes de B (UTA intra)
B_VARIANTES = {
    "principal": dict(janela=10, norm="c", com_ref=False),
    "sem_normalizacao": dict(janela=10, norm="bruta", com_ref=False),
    "janela_5s": dict(janela=5, norm="c", com_ref=False),
    "janela_15s": dict(janela=15, norm="c", com_ref=False),
    "uta_ref_no_teste": dict(janela=10, norm="c", com_ref=True),
}
DESC_VARIANTE = {
    "principal": "janela de 10 s; DMD: alerta = safe_drive da s2, sonolento = s5 com sinal anotado; características "
                 "centradas na referência do sujeito (`_c`; DMD: safe_drive da s3; UTA: último terço do vídeo 0)",
    "sem_normalizacao": "como a principal, com as características brutas (sem normalização por sujeito)",
    "alerta_s1": "alerta do DMD = safe_drive da s1 (carro em MOVIMENTO) no lugar da s2",
    "s5_sem_sinal_alerta": "janelas da s5 sem sinal anotado entram como alerta, junto com as da s2",
    "janela_5s": "janelas de 5 s (passo 5 s) nas duas bases",
    "janela_15s": "janelas de 15 s (passo 5 s) nas duas bases (no DMD quase não sobra alerta)",
    "ref_s2_espelho": "referência do DMD = último terço das janelas safe_drive da s2 e alerta = as anteriores "
                      "(a mesma estrutura da UTA: referência na sessão alerta, sonolento em outra sessão)",
    "sem_gE_29": "sem o motorista gE_29 (usa óculos; o detector de piscadas falhou nele) no DMD",
    "uta_ref_no_teste": "só B3: janelas de referência e de fronteira da UTA no treino e no teste, como nos `.npy` oficiais",
}
VARIANTES_CENTRAIS = ("principal", "sem_normalizacao")   # limiar aninhado, estratos, importância, predições por janela
METRICAS_LIMIAR = ("acuracia", "acuracia_balanceada", "f1_macro", "sensibilidade", "especificidade")
METRICAS_QUEDA = {"janela": ("acuracia_balanceada", "f1_macro", "auc", "auc_media_intra_sujeito"),
                  "video": ("acuracia", "acuracia_balanceada", "f1_macro", "auc", "acuracia_voto")}
CONTAGENS = ("tn", "fp", "fn", "tp")
SUJ_FALHA_DETECTOR = "gE_29"


# ======================================================================================
# Utilidades
# ======================================================================================
def configurar_log() -> logging.Logger:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("exp5b")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(LOG, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def rel(p: Path) -> str:
    """Caminho relativo à raiz do repositório quando possível (saídas do modo --rapido ficam fora dela)."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def gravar_csv(df: pd.DataFrame, path: Path, **kw) -> None:
    """Escrita atômica (outros scripts podem estar lendo os CSV desta pasta)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, **kw)
    os.replace(tmp, path)


def _bool(serie: pd.Series) -> pd.Series:
    return serie.map(lambda v: str(v).strip().lower() == "true")


def colunas(norm: str) -> list[str]:
    return list(FEATURES) if norm == "bruta" else [f + "_c" for f in FEATURES]


def attrs_variante(exp: str, var: str) -> dict:
    """Atributos de uma variante para as linhas do CSV (B tem as suas próprias variantes)."""
    if exp == "B_uta_intra":
        c = B_VARIANTES[var]
        return dict(janela_s=c["janela"], rotulo_dmd="nao_se_aplica", normalizacao=c["norm"], ref_dmd="nao_se_aplica")
    c = VARIANTES[var]
    return dict(janela_s=c["janela"], rotulo_dmd=c["rotulo"], normalizacao=c["norm"], ref_dmd=c["ref"])


# ======================================================================================
# Dados
# ======================================================================================
def ler_janelas(base: str, L: int) -> tuple[pd.DataFrame, str]:
    p = FEAT / base / f"windows_{L}s.csv"
    if not p.exists():
        raise SystemExit(f"Falta {rel(p)}; rode antes: bash src/features/run_features.sh")
    df = pd.read_csv(p, dtype={"rotulo_video": str, "sessao": str, "setup": str, "data_gravacao": str})
    for c in ("janela_ref", "sobrepoe_ref", "avaliavel", "parcial"):
        df[c] = _bool(df[c])
    if base == "dmd":
        df["oculos"] = _bool(df["oculos"])
        df["tem_s5"] = _bool(df["tem_s5"])
    else:
        df["id_sinal"] = df["arquivo"].astype(str)
        df["rotulo_kss"] = df["rotulo_kss"].astype(int)
        df["fold"] = df["fold"].astype(int)
    return df, md5(p)


def conferir_uta(uta: pd.DataFrame, log) -> dict:
    """A rodada final exige a UTA-RLDD completa: 5 folds, 60 sujeitos, 180 vídeos lógicos, e todo arquivo com sinal
    completo em data/signals/uta/index.csv tem janelas (e vice-versa)."""
    idx = pd.read_csv(SIG_UTA_INDEX, dtype={"label": str, "session": str})
    idx = idx[idx["complete"].astype(str).str.lower() == "true"]
    sinal, jan = set(idx["id"].astype(str)), set(uta["id_sinal"])
    faltam, sobram = sorted(sinal - jan), sorted(jan - sinal)
    folds = sorted(int(f) for f in uta["fold"].unique())
    info = dict(folds=folds, n_sujeitos=int(uta["sujeito"].nunique()), n_videos=int(uta["video"].nunique()),
                n_arquivos=int(uta["id_sinal"].nunique()), n_arquivos_sinal=len(sinal),
                arquivos_sem_janela=faltam, arquivos_sem_sinal=sobram)
    log.info("UTA: folds %s, %d sujeitos, %d vídeos lógicos, %d arquivos (index.csv: %d com sinal completo)",
             folds, info["n_sujeitos"], info["n_videos"], info["n_arquivos"], len(sinal))
    if faltam or sobram or folds != [1, 2, 3, 4, 5]:
        raise SystemExit(f"UTA incompleta: folds {folds}; arquivos com sinal sem janela {faltam[:5]}; "
                         f"com janela sem sinal {sobram[:5]}")
    return info


def dmd_espelho(dmd: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sensibilidade `ref_s2_espelho`: a regra de reserva do windows.py (referência = último terço, por sessão, das
    janelas safe_drive da s2; alerta = as anteriores que não a tocam; fronteira = as que a tocam) aplicada a TODOS os
    sujeitos, e a centragem refeita com essa referência. As características brutas (e o limiar de EAR, calibrado na
    s3) não mudam: só a referência da centragem."""
    import windows as W   # src/features/windows.py (mesma implementação testada da etapa de características)
    w = dmd.drop(columns=[c for c in dmd.columns if c.endswith("_c") or c.endswith("_z")]).copy()
    w, _ = W.marcar_dmd(w, n_min_ref=10 ** 9)
    w, stats = W.normalizar(w, w["janela_ref"], grupos="unidade_ref")
    return w, stats


def conjunto_dmd(dmd: pd.DataFrame, rotulo: str = "y", excluir=()) -> pd.DataFrame:
    """Janelas do DMD com rótulo na coluna `rotulo` (NaN = fora), sem referência/fronteira e sem `excluir`."""
    m = dmd[rotulo].notna() & ~dmd["janela_ref"] & ~dmd["sobrepoe_ref"] & ~dmd["sujeito"].isin(list(excluir))
    d = dmd[m].copy()
    d["y"] = d[rotulo].astype(int)
    assert d["tem_s5"].all(), "sujeito sem s5 no conjunto pareado"
    return d.reset_index(drop=True)


def conjunto_uta(uta: pd.DataFrame, esquema: str, com_ref: bool = False) -> pd.DataFrame:
    """Janelas da UTA para um esquema (positivo = sonolento). Só as avaliáveis (sem referência e sem fronteira),
    salvo com_ref (sensibilidade: todas as janelas, como nos .npy oficiais)."""
    u = uta if com_ref else uta[uta["avaliavel"]]
    if esquema == "B3":
        u = u[u["rotulo_kss"] != 5].copy()
        u["y"] = (u["rotulo_kss"] == 10).astype(int)
    elif esquema == "B1":
        u = u.copy()
        u["y"] = (u["rotulo_kss"] >= 5).astype(int)
    elif esquema == "B2":
        u = u.copy()
        u["y"] = (u["rotulo_kss"] == 10).astype(int)
    else:
        raise ValueError(esquema)
    return u.reset_index(drop=True)


def mascara_esquema(kss: np.ndarray, esquema: str) -> tuple[np.ndarray, np.ndarray]:
    """(máscara das janelas do esquema, y) a partir do KSS — para fatiar os escores de C (que não dependem do esquema)."""
    kss = np.asarray(kss)
    if esquema == "B3":
        m = kss != 5
        return m, (kss[m] == 10).astype(int)
    if esquema == "B1":
        return np.ones(len(kss), bool), (kss >= 5).astype(int)
    if esquema == "B2":
        return np.ones(len(kss), bool), (kss == 10).astype(int)
    raise ValueError(esquema)


def sobrepoe_referencia(df: pd.DataFrame, avaliadas: pd.Series) -> int:
    """Nº de janelas avaliadas que dividem algum trecho de tempo com uma janela de referência do MESMO arquivo
    (tem de ser 0 na rodada final: as de fronteira ficaram fora)."""
    ref = df[df["janela_ref"]][["arquivo", "t_ini", "t_fim"]]
    if ref.empty:
        return 0
    n = 0
    av = df[avaliadas]
    for arq, g in av.groupby("arquivo"):
        r = ref[ref["arquivo"] == arq]
        if r.empty:
            continue
        ini, fim = g["t_ini"].to_numpy()[:, None], g["t_fim"].to_numpy()[:, None]
        n += int(((ini < r["t_fim"].to_numpy()[None, :]) & (fim > r["t_ini"].to_numpy()[None, :])).any(1).sum())
    return n


# ======================================================================================
# Modelos, validação e limiares
# ======================================================================================
def novo_modelo(nome: str) -> Pipeline:
    imp = SimpleImputer(strategy="median", keep_empty_features=True)
    if nome == "RF":
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED, n_jobs=N_JOBS)
        return Pipeline([("imputacao", imp), ("clf", clf)])
    if nome == "LR":
        clf = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=SEED)
        return Pipeline([("imputacao", imp), ("padronizacao", StandardScaler()), ("clf", clf)])
    raise ValueError(nome)


def validacao_cruzada(X, y, grupos, sujeitos, nome):
    """Escores fora da amostra (probabilidade de 'sonolento') com uma partição por valor de `grupos`.
    Devolve (escores, [(índices de teste, modelo)])."""
    s = np.full(len(y), np.nan)
    modelos = []
    for g in pd.unique(grupos):
        te = grupos == g
        tr = ~te
        assert not (set(sujeitos[tr]) & set(sujeitos[te])), "sujeito em treino e teste"
        assert len(np.unique(y[tr])) == 2, f"treino sem as duas classes (partição {g})"
        m = novo_modelo(nome).fit(X[tr], y[tr])
        s[te] = m.predict_proba(X[te])[:, 1]
        modelos.append((np.flatnonzero(te), m))
    assert np.isfinite(s).all()
    return s, modelos


def limiar_otimo(y, s) -> tuple[float, float]:
    """Limiar t que maximiza a acurácia balanceada da regra `s >= t`. Candidatos: pontos médios entre escores
    distintos consecutivos e os dois extremos; empate -> o candidato mais próximo de 0,5 (o limiar a priori)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, float)
    u = np.unique(s)
    cand = np.r_[u[0] - 1e-6, (u[:-1] + u[1:]) / 2.0, u[-1] + 1e-6]
    sp, sn = np.sort(s[y]), np.sort(s[~y])
    tp = len(sp) - np.searchsorted(sp, cand, side="left")
    tn = np.searchsorted(sn, cand, side="left")
    ba = 0.5 * (tp / len(sp) + tn / len(sn))
    melhor = np.flatnonzero(ba >= ba.max() - 1e-12)
    j = melhor[np.argmin(np.abs(cand[melhor] - LIMIAR))]
    return float(cand[j]), float(ba[j])


def limiar_aninhado_loso(X, y, suj, nome):
    """A com limiar calibrado sem o sujeito de teste: para cada sujeito s, LOSO interno nos demais -> escores fora
    da amostra -> limiar ótimo t_s, aplicado ao escore de s (do modelo externo). Devolve (limiar por janela, {s: t_s})."""
    lim = np.full(len(y), np.nan)
    por = {}
    for s_out in pd.unique(suj):
        tr = suj != s_out
        s_in, _ = validacao_cruzada(X[tr], y[tr], suj[tr], suj[tr], nome)
        t, _ = limiar_otimo(y[tr], s_in)
        lim[suj == s_out] = t
        por[s_out] = t
    return lim, por


def limiar_folds_internos(X, y, folds, suj, nome):
    """B com limiar calibrado só no treino: para cada fold de teste k, validação interna deixando um fold de fora
    nos 4 folds de treino -> escores fora da amostra -> limiar ótimo t_k. Devolve (limiar por janela, {k: t_k})."""
    lim = np.full(len(y), np.nan)
    por = {}
    for k in pd.unique(folds):
        tr = folds != k
        s_in, _ = validacao_cruzada(X[tr], y[tr], folds[tr], suj[tr], nome)
        t, _ = limiar_otimo(y[tr], s_in)
        lim[folds == k] = t
        por[int(k)] = t
    return lim, por


# ======================================================================================
# Métricas com pesos (bootstrap por sujeito = multiplicidade de cada sujeito na reamostragem)
# ======================================================================================
class Reamostragem:
    """Matriz (N_BOOT x n_sujeitos) de quantas vezes cada sujeito entra em cada reamostragem."""

    def __init__(self, sujeitos, n_boot, rng):
        self.sujeitos = list(sujeitos)
        self.idx = {s: i for i, s in enumerate(self.sujeitos)}
        n = len(self.sujeitos)
        sorteio = rng.integers(0, n, size=(n_boot, n))
        self.contagens = np.stack([np.bincount(r, minlength=n) for r in sorteio]).astype(float)

    def pesos(self, sujeitos, linhas=slice(None)):
        return self.contagens[linhas][:, [self.idx[s] for s in sujeitos]]


def auc_ponderada(y, s, W):
    """AUC (Mann-Whitney, empates = 1/2) para cada linha de pesos W (B x n)."""
    ordem = np.argsort(s, kind="mergesort")
    so, yo = s[ordem], y[ordem].astype(float)
    ini = np.flatnonzero(np.r_[True, so[1:] != so[:-1]])
    Wo = W[:, ordem]
    wp = np.add.reduceat(Wo * yo, ini, axis=1)
    wn = np.add.reduceat(Wo * (1.0 - yo), ini, axis=1)
    Wp, Wn = wp.sum(1), wn.sum(1)
    abaixo = np.cumsum(wn, axis=1) - wn
    num = (wp * (abaixo + 0.5 * wn)).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((Wp > 0) & (Wn > 0), num / (Wp * Wn), np.nan)


def metricas_ponderadas(y, s, W, limiar=LIMIAR, com_auc=True):
    """Métricas para cada linha de pesos W. `limiar` pode ser escalar ou um vetor (um limiar por janela)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, float)
    yh = s >= np.asarray(limiar, float)
    tp, fn = W @ (y & yh), W @ (y & ~yh)
    fp, tn = W @ (~y & yh), W @ (~y & ~yh)
    P, N = tp + fn, tn + fp
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = np.where(P > 0, tp / P, np.nan)
        espec = np.where(N > 0, tn / N, np.nan)
        f1p = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
        f1n = np.where(2 * tn + fn + fp > 0, 2 * tn / (2 * tn + fn + fp), 0.0)
        ambas = (P > 0) & (N > 0)
        out = dict(acuracia=np.where(P + N > 0, (tp + tn) / (P + N), np.nan),
                   acuracia_balanceada=np.where(ambas, (sens + espec) / 2, np.nan),
                   f1_macro=np.where(ambas, (f1p + f1n) / 2, np.nan),
                   sensibilidade=sens, especificidade=espec, tn=tn, fp=fp, fn=fn, tp=tp)
    if com_auc:
        out["auc"] = auc_ponderada(y, s, W)
    return out


def boot_em_blocos(fn, ream, sujeitos):
    """Aplica fn(W) em blocos de reamostragens e concatena (memória: BLOCO_BOOT x n em vez de N_BOOT x n)."""
    partes = []
    for i in range(0, ream.contagens.shape[0], BLOCO_BOOT):
        partes.append(fn(ream.pesos(sujeitos, slice(i, i + BLOCO_BOOT))))
    return {k: np.concatenate([p[k] for p in partes]) for k in partes[0]}


def avaliar(y, s, sujeitos, ream, auc_por_sujeito=True, limiar=LIMIAR, com_auc=True):
    """Métricas no conjunto inteiro (ponto) e nas N_BOOT reamostragens de sujeitos (boot)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    sujeitos = np.asarray(sujeitos)
    lim = np.broadcast_to(np.asarray(limiar, float), s.shape)
    assert np.isfinite(s).all() and np.isfinite(lim).all() and len(y) == len(s) == len(sujeitos)
    um = np.ones((1, len(y)))
    ponto = {k: float(v[0]) for k, v in metricas_ponderadas(y, s, um, lim, com_auc).items()}
    boot = boot_em_blocos(lambda W: metricas_ponderadas(y, s, W, lim, com_auc), ream, sujeitos)
    meta = dict(n=int(len(y)), n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
                n_sujeitos=int(len(np.unique(sujeitos))))
    if com_auc and auc_por_sujeito:
        aucs = np.full(len(ream.sujeitos), np.nan)
        for su in np.unique(sujeitos):
            m = sujeitos == su
            if 0 < y[m].sum() < m.sum():
                aucs[ream.idx[su]] = roc_auc_score(y[m], s[m])
        ok = np.isfinite(aucs)
        meta["n_sujeitos_duas_classes"] = int(ok.sum())
        if ok.any():
            C = ream.contagens[:, ok]
            ponto["auc_media_intra_sujeito"] = float(aucs[ok].mean())
            with np.errstate(divide="ignore", invalid="ignore"):
                boot["auc_media_intra_sujeito"] = np.where(C.sum(1) > 0, (C @ aucs[ok]) / C.sum(1), np.nan)
    return dict(ponto=ponto, boot=boot, meta=meta)


def conferir_com_sklearn(y, s, res, log, rotulo, limiar=LIMIAR):
    """As métricas com pesos unitários têm de ser iguais às do scikit-learn."""
    yh = (np.asarray(s) >= np.asarray(limiar, float)).astype(int)
    ref = dict(acuracia_balanceada=balanced_accuracy_score(y, yh), f1_macro=f1_score(y, yh, average="macro"))
    if "auc" in res["ponto"]:
        ref["auc"] = roc_auc_score(y, s)
    for k, v in ref.items():
        if abs(v - res["ponto"][k]) > 1e-9:
            raise AssertionError(f"{rotulo}: {k} = {res['ponto'][k]} (nosso) x {v} (sklearn)")


def por_video(df: pd.DataFrame, s, limiar=LIMIAR) -> pd.DataFrame:
    """Uma linha por VÍDEO LÓGICO (coluna `video`; partes _p1/_p2 juntas): escore = média dos escores das janelas;
    fração de janelas >= limiar (para o voto). `limiar` escalar ou por janela (constante dentro do vídeo)."""
    g = pd.DataFrame({"video": df["video"].to_numpy(), "sujeito": df["sujeito"].to_numpy(),
                      "fold": df["fold"].to_numpy() if "fold" in df else np.nan,
                      "rotulo_kss": df["rotulo_kss"].to_numpy() if "rotulo_kss" in df else np.nan,
                      "y": df["y"].to_numpy(), "s": np.asarray(s, float),
                      "lim": np.broadcast_to(np.asarray(limiar, float), (len(df),))})
    g["pred"] = (g["s"] >= g["lim"]).astype(float)
    assert (g.groupby("video")["y"].nunique() == 1).all(), "vídeo com mais de um rótulo"
    assert (g.groupby("video")["lim"].nunique() == 1).all(), "limiar variando dentro do vídeo"
    return (g.groupby("video", sort=True)
            .agg(sujeito=("sujeito", "first"), fold=("fold", "first"), rotulo_kss=("rotulo_kss", "first"),
                 y=("y", "first"), s=("s", "mean"), lim=("lim", "first"), frac_pred=("pred", "mean"),
                 n_janelas=("s", "size")).reset_index())


def voto(v: pd.DataFrame) -> np.ndarray:
    """Decisão do vídeo por voto majoritário das janelas; empate -> escore médio contra o limiar (como no Exp. 1)."""
    return ((v["frac_pred"] > 0.5) | ((v["frac_pred"] == 0.5) & (v["s"] >= v["lim"]))).to_numpy()


def avaliar_video(v: pd.DataFrame, ream, com_auc=True) -> dict:
    res = avaliar(v["y"], v["s"], v["sujeito"], ream, auc_por_sujeito=False, limiar=v["lim"].to_numpy(),
                  com_auc=com_auc)
    acerto = (voto(v) == v["y"].astype(bool).to_numpy()).astype(float)
    res["ponto"]["acuracia_voto"] = float(acerto.mean())
    b = boot_em_blocos(lambda W: {"a": W @ acerto / W.sum(1)}, ream, v["sujeito"])
    res["boot"]["acuracia_voto"] = b["a"]
    return res


def registrar_diagnostico(reg, exp, var, nome, u_b3, s_b3, u5, s5, ream):
    """Avalia K10v5 e K5v0 com os escores de B3 (0/10) e das janelas KSS 5 (mesmo modelo por sujeito)."""
    base = pd.concat([u_b3[["sujeito", "video", "rotulo_kss"]].assign(s=s_b3),
                      u5[["sujeito", "video", "rotulo_kss"]].assign(s=s5)], ignore_index=True)
    for esq, (neg, pos) in (("K10v5", (5, 10)), ("K5v0", (0, 5))):
        q = base[base["rotulo_kss"].isin([neg, pos])]
        y = (q["rotulo_kss"] == pos).astype(int).to_numpy()
        reg[(exp, var, nome, esq, "janela", "todos", LIM_FIXO)] = avaliar(y, q["s"].to_numpy(), q["sujeito"].to_numpy(), ream)


def escores_kss5(mods, suj_treino_teste, u5, cols):
    """Escores das janelas KSS 5 de cada sujeito com o modelo da partição em que o sujeito foi TESTE."""
    suj5 = u5["sujeito"].to_numpy()
    s5 = np.full(len(u5), np.nan)
    for idx, m in mods:
        mk = np.isin(suj5, np.unique(suj_treino_teste[idx]))
        if mk.any():
            s5[mk] = m.predict_proba(u5.loc[mk, cols].to_numpy(float))[:, 1]
    assert np.isfinite(s5).all(), "janela KSS 5 sem modelo de teste"
    return s5


# ======================================================================================
# Importância, AUC univariada e coeficientes
# ======================================================================================
def importancia_permutacao(modelos, X, y, sujeitos, n_rep=N_REP_PERM, seed=SEED):
    """Queda da AUC agrupada (todas as janelas de teste) quando uma característica é embaralhada ENTRE AS
    JANELAS DO MESMO SUJEITO de teste (preserva a distribuição de cada sujeito e quebra só a relação com o
    rótulo). `modelos` = [(índices de teste, modelo)] — os modelos de cada partição (A, B) ou o único (C)."""
    rng = np.random.default_rng(seed)
    idx_all = np.concatenate([i for i, _ in modelos])
    base = np.full(len(y), np.nan)
    for idx, m in modelos:
        base[idx] = m.predict_proba(X[idx])[:, 1]
    auc_base = roc_auc_score(y[idx_all], base[idx_all])
    linhas = []
    for j in range(X.shape[1]):
        S = np.tile(base, (n_rep, 1))
        for idx, m in modelos:
            Xt = X[idx]
            st = sujeitos[idx]
            grande = np.repeat(Xt[None, :, :], n_rep, axis=0)
            for r in range(n_rep):
                col = Xt[:, j].copy()
                for su in np.unique(st):
                    pos = np.flatnonzero(st == su)
                    col[pos] = Xt[rng.permutation(pos), j]
                grande[r, :, j] = col
            S[:, idx] = m.predict_proba(grande.reshape(-1, X.shape[1]))[:, 1].reshape(n_rep, len(idx))
        aucs = np.array([roc_auc_score(y[idx_all], S[r, idx_all]) for r in range(n_rep)])
        linhas.append(dict(j=j, auc_base=auc_base, auc_permutada_media=float(aucs.mean()),
                           importancia=float(auc_base - aucs.mean()), importancia_dp=float((auc_base - aucs).std(ddof=1))))
    return pd.DataFrame(linhas)


def auc_univariada(df, col, ream):
    """AUC da característica sozinha como escore de 'sonolento' (> 0,5: maior no sonolento), IC por sujeito."""
    m = df[col].notna().to_numpy()
    y = df["y"].to_numpy()[m].astype(int)
    x = df[col].to_numpy(float)[m]
    suj = df["sujeito"].to_numpy()[m]
    if y.min() == y.max():
        return dict(auc=np.nan, ic95_inf=np.nan, ic95_sup=np.nan, n=int(m.sum()))
    pt = roc_auc_score(y, x)
    b = boot_em_blocos(lambda W: {"a": auc_ponderada(y.astype(bool), x, W)}, ream, suj)["a"]
    b = b[np.isfinite(b)]
    return dict(auc=float(pt), ic95_inf=float(np.percentile(b, 2.5)), ic95_sup=float(np.percentile(b, 97.5)),
                n=int(m.sum()), n_boot_validos=int(len(b)),
                media_alerta=float(np.mean(x[y == 0])), media_sonolento=float(np.mean(x[y == 1])),
                mediana_alerta=float(np.median(x[y == 0])), mediana_sonolento=float(np.median(x[y == 1])))


def coeficientes_lr(modelos):
    """Coeficientes da LR (escala padronizada) — média e desvio entre os modelos de partição."""
    C = np.array([m.named_steps["clf"].coef_[0] for _, m in modelos])
    return C.mean(0), (C.std(0, ddof=1) if len(C) > 1 else np.full(C.shape[1], np.nan))


# ======================================================================================
# Linhas do CSV
# ======================================================================================
def ic(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    if len(b) == 0:
        return np.nan, np.nan, 0
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi), int(len(b))


def linhas_resultado(chave, res, ctx, limiar_valor=np.nan):
    exp, var, modelo, esquema, nivel, estrato, lim = chave
    at = attrs_variante(exp, var)
    out = []
    for met, val in res["ponto"].items():
        if met in CONTAGENS:
            lo, hi, nb, un = np.nan, np.nan, np.nan, "contagem"
        else:
            lo, hi, nb = ic(res["boot"][met])
            un = "fracao"
        out.append(dict(experimento=exp, variante=var, **at, modelo=modelo, esquema=esquema, nivel=nivel,
                        estrato=estrato, limiar_tipo=lim, limiar_valor=limiar_valor, metrica=met, valor=val,
                        ic95_inf=lo, ic95_sup=hi, n_boot_validos=nb, unidade=un, n_unidades=res["meta"]["n"],
                        n_pos=res["meta"]["n_pos"], n_neg=res["meta"]["n_neg"], n_sujeitos=res["meta"]["n_sujeitos"],
                        n_sujeitos_duas_classes=res["meta"].get("n_sujeitos_duas_classes", np.nan),
                        validacao=ctx["validacao"][exp], execucao=ctx["execucao"]))
    return out


def linhas_diferenca(nome, chave_a, chave_b, reg, ctx, metricas, var_saida):
    """(a - b) em pontos percentuais; bootstrap elemento a elemento (pareado se as duas chaves usam a mesma
    reamostragem de sujeitos, independente se usam bases diferentes)."""
    if chave_a not in reg or chave_b not in reg:
        return []
    ra, rb = reg[chave_a], reg[chave_b]
    at = attrs_variante("C_dmd_para_uta", var_saida)
    out = []
    for met in metricas:
        if met not in ra["ponto"] or met not in rb["ponto"]:
            continue
        d = 100 * (ra["ponto"][met] - rb["ponto"][met])
        lo, hi, nb = ic(100 * (ra["boot"][met] - rb["boot"][met]))
        out.append(dict(experimento=nome, variante=var_saida, **at, modelo=chave_b[2], esquema=chave_b[3],
                        nivel=chave_b[4], estrato=chave_b[5], limiar_tipo=chave_b[6], limiar_valor=np.nan,
                        metrica=met, valor=d, ic95_inf=lo, ic95_sup=hi, n_boot_validos=nb, unidade="pp",
                        n_unidades=np.nan, n_pos=np.nan, n_neg=np.nan, n_sujeitos=np.nan,
                        n_sujeitos_duas_classes=np.nan, validacao=ctx["validacao"][nome], execucao=ctx["execucao"],
                        comparacao=f"{chave_a[0]}[{chave_a[1]},{chave_a[3]},{chave_a[6]}] - "
                                   f"{chave_b[0]}[{chave_b[1]},{chave_b[3]},{chave_b[6]}]"))
    return out


def datas_dmd(dmd: pd.DataFrame) -> pd.DataFrame:
    """Por sujeito do DMD: datas das sessões s2 e s5, s2 no mesmo dia da s5, óculos na s5 e na s2."""
    d = dmd.drop_duplicates("arquivo")[["sujeito", "sessao", "data_gravacao", "oculos"]].copy()
    d["dia"] = d["data_gravacao"].astype(str).str[:10]
    L = []
    for su, g in d.groupby("sujeito"):
        s2, s5 = g[g.sessao == "s2"], g[g.sessao == "s5"]
        L.append(dict(sujeito=su, dias_s2=";".join(sorted(s2["dia"])), dias_s5=";".join(sorted(s5["dia"])),
                      s2_mesmo_dia_s5=bool(len(s5) and set(s2["dia"]) & set(s5["dia"])),
                      oculos_s5=bool(s5["oculos"].any()) if len(s5) else np.nan,
                      oculos_s2=";".join("sim" if o else "não" for o in s2.sort_values("dia")["oculos"]),
                      tem_s5=bool(len(s5))))
    return pd.DataFrame(L)


def fatos_amostra(dados: dict, d_esp: pd.DataFrame, info_uta: dict) -> pd.DataFrame:
    """Fatos das bases citados no relatório (para nenhum número do .md ser digitado à mão)."""
    L = []

    def add(base, item, valor, descricao=""):
        L.append(dict(base=base, item=item, valor=float(valor), descricao=descricao))
    dmd, uta = dados[10]["dmd"], dados[10]["uta"]
    for L_ in JANELAS:
        dd, uu = dados[L_]["dmd"], dados[L_]["uta"]
        for uso, n in dd["uso"].value_counts().items():
            add("dmd", f"janelas_{uso}_{L_}s", n, f"janelas do DMD com uso = {uso} ({L_} s)")
        add("dmd", f"alerta_principal_{L_}s", (dd["y"] == 0).sum(), f"alerta no principal ({L_} s)")
        add("dmd", f"sonolento_principal_{L_}s", (dd["y"] == 1).sum(), f"sonolento no principal ({L_} s)")
        dp = dd[dd["y"].notna()]
        add("dmd", f"sujeitos_duas_classes_{L_}s", (dp.groupby("sujeito")["y"].nunique() == 2).sum(), "")
        add("dmd", f"sujeitos_com_alerta_{L_}s", dp.loc[dp.y == 0, "sujeito"].nunique(), "")
        av = uu[uu["avaliavel"]]
        for k in (0, 5, 10):
            add("uta", f"janelas_kss{k}_{L_}s", (av["rotulo_kss"] == k).sum(), f"janelas avaliáveis KSS {k} ({L_} s)")
        add("uta", f"janelas_referencia_{L_}s", uu["janela_ref"].sum(), "")
        add("uta", f"janelas_fronteira_{L_}s", uu["sobrepoe_ref"].sum(), "")
        add("uta", f"sobreposicao_ref_{L_}s", sobrepoe_referencia(uu, uu["avaliavel"]),
            "janelas avaliadas que dividem tempo com a referência do mesmo arquivo")
        add("dmd", f"sobreposicao_ref_{L_}s", sobrepoe_referencia(dd, dd["y"].notna()),
            "janelas do principal que dividem tempo com a referência do mesmo arquivo")
    add("dmd", "sessoes", dmd["arquivo"].nunique(), "sessões (todas)")
    for s in ("s1", "s2", "s3", "s5"):
        add("dmd", f"sessoes_{s}", dmd.loc[dmd["sessao"] == s, "arquivo"].nunique(), f"sessões {s} com janelas")
    add("dmd", "sujeitos", dmd["sujeito"].nunique(), "sujeitos (com gE_28)")
    add("dmd", "sujeitos_s5", dmd.loc[dmd["tem_s5"], "sujeito"].nunique(), "sujeitos com s5")
    ref_d = dmd[dmd["janela_ref"]].groupby("sujeito").size()
    add("dmd", "ref_por_sujeito_min", ref_d.min(), "janelas de referência por sujeito (10 s)")
    add("dmd", "ref_por_sujeito_max", ref_d.max(), "")
    dp = dmd[dmd["y"].notna()]
    add("dmd", "frac_sem_piscada", dp["dur_media_piscada_s"].isna().mean(), "janelas do principal sem piscada detectada")
    dt = datas_dmd(dmd)
    dt5 = dt[dt["tem_s5"]]
    add("dmd", "sujeitos_s2_mesmo_dia", dt5["s2_mesmo_dia_s5"].sum(),
        ", ".join(dt5.loc[dt5["s2_mesmo_dia_s5"], "sujeito"]))
    add("dmd", "sujeitos_oculos_s5", (dt5["oculos_s5"] == True).sum(),  # noqa: E712
        ", ".join(dt5.loc[dt5["oculos_s5"] == True, "sujeito"]))  # noqa: E712
    ee = d_esp[d_esp["y"].notna()]
    add("dmd", "espelho_alerta", (ee["y"] == 0).sum(), "alerta na referência espelhada (s2, antes do último terço)")
    add("dmd", "espelho_sonolento", (ee["y"] == 1).sum(), "")
    re_ = d_esp[d_esp["janela_ref"]].groupby("sujeito").size()
    add("dmd", "espelho_ref_min", re_.min(), "janelas de referência por sujeito (espelho)")
    add("dmd", "espelho_ref_max", re_.max(), "")
    add("dmd", "espelho_sobreposicao_ref", sobrepoe_referencia(d_esp, d_esp["y"].notna()), "")
    # UTA
    add("uta", "sujeitos", info_uta["n_sujeitos"], "sujeitos")
    add("uta", "videos", info_uta["n_videos"], "vídeos lógicos")
    add("uta", "arquivos", info_uta["n_arquivos"], "arquivos (partes dos vídeos divididos separadas)")
    ref_u = uta[uta["janela_ref"]].groupby("sujeito").size()
    add("uta", "ref_por_sujeito_min", ref_u.min(), "janelas de referência por sujeito (10 s)")
    add("uta", "ref_por_sujeito_mediana", ref_u.median(), "")
    add("uta", "ref_por_sujeito_max", ref_u.max(), "")
    av = uta[uta["avaliavel"]]
    add("uta", "frac_sem_piscada", av["dur_media_piscada_s"].isna().mean(), "janelas avaliáveis sem piscada detectada")
    add("uta", "frac_sem_rosto", av["perclos"].isna().mean(), "janelas avaliáveis sem nenhum quadro com rosto")
    bi = pd.read_csv(FEAT / "uta" / "blinks_index.csv")
    pv = bi.groupby("video")["n_piscadas"].sum()
    sem = sorted(pv[pv == 0].index)
    add("uta", "videos_sem_piscada", len(sem), ";".join(sem))
    for v in sem:
        q = av[av["video"] == v]
        add("uta", f"janelas_video_sem_piscada_{v}", len(q), f"janelas avaliáveis de {v} (KSS {int(q['rotulo_kss'].iloc[0])})")
        add("uta", f"perclos_max_{v}", q["perclos"].max(), "")
    # durações (do contêiner, data/signals/<base>/index.csv)
    idd = pd.read_csv(ROOT / "data" / "signals" / "dmd" / "index.csv")
    ses = dmd.drop_duplicates("arquivo").set_index("arquivo")["sessao"]
    idd["sessao"] = idd["id"].map(ses)
    for s in ("s1", "s2", "s3", "s5"):
        add("dmd", f"duracao_mediana_min_{s}", idd.loc[idd["sessao"] == s, "duration_s"].median() / 60.0,
            f"duração mediana das sessões {s} (min)")
    idu = pd.read_csv(SIG_UTA_INDEX)
    vlog = idu["id"].astype(str).str.replace(r"_p\d+$", "", regex=True)
    dv = idu.groupby(vlog)["duration_s"].sum()
    add("uta", "duracao_mediana_min_video", dv.median() / 60.0, "duração mediana dos vídeos lógicos (min)")
    # rodada preliminar (só para citar no texto; results/preliminar/exp5b_transferencia.csv)
    pre = RES / "preliminar" / "exp5b_transferencia.csv"
    if pre.exists():
        tp = pd.read_csv(pre)
        for mod in MODELOS:
            q = tp[(tp.experimento == "A_dmd_intra") & (tp.variante == "principal") & (tp.modelo == mod)
                   & (tp.esquema == "DMD") & (tp.nivel == "janela") & (tp.estrato == "todos") & (tp.metrica == "auc")]
            if len(q) == 1:
                add("preliminar", f"A_auc_{mod}", q["valor"].iloc[0], "AUC de A na rodada preliminar (30 s; alerta sobreposto à referência)")
                add("preliminar", "A_alerta", q["n_neg"].iloc[0], "janelas alerta de A na rodada preliminar")
                add("preliminar", "A_sujeitos_alerta", q["n_sujeitos_duas_classes"].iloc[0], "sujeitos com alerta na rodada preliminar")
    # estimabilidade do z-score (referência de 10 s)
    for base in ("dmd", "uta"):
        rf = pd.read_csv(FEAT / base / "windows_referencia_10s.csv")
        if base == "dmd":
            rf = rf[rf["unidade_ref"].isin(dmd.loc[dmd["tem_s5"], "sujeito"].unique())]
        tot = []
        for f in FEATURES:
            ok = rf[f + "_sd"].notna() & (rf[f + "_sd"] > 0)
            tot.append(ok.to_numpy())
            add(base, f"z_definido_{f}", ok.mean(), "fração das unidades de referência com z definido")
        add(base, "z_definido_celulas", np.concatenate(tot).mean(), "fração das células unidade x característica")
        add(base, "z_definido_todas_caracteristicas", np.logical_and.reduce(tot).mean(),
            "fração das unidades com z definido nas 11 características")
        add(base, "unidades_ref", len(rf), "")
        add(base, "unidades_ref_5_ou_mais", (rf["n_janelas_ref"] >= 5).sum(), "unidades com >= 5 janelas de referência")
    return pd.DataFrame(L)


def pastas_ausentes_oficial() -> list[str]:
    """Pastas da UTA sem sujeito no arquivo oficial de características: nos folds com < 12 sujeitos oficiais e todos
    eles com pasta atribuída (results/exp5_uta_mapeamento_oficial.csv), as pastas que sobram."""
    m = pd.read_csv(MAPA_UTA)
    out = []
    for k, g in m.groupby("fold"):
        k = int(k)
        if len(g) < 12 and g["pasta"].notna().all():
            todas = {f"F{k}_{i:02d}" for i in range(12 * (k - 1) + 1, 12 * k + 1)}
            out += sorted(todas - set(g["pasta"].astype(str)))
    return out


def detector_por_sujeito() -> pd.DataFrame:
    """Validação do detector de piscadas contra a anotação da s5 (results/exp5_dmd_validacao_detector.csv), média das
    sessões de cada sujeito."""
    v = pd.read_csv(DET_DMD)
    cols = {"f1_piscadas": "det_f1_piscadas", "precisao_piscadas": "det_precisao_piscadas",
            "revocacao_piscadas": "det_revocacao_piscadas", "f1_fech05": "det_f1_fech05",
            "revocacao_fech05": "det_revocacao_fech05", "q_acuracia_bal": "det_acur_bal_quadro",
            "q_kappa": "det_kappa_quadro", "ear_mediana_fechado": "det_ear_mediana_fechado",
            "ear_mediana_aberto": "det_ear_mediana_aberto", "n_piscadas_anot": "det_n_piscadas_anot",
            "n_eventos_det": "det_n_eventos_det"}
    g = v.groupby("sujeito").agg(**{novo: (c, "sum" if c.startswith("n_") else "mean") for c, novo in cols.items()},
                                 det_sessoes=("id", "size"), det_oculos=("oculos", "max"))
    g["det_separacao_ear"] = g["det_ear_mediana_aberto"] - g["det_ear_mediana_fechado"]
    return g.reset_index()


def ear_olhos_dmd(log):
    """PP3 (óculos): o EAR ainda separa olho fechado de aberto, independentemente do limiar? Por sessão s5: AUC de
    -EAR (e de cada olho) nos quadros com rosto anotados 'fechado' x 'aberto' (transições 'abrindo'/'fechando' fora),
    medianas do EAR nas duas classes, queda relativa e o limiar do sujeito (results/exp5_dmd_validacao_detector.csv).
    Depende só dos sinais e da anotação (não dos modelos): roda em segundos, também com --so-relatorio."""
    det = pd.read_csv(DET_DMD)
    rows = []
    for _, r in det.iterrows():
        est = np.load(FEAT / "dmd" / "rotulos_quadros" / f"{r['id']}.npz")["estado_olho"]
        s = np.load(ROOT / "data" / "signals" / "dmd" / f"{r['id']}.npz", allow_pickle=True)
        n = min(len(est), len(s["ear"]))
        est, face = est[:n], s["face"][:n].astype(bool)
        m = face & np.isin(est, [0, 1])
        y = (est[m] == 1).astype(int)
        linha = dict(id=r["id"], sujeito=r["sujeito"], oculos=bool(r["oculos"]), n_quadros_fechado=int(y.sum()),
                     n_quadros_aberto=int((1 - y).sum()), limiar=float(r["limiar"]),
                     det_sens_quadro=float(r["q_sensibilidade"]), det_f1_piscadas=float(r["f1_piscadas"]))
        for k in ("ear", "ear_l", "ear_r"):
            e = s[k][:n][m].astype(float)
            ok = np.isfinite(e)
            linha[f"auc_menos_{k}"] = float(roc_auc_score(y[ok], -e[ok]))
        e = s["ear"][:n][m].astype(float)
        linha["ear_mediana_aberto"] = float(np.nanmedian(e[y == 0]))
        linha["ear_mediana_fechado"] = float(np.nanmedian(e[y == 1]))
        linha["queda_relativa"] = 1.0 - linha["ear_mediana_fechado"] / linha["ear_mediana_aberto"]
        linha["frac_fechado_abaixo_limiar"] = float(np.mean(e[y == 1] < linha["limiar"]))
        rows.append(linha)
    df = pd.DataFrame(rows)
    gravar_csv(df, CSV_EAR)
    log.info("-> %s (%d sessões s5)", rel(CSV_EAR), len(df))
    return df


def executar(log):
    t0 = time.time()
    execucao = datetime.now().strftime("%Y%m%d-%H%M%S")
    log.info("Exp. 5b FINAL — execução %s | Python %s | scikit-learn %s | numpy %s | pandas %s | %s", execucao,
             platform.python_version(), sklearn.__version__, np.__version__, pd.__version__, platform.platform())
    dados = {}
    for L in JANELAS:
        dmd_L, md5_d = ler_janelas("dmd", L)
        uta_L, md5_u = ler_janelas("uta", L)
        dados[L] = dict(dmd=dmd_L, uta=uta_L, md5_dmd=md5_d, md5_uta=md5_u)
        log.info("Janelas de %d s: DMD %d (md5 %s) | UTA %d (md5 %s)", L, len(dmd_L), md5_d, len(uta_L), md5_u)
    info_uta = conferir_uta(dados[10]["uta"], log)
    for L in (5, 15):
        assert set(dados[L]["uta"]["video"]) == set(dados[10]["uta"]["video"]), f"vídeos da UTA diferem em {L} s"
    d_esp, _ = dmd_espelho(dados[10]["dmd"])
    uta10, dmd10 = dados[10]["uta"], dados[10]["dmd"]
    suj_uta = sorted(uta10["sujeito"].unique())
    suj_dmd = sorted(dmd10.loc[dmd10["tem_s5"], "sujeito"].unique())
    assert len(suj_uta) == 60 and len(suj_dmd) == 13, (len(suj_uta), len(suj_dmd))
    log.info("DMD: %d sujeitos com s5 (%s) | UTA: %d sujeitos", len(suj_dmd), ", ".join(suj_dmd), len(suj_uta))
    ctx = dict(execucao=execucao, validacao={
        "A_dmd_intra": f"LOSO ({len(suj_dmd)} sujeitos do DMD; sessões do mesmo sujeito juntas)",
        "B_uta_intra": "5 folds oficiais (teste = fold k, treino = os outros 4; mesma partição do Exp. 1)",
        "C_dmd_para_uta": f"treino: DMD inteiro ({len(suj_dmd)} sujeitos); teste: UTA inteira ({len(suj_uta)} sujeitos)",
        "queda_B_menos_C": "B - C; bootstrap pareado por sujeito da UTA",
        "choque_A_menos_C": "A - C; bootstraps independentes (sujeitos do DMD e da UTA)",
        "lab_A_menos_B": "A - B; bootstraps independentes (sujeitos do DMD e da UTA)"})
    gravar_csv(fatos_amostra(dados, d_esp, info_uta).assign(execucao=execucao), CSV_AMOSTRA)
    log.info("-> %s", rel(CSV_AMOSTRA))
    rng = np.random.default_rng(SEED)
    ream_uta = Reamostragem(suj_uta, N_BOOT, rng)   # sorteada primeiro
    ream_dmd = Reamostragem(suj_dmd, N_BOOT, rng)
    reg, lim_valor, lims, preds, preds_vid, cv = {}, {}, [], [], [], {}
    dt = datas_dmd(dmd10).set_index("sujeito")

    def dmd_da_variante(cfg):
        return d_esp if cfg["ref"] == "s2" else dados[cfg["janela"]]["dmd"]

    def linha_lim(**kw):
        lims.append(dict(execucao=execucao, **kw))

    # ---------------- (A) DMD intra: LOSO ----------------
    A = {}
    for var, cfg in VARIANTES.items():
        t1 = time.time()
        d = conjunto_dmd(dmd_da_variante(cfg), cfg["rotulo"], cfg["excluir"])
        cols = colunas(cfg["norm"])
        X, y, suj = d[cols].to_numpy(float), d["y"].to_numpy(), d["sujeito"].to_numpy()
        log.info("(A) %s: %d janelas (sonolento %d, alerta %d), %d sujeitos, %d com as duas classes", var, len(d),
                 int(y.sum()), int((1 - y).sum()), len(np.unique(suj)),
                 int((d.groupby("sujeito")["y"].nunique() == 2).sum()))
        for nome in MODELOS:
            s, mods = validacao_cruzada(X, y, suj, suj, nome)
            res = avaliar(y, s, suj, ream_dmd)
            conferir_com_sklearn(y, s, res, log, f"A {var} {nome}")
            reg[("A_dmd_intra", var, nome, "DMD", "janela", "todos", LIM_FIXO)] = res
            t_c, ba_c = limiar_otimo(y, s)
            A[(var, nome)] = dict(d=d, X=X, y=y, suj=suj, s=s, t=t_c, cols=cols)
            linha_lim(experimento="C_dmd_para_uta", variante=var, modelo=nome, esquema="todos", unidade="modelo_final",
                      id_unidade="DMD inteiro", limiar=t_c, acur_bal_na_calibracao=ba_c,
                      acur_bal_05_na_calibracao=res["ponto"]["acuracia_balanceada"], n_calibracao=len(y),
                      origem="escores fora da amostra do LOSO do DMD (os de A); a UTA não é consultada")
            lim_a = np.full(len(y), np.nan)
            if var in VARIANTES_CENTRAIS:
                lim_a, por = limiar_aninhado_loso(X, y, suj, nome)
                r_cal = avaliar(y, s, suj, ream_dmd, limiar=lim_a, com_auc=False)
                conferir_com_sklearn(y, s, r_cal, log, f"A {var} {nome} (limiar calibrado)", limiar=lim_a)
                reg[("A_dmd_intra", var, nome, "DMD", "janela", "todos", LIM_CAL)] = r_cal
                for su, t in por.items():
                    linha_lim(experimento="A_dmd_intra", variante=var, modelo=nome, esquema="DMD", unidade="sujeito_teste",
                              id_unidade=su, limiar=t, origem="LOSO aninhado nos outros 12 sujeitos")
                estratos = (("oculos_s5", dt.loc[suj, "oculos_s5"].to_numpy() == True),  # noqa: E712
                            ("sem_oculos_s5", dt.loc[suj, "oculos_s5"].to_numpy() == False),  # noqa: E712
                            ("s2_mesmo_dia_s5", dt.loc[suj, "s2_mesmo_dia_s5"].to_numpy() == True),  # noqa: E712
                            ("s2_outro_dia", dt.loc[suj, "s2_mesmo_dia_s5"].to_numpy() == False))  # noqa: E712
                for nome_e, mask in estratos:
                    ream_e = Reamostragem(sorted(np.unique(suj[mask])), N_BOOT, np.random.default_rng(SEED))
                    reg[("A_dmd_intra", var, nome, "DMD", "janela", nome_e, LIM_FIXO)] = avaliar(y[mask], s[mask], suj[mask], ream_e)
                cv[("A", var, nome)] = (mods, X, y, suj)
            preds.append(pd.DataFrame(dict(experimento="A_dmd_intra", variante=var, modelo=nome, esquema="DMD",
                                           base="dmd", sujeito=suj, fold=np.nan, video=d["arquivo"], arquivo=d["arquivo"],
                                           sessao=d["sessao"], janela=d["janela"], t_ini=d["t_ini"], rotulo=d["uso"], y=y,
                                           escore=s, limiar_calibrado=lim_a)))
            log.info("   %s: acur. bal. %.3f | F1 %.3f | AUC %.3f | limiar calibrado (LOSO) %.3f (acur. bal. %.3f)", nome,
                     res["ponto"]["acuracia_balanceada"], res["ponto"]["f1_macro"], res["ponto"]["auc"], t_c, ba_c)
        log.info("   (%.0f s)", time.time() - t1)

    # ---------------- (B) UTA intra: 5 folds oficiais ----------------
    Bm = {}
    for varb, cfg in B_VARIANTES.items():
        u_all = dados[cfg["janela"]]["uta"]
        cols = colunas(cfg["norm"])
        for esq in (("B3",) if cfg["com_ref"] else ESQUEMAS):
            t1 = time.time()
            u = conjunto_uta(u_all, esq, com_ref=cfg["com_ref"])
            X, y, suj, folds = u[cols].to_numpy(float), u["y"].to_numpy(), u["sujeito"].to_numpy(), u["fold"].to_numpy()
            log.info("(B) %s %s: %d janelas (sonolento %d, não %d), %d vídeos, %d sujeitos", varb, esq, len(u),
                     int(y.sum()), int((1 - y).sum()), u["video"].nunique(), len(np.unique(suj)))
            for nome in MODELOS:
                s, mods = validacao_cruzada(X, y, folds, suj, nome)
                res = avaliar(y, s, suj, ream_uta)
                conferir_com_sklearn(y, s, res, log, f"B {varb} {esq} {nome}")
                reg[("B_uta_intra", varb, nome, esq, "janela", "todos", LIM_FIXO)] = res
                v = por_video(u, s)
                reg[("B_uta_intra", varb, nome, esq, "video", "todos", LIM_FIXO)] = avaliar_video(v, ream_uta)
                lim_b = np.full(len(y), np.nan)
                v_cal = None
                if esq == "B3" and varb in VARIANTES_CENTRAIS:
                    lim_b, por = limiar_folds_internos(X, y, folds, suj, nome)
                    r = avaliar(y, s, suj, ream_uta, limiar=lim_b, com_auc=False)
                    conferir_com_sklearn(y, s, r, log, f"B {varb} {esq} {nome} (limiar calibrado)", limiar=lim_b)
                    reg[("B_uta_intra", varb, nome, esq, "janela", "todos", LIM_CAL)] = r
                    v_cal = por_video(u, s, lim_b)
                    reg[("B_uta_intra", varb, nome, esq, "video", "todos", LIM_CAL)] = avaliar_video(v_cal, ream_uta, com_auc=False)
                    for k, t in por.items():
                        linha_lim(experimento="B_uta_intra", variante=varb, modelo=nome, esquema=esq, unidade="fold_teste",
                                  id_unidade=str(k), limiar=t, origem="validação interna deixando um fold de fora nos 4 folds de treino")
                    cv[("B", varb, nome)] = (mods, X, y, suj)
                    u5 = u_all[u_all["avaliavel"] & (u_all["rotulo_kss"] == 5)].reset_index(drop=True)
                    s5 = escores_kss5(mods, suj, u5, cols)
                    registrar_diagnostico(reg, "B_uta_intra", varb, nome, u, s, u5, s5, ream_uta)
                    preds.append(pd.DataFrame(dict(experimento="B_uta_intra", variante=varb, modelo=nome,
                                                   esquema="B3_janelas_kss5", base="uta", sujeito=u5["sujeito"],
                                                   fold=u5["fold"], video=u5["video"], arquivo=u5["id_sinal"], sessao="",
                                                   janela=u5["janela"], t_ini=u5["t_ini"], rotulo=u5["rotulo_kss"],
                                                   y=np.nan, escore=s5, limiar_calibrado=np.nan)))
                if esq == "B3" and varb in ("principal", "uta_ref_no_teste"):
                    Bm[(varb, nome)] = (u, s)
                if varb in VARIANTES_CENTRAIS:
                    preds.append(pd.DataFrame(dict(experimento="B_uta_intra", variante=varb, modelo=nome, esquema=esq,
                                                   base="uta", sujeito=suj, fold=folds, video=u["video"],
                                                   arquivo=u["id_sinal"], sessao="", janela=u["janela"], t_ini=u["t_ini"],
                                                   rotulo=u["rotulo_kss"], y=y, escore=s, limiar_calibrado=lim_b)))
                pv = v.assign(experimento="B_uta_intra", variante=varb, modelo=nome, esquema=esq,
                              pred_05=(v["s"] >= LIMIAR).astype(int), voto_05=voto(v).astype(int))
                if v_cal is not None:
                    pv["limiar_calibrado"] = v_cal["lim"].to_numpy()
                    pv["pred_calibrado"] = (v_cal["s"] >= v_cal["lim"]).astype(int).to_numpy()
                preds_vid.append(pv)
                log.info("   %s: acur. bal. %.3f | F1 %.3f | AUC %.3f | AUC vídeo %.3f | acurácia vídeo %.3f", nome,
                         res["ponto"]["acuracia_balanceada"], res["ponto"]["f1_macro"], res["ponto"]["auc"],
                         reg[("B_uta_intra", varb, nome, esq, "video", "todos", LIM_FIXO)]["ponto"]["auc"],
                         reg[("B_uta_intra", varb, nome, esq, "video", "todos", LIM_FIXO)]["ponto"]["acuracia"])
            log.info("   (%.0f s)", time.time() - t1)

    # ---------------- (C) Transferência: todo o DMD -> UTA ----------------
    MODELS.mkdir(parents=True, exist_ok=True)
    meta_modelos = []
    for var, cfg in VARIANTES.items():
        t1 = time.time()
        a0 = A[(var, "RF")]
        d, cols = a0["d"], a0["cols"]
        Xd, yd = d[cols].to_numpy(float), d["y"].to_numpy()
        u_all = dados[cfg["janela"]]["uta"]
        ua = u_all[u_all["avaliavel"]].reset_index(drop=True)
        Xu, kss, suj_u = ua[cols].to_numpy(float), ua["rotulo_kss"].to_numpy(), ua["sujeito"].to_numpy()
        for nome in MODELOS:
            m = novo_modelo(nome).fit(Xd, yd)
            s_all = m.predict_proba(Xu)[:, 1]
            t_c = A[(var, nome)]["t"]
            if var in VARIANTES_CENTRAIS:
                arq = MODELS / f"{PREFIXO_MODELO}{var}_{nome}.joblib"
                joblib.dump(m, arq)
                meta_modelos.append(dict(
                    experimento="C_dmd_para_uta", arquivo=str(rel(arq)), variante=var, descricao=DESC_VARIANTE[var], modelo=nome,
                    pipeline=str(m), caracteristicas=cols, ordem_das_colunas="exatamente a de `caracteristicas`",
                    normalizacao=("<f>_c = x - média das janelas de referência alerta do próprio sujeito (DMD: safe_drive "
                                  "da s3; UTA: último terço do vídeo 0); ver src/features/windows.py" if cfg["norm"] == "c"
                                  else "nenhuma (características brutas)"),
                    janela_s=cfg["janela"], passo_s=5, n_treino=int(len(yd)), n_sonolento=int(yd.sum()),
                    n_alerta=int((1 - yd).sum()), sujeitos_treino=sorted(d["sujeito"].unique()),
                    rotulo="alerta = safe_drive da s2 (0); sonolento = s5 com fechamento anotado >= 0,5 s ou bocejo (1)",
                    positivo="sonolento (1)", limiar_fixo=LIMIAR, limiar_calibrado_dmd=t_c,
                    limiar_calibrado_origem="máxima acurácia balanceada nos escores fora da amostra do LOSO do DMD",
                    seed=SEED, sklearn=sklearn.__version__, numpy=np.__version__, execucao=execucao,
                    entrada=str(rel((FEAT / "dmd" / f"windows_{cfg['janela']}s.csv"))),
                    md5_entrada=dados[cfg["janela"]]["md5_dmd"]))
            for esq in ESQUEMAS:
                mk, y = mascara_esquema(kss, esq)
                u = ua[mk].reset_index(drop=True).assign(y=y)
                s, suj = s_all[mk], suj_u[mk]
                res = avaliar(y, s, suj, ream_uta)
                conferir_com_sklearn(y, s, res, log, f"C {var} {esq} {nome}")
                reg[("C_dmd_para_uta", var, nome, esq, "janela", "todos", LIM_FIXO)] = res
                v = por_video(u, s)
                reg[("C_dmd_para_uta", var, nome, esq, "video", "todos", LIM_FIXO)] = avaliar_video(v, ream_uta)
                r = avaliar(y, s, suj, ream_uta, limiar=t_c, com_auc=False)
                conferir_com_sklearn(y, s, r, log, f"C {var} {esq} {nome} (limiar do DMD)", limiar=t_c)
                reg[("C_dmd_para_uta", var, nome, esq, "janela", "todos", LIM_CAL)] = r
                lim_valor[("C_dmd_para_uta", var, nome, esq, "janela", "todos", LIM_CAL)] = t_c
                v_cal = por_video(u, s, t_c)
                reg[("C_dmd_para_uta", var, nome, esq, "video", "todos", LIM_CAL)] = avaliar_video(v_cal, ream_uta, com_auc=False)
                lim_valor[("C_dmd_para_uta", var, nome, esq, "video", "todos", LIM_CAL)] = t_c
                if esq == "B3":
                    t_o, ba_o = limiar_otimo(y, s)
                    reg[("C_dmd_para_uta", var, nome, esq, "janela", "todos", LIM_ORACULO)] = avaliar(
                        y, s, suj, ream_uta, limiar=t_o, com_auc=False)
                    lim_valor[("C_dmd_para_uta", var, nome, esq, "janela", "todos", LIM_ORACULO)] = t_o
                    linha_lim(experimento="C_dmd_para_uta", variante=var, modelo=nome, esquema=esq, unidade="oraculo_uta",
                              id_unidade="UTA inteira", limiar=t_o, acur_bal_na_calibracao=ba_o, n_calibracao=len(y),
                              origem="DIAGNÓSTICO: maximiza a acurácia balanceada na própria UTA (olha o rótulo do teste)")
                    if var in VARIANTES_CENTRAIS:
                        cv[("C", var, nome)] = ([(np.arange(len(y)), m)], Xu[mk], y, suj)
                        u5 = ua[kss == 5].reset_index(drop=True)
                        registrar_diagnostico(reg, "C_dmd_para_uta", var, nome, u, s, u5, s_all[kss == 5], ream_uta)
                pv = v.assign(experimento="C_dmd_para_uta", variante=var, modelo=nome, esquema=esq,
                              pred_05=(v["s"] >= LIMIAR).astype(int), voto_05=voto(v).astype(int),
                              limiar_calibrado=t_c, pred_calibrado=(v["s"] >= t_c).astype(int))
                preds_vid.append(pv)
            if var in VARIANTES_CENTRAIS:
                preds.append(pd.DataFrame(dict(experimento="C_dmd_para_uta", variante=var, modelo=nome, esquema="todos",
                                               base="uta", sujeito=suj_u, fold=ua["fold"], video=ua["video"],
                                               arquivo=ua["id_sinal"], sessao="", janela=ua["janela"], t_ini=ua["t_ini"],
                                               rotulo=kss, y=(kss == 10).astype(int), escore=s_all, limiar_calibrado=t_c)))
            r3 = reg[("C_dmd_para_uta", var, nome, "B3", "janela", "todos", LIM_FIXO)]["ponto"]
            r3c = reg[("C_dmd_para_uta", var, nome, "B3", "janela", "todos", LIM_CAL)]["ponto"]
            log.info("(C) %s %s (treino %d janelas, alerta %d): B3 AUC %.3f | acur. bal. 0,5 %.3f (sens. %.3f, espec. %.3f)"
                     " | limiar do DMD %.3f: acur. bal. %.3f (sens. %.3f, espec. %.3f)", var, nome, len(yd),
                     int((1 - yd).sum()), r3["auc"], r3["acuracia_balanceada"], r3["sensibilidade"], r3["especificidade"],
                     t_c, r3c["acuracia_balanceada"], r3c["sensibilidade"], r3c["especificidade"])
        log.info("   (%.0f s)", time.time() - t1)
    tmpj = JSON_MODELOS.with_suffix(".json.tmp")
    with open(tmpj, "w", encoding="utf-8") as fh:
        json.dump(meta_modelos, fh, ensure_ascii=False, indent=1)
    os.replace(tmpj, JSON_MODELOS)
    log.info("-> %s (%d modelos)", rel(JSON_MODELOS), len(meta_modelos))

    # ---------------- Tabela de resultados + quedas ----------------
    linhas = []
    for chave, res in reg.items():
        lv = lim_valor.get(chave, LIMIAR if chave[6] == LIM_FIXO else np.nan)
        linhas += linhas_resultado(chave, res, ctx, lv)
    sem_auc = lambda mets: tuple(m for m in mets if "auc" not in m)  # noqa: E731
    for var, cfg in VARIANTES.items():
        vb = cfg["b"]
        for nome in MODELOS:
            for esq in ESQUEMAS:
                for nivel in ("janela", "video"):
                    for lim in (LIM_FIXO, LIM_CAL):
                        mets = METRICAS_QUEDA[nivel] if lim == LIM_FIXO else sem_auc(METRICAS_QUEDA[nivel])
                        linhas += linhas_diferenca("queda_B_menos_C", ("B_uta_intra", vb, nome, esq, nivel, "todos", lim),
                                                   ("C_dmd_para_uta", var, nome, esq, nivel, "todos", lim), reg, ctx, mets, var)
                for lim in (LIM_FIXO, LIM_CAL):
                    mets = ("acuracia_balanceada", "f1_macro", "auc") if lim == LIM_FIXO else ("acuracia_balanceada", "f1_macro")
                    linhas += linhas_diferenca("choque_A_menos_C", ("A_dmd_intra", var, nome, "DMD", "janela", "todos", lim),
                                               ("C_dmd_para_uta", var, nome, esq, "janela", "todos", lim), reg, ctx, mets, var)
                    linhas += linhas_diferenca("lab_A_menos_B", ("A_dmd_intra", var, nome, "DMD", "janela", "todos", lim),
                                               ("B_uta_intra", vb, nome, esq, "janela", "todos", lim), reg, ctx, mets, var)
            for esq in ESQUEMAS_DIAG:
                linhas += linhas_diferenca("queda_B_menos_C", ("B_uta_intra", vb, nome, esq, "janela", "todos", LIM_FIXO),
                                           ("C_dmd_para_uta", var, nome, esq, "janela", "todos", LIM_FIXO), reg, ctx,
                                           ("auc", "auc_media_intra_sujeito"), var)
    tab = pd.DataFrame(linhas)
    gravar_csv(tab, CSV_RES)
    log.info("-> %s (%d linhas)", rel(CSV_RES), len(tab))
    gravar_csv(pd.DataFrame(lims), CSV_LIM)
    log.info("-> %s (%d linhas)", rel(CSV_LIM), len(lims))

    # ---------------- Importância, AUC univariada, coeficientes ----------------
    imp = []
    for (exp, var, nome), (mods, X, y, suj) in sorted(cv.items()):
        t1 = time.time()
        pi = importancia_permutacao(mods, X, y.astype(int), suj)
        esq = {"A": "DMD", "B": "B3", "C": "B3"}[exp]
        for _, r in pi.iterrows():
            imp.append(dict(tipo="permutacao_auc", experimento=exp, variante=var, modelo=nome, esquema=esq,
                            caracteristica=FEATURES[int(r["j"])], valor=r["importancia"], dp=r["importancia_dp"],
                            auc_base=r["auc_base"], n_repeticoes=N_REP_PERM, n_modelos=len(mods)))
        if nome == "LR":
            media, dp = coeficientes_lr(mods)
            for j, f in enumerate(FEATURES):
                imp.append(dict(tipo="coef_lr_padronizado", experimento=exp, variante=var, modelo=nome, esquema=esq,
                                caracteristica=f, valor=media[j], dp=dp[j], n_modelos=len(mods)))
        log.info("Importância por permutação %s/%s/%s: %.0f s", exp, var, nome, time.time() - t1)
    u_b3 = conjunto_uta(uta10, "B3")
    for var in VARIANTES_CENTRAIS:
        norm = VARIANTES[var]["norm"]
        for base, df_, ream, esq in (("dmd", A[(var, "RF")]["d"], ream_dmd, "DMD"), ("uta", u_b3, ream_uta, "B3")):
            for f in FEATURES:
                r = auc_univariada(df_, f if norm == "bruta" else f + "_c", ream)
                imp.append(dict(tipo="auc_univariada", experimento=base, variante=var, modelo="", esquema=esq,
                                caracteristica=f, valor=r["auc"], ic95_inf=r["ic95_inf"], ic95_sup=r["ic95_sup"],
                                n=r["n"], n_boot_validos=r.get("n_boot_validos"), media_alerta=r.get("media_alerta"),
                                media_sonolento=r.get("media_sonolento"), mediana_alerta=r.get("mediana_alerta"),
                                mediana_sonolento=r.get("mediana_sonolento")))
    imp = pd.DataFrame(imp)
    imp["execucao"] = execucao
    gravar_csv(imp, CSV_IMP)
    log.info("-> %s (%d linhas)", rel(CSV_IMP), len(imp))

    # ---------------- Predições ----------------
    P = pd.concat(preds, ignore_index=True)
    P["escore"] = P["escore"].round(6)
    gravar_csv(P, CSV_PRED)
    log.info("-> %s (%d linhas)", rel(CSV_PRED), len(P))
    PV = pd.concat(preds_vid, ignore_index=True).rename(columns={"s": "escore_medio", "lim": "limiar_aplicado",
                                                                 "frac_pred": "frac_janelas_sonolento"})
    gravar_csv(PV, CSV_PRED_VID)
    log.info("-> %s (%d linhas)", rel(CSV_PRED_VID), len(PV))

    # ---------------- Por sujeito ----------------
    mapa = pd.read_csv(MAPA_UTA)
    classe_mapa = dict(zip(mapa["pasta"].dropna().astype(str), mapa.loc[mapa["pasta"].notna(), "classe"]))
    ausentes = pastas_ausentes_oficial()
    sem_pisc = set(filter(None, str(pd.read_csv(CSV_AMOSTRA).query("item == 'videos_sem_piscada'")["descricao"].iloc[0]).split(";")))
    rows = {su: dict(base="uta", sujeito=su, fold=int(uta10.loc[uta10["sujeito"] == su, "fold"].iloc[0]),
                     mapeamento_oficial=("ausente_do_oficial" if su in ausentes else classe_mapa.get(su, "indeterminado")),
                     videos_sem_piscada=";".join(sorted(v for v in sem_pisc if v.startswith(su + "_"))))
            for su in suj_uta}
    rot_num = pd.to_numeric(P["rotulo"], errors="coerce")
    for var in VARIANTES_CENTRAIS:
        for nome in MODELOS:
            base_m = (P.variante == var) & (P.modelo == nome)
            for exp, filtro in (("B", (P.experimento == "B_uta_intra") & (P.esquema == "B3")),
                                ("C", (P.experimento == "C_dmd_para_uta") & (rot_num != 5))):
                sfx = f"{exp}_{nome}" + ("" if var == "principal" else "_bruta")
                for su, q in P[filtro & base_m].groupby("sujeito"):
                    linha = rows[su]
                    y_, e_ = q["y"].to_numpy(), q["escore"].to_numpy()
                    linha["n_alerta"], linha["n_sonolento"] = int((y_ == 0).sum()), int((y_ == 1).sum())
                    linha[f"auc_{sfx}"] = roc_auc_score(y_, e_) if len(np.unique(y_)) == 2 else np.nan
                    linha[f"escore_medio_alerta_{sfx}"] = float(e_[y_ == 0].mean())
                    linha[f"escore_medio_sonolento_{sfx}"] = float(e_[y_ == 1].mean())
    rows = [rows[su] for su in suj_uta]
    det = detector_por_sujeito().set_index("sujeito")
    for su in suj_dmd:
        linha = dict(base="dmd", sujeito=su, oculos_s5=bool(dt.loc[su, "oculos_s5"]), oculos_s2=dt.loc[su, "oculos_s2"],
                     s2_mesmo_dia_s5=bool(dt.loc[su, "s2_mesmo_dia_s5"]), dias_s2=dt.loc[su, "dias_s2"],
                     dias_s5=dt.loc[su, "dias_s5"])
        for var in VARIANTES_CENTRAIS:
            for nome in MODELOS:
                a = A[(var, nome)]
                mk = a["suj"] == su
                y, s = a["y"][mk], a["s"][mk]
                yh = s >= LIMIAR
                sfx = f"A_{nome}" + ("" if var == "principal" else "_bruta")
                linha["n_alerta"], linha["n_sonolento"] = int((y == 0).sum()), int((y == 1).sum())
                linha[f"auc_{sfx}"] = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
                linha[f"sens_{sfx}"] = float(yh[y == 1].mean()) if (y == 1).any() else np.nan
                linha[f"espec_{sfx}"] = float((~yh[y == 0]).mean()) if (y == 0).any() else np.nan
                linha[f"acur_bal_{sfx}"] = np.nanmean([linha[f"sens_{sfx}"], linha[f"espec_{sfx}"]])
                linha[f"escore_medio_alerta_{sfx}"] = float(s[y == 0].mean()) if (y == 0).any() else np.nan
                linha[f"escore_medio_sonolento_{sfx}"] = float(s[y == 1].mean()) if (y == 1).any() else np.nan
        if su in det.index:
            linha.update(det.loc[su].to_dict())
        d0 = A[("principal", "RF")]["d"]
        dd = d0[d0["sujeito"] == su]
        for f in ("perclos", "n_fechamentos_05s", "dur_max_fechamento_s", "taxa_piscadas_min"):
            linha[f"media_{f}_alerta"] = float(dd.loc[dd.y == 0, f].mean())
            linha[f"media_{f}_sonolento"] = float(dd.loc[dd.y == 1, f].mean())
        rows.append(linha)
    SJ = pd.DataFrame(rows).assign(execucao=execucao)
    gravar_csv(SJ, CSV_SUJ)
    log.info("-> %s (%d linhas)", rel(CSV_SUJ), len(SJ))

    # ---------------- Comparação com o Exp. 1 (mesmos folds) ----------------
    e1 = pd.read_csv(EXP1)
    e1 = e1[(e1["scheme"] == "B3") & e1["model"].isin(["RandomForest", "LogReg"])]
    comp = []
    for (varb, nome), (u, s) in Bm.items():
        for conj in ("todos", "oficiais"):
            mk = np.ones(len(u), bool) if conj == "todos" else ~u["sujeito"].isin(ausentes).to_numpy()
            for k in range(1, 6):
                mm = mk & (u["fold"].to_numpy() == k)
                uu, ss = u[mm], s[mm]
                v = por_video(uu, ss)
                vals = dict(auc=roc_auc_score(uu["y"], ss),
                            acuracia_balanceada=balanced_accuracy_score(uu["y"], (ss >= LIMIAR).astype(int)),
                            acuracia_voto=float((voto(v) == v["y"].astype(bool)).mean()),
                            acuracia_video=float(((v["s"] >= LIMIAR) == v["y"].astype(bool)).mean()),
                            n_videos=len(v), n_sujeitos=uu["sujeito"].nunique(), n_janelas=len(uu))
                for met, val in vals.items():
                    comp.append(dict(fonte="exp5b", variante=varb, conjunto=conj, modelo=nome, fold=k, metrica=met, valor=val))
    for nome, nome1 in (("RF", "RandomForest"), ("LR", "LogReg")):
        for _, r in e1[e1["model"] == nome1].iterrows():
            comp.append(dict(fonte="exp1", variante="caracteristicas_oficiais", conjunto="oficiais", modelo=nome,
                             fold=int(r["fold"]), metrica=r["metric"], valor=float(r["value"])))
    gravar_csv(pd.DataFrame(comp).assign(pastas_ausentes_do_oficial=";".join(ausentes), execucao=execucao), CSV_EXP1)
    log.info("-> %s", rel(CSV_EXP1))
    log.info("Tempo total das análises: %.1f min", (time.time() - t0) / 60)
    return dict(execucao=execucao)


# ======================================================================================
# Relatório (lê só os CSV)
# ======================================================================================
def br(x, nd=1):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return f"{x:.{nd}f}".replace("-", "−").replace(".", ",")


def pega(tab, **f):
    m = np.ones(len(tab), bool)
    for k, v in f.items():
        m &= (tab[k].astype(str) == str(v)).to_numpy()
    sub = tab[m]
    return sub.iloc[0] if len(sub) == 1 else None


def cel(r, escala=100.0, nd=1, com_ic=True):
    if r is None or not np.isfinite(r["valor"]):
        return "—"
    txt = br(r["valor"] * escala, nd)
    if com_ic and np.isfinite(r["ic95_inf"]):
        txt += f" [{br(r['ic95_inf'] * escala, nd)}; {br(r['ic95_sup'] * escala, nd)}]"
    return txt


def inclui(r, alvo, escala=100.0):
    """O IC 95% contém `alvo`? (None se a linha não existe)"""
    if r is None or not np.isfinite(r["ic95_inf"]):
        return None
    return bool(r["ic95_inf"] * escala <= alvo <= r["ic95_sup"] * escala)


def lado(r, alvo, escala=100.0):
    """+1 se o IC inteiro está acima de `alvo`, -1 se abaixo, 0 se contém (ou não existe)."""
    if r is None or not np.isfinite(r["ic95_inf"]):
        return 0
    if r["ic95_inf"] * escala > alvo:
        return 1
    if r["ic95_sup"] * escala < alvo:
        return -1
    return 0


def sentido_univ(r):
    """+1/-1 se a característica discrimina (IC exclui 0,5 e efeito >= EFEITO_MIN_UNIV); 0 caso contrário."""
    if r is None or not np.isfinite(r["valor"]) or abs(r["valor"] - 0.5) < EFEITO_MIN_UNIV:
        return 0
    return lado(r, 0.5, 1.0)


def pct(x, nd=1):
    return br(100 * x, nd)


class Leitor:
    def __init__(self):
        self.tab = pd.read_csv(CSV_RES)
        self.imp = pd.read_csv(CSV_IMP)
        self.sj = pd.read_csv(CSV_SUJ)
        self.lim = pd.read_csv(CSV_LIM)
        self.fatos = pd.read_csv(CSV_AMOSTRA)
        self.exp1 = pd.read_csv(CSV_EXP1)
        self.pv = pd.read_csv(CSV_PRED_VID)
        self.ear = pd.read_csv(CSV_EAR)
        self.pred = pd.read_csv(CSV_PRED, dtype={"rotulo": str, "sessao": str}, low_memory=False)
        chaves = ["experimento", "variante", "modelo", "esquema", "nivel", "estrato", "limiar_tipo", "metrica"]
        self._d = {tuple(str(v) for v in k): r for k, r in zip(self.tab[chaves].itertuples(index=False, name=None),
                                                               (row for _, row in self.tab.iterrows()))}

    def R(self, exp, var, mod, esq, nivel, met, estrato="todos", lim=LIM_FIXO):
        return self._d.get((exp, var, mod, esq, nivel, estrato, lim, met))

    def fato(self, base, item):
        q = self.fatos[(self.fatos.base == base) & (self.fatos.item == item)]
        return float(q["valor"].iloc[0]) if len(q) else np.nan

    def fato_desc(self, base, item):
        q = self.fatos[(self.fatos.base == base) & (self.fatos.item == item)]
        return str(q["descricao"].iloc[0]) if len(q) and isinstance(q["descricao"].iloc[0], str) else ""

    def limiar_c(self, var, mod):
        q = self.lim[(self.lim.experimento == "C_dmd_para_uta") & (self.lim.variante == var) & (self.lim.modelo == mod)
                     & (self.lim.unidade == "modelo_final")]
        return q.iloc[0] if len(q) else None

    def limiar_oraculo(self, var, mod):
        q = self.lim[(self.lim.experimento == "C_dmd_para_uta") & (self.lim.variante == var) & (self.lim.modelo == mod)
                     & (self.lim.unidade == "oraculo_uta")]
        return q.iloc[0] if len(q) else None

    def limiares(self, exp, var, mod):
        q = self.lim[(self.lim.experimento == exp) & (self.lim.variante == var) & (self.lim.modelo == mod)]
        return q["limiar"].to_numpy(float)

    def imp_perm(self, exp, var, mod, f):
        return pega(self.imp, tipo="permutacao_auc", experimento=exp, variante=var, modelo=mod, caracteristica=f)

    def univ(self, base, var, f):
        return pega(self.imp, tipo="auc_univariada", experimento=base, variante=var, caracteristica=f)

    def media(self, base, var, f, classe, estat="media"):
        r = self.univ(base, var, f)
        return np.nan if r is None else float(r[f"{estat}_{classe}"])

    def top(self, exp, var, mod, k=3):
        q = self.imp[(self.imp.tipo == "permutacao_auc") & (self.imp.experimento == exp) & (self.imp.variante == var)
                     & (self.imp.modelo == mod)].sort_values("valor", ascending=False)
        return list(zip(q["caracteristica"].head(k), q["valor"].head(k)))


def classifica_sinal(lt, f, var="principal"):
    sd = sentido_univ(lt.univ("dmd", var, f))
    su = sentido_univ(lt.univ("uta", var, f))
    if sd != 0 and sd == su:
        return "mesmo sentido (transfere)"
    if sd != 0 and su != 0:
        return "sentido oposto"
    if sd != 0:
        return "discrimina só no DMD"
    if su != 0:
        return "discrimina só na UTA"
    return "não discrimina em nenhuma"


def matriz(lt, exp, var, mod, esq, nivel, lim=LIM_FIXO):
    vals = [lt.R(exp, var, mod, esq, nivel, k, lim=lim) for k in CONTAGENS]
    return " / ".join("—" if v is None else str(int(round(v["valor"]))) for v in vals)


def secao_resposta(lt):
    R = lt.R
    L = ["## 1. Resposta curta (variante principal, esquema B3: KSS 0 × KSS 10)", ""]
    w = L.append
    w("| Métrica | Modelo | (A) DMD → DMD (atuado, LOSO) | (B) UTA → UTA (real, 5 folds) | (C) DMD → UTA (transferência) | "
      "Queda da PP1 (B − C), pp | Choque de realidade (A − C), pp |")
    w("|---|---|---|---|---|---|---|")
    linhas = (("janela", "auc", LIM_FIXO, "AUC por janela"),
              ("janela", "acuracia_balanceada", LIM_FIXO, "Acur. bal. por janela, limiar 0,5"),
              ("janela", "acuracia_balanceada", LIM_CAL, "Acur. bal. por janela, limiar calibrado no treino ¹"),
              ("video", "auc", LIM_FIXO, "AUC por vídeo"),
              ("video", "acuracia", LIM_FIXO, "Acurácia por vídeo, limiar 0,5"),
              ("video", "acuracia", LIM_CAL, "Acurácia por vídeo, limiar calibrado no treino ¹"))
    for nivel, met, lim, rot in linhas:
        for mod in MODELOS:
            a = R("A_dmd_intra", "principal", mod, "DMD", "janela", met, lim=lim) if nivel == "janela" else None
            b = R("B_uta_intra", "principal", mod, "B3", nivel, met, lim=lim)
            c = R("C_dmd_para_uta", "principal", mod, "B3", nivel, met, lim=lim)
            q = R("queda_B_menos_C", "principal", mod, "B3", nivel, met, lim=lim)
            k = R("choque_A_menos_C", "principal", mod, "B3", "janela", met, lim=lim) if nivel == "janela" else None
            ta = cel(a) if nivel == "janela" else "n/a ²"
            tk = cel(k, 1) if nivel == "janela" else "n/a"
            w(f"| {rot} | {mod} | {ta} | {cel(b)} | {cel(c)} | {cel(q, 1)} | {tk} |")
    w("")
    tc = {m: lt.limiar_c("principal", m) for m in MODELOS}
    w("¹ Limiar escolhido só com o treino de cada modelo, pela máxima acurácia balanceada nos escores fora da amostra: "
      f"C usa o limiar do LOSO do DMD (RF {br(tc['RF']['limiar'], 3)}; LR {br(tc['LR']['limiar'], 3)}), sem olhar a UTA; "
      "A usa um LOSO aninhado (o sujeito de teste fica fora da escolha); B, uma validação interna nos 4 folds de treino "
      "de cada fold (seção 6). ² Uma sessão do DMD mistura janelas alerta e sonolentas (e a classe alerta vem de outra "
      "sessão, a s2): não há rótulo por vídeo no DMD.")
    w("")
    w("Leitura automática (regras fixas sobre os IC; a interpretação está na seção 14):")
    w("")
    txt = {1: "acima do acaso", -1: "abaixo do acaso", 0: "indistinguível do acaso (IC contém 50)"}
    for mod in MODELOS:
        c_auc = R("C_dmd_para_uta", "principal", mod, "B3", "janela", "auc")
        q_auc = R("queda_B_menos_C", "principal", mod, "B3", "janela", "auc")
        q_bal = R("queda_B_menos_C", "principal", mod, "B3", "janela", "acuracia_balanceada")
        q_balc = R("queda_B_menos_C", "principal", mod, "B3", "janela", "acuracia_balanceada", lim=LIM_CAL)
        s05 = R("C_dmd_para_uta", "principal", mod, "B3", "janela", "sensibilidade")["valor"]
        e05 = R("C_dmd_para_uta", "principal", mod, "B3", "janela", "especificidade")["valor"]
        sc = R("C_dmd_para_uta", "principal", mod, "B3", "janela", "sensibilidade", lim=LIM_CAL)["valor"]
        ec = R("C_dmd_para_uta", "principal", mod, "B3", "janela", "especificidade", lim=LIM_CAL)["valor"]
        w(f"- **{NOME_MODELO[mod]}**: AUC da transferência {txt[lado(c_auc, 50)]}; queda de AUC B − C com IC que "
          f"{'inclui' if inclui(q_auc, 0, 1) else 'exclui'} 0; queda de acurácia balanceada com IC que "
          f"{'inclui' if inclui(q_bal, 0, 1) else 'exclui'} 0 no limiar 0,5 e que "
          f"{'inclui' if inclui(q_balc, 0, 1) else 'exclui'} 0 com os limiares calibrados no treino; na UTA o modelo do "
          f"DMD acerta {pct(s05)}% das janelas sonolentas e {pct(e05)}% das alertas no limiar 0,5, e {pct(sc)}% / "
          f"{pct(ec)}% no limiar calibrado no DMD ({br(tc[mod]['limiar'], 3)}).")
    w("")
    return L


def secao_desenho(lt):
    L = ["## 2. Desenho e decisões", ""]
    w = L.append
    zc = {b: lt.fato(b, "z_definido_celulas") for b in ("dmd", "uta")}
    zt = {b: lt.fato(b, "z_definido_todas_caracteristicas") for b in ("dmd", "uta")}
    feats_z_uta = [f for f in FEATURES if lt.fato("uta", f"z_definido_{f}") >= 1.0 - 1e-9]
    w("| Item | Decisão | Por quê |")
    w("|---|---|---|")
    w("| Unidade de análise | janela de **10 s, passo 5 s** (`data/features/final/<base>/windows_10s.csv`) | os trechos "
      "`driver_actions/safe_drive` do DMD são curtos demais para as janelas de 30 s da Entrega 01 "
      "(`results/exp5_caracteristicas.md`; relatório da Entrega 02); 5 s e 15 s como sensibilidade |")
    w("| Classe alerta (DMD) | janelas inteiramente dentro de um trecho safe_drive da **sessão s2** (carro parado, direção "
      "encenada) | mesma condição de carro parado da s5: a diferença entre as classes é a sonolência encenada; a s1 (carro "
      "em movimento) é sensibilidade |")
    w("| Classe sonolento (DMD) | janelas da **s5** com fechamento anotado ≥ 0,5 s ou bocejo anotado (≥ 0,5 s dentro da "
      "janela) | único rótulo por janela que a anotação OpenLABEL da s5 permite; janelas da s5 **sem** sinal ficam fora "
      "(ambíguas; como alerta, só na sensibilidade) |")
    w("| Rótulo da UTA | do vídeo (KSS autorrelatado 0/5/10): B3 = 0 × 10 (principal), B1 = 0 × 5+10, B2 = 0+5 × 10 | "
      "mesmas harmonizações do Exp. 1; o DMD é binário, então B3 é o par comparável |")
    w("| Referência de normalização | DMD: janelas safe_drive da **s3** do sujeito; UTA: **último terço do vídeo 0**; "
      "nunca entram em treino/teste, nem as janelas de fronteira (que dividem quadros com a referência) | regra final "
      "(item 10 das notas): na rodada preliminar as janelas alerta do DMD sobrepunham a referência e ficavam perto de 0 "
      f"por construção. Nesta rodada, janelas avaliadas que dividem tempo com a referência: DMD "
      f"{int(lt.fato('dmd', 'sobreposicao_ref_10s'))}, UTA {int(lt.fato('uta', 'sobreposicao_ref_10s'))} |")
    w("| Normalização por sujeito (principal) | **centragem** `<f>_c` = x − média das janelas de referência do sujeito; "
      "a escala é aprendida no treino (padronização da LR ajustada só no treino) | o z-score continua sem estimativa: "
      f"com desvio > 0 e ≥ 5 janelas de referência, ele existe em {pct(zc['dmd'], 0)}% das células sujeito × característica "
      f"do DMD ({int(lt.fato('dmd', 'unidades_ref_5_ou_mais'))} de {int(lt.fato('dmd', 'unidades_ref'))} sujeitos têm "
      f"≥ 5 janelas de referência) e em {pct(zc['uta'], 0)}% na UTA; nas 11 características ao mesmo tempo, em "
      f"{pct(zt['dmd'], 0)}% dos sujeitos do DMD e {pct(zt['uta'], 0)}% dos da UTA (na UTA, só "
      + ", ".join(f"`{f}`" for f in feats_z_uta) + " têm z em todos os sujeitos). Centrar é a parte do z-score que "
      "existe nas duas bases |")
    w("| Características | as 11 da janela: " + ", ".join(f"`{f}`" for f in FEATURES) + " | em segundos, Hz e razões: não "
      "dependem do fps (que muda entre vídeos do mesmo sujeito na UTA) |")
    w("| Modelos | RandomForest (300 árvores, `class_weight=\"balanced\"`, `random_state=42`); LogisticRegression "
      "(`class_weight=\"balanced\"`, `max_iter=5000`) após `StandardScaler`; os dois após `SimpleImputer(median)` "
      "ajustado no treino | hiperparâmetros fixos (os da rodada preliminar); nenhum ajuste olhando o teste. A imputação "
      "é necessária porque janelas de 10 s sem piscada não têm duração/amplitude/velocidade (DMD "
      f"{pct(lt.fato('dmd', 'frac_sem_piscada'))}% das janelas do principal; UTA {pct(lt.fato('uta', 'frac_sem_piscada'))}% "
      "das avaliáveis) |")
    w("| Limiar | (i) **0,5 fixo a priori**; (ii) **calibrado no treino**: máxima acurácia balanceada nos escores fora "
      "da amostra do próprio treino (C: LOSO do DMD, sem olhar a UTA; A: LOSO aninhado; B: validação interna nos 4 folds "
      "de treino); empate → o mais próximo de 0,5 | a mudança de domínio desloca o limiar ótimo; escolher o limiar no "
      "DMD é a única calibração honesta de C. O limiar que maximiza a acurácia na própria UTA aparece só como "
      "diagnóstico (seção 6), nunca como resultado |")
    w("| Validação | A: LOSO nos 13 sujeitos do DMD (as 2 sessões s5 de gB_10, gF_23 e gZ_33 ficam juntas; gE_28, sem s5, "
      "fica fora); B: os **5 folds oficiais** (teste = fold k, treino = os outros 4; mesma partição do Exp. 1); C: um "
      "modelo treinado em todo o DMD e aplicado a todas as janelas avaliáveis da UTA | por sujeito, sem vazamento de "
      "identidade (Exp. 2) |")
    w("| Por vídeo | escore do **vídeo lógico** = média dos escores das janelas (as partes `_p1`/`_p2` de F3_32_10 e "
      "F5_49_10 contam como um vídeo); limiar 0,5 (regra da Entrega 01) e o calibrado; AUC dos escores médios; a "
      "acurácia por voto majoritário serve só para comparar com o Exp. 1 | na UTA cada vídeo tem um rótulo |")
    w("| IC 95% | bootstrap percentil sobre **sujeitos**, 1000 reamostragens, seed 42 (UTA sorteada antes do DMD); os "
      "modelos não são retreinados na reamostragem | janelas do mesmo sujeito são correlacionadas (sobreposição de 5 s) |")
    w("| Quedas | PP1 = **B − C**, pareada (as mesmas reamostragens de sujeitos da UTA em B e C); choque de realidade = "
      "**A − C** e laboratório → real com treino real = A − B, com reamostragens independentes das duas bases | "
      "especificação |")
    w("| Controle do vídeo de referência | modelos de B3 (cada sujeito com o modelo do fold em que foi teste) e de C "
      "aplicados às janelas KSS 5: AUC KSS 10 × 5 e KSS 5 × 0 | as janelas alerta da UTA saem do mesmo vídeo da "
      "referência; as sonolentas, de outro (seção 9) |")
    w("| Importância | queda da AUC agrupada ao embaralhar a característica entre as janelas do MESMO sujeito de teste "
      "(20 repetições; A e B com o modelo de cada partição, C com o modelo único); AUC univariada por base; coeficientes "
      "padronizados da LR | a mesma permutação dentro do sujeito em A, B e C |")
    w("| Sensibilidades | sem normalização; alerta da s1; s5 sem sinal como alerta; janelas de 5 s e 15 s; referência do "
      "DMD na s2 (espelha a UTA); sem gE_29 (falha do detector) | pedidas na especificação, mais duas motivadas pelos "
      "dados (seções 8 e 11) |")
    w("")
    return L


def secao_amostra(lt):
    R = lt.R
    L = ["## 3. Tamanho da amostra", ""]
    w = L.append
    w("| Conjunto | Unidades | Sonolento | Alerta / não sonolento | Sujeitos | Sujeitos com as duas classes |")
    w("|---|---:|---:|---:|---:|---:|")
    itens = [("A_dmd_intra", v, "DMD", "janela", f"DMD, variante `{v}` (janelas)") for v in VARIANTES]
    itens += [("B_uta_intra", "principal", e, "janela", f"UTA {e} (janelas avaliáveis, 10 s)") for e in ESQUEMAS]
    itens += [("B_uta_intra", "principal", e, "video", f"UTA {e} (vídeos lógicos)") for e in ESQUEMAS]
    itens += [("B_uta_intra", "janela_5s", "B3", "janela", "UTA B3 (janelas de 5 s)"),
              ("B_uta_intra", "janela_15s", "B3", "janela", "UTA B3 (janelas de 15 s)"),
              ("B_uta_intra", "uta_ref_no_teste", "B3", "janela", "UTA B3 com referência e fronteira (sensibilidade)")]
    for exp, var, esq, nivel, rot in itens:
        r = R(exp, var, "RF", esq, nivel, "auc")
        if r is None:
            continue
        dc = r["n_sujeitos_duas_classes"]
        w(f"| {rot} | {int(r['n_unidades'])} | {int(r['n_pos'])} | {int(r['n_neg'])} | {int(r['n_sujeitos'])} | "
          f"{int(dc) if np.isfinite(dc) else ''} |")
    w("")
    sem = [v for v in lt.fato_desc("uta", "videos_sem_piscada").split(";") if v]
    if sem:
        partes = [f"`{v}` ({int(lt.fato('uta', f'janelas_video_sem_piscada_{v}'))} janelas; "
                  f"{lt.fato_desc('uta', f'janelas_video_sem_piscada_{v}').split('(')[-1].rstrip(')')})" for v in sem]
        w(f"**Vídeos sem nenhuma piscada detectada** ({len(sem)} de {int(lt.fato('uta', 'videos'))}): " + "; ".join(partes) +
          ". O EAR nunca cruza o limiar calibrado no último terço do vídeo 0 do sujeito (o nível de olho aberto subiu; "
          "`results/exp5_caracteristicas.md`, seção 5.5): nessas janelas PERCLOS, taxa de piscadas e fechamentos valem 0 "
          "e duração/amplitude/velocidade são imputadas. Eles ficam na avaliação (é o que o app veria); os escores "
          "deles estão em `results/exp5b_predicoes_video.csv`.")
        w("")
    pre = ""
    if np.isfinite(lt.fato("preliminar", "A_alerta")):
        pre = (f" (na rodada preliminar: {int(lt.fato('preliminar', 'A_alerta'))} alerta de "
               f"{int(lt.fato('preliminar', 'A_sujeitos_alerta'))} sujeitos, sobrepostas à referência)")
    w(f"Com janelas de 10 s o DMD passa a ter {int(lt.fato('dmd', 'alerta_principal_10s'))} janelas alerta (safe_drive da "
      f"s2) contra {int(lt.fato('dmd', 'sonolento_principal_10s'))} sonolentas, com os "
      f"{int(lt.fato('dmd', 'sujeitos_duas_classes_10s'))} sujeitos nas duas classes{pre}. Com 15 s sobram "
      f"{int(lt.fato('dmd', 'alerta_principal_15s'))} janelas alerta de {int(lt.fato('dmd', 'sujeitos_com_alerta_15s'))} "
      "sujeitos: essa sensibilidade é pouco informativa no DMD.")
    w("")
    return L


def tabela_janela(lt, var, titulo):
    R = lt.R
    L = [titulo, ""]
    w = L.append
    w("| Experimento | Modelo | Esquema | Acur. bal. | F1 macro | AUC | AUC média intra-sujeito | Sens. | Espec. | TN / FP / FN / TP |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    vb = VARIANTES[var]["b"]
    linhas = [("A_dmd_intra", "DMD → DMD (A)", "DMD", var)]
    for esq in ESQUEMAS:
        linhas += [("B_uta_intra", "UTA → UTA (B)", esq, vb), ("C_dmd_para_uta", "DMD → UTA (C)", esq, var)]
    for exp, rot, esq, vv in linhas:
        for mod in MODELOS:
            g = lambda met: R(exp, vv, mod, esq, "janela", met)  # noqa: E731
            if g("auc") is None:
                continue
            w(f"| {rot} | {mod} | {esq} | {cel(g('acuracia_balanceada'))} | {cel(g('f1_macro'))} | {cel(g('auc'))} | "
              f"{cel(g('auc_media_intra_sujeito'))} | {cel(g('sensibilidade'), com_ic=False)} | "
              f"{cel(g('especificidade'), com_ic=False)} | {matriz(lt, exp, vv, mod, esq, 'janela')} |")
    w("")
    return L


def secao_janela(lt):
    L = ["## 4. Resultados por janela (limiar 0,5)", ""]
    L += tabela_janela(lt, "principal", "### 4.1 Variante `principal` — " + DESC_VARIANTE["principal"])
    L += tabela_janela(lt, "sem_normalizacao", "### 4.2 Variante `sem_normalizacao` — " + DESC_VARIANTE["sem_normalizacao"])
    L += ["\"AUC média intra-sujeito\" = média, entre os sujeitos com as duas classes, da AUC calculada só com as janelas "
          "de cada sujeito: mede se o modelo ordena os estados DENTRO da mesma pessoa (o que importa num app calibrado por "
          "usuário), sem exigir que a escala dos escores seja comparável entre pessoas. Sens. = acerto nas janelas "
          "sonolentas; Espec. = acerto nas alertas.", ""]
    return L


def secao_video(lt):
    R = lt.R
    L = ["## 5. Resultados por vídeo (UTA; escore do vídeo lógico = média dos escores das janelas)", ""]
    w = L.append
    w("| Variante | Experimento | Modelo | Esquema | Acurácia | Acur. bal. | F1 macro | AUC | Acurácia por voto | TN / FP / FN / TP |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for var in VARIANTES_CENTRAIS:
        for esq in ESQUEMAS:
            for exp, rot in (("B_uta_intra", "UTA → UTA (B)"), ("C_dmd_para_uta", "DMD → UTA (C)")):
                for mod in MODELOS:
                    g = lambda met: R(exp, var, mod, esq, "video", met)  # noqa: E731
                    if g("auc") is None:
                        continue
                    w(f"| `{var}` | {rot} | {mod} | {esq} | {cel(g('acuracia'))} | {cel(g('acuracia_balanceada'))} | "
                      f"{cel(g('f1_macro'))} | {cel(g('auc'))} | {cel(g('acuracia_voto'))} | "
                      f"{matriz(lt, exp, var, mod, esq, 'video')} |")
    w("")
    return L


def secao_limiar(lt):
    R = lt.R
    L = ["## 6. Limiar: 0,5 fixo × calibrado só com o treino (B3)", ""]
    w = L.append
    w("A mudança de domínio desloca a escala dos escores. A única calibração honesta de C é a que usa só o DMD: o "
      "limiar que maximiza a acurácia balanceada nos escores fora da amostra do LOSO do DMD (os de A). Para comparar "
      "de igual para igual, A e B também recebem um limiar escolhido só com o próprio treino (A: LOSO aninhado, um "
      "limiar por sujeito de teste; B: validação interna nos 4 folds de treino, um limiar por fold). Limiares em "
      "`results/exp5b_limiares.csv`.")
    w("")
    w("| Variante | Modelo | Limiar de C (LOSO do DMD) | A: acur. bal. 0,5 → calibrado | B: acur. bal. 0,5 → calibrado (limiares por fold) | "
      "C: acur. bal. 0,5 → calibrado | C: sens. / espec. no 0,5 | C: sens. / espec. calibrado | "
      "C: acurácia por vídeo 0,5 → calibrado |")
    w("|---|---|---|---|---|---|---|---|---|")
    for var in VARIANTES_CENTRAIS:
        for mod in MODELOS:
            tc = lt.limiar_c(var, mod)
            lb = lt.limiares("B_uta_intra", var, mod)
            g = lambda exp, met, lim, esq="B3", nivel="janela": R(exp, var, mod, esq, nivel, met, lim=lim)  # noqa: E731
            w(f"| `{var}` | {mod} | {br(tc['limiar'], 3)} | {cel(g('A_dmd_intra', 'acuracia_balanceada', LIM_FIXO, 'DMD'))} → "
              f"{cel(g('A_dmd_intra', 'acuracia_balanceada', LIM_CAL, 'DMD'))} | "
              f"{cel(g('B_uta_intra', 'acuracia_balanceada', LIM_FIXO))} → {cel(g('B_uta_intra', 'acuracia_balanceada', LIM_CAL))} "
              f"({br(lb.min(), 3)}–{br(lb.max(), 3)}) | "
              f"{cel(g('C_dmd_para_uta', 'acuracia_balanceada', LIM_FIXO))} → {cel(g('C_dmd_para_uta', 'acuracia_balanceada', LIM_CAL))} | "
              f"{pct(g('C_dmd_para_uta', 'sensibilidade', LIM_FIXO)['valor'])} / {pct(g('C_dmd_para_uta', 'especificidade', LIM_FIXO)['valor'])} | "
              f"{pct(g('C_dmd_para_uta', 'sensibilidade', LIM_CAL)['valor'])} / {pct(g('C_dmd_para_uta', 'especificidade', LIM_CAL)['valor'])} | "
              f"{cel(g('C_dmd_para_uta', 'acuracia', LIM_FIXO, nivel='video'))} → {cel(g('C_dmd_para_uta', 'acuracia', LIM_CAL, nivel='video'))} |")
    w("")
    w("**Diagnóstico (não é resultado):** o limiar que maximiza a acurácia balanceada de C na própria UTA olha o "
      "rótulo do teste e mede só o teto de uma recalibração perfeita. Ele mostra quanto da queda no limiar é "
      "calibração e quanto é ordenação (a AUC não depende do limiar).")
    w("")
    w("| Variante | Modelo | Limiar oráculo na UTA | C: acur. bal. no oráculo | C: acur. bal. no limiar do DMD | Limiar do DMD |")
    w("|---|---|---|---|---|---|")
    for var in VARIANTES_CENTRAIS:
        for mod in MODELOS:
            to, tc = lt.limiar_oraculo(var, mod), lt.limiar_c(var, mod)
            w(f"| `{var}` | {mod} | {br(to['limiar'], 3)} | {pct(to['acur_bal_na_calibracao'])} | "
              f"{cel(R('C_dmd_para_uta', var, mod, 'B3', 'janela', 'acuracia_balanceada', lim=LIM_CAL), com_ic=False)} | "
              f"{br(tc['limiar'], 3)} |")
    w("")
    return L


def secao_quedas(lt):
    R = lt.R
    L = ["## 7. Quedas: PP1 (B − C), choque de realidade (A − C) e laboratório → real com treino real (A − B)", ""]
    w = L.append
    w("B − C é pareada (as mesmas reamostragens de sujeitos da UTA entram em B e em C); A − C e A − B combinam "
      "reamostragens independentes do DMD e da UTA. Em pp, com IC 95%; positivo = o segundo termo é pior. Com limiar "
      "calibrado, cada experimento usa o limiar escolhido no próprio treino (seção 6).")
    w("")
    w("| Comparação | Variante | Modelo | Esquema | Nível | Limiar | Δ acur. bal. | Δ F1 macro | Δ AUC | Δ AUC intra-sujeito | Δ acurácia | Δ acurácia por voto |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for nome, rot in (("queda_B_menos_C", "B − C"), ("choque_A_menos_C", "A − C"), ("lab_A_menos_B", "A − B")):
        for var in VARIANTES_CENTRAIS:
            for esq in ESQUEMAS:
                for nivel in (("janela", "video") if nome == "queda_B_menos_C" else ("janela",)):
                    for lim in (LIM_FIXO, LIM_CAL):
                        for mod in MODELOS:
                            g = lambda met: R(nome, var, mod, esq, nivel, met, lim=lim)  # noqa: E731
                            if g("acuracia_balanceada") is None:
                                continue
                            w(f"| {rot} | `{var}` | {mod} | {esq} | {nivel} | {'0,5' if lim == LIM_FIXO else 'calibrado'} | "
                              f"{cel(g('acuracia_balanceada'), 1)} | {cel(g('f1_macro'), 1)} | {cel(g('auc'), 1)} | "
                              f"{cel(g('auc_media_intra_sujeito'), 1)} | {cel(g('acuracia'), 1)} | {cel(g('acuracia_voto'), 1)} |")
    w("")
    return L


def secao_sensibilidades(lt):
    R = lt.R
    L = ["## 8. Sensibilidades (B3; por janela salvo indicação)", ""]
    w = L.append
    w("Cada linha muda UMA escolha em relação à principal. B é o da variante indicada em `b` (a UTA não muda nas "
      "variantes que só alteram o DMD). O limiar calibrado de C vem sempre do LOSO do DMD da própria variante.")
    w("")
    w("| Variante | DMD: sonolento / alerta (sujeitos) | A: AUC | B: AUC | C: AUC | Queda B − C (AUC), pp | "
      "C: acur. bal. 0,5 | C: acur. bal. no limiar do DMD (limiar) | C: AUC por vídeo | Queda B − C (AUC por vídeo), pp |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for mod in MODELOS:
        w(f"| **{NOME_MODELO[mod]}** | | | | | | | | | |")
        for var, cfg in VARIANTES.items():
            a = R("A_dmd_intra", var, mod, "DMD", "janela", "auc")
            if a is None:
                continue
            b = R("B_uta_intra", cfg["b"], mod, "B3", "janela", "auc")
            c = R("C_dmd_para_uta", var, mod, "B3", "janela", "auc")
            q = R("queda_B_menos_C", var, mod, "B3", "janela", "auc")
            cb = R("C_dmd_para_uta", var, mod, "B3", "janela", "acuracia_balanceada")
            cbc = R("C_dmd_para_uta", var, mod, "B3", "janela", "acuracia_balanceada", lim=LIM_CAL)
            cv_ = R("C_dmd_para_uta", var, mod, "B3", "video", "auc")
            qv = R("queda_B_menos_C", var, mod, "B3", "video", "auc")
            tc = lt.limiar_c(var, mod)
            w(f"| `{var}` | {int(a['n_pos'])} / {int(a['n_neg'])} ({int(a['n_sujeitos'])}) | {cel(a)} | {cel(b)} | {cel(c)} | "
              f"{cel(q, 1)} | {cel(cb)} | {cel(cbc)} ({br(tc['limiar'], 3)}) | {cel(cv_)} | {cel(qv, 1)} |")
    w("")
    w("Descrição das variantes: " + "; ".join(f"`{k}` = {DESC_VARIANTE[k]}" for k in VARIANTES if k != "principal") + ".")
    w("")
    w(f"Na variante `ref_s2_espelho` o alerta do DMD cai para {int(lt.fato('dmd', 'espelho_alerta'))} janelas e a "
      f"referência fica com {int(lt.fato('dmd', 'espelho_ref_min'))}–{int(lt.fato('dmd', 'espelho_ref_max'))} janelas por "
      "sujeito (mais ruidosa); o limiar de EAR continua o calibrado na s3 (só a centragem muda).")
    w("")
    return L


def secao_controle(lt):
    R = lt.R
    L = ["## 9. Controle: B e C separam sonolência ou só \"vídeo diferente da referência\"?", ""]
    w = L.append
    w("As janelas alerta (KSS 0) vêm do mesmo vídeo que a referência da normalização (os 2/3 iniciais do vídeo 0; a "
      "referência é o último terço); as sonolentas (KSS 10), de outro vídeo, gravado em outro momento. Um modelo pode "
      "separar B3 só por \"este trecho não se parece com a referência\" (luz, enquadramento, hora do dia). O KSS 5 também "
      "é outro vídeo: se o modelo separa KSS 10 de KSS 5, há sinal além do vídeo de referência. Os modelos são os mesmos "
      "de B3 (cada sujeito com o modelo do fold em que foi teste) e de C; as janelas KSS 5 nunca entraram no treino.")
    w("")
    w("| Variante | Experimento | Modelo | AUC KSS 0 × 10 (B3) | AUC KSS 5 × 10 | AUC média intra-sujeito 5 × 10 | AUC KSS 0 × 5 | Escore médio KSS 0 / 5 / 10 |")
    w("|---|---|---|---|---|---|---|---|")
    pred = lt.pred
    for var in VARIANTES_CENTRAIS:
        for exp, rot in (("B_uta_intra", "UTA → UTA (B)"), ("C_dmd_para_uta", "DMD → UTA (C)")):
            for mod in MODELOS:
                k105 = R(exp, var, mod, "K10v5", "janela", "auc")
                if k105 is None:
                    continue
                esqs = ["B3", "B3_janelas_kss5"] if exp == "B_uta_intra" else ["todos"]
                q = pred[(pred.experimento == exp) & (pred.variante == var) & (pred.modelo == mod) & pred.esquema.isin(esqs)]
                med = q.groupby(q["rotulo"].astype(float).astype(int))["escore"].mean()
                w(f"| `{var}` | {rot} | {mod} | {cel(R(exp, var, mod, 'B3', 'janela', 'auc'))} | {cel(k105)} | "
                  f"{cel(R(exp, var, mod, 'K10v5', 'janela', 'auc_media_intra_sujeito'))} | "
                  f"{cel(R(exp, var, mod, 'K5v0', 'janela', 'auc'))} | "
                  f"{br(med.get(0, np.nan), 2)} / {br(med.get(5, np.nan), 2)} / {br(med.get(10, np.nan), 2)} |")
    w("")
    w("Queda B − C nesses contrastes (pareada, pp de AUC × 100):")
    w("")
    w("| Variante | Modelo | Δ AUC 5 × 10 | Δ AUC intra-sujeito 5 × 10 | Δ AUC 0 × 5 |")
    w("|---|---|---|---|---|")
    for var in VARIANTES_CENTRAIS:
        for mod in MODELOS:
            a = R("queda_B_menos_C", var, mod, "K10v5", "janela", "auc")
            if a is None:
                continue
            w(f"| `{var}` | {mod} | {cel(a, 1)} | {cel(R('queda_B_menos_C', var, mod, 'K10v5', 'janela', 'auc_media_intra_sujeito'), 1)} | "
              f"{cel(R('queda_B_menos_C', var, mod, 'K5v0', 'janela', 'auc'), 1)} |")
    w("")
    return L


def secao_sinais(lt):
    L = ["## 10. Quais sinais transferem?", ""]
    w = L.append
    w("Importância por permutação = queda da AUC agrupada (× 100) ao embaralhar a característica entre as janelas do "
      "mesmo sujeito de teste (média ± dp de 20 repetições; variante principal; B3 na UTA). AUC univariada = AUC da "
      "característica sozinha (`_c`) como escore de sonolento: > 50 = maior nas janelas sonolentas. Uma característica "
      f"\"discrimina\" numa base quando o IC 95% da AUC univariada exclui 50 e a AUC se afasta ≥ {int(EFEITO_MIN_UNIV * 100)} "
      "pontos de 50 (regra descritiva, não é teste de hipótese). Tabelas ordenadas pela importância em C.")
    w("")
    for mod in MODELOS:
        w(f"**{NOME_MODELO[mod]}**")
        w("")
        w("| Característica | Import. A (DMD → DMD) | Import. B (UTA → UTA) | Import. C (DMD → UTA) | AUC univ. DMD | AUC univ. UTA B3 | Sentido nas duas bases |")
        w("|---|---:|---:|---:|---|---|---|")
        for f, _ in lt.top("C", "principal", mod, k=len(FEATURES)):
            cols_imp = []
            for exp in ("A", "B", "C"):
                r = lt.imp_perm(exp, "principal", mod, f)
                cols_imp.append("—" if r is None else f"{br(r['valor'] * 100)} ± {br(r['dp'] * 100)}")
            w(f"| `{f}` | " + " | ".join(cols_imp) + f" | {cel(lt.univ('dmd', 'principal', f))} | "
              f"{cel(lt.univ('uta', 'principal', f))} | {classifica_sinal(lt, f)} |")
        w("")
    w("As 3 características mais importantes (queda de AUC × 100) em cada modelo e variante:")
    w("")
    w("| Experimento | Variante | RF | LR |")
    w("|---|---|---|---|")
    for exp, rot in (("A", "A (DMD → DMD)"), ("B", "B (UTA → UTA, B3)"), ("C", "C (DMD → UTA, B3)")):
        for var in VARIANTES_CENTRAIS:
            cols_top = []
            for mod in MODELOS:
                t = lt.top(exp, var, mod)
                cols_top.append(", ".join(f"`{f}` {br(v * 100)}" for f, v in t) if t else "—")
            w(f"| {rot} | `{var}` | {cols_top[0]} | {cols_top[1]} |")
    w("")
    w("Coeficientes da regressão logística (escala padronizada; positivo = empurra para \"sonolento\"; A e B: média ± dp "
      "entre os modelos de partição; C: modelo único treinado em todo o DMD; variante principal):")
    w("")
    w("| Característica | A (DMD) | B3 (UTA) | C (todo o DMD) |")
    w("|---|---:|---:|---:|")
    for f in FEATURES:
        cc = []
        for exp in ("A", "B", "C"):
            r = pega(lt.imp, tipo="coef_lr_padronizado", experimento=exp, variante="principal", caracteristica=f)
            cc.append("—" if r is None else (br(r["valor"], 2) + ("" if not np.isfinite(r["dp"]) else f" ± {br(r['dp'], 2)}")))
        w(f"| `{f}` | " + " | ".join(cc) + " |")
    w("")
    w("Nível das características brutas (mediana nas janelas alerta → sonolentas; variante sem normalização):")
    w("")
    w("| Característica | DMD alerta → sonolento | UTA KSS 0 → KSS 10 |")
    w("|---|---|---|")
    for f in FEATURES:
        nd = 3 if f in ("perclos", "ear_media", "ear_dp", "amp_media", "frac_bocejo", "taxa_rosto", "dur_media_piscada_s") else 2
        w(f"| `{f}` | {br(lt.media('dmd', 'sem_normalizacao', f, 'alerta', 'mediana'), nd)} → "
          f"{br(lt.media('dmd', 'sem_normalizacao', f, 'sonolento', 'mediana'), nd)} | "
          f"{br(lt.media('uta', 'sem_normalizacao', f, 'alerta', 'mediana'), nd)} → "
          f"{br(lt.media('uta', 'sem_normalizacao', f, 'sonolento', 'mediana'), nd)} |")
    w("")
    return L


def secao_oculos(lt):
    R = lt.R
    sj = lt.sj[lt.sj["base"] == "dmd"].copy()
    L = ["## 11. PP3 no DMD: óculos (metadado `glasses` do OpenLABEL)", ""]
    w = L.append
    n_oc = int(sj["oculos_s5"].astype(str).str.lower().eq("true").sum())
    w(f"Na UTA-RLDD não há rótulo de óculos por vídeo e, por decisão da equipe, os rostos da UTA não foram anotados: "
      f"a PP3 sobre óculos só pode ser olhada no DMD, onde {n_oc} dos "
      f"{len(sj)} motoristas usam óculos na sessão de sonolência (s5). O desempenho de A por motorista e a validação do "
      "detector de piscadas contra a anotação da s5 (`results/exp5_dmd_validacao_detector.csv`, limiar final) ficam lado "
      "a lado:")
    w("")
    w("| Motorista | Óculos s5 / s2 | s2 no mesmo dia da s5 | Janelas sonol. / alerta | A: AUC RF | A: AUC LR | "
      "A-RF: sens. / espec. (0,5) | Detector: F1 piscadas | Revocação piscadas | Acur. bal. por quadro | "
      "EAR mediano aberto / fechado | PERCLOS médio alerta → sonol. |")
    w("|---|---|---|---|---:|---:|---|---:|---:|---:|---|---|")
    sj = sj.sort_values(["oculos_s5", "sujeito"], ascending=[False, True])
    for _, r in sj.iterrows():
        oc5 = "sim" if str(r["oculos_s5"]).lower() == "true" else "não"
        w(f"| {r['sujeito']} | {oc5} / {r['oculos_s2']} | {'sim' if str(r['s2_mesmo_dia_s5']).lower() == 'true' else 'não'} | "
          f"{int(r['n_sonolento'])} / {int(r['n_alerta'])} | {br(r['auc_A_RF'] * 100)} | {br(r['auc_A_LR'] * 100)} | "
          f"{pct(r['sens_A_RF'])} / {pct(r['espec_A_RF'])} | {br(r['det_f1_piscadas'], 3)} | {br(r['det_revocacao_piscadas'], 3)} | "
          f"{br(r['det_acur_bal_quadro'], 3)} | {br(r['det_ear_mediana_aberto'], 3)} / {br(r['det_ear_mediana_fechado'], 3)} | "
          f"{br(r['media_perclos_alerta'], 3)} → {br(r['media_perclos_sonolento'], 3)} |")
    w("")
    w("O EAR sem limiar: AUC de −EAR para separar os quadros anotados \"fechado\" dos \"aberto\" (transições fora), "
      "medianas do EAR nas duas classes e fração dos quadros fechados abaixo do limiar do sujeito (0,75 × EAR mediano da "
      "referência; `results/exp5b_oculos_ear_dmd.csv`). Uma AUC alta com queda relativa pequena indica que o sinal existe, "
      "mas o limiar relativo não o alcança:")
    w("")
    w("| Sessão s5 | Óculos | AUC −EAR (média dos olhos) | AUC −EAR esq. / dir. | EAR mediano aberto → fechado | Queda relativa | "
      "Limiar | Quadros fechados abaixo do limiar | F1 piscadas |")
    w("|---|---|---:|---|---|---:|---:|---:|---:|")
    ear = lt.ear.sort_values(["oculos", "sujeito", "id"], ascending=[False, True, True])
    for _, r in ear.iterrows():
        w(f"| {r['id'].split('+')[0].replace(';', ':')} | {'sim' if r['oculos'] else 'não'} | {br(r['auc_menos_ear'] * 100)} | "
          f"{br(r['auc_menos_ear_l'] * 100)} / {br(r['auc_menos_ear_r'] * 100)} | {br(r['ear_mediana_aberto'], 3)} → "
          f"{br(r['ear_mediana_fechado'], 3)} | {pct(r['queda_relativa'])}% | {br(r['limiar'], 3)} | "
          f"{pct(r['frac_fechado_abaixo_limiar'])}% | {br(r['det_f1_piscadas'], 3)} |")
    w("")
    w(f"A agregada por estrato de sujeitos (IC por bootstrap dos sujeitos do estrato; com {n_oc} sujeitos o IC é quase "
      "degenerado — ver a coluna de reamostragens válidas):")
    w("")
    w("| Variante | Modelo | Estrato | Sujeitos | Janelas sonol. / alerta | Acur. bal. (0,5) | AUC | AUC média intra-sujeito | Reamostragens válidas (AUC) |")
    w("|---|---|---|---:|---|---|---|---|---:|")
    for var in VARIANTES_CENTRAIS:
        for mod in MODELOS:
            for est, rot in (("oculos_s5", "com óculos (s5)"), ("sem_oculos_s5", "sem óculos (s5)"),
                             ("s2_mesmo_dia_s5", "s2 no mesmo dia da s5"), ("s2_outro_dia", "s2 em outro dia")):
                a = R("A_dmd_intra", var, mod, "DMD", "janela", "auc", estrato=est)
                if a is None:
                    continue
                w(f"| `{var}` | {mod} | {rot} | {int(a['n_sujeitos'])} | {int(a['n_pos'])} / {int(a['n_neg'])} | "
                  f"{cel(R('A_dmd_intra', var, mod, 'DMD', 'janela', 'acuracia_balanceada', estrato=est))} | {cel(a)} | "
                  f"{cel(R('A_dmd_intra', var, mod, 'DMD', 'janela', 'auc_media_intra_sujeito', estrato=est))} | "
                  f"{int(a['n_boot_validos'])} |")
    w("")
    w("Sensibilidade sem o motorista em que o detector falhou (`sem_gE_29`; seção 8):")
    w("")
    w("| Modelo | A: AUC (principal → sem gE_29) | C: AUC (principal → sem gE_29) | C: acur. bal. no limiar do DMD (principal → sem gE_29) |")
    w("|---|---|---|---|")
    for mod in MODELOS:
        g = lambda exp, var, met, lim=LIM_FIXO, esq="B3": R(exp, var, mod, esq, "janela", met, lim=lim)  # noqa: E731
        w(f"| {mod} | {cel(g('A_dmd_intra', 'principal', 'auc', esq='DMD'))} → {cel(g('A_dmd_intra', 'sem_gE_29', 'auc', esq='DMD'))} | "
          f"{cel(g('C_dmd_para_uta', 'principal', 'auc'))} → {cel(g('C_dmd_para_uta', 'sem_gE_29', 'auc'))} | "
          f"{cel(g('C_dmd_para_uta', 'principal', 'acuracia_balanceada', LIM_CAL))} → "
          f"{cel(g('C_dmd_para_uta', 'sem_gE_29', 'acuracia_balanceada', LIM_CAL))} |")
    w("")
    return L


def secao_por_sujeito(lt):
    sj = lt.sj[lt.sj["base"] == "uta"]
    L = ["## 12. Por sujeito da UTA (B3, variante principal)", ""]
    w = L.append
    for mod in MODELOS:
        b, c = sj[f"auc_B_{mod}"], sj[f"auc_C_{mod}"]
        ok = b.notna() & c.notna()
        rho = pd.Series(b[ok]).corr(pd.Series(c[ok]), method="spearman")
        w(f"- {mod}: AUC intra-sujeito > 50 em {int((b > 0.5).sum())} de {len(sj)} sujeitos em B e em "
          f"{int((c > 0.5).sum())} em C; mediana {br(b.median() * 100)} (B) e {br(c.median() * 100)} (C); C melhor que B "
          f"em {int((c > b).sum())} sujeitos; correlação de Spearman entre as AUC de B e C por sujeito {br(rho, 2)}.")
    for mod in MODELOS:
        q = lt.pred[(lt.pred.experimento == "C_dmd_para_uta") & (lt.pred.variante == "principal") & (lt.pred.modelo == mod)]
        k = q["rotulo"].astype(float).astype(int)
        f05 = (q["escore"] >= LIMIAR).groupby(k).mean()
        fc = (q["escore"] >= q["limiar_calibrado"]).groupby(k).mean()
        w(f"- C-{mod}: fração de janelas previstas \"sonolento\" por KSS 0 / 5 / 10 = {pct(f05.get(0, np.nan))}% / "
          f"{pct(f05.get(5, np.nan))}% / {pct(f05.get(10, np.nan))}% no limiar 0,5 e {pct(fc.get(0, np.nan))}% / "
          f"{pct(fc.get(5, np.nan))}% / {pct(fc.get(10, np.nan))}% no limiar do DMD.")
    for mod in MODELOS:
        inv = sj[sj[f"auc_C_{mod}"] < 0.5].sort_values(f"auc_C_{mod}")
        if len(inv):
            w(f"- C-{mod}: AUC intra-sujeito < 50 (ordem invertida) em {len(inv)} sujeitos: " + ", ".join(
                f"{r['sujeito']} ({br(r[f'auc_C_{mod}'] * 100)}; B {br(r[f'auc_B_{mod}'] * 100)})" for _, r in inv.iterrows()) + ".")
    com_sem = sj[sj["videos_sem_piscada"].fillna("").astype(str).str.len() > 0]
    for _, r in com_sem.iterrows():
        vids = str(r["videos_sem_piscada"]).split(";")
        no_b3 = [v for v in vids if v.rsplit("_", 1)[-1] in ("0", "10")]
        nota = ("o vídeo entra no B3: o limiar de EAR calibrado no vídeo 0 não é cruzado nele e todas as janelas parecem "
                "alerta" if no_b3 else "o vídeo é KSS 5 e não entra no B3; a AUC acima não depende dele")
        w(f"- {r['sujeito']} tem vídeo sem nenhuma piscada detectada ({r['videos_sem_piscada']}): AUC intra-sujeito B-RF "
          f"{br(r['auc_B_RF'] * 100)}, C-RF {br(r['auc_C_RF'] * 100)}, B-LR {br(r['auc_B_LR'] * 100)}, C-LR "
          f"{br(r['auc_C_LR'] * 100)} ({nota}; seção 3).")
    w("")
    w("Tabela completa no Apêndice A e em `results/exp5b_por_sujeito.csv`.")
    w("")
    return L


def secao_exp1(lt):
    e = lt.exp1
    L = ["## 13. Relação com o Exp. 1 (mesmos 5 folds oficiais, B3)", ""]
    w = L.append
    aus = str(e["pastas_ausentes_do_oficial"].iloc[0]).replace(";", ", ")
    w("O Exp. 1 (não recalculado; `results/exp1_intra.csv`) treinou nas **características oficiais** (sequências de 30 "
      "piscadas do detector dlib dos autores) e testou com a referência DENTRO do teste (como nos `.npy`). O B deste "
      "experimento usa as MESMAS partições (teste = fold k), mas as nossas janelas de 10 s do MediaPipe, sem a "
      "referência no teste. Métricas por fold, depois a média dos 5 folds (a agregação do Exp. 1). `oficiais` = sem as "
      f"pastas que não estão no arquivo oficial ({aus}), para ter os mesmos sujeitos; `com ref.` = a sensibilidade "
      "`uta_ref_no_teste` (referência e fronteira no treino e no teste).")
    w("")
    w("| Modelo | Fonte | AUC por fold (1 / 2 / 3 / 4 / 5) | AUC média | Acur. bal. média | VA por voto média | Acurácia por vídeo (média dos escores) média |")
    w("|---|---|---|---:|---:|---:|---:|")
    for mod in MODELOS:
        fontes = (("Exp. 1 (oficiais, com ref.)", e[(e.fonte == "exp1") & (e.modelo == mod)], "auc_macro", "bal_acc", "va", None),
                  ("Exp. 5b B, oficiais", e[(e.fonte == "exp5b") & (e.variante == "principal") & (e.conjunto == "oficiais") & (e.modelo == mod)],
                   "auc", "acuracia_balanceada", "acuracia_voto", "acuracia_video"),
                  ("Exp. 5b B, todos os 60", e[(e.fonte == "exp5b") & (e.variante == "principal") & (e.conjunto == "todos") & (e.modelo == mod)],
                   "auc", "acuracia_balanceada", "acuracia_voto", "acuracia_video"),
                  ("Exp. 5b B com ref., oficiais", e[(e.fonte == "exp5b") & (e.variante == "uta_ref_no_teste") & (e.conjunto == "oficiais") & (e.modelo == mod)],
                   "auc", "acuracia_balanceada", "acuracia_voto", "acuracia_video"))
        for rot, q, m_auc, m_bal, m_va, m_vid in fontes:
            if q.empty:
                continue
            por = q[q.metrica == m_auc].sort_values("fold")["valor"].to_numpy()
            med = lambda m: (q[q.metrica == m]["valor"].mean() if m else np.nan)  # noqa: E731
            w(f"| {mod} | {rot} | {' / '.join(br(v * 100) for v in por)} | {br(med(m_auc) * 100)} | {br(med(m_bal) * 100)} | "
              f"{br(med(m_va) * 100)} | {br(med(m_vid) * 100) if m_vid else '—'} |")
    w("")
    return L


def apendice_sujeitos(lt):
    sj = lt.sj[lt.sj["base"] == "uta"]
    L = ["## Apêndice A. Por sujeito da UTA (B3, variante principal): AUC intra-sujeito em B e em C", ""]
    w = L.append
    w("| Sujeito | Fold | Mapeamento oficial | Janelas alerta / sonolento | AUC B (RF) | AUC C (RF) | AUC B (LR) | AUC C (LR) | Escore médio de C-RF: alerta → sonolento | Vídeo sem piscada |")
    w("|---|---:|---|---|---:|---:|---:|---:|---|---|")
    for _, r in sj.iterrows():
        vs = r["videos_sem_piscada"] if isinstance(r["videos_sem_piscada"], str) else ""
        w(f"| {r['sujeito']} | {int(r['fold'])} | {r['mapeamento_oficial']} | {int(r['n_alerta'])} / {int(r['n_sonolento'])} | "
          f"{br(r['auc_B_RF'] * 100)} | {br(r['auc_C_RF'] * 100)} | {br(r['auc_B_LR'] * 100)} | {br(r['auc_C_LR'] * 100)} | "
          f"{br(r['escore_medio_alerta_C_RF'], 2)} → {br(r['escore_medio_sonolento_C_RF'], 2)} | {vs} |")
    w("")
    return L


def leitura(lt):
    """Seção de leitura: frases condicionadas aos IC desta execução."""
    R = lt.R
    L = ["## 14. Leitura", ""]
    w = L.append
    P = "principal"
    g = lambda exp, mod, met, nivel="janela", lim=LIM_FIXO, esq="B3", var=P, est="todos": R(  # noqa: E731
        exp, var, mod, "DMD" if exp == "A_dmd_intra" else esq, nivel, met, estrato=est, lim=lim)
    frase_ic = lambda r: ("o IC inclui 0" if inclui(r, 0, 1) else ("o IC fica todo acima de 0" if lado(r, 0, 1) == 1  # noqa: E731
                                                                  else "o IC fica todo abaixo de 0"))

    # (0) Síntese
    qa_ = {m: g("queda_B_menos_C", m, "auc") for m in MODELOS}
    qv_ = {m: g("queda_B_menos_C", m, "auc", "video") for m in MODELOS}
    qac_ = {m: g("queda_B_menos_C", m, "acuracia", "video") for m in MODELOS}
    cal_ = {m: (g("C_dmd_para_uta", m, "acuracia_balanceada"), g("C_dmd_para_uta", m, "acuracia_balanceada", lim=LIM_CAL))
            for m in MODELOS}
    txt = "**Síntese.** "
    if all(lado(qa_[m], 0, 1) == 1 for m in MODELOS):
        txt += ("Treinar em sonolência ATUADA (DMD) e testar em sonolência REAL (UTA-RLDD) custa, em relação a treinar na "
                f"própria UTA com as mesmas características, {cel(qa_['RF'], 1)} pp de AUC por janela no RF e "
                f"{cel(qa_['LR'], 1)} pp na LR (IC acima de 0 nos dois)")
    else:
        txt += (f"A queda de AUC por janela (B − C) é {cel(qa_['RF'], 1)} pp (RF) e {cel(qa_['LR'], 1)} pp (LR)")
    txt += (f"; por vídeo, {cel(qv_['RF'], 1)} e {cel(qv_['LR'], 1)} pp de AUC e {cel(qac_['RF'], 1)} e "
            f"{cel(qac_['LR'], 1)} pp de acurácia no limiar 0,5. ")
    if all(lado(g("C_dmd_para_uta", m, "auc"), 50) == 1 for m in MODELOS):
        txt += "A transferência fica acima do acaso, mas abaixo do teto real. "
    k105 = {m: g("queda_B_menos_C", m, "auc", esq="K10v5") for m in MODELOS}
    k50 = {m: g("queda_B_menos_C", m, "auc", esq="K5v0") for m in MODELOS}
    if all(k105[m] is not None and lado(k105[m], 0, 1) == 0 for m in MODELOS):
        partes_k = []
        for m in MODELOS:
            if k50[m] is not None and lado(k50[m], 0, 1) == 1:
                partes_k.append(f"{m}: {cel(k105[m], 1)} pp em 5 × 10 contra {cel(k50[m], 1)} pp em 0 × 5 — a vantagem de B "
                                "se concentra no contraste com o vídeo alerta, que é também o vídeo da referência")
            else:
                partes_k.append(f"{m}: {cel(k105[m], 1)} pp em 5 × 10 e {cel(k50[m], 1)} pp em 0 × 5, nenhum com IC "
                                "acima de 0")
        txt += ("**Ressalva importante:** entre dois vídeos que não são a referência da normalização (KSS 5 × 10), a "
                "queda B − C não é detectável (seção 9; " + "; ".join(partes_k) + "). Parte da queda da PP1 no B3 pode, "
                "portanto, vir de B explorar a estrutura da UTA-RLDD (as janelas alerta saem do mesmo vídeo da "
                "referência) — um atalho indisponível para o modelo treinado no DMD —, e não só de o treino atuado "
                "ensinar menos sobre sonolência; a base não permite separar as duas coisas. ")
    piora = [m for m in MODELOS if cal_[m][1]["valor"] < cal_[m][0]["valor"]]
    oposto = [m for m in MODELOS if (lt.limiar_c(P, m)["limiar"] - LIMIAR) * (lt.limiar_oraculo(P, m)["limiar"] - LIMIAR) < 0]
    if piora:
        txt += (f"Calibrar o limiar no DMD — a única calibração honesta — não recupera a perda ({', '.join(piora)}: a "
                "acurácia balanceada cai em vez de subir)")
        if set(piora) <= set(oposto):
            txt += (": o limiar que o DMD sugere se afasta de 0,5 para um lado e o que a UTA precisaria (diagnóstico), "
                    "para o outro")
        txt += ". "
    elif all(cal_[m][1]["valor"] > cal_[m][0]["valor"] for m in MODELOS):
        txt += "Calibrar o limiar no DMD melhora a acurácia balanceada de C nos dois modelos (seção 6). "
    w(txt)
    w("")

    # (1) PP1 — ordenação por janela
    qa = {m: g("queda_B_menos_C", m, "auc") for m in MODELOS}
    txt = (f"**PP1 — ordenação por janela (AUC, B3).** Treinado só no DMD atuado, o modelo ordena as janelas reais da "
           f"UTA com AUC {cel(g('C_dmd_para_uta', 'RF', 'auc'))} (RF) e {cel(g('C_dmd_para_uta', 'LR', 'auc'))} (LR); "
           f"treinado na própria UTA com as mesmas características (B, 5 folds), {cel(g('B_uta_intra', 'RF', 'auc'))} e "
           f"{cel(g('B_uta_intra', 'LR', 'auc'))}. A queda da PP1 (B − C) é {cel(qa['RF'], 1)} pp de AUC no RF "
           f"({frase_ic(qa['RF'])}) e {cel(qa['LR'], 1)} pp na LR ({frase_ic(qa['LR'])}). ")
    if all(lado(qa[m], 0, 1) == 1 for m in MODELOS):
        txt += ("**Há queda real na capacidade de ordenar** ao trocar o treino real pelo atuado, nos dois modelos; "
                "ainda assim, C fica " + ("acima do acaso nos dois" if all(lado(g('C_dmd_para_uta', m, 'auc'), 50) == 1 for m in MODELOS)
                                          else "perto do acaso em pelo menos um modelo") + ". ")
    elif all(inclui(qa[m], 0, 1) for m in MODELOS):
        pior = max(qa[m]["ic95_sup"] for m in MODELOS)
        txt += (f"Nos dois modelos o IC inclui 0: não há queda detectável na ordenação por janela; com "
                f"{int(lt.fato('uta', 'sujeitos'))} sujeitos, o IC ainda admite quedas de até {br(pior, 1)} pp — \"não "
                "detectável\" não é \"zero\". ")
    else:
        txt += "O resultado depende do classificador (um IC exclui 0, o outro não). "
    ret = {m: (g("C_dmd_para_uta", m, "auc")["valor"] - 0.5) / (g("B_uta_intra", m, "auc")["valor"] - 0.5) for m in MODELOS}
    txt += ("Em proporção, C preserva " + " e ".join(f"{pct(ret[m], 0)}% ({m})" for m in MODELOS)
            + " da distância de B ao acaso (AUC − 50). ")
    ai = {m: g("queda_B_menos_C", m, "auc_media_intra_sujeito") for m in MODELOS}
    txt += (f"Dentro de cada pessoa (AUC média intra-sujeito), a queda é {cel(ai['RF'], 1)} (RF) e {cel(ai['LR'], 1)} pp "
            "(LR).")
    w(txt)
    w("")

    # (2) PP1 — por vídeo
    qv = {m: g("queda_B_menos_C", m, "auc", "video") for m in MODELOS}
    qacc = {m: g("queda_B_menos_C", m, "acuracia", "video") for m in MODELOS}
    qaccc = {m: g("queda_B_menos_C", m, "acuracia", "video", LIM_CAL) for m in MODELOS}
    txt = (f"**PP1 — por vídeo (a decisão do app: média dos escores das janelas).** AUC por vídeo: B "
           f"{cel(g('B_uta_intra', 'RF', 'auc', 'video'))} × C {cel(g('C_dmd_para_uta', 'RF', 'auc', 'video'))} no RF "
           f"(queda {cel(qv['RF'], 1)} pp) e B {cel(g('B_uta_intra', 'LR', 'auc', 'video'))} × C "
           f"{cel(g('C_dmd_para_uta', 'LR', 'auc', 'video'))} na LR (queda {cel(qv['LR'], 1)} pp). Acurácia por vídeo no "
           f"limiar 0,5: B {cel(g('B_uta_intra', 'RF', 'acuracia', 'video'))} × C "
           f"{cel(g('C_dmd_para_uta', 'RF', 'acuracia', 'video'))} (RF; queda {cel(qacc['RF'], 1)} pp) e B "
           f"{cel(g('B_uta_intra', 'LR', 'acuracia', 'video'))} × C {cel(g('C_dmd_para_uta', 'LR', 'acuracia', 'video'))} "
           f"(LR; queda {cel(qacc['LR'], 1)} pp); com os limiares calibrados no treino, a queda fica em "
           f"{cel(qaccc['RF'], 1)} (RF) e {cel(qaccc['LR'], 1)} pp (LR).")
    w(txt)
    w("")

    # (3) Limiar
    partes = []
    for m in MODELOS:
        tc = lt.limiar_c(P, m)
        to = lt.limiar_oraculo(P, m)
        s05, e05 = g("C_dmd_para_uta", m, "sensibilidade")["valor"], g("C_dmd_para_uta", m, "especificidade")["valor"]
        sc, ec = (g("C_dmd_para_uta", m, "sensibilidade", lim=LIM_CAL)["valor"],
                  g("C_dmd_para_uta", m, "especificidade", lim=LIM_CAL)["valor"])
        b05, bc = g("C_dmd_para_uta", m, "acuracia_balanceada"), g("C_dmd_para_uta", m, "acuracia_balanceada", lim=LIM_CAL)
        fr = (f"{m}: no limiar 0,5 o modelo do DMD acerta {pct(s05)}% das janelas sonolentas e {pct(e05)}% das alertas "
              f"(acur. bal. {cel(b05)}); o limiar calibrado no DMD ({br(tc['limiar'], 3)}) leva a {pct(sc)}% / {pct(ec)}% "
              f"(acur. bal. {cel(bc)}); o limiar que seria ótimo na própria UTA (diagnóstico) é {br(to['limiar'], 3)}, com "
              f"acur. bal. {pct(to['acur_bal_na_calibracao'])}")
        partes.append(fr)
    txt = ("**Limiar.** " + "; ".join(partes) + ". ")
    desloc = [lt.limiar_oraculo(P, m)["limiar"] - lt.limiar_c(P, m)["limiar"] for m in MODELOS]
    sens_baixa = [m for m in MODELOS if g("C_dmd_para_uta", m, "sensibilidade")["valor"]
                  < g("C_dmd_para_uta", m, "especificidade")["valor"]]
    if sens_baixa:
        fm = [lt.media(b, "sem_normalizacao", "dur_max_fechamento_s", "sonolento", "mediana") for b in ("dmd", "uta")]
        pm = [lt.media(b, "sem_normalizacao", "perclos", "sonolento", "mediana") for b in ("dmd", "uta")]
        txt += (f"O modelo do DMD erra sobretudo por chamar de alerta janelas sonolentas reais ({', '.join(sens_baixa)}: "
                "sensibilidade abaixo da especificidade): a sonolência encenada é mais extrema que a real — nas janelas "
                f"sonolentas, o maior fechamento mediano é {br(fm[0], 2)} s no DMD e {br(fm[1], 2)} s na UTA, e o PERCLOS "
                f"mediano {br(pm[0], 3)} × {br(pm[1], 3)} (características brutas; seção 10). ")
    txt += ("O limiar ótimo na UTA fica " + ("abaixo" if all(d < 0 for d in desloc) else ("acima" if all(d > 0 for d in desloc)
                                                                                        else "de lados diferentes"))
            + " do escolhido no DMD nos dois modelos: a escala dos escores muda com o domínio, e a calibração no DMD "
            "não a corrige por completo. A parte da queda que é de ordenação aparece na AUC (item acima); a que é de "
            "calibração, na diferença entre a acurácia balanceada no limiar do DMD e no oráculo.")
    w(txt)
    w("")

    # (4) Choque de realidade e A − B
    ks = {m: g("choque_A_menos_C", m, "auc") for m in MODELOS}
    ab = {m: g("lab_A_menos_B", m, "auc") for m in MODELOS}
    txt = (f"**Choque de realidade (A − C).** O número de laboratório (A: {cel(g('A_dmd_intra', 'RF', 'auc'))} RF, "
           f"{cel(g('A_dmd_intra', 'LR', 'auc'))} LR, LOSO no DMD) encolhe {cel(ks['RF'], 1)} (RF) e {cel(ks['LR'], 1)} pp "
           f"(LR) de AUC no dado real. A − B = {cel(ab['RF'], 1)} (RF) e {cel(ab['LR'], 1)} pp (LR): ")
    fr = {1: "A acima de B (a tarefa do DMD é mais fácil que a da UTA mesmo com treino real)",
          0: "A e B indistinguíveis (a tarefa encenada não é, para as mesmas características, claramente mais fácil que a real)",
          -1: "A abaixo de B"}
    txt += "; ".join(f"{m}: {fr[lado(ab[m], 0, 1)]}" for m in MODELOS) + ". "
    txt += ("Ou seja, o choque de realidade (A − C) mistura duas coisas: a diferença de tarefa (A − B) e a queda da "
            "transferência propriamente dita (B − C).")
    w(txt)
    w("")

    # (5) Normalização
    cr = {m: g("C_dmd_para_uta", m, "auc", var="sem_normalizacao") for m in MODELOS}
    br_ = {m: g("B_uta_intra", m, "auc", var="sem_normalizacao") for m in MODELOS}
    txt = (f"**Normalização por sujeito.** Sem centrar as características, C vai a AUC {cel(cr['RF'])} (RF) e "
           f"{cel(cr['LR'])} (LR), e B a {cel(br_['RF'])} e {cel(br_['LR'])}, contra {cel(g('C_dmd_para_uta', 'RF', 'auc'))} / "
           f"{cel(g('C_dmd_para_uta', 'LR', 'auc'))} (C) e {cel(g('B_uta_intra', 'RF', 'auc'))} / "
           f"{cel(g('B_uta_intra', 'LR', 'auc'))} (B) centradas. ")
    ganhos = []
    for m in MODELOS:
        gb = 100 * (g("B_uta_intra", m, "auc")["valor"] - br_[m]["valor"])
        gc = 100 * (g("C_dmd_para_uta", m, "auc")["valor"] - cr[m]["valor"])
        ganhos.append(f"{m}: centrar muda a AUC de B em {br(gb, 1)} pp e a de C em {br(gc, 1)} pp")
    txt += "; ".join(ganhos) + ". "
    pa = [lt.media(b, "sem_normalizacao", "perclos", c, "mediana") for b in ("dmd", "uta") for c in ("alerta", "sonolento")]
    txt += (f"Em escala bruta as bases ocupam regiões diferentes (PERCLOS mediano alerta → sonolento: DMD {br(pa[0], 3)} → "
            f"{br(pa[1], 3)}; UTA {br(pa[2], 3)} → {br(pa[3], 3)}).")
    w(txt)
    w("")
    return L + leitura_resto(lt)


def leitura_resto(lt):
    """Continuação da leitura: sensibilidades, sinais, controle, PP3, Exp. 1 e amostra."""
    R = lt.R
    L = []
    w = L.append
    P = "principal"

    # (6) Sensibilidades
    partes = []
    for m in MODELOS:
        cs = {v: R("C_dmd_para_uta", v, m, "B3", "janela", "auc") for v in VARIANTES}
        qs = {v: R("queda_B_menos_C", v, m, "B3", "janela", "auc") for v in VARIANTES}
        cs = {v: r for v, r in cs.items() if r is not None}
        vmin, vmax = min(cs, key=lambda v: cs[v]["valor"]), max(cs, key=lambda v: cs[v]["valor"])
        excl = [v for v, r in qs.items() if r is not None and lado(r, 0, 1) == 1]
        incl = [v for v, r in qs.items() if r is not None and lado(r, 0, 1) == 0]
        abaixo = [v for v, r in qs.items() if r is not None and lado(r, 0, 1) == -1]
        fr = (f"{m}: AUC de C entre {cel(cs[vmin], com_ic=False)} (`{vmin}`) e {cel(cs[vmax], com_ic=False)} (`{vmax}`); "
              f"queda B − C com IC acima de 0 em {len(excl)} de {len(qs)} variantes")
        if incl:
            fr += " (IC contém 0: " + ", ".join(f"`{v}`" for v in incl) + ")"
        if abaixo:
            fr += " (IC abaixo de 0: " + ", ".join(f"`{v}`" for v in abaixo) + ")"
        partes.append(fr)
    txt = "**Sensibilidades (seção 8).** " + "; ".join(partes) + ". "
    for m in MODELOS:
        b0 = R("B_uta_intra", P, m, "B3", "janela", "auc")
        cmax = max((R("C_dmd_para_uta", v, m, "B3", "janela", "auc") for v in VARIANTES
                    if R("C_dmd_para_uta", v, m, "B3", "janela", "auc") is not None), key=lambda r: r["valor"])
        txt += (f"{m}: a maior AUC de C entre as variantes ({cel(cmax, com_ic=False)}) fica "
                + ("abaixo" if cmax["valor"] < b0["valor"] else "no nível ou acima") + f" da de B principal ({cel(b0, com_ic=False)}). ")
    for m in MODELOS:
        bc, bb = R("B_uta_intra", P, m, "B3", "janela", "auc"), R("B_uta_intra", "sem_normalizacao", m, "B3", "janela", "auc")
        cc, cb = R("C_dmd_para_uta", P, m, "B3", "janela", "auc"), R("C_dmd_para_uta", "sem_normalizacao", m, "B3", "janela", "auc")
        q = R("queda_B_menos_C", "sem_normalizacao", m, "B3", "janela", "auc")
        if q is not None and lado(q, 0, 1) == 0 and (bc["valor"] - bb["valor"]) > (cc["valor"] - cb["valor"]) + 0.03:
            txt += (f"Sem normalização, a queda {ART[m][1]} {m} deixa de ser detectável ({cel(q, 1)} pp) porque a centragem "
                    f"ajuda muito mais B ({cel(bb, com_ic=False)} → {cel(bc, com_ic=False)}) do que C "
                    f"({cel(cb, com_ic=False)} → {cel(cc, com_ic=False)}), e não porque C melhore. ")
    s1 = {m: R("C_dmd_para_uta", "alerta_s1", m, "B3", "janela", "auc") for m in MODELOS}
    c0 = {m: R("C_dmd_para_uta", P, m, "B3", "janela", "auc") for m in MODELOS}
    if all(s1[m] is not None and s1[m]["valor"] < c0[m]["valor"] for m in MODELOS):
        txt += ("Trocar o alerta da s2 (carro parado) pelo da s1 (carro em movimento) PIORA a transferência (AUC de C "
                + "; ".join(f"{cel(c0[m], com_ic=False)} → {cel(s1[m], com_ic=False)} ({m})" for m in MODELOS)
                + "). Uma explicação plausível, não testada, é que com o carro em movimento o alerta difere da s5 também "
                "em movimento, postura e luz, e o modelo aprende essas diferenças; o resultado é coerente com a escolha "
                "(feita antes) da s2, de carro parado como a s5, para o principal. ")
    es = {m: R("C_dmd_para_uta", "ref_s2_espelho", m, "B3", "janela", "auc") for m in MODELOS}
    if all(es[m] is not None for m in MODELOS):
        dif = {m: 100 * (es[m]["valor"] - c0[m]["valor"]) for m in MODELOS}
        txt += ("Espelhar no DMD a estrutura de referência da UTA (`ref_s2_espelho`) muda a AUC de C em "
                + " e ".join(f"{br(dif[m], 1)} pp ({m})" for m in MODELOS)
                + ("; o sentido é compatível com parte da queda vir da assimetria de referência entre as bases, mas "
                   f"com {int(lt.fato('dmd', 'espelho_alerta'))} janelas alerta e referência de "
                   f"{int(lt.fato('dmd', 'espelho_ref_min'))} a {int(lt.fato('dmd', 'espelho_ref_max'))} janelas por "
                   "sujeito não é uma evidência forte. " if all(d > 0 for d in dif.values()) else ". "))
    t15 = {m: lt.limiar_c("janela_15s", m) for m in MODELOS}
    if all(t is not None for t in t15.values()):
        txt += (f"Com 15 s o DMD tem só {int(lt.fato('dmd', 'alerta_principal_15s'))} janelas alerta "
                f"(de {int(lt.fato('dmd', 'sujeitos_com_alerta_15s'))} sujeitos) e o limiar calibrado no DMD vai a "
                + " e ".join(f"{br(t15[m]['limiar'], 3)} ({m})" for m in MODELOS)
                + ": essa variante é pouco informativa.")
    w(txt)
    w("")

    # (7) Sinais
    grupos = {}
    for f in FEATURES:
        grupos.setdefault(classifica_sinal(lt, f), []).append(f)
    txt = "**Quais sinais transferem (seção 10).** " + "; ".join(
        f"{k}: {', '.join('`' + x + '`' for x in v)}" for k, v in grupos.items()) + ". "
    opostos = []
    for f in FEATURES:
        cb = pega(lt.imp, tipo="coef_lr_padronizado", experimento="B", variante=P, caracteristica=f)
        cc_ = pega(lt.imp, tipo="coef_lr_padronizado", experimento="C", variante=P, caracteristica=f)
        ic_ = lt.imp_perm("C", P, "LR", f)
        if (cb is not None and cc_ is not None and ic_ is not None and abs(cb["valor"]) >= 0.2 and abs(cc_["valor"]) >= 0.2
                and np.sign(cb["valor"]) != np.sign(cc_["valor"])):
            opostos.append((f, cb["valor"], cc_["valor"], ic_["valor"]))
    if opostos:
        txt += ("Coeficientes da LR com sinal oposto nas duas bases (treinada no DMD × treinada na UTA): " + "; ".join(
            f"`{f}` {br(vc, 2)} × {br(vb, 2)} (importância por permutação em C-LR {br(vi * 100)})" for f, vb, vc, vi in opostos)
            + ". Como PERCLOS, fechamentos longos e EAR médio são correlacionados, um coeficiente isolado não se "
            "interpreta; o que importa é a importância por permutação em C: quando ela é negativa, embaralhar a "
            "característica MELHORA a AUC na UTA, ou seja, o modelo do DMD a usa num sentido que não vale no dado real. ")
    db, ub = lt.univ("dmd", P, "frac_bocejo"), lt.univ("uta", P, "frac_bocejo")
    if sentido_univ(db) == 1 and sentido_univ(ub) == 0:
        txt += (f"O bocejo não transfere: discrimina no DMD (AUC univariada {cel(db)}) e quase não aparece na UTA "
                f"({cel(ub)}). ")
    tops = {m: lt.top("C", P, m, 2) for m in MODELOS}
    txt += ("Em C, as duas características mais importantes são " + "; ".join(
        f"{m}: " + ", ".join(f"`{f}` ({br(v * 100)})" for f, v in tops[m]) for m in MODELOS) + ".")
    w(txt)
    w("")

    # (8) Controle do vídeo de referência
    kb = {m: R("B_uta_intra", P, m, "K10v5", "janela", "auc") for m in MODELOS}
    kc = {m: R("C_dmd_para_uta", P, m, "K10v5", "janela", "auc") for m in MODELOS}
    txt = (f"**Vídeo de referência (seção 9).** Entre KSS 10 e KSS 5 — dois vídeos que não são a referência — a AUC é "
           f"{cel(kb['RF'])} (B-RF), {cel(kb['LR'])} (B-LR), {cel(kc['RF'])} (C-RF) e {cel(kc['LR'])} (C-LR). ")
    acima = [f"{e}-{m}" for e, d in (("B", kb), ("C", kc)) for m in MODELOS if lado(d[m], 50) == 1]
    if acima:
        txt += (f"Em {', '.join(acima)} o IC fica acima de 50: há sinal de sonolência além de \"vídeo diferente da "
                "referência\". ")
    k50 = {m: R("B_uta_intra", P, m, "K5v0", "janela", "auc") for m in MODELOS}
    maior = [m for m in MODELOS if k50[m] is not None and kb[m] is not None and k50[m]["valor"] > kb[m]["valor"]]
    if maior:
        txt += (f"Em B ({', '.join(maior)}), KSS 0 × 5 separa mais que KSS 5 × 10: parte do que o modelo aprende é "
                "\"diferente do vídeo 0\" (luz, enquadramento, hora da gravação) — confundimento que vale para B, para C e "
                "para os `.npy` oficiais, e que a UTA-RLDD não permite eliminar.")
    else:
        txt += "Em B, KSS 0 × 5 separa menos que KSS 5 × 10. "
    kc50 = {m: R("C_dmd_para_uta", P, m, "K5v0", "janela", "auc") for m in MODELOS}
    iguais = [m for m in MODELOS if kc50[m] is not None and kc[m] is not None and abs(kc50[m]["valor"] - kc[m]["valor"]) < 0.03]
    if iguais:
        txt += ((" " if not txt.endswith(" ") else "") + "Em C (" + ", ".join(iguais) + "), KSS 0 × 5 e KSS 5 × 10 separam no mesmo patamar ("
                + "; ".join(f"{m}: {cel(kc50[m], com_ic=False)} e {cel(kc[m], com_ic=False)}" for m in iguais)
                + "): o modelo treinado no DMD não tem acesso a esse atalho da UTA.")
    w(txt)
    w("")

    # (9) PP3
    ear = lt.ear
    sj = lt.sj[lt.sj["base"] == "dmd"].set_index("sujeito")
    oc = ear[ear["oculos"]]
    ruim = ear[ear["frac_fechado_abaixo_limiar"] < 0.2]
    txt = (f"**PP3 — óculos (seção 11).** {len(oc['sujeito'].unique())} dos {int(lt.fato('dmd', 'sujeitos_s5'))} motoristas "
           "do DMD usam óculos na s5. ")
    for _, r in ruim.iterrows():
        su = r["sujeito"]
        txt += (f"O detector de piscadas falha em {su} ({'com' if r['oculos'] else 'sem'} óculos): F1 de piscadas "
                f"{br(r['det_f1_piscadas'], 3)} e {pct(r['frac_fechado_abaixo_limiar'])}% dos quadros anotados como olho "
                f"fechado abaixo do limiar. O EAR, porém, ainda ordena fechado × aberto (AUC por quadro "
                f"{br(r['auc_menos_ear'] * 100)}); o problema é a amplitude: o EAR cai só {pct(r['queda_relativa'])}% ao "
                f"fechar ({br(r['ear_mediana_aberto'], 3)} → {br(r['ear_mediana_fechado'], 3)}), contra um limiar de "
                f"{br(r['limiar'], 3)} (0,75 × EAR de referência). ")
        if su in sj.index:
            txt += (f"Em A, {su} tem AUC {br(sj.loc[su, 'auc_A_RF'] * 100)} (RF) e {br(sj.loc[su, 'auc_A_LR'] * 100)} (LR), "
                    f"e o RF acerta {pct(sj.loc[su, 'sens_A_RF'])}% das janelas sonolentas dele. ")
    for m in MODELOS:
        est_c = R("A_dmd_intra", P, m, "DMD", "janela", "auc", estrato="oculos_s5")
        est_s = R("A_dmd_intra", P, m, "DMD", "janela", "auc", estrato="sem_oculos_s5")
        if est_c is not None and est_s is not None and m == "RF":
            outros_oc = [s_ for s_ in oc["sujeito"].unique() if s_ not in set(ruim["sujeito"]) and s_ in sj.index]
            txt += (f"O estrato com óculos em A (RF: {cel(est_c, com_ic=False)} × {cel(est_s, com_ic=False)} sem óculos) "
                    "inclui esse motorista; os outros motoristas de óculos têm AUC individual de "
                    + " e ".join(f"{br(sj.loc[s_, 'auc_A_RF'] * 100)} ({s_})" for s_ in outros_oc) + " no RF. ")
    outros = oc[~oc["sujeito"].isin(ruim["sujeito"])]
    if len(outros):
        txt += ("Os outros motoristas de óculos (" + "; ".join(
            f"{r['sujeito']}: queda {pct(r['queda_relativa'])}%, F1 {br(r['det_f1_piscadas'], 3)}" for _, r in outros.iterrows())
            + ") têm detector normal. ")
    qsem = ear[~ear["oculos"]]
    txt += (f"Sem óculos, a queda relativa do EAR vai de {pct(qsem['queda_relativa'].min())}% a "
            f"{pct(qsem['queda_relativa'].max())}% e o F1 de piscadas de {br(qsem['det_f1_piscadas'].min(), 3)} a "
            f"{br(qsem['det_f1_piscadas'].max(), 3)}. Com {oc['sujeito'].nunique()} motoristas de óculos, dos quais "
            f"{'só um falha' if ruim['sujeito'].nunique() == 1 else str(ruim['sujeito'].nunique()) + ' falham'}, **não dá para "
            "atribuir a falha aos óculos**: ela pode vir da armação, de reflexo na lente, do formato do olho ou de um "
            "fechamento encenado incompleto — nada disso foi medido. O que se pode afirmar é mais estreito: um limiar de EAR "
            "relativo ao olho aberto falha quando a queda do EAR ao fechar é pequena, e isso aconteceu num motorista de "
            "óculos. ")
    q = {m: (R("C_dmd_para_uta", P, m, "B3", "janela", "auc"), R("C_dmd_para_uta", "sem_gE_29", m, "B3", "janela", "auc"))
         for m in MODELOS}
    if all(v[1] is not None for v in q.values()):
        txt += ("Tirar gE_29 do treino muda a AUC de C de " + "; ".join(
            f"{cel(a, com_ic=False)} para {cel(b, com_ic=False)} ({m})" for m, (a, b) in q.items()) + ". ")
    txt += "Na UTA não há rótulo de óculos: a PP3 sobre óculos em sonolência real fica sem resposta neste trabalho."
    w(txt)
    w("")

    # (10) Exp. 1
    e = lt.exp1
    partes, difs = [], {}
    for m in MODELOS:
        e1 = e[(e.fonte == "exp1") & (e.modelo == m)]
        b5 = e[(e.fonte == "exp5b") & (e.variante == P) & (e.conjunto == "oficiais") & (e.modelo == m)]
        br5 = e[(e.fonte == "exp5b") & (e.variante == "uta_ref_no_teste") & (e.conjunto == "oficiais") & (e.modelo == m)]
        med = lambda q, met: q[q.metrica == met]["valor"].mean()  # noqa: E731
        difs[m] = 100 * (med(b5, "acuracia_voto") - med(e1, "va"))
        partes.append(f"{m}: AUC média por fold {br(med(e1, 'auc_macro') * 100)} no Exp. 1 × {br(med(b5, 'auc') * 100)} "
                      f"aqui ({br(med(br5, 'auc') * 100)} com a referência no teste); VA por voto "
                      f"{br(med(e1, 'va') * 100)} × {br(med(b5, 'acuracia_voto') * 100)} "
                      f"({br(med(br5, 'acuracia_voto') * 100)})")
    w("**Relação com o Exp. 1 (seção 13).** " + "; ".join(partes) + ". A AUC do Exp. 1 é por sequência de 30 "
      "piscadas (que cobre minutos de vídeo) e a daqui é por janela de 10 s — unidades diferentes, então a AUC não se "
      "compara diretamente; a VA por voto é a mesma métrica nos dois (decisão por vídeo). " + " ".join(
          f"{m}: VA de B {'no mesmo patamar do' if abs(d) < 3 else ('acima do' if d > 0 else 'abaixo do')} Exp. 1 "
          f"({br(d, 1)} pp; médias de 5 folds, sem IC)." for m, d in difs.items())
      + " A queda da PP1 deve ser lida contra B (mesmo extrator, mesmas janelas), não contra o Exp. 1 (recomendação "
      "do Exp. 5a).")
    w("")

    # (11) Amostra
    larg = max(R("queda_B_menos_C", P, m, "B3", "janela", "auc")["ic95_sup"] - R("queda_B_menos_C", P, m, "B3", "janela", "auc")["ic95_inf"]
               for m in MODELOS)
    a_meta = R("A_dmd_intra", P, "RF", "DMD", "janela", "auc")
    txt = (f"**Amostra.** UTA com {int(lt.fato('uta', 'sujeitos'))} sujeitos (5 folds); DMD com {int(a_meta['n_neg'])} "
           f"janelas alerta de {int(a_meta['n_sujeitos_duas_classes'])} sujeitos. O IC da queda B − C em AUC por janela tem "
           f"até {br(larg, 1)} pp de largura. ")
    if np.isfinite(lt.fato("preliminar", "A_auc_RF")):
        txt += (f"Na rodada preliminar A era {br(lt.fato('preliminar', 'A_auc_RF') * 100)} (RF) e "
                f"{br(lt.fato('preliminar', 'A_auc_LR') * 100)} (LR) com janelas alerta sobrepostas à referência; agora, com a "
                f"referência separada e alerta de outra sessão, {cel(a_meta, com_ic=False)} e "
                f"{cel(R('A_dmd_intra', P, 'LR', 'DMD', 'janela', 'auc'), com_ic=False)}: o número de laboratório da rodada "
                "preliminar estava inflado, e os números preliminares de A, B e C não devem ser citados.")
    w(txt)
    w("")
    return L


def cabecalho(lt):
    execucao = str(lt.tab["execucao"].iloc[0])
    L = ["# Exp. 5b (final) — Transferência de sonolência atuada (DMD) para real (UTA-RLDD): resposta direta da PP1", ""]
    w = L.append
    w(f"Execução {execucao}. Seed 42. Gerado por `src/experiments/exp5_transfer.py` a partir de "
      "`results/exp5b_transferencia.csv` (métricas, quedas e diagnósticos), `results/exp5b_limiares.csv`, "
      "`results/exp5b_importancia.csv`, `results/exp5b_por_sujeito.csv`, `results/exp5b_comparacao_exp1.csv`, "
      "`results/exp5b_predicoes.csv`, `results/exp5b_predicoes_video.csv` e `results/exp5b_amostra.csv`; log em "
      "`results/logs/exp5b_transferencia.log`; modelos de transferência em `models/exp5/exp5b_final_C_*.joblib` "
      "(metadados em `models/exp5/exp5b_final_modelos.json`); figura `results/figures/fig_exp5b_transferencia.{pdf,png}`. "
      "Nenhum número deste arquivo foi digitado à mão. A rodada preliminar (janelas de 30 s, só o Fold 1, DMD só com "
      "a s5) foi descartada; seus arquivos foram removidos em 06/10/2026.")
    w("")
    w(f"**Dados.** UTA-RLDD completa: {int(lt.fato('uta', 'sujeitos'))} sujeitos, {int(lt.fato('uta', 'videos'))} vídeos "
      f"lógicos ({int(lt.fato('uta', 'arquivos'))} arquivos), 5 folds oficiais. DMD: {int(lt.fato('dmd', 'sujeitos_s5'))} "
      "sujeitos com sessão de sonolência (s5) e as sessões de distração s1/s2/s3 dos mesmos sujeitos. Janelas de 10 s "
      "com passo de 5 s de `data/features/final/` (`results/exp5_caracteristicas.md`).")
    w("")
    w("Valores em %, com IC 95% entre colchetes: bootstrap percentil sobre **sujeitos** (1000 reamostragens, seed 42) das "
      "predições de teste — os modelos não são retreinados na reamostragem. AUC × 100. Acaso = 50 em acurácia "
      "balanceada e AUC. Quedas em pontos percentuais (pp; para AUC, pontos de AUC × 100); valor positivo = o segundo "
      "termo é pior.")
    w("")
    return L


def limitacoes(lt):
    L = ["## 15. Limitações (valem para qualquer número acima)", ""]
    w = L.append
    det = lt.sj[lt.sj["base"] == "dmd"]
    f1m = det["det_f1_piscadas"].mean()
    w("1. **Rótulos de natureza diferente.** No DMD o rótulo sonolento vem de EVENTOS anotados (fechamento ≥ 0,5 s, "
      "bocejo) que o ator encenou, e o alerta, de trechos de direção segura de OUTRA sessão (s2); na UTA o rótulo é o "
      "ESTADO autorrelatado (KSS) do vídeo inteiro (duração mediana de "
      f"{br(lt.fato('uta', 'duracao_mediana_min_video'), 1)} min). O modelo do DMD aprende \"há um evento encenado nesta "
      "janela (e ela não é da s2)\"; a UTA pergunta \"esta pessoa está sonolenta\". Parte do que se mede em C é essa "
      "diferença de pergunta, não só de domínio.")
    w(f"2. **Condições de gravação opostas.** DMD: carro PARADO (s2 e s5), câmera fixa, encenação, sessão de sonolência "
      f"com duração mediana de {br(lt.fato('dmd', 'duracao_mediana_min_s5'), 1)} min; UTA: gravação caseira com "
      "celular/webcam (fps e resoluções variados), sonolência real, vídeos com duração mediana de "
      f"{br(lt.fato('uta', 'duracao_mediana_min_video'), 1)} min. C mistura troca de câmera, de pessoa, de cenário e de "
      "fenômeno; nenhuma análise aqui separa esses fatores.")
    w(f"3. **Poucos sujeitos atuados.** {int(lt.fato('dmd', 'sujeitos_s5'))} motoristas no DMD "
      f"({int(lt.fato('dmd', 'sujeitos_oculos_s5'))} com óculos), {int(lt.fato('dmd', 'alerta_principal_10s'))} janelas "
      "alerta; o modelo de C aprende \"alerta\" e \"sonolento\" com pouca variedade de pessoas, e o IC de A é largo.")
    w(f"4. **Alerta do DMD de outro dia.** A s2 (alerta) foi gravada em outro dia que a s5 em "
      f"{int(lt.fato('dmd', 'sujeitos_s5') - lt.fato('dmd', 'sujeitos_s2_mesmo_dia'))} dos "
      f"{int(lt.fato('dmd', 'sujeitos_s5'))} sujeitos, e a referência (s3) no mesmo dia da s5 em todos: diferenças de "
      "montagem, luz e óculos entre dias entram na classe alerta. É o espelho da UTA, onde a referência sai do mesmo "
      "vídeo das janelas alerta; a sensibilidade `ref_s2_espelho` e o estrato por dia (seções 8 e 11) medem o efeito, "
      "mas não o eliminam.")
    w("5. **Classe confundida com o vídeo na UTA**: cada estado é um vídeo gravado em outro momento, e as janelas alerta "
      "dividem o vídeo com a referência da normalização (controle na seção 9).")
    w(f"6. **Detector de piscadas imperfeito.** F1 médio de piscadas contra a anotação da s5: {br(f1m, 3)} por "
      "motorista, com falha quase total em gE_29 (seção 11); na UTA, "
      f"{int(lt.fato('uta', 'videos_sem_piscada'))} vídeos ficaram sem nenhuma piscada detectada (seção 3). As "
      "características herdam esses erros.")
    w(f"7. **Janela curta.** Em 10 s a taxa de piscadas é ruidosa e {pct(lt.fato('uta', 'frac_sem_piscada'))}% das janelas "
      "avaliáveis da UTA não têm piscada (duração, amplitude e velocidade imputadas pela mediana do treino).")
    w("8. **Bocejo quase ausente na UTA** com o limiar de MAR de 0,5 sustentado por 1 s: um sinal que define parte dos "
      "eventos do DMD praticamente não existe no dado real.")
    w("9. **IC sem a variância do treino.** O bootstrap reamostra as predições de teste (não retreina); com "
      f"{int(lt.fato('dmd', 'sujeitos_s5'))} sujeitos no DMD, a incerteza de C devida ao treino não aparece nos IC.")
    w("10. **Muitas comparações.** A variante principal, os modelos, os hiperparâmetros e o limiar 0,5 foram fixados "
      "antes; as sensibilidades são descritivas e não há correção para multiplicidade: um IC isolado que exclui 0 numa "
      "sensibilidade não é evidência forte.")
    w("")
    return L


def reproducao():
    return ["## 16. Como reproduzir", "",
            "```bash",
            "export PATH=\"$HOME/.local/bin:$PATH\"",
            "source .venv/bin/activate                              # na raiz do repositório",
            "bash src/features/run_features.sh                      # só se sinais/janelas mudarem (etapa de características)",
            "bash src/experiments/run_exp5b.sh                      # análises + CSV + relatório + figura (~20 min, log em results/logs/)",
            "python src/experiments/exp5_transfer.py --so-relatorio  # refaz .md e figura a partir dos CSV",
            "python src/experiments/exp5_transfer.py --rapido /tmp/x --sem-relatorio  # teste de fumaça (saídas fora de results/)",
            "```", ""]


def escrever_relatorio(log):
    lt = Leitor()
    L = (cabecalho(lt) + secao_resposta(lt) + secao_desenho(lt) + secao_amostra(lt) + secao_janela(lt) + secao_video(lt)
         + secao_limiar(lt) + secao_quedas(lt) + secao_sensibilidades(lt) + secao_controle(lt) + secao_sinais(lt)
         + secao_oculos(lt) + secao_por_sujeito(lt) + secao_exp1(lt) + leitura(lt) + limitacoes(lt) + reproducao()
         + apendice_sujeitos(lt))
    tmp = MD.with_suffix(".md.tmp")
    tmp.write_text("\n".join(L) + "\n", encoding="utf-8")
    os.replace(tmp, MD)
    log.info("-> %s", rel(MD))


# ======================================================================================
# Figura (lê só os CSV)
# ======================================================================================
def figura(log):
    """Ponto + IC 95% (a linha de base natural da AUC e da acurácia balanceada é o acaso, 50, não 0: barras a partir
    de 0 exagerariam as diferenças). Cor = modelo (azul/laranja da paleta validada pela skill dataviz, a mesma das
    outras figuras); preenchido = limiar 0,5, vazado = limiar calibrado só com o treino."""
    sys.path.insert(0, str(ROOT / "src" / "experiments"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import make_figures as MF

    MF.setup_style()
    lt = Leitor()
    R = lt.R
    cores = {"RF": MF.C1, "LR": MF.C2}
    grupos = [("A_dmd_intra", "DMD", "A\nDMD→\nDMD"), ("B_uta_intra", "B3", "B\nUTA→\nUTA"),
              ("C_dmd_para_uta", "B3", "C\nDMD→\nUTA")]
    paineis = [(0, 0, "auc", "janela", "AUC por janela", (LIM_FIXO,)),
               (0, 1, "auc", "video", "AUC por vídeo", (LIM_FIXO,)),
               (1, 0, "acuracia_balanceada", "janela", "Acur. balanceada por janela", (LIM_FIXO, LIM_CAL)),
               (1, 1, "acuracia", "video", "Acurácia por vídeo", (LIM_FIXO, LIM_CAL))]
    vals = []
    for _, _, met, nivel, _, lims in paineis:
        for exp, esq, _ in grupos:
            for mod in MODELOS:
                for lim in lims:
                    r = R(exp, "principal", mod, esq if exp != "A_dmd_intra" else "DMD", nivel if exp != "A_dmd_intra" else "janela", met, lim=lim)
                    if r is not None and (exp != "A_dmd_intra" or nivel == "janela"):
                        vals.append(r["ic95_inf"] * 100)
    ymin = max(0.0, min(40.0, np.floor(min(vals) / 10.0) * 10.0))
    fig, axes = plt.subplots(2, 2, figsize=(MF.COL_W_IN, MF.COL_W_IN * 1.08), sharey=True)
    for i, j, met, nivel, titulo, lims in paineis:
        ax = axes[i, j]
        n_series = len(MODELOS) * len(lims)
        passo = 0.17 if n_series == 4 else 0.24
        for k, (exp, esq, _) in enumerate(grupos):
            if exp == "A_dmd_intra" and nivel == "video":
                ax.text(k, (ymin + 100) / 2.0 + 5, "não se\naplica", ha="center", va="center", fontsize=MF.FS_S - 1.5,
                        color=MF.MUTED, linespacing=1.1)
                continue
            pos = 0
            for mod in MODELOS:
                for lim in lims:
                    x = k + (pos - (n_series - 1) / 2) * passo
                    pos += 1
                    r = R(exp, "principal", mod, esq, "janela" if exp == "A_dmd_intra" else nivel, met, lim=lim)
                    if r is None:
                        continue
                    v, lo, hi = 100 * r["valor"], 100 * r["ic95_inf"], 100 * r["ic95_sup"]
                    ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], fmt="none", zorder=3, **MF.errbar_kw(cores[mod]))
                    cheio = lim == LIM_FIXO
                    ax.plot(x, v, "o", ms=4.2, mfc=cores[mod] if cheio else MF.SURFACE, mec=cores[mod] if not cheio else MF.SURFACE,
                            mew=1.0 if not cheio else 0.7, zorder=4)
                    if i == 0 and j == 0 and exp == "A_dmd_intra":   # rótulo direto de identidade (texto em tinta)
                        esq_ = mod == MODELOS[0]
                        ax.text(x + (-0.075 if esq_ else 0.075), v, mod, ha="right" if esq_ else "left", va="center",
                                fontsize=MF.FS_S - 2, color=MF.INK2)
        ax.axhline(50, color=MF.MUTED, lw=0.6, zorder=1)
        ax.set_xticks(range(len(grupos)))
        ax.set_xticklabels([g[2] for g in grupos] if i == 1 else [""] * len(grupos), fontsize=MF.FS_S - 2, linespacing=0.95)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(ymin, 100)
        ax.set_yticks([t for t in (30, 40, 50, 60, 70, 80, 90, 100) if t >= ymin])
        ax.yaxis.set_major_formatter(MF.pct_formatter(0))
        ax.set_xlim(-0.55, len(grupos) - 0.45)
        MF.light_grid(ax)
        ax.set_title(titulo, fontsize=MF.FS_S - 0.5, color=MF.INK, pad=2.5)
        if j == 0:
            ax.set_ylabel("%" if i == 1 else "AUC × 100", fontsize=MF.FS_S - 0.5)
    axes[0, 0].text(-0.5, 50.8, "acaso", fontsize=MF.FS_S - 2, color=MF.MUTED, va="bottom", ha="left")
    handles = [Line2D([0], [0], marker="o", ls="none", ms=4.2, mfc=cores[m], mec=MF.SURFACE, mew=0.7, label=NOME_MODELO[m])
               for m in MODELOS]
    handles += [Line2D([0], [0], marker="o", ls="none", ms=4.2, mfc=MF.INK2, mec=MF.SURFACE, mew=0.7, label="limiar 0,5"),
                Line2D([0], [0], marker="o", ls="none", ms=4.2, mfc=MF.SURFACE, mec=MF.INK2, mew=1.0,
                       label="limiar calibrado no treino")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), fontsize=MF.FS_S - 1.5,
               handlelength=1.0, columnspacing=0.8, handletextpad=0.3)
    fig.text(0.01, 0.004, f"UTA-RLDD: {int(lt.fato('uta', 'sujeitos'))} sujeitos, 5 folds oficiais, KSS 0 × 10; DMD: "
             f"{int(lt.fato('dmd', 'sujeitos_s5'))} sujeitos (alerta = s2, sonolento = s5);\n"
             "janelas de 10 s, características centradas por sujeito; IC 95% por bootstrap de sujeitos (1000).",
             fontsize=MF.FS_S - 2.5, color=MF.INK2, linespacing=1.15)
    fig.tight_layout(rect=(0, 0.05, 1, 0.905), h_pad=0.6, w_pad=0.4)
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{FIG}.pdf")
    fig.savefig(FIGS / f"{FIG}.png", dpi=300)
    wi, he = fig.get_size_inches()
    plt.close(fig)
    log.info("-> %s/%s.{pdf,png} (%.1f x %.1f cm)", rel(FIGS), FIG, wi * 2.54, he * 2.54)


# ======================================================================================
# Entrada
# ======================================================================================
def modo_rapido(destino: Path) -> None:
    """Teste de fumaça: poucas reamostragens e variantes, TODAS as saídas em `destino` (nunca em results/)."""
    g = globals()
    destino.mkdir(parents=True, exist_ok=True)
    g.update(N_BOOT=40, BLOCO_BOOT=15, N_REP_PERM=2, RES_SAIDA=destino,
             CSV_RES=destino / CSV_RES.name, CSV_LIM=destino / CSV_LIM.name, CSV_IMP=destino / CSV_IMP.name,
             CSV_SUJ=destino / CSV_SUJ.name, CSV_EXP1=destino / CSV_EXP1.name, CSV_PRED=destino / CSV_PRED.name,
             CSV_PRED_VID=destino / CSV_PRED_VID.name, CSV_AMOSTRA=destino / CSV_AMOSTRA.name, MD=destino / MD.name,
             CSV_EAR=destino / CSV_EAR.name,
             LOG=destino / LOG.name, LOGS=destino, FIGS=destino, MODELS=destino / "models",
             JSON_MODELOS=destino / "models" / JSON_MODELOS.name)
    for k in [k for k in VARIANTES if k not in ("principal", "sem_normalizacao", "ref_s2_espelho", "janela_15s")]:
        VARIANTES.pop(k)
    for k in [k for k in B_VARIANTES if k not in ("principal", "sem_normalizacao", "uta_ref_no_teste", "janela_15s")]:
        B_VARIANTES.pop(k)


def main():
    ap = argparse.ArgumentParser(description="Exp. 5b (final) — transferência DMD (atuado) -> UTA-RLDD (real).")
    ap.add_argument("--so-relatorio", action="store_true", help="não recalcula; refaz .md e figura a partir dos CSV")
    ap.add_argument("--sem-relatorio", action="store_true", help="só as análises e os CSV")
    ap.add_argument("--sem-figura", action="store_true")
    ap.add_argument("--rapido", type=Path, default=None, metavar="PASTA",
                    help="teste de fumaça (poucas reamostragens/variantes) com TODAS as saídas em PASTA")
    args = ap.parse_args()
    if args.rapido is not None:
        modo_rapido(args.rapido.resolve())
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    if args.so_relatorio:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
        log = logging.getLogger("exp5b")
    else:
        log = configurar_log()
        executar(log)
    if not args.sem_relatorio:
        ear_olhos_dmd(log)
        escrever_relatorio(log)
        if not args.sem_figura:
            figura(log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
