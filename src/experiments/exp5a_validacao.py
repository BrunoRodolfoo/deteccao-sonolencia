#!/usr/bin/env python3
"""Exp. 5a (rodada final) — validação do NOSSO extrator contra as características OFICIAIS da UTA-RLDD, nos 5 folds.

Nosso extrator = MediaPipe Face Mesh (src/features/extract_signals.py) + detector de piscadas por limiar adaptativo
(src/features/blinks.py) + sequências de 30 piscadas no formato de Ghoddoosian et al. (src/features/sequences.py).
Oficial = dlib + SVM de olho fechado (blink_video.py dos autores): data/uta_rldd_blinks/*_30_Fold<k>.npy.

Rodar no .venv, DEPOIS de `bash src/features/run_features.sh` (ou tudo de uma vez: `bash src/experiments/run_exp5a.sh`):
    python src/experiments/exp5a_validacao.py                    # 1000 reamostragens, 10.000 permutações, 10 sementes
    python src/experiments/exp5a_validacao.py --bootstrap 2000 --permutacoes 20000 --sementes 20

Entradas (rodada final): data/features/final/uta/sequences.npz (nossas sequências dos 60 sujeitos, 178 vídeos lógicos com
sequência; terço de referência = último), results/exp5_uta_mapeamento_oficial.csv (mapeamento sujeito oficial <-> pasta
nos 5 folds: 47 confirmados, 7 prováveis, 4 indeterminados), results/exp5_uta_validacao_oficial_videos.csv (NF/T finais
das séries oficiais), data/features/final/uta/blinks_index.csv (uma linha por ARQUIVO: os vídeos divididos F3_32_10 e
F5_49_10 têm duas partes, agregadas aqui pelo vídeo lógico), data/signals/uta/index.csv, results/exp5a_dlib_rosto.csv
(src/experiments/exp5a_dlib_rosto.py, .venv-mp) e results/exp5a_mapeamento_indep.csv (exp5a_mapeamento_indep.py).

Conjuntos de sujeitos: PRINCIPAL = pares `confirmado` (47 sujeitos oficiais); sensibilidade = `confirmado` + `provavel`
(54). Os 4 indeterminados (Fold 3) não têm par: entram só na linha (i) com os 58 sujeitos oficiais (conferência com o
Exp. 1). Unidade de inferência: o sujeito. As predições de cada sujeito vêm sempre do modelo do fold dele; as métricas
agregam todos os sujeitos do conjunto (matriz de confusão somada; VA = vídeos certos / vídeos).

Partes:
 1. Mapeamento: checagens que não usam o escore do Exp. 5 — (a) húngaro sobre o nº de piscadas (mostra que a contagem
    não identifica o sujeito); (b) duração: NF da última piscada oficial <= nº de quadros do vídeo; (c) resumo da
    verificação independente (outro detector de vales); (d) diagnóstico das pastas sem par (F1_12 e F5_49).
 2. Custo de trocar o extrator (PRINCIPAL): RandomForest do Exp. 1 (300 árvores, estatísticas 24d de
    sequences_to_features) treinado nas oficiais dos outros 4 folds (Blinks_30_Fold<k>.npy) e testado (i) nas oficiais e
    (ii) nas nossas do fold k, para k = 1..5 (protocolo oficial). Esquemas 3c (como o Exp. 1) e B3 (alerta x sonolento,
    o do Exp. 5b). Incerteza: bootstrap pareado sobre sujeitos (IC 95% por percentis), teste de permutação pareado
    (troca aleatória dos lados (i)/(ii) dentro de cada sujeito, bicaudal), McNemar exato por vídeo (VA) e 10 sementes.
 3. Mesmo extrator nas duas pontas (sem troca entre treino e teste, situação do Exp. 5b), nos mesmos sujeitos:
    (a) 5 folds: treino nos sujeitos do conjunto dos outros folds, teste no fold; (b) LOSO nos 47 confirmados.
 4. Distribuições das 4 características pós z-score (piscadas; cada uma uma vez): KS de 2 amostras, D mediano por
    sujeito, Spearman das médias por vídeo; mecanismo da Frequency (Wilcoxon por sujeito).

Variante principal das nossas características: "estendida" (bordas estendidas até o olho aberto, em quadros), fixada
ANTES deste experimento pela etapa de características; "limiar" (chave X) e "segundos" são sensibilidade.

Saídas: results/exp5a_validacao.csv (métricas), exp5a_validacao_custo.csv (comparações pareadas),
exp5a_validacao_distribuicoes.csv, exp5a_validacao_mapeamento.csv, exp5a_validacao_mecanismo_frequencia.csv,
exp5a_validacao_videos.csv, exp5a_validacao_confusoes.csv, exp5a_videos_diagnostico.csv (incremental),
exp5a_validacao.md, results/figures/fig_exp5a_validacao.{pdf,png} (custo com IC) e fig_exp5a_distribuicoes.{pdf,png},
results/logs/exp5a_validacao_<data-hora>.log. A rodada preliminar (só o Fold 1) foi descartada
(arquivos removidos em 06/10/2026).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest, ks_2samp, spearmanr, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "experiments"))
sys.path.insert(0, str(ROOT / "src" / "features"))

import uta_common as U  # noqa: E402
import blinks as B  # noqa: E402
import uta_oficial as O  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

RES = ROOT / "results"
FIG = RES / "figures"
LOGS = RES / "logs"
FEAT_UTA = B.FEAT / "uta"                                  # data/features/final/uta (rodada final)
SEQ_NPZ = FEAT_UTA / "sequences.npz"
BIDX_CSV = FEAT_UTA / "blinks_index.csv"                   # uma linha por arquivo (partes _p1/_p2 separadas)
SINAIS_IDX = ROOT / "data" / "signals" / "uta" / "index.csv"
DIR_OF = ROOT / "data" / "uta_rldd_blinks"
MAPA_CSV = RES / "exp5_uta_mapeamento_oficial.csv"
MAPA_SENS_CSV = RES / "exp5_uta_mapeamento_sensibilidade.csv"
VAL_VIDEOS_CSV = RES / "exp5_uta_validacao_oficial_videos.csv"
SEQ_RESUMO_CSV = RES / "exp5_sequencias_resumo.csv"
EXP1_CSV = RES / "exp1_intra.csv"
DIAG_CSV = RES / "exp5a_videos_diagnostico.csv"
DLIB_CSV = RES / "exp5a_dlib_rosto.csv"
INDEP_CSV = RES / "exp5a_mapeamento_indep.csv"
PRELIM_CUSTO_CSV = RES / "preliminar" / "exp5a_validacao_custo.csv"
OUT_RES, OUT_FIG, OUT_LOGS = RES, FIG, LOGS    # saídas (--saida DIR redireciona tudo para DIR, para testes)

SEED = U.SEED            # 42
N_ARVORES = 300          # igual ao Exp. 1 (src/experiments/blink_models.py)
N_JOBS = 6               # regra do projeto (<= 6); o RF do scikit-learn dá o mesmo resultado com qualquer n_jobs
VARIANTES = {"limiar": "X", "segundos": "X_segundos", "estendida": "X_estendida"}
VARIANTE_PRINCIPAL = "estendida"
ORDEM_VAR = ["estendida", "limiar", "segundos"]
DESC_VARIANTE = {
    "limiar": "bordas no limiar de EAR; unidades em quadros; chave X",
    "segundos": "bordas no limiar; duração em s, frequência em Hz, velocidade em EAR/s",
    "estendida": "bordas estendidas até o olho aberto; unidades em quadros",
}
CONJUNTOS = {"confirmado": ("confirmado",), "confirmado_provavel": ("confirmado", "provavel")}
CONJ_P = "confirmado"
CONJ_NOME = {"confirmado": "confirmados", "confirmado_provavel": "confirmados + prováveis"}
CARACT = ["frequencia", "amplitude", "duracao", "velocidade"]
CARACT_NOME = {"frequencia": "Frequência", "amplitude": "Amplitude", "duracao": "Duração",
               "velocidade": "Velocidade de abertura"}
CLASSE_NOME = {0: "alerta", 1: "pouca vigilância", 2: "sonolento"}
KSS = {0: 0, 1: 5, 2: 10}
ESQUEMAS = ["3c", "B3"]
METRICAS = ["bsa", "bal_acc", "f1_macro", "auc_macro", "va"]
MET_NOME = {"bsa": "BSA", "bal_acc": "Acur. bal.", "f1_macro": "F1 macro", "auc_macro": "AUC macro", "va": "VA"}
MET_BOOT = ["bsa", "bal_acc", "f1_macro", "va"]   # métricas recalculáveis das confusões por sujeito (bootstrap/permutação)
SEC_TO = "treino_oficial_outros_folds"            # parte 2 (principal)
SEC_5F = "mesmo_extrator_5_folds"                 # parte 3a
SEC_LOSO = "mesmo_extrator_loso"                  # parte 3b
SEC_NOME = {SEC_TO: "RF treinado nas oficiais dos outros folds", SEC_5F: "5 folds, mesmo extrator",
            SEC_LOSO: "LOSO, mesmo extrator"}
TOL_NF = 5e-4            # folga relativa da checagem NF_final <= nº de quadros do vídeo
PENAL_NF = 10.0          # custo de cada violação na atribuição pela duração
TOL_FAIXA = 1e-3         # pastas ausentes: "fora da faixa" = mais de 0,1% além do mínimo/máximo dos arquivos processados
CLASSE_TXT = {"confirmado": "confirmado", "provavel": "provável", "indeterminado": "indeterminado"}


# ======================================================================================
# Utilidades
# ======================================================================================
class Log:
    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.f.write(str(msg) + "\n")
        self.f.flush()


def fin(x) -> bool:
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def br(x, nd=1) -> str:
    """Número com vírgula decimal e sinal de menos tipográfico (pt-BR)."""
    if not fin(x):
        return "—"
    s = f"{float(x):.{nd}f}"
    if s.startswith("-") and float(s) == 0:
        s = s[1:]
    return s.replace("-", "−").replace(".", ",")


def pct(x, nd=1) -> str:
    return "—" if not fin(x) else f"{br(100 * float(x), nd)}%"


def pp(x, nd=1) -> str:
    if not fin(x):
        return "—"
    v = 100 * float(x)
    s = br(v, nd)
    return (s if s.startswith("−") or float(f"{v:.{nd}f}") == 0 else "+" + s) + " pp"


def milhar(n) -> str:
    return f"{int(round(float(n))):,}".replace(",", ".")


def pval(p) -> str:
    if not fin(p):
        return "—"
    return "< 0,001" if p < 0.001 else br(p, 3)


def peq(p) -> str:
    """'p = 0,123' ou 'p < 0,001' (para o meio do texto)."""
    if not fin(p):
        return "p = —"
    return "p < 0,001" if p < 0.001 else f"p = {br(p, 3)}"


def lista_pt(itens) -> str:
    itens = [str(i) for i in itens]
    if not itens:
        return "—"
    return itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + " e " + itens[-1]


def kss_de_video(v: str) -> int:
    return int(str(v).rsplit("_", 1)[1])


def classe_de_video(v: str) -> int:
    return {0: 0, 5: 1, 10: 2}[kss_de_video(v)]


def sujeito_de_video(v: str) -> str:
    return str(v).rsplit("_", 1)[0]


def gravar_csv(df, path, log):
    tmp = path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    log(f"  -> {_rel(path)} ({len(df)} linhas)")


def _rel(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


# ======================================================================================
# Dados
# ======================================================================================
def carregar_nossas():
    if not SEQ_NPZ.exists():
        raise SystemExit(f"{SEQ_NPZ} não existe: rode antes `bash src/features/run_features.sh`.")
    z = np.load(SEQ_NPZ, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    terco = str(out.get("terco", np.array("?")))
    return out, terco


def carregar_mapa():
    mapa = pd.read_csv(MAPA_CSV)
    mapa["pasta"] = mapa["pasta"].fillna("").astype(str)
    return mapa


def montar_oficial(mapa, log):
    """Sequências oficiais de TESTE dos 5 folds (ids de vídeo/sujeito de U.load_fold_with_ids), com a pasta adotada
    (ou um rótulo próprio, F<k>_oficial<j>, para os indeterminados) e a classe do par. Também o treino oficial de cada
    fold (Blinks_30_Fold<k>.npy = sujeitos dos outros folds)."""
    partes, treino = [], {}
    for k in range(1, U.N_FOLDS + 1):
        xtr, ytr, xte, yte, vid, sid = U.load_fold_with_ids(k)          # float32, igual ao Exp. 1
        X64 = np.load(DIR_OF / f"BlinksTest_30_Fold{k}.npy", allow_pickle=True).astype(np.float64)
        # conferência: o treino oficial do fold k não pode conter nenhuma sequência do teste do fold k
        chaves = lambda A: {r.tobytes() for r in np.round(np.asarray(A, np.float64).reshape(len(A), -1), 6)}  # noqa: E731
        assert not (chaves(xtr) & chaves(xte)), f"fold {k}: sequência do teste aparece no treino oficial"
        mk = mapa[mapa["fold"] == k]
        pasta_de = {int(r.sujeito_oficial): (r.pasta or f"F{k}_oficial{int(r.sujeito_oficial):02d}") for r in mk.itertuples()}
        classe_de = {int(r.sujeito_oficial): str(r.classe) for r in mk.itertuples()}
        assert set(pasta_de) == set(int(s) for s in np.unique(sid)), f"fold {k}: mapeamento não cobre os sujeitos oficiais"
        suj = np.array([pasta_de[int(s)] for s in sid])
        partes.append(dict(X=xte, X64=X64, y=yte.astype(int), fold=np.full(len(yte), k), sid=sid.astype(int),
                           sujeito=suj, video=np.array([f"{p}_{KSS[int(c)]}" for p, c in zip(suj, yte)]),
                           classe=np.array([classe_de[int(s)] for s in sid])))
        treino[k] = dict(X=xtr, y=ytr.astype(int), n_sujeitos=sum(len(U.OFFICIAL_START_INDICES[f]) for f in
                                                                   range(1, U.N_FOLDS + 1) if f != k) // 3)
    OF = {c: np.concatenate([p[c] for p in partes]) for c in partes[0]}
    log(f"Oficiais (teste dos 5 folds): {len(OF['y'])} sequências, {len(set(OF['video']))} vídeos, "
        f"{len(set(OF['sujeito']))} sujeitos; por classe do par: "
        + ", ".join(f"{c} {len(set(OF['sujeito'][OF['classe'] == c]))}" for c in ("confirmado", "provavel", "indeterminado")))
    return OF, treino


def montar_nosso(z, mapa, log):
    classe_pasta = {r.pasta: str(r.classe) for r in mapa.itertuples() if r.pasta}
    NO = {var: z[key].astype(np.float64) for var, key in VARIANTES.items()}
    NO.update(y=z["y"].astype(int), fold=z["fold"].astype(int), sujeito=z["sujeito"].astype(str),
              video=z["video"].astype(str), seq=z["seq_no_video"].astype(int))
    NO["classe"] = np.array([classe_pasta.get(s, "sem_par") for s in NO["sujeito"]])
    log(f"Nossas (sequences.npz): {len(NO['y'])} sequências, {len(set(NO['video']))} vídeos lógicos com sequência, "
        f"{len(set(NO['sujeito']))} pastas")
    return NO


def mascara_conj(L, conj):
    return np.isin(L["classe"], CONJUNTOS[conj])


def videos_pareados(OF, NO, conj, log=None):
    """Vídeos do conjunto presentes nos DOIS lados (a comparação pareada usa só esses)."""
    vo = set(OF["video"][mascara_conj(OF, conj)])
    vn = set(NO["video"][mascara_conj(NO, conj)])
    so_of, so_no = sorted(vo - vn), sorted(vn - vo)
    if log is not None and (so_of or so_no):
        log(f"[{conj}] vídeos fora da comparação pareada: só no oficial {so_of}; só no nosso {so_no}")
    return sorted(vo & vn), so_of, so_no


def esquema_completo(y, esquema):
    """Rótulos do esquema no comprimento original (-1 = sequência descartada pelo esquema)."""
    ys, keep = U.apply_scheme(y, esquema)
    out = np.full(len(y), -1, dtype=int)
    out[keep] = ys
    return out


def indice_videos_logicos():
    """blinks_index (uma linha por ARQUIVO) agregado por vídeo lógico: nº de quadros e piscadas somados; taxas por
    minuto de rosto (tempo com rosto de cada parte = n_piscadas / taxa, ou taxa_rosto x duração se não houver
    piscadas); frac_fechado ponderada pelos quadros. ear_median = mediana do EAR (quadros com rosto) do vídeo inteiro,
    recalculada dos sinais nos vídeos divididos."""
    b = pd.read_csv(BIDX_CSV, dtype={"rotulo": str})
    b = b[b["dataset"] == "uta"].copy()
    idx = pd.read_csv(SINAIS_IDX)
    b = b.merge(idx[["id", "ear_median", "width_orig", "height_orig", "rotation_meta"]], on="id", how="left")
    b["tface_s"] = np.where(b["n_piscadas"] > 0, 60.0 * b["n_piscadas"] / b["taxa_piscadas_min"],
                            b["taxa_rosto"] * b["dur_s"])
    linhas = []
    for v, g in b.groupby("video"):
        g = g.sort_values("id")
        tf = float(g["tface_s"].sum())
        r = dict(video=v, sujeito=g["sujeito"].iloc[0], rotulo=str(g["rotulo"].iloc[0]), fold=int(g["fold"].iloc[0]),
                 n_arquivos=len(g), arquivos=";".join(g["id"]), n_quadros=int(g["n_quadros"].sum()),
                 dur_s=float(g["dur_s"].sum()), n_piscadas=int(g["n_piscadas"].sum()), limiar=float(g["limiar"].iloc[0]),
                 taxa_piscadas_min=float(g["n_piscadas"].sum()) / (tf / 60.0) if tf > 0 else np.nan,
                 vales12_por_min=float((g["vales12_por_min"] * g["tface_s"]).sum()) / tf if tf > 0 else np.nan,
                 vales25_por_min=float((g["vales25_por_min"] * g["tface_s"]).sum()) / tf if tf > 0 else np.nan,
                 frac_fechado=float((g["frac_fechado"] * g["n_quadros"]).sum() / g["n_quadros"].sum()),
                 taxa_rosto=float((g["taxa_rosto"] * g["n_quadros"]).sum() / g["n_quadros"].sum()))
        if len(g) == 1:
            r["ear_median"] = float(g["ear_median"].iloc[0])
        else:
            ears = []
            for cid in g["id"]:
                with np.load(ROOT / "data" / "signals" / "uta" / f"{cid}.npz", allow_pickle=False) as zz:
                    e = zz["ear"].astype(float)
                    ears.append(e[zz["face"] & np.isfinite(e)])
            r["ear_median"] = float(np.median(np.concatenate(ears)))
        linhas.append(r)
    return pd.DataFrame(linhas)


def series_videos(X, video, seq=None):
    """Série de piscadas (n x 4) de cada vídeo, reconstruída das sequências (passo 2; cada piscada uma vez)."""
    out = {}
    for v in pd.unique(video):
        idx = np.flatnonzero(video == v)
        if seq is not None:
            idx = idx[np.argsort(seq[idx], kind="stable")]
        S = X[idx]
        assert O.verificar_sobreposicao(S), f"{v}: sequências consecutivas não se sobrepõem em 28 piscadas"
        out[str(v)] = O.serie_do_video(S)
    return out


# ======================================================================================
# 1. Mapeamento: checagens que não usam o escore do Exp. 5
# ======================================================================================
def n_piscadas_aritmetica(N, n_linhas_reais):
    """Nº de piscadas cobertas pelas sequências de um vídeo, pela aritmética de unroll_in_time (janela 30, passo 2).

    N >= 2 -> n = 30 + 2(N-1) (a piscada final ímpar, se houver, não entra em nenhuma sequência);
    N == 1 -> n <= 31: o nº exato é o de linhas não preenchidas com zeros (preenchimento à esquerda quando n < 30)."""
    return 30 + 2 * (N - 1) if N >= 2 else int(n_linhas_reais)


def p_acaso_exata(acertos, n_s, m_p):
    """P(>= acertos) de uma atribuição injetiva uniforme de n_s sujeitos a m_p pastas coincidir com uma fixa
    (inclusão-exclusão: P(exatamente j) = sum_{s>=j} (-1)^(s-j) C(s,j) C(n,s) perm(m-s, n-s) / perm(m, n))."""
    tot = math.perm(m_p, n_s)

    def p_exata(j):
        return sum((-1) ** (s_ - j) * math.comb(s_, j) * math.comb(n_s, s_) * math.perm(m_p - s_, n_s - s_)
                   for s_ in range(j, n_s + 1)) / tot
    return float(max(0.0, 1.0 - sum(p_exata(j) for j in range(acertos))))


def mapear_por_contagem(k, OF, mapa, log):
    """Húngaro sobre o erro relativo médio |n_nosso - n_oficial| / n_oficial nos 3 vídeos, para os sujeitos oficiais
    com pasta adotada (confirmados e prováveis) do fold k contra todas as pastas do fold."""
    mk = mapa[(mapa["fold"] == k) & (mapa["pasta"] != "")]
    pasta_de = {int(r.sujeito_oficial): r.pasta for r in mk.itertuples()}
    classe_de = {int(r.sujeito_oficial): r.classe for r in mk.itertuples()}
    m = (OF["fold"] == k) & np.isin(OF["sujeito"], list(pasta_de.values()))
    ser_of = series_videos(OF["X64"][m], OF["video"][m])
    res = pd.read_csv(SEQ_RESUMO_CSV, dtype={"rotulo": int})
    res = res[res["fold"] == k]
    n_no = {(r.sujeito, int(r.rotulo)): int(r.n_piscadas_nas_sequencias) for r in res.itertuples()}
    pastas = sorted(res["sujeito"].unique())
    S = sorted(pasta_de)
    n_of, N_of, conf_arit = {}, {}, []
    for s in S:
        for lab in (0, 5, 10):
            v = f"{pasta_de[s]}_{lab}"
            N = int(np.sum(OF["video"][m] == v))
            n_exato = len(ser_of[v])
            n_ar = n_piscadas_aritmetica(N, n_exato)
            conf_arit.append(n_ar == n_exato)
            n_of[(s, lab)], N_of[(s, lab)] = n_ar, N
    C = np.full((len(S), len(pastas)), np.inf)
    for i, s in enumerate(S):
        for j, p in enumerate(pastas):
            if all((p, lab) in n_no for lab in (0, 5, 10)):
                C[i, j] = np.mean([abs(n_no[(p, lab)] - n_of[(s, lab)]) / n_of[(s, lab)] for lab in (0, 5, 10)])
    r, c = linear_sum_assignment(np.where(np.isfinite(C), C, 1e6))
    por_cont = {S[i]: pastas[j] for i, j in zip(r, c)}
    linhas = []
    for i, s in enumerate(S):
        j_ad = pastas.index(pasta_de[s])
        linhas.append(dict(
            fold=k, sujeito_oficial=s, classe=classe_de[s], pasta_adotada=pasta_de[s],
            **{f"n_seq_oficial_{lab}": N_of[(s, lab)] for lab in (0, 5, 10)},
            **{f"n_piscadas_oficial_{lab}": n_of[(s, lab)] for lab in (0, 5, 10)},
            **{f"n_piscadas_nossa_adotada_{lab}": n_no.get((pasta_de[s], lab), np.nan) for lab in (0, 5, 10)},
            erro_rel_medio_adotada=float(C[i, j_ad]), posto_adotada_na_contagem=int(1 + np.sum(C[i] < C[i, j_ad])),
            pasta_por_contagem=por_cont[s], erro_rel_medio_por_contagem=float(C[i, pastas.index(por_cont[s])]),
            contagem_coincide=bool(por_cont[s] == pasta_de[s])))
    tab = pd.DataFrame(linhas)
    acertos = int(tab["contagem_coincide"].sum())
    info = dict(acertos=acertos, n=len(S), n_pastas=len(pastas), esperado_acaso=len(S) / len(pastas),
                p_acaso=p_acaso_exata(acertos, len(S), len(pastas)), aritmetica_ok=int(sum(conf_arit)),
                aritmetica_n=len(conf_arit), acertos_confirmados=int(tab.loc[tab["classe"] == "confirmado",
                                                                               "contagem_coincide"].sum()),
                n_confirmados=int((tab["classe"] == "confirmado").sum()))
    log(f"[fold {k}] 1a. contagem de piscadas: aritmética confere em {info['aritmetica_ok']}/{info['aritmetica_n']} "
        f"vídeos; húngaro coincide com o adotado em {acertos}/{len(S)} (acaso: esperado {info['esperado_acaso']:.2f}, "
        f"P(>= {acertos}) = {info['p_acaso']:.3f})")
    return tab, info


def contar_atribuicoes(viavel):
    """Nº de atribuições injetivas linhas -> colunas que usam só pares viáveis (programação dinâmica em subconjuntos)."""
    nr, nc = viavel.shape
    dp = {0: 1}
    for i in range(nr):
        novo = defaultdict(int)
        for mask, cnt in dp.items():
            for j in range(nc):
                if viavel[i, j] and not (mask >> j) & 1:
                    novo[mask | (1 << j)] += cnt
        dp = novo
    return int(sum(dp.values()))


def checar_duracao(k, mapa, vlog, log):
    """NF da última piscada oficial <= nº de quadros do vídeo lógico da pasta candidata (sem EAR), para os sujeitos
    oficiais com pasta adotada do fold k contra todas as pastas do fold."""
    v = pd.read_csv(VAL_VIDEOS_CSV)
    v = v[v["fold"] == k]
    NF = {(int(r.sujeito_oficial), int(r.rotulo)): float(r.NF_final_oficial) for r in v.itertuples()}
    T = {(int(r.sujeito_oficial), int(r.rotulo)): int(r.T_final_oficial) for r in v.itertuples()}
    vk = vlog[vlog["fold"] == k]
    nq = dict(zip(vk["video"], vk["n_quadros"]))
    pastas = sorted(vk["sujeito"].unique())
    mk = mapa[(mapa["fold"] == k) & (mapa["pasta"] != "")]
    pasta_de = {int(r.sujeito_oficial): r.pasta for r in mk.itertuples()}
    classe_de = {int(r.sujeito_oficial): r.classe for r in mk.itertuples()}
    # série oficial confiável (Frequency recuperada com NF inteiro; src/features/uta_oficial.py): só elas restringem
    conf_serie = {(int(r.sujeito_oficial), lab): str(r.series_confiaveis).split(";")[i] == "1"
                  for r in mk.itertuples() for i, lab in enumerate((0, 5, 10))}
    S = sorted(pasta_de)
    viavel = np.zeros((len(S), len(pastas)), bool)
    custo = np.zeros((len(S), len(pastas)))
    for i, s in enumerate(S):
        for j, p in enumerate(pastas):
            viol, c = 0, 0.0
            for lab in (0, 5, 10):
                nf, n = NF.get((s, lab)), nq.get(f"{p}_{lab}")
                if not conf_serie[(s, lab)]:
                    continue
                if nf is None or n is None or not np.isfinite(nf):
                    viol += 1
                    continue
                if nf > n * (1 + TOL_NF):
                    viol += 1
                elif lab != 0:   # no vídeo alerta a série oficial termina em ~2/3 (o último terço é a referência)
                    c += math.log(n / nf)
            viavel[i, j] = viol == 0
            custo[i, j] = c + PENAL_NF * viol
    r, c = linear_sum_assignment(custo)
    por_dur = {S[i]: pastas[j] for i, j in zip(r, c)}
    linhas = []
    for i, s in enumerate(S):
        p = pasta_de[s]
        razoes = {lab: NF[(s, lab)] / nq[f"{p}_{lab}"] for lab in (0, 5, 10)}
        linhas.append(dict(fold=k, sujeito_oficial=s, n_series_confiaveis=int(sum(conf_serie[(s, lab)] for lab in (0, 5, 10))),
                           n_pastas_viaveis_duracao=int(viavel[i].sum()),
                           pastas_viaveis_duracao=";".join(pastas[j] for j in np.flatnonzero(viavel[i])),
                           adotada_viavel=bool(viavel[i, pastas.index(p)]),
                           pasta_por_duracao=por_dur[s], duracao_coincide=bool(por_dur[s] == p),
                           **{f"NF_final_oficial_{lab}": NF[(s, lab)] for lab in (0, 5, 10)},
                           **{f"T_final_oficial_{lab}": T[(s, lab)] for lab in (0, 5, 10)},
                           **{f"n_quadros_adotada_{lab}": int(nq[f"{p}_{lab}"]) for lab in (0, 5, 10)},
                           **{f"razao_NF_quadros_{lab}": razoes[lab] for lab in (0, 5, 10)}))
    tab = pd.DataFrame(linhas)
    conf = [i for i, s in enumerate(S) if classe_de[s] == "confirmado"]
    ok_ = lambda s_, lab: NF[(s_, lab)] <= nq[f"{pasta_de[s_]}_{lab}"] * (1 + TOL_NF)  # noqa: E731
    info = dict(n=len(S), n_pastas=len(pastas),
                videos_viaveis=int(sum(ok_(s, lab) for s in S for lab in (0, 5, 10) if conf_serie[(s, lab)])),
                n_videos=int(sum(conf_serie[(s, lab)] for s in S for lab in (0, 5, 10))),
                videos_viaveis_conf=int(sum(ok_(S[i], lab) for i in conf for lab in (0, 5, 10) if conf_serie[(S[i], lab)])),
                n_videos_conf=int(sum(conf_serie[(S[i], lab)] for i in conf for lab in (0, 5, 10))),
                violacoes=[(s, pasta_de[s], lab, NF[(s, lab)] / nq[f"{pasta_de[s]}_{lab}"]) for s in S
                           for lab in (0, 5, 10) if conf_serie[(s, lab)] and not ok_(s, lab)],
                max_razao_conf=float(max([NF[(S[i], lab)] / nq[f"{pasta_de[S[i]]}_{lab}"] for i in conf
                                          for lab in (0, 5, 10) if conf_serie[(S[i], lab)]], default=np.nan)),
                n_atrib_viaveis=contar_atribuicoes(viavel), n_atrib_total=math.perm(len(pastas), len(S)),
                acertos=int(tab["duracao_coincide"].sum()))
    log(f"[fold {k}] 1b. duração (só séries confiáveis): pares adotados respeitam NF_final <= quadros em "
        f"{info['videos_viaveis']}/{info['n_videos']} vídeos (confirmados: {info['videos_viaveis_conf']}/"
        f"{info['n_videos_conf']}; violações: {info['violacoes']}); atribuições "
        f"completas viáveis {info['n_atrib_viaveis']} de {info['n_atrib_total']}; húngaro pela duração coincide em "
        f"{info['acertos']}/{len(S)}")
    return tab, info


def resumo_indep(mapa):
    """Verificação independente do mapeamento (src/experiments/exp5a_mapeamento_indep.py): por sujeito oficial com
    pasta adotada, a melhor pasta pelo outro detector de vales, o escore da adotada e o da melhor alternativa."""
    if not INDEP_CSV.exists():
        return pd.DataFrame()
    d = pd.read_csv(INDEP_CSV)
    d["pasta_adotada"] = d["pasta_adotada"].fillna("").astype(str)
    linhas = []
    for (k, s), g in d.groupby(["fold", "sujeito_oficial"]):
        g = g.sort_values("escore", ascending=False)
        ad = str(g["pasta_adotada"].iloc[0])
        cl = mapa[(mapa["fold"] == k) & (mapa["sujeito_oficial"] == s)]["classe"]
        e_ad = float(g.loc[g["pasta"] == ad, "escore"].iloc[0]) if ad else np.nan
        alt = g[g["pasta"] != ad].iloc[0] if ad else g.iloc[1]     # sem pasta adotada: a 2ª melhor
        linhas.append(dict(fold=int(k), sujeito_oficial=int(s), classe=str(cl.iloc[0]) if len(cl) else "",
                           pasta_adotada=ad, indep_melhor=str(g["pasta"].iloc[0]), indep_escore_melhor=float(g["escore"].iloc[0]),
                           indep_escore_adotada=e_ad, indep_melhor_alternativa=str(alt["pasta"]),
                           indep_escore_alternativa=float(alt["escore"]),
                           indep_margem=e_ad - float(alt["escore"]) if ad else np.nan,
                           indep_coincide=bool(ad and g["pasta"].iloc[0] == ad)))
    return pd.DataFrame(linhas)


def diagnostico_videos(log):
    """Propriedades de cada ARQUIVO de vídeo (codec, contêiner, fps, duração, tamanho do rosto, brilho). Incremental."""
    import cv2  # opencv está no .venv (só leitura de alguns quadros; o Face Mesh não é usado aqui)
    idx = pd.read_csv(SINAIS_IDX)
    idx = idx[idx["complete"].astype(bool)].copy()
    idx["fold"] = idx["extra_json"].map(lambda e: json.loads(e).get("fold"))
    antigo = pd.read_csv(DIAG_CSV) if DIAG_CSV.exists() else pd.DataFrame(columns=["id"])
    feitos = set(antigo["id"].astype(str))
    novos = []
    for r in idx.itertuples():
        if r.id in feitos:
            continue
        with np.load(ROOT / "data" / "signals" / "uta" / f"{r.id}.npz", allow_pickle=False) as z:
            lm, li = z["lm"], list(z["lm_idx"])
            wp = float(z["meta_width_proc"])
            path = ROOT / str(z["meta_path"])
        esc = 450.0 / wp   # quadro exibido (em pé) reduzido a 450 px de largura, como imutils.resize(frame, width=450)
        with np.errstate(invalid="ignore"):
            alt = np.linalg.norm(lm[:, li.index(10)] - lm[:, li.index(152)], axis=1)
            iod = np.linalg.norm(lm[:, li.index(33)] - lm[:, li.index(263)], axis=1)
        cap = cv2.VideoCapture(str(path))
        fcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fcc >> 8 * i) & 0xFF) for i in range(4))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        brilho = []
        for f in np.linspace(0, max(n - 10, 0), 12).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, fr = cap.read()
            if ok:
                brilho.append(float(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).mean()))
        cap.release()
        with open(path, "rb") as fh:
            cab = fh.read(16)
        novos.append(dict(id=r.id, codec=codec, conteiner=cab[8:12].decode("latin-1").strip(),
                          tamanho_mb=path.stat().st_size / 1e6,
                          altura_rosto_px_450=float(np.nanmedian(alt) * esc),
                          distancia_interocular_px_450=float(np.nanmedian(iod) * esc),
                          brilho_medio=float(np.mean(brilho)) if brilho else np.nan))
        if len(novos) % 20 == 0:
            log(f"  diagnóstico de vídeo: {len(novos)} arquivo(s) novo(s) lido(s)")
    if novos:
        log(f"Diagnóstico de vídeo: {len(novos)} arquivo(s) novo(s) lido(s); {len(feitos)} já estavam em {DIAG_CSV.name}")
        todos = pd.concat([antigo, pd.DataFrame(novos)], ignore_index=True).sort_values("id")
        tmp = DIAG_CSV.with_suffix(".tmp.csv")
        todos.to_csv(tmp, index=False)
        tmp.replace(DIAG_CSV)
    else:
        log(f"Diagnóstico de vídeo: nenhum arquivo novo ({len(feitos)} em {DIAG_CSV.name})")
    diag = pd.read_csv(DIAG_CSV)
    bidx = pd.read_csv(BIDX_CSV, dtype={"rotulo": str})
    bidx = bidx[bidx["dataset"] == "uta"]
    cols_b = ["id", "sujeito", "rotulo", "video", "parte", "fold", "fps", "n_quadros", "dur_s", "taxa_rosto", "n_piscadas",
              "taxa_piscadas_min", "n_fechamentos_longos", "vales12_por_min", "vales25_por_min", "limiar"]
    cols_s = ["id", "width_orig", "height_orig", "rotation_meta"]
    out = bidx[cols_b].merge(idx[cols_s], on="id", how="left").merge(diag, on="id", how="left")
    if DLIB_CSV.exists():   # src/experiments/exp5a_dlib_rosto.py (.venv-mp): detector do dlib com o pré-processamento original
        dl = pd.read_csv(DLIB_CSV)[["id", "taxa_dlib_sem_ampliacao", "taxa_dlib_com_1_ampliacao",
                                    "largura_caixa_mediana_px", "quadros_testados"]]
        out = out.merge(dl, on="id", how="left")
        log(f"Detector do dlib: {int(out['taxa_dlib_sem_ampliacao'].notna().sum())} de {len(out)} arquivos medidos "
            f"({DLIB_CSV.name})")
    else:
        log(f"Detector do dlib: {DLIB_CSV.name} não existe (rode src/experiments/exp5a_dlib_rosto.py no .venv-mp)")
    return out.reset_index(drop=True)


# ======================================================================================
# 4. Distribuições (piscadas; mesmos sujeitos e vídeos nos dois lados)
# ======================================================================================
def resumo_amostra(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return dict(n=0, media=np.nan, dp=np.nan, p5=np.nan, q1=np.nan, mediana=np.nan, q3=np.nan, p95=np.nan)
    p5, q1, med, q3, p95 = np.percentile(x, [5, 25, 50, 75, 95])
    return dict(n=len(x), media=float(x.mean()), dp=float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                p5=p5, q1=q1, mediana=med, q3=q3, p95=p95)


def distribuicoes(OF, NO, conj, vids_par, log):
    """KS (piscadas de todos os sujeitos; e D mediano por sujeito) e Spearman entre as médias por vídeo, por variante,
    classe e fold (fold = "todos" é o agregado dos 5 folds)."""
    mo = mascara_conj(OF, conj) & np.isin(OF["video"], vids_par)
    ser_of = series_videos(OF["X64"][mo], OF["video"][mo])
    fold_de = {str(v): int(f) for v, f in zip(OF["video"][mo], OF["fold"][mo])}
    linhas, pool = [], {"oficial": ser_of}
    for var in VARIANTES:
        mn = mascara_conj(NO, conj) & np.isin(NO["video"], vids_par)
        ser_no = series_videos(NO[var][mn], NO["video"][mn], NO["seq"][mn])
        pool[var] = ser_no
        vids = [v for v in ser_of if v in ser_no]
        assert len(vids) == len(ser_of) == len(ser_no), "vídeos sem par entre as séries oficiais e as nossas"
        for fold in ["todos"] + sorted(set(fold_de.values())):
            vf = [v for v in vids if fold == "todos" or fold_de[v] == fold]
            for cls in ["todas", 0, 1, 2]:
                sel = [v for v in vf if cls == "todas" or classe_de_video(v) == cls]
                A = np.concatenate([ser_of[v] for v in sel])
                Bn = np.concatenate([ser_no[v] for v in sel])
                sujs = sorted({sujeito_de_video(v) for v in sel})
                for j, c in enumerate(CARACT):
                    a, b = A[:, j], Bn[:, j]
                    ks = ks_2samp(a, b)
                    Ds = []
                    for s in sujs:
                        xa = np.concatenate([ser_of[v][:, j] for v in sel if sujeito_de_video(v) == s])
                        xb = np.concatenate([ser_no[v][:, j] for v in sel if sujeito_de_video(v) == s])
                        if len(xa) and len(xb):
                            Ds.append(ks_2samp(xa, xb).statistic)
                    ra, rb = resumo_amostra(a), resumo_amostra(b)
                    linhas.append(dict(tipo="distribuicao", conjunto_sujeitos=conj, fold=str(fold), variante=var,
                                       classe=str(cls), caracteristica=c,
                                       **{f"{kk}_oficial": vv for kk, vv in ra.items()},
                                       **{f"{kk}_nosso": vv for kk, vv in rb.items()},
                                       ks_D=float(ks.statistic), ks_p=float(ks.pvalue),
                                       ks_D_mediana_por_sujeito=float(np.median(Ds)) if Ds else np.nan,
                                       ks_D_q1_por_sujeito=float(np.percentile(Ds, 25)) if Ds else np.nan,
                                       ks_D_q3_por_sujeito=float(np.percentile(Ds, 75)) if Ds else np.nan,
                                       n_sujeitos=len(Ds), n_videos=len(sel)))
            for subset, sel in (("todos", vf), ("so_5_e_10", [v for v in vf if kss_de_video(v) in (5, 10)])):
                for j, c in enumerate(CARACT):
                    a = np.array([ser_of[v][:, j].mean() for v in sel])
                    b = np.array([ser_no[v][:, j].mean() for v in sel])
                    ok = np.isfinite(a) & np.isfinite(b)
                    sr = spearmanr(a[ok], b[ok])
                    linhas.append(dict(tipo="spearman_medias_video", conjunto_sujeitos=conj, fold=str(fold),
                                       variante=var, classe=subset, caracteristica=c, n_videos=int(ok.sum()),
                                       spearman=float(sr.statistic), spearman_p=float(sr.pvalue)))
    df = pd.DataFrame(linhas)
    d = df[(df["tipo"] == "distribuicao") & (df["classe"] == "todas") & (df["variante"] == VARIANTE_PRINCIPAL)
           & (df["fold"] == "todos")]
    log(f"[{conj}] 4. KS (5 folds, todas as classes, {VARIANTE_PRINCIPAL}): " +
        "; ".join(f"{r.caracteristica} D={r.ks_D:.3f} (por sujeito {r.ks_D_mediana_por_sujeito:.3f})" for r in d.itertuples()))
    return df, pool


def mecanismo_frequencia(vlog, sujeitos, log):
    """Por que a nossa Frequency cresce nos vídeos 5 e 10? Diferença (vídeo 5 ou 10) - (vídeo 0) por sujeito de:
    taxa de piscadas por limiar, quedas do EAR sem limiar (>= 12% da linha de base), fração de quadros abaixo do
    limiar e mediana do EAR (por vídeo lógico). Wilcoxon pareado (bicaudal) nos sujeitos."""
    b = vlog[vlog["sujeito"].isin(sujeitos)]
    cols = {"taxa_piscadas_min": "piscadas/min (limiar)", "vales12_por_min": "quedas do EAR >= 12%/min (sem limiar)",
            "frac_fechado": "fração de quadros abaixo do limiar", "ear_median": "mediana do EAR"}
    linhas = []
    for col, desc in cols.items():
        pv = b.pivot_table(index="sujeito", columns="rotulo", values=col)
        for lab in ("5", "10"):
            d = (pv[lab] - pv["0"]).dropna()
            w = wilcoxon(d) if len(d) >= 5 and np.any(d != 0) else None
            linhas.append(dict(medida=col, descricao=desc, comparacao=f"{lab} - 0", n_sujeitos=len(d),
                               mediana_video0=float(pv["0"].median()), mediana_video=float(pv[lab].median()),
                               mediana_diferenca=float(d.median()), n_sobe=int((d > 0).sum()), n_desce=int((d < 0).sum()),
                               wilcoxon_p=float(w.pvalue) if w is not None else np.nan))
    df = pd.DataFrame(linhas)
    r = df[df["comparacao"] == "10 - 0"].set_index("medida")
    log(f"mecanismo da Frequency (10 - 0, {len(set(b['sujeito']))} sujeitos): " +
        "; ".join(f"{m} sobe em {int(r.loc[m, 'n_sobe'])} (p={r.loc[m, 'wilcoxon_p']:.3g})" for m in cols))
    return df


# ======================================================================================
# 2-3. RandomForest: métricas e estatística pareada por sujeito
# ======================================================================================
def novo_rf(seed):
    return RandomForestClassifier(n_estimators=N_ARVORES, random_state=seed, n_jobs=N_JOBS)


def metricas_conf(C):
    """BSA, acurácia balanceada e F1 macro a partir da matriz de confusão (mesmas convenções do scikit-learn)."""
    C = np.asarray(C, float)
    tot = C.sum()
    lin, col, dia = C.sum(1), C.sum(0), np.diag(C)
    presentes_true = lin > 0
    rec = np.divide(dia, lin, out=np.zeros_like(dia), where=lin > 0)
    prec = np.divide(dia, col, out=np.zeros_like(dia), where=col > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(dia), where=(prec + rec) > 0)
    rotulos = (lin > 0) | (col > 0)
    return dict(bsa=float(dia.sum() / tot) if tot else np.nan,
                bal_acc=float(rec[presentes_true].mean()) if presentes_true.any() else np.nan,
                f1_macro=float(f1[rotulos].mean()) if rotulos.any() else np.nan)


def metricas_conf_vet(C):
    """metricas_conf para uma pilha de matrizes (P x K x K), vetorizada (bootstrap e permutação)."""
    C = np.asarray(C, float)
    lin, col = C.sum(2), C.sum(1)
    dia = np.diagonal(C, axis1=1, axis2=2)
    tot = C.sum((1, 2))
    rec = np.where(lin > 0, dia / np.where(lin > 0, lin, 1.0), 0.0)
    prec = np.where(col > 0, dia / np.where(col > 0, col, 1.0), 0.0)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.where(prec + rec > 0, prec + rec, 1.0), 0.0)
    pres, rot = lin > 0, (lin > 0) | (col > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(bsa=dia.sum(1) / tot, bal_acc=(rec * pres).sum(1) / pres.sum(1), f1_macro=(f1 * rot).sum(1) / rot.sum(1))


def resultados_predicao(ys, pred, proba, video, sujeito, K):
    vid_int, vid_nomes = pd.factorize(video)
    m = U.sequence_metrics(ys, pred, proba)
    va, dec, verd = U.video_accuracy(ys, pred, proba, vid_int)
    ch_seq, ch_vid = U.chance_levels(ys, vid_int)
    conf_s, hits_s, vids = {}, {}, []
    for s in pd.unique(sujeito):
        ms = sujeito == s
        conf_s[s] = confusion_matrix(ys[ms], pred[ms], labels=list(range(K)))
        vs = np.unique(vid_int[ms])
        hits_s[s] = (int(sum(dec[int(v)] == verd[int(v)] for v in vs)), len(vs))
    for v in range(len(vid_nomes)):
        mv = vid_int == v
        vids.append(dict(video=str(vid_nomes[v]), rotulo_verdadeiro=int(verd[v]), decisao=int(dec[v]),
                         acerto=bool(dec[v] == verd[v]), n_seq=int(mv.sum()),
                         frac_votos_decisao=float(np.mean(pred[mv] == dec[v])),
                         **{f"prob_media_{c}": float(proba[mv, c].mean()) for c in range(K)}))
    # conferência: métricas a partir das confusões por sujeito = scikit-learn
    mc = metricas_conf(sum(conf_s.values()))
    assert abs(mc["bsa"] - m["bsa"]) < 1e-12 and abs(mc["bal_acc"] - m["bal_acc"]) < 1e-12
    assert abs(mc["f1_macro"] - f1_score(ys, pred, average="macro")) < 1e-12
    return dict(bsa=m["bsa"], bal_acc=m["bal_acc"], f1_macro=m["f1_macro"], auc_macro=m["auc_macro"], va=va,
                n_seq=len(ys), n_videos=len(vid_nomes), n_sujeitos=len(conf_s), chance_seq=ch_seq, chance_video=ch_vid,
                confusao=m["confusion"], conf_s=conf_s, hits_s=hits_s, videos=vids,
                classes_seq=np.bincount(ys, minlength=K).tolist())


def _pilhas(ra, rb, S):
    Ca = np.stack([ra["conf_s"][s] for s in S]).astype(float)
    Cb = np.stack([rb["conf_s"][s] for s in S]).astype(float)
    Ha = np.array([ra["hits_s"][s] for s in S], float)
    Hb = np.array([rb["hits_s"][s] for s in S], float)
    return Ca, Cb, Ha, Hb


def bootstrap_pareado(ra, rb, S, n_boot, seed):
    """IC 95% (percentil) de A, B e A - B reamostrando SUJEITOS com reposição (mesma reamostragem para A e B)."""
    Ca, Cb, Ha, Hb = _pilhas(ra, rb, S)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(S), size=(n_boot, len(S)))
    W = np.zeros((n_boot, len(S)))
    np.add.at(W, (np.repeat(np.arange(n_boot), len(S)), idx.ravel()), 1.0)
    ma = metricas_conf_vet(np.einsum("bs,sij->bij", W, Ca))
    mb = metricas_conf_vet(np.einsum("bs,sij->bij", W, Cb))
    ma["va"] = (W @ Ha[:, 0]) / (W @ Ha[:, 1])
    mb["va"] = (W @ Hb[:, 0]) / (W @ Hb[:, 1])
    res = {}
    for m in MET_BOOT:
        a, b_ = ma[m], mb[m]
        d = a - b_
        res[m] = dict(A_ic_inf=np.nanpercentile(a, 2.5), A_ic_sup=np.nanpercentile(a, 97.5),
                      B_ic_inf=np.nanpercentile(b_, 2.5), B_ic_sup=np.nanpercentile(b_, 97.5),
                      dif_ic_inf=np.nanpercentile(d, 2.5), dif_ic_sup=np.nanpercentile(d, 97.5),
                      frac_dif_pos=float(np.nanmean(d > 0)), dif_dp=float(np.nanstd(d, ddof=1)))
    return res


def permutacao_pareada(ra, rb, S, n_perm, seed):
    """Teste de permutação pareado por sujeito (bicaudal): sob H0 os dois lados de cada sujeito são trocáveis; cada
    sorteio troca (i) e (ii) dentro de um subconjunto aleatório de sujeitos e recalcula a diferença agregada.
    p = (1 + nº de |dif. sorteada| >= |dif. observada|) / (1 + nº de sorteios)."""
    Ca, Cb, Ha, Hb = _pilhas(ra, rb, S)
    assert np.array_equal(Ha[:, 1], Hb[:, 1]), "a permutação exige os mesmos vídeos nos dois lados"
    rng = np.random.default_rng(seed)
    M = rng.integers(0, 2, size=(n_perm, len(S))).astype(float)
    D = Cb - Ca
    A = Ca.sum(0)[None] + np.einsum("ps,sij->pij", M, D)
    Bm = Cb.sum(0)[None] - np.einsum("ps,sij->pij", M, D)
    ma, mb = metricas_conf_vet(A), metricas_conf_vet(Bm)
    dH = Hb - Ha
    HA, HB = Ha.sum(0)[None, :] + M @ dH, Hb.sum(0)[None, :] - M @ dH
    ma["va"], mb["va"] = HA[:, 0] / HA[:, 1], HB[:, 0] / HB[:, 1]
    oa, ob = metricas_conf(Ca.sum(0)), metricas_conf(Cb.sum(0))
    oa["va"], ob["va"] = Ha[:, 0].sum() / Ha[:, 1].sum(), Hb[:, 0].sum() / Hb[:, 1].sum()
    out = {}
    for m in MET_BOOT:
        nulo = ma[m] - mb[m]
        d0 = abs(oa[m] - ob[m])
        out[m] = dict(perm_p=float((1 + np.sum(np.abs(nulo) >= d0 - 1e-12)) / (n_perm + 1)), perm_n=n_perm)
    return out


def por_sujeito(ra, rb, S):
    """Quantos sujeitos têm (i) > (ii), (i) < (ii) e empate na acurácia balanceada e na VA do próprio sujeito."""
    out = {}
    for m in ("bal_acc", "va"):
        if m == "bal_acc":
            a = np.array([metricas_conf(ra["conf_s"][s])["bal_acc"] for s in S])
            b = np.array([metricas_conf(rb["conf_s"][s])["bal_acc"] for s in S])
        else:
            a = np.array([ra["hits_s"][s][0] / ra["hits_s"][s][1] for s in S])
            b = np.array([rb["hits_s"][s][0] / rb["hits_s"][s][1] for s in S])
        out[m] = dict(suj_oficial_maior=int(np.sum(a > b + 1e-12)), suj_nosso_maior=int(np.sum(b > a + 1e-12)),
                      suj_empate=int(np.sum(np.abs(a - b) <= 1e-12)), suj_dif_mediana=float(np.median(a - b)))
    return out


def mcnemar_videos(ra, rb):
    a = {v["video"]: v["acerto"] for v in ra["videos"]}
    b = {v["video"]: v["acerto"] for v in rb["videos"]}
    comuns = sorted(set(a) & set(b))
    so_a = sum(a[v] and not b[v] for v in comuns)
    so_b = sum(b[v] and not a[v] for v in comuns)
    p = binomtest(min(so_a, so_b), so_a + so_b, 0.5).pvalue if so_a + so_b else 1.0
    return dict(n_videos_pareados=len(comuns), so_oficial_acerta=int(so_a), so_nosso_acerta=int(so_b),
                ambos_acertam=int(sum(a[v] and b[v] for v in comuns)),
                nenhum_acerta=int(sum((not a[v]) and (not b[v]) for v in comuns)), mcnemar_p=float(p))


def comparar(ra, rb, n_boot, n_perm, perm=True):
    """Estatística pareada (i) - (ii) nos mesmos sujeitos e vídeos: bootstrap, permutação, McNemar e contagem por sujeito."""
    S = sorted(ra["conf_s"])
    assert set(S) == set(rb["conf_s"]), "os dois lados precisam ter os mesmos sujeitos"
    bt = bootstrap_pareado(ra, rb, S, n_boot, SEED)
    pm = permutacao_pareada(ra, rb, S, n_perm, SEED) if perm else {}
    mc = mcnemar_videos(ra, rb)
    ps = por_sujeito(ra, rb, S)
    linhas = {}
    for m in METRICAS:
        row = dict(metrica=m, oficial=float(ra[m]), nosso=float(rb[m]), diferenca=float(ra[m] - rb[m]),
                   n_sujeitos=len(S), n_videos=int(ra["n_videos"]), n_seq_oficial=int(ra["n_seq"]),
                   n_seq_nosso=int(rb["n_seq"]))
        if m in bt:
            row.update({f"boot_{kk}": vv for kk, vv in bt[m].items()}, boot_n=n_boot)
        if m in pm:
            row.update(pm[m])
        if m in ps:
            row.update(ps[m])
        if m == "va":
            row.update(mc)
        linhas[m] = row
    return linhas


def videos_preenchidos(conjuntos):
    """Vídeos representados por uma única sequência com preenchimento de zeros à esquerda (< 30 piscadas)."""
    out = set()
    for X, video in conjuntos:
        for v in pd.unique(video):
            mv = video == v
            if mv.sum() == 1 and np.all(X[mv][0] == 0, axis=1).any():
                out.add(str(v))
    return out


def conferir_exp1(k, esquema, r, log):
    if not EXP1_CSV.exists():
        return None
    e = pd.read_csv(EXP1_CSV)
    e = e[(e["model"] == "RandomForest") & (e["fold"] == k) & (e["scheme"] == esquema)]
    if e.empty:
        return None
    dif = max(abs(float(e.loc[e["metric"] == m, "value"].iloc[0]) - float(r[m])) for m in METRICAS
              if (e["metric"] == m).any() and fin(r[m]))
    log(f"[fold {k}] [{esquema}] conferência com o Exp. 1 (RandomForest, fold {k}): maior |diferença| = {dif:.2e} "
        f"-> {'reproduz' if dif < 1e-9 else 'NÃO reproduz'}")
    return dif


def avaliar_lado(L, pr, esquema, mask):
    y = esquema_completo(L["y"], esquema)
    m = mask & (y >= 0)
    return resultados_predicao(y[m], pr[0][m], pr[1][m], L["video"][m], L["sujeito"][m], U.n_classes(esquema))


def prever_treino_oficial(OF, NO, treino, sementes, log):
    """Parte 2: para cada esquema e semente, RF treinado no Blinks_30_Fold<k> (oficiais dos outros 4 folds) e aplicado
    às sequências de teste do fold k — oficiais e nossas. Devolve predições alinhadas a OF e NO (-1 = fora do esquema)."""
    P = {}
    F_of = U.sequences_to_features(OF["X"])
    F_no = {var: U.sequences_to_features(NO[var]) for var in VARIANTES}
    for esquema in ESQUEMAS:
        K = U.n_classes(esquema)
        y_of, y_no = esquema_completo(OF["y"], esquema), esquema_completo(NO["y"], esquema)
        for seed in sementes:
            t0 = time.time()
            pr = {"oficial": [np.full(len(y_of), -1), np.full((len(y_of), K), np.nan)]}
            pr.update({f"nosso_{v}": [np.full(len(y_no), -1), np.full((len(y_no), K), np.nan)] for v in VARIANTES})
            for k in range(1, U.N_FOLDS + 1):
                ytr = esquema_completo(treino[k]["y"], esquema)
                ktr = ytr >= 0
                rf = novo_rf(seed).fit(U.sequences_to_features(treino[k]["X"][ktr]), ytr[ktr])
                assert list(rf.classes_) == list(range(K))
                m = (OF["fold"] == k) & (y_of >= 0)
                pr["oficial"][0][m], pr["oficial"][1][m] = rf.predict(F_of[m]), rf.predict_proba(F_of[m])
                for var in VARIANTES:
                    m = (NO["fold"] == k) & (y_no >= 0)
                    pr[f"nosso_{var}"][0][m], pr[f"nosso_{var}"][1][m] = rf.predict(F_no[var][m]), rf.predict_proba(F_no[var][m])
            P[(esquema, seed)] = pr
            log(f"  [{esquema}] semente {seed}: 5 RF treinados nas oficiais dos outros folds ({time.time() - t0:.0f} s)")
    return P


def cv_grupos(X, y, video, sujeito, grupo, esquema, seed):
    """Validação cruzada por grupos com o MESMO extrator no treino e no teste (grupo = fold ou sujeito)."""
    ys, keep = U.apply_scheme(y, esquema)
    F = U.sequences_to_features(X[keep])
    vn, sj, gr = video[keep], sujeito[keep], grupo[keep]
    K = U.n_classes(esquema)
    pred = np.zeros(len(ys), int)
    proba = np.zeros((len(ys), K))
    for g in pd.unique(gr):
        te = gr == g
        rf = novo_rf(seed).fit(F[~te], ys[~te])
        p = np.zeros((int(te.sum()), K))
        p[:, rf.classes_] = rf.predict_proba(F[te])
        proba[te] = p
        pred[te] = rf.predict(F[te])
    return dict(ys=ys, pred=pred, proba=proba, video=vn, sujeito=sj, grupo=gr, K=K)


def res_cv(cv, mask=None):
    m = np.ones(len(cv["ys"]), bool) if mask is None else mask
    return resultados_predicao(cv["ys"][m], cv["pred"][m], cv["proba"][m], cv["video"][m], cv["sujeito"][m], cv["K"])


# ======================================================================================
# Execução das partes 2 e 3
# ======================================================================================
EXTRA_MET = ["n_seq", "n_videos", "n_sujeitos", "chance_seq", "chance_video"]


def linhas_met(secao, conj, fold, esquema, conjunto, semente, r):
    return [dict(secao=secao, conjunto_sujeitos=conj, fold=str(fold), esquema=esquema, conjunto_teste=conjunto,
                 semente=semente, metrica=m, valor=float(r[m])) for m in METRICAS + EXTRA_MET]


def linhas_videos(secao, conj, esquema, conjunto, r, fold_de):
    return [dict(secao=secao, conjunto_sujeitos=conj, esquema=esquema, conjunto_teste=conjunto, semente=SEED,
                 fold=fold_de.get(sujeito_de_video(v["video"])), **v) for v in r["videos"]]


def linha_conf(secao, conj, esquema, conjunto, r):
    return dict(secao=secao, conjunto_sujeitos=conj, esquema=esquema, conjunto_teste=conjunto,
                confusao=json.dumps(np.asarray(r["confusao"]).tolist()))


def linhas_custo(secao, conj, fold, esquema, var, cmp):
    return [dict(secao=secao, conjunto_sujeitos=conj, fold=str(fold), esquema=esquema, variante=var,
                 comparacao=f"oficial - nosso_{var}", **row) for row in cmp.values()]


def rodar_parte2(OF, NO, treino, sementes, n_boot, n_perm, vids_par, fold_de, log):
    log("\n===== Parte 2: RF treinado nas oficiais dos outros folds (protocolo oficial, 5 folds)")
    P = prever_treino_oficial(OF, NO, treino, sementes, log)
    met, custo, vids, confs, det, repro = [], [], [], [], {}, {}
    todos_of = np.ones(len(OF["y"]), bool)
    for esquema in ESQUEMAS:
        pr = P[(esquema, SEED)]
        for k in range(1, U.N_FOLDS + 1):          # (i) com os 58 sujeitos oficiais: conferência com o Exp. 1
            r = avaliar_lado(OF, pr["oficial"], esquema, OF["fold"] == k)
            repro[(esquema, k)] = conferir_exp1(k, esquema, r, log)
            met += linhas_met(SEC_TO, "todos_oficiais", k, esquema, "oficial", SEED, r)
        r = avaliar_lado(OF, pr["oficial"], esquema, todos_of)
        met += linhas_met(SEC_TO, "todos_oficiais", "todos", esquema, "oficial", SEED, r)
        det[(SEC_TO, "todos_oficiais", esquema, "oficial")] = r
        for var in VARIANTES:                        # (ii) nossas, todas as 60 pastas (contexto, sem pareamento)
            r = avaliar_lado(NO, pr[f"nosso_{var}"], esquema, np.ones(len(NO["y"]), bool))
            met += linhas_met(SEC_TO, "todas_as_pastas", "todos", esquema, f"nosso_{var}", SEED, r)
            det[(SEC_TO, "todas_as_pastas", esquema, f"nosso_{var}")] = r
        for conj in CONJUNTOS:
            vp = vids_par[conj]
            mo = mascara_conj(OF, conj) & np.isin(OF["video"], vp)
            mn = mascara_conj(NO, conj) & np.isin(NO["video"], vp)
            for fold in ["todos"] + list(range(1, U.N_FOLDS + 1)):
                fo = mo if fold == "todos" else mo & (OF["fold"] == fold)
                fn = mn if fold == "todos" else mn & (NO["fold"] == fold)
                ra = avaliar_lado(OF, pr["oficial"], esquema, fo)
                met += linhas_met(SEC_TO, conj, fold, esquema, "oficial", SEED, ra)
                if fold == "todos":
                    det[(SEC_TO, conj, esquema, "oficial")] = ra
                    vids += linhas_videos(SEC_TO, conj, esquema, "oficial", ra, fold_de)
                    confs.append(linha_conf(SEC_TO, conj, esquema, "oficial", ra))
                for var in VARIANTES:
                    rb = avaliar_lado(NO, pr[f"nosso_{var}"], esquema, fn)
                    met += linhas_met(SEC_TO, conj, fold, esquema, f"nosso_{var}", SEED, rb)
                    custo += linhas_custo(SEC_TO, conj, fold, esquema, var, comparar(ra, rb, n_boot, n_perm))
                    if fold == "todos":
                        det[(SEC_TO, conj, esquema, f"nosso_{var}")] = rb
                        vids += linhas_videos(SEC_TO, conj, esquema, f"nosso_{var}", rb, fold_de)
                        confs.append(linha_conf(SEC_TO, conj, esquema, f"nosso_{var}", rb))
            ra, rb = det[(SEC_TO, conj, esquema, "oficial")], det[(SEC_TO, conj, esquema, f"nosso_{VARIANTE_PRINCIPAL}")]
            log(f"[{esquema}] [{conj}] agregado: oficial bal={ra['bal_acc']:.3f} VA={ra['va']:.3f} | nosso "
                f"{VARIANTE_PRINCIPAL} bal={rb['bal_acc']:.3f} VA={rb['va']:.3f} ({ra['n_sujeitos']} sujeitos, "
                f"{ra['n_videos']} vídeos, {ra['n_seq']} x {rb['n_seq']} seq.)")
            for seed in sementes:                   # sensibilidade à semente do RF (agregado dos 5 folds)
                if seed == SEED:
                    continue
                prs = P[(esquema, seed)]
                met += linhas_met(SEC_TO, conj, "todos", esquema, "oficial", seed, avaliar_lado(OF, prs["oficial"], esquema, mo))
                for var in VARIANTES:
                    met += linhas_met(SEC_TO, conj, "todos", esquema, f"nosso_{var}", seed,
                                      avaliar_lado(NO, prs[f"nosso_{var}"], esquema, mn))
        # sensibilidade: sem os vídeos representados por UMA sequência com zeros à esquerda (< 30 piscadas) em qualquer
        # lado (exclusão pareada), no conjunto principal
        vp = vids_par[CONJ_P]
        mo = mascara_conj(OF, CONJ_P) & np.isin(OF["video"], vp)
        mn = mascara_conj(NO, CONJ_P) & np.isin(NO["video"], vp)
        excl = videos_preenchidos([(OF["X"][mo], OF["video"][mo])] + [(NO[v][mn], NO["video"][mn]) for v in VARIANTES])
        det[("excluidos_preenchidos", esquema)] = sorted(excl)
        fo, fn = mo & ~np.isin(OF["video"], sorted(excl)), mn & ~np.isin(NO["video"], sorted(excl))
        ra = avaliar_lado(OF, pr["oficial"], esquema, fo)
        met += linhas_met(SEC_TO + "_sem_preenchidos", CONJ_P, "todos", esquema, "oficial", SEED, ra)
        det[(SEC_TO + "_sem_preenchidos", CONJ_P, esquema, "oficial")] = ra
        for var in VARIANTES:
            rb = avaliar_lado(NO, pr[f"nosso_{var}"], esquema, fn)
            met += linhas_met(SEC_TO + "_sem_preenchidos", CONJ_P, "todos", esquema, f"nosso_{var}", SEED, rb)
            custo += linhas_custo(SEC_TO + "_sem_preenchidos", CONJ_P, "todos", esquema, var, comparar(ra, rb, n_boot, n_perm))
            det[(SEC_TO + "_sem_preenchidos", CONJ_P, esquema, f"nosso_{var}")] = rb
        log(f"[{esquema}] sem os {len(excl)} vídeos preenchidos: oficial bal={ra['bal_acc']:.3f} VA={ra['va']:.3f}; nosso "
            f"{VARIANTE_PRINCIPAL} bal={det[(SEC_TO + '_sem_preenchidos', CONJ_P, esquema, 'nosso_' + VARIANTE_PRINCIPAL)]['bal_acc']:.3f}")
    met_df, custo_df = pd.DataFrame(met), pd.DataFrame(custo)
    # resumo das sementes nas linhas de custo agregadas (fold "todos")
    for c in ["sementes_n", "sementes_dif_media", "sementes_dif_dp", "sementes_dif_min", "sementes_dif_max",
              "sementes_dif_positiva", "sementes_dif_negativa"]:
        custo_df[c] = np.nan
    for i, row in custo_df[(custo_df["secao"] == SEC_TO) & (custo_df["fold"] == "todos")].iterrows():
        q = met_df[(met_df["secao"] == SEC_TO) & (met_df["conjunto_sujeitos"] == row["conjunto_sujeitos"])
                   & (met_df["fold"] == "todos") & (met_df["esquema"] == row["esquema"]) & (met_df["metrica"] == row["metrica"])]
        a_s = q[q["conjunto_teste"] == "oficial"].set_index("semente")["valor"]
        b_s = q[q["conjunto_teste"] == f"nosso_{row['variante']}"].set_index("semente")["valor"]
        d_s = (a_s - b_s).dropna()
        custo_df.loc[i, ["sementes_n", "sementes_dif_media", "sementes_dif_dp", "sementes_dif_min", "sementes_dif_max",
                         "sementes_dif_positiva", "sementes_dif_negativa"]] = [
            len(d_s), d_s.mean(), d_s.std(ddof=1) if len(d_s) > 1 else np.nan, d_s.min(), d_s.max(),
            int((d_s > 0).sum()), int((d_s < 0).sum())]
    return met_df, custo_df, vids, confs, det, repro


def rodar_parte3(OF, NO, n_boot, n_perm, vids_par, fold_de, com_loso, log):
    log("\n===== Parte 3: mesmo extrator no treino e no teste (5 folds e LOSO), mesmos sujeitos")
    met, custo, vids, confs, det = [], [], [], [], {}
    for conj in CONJUNTOS:
        vp = vids_par[conj]
        mo = mascara_conj(OF, conj) & np.isin(OF["video"], vp)
        mn = mascara_conj(NO, conj) & np.isin(NO["video"], vp)
        protocolos = [(SEC_5F, "fold")] + ([(SEC_LOSO, "sujeito")] if (conj == CONJ_P and com_loso) else [])
        for secao, gcol in protocolos:
            for esquema in ESQUEMAS:
                t0 = time.time()
                cvo = cv_grupos(OF["X"][mo], OF["y"][mo], OF["video"][mo], OF["sujeito"][mo], OF[gcol][mo], esquema, SEED)
                fo_cv = np.array([fold_de[s] for s in cvo["sujeito"]])
                ra = res_cv(cvo)
                cvn = {var: cv_grupos(NO[var][mn], NO["y"][mn], NO["video"][mn], NO["sujeito"][mn], NO[gcol][mn],
                                      esquema, SEED) for var in VARIANTES}
                fn_cv = np.array([fold_de[s] for s in cvn[VARIANTE_PRINCIPAL]["sujeito"]])
                for fold in ["todos"] + list(range(1, U.N_FOLDS + 1)):
                    ra_f = ra if fold == "todos" else res_cv(cvo, fo_cv == fold)
                    met += linhas_met(secao, conj, fold, esquema, "oficial", SEED, ra_f)
                    for var in VARIANTES:
                        rb_f = res_cv(cvn[var]) if fold == "todos" else res_cv(cvn[var], fn_cv == fold)
                        met += linhas_met(secao, conj, fold, esquema, f"nosso_{var}", SEED, rb_f)
                        custo += linhas_custo(secao, conj, fold, esquema, var, comparar(ra_f, rb_f, n_boot, n_perm))
                        if fold == "todos":
                            det[(secao, conj, esquema, f"nosso_{var}")] = rb_f
                            vids += linhas_videos(secao, conj, esquema, f"nosso_{var}", rb_f, fold_de)
                            confs.append(linha_conf(secao, conj, esquema, f"nosso_{var}", rb_f))
                det[(secao, conj, esquema, "oficial")] = ra
                vids += linhas_videos(secao, conj, esquema, "oficial", ra, fold_de)
                confs.append(linha_conf(secao, conj, esquema, "oficial", ra))
                rb = det[(secao, conj, esquema, f"nosso_{VARIANTE_PRINCIPAL}")]
                log(f"[{secao}] [{conj}] [{esquema}] oficial bal={ra['bal_acc']:.3f} VA={ra['va']:.3f} | nosso "
                    f"{VARIANTE_PRINCIPAL} bal={rb['bal_acc']:.3f} VA={rb['va']:.3f} ({time.time() - t0:.0f} s)")
        if conj == CONJ_P:   # sensibilidade: 5 folds sem os vídeos preenchidos (exclusão pareada)
            excl = videos_preenchidos([(OF["X"][mo], OF["video"][mo])] + [(NO[v][mn], NO["video"][mn]) for v in VARIANTES])
            fo, fn = mo & ~np.isin(OF["video"], sorted(excl)), mn & ~np.isin(NO["video"], sorted(excl))
            for esquema in ESQUEMAS:
                ra = res_cv(cv_grupos(OF["X"][fo], OF["y"][fo], OF["video"][fo], OF["sujeito"][fo], OF["fold"][fo], esquema, SEED))
                met += linhas_met(SEC_5F + "_sem_preenchidos", conj, "todos", esquema, "oficial", SEED, ra)
                det[(SEC_5F + "_sem_preenchidos", conj, esquema, "oficial")] = ra
                for var in (VARIANTE_PRINCIPAL, "limiar"):
                    rb = res_cv(cv_grupos(NO[var][fn], NO["y"][fn], NO["video"][fn], NO["sujeito"][fn], NO["fold"][fn],
                                          esquema, SEED))
                    met += linhas_met(SEC_5F + "_sem_preenchidos", conj, "todos", esquema, f"nosso_{var}", SEED, rb)
                    custo += linhas_custo(SEC_5F + "_sem_preenchidos", conj, "todos", esquema, var, comparar(ra, rb, n_boot, n_perm))
                    det[(SEC_5F + "_sem_preenchidos", conj, esquema, f"nosso_{var}")] = rb
    return pd.DataFrame(met), pd.DataFrame(custo), vids, confs, det


# ======================================================================================
# Figuras
# ======================================================================================
def figura_custo(ctx, log):
    """Figura principal: diferença oficial − nosso (variante principal), com IC 95% por bootstrap sobre sujeitos, nos
    três protocolos e nos dois esquemas; painéis = acurácia balanceada e VA."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MultipleLocator
    import make_figures as MF   # mesmo estilo (paleta validada, fontes, vírgula decimal) das outras figuras

    MF.setup_style()
    custo = ctx["tabs"]["custo"]
    secoes = [s for s in (SEC_TO, SEC_5F, SEC_LOSO) if ((custo["secao"] == s) & (custo["conjunto_sujeitos"] == CONJ_P)).any()]
    rot_sec = {SEC_TO: "Treino nas oficiais,\nteste nas nossas", SEC_5F: "Mesmo extrator:\n5 folds",
               SEC_LOSO: "Mesmo extrator:\nLOSO"}
    estilo = {"3c": dict(color=MF.C1, marker="o", label="3 classes"),
              "B3": dict(color=MF.C2, marker="D", label="Alerta × sonolento (B3)")}
    off = {"3c": -0.15, "B3": 0.15}
    fig, axes = plt.subplots(1, 2, figsize=(17.6 / 2.54, 6.4 / 2.54), sharey=True)
    lo, hi = 0.0, 0.0
    for ax, met in zip(axes, ("bal_acc", "va")):
        for i, sec in enumerate(secoes):
            for esq in ESQUEMAS:
                r = _c(custo, sec, CONJ_P, "todos", esq, VARIANTE_PRINCIPAL, met)
                if r is None:
                    continue
                d, a, b = 100 * r["diferenca"], 100 * r["boot_dif_ic_inf"], 100 * r["boot_dif_ic_sup"]
                lo, hi = min(lo, a), max(hi, b)
                y = i + off[esq]
                ax.plot([a, b], [y, y], color=estilo[esq]["color"], lw=2.0, solid_capstyle="round", zorder=3)
                ax.plot([d], [y], marker=estilo[esq]["marker"], ms=6 if esq == "3c" else 5.4, color=estilo[esq]["color"],
                        mec=MF.SURFACE, mew=1.2, zorder=4, ls="none")
        ax.axvline(0, color=MF.INK2, lw=0.8, zorder=2)
        ax.set_yticks(range(len(secoes)), [rot_sec[s] for s in secoes])
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color=MF.GRID, lw=0.5)
        ax.grid(axis="y", visible=False)
        ax.set_title("Acurácia balanceada" if met == "bal_acc" else "Acurácia por vídeo (VA)", loc="left",
                     fontsize=MF.FS, color=MF.INK, fontweight="bold")
        ax.set_xlabel("Oficial − nosso (pp)   ·   positivo: o nosso extrator perde", color=MF.INK2)
    pad = 0.08 * (hi - lo)
    for ax in axes:
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(len(secoes) - 0.5, -0.5)
        ax.xaxis.set_major_locator(MultipleLocator(10 if hi - lo > 45 else 5))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 0)))
    n_suj = int(ctx["n_sujeitos_conj"][CONJ_P])
    fig.legend(handles=[Line2D([0], [0], color=estilo[e]["color"], marker=estilo[e]["marker"], lw=2.0, ms=5.5,
                               mec=MF.SURFACE, label=estilo[e]["label"]) for e in ESQUEMAS],
               loc="upper left", ncol=2, bbox_to_anchor=(0.005, 1.0), frameon=False, fontsize=MF.FS_S)
    fig.text(0.012, 0.885, f"UTA-RLDD, 5 folds, {n_suj} sujeitos com par confirmado  ·  ponto: diferença agregada nos "
             f"sujeitos  ·  barra: IC 95% (bootstrap sobre sujeitos, {milhar(ctx['n_boot'])} reamostragens)",
             ha="left", va="top", fontsize=MF.FS_S - 1, color=MF.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.83), w_pad=1.5)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG / "fig_exp5a_validacao.pdf")
    fig.savefig(OUT_FIG / "fig_exp5a_validacao.png", dpi=300)
    plt.close(fig)
    log(f"  -> {_rel(OUT_FIG / 'fig_exp5a_validacao.pdf')} (+ .png 300 dpi)")


def figura_distribuicoes(ctx, log):
    """4 painéis (uma característica cada): distribuição por classe, oficial x nosso (variante principal), mesma escala
    vertical dentro do painel. Caixa = quartis, traço na cor da superfície = mediana, haste = percentis 5-95."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import FuncFormatter
    import make_figures as MF

    MF.setup_style()
    cores = {"oficial": MF.C1, "nosso": MF.C2}
    pool = ctx["pool"]
    dados = {"oficial": defaultdict(list), "nosso": defaultdict(list)}
    for v, S in pool["oficial"].items():
        dados["oficial"][classe_de_video(v)].append(S)
    for v, S in pool[VARIANTE_PRINCIPAL].items():
        dados["nosso"][classe_de_video(v)].append(S)
    dados = {f: {c: np.concatenate(L) for c, L in d.items()} for f, d in dados.items()}
    dist = ctx["tabs"]["dist"]
    ks = dist[(dist["tipo"] == "distribuicao") & (dist["classe"] == "todas") & (dist["variante"] == VARIANTE_PRINCIPAL)
              & (dist["fold"] == "todos") & (dist["conjunto_sujeitos"] == CONJ_P)]
    fig, axes = plt.subplots(2, 2, figsize=(17.6 / 2.54, 10.6 / 2.54))
    w, off = 0.28, 0.17
    for j, (ax, c) in enumerate(zip(axes.ravel(), CARACT)):
        lo, hi = np.inf, -np.inf
        for ci in range(3):
            for fonte, dx in (("oficial", -off), ("nosso", off)):
                x = dados[fonte][ci][:, j]
                p5, q1, med, q3, p95 = np.percentile(x, [5, 25, 50, 75, 95])
                lo, hi = min(lo, p5), max(hi, p95)
                xc = ci + dx
                ax.add_patch(Rectangle((xc - w / 2, q1), w, q3 - q1, facecolor=cores[fonte], edgecolor="none", zorder=3))
                ax.plot([xc, xc], [p5, q1], color=cores[fonte], lw=1.0, solid_capstyle="butt", zorder=3)
                ax.plot([xc, xc], [q3, p95], color=cores[fonte], lw=1.0, solid_capstyle="butt", zorder=3)
                ax.plot([xc - w / 2, xc + w / 2], [med, med], color=MF.SURFACE, lw=1.6, solid_capstyle="butt", zorder=4)
        pad = 0.06 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.55, 2.55)
        ax.axhline(0, color=MF.AXIS, lw=0.8, zorder=2)
        ax.set_xticks([0, 1, 2], ["Alerta", "Pouca vigilância", "Sonolento"])
        ax.tick_params(axis="x", length=0)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: br(v, 0) if abs(v - round(v)) < 1e-9 else br(v, 1)))
        if j % 2 == 0:
            ax.set_ylabel("z (normalização por sujeito)", color=MF.INK2)
        r = ks[ks["caracteristica"] == c].iloc[0]
        ax.set_title(CARACT_NOME[c], loc="left", fontsize=MF.FS, color=MF.INK, fontweight="bold")
        ax.set_title(f"KS D = {br(r['ks_D'], 2)}  ·  por sujeito {br(r['ks_D_mediana_por_sujeito'], 2)}",
                     loc="right", fontsize=MF.FS_S - 0.5, color=MF.INK2)
    n_suj = int(ctx["n_sujeitos_conj"][CONJ_P])
    fig.legend(handles=[Patch(color=cores["oficial"], label="Oficial: dlib + SVM (Ghoddoosian et al., 2019)"),
                        Patch(color=cores["nosso"], label="Nosso: MediaPipe Face Mesh + limiar de EAR (bordas no olho aberto)")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=MF.FS_S)
    fig.text(0.5, 0.935, f"Piscadas das sequências de 30 (cada uma contada uma vez)  ·  {n_suj} sujeitos com par confirmado  ·  "
             "5 folds\nCaixa: quartis  ·  traço branco: mediana  ·  haste: percentis 5 a 95  ·  linha cinza: z = 0 "
             "(média da referência)", ha="center", va="top", fontsize=MF.FS_S - 1, color=MF.INK2, linespacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.87), h_pad=1.2, w_pad=1.6)
    fig.savefig(OUT_FIG / "fig_exp5a_distribuicoes.pdf")
    fig.savefig(OUT_FIG / "fig_exp5a_distribuicoes.png", dpi=300)
    plt.close(fig)
    log(f"  -> {_rel(OUT_FIG / 'fig_exp5a_distribuicoes.pdf')} (+ .png 300 dpi)")


# ======================================================================================
# Execução
# ======================================================================================
def main():
    global N_JOBS, OUT_RES, OUT_FIG, OUT_LOGS
    ap = argparse.ArgumentParser(description="Exp. 5a (final): nosso extrator x características oficiais da UTA-RLDD.")
    ap.add_argument("--bootstrap", type=int, default=1000, help="reamostragens de sujeitos (padrão 1000)")
    ap.add_argument("--permutacoes", type=int, default=10000, help="sorteios do teste de permutação pareado (padrão 10000)")
    ap.add_argument("--sementes", type=int, default=10, help="nº de sementes do RF a partir de 42 (padrão 10)")
    ap.add_argument("--jobs", type=int, default=N_JOBS, help="n_jobs do RandomForest (padrão 6; não muda o resultado)")
    ap.add_argument("--sem-diagnostico-video", action="store_true", help="não lê quadros dos vídeos (parte 1d)")
    ap.add_argument("--sem-loso", action="store_true", help="pula o LOSO (só para testes rápidos)")
    ap.add_argument("--sem-relatorio", action="store_true", help="não escreve o .md (só para testes)")
    ap.add_argument("--saida", type=str, default=None, help="grava CSVs, figuras, log e .md neste diretório (testes)")
    args = ap.parse_args()
    if args.saida:
        OUT_RES = OUT_FIG = OUT_LOGS = Path(args.saida).resolve()
    N_JOBS = max(1, min(6, args.jobs))
    ts = time.strftime("%Y%m%d-%H%M%S")
    OUT_LOGS.mkdir(parents=True, exist_ok=True)
    log = Log(OUT_LOGS / f"exp5a_validacao_{ts}.log")
    t0 = time.time()
    log(f"Exp. 5a (rodada final) — validação do extrator contra as características oficiais da UTA-RLDD ({ts})")
    log(f"Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, scipy {scipy.__version__}, "
        f"scikit-learn {sklearn.__version__}")
    log(f"RF: {N_ARVORES} árvores, sementes {SEED}..{SEED + args.sementes - 1} (principal {SEED}), n_jobs={N_JOBS}; "
        f"bootstrap por sujeito: {args.bootstrap} reamostragens; permutação pareada: {args.permutacoes} sorteios "
        f"(semente {SEED}); variante principal: {VARIANTE_PRINCIPAL}; conjunto principal: {CONJ_P}")
    z, terco = carregar_nossas()
    assert terco == "ultimo", f"sequences.npz com terço de referência '{terco}' (esperado 'ultimo', como os .npy oficiais)"
    log(f"Nossas sequências: {SEQ_NPZ.relative_to(ROOT)} (terço de referência = {terco})")
    mapa = carregar_mapa()
    OF, treino = montar_oficial(mapa, log)
    NO = montar_nosso(z, mapa, log)
    fold_de = {**{str(s): int(f) for s, f in zip(OF["sujeito"], OF["fold"])},
               **{str(s): int(f) for s, f in zip(NO["sujeito"], NO["fold"])}}
    vids_par, fora_par, n_suj_conj = {}, {}, {}
    for conj in CONJUNTOS:
        vp, so_of, so_no = videos_pareados(OF, NO, conj, log)
        vids_par[conj], fora_par[conj] = vp, (so_of, so_no)
        n_suj_conj[conj] = len({sujeito_de_video(v) for v in vp})
        log(f"[{conj}] {n_suj_conj[conj]} sujeitos, {len(vp)} vídeos pareados")
    vlog = indice_videos_logicos()
    # ---------------------------------------------------------------- 1. mapeamento
    log("\n===== Parte 1: checagens do mapeamento")
    tabs_c, infos_c, tabs_d, infos_d = [], {}, [], {}
    for k in range(1, U.N_FOLDS + 1):
        tc, ic = mapear_por_contagem(k, OF, mapa, log)
        td, id_ = checar_duracao(k, mapa, vlog, log)
        tabs_c.append(tc)
        tabs_d.append(td)
        infos_c[k], infos_d[k] = ic, id_
    indep = resumo_indep(mapa)
    mapa_t = mapa[["fold", "sujeito_oficial", "pasta", "classe", "motivo", "escore", "pasta_segunda", "segundo_escore",
                   "margem"]].rename(columns={"escore": "escore_vales_ear", "pasta_segunda": "segunda_pasta_vales_ear",
                                              "segundo_escore": "segundo_escore_vales_ear", "margem": "margem_vales_ear"})
    mapa_t = mapa_t.merge(pd.concat(tabs_c, ignore_index=True).drop(columns=["classe", "pasta_adotada"]),
                          on=["fold", "sujeito_oficial"], how="left") \
                   .merge(pd.concat(tabs_d, ignore_index=True), on=["fold", "sujeito_oficial"], how="left")
    if len(indep):
        mapa_t = mapa_t.merge(indep.drop(columns=["classe", "pasta_adotada"]), on=["fold", "sujeito_oficial"], how="left")
    # ---------------------------------------------------------------- 2 e 3. RandomForest
    sementes = list(range(SEED, SEED + args.sementes))
    met2, custo2, vids2, confs2, det2, repro = rodar_parte2(OF, NO, treino, sementes, args.bootstrap, args.permutacoes,
                                                            vids_par, fold_de, log)
    met3, custo3, vids3, confs3, det3 = rodar_parte3(OF, NO, args.bootstrap, args.permutacoes, vids_par, fold_de,
                                                     not args.sem_loso, log)
    # ---------------------------------------------------------------- 4. distribuições e mecanismo
    log("\n===== Parte 4: distribuições")
    dist_p, pool = distribuicoes(OF, NO, CONJ_P, vids_par[CONJ_P], log)
    dist_s, _ = distribuicoes(OF, NO, "confirmado_provavel", vids_par["confirmado_provavel"], log)
    suj_p = sorted({sujeito_de_video(v) for v in vids_par[CONJ_P]})
    mec = mecanismo_frequencia(vlog, suj_p, log)
    diag = None if args.sem_diagnostico_video else diagnostico_videos(log)
    # ---------------------------------------------------------------- arquivos
    log("\nArquivos:")
    tabs = dict(met=pd.concat([met2, met3], ignore_index=True), custo=pd.concat([custo2, custo3], ignore_index=True),
                dist=pd.concat([dist_p, dist_s], ignore_index=True), mapeamento=mapa_t, mecanismo=mec)
    vids_all = pd.DataFrame(vids2 + vids3)
    confs_all = pd.DataFrame(confs2 + confs3)
    gravar_csv(tabs["met"], OUT_RES / "exp5a_validacao.csv", log)
    gravar_csv(tabs["custo"], OUT_RES / "exp5a_validacao_custo.csv", log)
    gravar_csv(tabs["dist"], OUT_RES / "exp5a_validacao_distribuicoes.csv", log)
    gravar_csv(tabs["mapeamento"], OUT_RES / "exp5a_validacao_mapeamento.csv", log)
    gravar_csv(tabs["mecanismo"], OUT_RES / "exp5a_validacao_mecanismo_frequencia.csv", log)
    gravar_csv(vids_all, OUT_RES / "exp5a_validacao_videos.csv", log)
    gravar_csv(confs_all, OUT_RES / "exp5a_validacao_confusoes.csv", log)
    ctx = dict(tabs=tabs, det={**det2, **det3}, repro=repro, pool=pool, diag=diag, vlog=vlog, mapa=mapa, OF=OF, NO=NO,
               vids_par=vids_par, fora_par=fora_par, n_sujeitos_conj=n_suj_conj, infos_c=infos_c, infos_d=infos_d,
               indep=indep, sementes=sementes, n_boot=args.bootstrap, n_perm=args.permutacoes, ts=ts,
               log_path=log.path, treino=treino, terco=terco, com_loso=not args.sem_loso)
    figura_custo(ctx, log)
    figura_distribuicoes(ctx, log)
    if not args.sem_relatorio:
        relatorio(ctx, log)
    log(f"\nConcluído em {time.time() - t0:.0f} s. Log: {_rel(log.path)}")
    return 0


# ======================================================================================
# Relatório — utilidades
# ======================================================================================
def _m(met, secao, conj, fold, esquema, conjunto, metrica, semente=SEED):
    d = met[(met["secao"] == secao) & (met["conjunto_sujeitos"] == conj) & (met["fold"] == str(fold))
            & (met["esquema"] == esquema) & (met["conjunto_teste"] == conjunto) & (met["semente"] == semente)
            & (met["metrica"] == metrica)]
    return float(d["valor"].iloc[0]) if len(d) else np.nan


def _c(custo, secao, conj, fold, esquema, var, metrica):
    d = custo[(custo["secao"] == secao) & (custo["conjunto_sujeitos"] == conj) & (custo["fold"] == str(fold))
              & (custo["esquema"] == esquema) & (custo["variante"] == var) & (custo["metrica"] == metrica)]
    return d.iloc[0] if len(d) else None


def _ic(row, lado="dif"):
    if row is None or not fin(row.get(f"boot_{lado}_ic_inf", np.nan)):
        return "—"
    f = pp if lado == "dif" else pct
    return f"[{f(row[f'boot_{lado}_ic_inf'])}; {f(row[f'boot_{lado}_ic_sup'])}]"


def _exclui_zero(row):
    return row is not None and fin(row.get("boot_dif_ic_inf")) and (row["boot_dif_ic_inf"] > 0 or row["boot_dif_ic_sup"] < 0)


def _meia(row):
    return (row["boot_dif_ic_sup"] - row["boot_dif_ic_inf"]) / 2 if row is not None else np.nan


def _nvid(dif, n):
    m = abs(int(round(float(dif) * float(n))))
    return f"{m} vídeo" if m == 1 else f"{m} vídeos"


# ======================================================================================
# Relatório
# ======================================================================================
def _maiuscula(t):
    return t[:1].upper() + t[1:]


def _txt_ic_zero(row, metrica_txt):
    if row is None:
        return ""
    if _exclui_zero(row):
        lado = "o nosso extrator perde" if row["boot_dif_ic_inf"] > 0 else "o nosso extrator ganha"
        return f"o IC 95% da {metrica_txt} exclui zero ({lado})"
    return f"o IC 95% da {metrica_txt} inclui zero"


def _custo_frase(custo, secao, conj, esquema, var, n_videos):
    """'bal. +x pp (IC [..]; permutação p = ..), VA +y pp (n de N vídeos; IC [..]; permutação p = ..; McNemar p = ..)'."""
    cb, cv = _c(custo, secao, conj, "todos", esquema, var, "bal_acc"), _c(custo, secao, conj, "todos", esquema, var, "va")
    if cb is None or cv is None:
        return "—"
    return (f"acurácia balanceada {pp(cb['diferenca'])} (IC 95% {_ic(cb)}; permutação {peq(cb.get('perm_p'))}), "
            f"VA {pp(cv['diferenca'])} ({_nvid(cv['diferenca'], n_videos)} de {n_videos}; IC 95% {_ic(cv)}; "
            f"permutação {peq(cv.get('perm_p'))}; McNemar {peq(cv.get('mcnemar_p'))})")


def _rel_cabecalho(ctx, a):
    mapa = ctx["mapa"]
    agora = time.strftime("%d/%m/%Y %H:%M")
    n_suj_npy = {f: len(v) // 3 for f, v in U.OFFICIAL_START_INDICES.items()}
    f11 = [f for f, n in n_suj_npy.items() if n < 12]
    cnt = mapa["classe"].value_counts()
    met = ctx["tabs"]["met"]
    so_of, so_no = ctx["fora_par"]["confirmado_provavel"]
    nseq_of = _m(met, SEC_TO, CONJ_P, "todos", "3c", "oficial", "n_seq")
    nseq_no = _m(met, SEC_TO, CONJ_P, "todos", "3c", f"nosso_{VARIANTE_PRINCIPAL}", "n_seq")
    a("# Exp. 5a — Validação do nosso extrator contra as características oficiais da UTA-RLDD (rodada final, 5 folds)")
    a("")
    a(f"Gerado por `src/experiments/exp5a_validacao.py` em {agora} (log: `{_rel(ctx['log_path'])}`). Nenhum número deste "
      "arquivo foi digitado à mão: tudo vem dos CSVs da seção 9. Reproduzir, depois de `bash src/features/run_features.sh`: "
      "`bash src/experiments/run_exp5a.sh` (detector de rosto do dlib no `.venv-mp`, `src/experiments/exp5a_dlib_rosto.py`, "
      "incremental; verificação independente do mapeamento, `src/experiments/exp5a_mapeamento_indep.py`; e este script, no "
      "`.venv`). A rodada preliminar (só o Fold 1) foi descartada; seus arquivos foram removidos em 06/10/2026.")
    a("")
    a("**Pergunta.** Um modelo treinado nas características oficiais da UTA-RLDD (dlib + SVM de olho fechado, "
      "Ghoddoosian et al., 2019) mantém o desempenho quando recebe as NOSSAS características dos mesmos vídeos "
      "(MediaPipe Face Mesh + detector de piscadas por limiar, `src/features/`)? A diferença isola o custo de trocar o "
      "extrator, sem trocar de base. O Exp. 5b troca a base (DMD atuado → UTA real), mas usa o nosso extrator nas duas pontas.")
    a("")
    a(f"**Sujeitos.** Os `.npy` oficiais têm {sum(n_suj_npy.values())} sujeitos em 5 folds (" +
      ("os folds " + " e ".join(map(str, f11)) + " têm 11" if f11 else "todos com 12") + "); o artigo descreve 60. O "
      "mapeamento sujeito oficial ↔ pasta (Exp. 5, `results/exp5_uta_mapeamento_oficial.csv`) tem "
      f"{int(cnt.get('confirmado', 0))} pares confirmados, {int(cnt.get('provavel', 0))} prováveis e "
      f"{int(cnt.get('indeterminado', 0))} indeterminados. **Conjunto principal: os {ctx['n_sujeitos_conj'][CONJ_P]} "
      f"confirmados** ({len(ctx['vids_par'][CONJ_P])} vídeos; {milhar(nseq_of)} sequências oficiais e {milhar(nseq_no)} "
      f"nossas, variante `{VARIANTE_PRINCIPAL}`). Sensibilidade: confirmados + prováveis "
      f"({ctx['n_sujeitos_conj']['confirmado_provavel']} sujeitos, {len(ctx['vids_par']['confirmado_provavel'])} vídeos "
      "pareados" + (f"; {lista_pt(so_of)} fica de fora porque o nosso detector não achou nenhuma piscada nele"
                    if so_of else "") + "). Os indeterminados (Fold 3) não têm par: entram só na linha (i) com os "
      f"{sum(n_suj_npy.values())} sujeitos oficiais, que reproduz o Exp. 1.")
    a("")
    a(f"**Variante principal das nossas características: `{VARIANTE_PRINCIPAL}`** ({DESC_VARIANTE[VARIANTE_PRINCIPAL]}), "
      "fixada pela etapa de características antes deste experimento por ser a mais próxima da definição original (no "
      "`blink_video.py`, o início e o fim da piscada são extremos altos do EAR suavizado) e por ser a usada nas janelas do "
      f"Exp. 5b. As variantes `limiar` ({DESC_VARIANTE['limiar']}) e `segundos` ({DESC_VARIANTE['segundos']}) aparecem "
      "como sensibilidade. As nossas sequências usam o **último terço** do vídeo alerta como referência do z-score, como "
      f"os `.npy` oficiais (`data/features/final/uta/sequences.npz`: terço = `{ctx['terco']}`).")
    a("")
    a("**Estatística.** A unidade de inferência é o sujeito. Cada sujeito é avaliado pelo modelo do seu fold, e as "
      "métricas agregam todos os sujeitos do conjunto (matrizes de confusão somadas; VA = vídeos certos / vídeos). "
      f"Incerteza: IC 95% por bootstrap sobre sujeitos ({milhar(ctx['n_boot'])} reamostragens, semente {SEED}, a mesma "
      "reamostragem para (i) e (ii), percentis 2,5 e 97,5). Teste pareado: permutação com troca aleatória dos lados (i) e "
      f"(ii) dentro de cada sujeito ({milhar(ctx['n_perm'])} sorteios, semente {SEED}, bicaudal). Para a VA, também "
      f"McNemar exato nos vídeos pareados. Sensibilidade à semente do RF: {len(ctx['sementes'])} sementes "
      f"({ctx['sementes'][0]} a {ctx['sementes'][-1]}).")
    a("")


def _pastas_ausentes(ctx):
    """Pastas sem sujeito oficial: nos folds em que o nº de sujeitos oficiais é menor que o de pastas e não há
    indeterminados, as pastas que nenhum sujeito oficial ocupa."""
    mapa, vlog = ctx["mapa"], ctx["vlog"]
    out = {}
    for k in range(1, U.N_FOLDS + 1):
        pastas = sorted(vlog.loc[vlog["fold"] == k, "sujeito"].unique())
        n_of = len(U.OFFICIAL_START_INDICES[k]) // 3
        mk = mapa[mapa["fold"] == k]
        if n_of < len(pastas) and not (mk["classe"] == "indeterminado").any():
            out[k] = sorted(set(pastas) - set(mk["pasta"]))
    return out


def _info_ausentes(ctx):
    """Para cada pasta ausente: propriedades de cada arquivo contra a faixa dos arquivos que os autores processaram
    (todas as outras pastas dos 5 folds) e a taxa de detecção do dlib."""
    diag = ctx.get("diag")
    aus = _pastas_ausentes(ctx)
    if diag is None:
        return aus, {}
    d = diag.copy()
    d["sujeito"] = d["sujeito"].astype(str)
    todas_aus = [p for L in aus.values() for p in L]
    proc = d[~d["sujeito"].isin(todas_aus)]
    props = [("Resolução original (L × A)", None), ("Rotação (metadado)", "rotation_meta"), ("Codec", "codec"),
             ("Contêiner", "conteiner"), ("Tamanho do arquivo (MB)", "tamanho_mb"), ("fps", "fps"),
             ("Duração (s)", "dur_s"), ("Quadros", "n_quadros"), ("Rosto detectado (MediaPipe)", "taxa_rosto"),
             ("Altura do rosto a 450 px de largura (px)", "altura_rosto_px_450"),
             ("Distância interocular a 450 px (px)", "distancia_interocular_px_450"), ("Brilho médio (0–255)", "brilho_medio"),
             ("Rosto achado pelo dlib (pré-processamento dos autores)", "taxa_dlib_sem_ampliacao"),
             ("Idem, com 1 ampliação", "taxa_dlib_com_1_ampliacao"),
             ("Largura da caixa do dlib (px, mediana)", "largura_caixa_mediana_px"),
             ("Nossas piscadas", "n_piscadas"), ("Nossas piscadas/min", "taxa_piscadas_min"),
             ("Quedas do EAR ≥ 25%/min", "vales25_por_min")]
    fmt = {"rotation_meta": lambda x: f"{int(x)}°", "fps": lambda x: br(x, 2), "dur_s": lambda x: br(x, 0),
           "n_quadros": lambda x: milhar(x), "taxa_rosto": lambda x: pct(x, 1), "tamanho_mb": lambda x: br(x, 0),
           "altura_rosto_px_450": lambda x: br(x, 0), "distancia_interocular_px_450": lambda x: br(x, 0),
           "brilho_medio": lambda x: br(x, 0), "n_piscadas": lambda x: milhar(x),
           "taxa_piscadas_min": lambda x: br(x, 1), "vales25_por_min": lambda x: br(x, 1),
           "taxa_dlib_sem_ampliacao": lambda x: pct(x, 1), "taxa_dlib_com_1_ampliacao": lambda x: pct(x, 1),
           "largura_caixa_mediana_px": lambda x: br(x, 0)}
    info = {}
    for k, L in aus.items():
        for sp_ in L:
            vs = d[d["sujeito"] == sp_].sort_values("id", key=lambda c: c.map(
                lambda i: (int(i.split("_")[2]), i)))
            linhas, fora = [], []
            for nome, col in props:
                if col is None:
                    vals = [f"{int(r.width_orig)} × {int(r.height_orig)}" for r in vs.itertuples()]
                    uni = sorted({f"{int(w)} × {int(h)}" for w, h in zip(proc["width_orig"], proc["height_orig"])})
                    px = proc["width_orig"] * proc["height_orig"]
                    menor, maior = proc.loc[px.idxmin()], proc.loc[px.idxmax()]
                    fora_l = [not (px.min() * (1 - TOL_FAIXA) <= int(r.width_orig) * int(r.height_orig) <= px.max() * (1 + TOL_FAIXA))
                              for r in vs.itertuples()]
                    faixa = (f"{len(uni)} resoluções distintas; pixels por quadro de {int(menor['width_orig'])} × "
                             f"{int(menor['height_orig'])} ({menor['id']}) a {int(maior['width_orig'])} × "
                             f"{int(maior['height_orig'])}")
                elif col in ("codec", "conteiner"):
                    vals = [str(x) for x in vs[col]]
                    uni = sorted(set(proc[col].astype(str)))
                    fora_l = [v not in uni for v in vals]
                    faixa = ", ".join(uni)
                elif col == "rotation_meta":
                    vals = [fmt[col](x) for x in vs[col]]
                    uni = sorted(set(int(x) for x in proc[col]))
                    fora_l = [int(x) not in uni for x in vs[col]]
                    faixa = ", ".join(f"{u}°" for u in uni)
                else:
                    if col not in vs.columns or vs[col].isna().all() or proc[col].isna().all():
                        continue
                    f_ = fmt[col]
                    lo_, hi_ = float(proc[col].min()), float(proc[col].max())
                    vals = [f_(x) for x in vs[col]]
                    fora_l = [not (lo_ * (1 - TOL_FAIXA) <= float(x) <= hi_ * (1 + TOL_FAIXA)) for x in vs[col]]
                    faixa = f"{f_(lo_)} – {f_(hi_)}"
                    for (i_, x), fl in zip(vs[col].items(), fora_l):
                        if fl:
                            acima = float(x) > hi_
                            lim = hi_ if acima else lo_
                            fora.append(dict(prop=nome, col=col, arquivo=vs.loc[i_, "id"], valor=f_(x), limite=f_(lim),
                                             acima=acima, desvio_pct=100 * abs(float(x) - lim) / abs(lim) if lim else np.nan))
                if col is None or col in ("codec", "conteiner", "rotation_meta"):
                    for i_, (v, fl) in enumerate(zip(vals, fora_l)):
                        if fl:
                            lim = (f"{int(menor['width_orig'])} × {int(menor['height_orig'])}" if col is None else None)
                            fora.append(dict(prop=nome, col=col or "resolucao", arquivo=vs["id"].iloc[i_], valor=v,
                                             limite=lim, acima=False if col is None else None, desvio_pct=np.nan))
                linhas.append((nome, vals, faixa, fora_l))
            dl = vs["taxa_dlib_sem_ampliacao"] if "taxa_dlib_sem_ampliacao" in vs else pd.Series(dtype=float)
            info[sp_] = dict(fold=k, arquivos=vs["id"].tolist(), linhas=linhas, fora=fora,
                             n_proc=len(proc), n_proc_pastas=proc["sujeito"].nunique(),
                             dlib=dict(zip(vs["id"], dl)), dlib1=dict(zip(vs["id"], vs.get("taxa_dlib_com_1_ampliacao", dl))),
                             dlib_proc_min=float(proc["taxa_dlib_sem_ampliacao"].min()) if dl.notna().any() else np.nan,
                             dlib_proc_p5=float(proc["taxa_dlib_sem_ampliacao"].quantile(0.05)) if dl.notna().any() else np.nan,
                             dlib_proc_med=float(proc["taxa_dlib_sem_ampliacao"].median()) if dl.notna().any() else np.nan,
                             dlib_proc_arq_min=str(proc.loc[proc["taxa_dlib_sem_ampliacao"].idxmin(), "id"]) if dl.notna().any() else "",
                             dlib1_proc_min=float(proc["taxa_dlib_com_1_ampliacao"].min()) if dl.notna().any() else np.nan,
                             dlib1_proc_arq_min=str(proc.loc[proc["taxa_dlib_com_1_ampliacao"].idxmin(), "id"]) if dl.notna().any() else "",
                             caixa=dict(zip(vs["id"], vs.get("largura_caixa_mediana_px", dl))),
                             mp=dict(zip(vs["id"], vs["taxa_rosto"])),
                             posicao=("primeira" if sp_ == sorted(ctx["vlog"].loc[ctx["vlog"]["fold"] == k, "sujeito"].unique())[0]
                                      else "última" if sp_ == sorted(ctx["vlog"].loc[ctx["vlog"]["fold"] == k, "sujeito"].unique())[-1]
                                      else "intermediária"))
    return aus, info


def _txt_variantes(custo):
    """Frase sobre as variantes com bordas no limiar (3c), escrita a partir das linhas de custo."""
    VP = VARIANTE_PRINCIPAL
    base = {m: _c(custo, SEC_TO, CONJ_P, "todos", "3c", VP, m) for m in ("bal_acc", "va")}
    maior, excl, rej = [], [], []
    for var in ("limiar", "segundos"):
        for m in ("bal_acc", "va"):
            r = _c(custo, SEC_TO, CONJ_P, "todos", "3c", var, m)
            if r is None:
                continue
            nome = f"`{var}` {MET_NOME[m]}"
            if r["diferenca"] > base[m]["diferenca"]:
                maior.append(nome)
            if _exclui_zero(r):
                excl.append(nome)
                if r.get("perm_p", 1) < 0.05:
                    rej.append(nome)
    txt = (f"O custo com as bordas no limiar é maior que o da variante `{VP}` em {len(maior)} de 4 comparações. " if maior else "")
    if excl:
        txt += (f"O IC por bootstrap exclui zero em {lista_pt(excl)}; " +
                (f"a permutação rejeita a igualdade a 5% em {lista_pt(rej)}." if rej else
                 "a permutação não rejeita a igualdade a 5% em nenhum desses casos (resultado no limite)."))
    return txt


def _rel_numeros(ctx, a):
    t = ctx["tabs"]
    met, custo, dist = t["met"], t["custo"], t["dist"]
    VP = VARIANTE_PRINCIPAL
    nv = {e: int(_m(met, SEC_TO, CONJ_P, "todos", e, "oficial", "n_videos")) for e in ESQUEMAS}
    a("## 1. Números-chave")
    a("")
    for esq, tit in (("3c", "3 classes"), ("B3", "alerta × sonolento (B3)")):
        o = {m: _m(met, SEC_TO, CONJ_P, "todos", esq, "oficial", m) for m in METRICAS}
        n = {m: _m(met, SEC_TO, CONJ_P, "todos", esq, f"nosso_{VP}", m) for m in METRICAS}
        cb, cv = _c(custo, SEC_TO, CONJ_P, "todos", esq, VP, "bal_acc"), _c(custo, SEC_TO, CONJ_P, "todos", esq, VP, "va")
        cb1, cv1 = _c(custo, SEC_TO, CONJ_P, 1, esq, VP, "bal_acc"), _c(custo, SEC_TO, CONJ_P, 1, esq, VP, "va")
        a(f"- **Custo de trocar o extrator, {tit} ({ctx['n_sujeitos_conj'][CONJ_P]} sujeitos, {nv[esq]} vídeos).** RF do "
          "Exp. 1 treinado nas oficiais dos outros 4 folds: (i) oficial — acurácia balanceada "
          f"{pct(o['bal_acc'])}, VA {pct(o['va'])}; (ii) nosso `{VP}` — {pct(n['bal_acc'])}, {pct(n['va'])}. **(i) − (ii): "
          f"{_custo_frase(custo, SEC_TO, CONJ_P, esq, VP, nv[esq])}.** "
          f"{_maiuscula(_txt_ic_zero(cb, 'acurácia balanceada'))} e {_txt_ic_zero(cv, 'VA')}. Só com o Fold 1 "
          f"({int(cb1['n_sujeitos'])} sujeitos, mesmas características e mesma análise), a meia-largura do IC era "
          f"{br(100 * _meia(cb1), 1)} pp (acur. bal.) e {br(100 * _meia(cv1), 1)} pp (VA); com os 5 folds, "
          f"{br(100 * _meia(cb), 1)} e {br(100 * _meia(cv), 1)} pp.")
    cl3, cl3v = _c(custo, SEC_TO, CONJ_P, "todos", "3c", "limiar", "bal_acc"), _c(custo, SEC_TO, CONJ_P, "todos", "3c", "limiar", "va")
    cs3, cs3v = _c(custo, SEC_TO, CONJ_P, "todos", "3c", "segundos", "bal_acc"), _c(custo, SEC_TO, CONJ_P, "todos", "3c", "segundos", "va")
    a(f"- **Outras variantes (3 classes).** Bordas no limiar (`limiar`): custo de {pp(cl3['diferenca'])} na acurácia "
      f"balanceada (IC {_ic(cl3)}; permutação {peq(cl3.get('perm_p'))}) e {pp(cl3v['diferenca'])} na VA (IC {_ic(cl3v)}; "
      f"{peq(cl3v.get('perm_p'))}); `segundos`: {pp(cs3['diferenca'])} (IC {_ic(cs3)}; {peq(cs3.get('perm_p'))}) e "
      f"{pp(cs3v['diferenca'])} (IC {_ic(cs3v)}; {peq(cs3v.get('perm_p'))}). " + _txt_variantes(custo))
    partes = []
    for sec in (SEC_5F, SEC_LOSO):
        if not ((met["secao"] == sec) & (met["conjunto_sujeitos"] == CONJ_P)).any():
            continue
        txt = []
        for esq, tit in (("3c", "3 classes"), ("B3", "B3")):
            o = {m: _m(met, sec, CONJ_P, "todos", esq, "oficial", m) for m in METRICAS}
            n = {m: _m(met, sec, CONJ_P, "todos", esq, f"nosso_{VP}", m) for m in METRICAS}
            cb, cv = _c(custo, sec, CONJ_P, "todos", esq, VP, "bal_acc"), _c(custo, sec, CONJ_P, "todos", esq, VP, "va")
            txt.append(f"{tit}: oficial {pct(o['bal_acc'])} / VA {pct(o['va'])} × nosso {pct(n['bal_acc'])} / "
                       f"{pct(n['va'])} (oficial − nosso: {pp(cb['diferenca'])}, IC {_ic(cb)}, {peq(cb.get('perm_p'))}; "
                       f"VA {pp(cv['diferenca'])}, IC {_ic(cv)}, {peq(cv.get('perm_p'))})")
        partes.append(f"**{SEC_NOME[sec]}** — " + "; ".join(txt))
    a("- **Mesmo extrator no treino e no teste** (sem troca, como no Exp. 5b; mesmos sujeitos; acurácia balanceada / "
      "VA). " + ". ".join(partes) + ".")
    d = dist[(dist["tipo"] == "distribuicao") & (dist["conjunto_sujeitos"] == CONJ_P) & (dist["fold"] == "todos")
             & (dist["variante"] == VP) & (dist["classe"] == "todas")].set_index("caracteristica")
    sp = dist[(dist["tipo"] == "spearman_medias_video") & (dist["conjunto_sujeitos"] == CONJ_P) & (dist["fold"] == "todos")
              & (dist["variante"] == VP) & (dist["classe"] == "todos")].set_index("caracteristica")
    a(f"- **Distribuições (`{VP}`, piscadas dos {ctx['n_sujeitos_conj'][CONJ_P]} sujeitos).** KS D com todas as classes: " +
      ", ".join(f"{CARACT_NOME[c].lower()} {br(d.loc[c, 'ks_D'], 2)}" for c in CARACT) + "; D mediano por sujeito: " +
      ", ".join(br(d.loc[c, "ks_D_mediana_por_sujeito"], 2) for c in CARACT) + f". Spearman entre as médias por vídeo "
      f"({int(sp['n_videos'].iloc[0])} vídeos): " +
      ", ".join(f"{CARACT_NOME[c].lower()} {br(sp.loc[c, 'spearman'], 2)}" for c in CARACT) + ".")
    ic_, id_ = ctx["infos_c"], ctx["infos_d"]
    indep = ctx["indep"]
    ac = sum(v["acertos"] for v in ic_.values())
    nn = sum(v["n"] for v in ic_.values())
    esp = sum(v["esperado_acaso"] for v in ic_.values())
    vv_c = sum(v["videos_viaveis_conf"] for v in id_.values())
    nv_c = sum(v["n_videos_conf"] for v in id_.values())
    viol = [(k, x) for k, v in id_.items() for x in v["violacoes"]]
    classe_pasta = {r.pasta: r.classe for r in ctx["mapa"].itertuples() if r.pasta}
    txt_indep = ""
    if len(indep):
        ii = indep[indep["pasta_adotada"] != ""]
        ic_conf = ii[ii["classe"] == "confirmado"]
        ic_prov = ii[ii["classe"] == "provavel"]
        nao = ic_prov.loc[~ic_prov["indep_coincide"], "pasta_adotada"].tolist()
        txt_indep = (f" (ii) Outro detector de vales (`exp5a_mapeamento_indep.py`): a melhor pasta é a adotada em "
                     f"{int(ic_conf['indep_coincide'].sum())}/{len(ic_conf)} confirmados e "
                     f"{int(ic_prov['indep_coincide'].sum())}/{len(ic_prov)} prováveis" +
                     (f" (não confirma {lista_pt(nao)})" if nao else "") + ".")
    a("- **Mapeamento: checagens que não usam o escore do Exp. 5.** (i) Duração: nos vídeos com série oficial confiável, "
      f"os pares confirmados respeitam NF da última piscada ≤ nº de quadros em {vv_c}/{nv_c}" +
      (("; violam: " + "; ".join(f"{p} vídeo {lab} (razão {br(r, 3)}, {CLASSE_TXT.get(classe_pasta.get(p, '?'), '?')})"
                                  for k, (s, p, lab, r) in viol)) if viol else "") + "." + txt_indep +
      f" (iii) Nº de piscadas: o húngaro pela contagem coincide com o adotado em {ac} de {nn} sujeitos (acaso: "
      f"{br(esp, 1)}): a contagem carrega alguma identidade, mas não basta para mapear (seção 2.2).")
    aus, info = ctx["ausentes"]
    if aus:
        partes = []
        for k, L in aus.items():
            for sp_ in L:
                inf = info.get(sp_)
                if not inf:
                    partes.append(f"{sp_} (Fold {k}; diagnóstico de vídeo não executado)")
                    continue
                abaixo = [(f, v) for f, v in inf["dlib1"].items() if fin(v) and v < inf["dlib1_proc_min"]]
                partes.append(f"{sp_} ({inf['posicao']} pasta do Fold {k}; " + (
                    "o dlib com o pré-processamento dos autores e 1 ampliação acha o rosto em " +
                    ", ".join(f"{pct(v)} dos quadros amostrados de {f}" for f, v in abaixo) +
                    f", abaixo de todos os {inf['n_proc']} arquivos que os autores processaram (mínimo "
                    f"{pct(inf['dlib1_proc_min'])})" if abaixo else
                    "nenhuma propriedade medida, nem o detector de rosto do dlib, a separa dos vídeos processados") + ")")
        a("- **Pastas sem sujeito oficial:** " + "; ".join(partes) + ". A ausência não segue uma posição fixa na "
          "listagem (seção 2.3).")
    cb_s = _c(custo, SEC_TO, "confirmado_provavel", "todos", "3c", VP, "bal_acc")
    cv_s = _c(custo, SEC_TO, "confirmado_provavel", "todos", "3c", VP, "va")
    cbr = _c(custo, SEC_TO, CONJ_P, "todos", "3c", VP, "bal_acc")
    cvr = _c(custo, SEC_TO, CONJ_P, "todos", "3c", VP, "va")
    exc = ctx["det"].get(("excluidos_preenchidos", "3c"), [])
    cb_p = _c(custo, SEC_TO + "_sem_preenchidos", CONJ_P, "todos", "3c", VP, "bal_acc")
    cv_p = _c(custo, SEC_TO + "_sem_preenchidos", CONJ_P, "todos", "3c", VP, "va")
    a(f"- **Sensibilidades do custo (3 classes, `{VP}`).** Confirmados + prováveis "
      f"({ctx['n_sujeitos_conj']['confirmado_provavel']} sujeitos): {pp(cb_s['diferenca'])} (IC {_ic(cb_s)}) e VA "
      f"{pp(cv_s['diferenca'])} (IC {_ic(cv_s)}). {len(ctx['sementes'])} sementes do RF: diferença média "
      f"{pp(cbr['sementes_dif_media'])} ({pp(cbr['sementes_dif_min'])} a {pp(cbr['sementes_dif_max'])}) e VA "
      f"{pp(cvr['sementes_dif_media'])} ({pp(cvr['sementes_dif_min'])} a {pp(cvr['sementes_dif_max'])}). Sem os "
      f"{len(exc)} vídeos representados por uma única sequência com zeros à esquerda: {pp(cb_p['diferenca'])} "
      f"(IC {_ic(cb_p)}) e VA {pp(cv_p['diferenca'])} (IC {_ic(cv_p)}).")
    a("")


def _rel_mapeamento(ctx, a):
    mapa, met, vlog = ctx["mapa"], ctx["tabs"]["met"], ctx["vlog"]
    ic_, id_, indep = ctx["infos_c"], ctx["infos_d"], ctx["indep"]
    a("## 2. Sujeitos e mapeamento")
    a("")
    a("### 2.1 Sujeitos por fold")
    a("")
    a("Fontes: `results/exp5_uta_mapeamento_oficial.csv` (classes do par), `data/uta_rldd_blinks/BlinksTest_30_Fold<k>.npy` "
      "e `data/features/final/uta/sequences.npz`. Vídeos e sequências da coluna principal = sujeitos confirmados, "
      "esquema 3c.")
    a("")
    a("| Fold | Sujeitos oficiais | Pastas | Confirmados | Prováveis | Indeterminados | Pastas sem par atribuído | "
      "Vídeos (principal) | Sequências oficiais / nossas (principal) |")
    a("|---:|---:|---:|---:|---:|---:|---|---:|---:|")
    for k in range(1, U.N_FOLDS + 1):
        mk = mapa[mapa["fold"] == k]
        pastas = sorted(vlog.loc[vlog["fold"] == k, "sujeito"].unique())
        sem = sorted(set(pastas) - set(mk["pasta"]))
        cnt = mk["classe"].value_counts()
        a(f"| {k} | {len(mk)} | {len(pastas)} | {int(cnt.get('confirmado', 0))} | {int(cnt.get('provavel', 0))} | "
          f"{int(cnt.get('indeterminado', 0))} | {', '.join(sem) or '—'} | "
          f"{int(_m(met, SEC_TO, CONJ_P, k, '3c', 'oficial', 'n_videos'))} | "
          f"{milhar(_m(met, SEC_TO, CONJ_P, k, '3c', 'oficial', 'n_seq'))} / "
          f"{milhar(_m(met, SEC_TO, CONJ_P, k, '3c', 'nosso_' + VARIANTE_PRINCIPAL, 'n_seq'))} |")
    cnt = mapa["classe"].value_counts()
    a(f"| **total** | {len(mapa)} | {vlog['sujeito'].nunique()} | {int(cnt.get('confirmado', 0))} | "
      f"{int(cnt.get('provavel', 0))} | {int(cnt.get('indeterminado', 0))} | | "
      f"{int(_m(met, SEC_TO, CONJ_P, 'todos', '3c', 'oficial', 'n_videos'))} | "
      f"{milhar(_m(met, SEC_TO, CONJ_P, 'todos', '3c', 'oficial', 'n_seq'))} / "
      f"{milhar(_m(met, SEC_TO, CONJ_P, 'todos', '3c', 'nosso_' + VARIANTE_PRINCIPAL, 'n_seq'))} |")
    a("")
    frases = []
    for k in range(1, U.N_FOLDS + 1):
        mk = mapa[mapa["fold"] == k]
        n_ind = int((mk["classe"] == "indeterminado").sum())
        pastas = sorted(vlog.loc[vlog["fold"] == k, "sujeito"].unique())
        if n_ind and len(mk) == len(pastas):
            frases.append(f"No Fold {k} as {len(set(pastas) - set(mk['pasta']))} pastas sem par não são ausentes: os "
                          f"autores processaram {len(mk)} sujeitos (tantos quanto as pastas), mas {n_ind} deles ficaram "
                          "indeterminados no Exp. 5 (arquivo oficial com defeitos; `results/exp5_caracteristicas.md`, "
                          "seção 5.4).")
    a(" ".join(frases) + " Ausentes de fato (fold com menos sujeitos oficiais que pastas) são só as da seção 2.3.")
    a("")
    a("### 2.2 Checagens do mapeamento que não usam o escore do Exp. 5")
    a("")
    a("O mapeamento vem do Exp. 5 (coincidência dos instantes das piscadas oficiais, recuperados da Frequency gravada "
      "com 4 casas, com os vales do nosso EAR). Três checagens independentes dele, por fold, para os sujeitos oficiais "
      "com pasta adotada (confirmados e prováveis) contra todas as pastas do fold:")
    a("")
    a("- **Contagem de piscadas** (algoritmo húngaro sobre o erro relativo médio |n_nosso − n_oficial| / n_oficial nos "
      "3 vídeos; o nº oficial vem da aritmética de `unroll_in_time`, n = 30 + 2(N − 1) para N ≥ 2 sequências, conferida "
      "contra a série reconstruída). Acaso: atribuição injetiva uniforme, P(≥ acertos) exata por inclusão-exclusão.")
    a("- **Duração** (sem EAR): o NF da última piscada oficial (quadros com rosto no dlib) não pode passar do nº de quadros "
      "do vídeo lógico da pasta (partes somadas). Só entram as séries oficiais confiáveis (Frequency recuperada com NF "
      "inteiro; coluna `series_confiaveis` do mapeamento): nas outras o NF recuperado não tem sentido.")
    a("- **Outro detector de vales** (`src/experiments/exp5a_mapeamento_indep.py`, implementação independente da rodada "
      "preliminar): EAR ÷ mediana móvel, `find_peaks` com proeminência 0,10, defasagem local por piscada; não reutiliza "
      "nada de `src/features/uta_oficial.py`.")
    a("")
    a("| Fold | Contagem: coincide com o adotado (acaso esperado; P) | Aritmética confere | Duração: vídeos confiáveis dentro "
      "do limite (confirmados) | Atribuições completas viáveis pela duração | Outro detector: melhor = adotada "
      "(confirmados; prováveis) |")
    a("|---:|---|---:|---|---|---|")
    for k in range(1, U.N_FOLDS + 1):
        c, d = ic_[k], id_[k]
        ii = indep[(indep["fold"] == k) & (indep["pasta_adotada"] != "")] if len(indep) else pd.DataFrame()
        cc = ii[ii["classe"] == "confirmado"] if len(ii) else ii
        pv = ii[ii["classe"] == "provavel"] if len(ii) else ii
        a(f"| {k} | {c['acertos']}/{c['n']} ({br(c['esperado_acaso'], 2)}; {pval(c['p_acaso'])}) | "
          f"{c['aritmetica_ok']}/{c['aritmetica_n']} | {d['videos_viaveis']}/{d['n_videos']} "
          f"({d['videos_viaveis_conf']}/{d['n_videos_conf']}) | {milhar(d['n_atrib_viaveis'])} de {milhar(d['n_atrib_total'])} | "
          + (f"{int(cc['indep_coincide'].sum())}/{len(cc)}; {int(pv['indep_coincide'].sum())}/{len(pv)}" if len(ii) else "—") + " |")
    ac = sum(v["acertos"] for v in ic_.values())
    nn = sum(v["n"] for v in ic_.values())
    esp = sum(v["esperado_acaso"] for v in ic_.values())
    a(f"| **total** | {ac}/{nn} ({br(esp, 1)}) | {sum(v['aritmetica_ok'] for v in ic_.values())}/"
      f"{sum(v['aritmetica_n'] for v in ic_.values())} | {sum(v['videos_viaveis'] for v in id_.values())}/"
      f"{sum(v['n_videos'] for v in id_.values())} ({sum(v['videos_viaveis_conf'] for v in id_.values())}/"
      f"{sum(v['n_videos_conf'] for v in id_.values())}) | | " +
      (f"{int(indep[(indep['classe'] == 'confirmado')]['indep_coincide'].sum())}/{int((indep['classe'] == 'confirmado').sum())}; "
       f"{int(indep[(indep['classe'] == 'provavel')]['indep_coincide'].sum())}/{int((indep['classe'] == 'provavel').sum())}"
       if len(indep) else "—") + " |")
    a("")
    viol = [(k, x) for k, v in id_.items() for x in v["violacoes"]]
    max_conf = max(v["max_razao_conf"] for v in id_.values() if fin(v["max_razao_conf"]))
    a(f"**Contagem.** O húngaro pela contagem coincide com o adotado em {ac} de {nn} sujeitos, contra {br(esp, 1)} "
      "esperados ao acaso. A contagem tem, portanto, alguma informação de identidade (mais nos folds " +
      lista_pt([str(k) for k in ic_ if ic_[k]["p_acaso"] < 0.01]) + "), mas erra o par em " +
      f"{nn - ac} dos {nn} sujeitos: os dois detectores contam piscadas de forma diferente demais para a contagem, "
      "sozinha, mapear (tabela por sujeito em `results/exp5a_validacao_mapeamento.csv`). No Fold 1, o único da rodada "
      f"preliminar, a coincidência é {ic_[1]['acertos']} de {ic_[1]['n']}.")
    a("")
    a(f"**Duração.** Nos vídeos com série confiável, a maior razão NF/quadros entre os pares confirmados é "
      f"{br(max_conf, 4)}" + (". Violações (pares não confirmados): " + "; ".join(
          f"Fold {k}, sujeito oficial {s} → {p}, vídeo {lab}: {br(r, 3)}" for k, (s, p, lab, r) in viol)
          if viol else "; nenhum par adotado viola o limite") + ". Atribuições completas (sujeitos com pasta adotada → "
      "pastas do fold) que respeitam todos os limites: " + "; ".join(
          f"Fold {k}: {milhar(d['n_atrib_viaveis'])} de {milhar(d['n_atrib_total'])}" for k, d in id_.items()) +
      ". A duração restringe muito, mas não identifica sozinha, porque muitos vídeos têm durações parecidas." +
      "".join(_txt_zero_atrib(ctx, k) for k, d in id_.items() if d["n_atrib_viaveis"] == 0))
    a("")
    if len(indep):
        ii = indep[indep["pasta_adotada"] != ""]
        cc = ii[ii["classe"] == "confirmado"].sort_values("indep_margem")
        w = cc.iloc[0]
        dono = ii.loc[ii["pasta_adotada"] == w["indep_melhor_alternativa"]]
        dono_txt = (f", que é a pasta adotada do sujeito oficial {int(dono['sujeito_oficial'].iloc[0])} "
                    f"({CLASSE_TXT.get(dono['classe'].iloc[0], dono['classe'].iloc[0])}, escore {br(dono['indep_escore_adotada'].iloc[0], 3)} lá)"
                    if len(dono) else "")
        pv = ii[(ii["classe"] == "provavel")]
        nao = pv[~pv["indep_coincide"]]
        ind = indep[indep["pasta_adotada"] == ""]
        livres = set()
        for k in ind["fold"].unique():
            pastas = set(vlog.loc[vlog["fold"] == k, "sujeito"])
            livres |= pastas - set(mapa.loc[mapa["fold"] == k, "pasta"])
        sug = ind[ind["indep_melhor"].isin(livres) & (ind["indep_escore_melhor"] >= 0.3)]
        a(f"**Outro detector de vales.** A melhor pasta é a adotada em {int(cc['indep_coincide'].sum())} de {len(cc)} "
          f"confirmados. A menor margem entre eles é a do sujeito oficial {int(w['sujeito_oficial'])} do Fold "
          f"{int(w['fold'])} ({w['pasta_adotada']}: {br(w['indep_escore_adotada'], 3)} contra "
          f"{br(w['indep_escore_alternativa'], 3)} de {w['indep_melhor_alternativa']}{dono_txt}). Entre os prováveis, "
          f"{int(pv['indep_coincide'].sum())} de {len(pv)} coincidem" +
          ("; não coincidem " + "; ".join(f"{r.pasta_adotada} (Fold {int(r.fold)}, oficial {int(r.sujeito_oficial)}: "
                                         f"adotada {br(r.indep_escore_adotada, 3)}, melhor {r.indep_melhor} "
                                         f"{br(r.indep_escore_melhor, 3)})" for r in nao.itertuples()) if len(nao) else "") +
          ". Por isso o principal usa só os confirmados." +
          (" **Observação para a etapa de características** (não usada aqui): entre os indeterminados, " + "; ".join(
              f"o sujeito oficial {int(r.sujeito_oficial)} do Fold {int(r.fold)} tem a melhor pasta em {r.indep_melhor} "
              f"({br(r.indep_escore_melhor, 3)} contra {br(r.indep_escore_alternativa, 3)} da 2ª, {r.indep_melhor_alternativa}), "
              "pasta ainda livre" for r in sug.itertuples()) + "." if len(sug) else ""))
        a("")
    _rel_ausentes(ctx, a)


def _txt_zero_atrib(ctx, k):
    mt = ctx["tabs"]["mapeamento"]
    mt = mt[(mt["fold"] == k) & (mt["pasta"] != "")]
    zero = mt[mt["n_pastas_viaveis_duracao"] == 0]
    if len(zero):
        return (f" No Fold {k} nenhuma atribuição completa respeita todos os limites: " + "; ".join(
            f"o sujeito oficial {int(r.sujeito_oficial)} (pasta adotada {r.pasta}, {r.classe}) não cabe em nenhuma pasta "
            "do fold" for r in zero.itertuples()) + ".")
    return (f" No Fold {k} nenhuma atribuição completa respeita todos os limites ao mesmo tempo (as pastas compatíveis "
            "não bastam para todos os sujeitos).")


def _quadros_dlib():
    d = pd.read_csv(DLIB_CSV) if DLIB_CSV.exists() else pd.DataFrame()
    if not len(d):
        return "—"
    lo, hi = int(d["quadros_testados"].min()), int(d["quadros_testados"].max())
    return str(hi) if lo == hi else f"{lo} a {hi}"


def _rel_ausentes(ctx, a):
    aus, info = ctx["ausentes"]
    a("### 2.3 Pastas sem sujeito oficial: " + (", ".join(p for L in aus.values() for p in L) or "nenhuma"))
    a("")
    if not aus:
        a("Todos os folds têm tantos sujeitos oficiais quanto pastas.")
        a("")
        return
    if not info:
        a("Diagnóstico de vídeo não executado (`--sem-diagnostico-video`).")
        a("")
        return
    a("Faixa de referência = todos os arquivos das outras pastas dos 5 folds, isto é, os vídeos que os autores "
      "processaram (inclusive os dos sujeitos indeterminados, que estão no arquivo oficial). Fontes: "
      "`results/exp5a_videos_diagnostico.csv` (codec e contêiner do arquivo; brilho de 12 quadros espaçados; altura do rosto "
      "= pontos 10→152 do Face Mesh e distância interocular = 33→263, medianas, na escala do quadro exibido reduzido a "
      "450 px de largura, que é o `imutils.resize(frame, width=450)` do código original), `results/exp5a_dlib_rosto.csv` "
      f"(detector do dlib em {_quadros_dlib()} quadros amostrados por arquivo, com o pré-processamento do `blink_video.py`: 450 px de "
      "largura, cinza, gama 1,5, `detector(gray, 0)`; nos quadros sem rosto, também `detector(gray, 1)`, com 1 ampliação; "
      "`src/experiments/exp5a_dlib_rosto.py`, `.venv-mp`), "
      "`data/features/final/uta/blinks_index.csv` e `data/signals/uta/index.csv`.")
    a("")
    for k, L in aus.items():
        for sp_ in L:
            inf = info[sp_]
            a(f"**{sp_}** ({inf['posicao']} pasta do Fold {k}).")
            a("")
            a("| Propriedade | " + " | ".join(inf["arquivos"]) + f" | Faixa nos {inf['n_proc']} arquivos processados "
              "(mín – máx) | Fora da faixa |")
            a("|---|" + "---|" * len(inf["arquivos"]) + "---|---|")
            for nome, vals, faixa, fora_l in inf["linhas"]:
                fora_txt = ", ".join(f for f, fl in zip(inf["arquivos"], fora_l) if fl) or "—"
                a(f"| {nome} | " + " | ".join(vals) + f" | {faixa} | {fora_txt} |")
            a("")
    # leitura
    diag = ctx["diag"].copy()
    diag["sujeito"] = diag["sujeito"].astype(str)
    todas_aus = [p_ for L in aus.values() for p_ in L]
    proc = diag[~diag["sujeito"].isin(todas_aus)]
    s0, s1 = "taxa_dlib_sem_ampliacao", "taxa_dlib_com_1_ampliacao"
    i0, i1 = proc[s0].idxmin(), proc[s1].idxmin()
    min0, arq0, min1, arq1 = float(proc.loc[i0, s0]), proc.loc[i0, "id"], float(proc.loc[i1, s1]), proc.loc[i1, "id"]
    # a réplica sem ampliação subestima o detector dos autores em rostos pequenos? (série oficial confiável cobrindo
    # quase todo o vídeo, num arquivo em que a réplica sem ampliação quase não acha o rosto)
    vv = pd.read_csv(VAL_VIDEOS_CSV)
    nq = dict(zip(ctx["vlog"]["video"], ctx["vlog"]["n_quadros"]))
    mp_ = ctx["mapa"].set_index("pasta")
    contra = []
    for r in proc[(proc[s0] < 0.5) & (proc["parte"].isna())].itertuples():
        if r.sujeito not in mp_.index or str(r.rotulo) == "0":
            continue
        conf = str(mp_.loc[r.sujeito, "series_confiaveis"]).split(";")[(0, 5, 10).index(int(r.rotulo))] == "1"
        q = vv[vv["video"] == r.id]
        if conf and len(q):
            contra.append((r.id, float(getattr(r, s0)), float(getattr(r, s1)),
                           float(q["NF_final_oficial"].iloc[0]) / nq[r.id], mp_.loc[r.sujeito, "classe"],
                           float(r.largura_caixa_mediana_px)))
    frases = []
    txt_rep = (f"No conjunto dos {len(proc)} arquivos processados, a réplica sem ampliação chega a {pct(min0)} ({arq0}), e "
               f"a com 1 ampliação a {pct(min1)} ({arq1}). ")
    if contra:
        txt_rep += ("A réplica sem ampliação subestima o detector efetivo dos autores em rostos pequenos: " + "; ".join(
            f"em {f} (par {CLASSE_TXT.get(cl, cl)}; caixa mediana de {br(cx, 0)} px) ela acha o rosto em {pct(t0_)}, mas a série oficial "
            f"confiável vai até NF/quadros = {br(rz, 3)}, e com 1 ampliação a réplica sobe para {pct(t1_)}"
            for f, t0_, t1_, rz, cl, cx in contra) + ". A taxa com 1 ampliação é, portanto, o termo de comparação mais "
            "justo.")
    frases.append("**Detector de rosto do dlib.** " + txt_rep)
    for k, L in aus.items():
        for sp_ in L:
            inf = info[sp_]
            fora = [f for f in inf["fora"] if f["col"] not in (s0, s1)]
            abaixo = [(f, v) for f, v in inf["dlib1"].items() if fin(v) and v < min1]
            txt = (f"**{sp_}.** " + (f"Fora da faixa dos processados (tolerância de {br(100 * TOL_FAIXA, 1)}%): " + "; ".join(
                f"{f['prop'].split(' (')[0].lower()} de {f['arquivo']} = {f['valor']}" +
                (f" ({'máximo' if f['acima'] else 'mínimo'} dos processados {f['limite']})" if f["limite"] else "")
                for f in fora) + ". " if fora else "Nenhuma propriedade de vídeo medida fica fora da faixa dos "
                f"processados (tolerância de {br(100 * TOL_FAIXA, 1)}%). "))
            txt += ("Detector do dlib (sem / com 1 ampliação): " + "; ".join(
                f"{f} {pct(inf['dlib'][f])} / {pct(inf['dlib1'][f])}" for f in inf["arquivos"]) + ". ")
            if abaixo:
                fpior, vpior = min(abaixo, key=lambda fv: fv[1])
                alerta = fpior.split("_")[2] == "0"
                txt += (f"**Com 1 ampliação, {lista_pt([f for f, _ in abaixo])} "
                        f"{'fica' if len(abaixo) == 1 else 'ficam'} abaixo de todos os arquivos processados** (mínimo "
                        f"{pct(min1)}, {arq1})" +
                        "".join(f"; em {f} a caixa, quando achada, tem mediana de {br(inf['caixa'][f], 0)} px e o "
                                f"MediaPipe acha o rosto em {pct(inf['mp'][f])} dos quadros, então não é rosto pequeno"
                                for f, _ in abaixo if fin(inf["caixa"].get(f)) and inf["caixa"][f] >= 100) +
                        f". É a explicação candidata mais direta para a ausência: com o rosto achado em só {pct(vpior)} "
                        f"dos quadros amostrados de {fpior}" +
                        (" — o vídeo alerta, cujo último terço é a referência do z-score de todo o sujeito —" if alerta else "") +
                        ", o pré-processamento dos autores teria poucas piscadas nele (hipótese: o código publicado não "
                        "registra por que um sujeito ficou de fora). ")
            else:
                txt += (f"Com 1 ampliação, nenhum arquivo de {sp_} fica abaixo do mínimo dos processados: o detector de "
                        "rosto dos autores não explica a ausência. ")
            frases.append(txt)
    for f in frases:
        a(f)
        a("")
    pos = [(sp_, info[sp_]["posicao"], k) for k, L in aus.items() for sp_ in L if sp_ in info]
    divididos = sorted(ctx["vlog"].loc[ctx["vlog"]["n_arquivos"] > 1, "video"])
    div_txt = []
    for v in divididos:
        suj = sujeito_de_video(v)
        cl = ctx["mapa"].loc[ctx["mapa"]["pasta"] == suj, "classe"]
        div_txt.append(f"{v} ({'no arquivo oficial, par ' + CLASSE_TXT.get(cl.iloc[0], cl.iloc[0]) if len(cl) else 'pasta sem sujeito oficial'})")
    a("**Posição e vídeos divididos.** " + "; ".join(f"{s} é a {p} pasta do Fold {k}" for s, p, k in pos) +
      ": a ausência não é um corte fixo no fim (ou no início) da listagem, como a rodada preliminar cogitou com só o "
      "Fold 1. Vídeos lógicos divididos em duas partes: " + "; ".join(div_txt) + ", então a divisão em si não impediu o "
      "processamento pelos autores. O artigo descreve 60 participantes em 5 folds de 12, e o repositório "
      "(github.com/rezaghoddoosian/Early-Drowsiness-Detection, consultado em 27/09/2026) não explica a diferença; na issue "
      "#2, sem resposta, um terceiro relata que o próprio código dos autores gera saídas diferentes dos `.npy` publicados.")
    a("")


def _nome_teste(conj, conjunto, n_suj):
    if conjunto == "oficial":
        return ("(i) Oficial, todos os sujeitos oficiais (reproduz o Exp. 1)" if conj == "todos_oficiais"
                else f"(i) Oficial, {n_suj} {CONJ_NOME.get(conj, conj)}")
    var = conjunto.replace("nosso_", "")
    if conj == "todas_as_pastas":
        return f"(ii) Nosso `{var}`, todas as pastas (contexto, sem pareamento)"
    return f"(ii) Nosso `{var}`, {n_suj} {CONJ_NOME.get(conj, conj)}"


def _tab_metricas(ctx, a, secao, esquema, linhas_def):
    met, det = ctx["tabs"]["met"], ctx["det"]
    a("| Conjunto de teste | Sujeitos | Vídeos | Sequências por classe | BSA | Acur. bal. | F1 macro | AUC macro | VA |")
    a("|---|---:|---:|---|---:|---:|---:|---:|---:|")
    for conj, conjunto in linhas_def:
        r = det.get((secao, conj, esquema, conjunto))
        if r is None:
            continue
        a(f"| {_nome_teste(conj, conjunto, r['n_sujeitos'])} | {r['n_sujeitos']} | {r['n_videos']} | "
          f"{' / '.join(milhar(x) for x in r['classes_seq'])} | " +
          " | ".join(pct(_m(met, secao, conj, 'todos', esquema, conjunto, m)) for m in METRICAS) + " |")
    a("")


def _tab_custo(ctx, a, secao, conj, esquema, variantes, com_sementes=True):
    custo = ctx["tabs"]["custo"]
    cab = ("| Variante | Métrica | (i) Oficial | (ii) Nosso | (i) − (ii) | IC 95% da diferença | Permutação p | "
           "Sujeitos: (i) > (ii) / (i) < (ii) / empate")
    if com_sementes:
        cab += f" | {len(ctx['sementes'])} sementes: média (mín; máx) | Sementes com (i) > (ii)"
    a(cab + " |")
    a("|---|---|---:|---:|---:|---:|---:|---:|" + ("---:|---:|" if com_sementes else ""))
    for var in variantes:
        for m in METRICAS:
            r = _c(custo, secao, conj, "todos", esquema, var, m)
            if r is None:
                continue
            suj = (f"{int(r['suj_oficial_maior'])} / {int(r['suj_nosso_maior'])} / {int(r['suj_empate'])}"
                   if fin(r.get("suj_oficial_maior")) else "—")
            lin = (f"| `{var}` | {MET_NOME[m]} | {pct(r['oficial'])} | {pct(r['nosso'])} | **{pp(r['diferenca'])}** | "
                   f"{_ic(r)} | {pval(r.get('perm_p'))} | {suj}")
            if com_sementes:
                lin += (f" | {pp(r['sementes_dif_media'])} ({pp(r['sementes_dif_min'])}; {pp(r['sementes_dif_max'])}) | "
                        f"{int(r['sementes_dif_positiva'])}/{int(r['sementes_n'])}" if fin(r.get("sementes_n")) else " | — | —")
            a(lin + " |")
    a("")
    a("McNemar exato por vídeo (VA): " + "; ".join(
        f"`{var}`: só o oficial acerta {int(r['so_oficial_acerta'])}, só o nosso acerta {int(r['so_nosso_acerta'])}, "
        f"ambos {int(r['ambos_acertam'])}, nenhum {int(r['nenhum_acerta'])}; {peq(r['mcnemar_p'])}"
        for var in variantes for r in [_c(custo, secao, conj, "todos", esquema, var, "va")] if r is not None) + ".")
    a("")


def _tab_por_fold(ctx, a, secao, conj, esquema, var):
    custo = ctx["tabs"]["custo"]
    a(f"| Fold | Sujeitos | Vídeos | Oficial: acur. bal. | Nosso: acur. bal. | Oficial − nosso [IC 95%] | Oficial: VA | "
      "Nosso: VA | Oficial − nosso [IC 95%] |")
    a("|---:|---:|---:|---:|---:|---|---:|---:|---|")
    for fold in list(range(1, U.N_FOLDS + 1)) + ["todos"]:
        cb, cv = _c(custo, secao, conj, fold, esquema, var, "bal_acc"), _c(custo, secao, conj, fold, esquema, var, "va")
        if cb is None:
            continue
        rot = "**5 folds**" if fold == "todos" else str(fold)
        a(f"| {rot} | {int(cb['n_sujeitos'])} | {int(cb['n_videos'])} | {pct(cb['oficial'])} | {pct(cb['nosso'])} | "
          f"{pp(cb['diferenca'])} {_ic(cb)} | {pct(cv['oficial'])} | {pct(cv['nosso'])} | {pp(cv['diferenca'])} {_ic(cv)} |")
    a("")


def _rel_custo(ctx, a):
    met, custo, det = ctx["tabs"]["met"], ctx["tabs"]["custo"], ctx["det"]
    VP = VARIANTE_PRINCIPAL
    treino = ctx["treino"]
    a("## 3. Custo de trocar o extrator: RandomForest treinado nas características oficiais dos outros folds")
    a("")
    a("Para cada fold k = 1…5, o treino é `Blinks_30_Fold<k>.npy` / `Labels_30_Fold<k>.npy` (sequências oficiais dos "
      "sujeitos dos outros 4 folds: " + ", ".join(f"fold {k} {milhar(len(treino[k]['y']))} sequências de "
                                                    f"{treino[k]['n_sujeitos']} sujeitos" for k in treino) +
      "); nenhum sujeito do fold k entra no treino do seu modelo. Modelo igual ao do Exp. 1: "
      f"`RandomForestClassifier(n_estimators={N_ARVORES}, random_state={SEED})`, demais hiperparâmetros no padrão do "
      f"scikit-learn {sklearn.__version__}, n_jobs = {N_JOBS} (o resultado não depende disso). Entrada: as estatísticas "
      "24d de `sequences_to_features` (média, desvio, mínimo, máximo, mediana e IQR de cada característica). VA: voto "
      "majoritário das sequências do vídeo, empate resolvido pela maior probabilidade média. O mesmo modelo avalia (i) as "
      "sequências oficiais e (ii) as nossas do fold k; os resultados dos 5 folds são agregados por sujeito.")
    a("")
    for esq, num, tit in (("3c", "3.1", "3 classes"), ("B3", "3.2", "Alerta × sonolento (B3), sem a classe intermediária")):
        a(f"### {num} {tit}")
        a("")
        _tab_metricas(ctx, a, SEC_TO, esq, [("todos_oficiais", "oficial"), (CONJ_P, "oficial")] +
                      [(CONJ_P, f"nosso_{v}") for v in ORDEM_VAR] + [("todas_as_pastas", f"nosso_{VP}")])
        rep = [ctx["repro"].get((esq, k)) for k in range(1, U.N_FOLDS + 1)]
        e1 = pd.read_csv(EXP1_CSV) if EXP1_CSV.exists() else pd.DataFrame()
        va_e1 = (float(e1[(e1["model"] == "RandomForest") & (e1["scheme"] == esq) & (e1["metric"] == "va")]["value"].mean())
                 if len(e1) else np.nan)
        va_f = np.mean([_m(met, SEC_TO, "todos_oficiais", k, esq, "oficial", "va") for k in range(1, U.N_FOLDS + 1)])
        cs_o = _m(met, SEC_TO, CONJ_P, "todos", esq, "oficial", "chance_seq")
        cs_n = _m(met, SEC_TO, CONJ_P, "todos", esq, f"nosso_{VP}", "chance_seq")
        cv_o = _m(met, SEC_TO, CONJ_P, "todos", esq, "oficial", "chance_video")
        a(f"Chance (classe majoritária) no conjunto principal: BSA {pct(cs_o)} no oficial e {pct(cs_n)} no nosso; VA "
          f"{pct(cv_o)}; acurácia balanceada {pct(1 / U.n_classes(esq))}. " +
          (f"A linha (i) com todos os sujeitos oficiais reproduz o RandomForest do Exp. 1 fold a fold "
           "(`results/exp1_intra.csv`; maior diferença " +
           f"{max(r for r in rep if r is not None):.1e}".replace(".", ",").replace("-", "−") +
           "); a média das VA dos 5 "
           f"folds é {pct(va_f)} (Exp. 1: {pct(va_e1)}). As métricas desta seção agregam os sujeitos (e não fazem a "
           "média dos folds)." if all(r is not None for r in rep) else ""))
        a("")
        a(f"**Custo (i) − (ii), pareado nos {ctx['n_sujeitos_conj'][CONJ_P]} sujeitos confirmados.** Valor positivo = "
          "o nosso extrator perde. A AUC não tem IC (não se recalcula das confusões por sujeito).")
        a("")
        _tab_custo(ctx, a, SEC_TO, CONJ_P, esq, ORDEM_VAR)
        a(f"**Por fold (`{VP}`; IC por bootstrap sobre os sujeitos do fold).**")
        a("")
        _tab_por_fold(ctx, a, SEC_TO, CONJ_P, esq, VP)
        C_o = np.asarray(det[(SEC_TO, CONJ_P, esq, "oficial")]["confusao"])
        C_n = np.asarray(det[(SEC_TO, CONJ_P, esq, f"nosso_{VP}")]["confusao"])
        C_l = np.asarray(det[(SEC_TO, CONJ_P, esq, "nosso_limiar")]["confusao"])
        nomes = ([CLASSE_NOME[i] for i in range(3)] if esq == "3c" else ["alerta", "sonolento"])
        a(f"**Matrizes de confusão (sequências, 5 folds agregados; linhas = verdadeiro, colunas = predito: "
          f"{' / '.join(nomes)}).**")
        a("")
        a(f"| Verdadeiro | (i) Oficial | (ii) Nosso `{VP}` | (ii) Nosso `limiar` |")
        a("|---|---:|---:|---:|")
        for i, nm in enumerate(nomes):
            a(f"| {nm} | " + " / ".join(milhar(x) for x in C_o[i]) + " | " + " / ".join(milhar(x) for x in C_n[i]) +
              " | " + " / ".join(milhar(x) for x in C_l[i]) + " |")
        rec = lambda C: np.diag(C) / C.sum(1)  # noqa: E731
        a("")
        a(f"Revocação por classe ({' / '.join(nomes)}): oficial " + " / ".join(pct(x) for x in rec(C_o)) +
          f"; nosso `{VP}` " + " / ".join(pct(x) for x in rec(C_n)) + "; nosso `limiar` " +
          " / ".join(pct(x) for x in rec(C_l)) + ".")
        a("")
    a("### 3.3 Sensibilidades")
    a("")
    for esq in ESQUEMAS:
        a(f"**{esq}, confirmados + prováveis ({ctx['n_sujeitos_conj']['confirmado_provavel']} sujeitos).**")
        a("")
        _tab_custo(ctx, a, SEC_TO, "confirmado_provavel", esq, [VP, "limiar"], com_sementes=False)
    exc = ctx["det"].get(("excluidos_preenchidos", "3c"), [])
    a(f"**Sem os vídeos representados por uma única sequência com zeros à esquerda** (< 30 piscadas em algum dos lados; "
      f"exclusão pareada; {len(exc)} vídeos no 3c: {', '.join(exc)}). As estatísticas dessas sequências ficam perto de "
      "zero, como a classe alerta, e o nosso detector acha menos piscadas que o original em vários vídeos alerta.")
    a("")
    for esq in ESQUEMAS:
        a(f"*{esq}:*")
        a("")
        _tab_custo(ctx, a, SEC_TO + "_sem_preenchidos", CONJ_P, esq, ORDEM_VAR, com_sementes=False)
    if PRELIM_CUSTO_CSV.exists():   # rastreabilidade: o mesmo cálculo no Fold 1 com as características preliminares
        pc = pd.read_csv(PRELIM_CUSTO_CSV)
        pc = pc[(pc["secao"] == SEC_TO) & (pc["variante"] == VARIANTE_PRINCIPAL) & (pc["fold"].astype(str) == "1")]
        frases, fora, nb = [], [], set()
        for esq in ESQUEMAS:
            for m in ("bal_acc", "va"):
                q = pc[(pc["esquema"] == esq) & (pc["metrica"] == m)]
                r = _c(custo, SEC_TO, CONJ_P, 1, esq, VARIANTE_PRINCIPAL, m)
                if len(q) and r is not None:
                    q = q.iloc[0]
                    nb.add(int(q["boot_n"]))
                    frases.append(f"{esq}, {MET_NOME[m]}: {pp(q['diferenca'])} [{pp(q['boot_dif_ic_inf'])}; "
                                  f"{pp(q['boot_dif_ic_sup'])}] → {pp(r['diferenca'])} {_ic(r)}")
                    if not (q["boot_dif_ic_inf"] <= r["diferenca"] <= q["boot_dif_ic_sup"]):
                        fora.append(f"{esq}, {MET_NOME[m]}")
        if frases:
            n1 = int(_c(custo, SEC_TO, CONJ_P, 1, "3c", VARIANTE_PRINCIPAL, "bal_acc")["n_sujeitos"])
            a(f"**Fold 1, rodada preliminar → final** (mesmos {n1} sujeitos, variante `{VARIANTE_PRINCIPAL}`; custo (i) − (ii) "
              f"[IC 95%]; a preliminar usou {' e '.join(milhar(x) for x in sorted(nb))} reamostragens e as características "
              "daquela rodada, com o limiar de EAR calibrado no vídeo 0 inteiro — "
              "`results/preliminar/exp5a_validacao_custo.csv`): " + "; ".join(frases) + ". " +
              ("Todas as estimativas finais ficam dentro dos IC preliminares." if not fora else
               "Estimativas finais fora do IC preliminar: " + "; ".join(fora) + "."))
            a("")


def _rel_mesmo(ctx, a):
    met, custo = ctx["tabs"]["met"], ctx["tabs"]["custo"]
    VP = VARIANTE_PRINCIPAL
    a("## 4. Mesmo extrator no treino e no teste (5 folds e LOSO)")
    a("")
    a("Aqui cada lado treina e testa com as próprias características, nos mesmos sujeitos: não há troca de extrator entre "
      "treino e teste, que é a situação do Exp. 5b. **5 folds**: para cada fold, treino nos sujeitos confirmados dos "
      "outros 4 folds e teste nos do fold (o treino oficial aqui é menor que o do Exp. 1, que usa todos os sujeitos "
      "oficiais dos outros folds). **LOSO**: treino nos outros sujeitos confirmados e teste no que ficou de fora, "
      "girando os sujeitos. Mesmo RF (300 árvores, semente 42) e mesmas métricas.")
    a("")
    a("| Protocolo | Esquema | Características | BSA | Acur. bal. | F1 macro | AUC macro | VA | Oficial − nosso: acur. bal. "
      "[IC 95%]; p | Oficial − nosso: VA [IC 95%]; p; McNemar p |")
    a("|---|---|---|---:|---:|---:|---:|---:|---|---|")
    for sec in (SEC_5F, SEC_LOSO):
        for esq in ESQUEMAS:
            for conjunto in ["oficial"] + [f"nosso_{v}" for v in ORDEM_VAR]:
                if not np.isfinite(_m(met, sec, CONJ_P, "todos", esq, conjunto, "va")):
                    continue
                vals = " | ".join(pct(_m(met, sec, CONJ_P, "todos", esq, conjunto, m)) for m in METRICAS)
                if conjunto == "oficial":
                    d1 = d2 = "—"
                else:
                    var = conjunto.replace("nosso_", "")
                    cb, cv = _c(custo, sec, CONJ_P, "todos", esq, var, "bal_acc"), _c(custo, sec, CONJ_P, "todos", esq, var, "va")
                    d1 = f"{pp(cb['diferenca'])} {_ic(cb)}; {pval(cb.get('perm_p'))}"
                    d2 = f"{pp(cv['diferenca'])} {_ic(cv)}; {pval(cv.get('perm_p'))}; {pval(cv.get('mcnemar_p'))}"
                nome = "Oficial" if conjunto == "oficial" else f"Nosso `{conjunto.replace('nosso_', '')}`"
                a(f"| {SEC_NOME[sec]} | {esq} | {nome} | {vals} | {d1} | {d2} |")
    a("")
    a(f"**Por fold (5 folds, `{VP}`).**")
    a("")
    for esq in ESQUEMAS:
        a(f"*{esq}:*")
        a("")
        _tab_por_fold(ctx, a, SEC_5F, CONJ_P, esq, VP)
    a(f"**Sensibilidades (5 folds, `{VP}`).** " + " ".join(
        f"{esq}: confirmados + prováveis ({ctx['n_sujeitos_conj']['confirmado_provavel']} sujeitos) — oficial "
        f"{pct(_m(met, SEC_5F, 'confirmado_provavel', 'todos', esq, 'oficial', 'bal_acc'))} / VA "
        f"{pct(_m(met, SEC_5F, 'confirmado_provavel', 'todos', esq, 'oficial', 'va'))} × nosso "
        f"{pct(_m(met, SEC_5F, 'confirmado_provavel', 'todos', esq, f'nosso_{VP}', 'bal_acc'))} / "
        f"{pct(_m(met, SEC_5F, 'confirmado_provavel', 'todos', esq, f'nosso_{VP}', 'va'))} (dif. acur. bal. "
        f"{pp(_c(custo, SEC_5F, 'confirmado_provavel', 'todos', esq, VP, 'bal_acc')['diferenca'])} "
        f"{_ic(_c(custo, SEC_5F, 'confirmado_provavel', 'todos', esq, VP, 'bal_acc'))}); sem os vídeos preenchidos — oficial "
        f"{pct(_m(met, SEC_5F + '_sem_preenchidos', CONJ_P, 'todos', esq, 'oficial', 'bal_acc'))} / "
        f"{pct(_m(met, SEC_5F + '_sem_preenchidos', CONJ_P, 'todos', esq, 'oficial', 'va'))} × nosso "
        f"{pct(_m(met, SEC_5F + '_sem_preenchidos', CONJ_P, 'todos', esq, f'nosso_{VP}', 'bal_acc'))} / "
        f"{pct(_m(met, SEC_5F + '_sem_preenchidos', CONJ_P, 'todos', esq, f'nosso_{VP}', 'va'))} (dif. acur. bal. "
        f"{pp(_c(custo, SEC_5F + '_sem_preenchidos', CONJ_P, 'todos', esq, VP, 'bal_acc')['diferenca'])} "
        f"{_ic(_c(custo, SEC_5F + '_sem_preenchidos', CONJ_P, 'todos', esq, VP, 'bal_acc'))})."
        for esq in ESQUEMAS))
    a("")


def _texto_mecanismo(mec):
    """Frase sobre o mecanismo da Frequency, escrita a partir de results/exp5a_validacao_mecanismo_frequencia.csv."""
    if mec is None or not len(mec):
        return ""
    r = mec[mec["comparacao"] == "10 - 0"].set_index("medida")
    n = int(r["n_sujeitos"].max())

    def f(m, nd=1, plural=False):
        x = r.loc[m]
        verbo = ("sobe" if x["n_sobe"] >= x["n_desce"] else "cai") + ("m" if plural else "")
        verbo = verbo.replace("caim", "caem")
        k = int(x["n_sobe"] if verbo.startswith("sob") else x["n_desce"])
        return (f"{verbo} em {k} de {n} sujeitos (mediana da diferença {br(x['mediana_diferenca'], nd)}; "
                f"Wilcoxon {peq(x['wilcoxon_p'])})")

    ear = r.loc["ear_median"]
    ear_cai = ear["wilcoxon_p"] < 0.05 and ear["mediana_diferenca"] < 0
    v12 = r.loc["vales12_por_min"]
    tx = r.loc["taxa_piscadas_min"]
    txt = ("Do vídeo alerta (0) para o sonolento (10), a nossa taxa de piscadas por limiar " + f("taxa_piscadas_min") +
           "; a fração de quadros abaixo do limiar " + f("frac_fechado", 3) + "; as quedas do EAR contadas sem limiar "
           "(≥ 12% da linha de base local) " + f("vales12_por_min", plural=True) + "; e a mediana do EAR do vídeo " + f("ear_median", 3) + ". ")
    if v12["wilcoxon_p"] < 0.05 and v12["mediana_diferenca"] > 0:
        txt += "O número de quedas também sobe, então parte do aumento da Frequency é piscar mais. "
    else:
        txt += "O número de quedas não sobe de forma sistemática; o que muda é quantas quedas cruzam o limiar. "
    if ear_cai:
        txt += ("A linha de base do EAR cai no vídeo sonolento: como o limiar do sujeito é fixo (calibrado no último terço "
                "do vídeo alerta), mais quedas e mais quadros ficam abaixo dele. ")
    txt += ("A nossa Frequency mistura, portanto, a taxa de piscadas com a profundidade das quedas e com o nível do olho "
            "aberto, e não é a mesma grandeza da Frequency oficial (piscadas contadas pelo SVM de olho fechado).")
    return txt


def _rel_dist(ctx, a):
    dist, mec = ctx["tabs"]["dist"], ctx["tabs"]["mecanismo"]
    VP = VARIANTE_PRINCIPAL
    D = dist[(dist["conjunto_sujeitos"] == CONJ_P)]

    def g(tipo, fold, var, classe, c):
        q = D[(D["tipo"] == tipo) & (D["fold"] == str(fold)) & (D["variante"] == var) & (D["classe"] == str(classe))
              & (D["caracteristica"] == c)]
        return q.iloc[0] if len(q) else None
    a("## 5. Distribuições das 4 características pós z-score (nosso × oficial, mesmos sujeitos e vídeos)")
    a("")
    a("Unidade: a piscada. A série de cada vídeo é reconstruída das sequências (passo 2), de modo que cada piscada conta "
      "uma vez e o preenchimento de zeros é descartado. Os dois lados usam o z-score por sujeito com o último terço do "
      f"vídeo alerta e os mesmos {ctx['n_sujeitos_conj'][CONJ_P]} sujeitos confirmados. O KS de 2 amostras junta as "
      "piscadas de todos os sujeitos; o p supõe piscadas independentes, o que não vale dentro de um sujeito, então **leia "
      "o D como tamanho de efeito**. O D por sujeito compara os dois lados dentro de cada sujeito e não mistura sujeitos. "
      "Desvio com ddof = 1.")
    a("")
    a("![Distribuições nosso × oficial](figures/fig_exp5a_distribuicoes.png)")
    a("")
    a("*Figura (`results/figures/fig_exp5a_distribuicoes.pdf`).* Para cada característica, a distribuição por classe das "
      f"piscadas oficiais (azul) e das nossas (laranja, variante `{VP}`), na mesma escala dentro do painel. Caixa: "
      "quartis; traço branco: mediana; haste: percentis 5 a 95; linha cinza: z = 0, a média da referência. No título de "
      "cada painel, o D do KS com todas as classes juntas e a mediana do D por sujeito.")
    a("")
    r0 = g("distribuicao", "todos", VP, "todas", "frequencia")
    s0 = g("spearman_medias_video", "todos", VP, "todos", "frequencia")
    s1 = g("spearman_medias_video", "todos", VP, "so_5_e_10", "frequencia")
    a(f"### 5.1 Variante `{VP}`, todas as classes (5 folds)")
    a("")
    a("| Característica | Oficial: média (dp) | Oficial: Q1 / mediana / Q3 | Nosso: média (dp) | Nosso: Q1 / mediana / Q3 | "
      f"KS D (p) | D por sujeito: mediana (Q1 – Q3) | Spearman das médias por vídeo, {int(s0['n_videos'])} vídeos (p) | "
      f"só vídeos 5 e 10, {int(s1['n_videos'])} (p) |")
    a("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for c in CARACT:
        r = g("distribuicao", "todos", VP, "todas", c)
        sa, sb = g("spearman_medias_video", "todos", VP, "todos", c), g("spearman_medias_video", "todos", VP, "so_5_e_10", c)
        a(f"| {CARACT_NOME[c]} | {br(r.media_oficial, 2)} ({br(r.dp_oficial, 2)}) | {br(r.q1_oficial, 2)} / "
          f"{br(r.mediana_oficial, 2)} / {br(r.q3_oficial, 2)} | {br(r.media_nosso, 2)} ({br(r.dp_nosso, 2)}) | "
          f"{br(r.q1_nosso, 2)} / {br(r.mediana_nosso, 2)} / {br(r.q3_nosso, 2)} | {br(r.ks_D, 3)} ({pval(r.ks_p)}) | "
          f"{br(r.ks_D_mediana_por_sujeito, 3)} ({br(r.ks_D_q1_por_sujeito, 2)} – {br(r.ks_D_q3_por_sujeito, 2)}) | "
          f"{br(sa.spearman, 2)} ({pval(sa.spearman_p)}) | {br(sb.spearman, 2)} ({pval(sb.spearman_p)}) |")
    a("")
    a(f"Piscadas: {milhar(r0.n_oficial)} oficiais e {milhar(r0.n_nosso)} nossas, de {int(r0.n_sujeitos)} sujeitos e "
      f"{int(r0.n_videos)} vídeos.")
    a("")
    a("### 5.2 As três variantes (5 folds): KS D / D mediano por sujeito / Spearman das médias por vídeo")
    a("")
    a("| Característica | " + " | ".join(f"`{v}`" for v in ORDEM_VAR) + " |")
    a("|---|" + "---:|" * len(ORDEM_VAR))
    for c in CARACT:
        cel = []
        for var in ORDEM_VAR:
            r = g("distribuicao", "todos", var, "todas", c)
            s = g("spearman_medias_video", "todos", var, "todos", c)
            cel.append(f"{br(r.ks_D, 2)} / {br(r.ks_D_mediana_por_sujeito, 2)} / {br(s.spearman, 2)}")
        a(f"| {CARACT_NOME[c]} | " + " | ".join(cel) + " |")
    a("")
    a(f"### 5.3 Por classe (5 folds, `{VP}`): mediana das piscadas e KS D")
    a("")
    a("| Característica | Oficial: alerta / pouca vig. / sonolento | Nosso: alerta / pouca vig. / sonolento | "
      "KS D: alerta / pouca vig. / sonolento |")
    a("|---|---:|---:|---:|")
    for c in CARACT:
        rr = {cl: g("distribuicao", "todos", VP, cl, c) for cl in (0, 1, 2)}
        a(f"| {CARACT_NOME[c]} | " + " / ".join(br(rr[cl].mediana_oficial, 2) for cl in (0, 1, 2)) + " | " +
          " / ".join(br(rr[cl].mediana_nosso, 2) for cl in (0, 1, 2)) + " | " +
          " / ".join(br(rr[cl].ks_D, 2) for cl in (0, 1, 2)) + " |")
    a("")
    a(f"### 5.4 Por fold (`{VP}`, todas as classes): KS D e Spearman das médias por vídeo")
    a("")
    a("| Fold | Sujeitos | " + " | ".join(f"{CARACT_NOME[c]}: D / ρ" for c in CARACT) + " |")
    a("|---:|---:|" + "---:|" * len(CARACT))
    for k in range(1, U.N_FOLDS + 1):
        r1 = g("distribuicao", k, VP, "todas", "frequencia")
        if r1 is None:
            continue
        a(f"| {k} | {int(r1.n_sujeitos)} | " + " | ".join(
            f"{br(g('distribuicao', k, VP, 'todas', c).ks_D, 2)} / {br(g('spearman_medias_video', k, VP, 'todos', c).spearman, 2)}"
            for c in CARACT) + " |")
    a("")
    a("### 5.5 Por que a nossa Frequency cresce mais que a oficial")
    a("")
    a(_texto_mecanismo(mec) + " Fonte: `results/exp5a_validacao_mecanismo_frequencia.csv` (vídeo 5 − 0 e 10 − 0, por "
      "vídeo lógico; `data/features/final/uta/blinks_index.csv` e `data/signals/uta/index.csv`).")
    a("")


def _rel_leitura(ctx, a):
    t = ctx["tabs"]
    met, custo, dist = t["met"], t["custo"], t["dist"]
    VP = VARIANTE_PRINCIPAL
    n = ctx["n_sujeitos_conj"][CONJ_P]
    D = dist[(dist["conjunto_sujeitos"] == CONJ_P) & (dist["fold"] == "todos") & (dist["variante"] == VP)]
    ks = D[(D["tipo"] == "distribuicao") & (D["classe"] == "todas")].set_index("caracteristica")
    sp = D[(D["tipo"] == "spearman_medias_video") & (D["classe"] == "todos")].set_index("caracteristica")
    mantem = [CARACT_NOME[c].lower() for c in CARACT if sp.loc[c, "spearman"] >= 0.4]
    fraca = [CARACT_NOME[c].lower() for c in CARACT if sp.loc[c, "spearman"] < 0.4]
    a("## 6. Leitura")
    a("")
    a(f"1. **As nossas características não são as oficiais, mas guardam parte da estrutura.** KS D de "
      f"{br(ks['ks_D'].min(), 2)} a {br(ks['ks_D'].max(), 2)} com todas as piscadas; dentro de cada sujeito, D mediano de "
      f"{br(ks['ks_D_mediana_por_sujeito'].min(), 2)} a {br(ks['ks_D_mediana_por_sujeito'].max(), 2)}. A ordem entre "
      f"vídeos (Spearman das médias ≥ 0,4) se mantém em: {lista_pt(mantem)}" +
      (f"; fica fraca em: {lista_pt(fraca)}" if fraca else "") + ". A maior diferença de forma é a da frequência, que "
      "no nosso extrator mistura taxa de piscadas, profundidade das quedas e nível do olho aberto (seção 5.5).")
    for i, (esq, tit) in enumerate((("3c", "3 classes"), ("B3", "alerta × sonolento"))):
        cb, cv = _c(custo, SEC_TO, CONJ_P, "todos", esq, VP, "bal_acc"), _c(custo, SEC_TO, CONJ_P, "todos", esq, VP, "va")
        cl = _c(custo, SEC_TO, CONJ_P, "todos", esq, "limiar", "bal_acc")
        ambos_zero = not _exclui_zero(cb) and not _exclui_zero(cv)
        txt = (f"{2 + i}. **Custo de trocar dlib por MediaPipe com o modelo treinado nas oficiais ({tit}, {n} sujeitos):** "
               f"{pp(cb['diferenca'])} de acurácia balanceada (IC {_ic(cb)}; permutação {peq(cb.get('perm_p'))}) e "
               f"{pp(cv['diferenca'])} de VA (IC {_ic(cv)}; {peq(cv.get('perm_p'))}). ")
        cb1 = _c(custo, SEC_TO, CONJ_P, 1, esq, VP, "bal_acc")
        if ambos_zero:
            txt += (f"Com {n} sujeitos (a rodada preliminar tinha {int(cb1['n_sujeitos'])}, só do Fold 1), os IC "
                    "continuam incluindo zero: **o custo não se distingue de zero**, e o limite superior do IC o restringe "
                    f"a no máximo {pp(cb['boot_dif_ic_sup'])} de acurácia balanceada e {pp(cv['boot_dif_ic_sup'])} de VA. ")
        else:
            txt += f"Com {n} sujeitos, pelo menos um IC exclui zero: " + ("; ".join(
                f"{nome} {'(o nosso perde)' if r['boot_dif_ic_inf'] > 0 else '(o nosso ganha)'}"
                for nome, r in (("acurácia balanceada", cb), ("VA", cv)) if _exclui_zero(r))) + ". "
        pf = [(k, _c(custo, SEC_TO, CONJ_P, k, esq, VP, "bal_acc")) for k in range(1, U.N_FOLDS + 1)]
        pf = [(k, r) for k, r in pf if r is not None]
        kmin = min(pf, key=lambda kr: kr[1]["diferenca"])
        kmax = max(pf, key=lambda kr: kr[1]["diferenca"])
        nmin = min(int(r["n_sujeitos"]) for _, r in pf)
        nmax = max(int(r["n_sujeitos"]) for _, r in pf)
        txt += (f"Entre folds a diferença na acurácia balanceada vai de {pp(kmin[1]['diferenca'])} (Fold {kmin[0]}) a "
                f"{pp(kmax[1]['diferenca'])} (Fold {kmax[0]})" +
                (": o agregado junta folds com diferenças de sinais opostos" if kmin[1]["diferenca"] < 0 < kmax[1]["diferenca"]
                 else "") + f"; com {nmin} a {nmax} sujeitos por fold, nenhum fold sozinho decide. ")
        txt += (f"Com as bordas no limiar (`limiar`) a diferença é {pp(cl['diferenca'])} (IC {_ic(cl)}; permutação "
                f"{peq(cl.get('perm_p'))})" +
                (": o IC por bootstrap exclui zero" + (", mas a permutação não rejeita a igualdade a 5% — resultado no "
                                                      "limite" if cl.get("perm_p", 0) >= 0.05 else "")
                 if _exclui_zero(cl) else "") + ".")
        a(txt)
    partes = []
    for sec in (SEC_5F, SEC_LOSO):
        for esq in ESQUEMAS:
            cb = _c(custo, sec, CONJ_P, "todos", esq, VP, "bal_acc")
            cv = _c(custo, sec, CONJ_P, "todos", esq, VP, "va")
            if cb is None:
                continue
            quem = ("o nosso separa melhor" if cb["boot_dif_ic_sup"] < 0 else "o oficial separa melhor"
                    if cb["boot_dif_ic_inf"] > 0 else "sem diferença distinguível")
            partes.append(f"{SEC_NOME[sec]}, {esq}: {pp(cb['diferenca'])} {_ic(cb)} ({quem}; VA {pp(cv['diferenca'])} {_ic(cv)})")
    perdas = []
    for sec in (SEC_5F, SEC_LOSO):
        for esq in ESQUEMAS:
            for m in ("bal_acc", "va"):
                r = _c(custo, sec, CONJ_P, "todos", esq, VP, m)
                if r is not None and r["boot_dif_ic_inf"] > 0:
                    perdas.append(f"{SEC_NOME[sec]}, {esq}, {MET_NOME[m]}")
    tit4 = ("Com o mesmo extrator no treino e no teste, as nossas características não perdem para as oficiais nos mesmos "
            "sujeitos" if not perdas else "Com o mesmo extrator no treino e no teste, as nossas características perdem "
            "para as oficiais em: " + "; ".join(perdas))
    a(f"4. **{tit4}.** Oficial − nosso na acurácia balanceada [IC 95%]: " + "; ".join(partes) + ". A classe "
      "intermediária (pouca vigilância) é a parte difícil para os dois extratores (matrizes da seção 3).")
    aus, info = ctx["ausentes"]
    expl = [sp_ for sp_, inf in info.items() if any(fin(v) and v < inf["dlib1_proc_min"] for v in inf["dlib1"].values())]
    sem = [sp_ for sp_ in info if sp_ not in expl]
    ind = ctx["indep"]
    conf_ok = (len(ind) and bool(ind.loc[ind["classe"] == "confirmado", "indep_coincide"].all()))
    viol_conf = [x for v in ctx["infos_d"].values() for x in v["violacoes"]
                 if ctx["mapa"].set_index("pasta")["classe"].get(x[1]) == "confirmado"]
    a("5. **Mapeamento e pastas ausentes.** " +
      ("Os pares confirmados passam na checagem pela duração e no outro detector de vales (seção 2.2). "
       if conf_ok and not viol_conf else "Há pares confirmados que falham em alguma checagem independente (seção 2.2). ") +
      (f"A ausência de {lista_pt(expl)} tem explicação candidata no detector de rosto do dlib com o pré-processamento dos "
       "autores (seção 2.3)" if expl else "") + ("; " if expl and sem else "") +
      (f"a de {lista_pt(sem)} continua sem explicação pelos vídeos" if sem else "") +
      ("." if (expl or sem) else "Diagnóstico das pastas ausentes não executado."))
    a("")


def _rel_implicacoes(ctx, a):
    custo, met = ctx["tabs"]["custo"], ctx["tabs"]["met"]
    VP = VARIANTE_PRINCIPAL
    cv = _c(custo, SEC_TO, CONJ_P, "todos", "3c", VP, "va")
    cb = _c(custo, SEC_TO, CONJ_P, "todos", "3c", VP, "bal_acc")
    cv3 = _c(custo, SEC_TO, CONJ_P, "todos", "B3", VP, "va")
    va5f = _m(met, SEC_5F, CONJ_P, "todos", "B3", f"nosso_{VP}", "va")
    bal5f = _m(met, SEC_5F, CONJ_P, "todos", "B3", f"nosso_{VP}", "bal_acc")
    a("## 7. Implicações para o Exp. 5b e para o artigo")
    a("")
    a("1. **Usar o mesmo extrator nas duas pontas (DMD e UTA), como já está no desenho.** O ponto de comparação intra-UTA "
      "da PP1 tem de ser calculado com o nosso extrator; aqui, com sequências de 30 piscadas e 5 folds, o RF nas nossas "
      f"características dá B3 acurácia balanceada {pct(bal5f)} e VA {pct(va5f)} (seção 4). O Exp. 5b usa janelas de 10 s, "
      "não sequências, então esse número é referência de ordem de grandeza, não o teto do 5b. Não use o Exp. 1 "
      "(características oficiais) como teto do Exp. 5b.")
    a(f"2. **Manter a variante `{VP}`.** É a principal fixada antes deste experimento; o custo das outras variantes está "
      "nas seções 3 e 4. A duração, em qualquer variante, é definição própria (Spearman baixo com a oficial, seção 5): "
      "descreva-a assim no artigo.")
    a("3. **B3 como esquema principal é coerente com estes dados:** a classe intermediária é a parte fraca para os dois "
      "extratores.")
    a("4. **Ler a taxa de piscadas como sinal composto** (seção 5.5): com o limiar fixo por sujeito, a contagem responde "
      "também à profundidade das quedas e ao nível do olho aberto. No DMD atuado, piscadas e fechamentos encenados podem "
      "ter outra profundidade — um mecanismo concreto para a queda da PP1. Relate PERCLOS e quedas sem limiar ao lado da "
      "taxa de piscadas.")
    a(f"5. **Tamanho de efeito detectável.** Com {cb['n_sujeitos']:.0f} sujeitos, a meia-largura do IC 95% de uma "
      f"diferença pareada foi {br(100 * _meia(cb), 1)} pp (acurácia balanceada, 3 classes), {br(100 * _meia(cv), 1)} pp "
      f"(VA, 3 classes) e {br(100 * _meia(cv3), 1)} pp (VA, B3). Diferenças menores que isso não serão distinguíveis de "
      "zero com o mesmo número de sujeitos; a PP1 deve ser relatada com bootstrap por sujeito.")
    a("6. **No artigo:** o custo de trocar o extrator deve ser apresentado com o IC (não só a estimativa pontual) e com o "
      "conjunto de sujeitos explícito (pares confirmados), porque o mapeamento sujeito ↔ pasta é uma reconstrução nossa, "
      "não um dado publicado.")
    a("")


def _rel_limitacoes(ctx, a):
    custo = ctx["tabs"]["custo"]
    VP = VARIANTE_PRINCIPAL
    cb, cbs = _c(custo, SEC_TO, CONJ_P, "todos", "3c", VP, "bal_acc"), _c(custo, SEC_TO, "confirmado_provavel", "todos", "3c", VP, "bal_acc")
    mapa = ctx["mapa"]
    n_fora = len(mapa) - ctx["n_sujeitos_conj"][CONJ_P]
    a("## 8. Limitações e pendências")
    a("")
    a(f"- **Seleção pelo mapeamento.** O conjunto principal deixa de fora {n_fora} dos {len(mapa)} sujeitos oficiais "
      "(prováveis e indeterminados). Um par é confirmado quando as piscadas oficiais coincidem com os vales do nosso EAR, "
      "o que tende a favorecer sujeitos em que o nosso extrator funciona bem: o custo no conjunto principal pode estar "
      f"subestimado. Com os prováveis, o custo em 3 classes vai de {pp(cb['diferenca'])} para {pp(cbs['diferenca'])} "
      "(acurácia balanceada)" + (", sem sinal desse viés nos sujeitos que dá para acrescentar" if cbs["diferenca"] <= cb["diferenca"]
                                 else ", no sentido do viés") + "; os indeterminados não podem entrar.")
    norm_csv = FEAT_UTA / "sequences_normalizacao.csv"
    if norm_csv.exists():
        nz = pd.read_csv(norm_csv)
        suj_p = sorted({sujeito_de_video(v) for v in ctx["vids_par"][CONJ_P]})
        nz = nz[(nz["variante"] == VP) & (nz["sujeito"].isin(suj_p))]
        poucos = nz[nz["n_referencia"] < 3].sort_values("sujeito")
        if len(poucos):
            a(f"- **Referência do z-score com poucas piscadas no nosso lado.** Em {len(poucos)} dos {len(nz)} sujeitos do "
              "conjunto principal, o nosso detector acha tão poucas piscadas no vídeo alerta que o último terço (a "
              "referência) tem menos de 3: " + "; ".join(f"{r.sujeito} ({int(r.n_alerta)} piscadas no vídeo alerta, "
                                                          f"{int(r.n_referencia)} na referência)" for r in poucos.itertuples()) +
              ". Com desvio zero na referência, vale a regra do `Preprocessing.py` original (desvio de todas as piscadas "
              "do vídeo alerta; `data/features/final/uta/sequences_normalizacao.csv`), e o z-score desses sujeitos fica "
              "mal estimado. Isso é parte do custo de trocar o extrator (o nosso detector acha menos piscadas nos vídeos "
              "alerta), não um erro de processamento.")
    if norm_csv.exists():   # conferência: desvio de referência que é só ruído de ponto flutuante (não cai na regra do 0)
        nz = pd.read_csv(norm_csv)
        suj_p = sorted({sujeito_de_video(v) for v in ctx["vids_par"][CONJ_P]})
        nz = nz[nz["sujeito"].isin(suj_p)]
        zz = np.load(SEQ_NPZ, allow_pickle=False)
        itens = []
        for r in nz.itertuples():
            for j, c in enumerate(CARACT):
                sd, mu = float(getattr(r, f"sd_{c}")), float(getattr(r, f"mu_{c}"))
                if np.isfinite(sd) and 0 < sd < 1e-9 * max(1.0, abs(mu)):
                    Xv = zz[VARIANTES[r.variante]][zz["sujeito"] == r.sujeito][:, :, j]
                    itens.append(f"{r.sujeito}, variante `{r.variante}`, {CARACT_NOME[c].lower()}: desvio "
                                 + f"{sd:.1e}".replace(".", ",").replace("-", "−")
                                 + " (|z| máximo " + f"{np.abs(Xv).max():.1e}".replace(".", ",").replace("+", "") + ")")
        if itens:
            a("- **Desvio de referência quase nulo (ruído de ponto flutuante).** " + "; ".join(itens) + ". Em "
              "`src/features/sequences.py` a regra do desvio zero compara com 0 exato, e este desvio (valores iguais "
              "em segundos, com erro de arredondamento) passa por positivo, gerando z-scores enormes. Só a variante "
              "indicada é afetada (as outras usam quadros e caíram na regra); ela é sensibilidade aqui, mas as médias e "
              "o Spearman por vídeo dessa característica nessa variante ficam distorcidos. Correção pendente na etapa "
              "de características (tolerância relativa na regra do desvio zero).")
    a("- O p do KS trata as piscadas como independentes; o D por sujeito e o Spearman por vídeo são as leituras robustas.")
    a("- A AUC não tem IC (o bootstrap e a permutação recalculam as métricas a partir das confusões por sujeito).")
    a("- Os `.npy` oficiais são um instantâneo que nem o código dos autores reproduz exatamente (issue #2 do repositório); "
      "a comparação é com esse instantâneo, não com o método ideal.")
    a("- A explicação das pastas ausentes pelo dlib é uma hipótese compatível com os dados (seção 2.3), não um registro dos "
      "autores; a orientação dos quadros usada no teste (quadro exibido, com a rotação do contêiner aplicada) pode "
      "diferir da que o OpenCV dos autores lia em 2019 nos vídeos com metadado de rotação.")
    a("- As sequências de 30 piscadas não são a unidade do Exp. 5b (janelas de 10 s); este experimento valida as "
      "características de piscada, não as janelas.")
    a("")


def _rel_arquivos(ctx, a):
    a("## 9. Arquivos")
    a("")
    a("| Arquivo | Conteúdo |")
    a("|---|---|")
    for arq, desc in [
        ("results/exp5a_validacao.csv", "Métricas em formato longo (seção, conjunto de sujeitos, fold ou `todos`, esquema, "
         "conjunto de teste, semente, métrica, valor): RF treinado nas oficiais, 5 folds e LOSO com o mesmo extrator, "
         "sensibilidades"),
        ("results/exp5a_validacao_custo.csv", "Diferenças oficial − nosso por seção, conjunto de sujeitos, fold, esquema, "
         "variante e métrica: IC por bootstrap sobre sujeitos, permutação pareada, McNemar, contagem por sujeito, sementes"),
        ("results/exp5a_validacao_distribuicoes.csv", "Estatísticas das piscadas (n, média, dp, percentis), KS (D, p, D por "
         "sujeito) e Spearman das médias por vídeo, por conjunto, fold, variante e classe"),
        ("results/exp5a_validacao_mapeamento.csv", "Por sujeito oficial: classe do par, escore do Exp. 5, casamento pela "
         "contagem, checagem pela duração, verificação independente"),
        ("results/exp5a_validacao_mecanismo_frequencia.csv", "Mudança (vídeo 5 ou 10 − vídeo 0) por sujeito da taxa de "
         "piscadas, das quedas do EAR sem limiar, da fração abaixo do limiar e da mediana do EAR, com Wilcoxon"),
        ("results/exp5a_validacao_videos.csv", "Decisão por vídeo (voto, probabilidades médias, acerto) de cada seção, "
         "conjunto e esquema (semente 42)"),
        ("results/exp5a_validacao_confusoes.csv", "Matrizes de confusão agregadas (JSON)"),
        ("results/exp5a_videos_diagnostico.csv", "Propriedades de cada arquivo de vídeo (codec, contêiner, tamanho do rosto a "
         "450 px, brilho); incremental"),
        ("results/exp5a_dlib_rosto.csv", "Detector de rosto do dlib com o pré-processamento dos autores, por arquivo "
         "(`src/experiments/exp5a_dlib_rosto.py`, `.venv-mp`)"),
        ("results/exp5a_mapeamento_indep.csv", "Verificação independente do mapeamento: escore por sujeito oficial × pasta "
         "(`src/experiments/exp5a_mapeamento_indep.py`)"),
        ("results/figures/fig_exp5a_validacao.pdf / .png", "Figura principal: oficial − nosso com IC 95% nos três protocolos"),
        ("results/figures/fig_exp5a_distribuicoes.pdf / .png", "Distribuições das 4 características por classe (seção 5)"),
        (_rel(ctx["log_path"]), "Log da execução"),
    ]:
        a(f"| `{arq}` | {desc} |")
    a("")


def relatorio(ctx, log):
    ctx["ausentes"] = _info_ausentes(ctx)
    L = []
    a = L.append
    _rel_cabecalho(ctx, a)
    _rel_numeros(ctx, a)
    a("![Custo de trocar o extrator](figures/fig_exp5a_validacao.png)")
    a("")
    a("*Figura principal (`results/figures/fig_exp5a_validacao.pdf`).* Diferença oficial − nosso (variante "
      f"`{VARIANTE_PRINCIPAL}`, {ctx['n_sujeitos_conj'][CONJ_P]} sujeitos com par confirmado, 5 folds) na acurácia "
      "balanceada e na VA, com IC 95% por bootstrap sobre sujeitos. Linha de cima: RF treinado nas características oficiais "
      "e testado nas nossas (custo de trocar o extrator). Linhas de baixo: cada lado treinado e testado com o próprio "
      "extrator (5 folds e LOSO). Positivo = o nosso extrator perde.")
    a("")
    _rel_mapeamento(ctx, a)
    _rel_custo(ctx, a)
    _rel_mesmo(ctx, a)
    _rel_dist(ctx, a)
    _rel_leitura(ctx, a)
    _rel_implicacoes(ctx, a)
    _rel_limitacoes(ctx, a)
    _rel_arquivos(ctx, a)
    out = OUT_RES / "exp5a_validacao.md"
    out.write_text("\n".join(L), encoding="utf-8")
    log(f"  -> {_rel(out)}")



if __name__ == "__main__":
    sys.exit(main())
