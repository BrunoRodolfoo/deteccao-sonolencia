"""Exp. 6 — Robustez noturna com o NITYMED (PP3), RODADA FINAL: o pipeline de EAR/piscadas/janelas funciona à noite?

Rodar no .venv, DEPOIS de `bash src/features/run_features.sh` (piscadas e janelas da rodada final, em
data/features/final/) e de `src/experiments/exp6_luminancia.py` (no .venv-mp; iluminação medida):

    bash src/experiments/run_exp6.sh                 # luminância (.venv-mp) + esta análise (.venv); incremental
    python src/experiments/exp6_noite.py             # só a análise (medidas por vídeo em cache)
    python src/experiments/exp6_noite.py --force     # recalcula as medidas por vídeo
    python src/experiments/exp6_noite.py --treinar-dmd   # ignora models/exp5/ e treina o modelo do DMD aqui

O NITYMED (noite real, carros em movimento, Patras) NÃO tem classe alerta nem anotação quadro a quadro:
NENHUMA acurácia/AUC/F1 é calculada nele. O que se mede (janelas de 10 s, passo 5 s — regra final, notas item 9):
  (a) taxa de detecção de rosto por quadro — noite (NITYMED) × bases de referência: UTA-RLDD completa (60 sujeitos,
      180 vídeos lógicos, autogravada) e DMD (dia: s5 = sonolência encenada com o carro parado; s1 = direção real com o
      carro em movimento; s2 = direção encenada com o carro parado — nas sessões de distração, só os trechos
      "safe_drive" entram nas medidas de sinal);
  (b) qualidade do sinal de EAR — nível com olho aberto, ruído, amplitude das piscadas, SNR, piscadas/min,
      tamanho do olho em pixels e luminância medida (results/exp6_luminancia_videos.csv);
  (c) taxa de DISPARO dos detectores de janela (fração das janelas de 10 s marcadas "sonolento"): o modelo FINAL de
      transferência do Exp. 5b (treinado no DMD: alerta = safe_drive da s2 × sonolento = s5 com fechamento anotado
      >= 0,5 s ou bocejo), o modelo intra-UTA B3 final (KSS 0 × 10, treinado aqui com a mesma receita) e a regra do
      rótulo do DMD aplicada aos NOSSOS sinais (sem treino); nos vídeos de Microssono e Bocejo (noite) × falso alarme
      nas janelas alerta da UTA-RLDD e do DMD (s2 carro parado; s1 carro em movimento);
  (d) PERCLOS e fechamentos longos nos vídeos de microssono (distribuição e exemplos), com a sensibilidade do limiar
      sem referência alerta (p90 do próprio vídeo);
  (e) ficha do dataset: data/nitymed/index.csv e results/exp6_nitymed_ficha.md.

Decisões (detalhadas em results/exp6_noite.md):
  * Modelo do DMD: o salvo pelo Exp. 5b FINAL em models/exp5/ quando o md5 das janelas de treino registrado nos
    metadados confere com data/features/final/dmd/windows_10s.csv; senão (ou com --treinar-dmd) é treinado aqui com
    a mesma receita e isso fica registrado. O modelo intra-UTA B3 é treinado aqui (60 sujeitos). Receita: a do
    Exp. 5b (ver RECEITA abaixo).
  * Variantes: `sem_normalizacao` (brutas; PRINCIPAL no Exp. 6), `principal` (centradas `_c`; a principal do 5b) e
    `sem_taxa_rosto` (brutas sem taxa_rosto). O NITYMED não tem trecho alerta: a referência de `_c` e do limiar de EAR
    nele é o primeiro terço do PRÓPRIO vídeo (desvio documentado no Exp. 5) — por isso a análise noturna usa as brutas.
  * AVALIAÇÃO só nas janelas `avaliavel` (UTA e NITYMED: `uso` = exemplo, sem referência nem fronteira; DMD: alerta s2
    e sonolento s5 com rótulo, mais o alerta da s1 como grupo de sensibilidade). No próprio domínio, predição fora da
    amostra: UTA pelos 5 folds oficiais (sujeitos inteiros), DMD por deixa-um-sujeito-de-fora; o modelo da outra base e
    o NITYMED usam o modelo inteiro. Sensibilidade: NITYMED com todas as janelas (brutas).
  * Disparo = probabilidade da classe sonolenta >= 0,5. Secundário: limiar com 10% de falso alarme nas janelas alerta
    da UTA (KSS 0), aplicado à noite.
  * Por vídeo lógico: as partes _p1/_p2 dos vídeos divididos da UTA (F3_32_10, F5_49_10) são concatenadas em ordem
    (tempo contínuo) antes das medidas; métricas por vídeo agrupam pela coluna `video`.
  * IC 95%: bootstrap (1000 reamostragens, semente 42) por AGRUPAMENTO — vídeo no NITYMED (não há ID de motorista),
    sujeito na UTA-RLDD e no DMD. Vídeos do mesmo motorista do NITYMED não são independentes: IC otimista.
Saídas: results/exp6_noite.csv (longo), results/exp6_noite_videos.csv, results/exp6_noite_janelas.csv,
results/exp6_noite_importancias.csv, results/exp6_noite_contrafactual.csv, results/exp6_noite.md,
results/exp6_nitymed_ficha.md, data/nitymed/index.csv, models/exp6/*.joblib + modelos.json,
results/figures/fig_exp6_noite.{pdf,png}, results/figures/fig_exp6_microssono.{pdf,png};
log em results/logs/exp6_final_noite.log. A rodada preliminar (30 s, Fold 1, só s5) foi descartada
(arquivos removidos em 06/10/2026).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "features"))
import blinks as B  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
LOGS = RES / "logs"
FEAT = B.FEAT                          # data/features/final (rodada final)
NITY = ROOT / "data" / "nitymed"
MOD5 = ROOT / "models" / "exp5"
MOD6 = ROOT / "models" / "exp6"
LOG_ARQ = LOGS / "exp6_final_noite.log"

VERSAO = "2.2"   # 2.2: texto do setup fora do protocolo calculado (verificação);  2.0: rodada final (janelas de 10 s, UTA completa, DMD com distração, vídeo lógico, modelos finais);
                 # 2.1: + resoluções de todas as partes por vídeo (a versão entra na chave do cache por vídeo)
SEED = 42
N_BOOT = 1000
N_JOBS = 6
JANELA_S = 10.0
BASES = ("dmd", "nitymed", "uta")
NOME = {"dmd": "DMD", "nitymed": "NITYMED", "uta": "UTA-RLDD"}
FEATS = ["perclos", "taxa_piscadas_min", "dur_media_piscada_s", "dur_max_fechamento_s", "n_fechamentos_05s",
         "ear_media", "ear_dp", "amp_media", "vel_media", "frac_bocejo"]
FEATS11 = FEATS + ["taxa_rosto"]      # as 11 características da janela (Exp. 5)
VARIANTES = {"sem_normalizacao": FEATS11,                    # PRINCIPAL no Exp. 6 (ver docstring)
             "principal": [f + "_c" for f in FEATS11],        # principal do Exp. 5b (centradas na referência)
             "sem_taxa_rosto": FEATS}                         # sensibilidade do Exp. 6
VAR_PRINCIPAL = "sem_normalizacao"
NOME_VAR = {"sem_normalizacao": "brutas, 11 características (principal no Exp. 6)",
            "principal": "centradas `_c`, 11 características (principal do Exp. 5b)",
            "sem_taxa_rosto": "brutas sem `taxa_rosto`, 10 características (sensibilidade)"}
LIMIAR_DISPARO = 0.5        # disparo: probabilidade >= 0,5 (a regra do Exp. 5b: s >= limiar)
FAR_ALVO = 0.10
MARGEM_EVENTO_S = 0.10      # (b) quadros a menos de 0,1 s de uma piscada/fechamento não contam como "olho aberto"
JANELA_RUIDO_S = 1.0        # (b) ruído rápido = resíduo em torno da mediana móvel de 1 s
P_ABERTO = 90               # (d) sensibilidade: nível de olho aberto = percentil 90 do EAR do vídeo inteiro
FECH_S = 0.5                # fechamento "longo o bastante" para o rótulo do DMD e para a regra
LONGO_S = B.FECHAMENTO_LONGO_S  # 1,5 s
MAR_LIMIAR = 0.5            # bocejo = MAR > 0,5 sustentado >= 1 s (mesma definição de frac_bocejo em windows.py)
MAR_SUST_S = 1.0
SUSPEITO_DUR_S = 90.0      # (e) vídeo da pasta Yawning com > 90 s (declarado: 15–25 s) e SEM nenhum bocejo detectado
                           #     = suspeito de estar na pasta errada (duração típica dos de microssono, ~2 min)

# grupos de janelas da seção (c) e rótulos (a condição de luz/movimento vai no rótulo)
GRUPOS_C = [("nitymed", "Microsleep"), ("nitymed", "Yawning"), ("uta", "KSS 0"), ("uta", "KSS 5"), ("uta", "KSS 10"),
            ("dmd", "alerta_movimento"), ("dmd", "alerta_parado"), ("dmd", "sonolento"),
            ("dmd", "alerta_s1"), ("dmd", "alerta")]   # os dois últimos: por sessão do protocolo (sensibilidade)
GRUPOS_C_TAB = GRUPOS_C[:8]
ROTULO_GRUPO = {("nitymed", "Microsleep"): "NITYMED Microssono (noite, carro em movimento)",
                ("nitymed", "Yawning"): "NITYMED Bocejo (noite, carro em movimento)",
                ("uta", "KSS 0"): "UTA-RLDD alerta, KSS 0 (autogravado)",
                ("uta", "KSS 5"): "UTA-RLDD pouca vigilância, KSS 5 (autogravado)",
                ("uta", "KSS 10"): "UTA-RLDD sonolento, KSS 10 (autogravado)",
                ("dmd", "alerta_movimento"): "DMD alerta, direção segura com o carro em movimento (dia)",
                ("dmd", "alerta_parado"): "DMD alerta, direção segura com o carro parado (dia)",
                ("dmd", "sonolento"): "DMD sonolento atuado s5 (dia, carro parado)",
                ("dmd", "alerta_s1"): "DMD alerta da sessão s1 (protocolo: carro em movimento)",
                ("dmd", "alerta"): "DMD alerta da sessão s2 (protocolo: carro parado; classe alerta de treino)"}
ROTULO_CURTO = {("nitymed", "Microsleep"): "NITYMED microssono", ("nitymed", "Yawning"): "NITYMED bocejo",
                ("uta", "KSS 0"): "UTA KSS 0", ("uta", "KSS 5"): "UTA KSS 5", ("uta", "KSS 10"): "UTA KSS 10",
                ("dmd", "alerta_movimento"): "DMD alerta, em movimento", ("dmd", "alerta_parado"): "DMD alerta, parado",
                ("dmd", "sonolento"): "DMD sonolento s5", ("dmd", "alerta_s1"): "DMD alerta s1",
                ("dmd", "alerta"): "DMD alerta s2"}
MOVIMENTO, PARADO = "Car Moving", "Car Stopped"   # campo `setup` do OpenLABEL do DMD

log = logging.getLogger("exp6")


# ======================================================================================
# Utilidades
# ======================================================================================
def fnum(x, nd=3):
    """Número em pt-BR (vírgula decimal, ponto de milhar); '—' para ausente."""
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
    except (TypeError, ValueError):
        return str(x)
    s = f"{float(x):,.{nd}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def fpct(x, nd=1):
    try:
        if x is None or not np.isfinite(float(x)):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return fnum(100.0 * float(x), nd) + "%"


def fint(n):
    try:
        return f"{int(n):,}".replace(",", ".")
    except (TypeError, ValueError):
        return "—"


def fic(est, lo, hi, nd=3, pct=False):
    f = (lambda v: fpct(v, nd)) if pct else (lambda v: fnum(v, nd))
    try:
        ok = np.isfinite(float(lo)) and np.isfinite(float(hi))
    except (TypeError, ValueError):
        ok = False
    return f"{f(est)} [{f(lo)}; {f(hi)}]" if ok else f(est)


def boot_cluster(valores, clusters, stat=np.mean, n_boot=N_BOOT, seed=SEED):
    """Estatística de valores por unidade (ex.: por vídeo) com IC 95% por bootstrap de AGRUPAMENTOS."""
    v = np.asarray(valores, float)
    c = np.asarray(clusters)
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    est = float(stat(v))
    uc, inv = np.unique(c, return_inverse=True)
    if len(uc) < 2:
        return est, np.nan, np.nan
    grupos = [v[inv == k] for k in range(len(uc))]
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uc), len(uc))
        bs[i] = stat(np.concatenate([grupos[k] for k in pick]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, float(lo), float(hi)


def boot_dif(va, ca, vb, cb, stat=np.mean, n_boot=N_BOOT, seed=SEED):
    """Diferença stat(A) − stat(B) com bootstrap independente de agrupamentos em cada base."""
    va, vb = np.asarray(va, float), np.asarray(vb, float)
    ca, cb = np.asarray(ca), np.asarray(cb)
    oka, okb = np.isfinite(va), np.isfinite(vb)
    va, ca, vb, cb = va[oka], ca[oka], vb[okb], cb[okb]
    if len(va) == 0 or len(vb) == 0:
        return np.nan, np.nan, np.nan
    est = float(stat(va) - stat(vb))
    ua, ia = np.unique(ca, return_inverse=True)
    ub, ib = np.unique(cb, return_inverse=True)
    ga = [va[ia == k] for k in range(len(ua))]
    gb = [vb[ib == k] for k in range(len(ub))]
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        pa = rng.integers(0, len(ua), len(ua))
        pb = rng.integers(0, len(ub), len(ub))
        bs[i] = stat(np.concatenate([ga[k] for k in pa])) - stat(np.concatenate([gb[k] for k in pb]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, float(lo), float(hi)


def taxa_boot(flag, clusters, n_boot=N_BOOT, seed=SEED):
    """Fração de janelas marcadas (ponderada por janela) com IC 95% por bootstrap de agrupamentos."""
    df = pd.DataFrame({"f": np.asarray(flag, float), "c": np.asarray(clusters)})
    if len(df) == 0:
        return np.nan, np.nan, np.nan, 0, 0
    g = df.groupby("c")["f"].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(), g["count"].to_numpy()
    est = float(s.sum() / n.sum())
    if len(g) < 2:
        return est, np.nan, np.nan, len(g), int(n.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    bs = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, float(lo), float(hi), len(g), int(n.sum())


def md5_arquivo(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b_ in iter(lambda: f.read(1 << 20), b""):
            h.update(b_)
    return h.hexdigest()


def rotulo_L(janela):
    return f"{int(janela)}s" if float(janela).is_integer() else f"{janela}s".replace(".", "p")


class Linhas:
    """Acumula as linhas do CSV longo results/exp6_noite.csv."""

    def __init__(self):
        self.rows = []

    def add(self, secao, base, grupo, metrica, valor, ic_inf=np.nan, ic_sup=np.nan, n_videos=np.nan,
            n_janelas=np.nan, n_agrup=np.nan, modelo="", variante="", nota=""):
        self.rows.append(dict(secao=secao, base=base, grupo=grupo, modelo=modelo, variante=variante,
                              metrica=metrica, valor=valor, ic95_inf=ic_inf, ic95_sup=ic_sup,
                              n_videos=n_videos, n_janelas=n_janelas, n_agrupamentos=n_agrup, nota=nota))

    def df(self):
        return pd.DataFrame(self.rows)


# ======================================================================================
# Dados
# ======================================================================================
def carregar_janelas(janela=JANELA_S):
    """Janelas da rodada final (data/features/final/<base>/windows_<L>s.csv) com `grupo`, `cluster` e `aval`.

    aval (conjunto avaliado na seção c): UTA e NITYMED = `avaliavel` (uso exemplo: nem referência nem fronteira);
    DMD = alerta s2 e sonolento s5 com rótulo (`avaliavel`) + alerta da s1 (grupo de sensibilidade, sem treino)."""
    W = {}
    L_ = rotulo_L(janela)
    for ds in BASES:
        p = FEAT / ds / f"windows_{L_}.csv"
        if not p.exists():
            raise SystemExit(f"Falta {p.relative_to(ROOT)} — rode antes: bash src/features/run_features.sh")
        w = pd.read_csv(p, low_memory=False, dtype={"rotulo_video": str, "sessao": str, "parte": str})
        for c in ("janela_ref", "sobrepoe_ref", "avaliavel", "parcial"):
            w[c] = w[c].astype(str).str.lower().eq("true")
        if ds == "nitymed":
            w["grupo"] = w["categoria"]
        elif ds == "uta":
            w["grupo"] = "KSS " + w["rotulo_kss"].astype(int).astype(str)
        else:
            w["grupo"] = w["uso"]
        w["cluster"] = w["video"] if ds == "nitymed" else w["sujeito"]
        if ds == "dmd":
            w["aval"] = (w["avaliavel"] & w["uso"].isin(["alerta", "sonolento"]) & w["y"].notna()) | (w["uso"] == "alerta_s1")
            alerta = w["uso"].isin(["alerta", "alerta_s1"])       # direção segura das sessões s1 e s2 (inclui gE_28)
            w["grupo_setup"] = np.where(alerta & (w["setup"] == MOVIMENTO), "alerta_movimento",
                                        np.where(alerta & (w["setup"] == PARADO), "alerta_parado", ""))
        else:
            w["aval"] = w["avaliavel"]
        W[ds] = w.reset_index(drop=True)
    return W


def sel_grupo(df, ds, g):
    """Janelas do grupo g da seção (c) (DMD: alerta por `setup` ou por sessão; demais: coluna grupo), só `aval`
    — exceto os grupos do DMD por `setup`, que incluem as janelas de direção segura de todos os 14 sujeitos."""
    if ds == "dmd" and g in ("alerta_movimento", "alerta_parado"):
        return df[df["grupo_setup"] == g]
    return df[df["aval"] & (df["grupo"] == g)]


def unidades_video(ds):
    """Vídeos LÓGICOS da base: lista de (video, DataFrame das partes em ordem)."""
    idx = B.listar_videos(ds)
    idx["parte_n"] = pd.to_numeric(idx["parte"], errors="coerce").fillna(0)
    return [(vid, g.sort_values("parte_n")) for vid, g in idx.groupby("video", sort=True)]


def carregar_video(ds, partes):
    """Sinal e eventos de um vídeo lógico: partes concatenadas em ordem, com tempo e índices de quadro contínuos."""
    itens = []
    for _, r in partes.iterrows():
        s = B.carregar_sinal(ROOT / r["npz"])
        with np.load(ROOT / r["npz"], allow_pickle=False) as z:
            s["lm"] = z["lm"].astype(float)
            s["face_mp"] = z["face"].astype(bool)
        ev = pd.read_csv(FEAT / ds / "blinks" / f"{r['id']}.csv")
        itens.append((r, s, ev))
    if len(itens) == 1:
        return itens[0][1], itens[0][2]
    t_off, q_off = 0.0, 0
    cat = {k: [] for k in ("ear", "mar", "face", "face_mp", "lm", "t", "t_nominal", "d")}
    evs = []
    for r, s, ev in itens:
        for k in ("ear", "mar", "face", "face_mp", "lm", "d"):
            cat[k].append(s[k])
        cat["t"].append(s["t"] + t_off)
        cat["t_nominal"].append(s["t_nominal"] + t_off)
        e2 = ev.copy()
        for c in ("inicio", "fundo", "fim", "inicio_ext", "fim_ext"):
            e2[c] = e2[c].astype(int) + q_off
        for c in ("t_inicio_s", "t_fim_s"):
            e2[c] = e2[c] + t_off
        evs.append(e2)
        t_off += B.duracao_total(s)
        q_off += s["n"]
    S = {k: np.concatenate(v) for k, v in cat.items()}
    S["n"] = q_off
    S["fps"] = float(np.average([s["fps"] for _, s, _ in itens], weights=[s["n"] for _, s, _ in itens]))
    S["meta"], S["extra"] = itens[0][1]["meta"], itens[0][1]["extra"]
    return S, pd.concat(evs, ignore_index=True)


def mascara_safe(anotacao, n):
    """Quadros dos trechos driver_actions/safe_drive (DMD, sessões de distração)."""
    m = np.zeros(n, bool)
    for s_, e_ in B.trechos_acao(anotacao):
        m[max(s_, 0):min(e_, n - 1) + 1] = True
    return m


def grupo_video(ds, r):
    if ds == "nitymed":
        return str(r["categoria"])
    if ds == "uta":
        return f"KSS {r['label']}"
    return str(r["sessao"])


# ======================================================================================
# (a), (b), (d): medidas por vídeo a partir dos sinais e das piscadas
# ======================================================================================
def medidas_video(S, EV, lim, med_ref, mask=None):
    """Medidas de captura, de sinal e de fechamento de um vídeo (ou dos quadros de `mask`, ex.: safe_drive)."""
    n = S["n"]
    face_mp, face, ear, mar, t, d = S["face_mp"], S["face"], S["ear"], S["mar"], S["t"], S["d"]
    m = np.ones(n, bool) if mask is None else np.asarray(mask, bool)
    fm = face & m
    fps = S["fps"] if np.isfinite(S["fps"]) and S["fps"] > 0 else 30.0
    dur = float(d[m].sum())
    out = dict(fps=fps, n_quadros=int(m.sum()), dur_s=dur)
    # ---- (a) detecção de rosto
    out["taxa_rosto"] = float(face_mp[m].mean()) if m.any() else np.nan       # Face Mesh devolveu landmarks
    out["taxa_rosto_ear_valido"] = float(face[m].mean()) if m.any() else np.nan
    s_arr, e_arr = B.runs(~face_mp & m)
    lac = np.array([float(d[a:b_ + 1].sum()) for a, b_ in zip(s_arr, e_arr)])
    out["n_lacunas"] = int(len(lac))
    out["maior_lacuna_s"] = float(lac.max()) if len(lac) else 0.0
    out["lacunas_05s_por_min"] = float((lac >= 0.5).sum() / (dur / 60.0)) if dur > 0 else np.nan
    out["frac_tempo_sem_rosto"] = float(d[~face_mp & m].sum() / dur) if dur > 0 else np.nan
    # ---- (b) qualidade do EAR
    out["limiar_ear"] = float(lim)
    out["ear_ref_mediana"] = float(med_ref)
    marg = max(1, int(round(MARGEM_EVENTO_S * fps)))
    evento = np.zeros(n, bool)
    if len(EV):
        for a, z_ in zip(EV["inicio_ext"].to_numpy(int), EV["fim_ext"].to_numpy(int)):
            evento[max(0, a - marg):min(n, z_ + marg + 1)] = True
    aberto = fm & (ear >= lim) & ~evento
    ea = ear[aberto]
    out["frac_quadros_olho_aberto"] = float(aberto.sum() / fm.sum()) if fm.sum() else np.nan
    out["ear_aberto_media"] = float(ea.mean()) if len(ea) else np.nan
    out["ear_aberto_mediana"] = float(np.median(ea)) if len(ea) else np.nan
    out["ear_aberto_dp"] = float(ea.std()) if len(ea) > 1 else np.nan
    w = int(round(JANELA_RUIDO_S * fps)) | 1
    e_nan = np.where(face, ear, np.nan)
    base = pd.Series(e_nan).rolling(w, center=True, min_periods=max(3, w // 3)).median().to_numpy()
    res = (e_nan - base)[aberto]
    res = res[np.isfinite(res)]
    out["ruido_rapido"] = float(1.4826 * np.median(np.abs(res - np.median(res)))) if len(res) > 10 else np.nan
    if len(EV) and mask is not None:        # evento inteiro dentro do recorte
        EVm = EV[m[EV["inicio"].to_numpy(int)] & m[EV["fim"].to_numpy(int)]]
    else:
        EVm = EV
    pis = EVm[EVm["tipo"] == "piscada"] if len(EVm) else EVm
    tface_min = float(d[fm].sum() / 60.0)
    out["n_piscadas"] = int(len(pis))
    out["taxa_piscadas_min"] = len(pis) / tface_min if tface_min > 0 else np.nan
    out["amp_ext_media"] = float(pis["amplitude_ext"].mean()) if len(pis) else np.nan
    out["amp_limiar_media"] = float(pis["amplitude"].mean()) if len(pis) else np.nan
    out["amp_ext_relativa"] = (out["amp_ext_media"] / out["ear_aberto_media"]
                               if np.isfinite(out["amp_ext_media"]) and out["ear_aberto_media"] else np.nan)
    out["snr"] = (out["amp_ext_media"] / out["ear_aberto_dp"]
                  if np.isfinite(out["amp_ext_media"]) and out["ear_aberto_dp"] > 0 else np.nan)
    out["snr_rapido"] = (out["amp_ext_media"] / out["ruido_rapido"]
                         if np.isfinite(out["amp_ext_media"]) and out["ruido_rapido"] > 0 else np.nan)
    pk, prof = B.vales_ear(S)                # vales do EAR (diagnóstico independente do limiar)
    if len(pk):
        sel = m[pk]
        pk, prof = pk[sel], prof[sel]
    out["vales12_por_min"] = float(np.sum(prof >= 0.12) / tface_min) if tface_min > 0 else np.nan
    out["vales25_por_min"] = float(np.sum(prof >= 0.25) / tface_min) if tface_min > 0 else np.nan
    out["prof_rel_p90_vales"] = float(np.percentile(prof, 90)) if len(prof) else np.nan
    lm = S["lm"]
    olho = (np.linalg.norm(lm[:, 0] - lm[:, 3], axis=1) + np.linalg.norm(lm[:, 6] - lm[:, 9], axis=1)) / 2.0
    out["olho_largura_px"] = float(np.nanmedian(olho[fm])) if fm.any() else np.nan
    out["rosto_largura_px"] = float(np.nanmedian(np.linalg.norm(lm[:, 19] - lm[:, 20], axis=1)[fm])) if fm.any() else np.nan
    # ---- (d) PERCLOS e fechamentos (limiar do pipeline = 0,75 × mediana da referência)
    out["perclos"] = float((fm & (ear < lim)).sum() / fm.sum()) if fm.sum() else np.nan
    dur_ev = EVm["dur_s"].to_numpy(float) if len(EVm) else np.zeros(0)
    out["n_fech_05s"] = int((dur_ev >= FECH_S).sum())
    out["n_fech_1s"] = int((dur_ev >= 1.0).sum())
    out["n_fech_longos"] = int((EVm["tipo"] == "fechamento_longo").sum()) if len(EVm) else 0
    out["fech_05s_por_min"] = out["n_fech_05s"] / tface_min if tface_min > 0 else np.nan
    out["dur_max_fech_s"] = float(dur_ev.max()) if len(dur_ev) else 0.0
    # ---- (d) sensibilidade: nível aberto = percentil 90 do EAR do próprio vídeo (não depende de trecho alerta)
    if fm.sum() > 10:
        p90 = float(np.percentile(ear[fm], P_ABERTO))
        lim90 = B.K_LIMIAR * p90
        f90 = fm & (ear < lim90)
        a90, b90 = B.runs(f90)
        keep = (b90 - a90 + 1) >= B.MIN_QUADROS
        du90 = np.array([float(t[e] + d[e] - t[a]) for a, e in zip(a90[keep], b90[keep])])
        out.update(ear_p90=p90, limiar_p90=lim90, perclos_p90=float(f90.sum() / fm.sum()),
                   n_fech_05s_p90=int((du90 >= FECH_S).sum()), n_fech_longos_p90=int((du90 > LONGO_S).sum()),
                   fech_05s_por_min_p90=float((du90 >= FECH_S).sum() / tface_min) if tface_min > 0 else np.nan,
                   dur_max_fech_s_p90=float(du90.max()) if len(du90) else 0.0)
    # ---- bocejos (MAR > 0,5 sustentado >= 1 s entre quadros com rosto; a definição de frac_bocejo)
    sb, eb = B.runs(fm & (mar > MAR_LIMIAR))
    durb = np.array([float(d[a:b_ + 1].sum()) for a, b_ in zip(sb, eb)])
    out["n_bocejos"] = int((durb >= MAR_SUST_S).sum())
    out["bocejos_por_min"] = out["n_bocejos"] / tface_min if tface_min > 0 else np.nan
    return out


def info_video(ds, vid, partes):
    r0 = partes.iloc[0]
    tam = sum(os.path.getsize(ROOT / p) for p in partes["path"])
    n_tot = int(partes["n_frames"].sum())
    pix = float((partes["n_frames"] * partes["width_orig"] * partes["height_orig"]).sum())
    fr = float((partes["face_rate"] * partes["n_frames"]).sum() / n_tot) if n_tot else np.nan
    return dict(dataset=ds, id=vid, video=vid, n_partes=int(len(partes)), arquivos=";".join(partes["id"]),
                sujeito=r0["subject"], grupo=grupo_video(ds, r0), sessao=r0.get("sessao"), setup=r0.get("setup"),
                data_gravacao=r0.get("data_gravacao"), categoria=r0.get("categoria"), genero=r0.get("genero"),
                rotulo=r0.get("label"), oculos=r0.get("oculos"), fold=r0.get("fold"),
                escala_processamento=float(partes["scale"].mean()), largura=int(r0["width_orig"]),
                altura=int(r0["height_orig"]), resolucoes=";".join(f"{int(a)}x{int(b)}" for a, b in
                                                                   zip(partes["width_orig"], partes["height_orig"])),
                tamanho_bytes=int(tam), _pix=pix,
                taxa_rosto_index_extracao=fr)


def tabela_videos(force=False):
    """Uma linha por vídeo LÓGICO × recorte ('inteiro'; nas sessões de distração do DMD também 'safe_drive'),
    com cache por vídeo (data/features/final/<base>/exp6_videos/<video>__<recorte>.json)."""
    linhas, feitos, pulados = [], 0, 0
    for ds in BASES:
        bidx = pd.read_csv(FEAT / ds / "blinks_index.csv", dtype={"rotulo": str}).set_index("id")
        cache = FEAT / ds / "exp6_videos"
        cache.mkdir(parents=True, exist_ok=True)
        for vid, partes in unidades_video(ds):
            ids = list(partes["id"])
            falta = [i for i in ids if i not in bidx.index or not (FEAT / ds / "blinks" / f"{i}.csv").exists()]
            if falta:
                log.warning("%s/%s: sem piscadas para %s (rode bash src/features/run_features.sh) — ignorado", ds, vid, falta)
                continue
            lims = bidx.loc[ids, "limiar"].to_numpy(float)
            if np.nanmax(lims) - np.nanmin(lims) > 1e-12:
                log.warning("%s/%s: limiares diferentes entre as partes %s", ds, vid, lims)
            lim, med = float(lims[0]), float(bidx.loc[ids[0], "ear_ref_mediana"])
            recortes = ["inteiro"]
            if ds == "dmd" and str(partes.iloc[0]["sessao"]) in ("s1", "s2", "s3"):
                recortes.append("safe_drive")
            info = info_video(ds, vid, partes)
            carregado = None
            for rec in recortes:
                chave = hashlib.sha1(json.dumps(
                    [VERSAO, rec] + [os.path.getmtime(ROOT / p) for p in partes["npz"]]
                    + [os.path.getmtime(FEAT / ds / "blinks" / f"{i}.csv") for i in ids]
                    + [str(bidx.loc[i, "params_hash"]) for i in ids]).encode()).hexdigest()[:12]
                cj = cache / f"{vid}__{rec}.json"
                if not force and cj.exists():
                    try:
                        c = json.load(open(cj))
                        if c.get("_chave") == chave:
                            linhas.append(c)
                            pulados += 1
                            continue
                    except (json.JSONDecodeError, OSError):
                        pass
                if carregado is None:
                    carregado = carregar_video(ds, partes)
                S, EV = carregado
                mask = mascara_safe(partes.iloc[0]["anotacao"], S["n"]) if rec == "safe_drive" else None
                mm = medidas_video(S, EV, lim, med, mask)
                dur_tot = B.duracao_total(S)
                m = dict(info)
                m.update(mm)
                m["recorte"] = rec
                m["dur_video_s"] = dur_tot
                m["taxa_bits_mbps"] = info["tamanho_bytes"] * 8 / dur_tot / 1e6 if dur_tot > 0 else np.nan
                m["bits_por_pixel"] = info["tamanho_bytes"] * 8 / info["_pix"] if info["_pix"] > 0 else np.nan
                m["_chave"] = chave
                m.pop("_pix", None)
                tmp = cj.with_suffix(".tmp.json")
                json.dump({k: (None if isinstance(v, float) and not np.isfinite(v) else
                               (v.item() if hasattr(v, "item") else v)) for k, v in m.items()}, open(tmp, "w"),
                          ensure_ascii=False)
                os.replace(tmp, cj)
                linhas.append(json.load(open(cj)))
                feitos += 1
    V = pd.DataFrame(linhas)
    txt = ("dataset", "id", "video", "arquivos", "sujeito", "grupo", "sessao", "setup", "data_gravacao", "categoria",
           "genero", "rotulo", "oculos", "recorte", "resolucoes", "_chave")
    num = [c for c in V.columns if c not in txt]
    V[num] = V[num].apply(pd.to_numeric, errors="coerce")
    V["oculos"] = V["oculos"].astype(str).str.lower().eq("true")
    log.info("Medidas por vídeo lógico: %d linhas (%d calculadas, %d do cache); vídeos por base: %s", len(V), feitos,
             pulados, V[V["recorte"] == "inteiro"].groupby("dataset")["video"].nunique().to_dict())
    V = juntar_luminancia(V)
    V["cluster"] = np.where(V["dataset"] == "nitymed", V["video"], V["sujeito"])
    return V


LUM_COLS = ["lum_quadro", "lum_rosto", "lum_olhos", "contraste_rosto", "frac_escuro_rosto", "area_rosto_frac"]


def juntar_luminancia(V):
    """Luminância por vídeo lógico: mediana das amostras de todas as partes (cache por arquivo do exp6_luminancia.py)."""
    lum = RES / "exp6_luminancia_videos.csv"
    if not lum.exists():
        log.warning("Sem %s — rode src/experiments/exp6_luminancia.py no .venv-mp (luminância fica vazia)",
                    lum.relative_to(ROOT))
        for c in LUM_COLS + ["codec"]:
            V[c] = np.nan
        return V
    L = pd.read_csv(lum)
    rows = []
    for (ds, vid), g in L.groupby(["dataset", "video"]):
        r = dict(dataset=ds, video=vid, codec=";".join(sorted(set(g["codec"].astype(str)))),
                 n_amostras_lum=int(g["n_amostras"].sum()))
        if len(g) == 1:
            for c in LUM_COLS:
                r[c] = float(g[c].iloc[0])
        else:                                   # vídeo dividido: mediana das amostras das partes juntas
            amostras = []
            for i in g["id"]:
                c = json.load(open(FEAT / ds / "luminancia" / f"{i}.json"))
                amostras += c.get("amostras", [])
            for c in LUM_COLS:
                v = [a[c] for a in amostras]
                r[c] = float(np.median(v)) if v else np.nan
        rows.append(r)
    Lv = pd.DataFrame(rows)
    V = V.merge(Lv, on=["dataset", "video"], how="left")
    falta = V.loc[V["recorte"] == "inteiro"].groupby("dataset")["lum_rosto"].apply(lambda s: int(s.isna().sum())).to_dict()
    log.info("Luminância: %s (vídeos sem medida por base: %s)", lum.relative_to(ROOT), falta)
    return V


# ======================================================================================
# (c) modelos de janela
# ======================================================================================
RECEITA = ("Pipeline(SimpleImputer(median, keep_empty_features=True), RandomForestClassifier(n_estimators=300, "
           "class_weight='balanced', random_state=42, n_jobs=6)) — a do Exp. 5b; quando o modelo do 5b é carregado, "
           "os modelos fora da amostra são clones dele (sklearn.base.clone: mesmos hiperparâmetros)")


def novo_rf():
    """Receita do Exp. 5b (src/experiments/exp5_transfer.py, novo_modelo('RF'))."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    return Pipeline([("imputacao", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED,
                                                    n_jobs=N_JOBS))])


def conjuntos_treino(W):
    """DMD: alerta (safe_drive da s2) × sonolento (s5 com fechamento anotado >= 0,5 s ou bocejo), só `avaliavel`
    (y definido; gE_28 sem s5 fica fora); UTA: B3 (KSS 0 × 10), só `avaliavel` (sem referência nem fronteira)."""
    d = W["dmd"]
    dmd = d[d["avaliavel"] & d["uso"].isin(["alerta", "sonolento"]) & d["y"].notna()].reset_index(drop=True)
    u = W["uta"]
    uta = u[u["avaliavel"] & u["B3"].notna()].reset_index(drop=True)
    return {"dmd": (dmd, dmd["y"].astype(int).to_numpy()), "uta": (uta, uta["B3"].astype(int).to_numpy())}


def _entradas_json(obj):
    if isinstance(obj, list):
        for e in obj:
            yield from _entradas_json(e)
    elif isinstance(obj, dict):
        if "arquivo" in obj and ("caracteristicas" in obj or "features" in obj):
            yield obj
        else:
            for v in obj.values():
                if isinstance(v, (list, dict)):
                    yield from _entradas_json(v)


def _md5s(e):
    out = set()
    for k, v in e.items():
        if "md5" in str(k).lower():
            if isinstance(v, str):
                out.add(v)
            elif isinstance(v, dict):
                out.update(x for x in v.values() if isinstance(x, str))
    return out


def modelos_exp5b(md5_dmd):
    """Entradas de metadados de models/exp5/*.json cujos modelos foram treinados com as janelas FINAIS do DMD
    (algum campo md5 = md5 de data/features/final/dmd/windows_10s.csv) e cujo arquivo existe."""
    out = []
    for js in sorted(MOD5.glob("*.json")):
        try:
            obj = json.load(open(js, encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("%s ilegível (%r)", js.relative_to(ROOT), exc)
            continue
        for e in _entradas_json(obj):
            arq = Path(e["arquivo"])
            arq = arq if arq.is_absolute() else (ROOT / arq if (ROOT / arq).exists() else MOD5 / arq.name)
            if not arq.exists():
                continue
            if md5_dmd not in _md5s(e):
                continue
            e = dict(e)
            e["_arquivo"] = arq
            e["_json"] = str(js.relative_to(ROOT))
            e["caracteristicas"] = list(e.get("caracteristicas", e.get("features", [])))
            out.append(e)
    return out


def escolher_modelo(cand, cols, tipo="RF"):
    """Modelo de transferência (C: treinado em todo o DMD) com as características `cols` e o tipo pedido."""
    for e in cand:
        exp = str(e.get("experimento", e.get("exp", "C"))).upper()
        if exp and not exp.startswith("C") and "TRANSF" not in exp:
            continue
        if str(e.get("modelo", "")).upper() != tipo:
            continue
        if list(e["caracteristicas"]) == list(cols):
            return e
    return None


def limiar_otimo(y, s):
    """Cópia da regra do Exp. 5b (exp5_transfer.limiar_otimo): limiar t que maximiza a acurácia balanceada de
    `s >= t`; candidatos = pontos médios entre escores distintos consecutivos e os extremos; empate -> o mais
    próximo de 0,5."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, float)
    u = np.unique(s)
    cand = np.r_[u[0] - 1e-6, (u[:-1] + u[1:]) / 2.0, u[-1] + 1e-6]
    sp, sn = np.sort(s[y]), np.sort(s[~y])
    tp = len(sp) - np.searchsorted(sp, cand, side="left")
    tn = np.searchsorted(sn, cand, side="left")
    ba = 0.5 * (tp / len(sp) + tn / len(sn))
    melhor = np.flatnonzero(ba >= ba.max() - 1e-12)
    j = melhor[np.argmin(np.abs(cand[melhor] - 0.5))]
    return float(cand[j]), float(ba[j])


def treinar_e_prever(W, forcar_treino_dmd=False):
    """Modelo do DMD (o final do Exp. 5b quando disponível) e modelo intra-UTA B3 (treinado aqui), 3 variantes.

    Devolve (janelas com as probabilidades p_<fonte>_<variante>, importâncias, metadados, modelos inteiros)."""
    import joblib
    import sklearn
    from sklearn.base import clone
    MOD6.mkdir(parents=True, exist_ok=True)
    for velho in MOD6.glob("rf_*.joblib"):            # modelos desta pasta são sempre regerados
        velho.unlink()
    tr = conjuntos_treino(W)
    P = {ds: W[ds].copy() for ds in BASES}
    L_ = rotulo_L(JANELA_S)
    md5_dmd = md5_arquivo(FEAT / "dmd" / f"windows_{L_}.csv")
    md5_uta = md5_arquivo(FEAT / "uta" / f"windows_{L_}.csv")
    cand = [] if forcar_treino_dmd else modelos_exp5b(md5_dmd)
    log.info("Modelos do Exp. 5b com as janelas finais do DMD (md5 %s): %d entradas%s", md5_dmd, len(cand),
             " (ignorados: --treinar-dmd)" if forcar_treino_dmd else "")
    meta = dict(versao=VERSAO, sklearn=sklearn.__version__, receita=RECEITA, variantes=VARIANTES,
                limiar_disparo=LIMIAR_DISPARO, janela_s=JANELA_S, md5_windows_dmd=md5_dmd, md5_windows_uta=md5_uta,
                validacao_fora_da_amostra={"dmd": "deixa-um-sujeito-de-fora (todas as janelas do sujeito de fora)",
                                           "uta": "5 folds oficiais da UTA-RLDD (sujeitos inteiros)"},
                modelos={})
    imp_rows, inteiros = [], {}
    for fonte, (Xdf, y) in tr.items():
        for var, cols in VARIANTES.items():
            t0 = time.time()
            X = Xdf[cols].to_numpy(float)
            m, origem, e5 = None, None, None
            if fonte == "dmd":
                e5 = escolher_modelo(cand, cols, "RF")
                if e5 is not None:
                    m = joblib.load(e5["_arquivo"])
                    origem = (f"{e5['_arquivo'].relative_to(ROOT)} (Exp. 5b final, execução {e5.get('execucao', '?')}; "
                              f"metadados {e5['_json']})")
                    n_meta = e5.get("n_treino")
                    if n_meta is not None and int(n_meta) != len(y):
                        log.warning("RF do DMD (%s): o 5b registra %s janelas de treino; aqui o conjunto tem %d", var,
                                    n_meta, len(y))
                else:
                    log.info("RF do DMD (%s): nenhum modelo final do Exp. 5b com estas características -> treinado "
                             "aqui com a mesma receita", var)
            if m is None:
                m = novo_rf().fit(X, y)
                arq = MOD6 / f"rf_{fonte}_{var}.joblib"
                joblib.dump(m, arq, compress=3)
                origem = f"models/exp6/{arq.name} (treinado aqui, receita do Exp. 5b)"
            meta["modelos"][f"{fonte}_{var}"] = dict(
                origem=origem, n_janelas=int(len(y)), n_sonolento=int(y.sum()), n_alerta=int((y == 0).sum()),
                n_sujeitos=int(Xdf["sujeito"].nunique()), caracteristicas=cols, pipeline=repr(m),
                rotulo=("DMD: y (alerta = safe_drive da s2 = 0; sonolento = s5 com fechamento anotado >= 0,5 s ou "
                        "bocejo = 1); só janelas avaliáveis" if fonte == "dmd" else
                        "UTA-RLDD B3 (KSS 10 = 1, KSS 0 = 0; KSS 5 fora); só janelas avaliáveis (sem referência nem fronteira)"))
            inteiros[(fonte, var)] = m
            clf = m.named_steps["clf"] if hasattr(m, "named_steps") else m
            if hasattr(clf, "feature_importances_"):
                for f, v in zip(cols, clf.feature_importances_):
                    imp_rows.append(dict(fonte=fonte, variante=var, caracteristica=f, importancia_impureza=float(v)))
            for ds in BASES:                                   # modelo inteiro nas bases que ele não viu
                if ds != fonte:
                    P[ds][f"p_{fonte}_{var}"] = m.predict_proba(P[ds][cols].to_numpy(float))[:, 1]
            # no próprio domínio: fora da amostra com a MESMA receita (clone do modelo usado)
            col = f"p_{fonte}_{var}"
            P[fonte][col] = np.nan
            chave = "sujeito" if fonte == "dmd" else "fold"
            g_tr = Xdf[chave].to_numpy()
            for g in sorted(P[fonte][chave].dropna().unique()):
                msk = g_tr != g
                if len(np.unique(y[msk])) < 2:
                    continue
                mm = clone(m).fit(X[msk], y[msk])
                sel = (P[fonte][chave] == g).to_numpy()
                P[fonte].loc[sel, col] = mm.predict_proba(P[fonte].loc[sel, cols].to_numpy(float))[:, 1]
            # limiar calibrado só com o treino (regra do Exp. 5b): máxima acurácia balanceada nos escores fora da
            # amostra das janelas de treino (DMD: LOSO = os escores de A; UTA: fora do fold)
            sel_tr = (P[fonte]["avaliavel"] & P[fonte]["uso"].isin(["alerta", "sonolento"]) & P[fonte]["y"].notna()
                      if fonte == "dmd" else P[fonte]["avaliavel"] & P[fonte]["B3"].notna()).to_numpy()
            s_tr = P[fonte].loc[sel_tr, col].to_numpy(float)
            t_aqui, ba_aqui = limiar_otimo(y, s_tr)
            t_5b = e5.get("limiar_calibrado_dmd") if e5 is not None else None
            meta["modelos"][f"{fonte}_{var}"].update(
                limiar_calibrado_aqui=t_aqui, acur_bal_fora_da_amostra_no_limiar=ba_aqui,
                limiar_calibrado_exp5b=t_5b,
                limiar_calibrado=float(t_5b) if t_5b is not None else t_aqui,
                limiar_calibrado_origem=("metadados do Exp. 5b final (limiar_calibrado_dmd)" if t_5b is not None else
                                         "calculado aqui com a regra do Exp. 5b nos escores fora da amostra do treino"))
            log.info("RF %s/%s: %d janelas (%d sonolentas) de %d sujeitos; origem: %s; + fora da amostra por %s; "
                     "limiar calibrado aqui %.4f (acur. bal. %.3f)%s (%.1f s)", fonte, var, len(y), int(y.sum()),
                     Xdf["sujeito"].nunique(), origem, chave, t_aqui, ba_aqui,
                     f"; no Exp. 5b {float(t_5b):.4f}" if t_5b is not None else "", time.time() - t0)
    json.dump(meta, open(MOD6 / "modelos.json", "w"), ensure_ascii=False, indent=1)
    return P, pd.DataFrame(imp_rows), meta, inteiros, cand


GRUPO_SINAL = ["ear_dp", "amp_media", "vel_media"]   # características que o ruído/compressão do EAR afeta


def contrafactual(P, inteiros, L):
    """Diagnóstico: disparo noturno se UMA característica (ou o grupo de qualidade do sinal) tivesse o valor típico
    do dia. Duas referências: mediana das janelas alerta KSS 0 da UTA (avaliáveis) e mediana das janelas alerta da s2
    do DMD (a classe alerta de treino do modelo do DMD). Modelos inteiros, variante principal (brutas)."""
    cols = VARIANTES[VAR_PRINCIPAL]
    refs = {"uta_kss0": P["uta"][P["uta"]["aval"] & (P["uta"]["grupo"] == "KSS 0")],
            "dmd_alerta_s2": P["dmd"][P["dmd"]["aval"] & (P["dmd"]["grupo"] == "alerta")]}
    T = []
    for nome_ref, ref in refs.items():
        med = {c: float(ref[c].median()) for c in cols}
        for fonte in ("dmd", "uta"):
            m = inteiros[(fonte, VAR_PRINCIPAL)]
            for g in ("Microsleep", "Yawning"):
                df = P["nitymed"][P["nitymed"]["aval"] & (P["nitymed"]["grupo"] == g)]
                X = df[cols].to_numpy(float)
                base = float(np.mean(m.predict_proba(X)[:, 1] >= LIMIAR_DISPARO))
                trocas = [(c, [c]) for c in cols] + [("+".join(GRUPO_SINAL), GRUPO_SINAL)]
                for nome, cs in trocas:
                    Xc = X.copy()
                    for c in cs:
                        Xc[:, cols.index(c)] = med[c]
                    t = float(np.mean(m.predict_proba(Xc)[:, 1] >= LIMIAR_DISPARO))
                    T.append(dict(referencia=nome_ref, fonte=fonte, grupo=g, troca=nome,
                                  valor_dia=(med[cs[0]] if len(cs) == 1 else np.nan), taxa_original=base,
                                  taxa_contrafactual=t, delta=t - base, n_janelas=len(df)))
                    L.add("c", "nitymed", g, "taxa_disparo_contrafactual", t, n_janelas=len(df), modelo=FONTE_NOME[fonte],
                          variante=VAR_PRINCIPAL, nota=f"{nome} := mediana de {nome_ref}; original = {base:.4f}")
    return pd.DataFrame(T)


FONTE_NOME = {"dmd": "Modelo do DMD", "uta": "Modelo da UTA-RLDD (B3)"}   # a origem do modelo do DMD vai no relatório


def conferir_exp5b(P, cand):
    """Confere nossas probabilidades com as predições salvas pelo Exp. 5b final (results/exp5b_predicoes.csv): mesma
    receita e mesmos dados => mesmos números. Só roda quando há modelos finais do 5b com as janelas finais (cand)."""
    p = RES / "exp5b_predicoes.csv"
    if not p.exists() or not cand:
        return pd.DataFrame()
    js = min(os.path.getmtime(ROOT / e["_json"]) for e in cand)
    if os.path.getmtime(p) < js - 3600:
        log.warning("results/exp5b_predicoes.csv é mais antigo que os metadados dos modelos finais — conferência pulada")
        return pd.DataFrame()
    try:
        E = pd.read_csv(p, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("results/exp5b_predicoes.csv ilegível (%r)", exc)
        return pd.DataFrame()
    need = {"experimento", "variante", "video", "janela", "escore"}
    if not need.issubset(E.columns):
        log.info("results/exp5b_predicoes.csv sem as colunas %s — conferência pulada", sorted(need - set(E.columns)))
        return pd.DataFrame()
    if "modelo" in E.columns:
        E = E[E["modelo"] == "RF"]
    T = []
    for exp, ds, fonte, desc in (("C", "uta", "dmd", "modelo do DMD inteiro aplicado à UTA"),
                                 ("A", "dmd", "dmd", "modelo do DMD, fora da amostra (LOSO)"),
                                 ("B", "uta", "uta", "modelo da UTA B3, fora da amostra")):
        for var in ("sem_normalizacao", "principal"):
            e = E[E["experimento"].astype(str).str.upper().str.startswith(exp) & (E["variante"] == var)]
            if "esquema" in e.columns and ds == "uta":
                e = e[e["esquema"].isin(["B3", "todos"])]
            col = f"p_{fonte}_{var}"
            if e.empty or col not in P[ds].columns:
                continue
            chave = ["video", "janela"] + (["arquivo"] if "arquivo" in e.columns and "arquivo" in P[ds].columns else [])
            e = e.drop_duplicates(chave)
            mg = e.merge(P[ds][chave + [col]].drop_duplicates(chave), on=chave, how="left")
            d = (mg["escore"] - mg[col]).abs()
            T.append(dict(comparacao=desc, experimento_5b=exp, variante=var, n_janelas=int(len(mg)),
                          n_sem_par=int(mg[col].isna().sum()), dif_max=float(d.max()), dif_media=float(d.mean())))
    return pd.DataFrame(T)


def outros_modelos_exp5(P, cand):
    """Todos os modelos finais salvos pelo Exp. 5b (treinados no DMD) aplicados ao NITYMED e às janelas alerta/
    sonolentas de referência (fora do domínio de treino), só janelas avaliáveis."""
    import joblib
    out = []
    for e in cand:
        m = joblib.load(e["_arquivo"])
        cols = list(e["caracteristicas"])
        for ds, grupos in (("nitymed", ["Microsleep", "Yawning"]), ("uta", ["KSS 0", "KSS 10"]),
                           ("dmd", ["alerta_movimento"])):
            if not all(c in P[ds].columns for c in cols):
                continue
            for g in grupos:
                df = sel_grupo(P[ds], ds, g)
                pr = m.predict_proba(df[cols].to_numpy(float))[:, 1]
                est, lo, hi, nc, nj = taxa_boot(pr >= LIMIAR_DISPARO, df["cluster"].to_numpy())
                out.append(dict(arquivo=e["_arquivo"].name, modelo=e.get("modelo"), variante=e.get("variante"), base=ds,
                                grupo=g, taxa=est, lo=lo, hi=hi, n_agrup=nc, n_janelas=nj))
    return out


def _flags_regra(df):
    fch = df["dur_max_fechamento_s"].to_numpy(float) >= FECH_S
    boc = df["frac_bocejo"].to_numpy(float) > 0
    return fch, boc


def taxas_disparo(P, L, meta_mod):
    """Seção (c): frações de janelas/vídeos marcados, por detector × variante × grupo (só janelas `aval`)."""
    linhas = []
    detectores = [(f, v) for f in ("dmd", "uta") for v in VARIANTES]
    limiares_far = {}
    for fonte, var in detectores:
        col = f"p_{fonte}_{var}"
        u0 = P["uta"][P["uta"]["aval"] & (P["uta"]["grupo"] == "KSS 0")][col].dropna().to_numpy()
        tau = float(np.quantile(u0, 1 - FAR_ALVO, method="higher")) if len(u0) else np.nan
        limiares_far[(fonte, var)] = (tau, float(np.mean(u0 > tau)) if len(u0) else np.nan,
                                      float(np.mean(u0 >= tau)) if len(u0) else np.nan)
    for ds, g in GRUPOS_C:
        df = sel_grupo(P[ds], ds, g)
        if df.empty:
            continue
        cl = df["cluster"].to_numpy()
        fch, boc = _flags_regra(df)
        sem_ev_all = ~(fch | boc)
        for fonte, var in detectores:
            col = f"p_{fonte}_{var}"
            p = df[col].to_numpy(float)
            ok = np.isfinite(p)
            if not ok.any():
                continue
            fora = "fora da amostra (própria base)" if ds == fonte else "outra base (modelo inteiro)"
            est, lo, hi, nc, nj = taxa_boot(p[ok] >= LIMIAR_DISPARO, cl[ok])
            L.add("c", ds, g, "taxa_disparo_janela", est, lo, hi, df.loc[ok, "video"].nunique(), nj, nc,
                  FONTE_NOME[fonte], var, fora)
            t_cal = meta_mod["modelos"][f"{fonte}_{var}"]["limiar_calibrado"]
            ec, lc, hc, _, _ = taxa_boot(p[ok] >= t_cal, cl[ok])
            L.add("c", ds, g, "taxa_disparo_janela_limiar_calibrado", ec, lc, hc, df.loc[ok, "video"].nunique(), nj, nc,
                  FONTE_NOME[fonte], var, f"limiar calibrado no treino = {t_cal:.4f} "
                  f"({meta_mod['modelos'][f'{fonte}_{var}']['limiar_calibrado_origem']})")
            tau, far, _ = limiares_far[(fonte, var)]
            est2, lo2, hi2, _, _ = taxa_boot(p[ok] > tau, cl[ok])
            L.add("c", ds, g, "taxa_disparo_janela_far10", est2, lo2, hi2, df.loc[ok, "video"].nunique(), nj, nc,
                  FONTE_NOME[fonte], var, f"limiar {tau:.4f} (FAR alcançado na UTA KSS 0 = {far:.4f})")
            L.add("c", ds, g, "escore_mediano", float(np.median(p[ok])), n_janelas=nj, n_agrup=nc,
                  modelo=FONTE_NOME[fonte], variante=var)
            pv = df.loc[ok].assign(p=p[ok]).groupby("video").agg(media=("p", "mean"), maximo=("p", "max"),
                                                                 cl=("cluster", "first"))
            e3, l3, h3 = boot_cluster((pv["media"] >= LIMIAR_DISPARO).astype(float), pv["cl"])
            L.add("c", ds, g, "frac_videos_media_escore_gt_0_5", e3, l3, h3, len(pv), nj, pv["cl"].nunique(),
                  FONTE_NOME[fonte], var, "regra por vídeo da Entrega 01: média dos escores das janelas >= 0,5")
            e4, l4, h4 = boot_cluster((pv["maximo"] >= LIMIAR_DISPARO).astype(float), pv["cl"])
            L.add("c", ds, g, "frac_videos_alguma_janela", e4, l4, h4, len(pv), nj, pv["cl"].nunique(),
                  FONTE_NOME[fonte], var, "vídeo com pelo menos uma janela marcada")
            sem_ev = ok & sem_ev_all
            e5, l5, h5, _, n5 = taxa_boot(p[sem_ev] >= LIMIAR_DISPARO, cl[sem_ev])
            L.add("c", ds, g, "taxa_disparo_janela_sem_evento", e5, l5, h5, n_janelas=n5, modelo=FONTE_NOME[fonte],
                  variante=var, nota="só janelas sem fechamento >= 0,5 s e sem bocejo detectados pelos nossos sinais")
            linhas.append(dict(base=ds, grupo=g, fonte=fonte, variante=var, taxa=est, lo=lo, hi=hi, n_janelas=nj,
                               n_agrup=nc, n_videos=df.loc[ok, "video"].nunique(), taxa_far10=est2, lo_far10=lo2,
                               hi_far10=hi2, escore_mediano=float(np.median(p[ok])), videos_media=e3,
                               videos_media_lo=l3, videos_media_hi=h3, videos_alguma=e4, videos_alguma_lo=l4,
                               videos_alguma_hi=h4, amostra=fora, taxa_sem_evento=e5, lo_sem_evento=l5,
                               hi_sem_evento=h5, n_sem_evento=n5, frac_sem_evento=n5 / nj if nj else np.nan,
                               limiar_cal=t_cal, taxa_cal=ec, lo_cal=lc, hi_cal=hc))
        for nome, fl in (("regra_fechamento", fch), ("regra_bocejo", boc), ("regra_dmd", fch | boc)):
            est, lo, hi, nc, nj = taxa_boot(fl, cl)
            L.add("c", ds, g, "taxa_disparo_janela", est, lo, hi, df["video"].nunique(), nj, nc, nome, "",
                  "fechamento >= 0,5 s na janela e/ou MAR > 0,5 por >= 1 s (sem treino)")
            linhas.append(dict(base=ds, grupo=g, fonte=nome, variante="", taxa=est, lo=lo, hi=hi, n_janelas=nj,
                               n_agrup=nc, n_videos=df["video"].nunique()))
    for (fonte, var), (tau, far, empate) in limiares_far.items():
        L.add("c", "uta", "KSS 0", "limiar_far10", tau, modelo=FONTE_NOME[fonte], variante=var,
              nota=f"FAR alcançado = {far:.4f}; fração das janelas alerta com p >= limiar = {empate:.4f}")
    return pd.DataFrame(linhas), limiares_far


def sensibilidade_nitymed(P, F, L):
    """(c) sensibilidades no NITYMED, variante principal (brutas): (i) todas as janelas (inclui as do primeiro terço,
    referência e fronteira); (ii) grupo Bocejo SEM os vídeos suspeitos de pasta errada e Microssono COM eles."""
    sus = set(F.loc[F["suspeito_pasta"], "id"])
    todas = P["nitymed"]
    aval = todas[todas["aval"]]
    casos = [("Microsleep", "avaliáveis (principal)", aval, aval["grupo"] == "Microsleep"),
             ("Microsleep", "todas as janelas", todas, todas["grupo"] == "Microsleep"),
             ("Yawning", "avaliáveis (principal)", aval, aval["grupo"] == "Yawning"),
             ("Yawning", "todas as janelas", todas, todas["grupo"] == "Yawning"),
             ("Yawning", "avaliáveis, sem os suspeitos", aval, (aval["grupo"] == "Yawning") & ~aval["video"].isin(sus)),
             ("Yawning", "avaliáveis, só os suspeitos", aval, (aval["grupo"] == "Yawning") & aval["video"].isin(sus)),
             ("Microsleep", "avaliáveis, com os suspeitos", aval, (aval["grupo"] == "Microsleep") | aval["video"].isin(sus))]
    T = []
    for g, nome, base, msk in casos:
        d = base[msk.to_numpy()]
        cl = d["cluster"].to_numpy()
        fch, boc = _flags_regra(d)
        flags = {"dmd": d[f"p_dmd_{VAR_PRINCIPAL}"].to_numpy(float) >= LIMIAR_DISPARO,
                 "uta": d[f"p_uta_{VAR_PRINCIPAL}"].to_numpy(float) >= LIMIAR_DISPARO,
                 "regra_dmd": fch | boc}
        for det, fl in flags.items():
            est, lo, hi, nc, nj = taxa_boot(fl, cl)
            T.append(dict(grupo=g, conjunto=nome, detector=det, taxa=est, lo=lo, hi=hi, n_videos=nc, n_janelas=nj))
            L.add("c", "nitymed", g, "taxa_disparo_janela_sensibilidade", est, lo, hi, nc, nj, nc,
                  FONTE_NOME.get(det, det), VAR_PRINCIPAL if det != "regra_dmd" else "",
                  f"{nome}; suspeitos = pasta Yawning, > {SUSPEITO_DUR_S:.0f} s, nenhum bocejo detectado: {sorted(sus)}")
    return pd.DataFrame(T)


def medianas_caracteristicas(P):
    T = []
    for ds, g in GRUPOS_C_TAB:
        df = sel_grupo(P[ds], ds, g)
        r = dict(base=ds, grupo=g, n_janelas=len(df))
        for f in FEATS11:
            r[f] = float(df[f].median()) if df[f].notna().any() else np.nan
            r[f + "_nan"] = float(df[f].isna().mean()) if len(df) else np.nan
        T.append(r)
    return pd.DataFrame(T)


# ======================================================================================
# (e) ficha do NITYMED
# ======================================================================================
CITACOES_VERIFICADAS = [
    # conferidas em 27/09/2026 (rodada preliminar) na API do Crossref (artigos) e do DataCite (conjuntos de dados)
    dict(chave="petrellis2021dsd", tipo="artigo (exigido: opção 3 da lista dos autores)",
         texto="N. Petrellis, S. Zogas, P. Christakos, G. Keramidas, P. Mousouliotis, N. Voros, C. Antonopoulos. "
               "High Speed Implementation of the Deformable Shape Tracking Face Alignment Algorithm. "
               "2021 24th Euromicro Conference on Digital System Design (DSD), Palermo, 1–3 set. 2021, pp. 174–177.",
         doi="10.1109/DSD53832.2021.00035", fonte="Crossref"),
    dict(chave="petrellis2021vlsisoc", tipo="artigo (opção 1 da lista dos autores)",
         texto="N. Petrellis, P. Christakos, S. Zogas, P. Mousouliotis, G. Keramidas, N. Voros, C. Antonopoulos. "
               "Challenges Towards Hardware Acceleration of the Deformable Shape Tracking Application. "
               "2021 IFIP/IEEE 29th Int. Conf. on Very Large Scale Integration (VLSI-SoC), Singapura, out. 2021, pp. 1–4.",
         doi="10.1109/VLSI-SoC53125.2021.9606999", fonte="Crossref"),
    dict(chave="(não incluída no .bib)", tipo="artigo (opção 2 da lista dos autores)",
         texto="N. Petrellis, S. Zogas, P. Christakos, P. Mousouliotis, G. Keramidas, N. Voros, C. Antonopoulos. "
               "Software Acceleration of the Deformable Shape Tracking Application. ESSE 2021 (2nd European Symposium "
               "on Software Engineering), Larissa, ACM, 2021, pp. 51–57.",
         doi="10.1145/3501774.3501782", fonte="Crossref"),
    dict(chave="petrellis2022kaggle", tipo="conjunto de dados — cópia usada (Kaggle, opção 4)",
         texto="N. Petrellis, N. Voros, C. Antonopoulos, G. Keramidas, P. Christakos, P. Mousouliotis. NITYMED "
               "[Data set]. Kaggle, 2022.", doi="10.34740/KAGGLE/DSV/3921886", fonte="DataCite"),
    dict(chave="petrellis2022", tipo="conjunto de dados — registro no IEEE DataPort (já existia no .bib)",
         texto="N. Petrellis, N. Voros, C. Antonopoulos, G. Keramidas, P. Christakos, P. Mousouliotis. NITYMED. "
               "IEEE DataPort, 2022.", doi="10.21227/85xe-3f88", fonte="DataCite"),
]


def sha256(p, bloco=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b_ in iter(lambda: f.read(bloco), b""):
            h.update(b_)
    return h.hexdigest()


def ficha_nitymed(V, W, lum_ok):
    base = NITY / "DSM_Dataset-HDTV720"
    arqs = sorted(base.glob("*/*/HDTV720/*.mp4"))
    sig = pd.read_csv(ROOT / "data" / "signals" / "nitymed" / "index.csv").set_index("path")
    zpath = NITY / "nitymed_720p.zip"
    zinfo = {}
    if zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            zinfo = {i.filename: i.file_size for i in z.infolist() if not i.is_dir()}
    vn = V[(V["dataset"] == "nitymed") & (V["recorte"] == "inteiro")].set_index("video")
    wn = W["nitymed"]
    rows = []
    for p in arqs:
        rel = str(p.relative_to(ROOT))
        cat, gen = p.parts[-4], p.parts[-3]
        vid = f"{cat}_{gen}_{p.stem}"
        sr = sig.loc[rel] if rel in sig.index else None
        tam = p.stat().st_size
        zrel = str(p.relative_to(NITY))
        fps = float(sr["fps"]) if sr is not None else np.nan
        nq = int(sr["n_frames_meta"]) if sr is not None else np.nan
        durs = nq / fps if sr is not None and fps > 0 else np.nan
        wv = wn[wn["video"] == vid]
        v = vn.loc[vid] if vid in vn.index else None
        rows.append(dict(
            id=vid, categoria=cat, genero=gen, arquivo=p.name, caminho=rel, tamanho_bytes=tam, sha256=sha256(p),
            no_zip=zrel in zinfo, tamanho_confere_zip=(zinfo.get(zrel) == tam) if zinfo else None,
            codec=(v["codec"] if v is not None and lum_ok else None),
            fps=fps, n_quadros_conteiner=nq,
            n_quadros_lidos=int(sr["n_frames"]) if sr is not None else np.nan, duracao_s=durs,
            largura=int(sr["width_orig"]) if sr is not None else np.nan,
            altura=int(sr["height_orig"]) if sr is not None else np.nan,
            taxa_bits_mbps=(tam * 8 / durs / 1e6) if np.isfinite(durs) and durs > 0 else np.nan,
            taxa_rosto=float(v["taxa_rosto"]) if v is not None else np.nan,
            n_bocejos_detectados=int(v["n_bocejos"]) if v is not None else np.nan,
            n_janelas_10s=int(len(wv)), n_janelas_10s_avaliaveis=int(wv["avaliavel"].sum()) if len(wv) else 0,
            janela_parcial=bool(wv["parcial"].any()) if len(wv) else False,
            n_janelas_com_bocejo=int((wv["frac_bocejo"] > 0).sum()) if len(wv) else 0,
            lum_rosto=(float(v["lum_rosto"]) if v is not None and lum_ok else np.nan)))
    F = pd.DataFrame(rows)
    F["suspeito_pasta"] = ((F["categoria"] == "Yawning") & (F["duracao_s"] > SUSPEITO_DUR_S)
                           & (F["n_bocejos_detectados"] == 0))
    out = NITY / "index.csv"
    tmp = out.with_suffix(".tmp.csv")
    F.to_csv(tmp, index=False)
    os.replace(tmp, out)
    log.info("Ficha NITYMED: %d vídeos -> %s", len(F), out.relative_to(ROOT))
    return F, zinfo


def chaves_bib():
    bib = ROOT / "paper" / "refs.bib"
    if not bib.exists():
        return {}
    s = bib.read_text(encoding="utf-8")
    return {c["chave"]: (f"{{{c['chave']},") in s for c in CITACOES_VERIFICADAS if not c["chave"].startswith("(")}


def zpath_size():
    z = NITY / "nitymed_720p.zip"
    return z.stat().st_size if z.exists() else 0


def escrever_ficha(F, zinfo):
    n = len(F)
    ct = F.groupby(["categoria", "genero"]).agg(videos=("id", "size"), dur_total_s=("duracao_s", "sum"),
                                                 dur_mediana_s=("duracao_s", "median"), dur_min_s=("duracao_s", "min"),
                                                 dur_max_s=("duracao_s", "max"), quadros=("n_quadros_conteiner", "sum"),
                                                 bytes_=("tamanho_bytes", "sum"))
    L = []
    L.append("# Ficha do dataset NITYMED (cópia usada no TCC)\n")
    L.append(f"Gerado por `src/experiments/exp6_noite.py` (rodada final) em {time.strftime('%d/%m/%Y')} a partir dos arquivos em disco "
             "(`data/nitymed/`), dos sinais extraídos (`data/signals/nitymed/index.csv`), das medidas por vídeo do Exp. 6 "
             "(`results/exp6_noite_videos.csv`) e da luminância medida (`results/exp6_luminancia_videos.csv`). "
             "Índice por vídeo: `data/nitymed/index.csv` (inclui SHA-256 de cada arquivo). A rodada preliminar "
             "(janelas de 30 s) foi descartada; seus arquivos foram removidos em 06/10/2026.\n")
    L.append("## 1. Identificação\n")
    L.append("| Campo | Valor | Fonte |\n|---|---|---|")
    L.append("| Nome | NITYMED — NIght Time Yawning-Microsleep-Eyeblink-driver Distraction | Kaggle `nikospetrellis/nitymed`; IEEE DataPort |")
    L.append("| Autores | Nikos Petrellis, Nikolaos Voros, Christos Antonopoulos, Georgios Keramidas, Panagiotis Christakos, Panagiotis Mousouliotis | DataCite (DOIs 10.21227/85xe-3f88 e 10.34740/KAGGLE/DSV/3921886) |")
    L.append("| Cópia usada | Kaggle, versão 1 (\"Initial release\", 09/07/2022), arquivo `nitymed_720p.zip` (HDTV720) | API do Kaggle, consultada em 27/09/2026 (rodada preliminar) |")
    L.append("| Conteúdo declarado | 130 vídeos: \"the drivers yawn 3 times in each video lasting approximately 15-25 seconds (107 videos)\" e \"the drivers talk, look around and have microsleeps in videos lasting approximately 2 minutes (21 videos)\"; 21 motoristas (11 homens, 10 mulheres); Patras, Grécia; carros reais em movimento, à noite; mp4, 25 quadros/s, sem áudio; 720p (Kaggle) e 1080p (site da universidade) | página do IEEE DataPort (lida pelo orquestrador em 27/09/2026) e descrição do Kaggle |")
    L.append("| Iluminação declarada | \"natural, com um leve reforço pelas luzes internas mais fracas do carro, para simular a iluminação de uma avenida\"; a maioria dos vídeos foi gravada numa estrada escura e pouco movimentada | descrição no Kaggle |")
    L.append("| Autenticidade dos comportamentos | **NÃO documentada**: a fonte primária não diz se bocejos e microssonos foram espontâneos ou encenados; o número fixo de 3 bocejos por vídeo sugere execução sob instrução. Enquadramento: condução noturna REAL com comportamentos de autenticidade não documentada — serve para a condição NOITE da PP3 (captura, sinal, disparo), **não** como evidência de sonolência real para a PP1 | página do IEEE DataPort (27/09/2026) |")
    L.append("")
    L.append("## 2. Licença e citação\n")
    L.append("- **Licença da cópia usada (Kaggle): CC BY-ND 4.0** (Attribution-NoDerivatives 4.0 International) — campo `licenseName` da API do Kaggle.")
    L.append("- A página do IEEE DataPort não traz licença padronizada. Como a cópia baixada é a do Kaggle, vale a **BY-ND**.")
    L.append("- CC BY-ND permite reproduzir quadros **sem modificação** e com atribuição (recortes, anotações sobrepostas ou borrões são modificações). "
             "**Nenhuma figura com rosto foi gerada**: a decisão de publicar quadros é da equipe. As figuras do Exp. 6 só têm números e séries temporais.")
    L.append("- Exigência dos autores: \"free to be used provided that one or more of the following papers are cited\". Metadados conferidos em 27/09/2026 (Crossref/DataCite, rodada preliminar; nenhuma consulta externa nesta rodada):\n")
    L.append("| Chave no `paper/refs.bib` | Tipo | Referência | DOI | Conferido em |\n|---|---|---|---|---|")
    presentes = chaves_bib()
    for c in CITACOES_VERIFICADAS:
        est = ""
        if c["chave"] in presentes:
            est = " (presente no .bib)" if presentes[c["chave"]] else " (AUSENTE do .bib)"
        L.append(f"| `{c['chave']}`{est} | {c['tipo']} | {c['texto']} | {c['doi']} | {c['fonte']} |")
    L.append("\nRecomendação: citar o conjunto de dados (`petrellis2022`, IEEE DataPort, ou `petrellis2022kaggle`, a cópia usada) e **pelo menos um** dos artigos (`petrellis2021dsd` ou `petrellis2021vlsisoc`). "
             "Divergência registrada: a lista do Kaggle diz VLSI-SoC \"4–8 October 2021\"; o Crossref registra o evento de 4 a 7/10/2021 — o .bib segue o Crossref.\n")
    L.append("## 3. Estrutura em disco\n")
    L.append("```\ndata/nitymed/\n├── nitymed_720p.zip                      (" + fint(zpath_size()) + " bytes)\n"
             "└── DSM_Dataset-HDTV720/\n    ├── Microsleep/{Female,Male}/HDTV720/P10xxxxx_720.mp4\n"
             "    └── Yawning/{Female,Male}/HDTV720/P10xxxxx_720.mp4\n```\n")
    L.append(f"- {n} arquivos .mp4 em disco; {len(zinfo)} arquivos no .zip; todos os do disco estão no .zip: "
             f"{'sim' if F['no_zip'].all() else 'NÃO'}; tamanhos conferem: {'sim' if F['tamanho_confere_zip'].all() else 'NÃO'}.")
    L.append("- Rótulo = pasta (categoria); gênero = pasta. **Não há identificador de motorista** nos nomes (P10xxxxx = número do arquivo da câmera): "
             "vídeos do mesmo motorista não podem ser agrupados — qualquer estatística por vídeo trata como independentes vídeos que não são.")
    L.append(f"- Contagem real × declarada: {n} × 130 vídeos; Microssono {int((F['categoria'] == 'Microsleep').sum())} × 21; "
             f"Bocejo {int((F['categoria'] == 'Yawning').sum())} × 107. O total de bytes do Kaggle (API: `totalBytes` = 697.049.508) "
             f"{'coincide' if sum(zinfo.values()) == 697049508 else 'NÃO coincide'} com a soma dos {len(zinfo)} arquivos do .zip "
             f"({fint(sum(zinfo.values()))} bytes): a diferença para 130 já está na versão publicada, não é falha de download.\n")
    L.append("## 4. Contagens por categoria e gênero\n")
    L.append("| Categoria | Gênero | Vídeos | Duração total (min) | Duração mediana (s) | Mín–máx (s) | Quadros | Tamanho (MB) |\n|---|---|---:|---:|---:|---:|---:|---:|")
    for (cat, gen), r in ct.iterrows():
        L.append(f"| {cat} | {gen} | {int(r['videos'])} | {fnum(r['dur_total_s'] / 60, 1)} | {fnum(r['dur_mediana_s'], 1)} | "
                 f"{fnum(r['dur_min_s'], 1)}–{fnum(r['dur_max_s'], 1)} | {fint(r['quadros'])} | {fnum(r['bytes_'] / 1e6, 1)} |")
    L.append(f"| **Total** | | {n} | {fnum(F['duracao_s'].sum() / 60, 1)} | {fnum(F['duracao_s'].median(), 1)} | "
             f"{fnum(F['duracao_s'].min(), 1)}–{fnum(F['duracao_s'].max(), 1)} | {fint(F['n_quadros_conteiner'].sum())} | {fnum(F['tamanho_bytes'].sum() / 1e6, 1)} |\n")
    curtos = F[(F["categoria"] == "Microsleep") & (F["duracao_s"] < 60)]
    if len(curtos):
        L.append("Vídeos de microssono com menos de 60 s (fora do \"~2 min\" declarado): " +
                 ", ".join(f"`{r.arquivo}` ({fnum(r.duracao_s, 1)} s)" for r in curtos.itertuples()) + ".\n")
    yw = F[F["categoria"] == "Yawning"]
    L.append(f"Duração dos vídeos de bocejo × declarada (15–25 s): {int(yw['duracao_s'].between(15, 25).sum())} de {len(yw)} dentro; "
             f"{int((yw['duracao_s'] < 15).sum())} abaixo; {int((yw['duracao_s'] > 25).sum())} acima (mediana {fnum(yw['duracao_s'].median(), 1)} s); "
             f"{int((yw['duracao_s'] > 60).sum())} com mais de 60 s.\n")
    sus = F[F["suspeito_pasta"]]
    ok_y = yw[~yw["suspeito_pasta"]]
    L.append(f"Bocejos detectados nos nossos sinais (MAR > 0,5 sustentado ≥ 1 s, a definição de `frac_bocejo`, contados no vídeo inteiro): "
             f"mediana {fnum(ok_y['n_bocejos_detectados'].median(), 0)} por vídeo de bocejo (fora os suspeitos abaixo); "
             f"{int((yw['n_bocejos_detectados'] == 0).sum())} dos {len(yw)} vídeos de bocejo sem nenhum; "
             f"{int((yw['n_bocejos_detectados'] >= 3).sum())} com 3 ou mais (declarado: 3 por vídeo).\n")
    if len(sus):
        L.append(f"**Possível erro de pasta na versão publicada**: {len(sus)} vídeos da pasta Yawning têm duração de vídeo de microssono "
                 f"(> {SUSPEITO_DUR_S:.0f} s) e **nenhum bocejo detectado** no vídeo inteiro: " +
                 ", ".join(f"`{r.categoria}/{r.genero}/{r.arquivo}` ({fnum(r.duracao_s, 1)} s)" for r in sus.itertuples()) +
                 ". Pela duração e pela ausência de bocejo, parecem vídeos de microssono na pasta errada (com eles, o Microssono "
                 f"teria {int((F['categoria'] == 'Microsleep').sum()) + len(sus)} vídeos × 21 declarados). NÃO confirmado visualmente nem "
                 "pelos autores: os rótulos de pasta foram mantidos, e o efeito nas taxas de disparo está em `results/exp6_noite.md` "
                 "(sensibilidade da seção 5).\n")
    L.append("## 5. Formato\n")
    res = F.groupby(["largura", "altura"]).size()
    L.append("| Campo | Valor |\n|---|---|")
    L.append("| Resolução | " + "; ".join(f"{int(a)}×{int(b)} ({c} vídeos)" for (a, b), c in res.items()) + " |")
    L.append("| fps (contêiner) | " + "; ".join(f"{fnum(k, 2)} ({v} {'vídeo' if v == 1 else 'vídeos'})" for k, v in F["fps"].value_counts().sort_index().items())
             + ("; fora de 25: " + ", ".join(f"`{r.arquivo}`" for r in F[F["fps"] != 25].itertuples()) if (F["fps"] != 25).any() else "") + " |")
    L.append("| Codec (OpenCV, FOURCC) | " + ("; ".join(f"{k} ({v})" for k, v in F["codec"].value_counts().items()) if F["codec"].notna().any() else "— (rode exp6_luminancia.py)") + " |")
    L.append(f"| Taxa de bits | mediana {fnum(F['taxa_bits_mbps'].median(), 2)} Mbit/s (mín {fnum(F['taxa_bits_mbps'].min(), 2)}, máx {fnum(F['taxa_bits_mbps'].max(), 2)}) |")
    L.append(f"| Quadros decodificados = quadros do contêiner | {int((F['n_quadros_lidos'] == F['n_quadros_conteiner']).sum())} de {n} vídeos |")
    L.append(f"| Janelas de 10 s (passo 5 s) | {fint(F['n_janelas_10s'].sum())}, das quais {fint(F['n_janelas_10s_avaliaveis'].sum())} avaliáveis "
             f"(fora do primeiro terço, que é a referência; {int(F['janela_parcial'].sum())} vídeos com janela parcial); vídeos sem nenhuma janela avaliável: "
             f"{int((F['n_janelas_10s_avaliaveis'] == 0).sum())} |")
    L.append(f"| Taxa de detecção de rosto (Face Mesh) | mediana por vídeo {fnum(F['taxa_rosto'].median(), 4)}, mín {fnum(F['taxa_rosto'].min(), 4)} |")
    if F["lum_rosto"].notna().any():
        L.append(f"| Luminância média do rosto (0–255) | mediana por vídeo {fnum(F['lum_rosto'].median(), 1)} (mín {fnum(F['lum_rosto'].min(), 1)}, máx {fnum(F['lum_rosto'].max(), 1)}) |")
    L.append("")
    L.append("## 6. Uso no TCC\n")
    L.append("- Só na **PP3 (condição noite)**: detecção de rosto, qualidade do sinal de EAR e taxa de disparo dos detectores (Exp. 6, `results/exp6_noite.md`).")
    L.append("- **Não** serve como evidência de sonolência real para a PP1 (autenticidade não documentada) e **não** permite acurácia: não há classe alerta nem anotação quadro a quadro.")
    L.append("- A referência de normalização e do limiar de EAR no NITYMED é o primeiro terço do próprio vídeo (não existe vídeo alerta) — desvio documentado no Exp. 5.")
    return "\n".join(L) + "\n"


# ======================================================================================
# Resumos por seção
# ======================================================================================
def _m(V, ds, rec="inteiro"):
    return (V["dataset"] == ds) & (V["recorte"] == rec)


# linhas das tabelas (a)/(b)/(d): chave -> (rótulo, filtro)
LINHAS = {
    "nitymed|todos": ("**NITYMED (noite, carro em movimento)**", lambda V: _m(V, "nitymed")),
    "nitymed|Microsleep": ("Microssono", lambda V: _m(V, "nitymed") & (V["grupo"] == "Microsleep")),
    "nitymed|Yawning": ("Bocejo", lambda V: _m(V, "nitymed") & (V["grupo"] == "Yawning")),
    "uta|todos": ("**UTA-RLDD (autogravado)**", lambda V: _m(V, "uta")),
    "uta|KSS 0": ("KSS 0 (alerta)", lambda V: _m(V, "uta") & (V["grupo"] == "KSS 0")),
    "uta|KSS 5": ("KSS 5", lambda V: _m(V, "uta") & (V["grupo"] == "KSS 5")),
    "uta|KSS 10": ("KSS 10 (sonolento)", lambda V: _m(V, "uta") & (V["grupo"] == "KSS 10")),
    "dmd_s5|todos": ("**DMD s5 (dia, carro parado, sonolência encenada)**", lambda V: _m(V, "dmd") & (V["sessao"] == "s5")),
    "dmd_s5|sem óculos": ("sem óculos", lambda V: _m(V, "dmd") & (V["sessao"] == "s5") & ~V["oculos"]),
    "dmd_s5|com óculos": ("com óculos", lambda V: _m(V, "dmd") & (V["sessao"] == "s5") & V["oculos"]),
    "dmd_mov|safe": ("**DMD, direção segura com o carro em movimento (dia)**",
                     lambda V: _m(V, "dmd", "safe_drive") & V["sessao"].isin(["s1", "s2"]) & (V["setup"] == MOVIMENTO)),
    "dmd_par|safe": ("**DMD, direção segura com o carro parado (dia)**",
                     lambda V: _m(V, "dmd", "safe_drive") & V["sessao"].isin(["s1", "s2"]) & (V["setup"] == PARADO)),
    "dmd_mov|inteiro": ("DMD, carro em movimento, sessão inteira (com as distrações encenadas)",
                        lambda V: _m(V, "dmd") & V["sessao"].isin(["s1", "s2"]) & (V["setup"] == MOVIMENTO)),
    "dmd_s1|safe": ("DMD s1, direção segura (protocolo: carro em movimento)",
                    lambda V: _m(V, "dmd", "safe_drive") & (V["sessao"] == "s1")),
    "dmd_s2|safe": ("DMD s2, direção segura (protocolo: carro parado)",
                    lambda V: _m(V, "dmd", "safe_drive") & (V["sessao"] == "s2")),
}
LINHAS_A = [k for k in LINHAS if k not in ("dmd_s1|safe", "dmd_s2|safe")]
LINHAS_B = [k for k in LINHAS if k not in ("dmd_mov|inteiro", "dmd_s1|safe", "dmd_s2|safe")]
LINHAS_D = ["nitymed|Microsleep", "nitymed|Yawning", "uta|KSS 0", "uta|KSS 5", "uta|KSS 10", "dmd_s5|todos",
            "dmd_mov|safe", "dmd_par|safe"]
CONTRASTES = [("nitymed|todos", "uta|todos"), ("nitymed|todos", "dmd_s5|todos"), ("nitymed|todos", "dmd_mov|safe"),
              ("nitymed|todos", "dmd_par|safe")]
PAREADOS = [("dmd_mov|safe", "dmd_par|safe", "setup (Car Moving − Car Stopped), sessões s1/s2"),
            ("dmd_s1|safe", "dmd_s2|safe", "sessão do protocolo (s1 − s2)")]
NOME_CURTO = {"uta|todos": "UTA-RLDD", "dmd_s5|todos": "DMD s5 (parado)", "dmd_mov|safe": "DMD em movimento",
              "dmd_par|safe": "DMD parado", "nitymed|todos": "NITYMED", "dmd_s1|safe": "DMD s1", "dmd_s2|safe": "DMD s2"}


def sub(V, chave):
    return V[LINHAS[chave][1](V)]


def base_da_linha(chave):
    return chave.split("|")[0].split("_")[0]


def boot_pareado_sujeito(V, k1, k2, metrica, n_boot=N_BOOT, seed=SEED):
    """Diferença k1 − k2 pareada por sujeito (média das sessões de cada sujeito em cada condição);
    estimativa = mediana das diferenças por sujeito; IC 95% por bootstrap de sujeitos."""
    a = sub(V, k1).groupby("sujeito")[metrica].mean()
    b = sub(V, k2).groupby("sujeito")[metrica].mean()
    d = (a - b).dropna()
    if len(d) < 2:
        return np.nan, np.nan, np.nan, len(d)
    v = d.to_numpy()
    rng = np.random.default_rng(seed)
    bs = np.array([np.median(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(np.median(v)), float(lo), float(hi), len(d)


def sinais_pareados(V, k1, k2, metrica):
    """Nº de sujeitos com diferença k1 − k2 positiva e negativa (média das sessões de cada condição)."""
    d = (sub(V, k1).groupby("sujeito")[metrica].mean() - sub(V, k2).groupby("sujeito")[metrica].mean()).dropna()
    return int((d > 0).sum()), int((d < 0).sum())


def janelas_da_linha(W, chave):
    """Janelas (não referência, não fronteira) dos vídeos de uma linha da tabela (d)."""
    ds = base_da_linha(chave)
    w = W[ds]
    if ds in ("nitymed", "uta"):
        g = chave.split("|")[1]
        sel = w["aval"] if g == "todos" else (w["aval"] & (w["grupo"] == g))
        return w[sel]
    if chave.startswith("dmd_s5"):
        return w[w["uso"].isin(["sonolento", "s5_sem_sinal"])]
    if chave == "dmd_mov|safe":
        return w[w["grupo_setup"] == "alerta_movimento"]
    if chave == "dmd_par|safe":
        return w[w["grupo_setup"] == "alerta_parado"]
    if chave == "dmd_s1|safe":
        return w[w["uso"] == "alerta_s1"]
    if chave == "dmd_s2|safe":
        return w[w["uso"] == "alerta"]
    return w.iloc[0:0]


def resumo_a(V, W, L):
    T = []
    for chave in LINHAS_A:
        s = sub(V, chave)
        if s.empty:
            continue
        fr = s["taxa_rosto"].to_numpy()
        m, lo, hi = boot_cluster(fr, s["cluster"])
        ds = base_da_linha(chave)
        wv = W[ds][W[ds]["video"].isin(s["video"])]
        if chave in ("dmd_mov|safe", "dmd_par|safe", "dmd_s1|safe", "dmd_s2|safe"):
            wv = janelas_da_linha(W, chave)
        elif chave == "dmd_mov|inteiro":         # as janelas da distração só existem dentro dos trechos safe_drive
            wv = wv.iloc[0:0]
        r = dict(chave=chave, base=ds, n_videos=len(s), n_agrup=s["cluster"].nunique(),
                 n_quadros=int(s["n_quadros"].sum()), media=m, lo=lo, hi=hi, mediana=float(np.median(fr)),
                 minimo=float(fr.min()), p10=float(np.percentile(fr, 10)),
                 ponderada=float((s["taxa_rosto"] * s["n_quadros"]).sum() / s["n_quadros"].sum()),
                 n_lt_095=int((fr < 0.95).sum()), n_lt_099=int((fr < 0.99).sum()),
                 n_janelas=len(wv), frac_jan_lt_090=float((wv["taxa_rosto"] < 0.90).mean()) if len(wv) else np.nan,
                 maior_lacuna_mediana=float(s["maior_lacuna_s"].median()), maior_lacuna_max=float(s["maior_lacuna_s"].max()),
                 lacunas05_por_min_media=float(s["lacunas_05s_por_min"].mean()),
                 id_minimo=str(s.loc[s["taxa_rosto"].idxmin(), "video"]))
        T.append(r)
        L.add("a", ds, chave, "taxa_rosto_media_por_video", m, lo, hi, r["n_videos"], r["n_janelas"], r["n_agrup"])
        for k in ("mediana", "minimo", "p10", "ponderada"):
            L.add("a", ds, chave, f"taxa_rosto_{k}", r[k], n_videos=r["n_videos"], n_agrup=r["n_agrup"],
                  nota=("ponderada pelo nº de quadros" if k == "ponderada" else "por vídeo"))
        L.add("a", ds, chave, "n_videos_taxa_rosto_lt_0_95", r["n_lt_095"], n_videos=r["n_videos"])
        L.add("a", ds, chave, "n_videos_taxa_rosto_lt_0_99", r["n_lt_099"], n_videos=r["n_videos"])
        L.add("a", ds, chave, "frac_janelas_taxa_rosto_lt_0_90", r["frac_jan_lt_090"], n_janelas=r["n_janelas"],
              nota="todas as janelas de 10 s dos vídeos da linha (DMD s1/s2: as janelas de direção segura)")
        L.add("a", ds, chave, "maior_lacuna_sem_rosto_s_mediana", r["maior_lacuna_mediana"], n_videos=r["n_videos"])
        L.add("a", ds, chave, "maior_lacuna_sem_rosto_s_max", r["maior_lacuna_max"], n_videos=r["n_videos"])
        L.add("a", ds, chave, "lacunas_ge_0_5s_por_min_media", r["lacunas05_por_min_media"], n_videos=r["n_videos"])
    A = pd.DataFrame(T)
    dif = []
    for ka, kb in CONTRASTES:
        sa, sb = sub(V, ka), sub(V, kb)
        e, lo, hi = boot_dif(sa["taxa_rosto"], sa["cluster"], sb["taxa_rosto"], sb["cluster"])
        dif.append(dict(metrica="taxa_rosto", a=ka, b=kb, dif=e, lo=lo, hi=hi))
        L.add("a", "nitymed", ka, f"dif_taxa_rosto_media_{ka}_menos_{kb}", e, lo, hi,
              nota="média por vídeo; bootstrap independente por agrupamento em cada base")
    for ka, kb, desc in PAREADOS:
        e, lo, hi, n = boot_pareado_sujeito(V, ka, kb, "taxa_rosto")
        dif.append(dict(metrica="taxa_rosto", a=ka, b=kb, dif=e, lo=lo, hi=hi, pareado=True, n=n, desc=desc))
        L.add("a", "dmd", ka, f"dif_pareada_taxa_rosto_{ka}_menos_{kb}", e, lo, hi, n_agrup=n,
              nota=f"{desc}; mediana das diferenças por sujeito (média das sessões); bootstrap de sujeitos")
    return A, pd.DataFrame(dif)


MET_B = [("ear_aberto_media", "EAR médio com olho aberto", 3), ("ear_aberto_dp", "DP do EAR com olho aberto", 4),
         ("ruido_rapido", "Ruído rápido do EAR (DP robusto, resíduo de 1 s)", 4),
         ("amp_ext_media", "Amplitude média das piscadas (estendida)", 4),
         ("amp_ext_relativa", "Amplitude ÷ nível aberto", 3), ("snr", "SNR = amplitude ÷ DP aberto", 2),
         ("snr_rapido", "SNR rápida = amplitude ÷ ruído rápido", 2),
         ("taxa_piscadas_min", "Piscadas por minuto", 1), ("prof_rel_p90_vales", "Profundidade relativa p90 dos vales", 3),
         ("vales12_por_min", "Quedas ≥ 12% por minuto", 1), ("olho_largura_px", "Largura do olho (px processados)", 1),
         ("lum_rosto", "Luminância do rosto (0–255)", 1), ("lum_olhos", "Luminância dos olhos (0–255)", 1),
         ("contraste_rosto", "Contraste RMS do rosto", 1), ("frac_escuro_rosto", "Fração do rosto com cinza < 40", 3),
         ("taxa_bits_mbps", "Taxa de bits do arquivo (Mbit/s)", 2), ("bits_por_pixel", "Bits por pixel (compressão)", 3)]
MET_PAREADO = ["ruido_rapido", "snr_rapido", "ear_aberto_dp", "amp_ext_media", "taxa_rosto", "lum_rosto"]


def resumo_b(V, L):
    T = []
    for chave in LINHAS_B:
        s = sub(V, chave)
        if s.empty:
            continue
        r = dict(chave=chave, base=base_da_linha(chave), n_videos=len(s), n_agrup=s["cluster"].nunique())
        for k, _, _ in MET_B:
            v = s[k].to_numpy(float) if k in s.columns else np.full(len(s), np.nan)
            e, lo, hi = boot_cluster(v, s["cluster"], stat=np.median)
            r[k], r[k + "_lo"], r[k + "_hi"] = e, lo, hi
            r[k + "_n"] = int(np.isfinite(v).sum())
            L.add("b", r["base"], chave, f"{k}_mediana_por_video", e, lo, hi, int(np.isfinite(v).sum()), n_agrup=r["n_agrup"])
        T.append(r)
    Bq = pd.DataFrame(T)
    dif = []
    for ka, kb in CONTRASTES:
        sa, sb = sub(V, ka), sub(V, kb)
        for k, rot, _ in MET_B:
            if k not in V.columns or sa[k].isna().all() or sb[k].isna().all():
                continue
            e, lo, hi = boot_dif(sa[k], sa["cluster"], sb[k], sb["cluster"], stat=np.median)
            dif.append(dict(metrica=k, rotulo=rot, a=ka, b=kb, dif=e, lo=lo, hi=hi))
            L.add("b", "nitymed", ka, f"dif_mediana_{k}_{ka}_menos_{kb}", e, lo, hi,
                  nota="diferença de medianas por vídeo; bootstrap independente por agrupamento")
    rot = dict((a, b) for a, b, _ in MET_B)
    rot["taxa_rosto"] = "Taxa de detecção de rosto"
    for ka, kb, desc in PAREADOS:
        for k in MET_PAREADO:
            e, lo, hi, n = boot_pareado_sujeito(V, ka, kb, k)
            n_pos, n_neg = sinais_pareados(V, ka, kb, k)
            dif.append(dict(metrica=k, rotulo=rot.get(k, k), a=ka, b=kb, dif=e, lo=lo, hi=hi, pareado=True, n=n,
                            n_pos=n_pos, n_neg=n_neg, desc=desc))
            L.add("b", "dmd", ka, f"dif_pareada_{k}_{ka}_menos_{kb}", e, lo, hi, n_agrup=n,
                  nota=f"{desc}; mesmos sujeitos, direção segura; mediana das diferenças por sujeito")
    return Bq, pd.DataFrame(dif)


def correlacoes(V, L):
    from scipy.stats import spearmanr
    pares = [("lum_rosto", "taxa_rosto"), ("lum_rosto", "snr_rapido"), ("olho_largura_px", "snr_rapido"),
             ("contraste_rosto", "snr_rapido"), ("lum_olhos", "ruido_rapido"), ("olho_largura_px", "taxa_rosto"),
             ("bits_por_pixel", "ruido_rapido")]
    T = []
    for chave in ("nitymed|todos", "uta|todos"):
        s = sub(V, chave)
        for x, y in pares:
            if x not in s.columns or s[x].isna().all():
                continue
            ok = s[x].notna() & s[y].notna()
            if ok.sum() < 8:
                continue
            rho, p = spearmanr(s.loc[ok, x], s.loc[ok, y])
            T.append(dict(base=base_da_linha(chave), x=x, y=y, rho=float(rho), p=float(p), n=int(ok.sum())))
            L.add("b", base_da_linha(chave), "todos", f"spearman_{x}_x_{y}", float(rho), n_videos=int(ok.sum()),
                  nota=f"p = {p:.3g} (vídeos não independentes: IC/p otimistas)")
    return pd.DataFrame(T)


def resumo_d(V, W, L):
    T = []
    for chave in LINHAS_D:
        s = sub(V, chave)
        if s.empty:
            continue
        wv = janelas_da_linha(W, chave)
        r = dict(chave=chave, base=base_da_linha(chave), n_videos=len(s), minutos=float(s["dur_s"].sum() / 60))
        for k in ("perclos", "perclos_p90", "fech_05s_por_min", "fech_05s_por_min_p90", "dur_max_fech_s"):
            e, lo, hi = boot_cluster(s[k], s["cluster"], stat=np.median)
            r[k], r[k + "_lo"], r[k + "_hi"] = e, lo, hi
            r[k + "_p90v"] = float(np.nanpercentile(s[k], 90))
            L.add("d", r["base"], chave, f"{k}_mediana_por_video", e, lo, hi, len(s), n_agrup=s["cluster"].nunique())
        r["frac_videos_fech05"] = float((s["n_fech_05s"] > 0).mean())
        r["frac_videos_fech_longo"] = float((s["n_fech_longos"] > 0).mean())
        r["n_fech_05s_total"] = int(s["n_fech_05s"].sum())
        r["n_fech_longos_total"] = int(s["n_fech_longos"].sum())
        r["frac_videos_fech05_p90"] = float((s["n_fech_05s_p90"] > 0).mean())
        r["frac_videos_fech_longo_p90"] = float((s["n_fech_longos_p90"] > 0).mean())
        r["n_fech_longos_total_p90"] = int(s["n_fech_longos_p90"].sum())
        r["perclos_janela_mediana"] = float(wv["perclos"].median()) if len(wv) else np.nan
        r["perclos_janela_p90"] = float(wv["perclos"].quantile(0.9)) if len(wv) else np.nan
        r["frac_janelas_fech05"] = float((wv["dur_max_fechamento_s"] >= FECH_S).mean()) if len(wv) else np.nan
        r["n_janelas"] = len(wv)
        for k in ("frac_videos_fech05", "frac_videos_fech_longo", "n_fech_05s_total", "n_fech_longos_total",
                  "frac_videos_fech05_p90", "frac_videos_fech_longo_p90", "n_fech_longos_total_p90",
                  "perclos_janela_mediana", "perclos_janela_p90", "frac_janelas_fech05"):
            L.add("d", r["base"], chave, k, r[k], n_videos=len(s), n_janelas=len(wv))
        T.append(r)
    Dd = pd.DataFrame(T)
    ms = sub(V, "nitymed|Microsleep").copy()
    pos = ms.sort_values(["n_fech_05s", "perclos"], ascending=False).iloc[0]["video"]
    cand = ms[ms["dur_s"] >= 60].sort_values(["n_fech_05s", "perclos"], ascending=True)
    neg = cand.iloc[0]["video"] if len(cand) else None
    exemplos = [(pos, "Mais fechamentos ≥ 0,5 s")]
    if neg is not None and neg != pos:
        exemplos.append((neg, "Menos fechamentos e menor PERCLOS (≥ 60 s)"))
    return Dd, exemplos, ms


# ======================================================================================
# Figuras (sem rostos: só números e séries temporais). Paleta: 3 primeiros slots categóricos da paleta de
# referência (validados com o validate_palette.js do skill dataviz: CVD ΔE 9,2, visão normal 24,0; o aqua fica
# abaixo de 3:1 no branco -> rótulos de valor visíveis nas barras e tabelas no relatório).
# ======================================================================================
COR = dict(s1="#2a78d6", s2="#eb6834", s3="#1baf7a", tinta="#0b0b0b", tinta2="#52514e", muda="#898781",
           grade="#e1e0d9", eixo="#c3c2b7", fundo="#ffffff")


def estilo():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.edgecolor": COR["eixo"], "axes.linewidth": 0.8, "axes.labelcolor": COR["tinta2"],
        "xtick.color": COR["tinta2"], "ytick.color": COR["tinta2"], "text.color": COR["tinta"],
        "axes.grid": True, "grid.color": COR["grade"], "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": COR["fundo"],
        "axes.facecolor": COR["fundo"], "savefig.facecolor": COR["fundo"], "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


def virgula(nd):
    """Formatador de eixo com vírgula decimal (pt-BR)."""
    from matplotlib.ticker import FuncFormatter
    return FuncFormatter(lambda v, _pos: f"{v:.{nd}f}".replace(".", ","))


def _faixa(ax, V, chaves, coluna, escala=1.0, rng=None):
    """Um ponto por vídeo (série única, slot 1) e traço na mediana (tinta)."""
    for i, (chave, _) in enumerate(chaves):
        v = sub(V, chave)[coluna].to_numpy(float) * escala
        v = v[np.isfinite(v)]
        x = i + rng.uniform(-0.22, 0.22, len(v))
        ax.scatter(x, v, s=7, color=COR["s1"], alpha=0.5, linewidths=0, zorder=3)
        if len(v):
            med = float(np.median(v))
            ax.plot([i - 0.32, i + 0.32], [med, med], color=COR["tinta"], lw=1.5, zorder=4, solid_capstyle="round")
    ax.set_xticks(range(len(chaves)), [r for _, r in chaves])
    ax.set_xlim(-0.6, len(chaves) - 0.4)
    ax.grid(axis="x", visible=False)


def figura_principal(V, D, A, Bq, rot_dmd="Modelo do DMD (Exp. 5b)"):
    plt = estilo()
    fig = plt.figure(figsize=(7.16, 2.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 1.85], wspace=0.36)
    a1, a2, a3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    rng = np.random.default_rng(SEED)
    chaves = [("nitymed|todos", "NITYMED\nnoite"), ("dmd_mov|safe", "DMD\nmovim.\n(dia)"),
              ("dmd_par|safe", "DMD\nparado\n(dia)"), ("uta|todos", "UTA-\nRLDD")]
    # (A) fração de quadros SEM rosto por vídeo (escala log simétrica: 0 fica visível)
    for i, (chave, _) in enumerate(chaves):
        v = 100.0 * (1.0 - sub(V, chave)["taxa_rosto"].to_numpy(float))
        x = i + rng.uniform(-0.22, 0.22, len(v))
        a1.scatter(x, v, s=7, color=COR["s1"], alpha=0.5, linewidths=0, zorder=3)
        med = float(np.median(v))
        a1.plot([i - 0.32, i + 0.32], [med, med], color=COR["tinta"], lw=1.5, zorder=4, solid_capstyle="round")
    a1.set_yscale("symlog", linthresh=0.1, linscale=0.6)
    a1.set_ylim(-0.02, 100)
    a1.set_yticks([0, 0.1, 1, 10, 100], ["0", "0,1", "1", "10", "100"])
    a1.set_xticks(range(len(chaves)), [r for _, r in chaves], fontsize=6.1)
    a1.set_xlim(-0.6, len(chaves) - 0.4)
    a1.grid(axis="x", visible=False)
    a1.set_ylabel("Quadros sem rosto (% por vídeo)")
    a1.set_title("A. Captura do rosto", loc="left", color=COR["tinta"], fontweight="bold")
    # (B) SNR rápida por vídeo
    _faixa(a2, V, chaves, "snr_rapido", rng=rng)
    a2.set_xticklabels([r for _, r in chaves], fontsize=6.1)
    a2.set_ylabel("SNR do EAR (amplitude ÷ ruído rápido)")
    a2.set_ylim(bottom=0)
    a2.yaxis.set_major_formatter(virgula(0))
    a2.set_title("B. Qualidade do sinal", loc="left", color=COR["tinta"], fontweight="bold")
    # (C) taxa de disparo (variante principal = brutas)
    grupos = [("uta", "KSS 0", "UTA\nKSS 0"), ("dmd", "alerta_parado", "DMD\nparado"),
              ("dmd", "alerta_movimento", "DMD\nmovim."), ("uta", "KSS 10", "UTA\nKSS 10"),
              ("nitymed", "Yawning", "NITYMED\nbocejo"), ("nitymed", "Microsleep", "NITYMED\nmicross.")]
    series = [("dmd", VAR_PRINCIPAL, rot_dmd, COR["s1"]),
              ("uta", VAR_PRINCIPAL, "Modelo da UTA-RLDD (B3)", COR["s2"]),
              ("regra_dmd", "", "Regra: fech. ≥ 0,5 s ou bocejo", COR["s3"])]
    larg = 0.25
    for j, (fonte, var, rot, cor) in enumerate(series):
        for i, (ds, g, _) in enumerate(grupos):
            r = D[(D["base"] == ds) & (D["grupo"] == g) & (D["fonte"] == fonte) & (D["variante"].fillna("") == var)]
            if r.empty:
                continue
            r = r.iloc[0]
            x = i + (j - 1) * larg
            a3.bar(x, r["taxa"], width=larg, color=cor, edgecolor=COR["fundo"], linewidth=0.9, zorder=3,
                   label=rot if i == 0 else None)
            topo = r["taxa"]
            if np.isfinite(r["lo"]):
                a3.plot([x, x], [r["lo"], r["hi"]], color=COR["tinta2"], lw=0.7, zorder=4)
                topo = r["hi"]
            a3.annotate(fnum(100 * r["taxa"], 0), (x, topo + 0.015), ha="center", va="bottom", fontsize=5.8,
                        color=COR["tinta"])
    a3.set_xticks(range(len(grupos)), [g for _, _, g in grupos], fontsize=6.1)
    a3.set_xlim(-0.55, len(grupos) - 0.45)
    a3.set_ylim(0, 1.12)
    a3.set_yticks(np.linspace(0, 1, 6), [f"{int(round(v * 100))}%" for v in np.linspace(0, 1, 6)])
    a3.set_ylabel("Janelas de 10 s marcadas \"sonolento\"")
    a3.set_title("C. Taxa de disparo (p ≥ 0,5; IC 95%)", loc="left", color=COR["tinta"], fontweight="bold")
    a3.grid(axis="x", visible=False)
    a3.axvline(2.5, color=COR["eixo"], lw=0.8, zorder=1)
    a3.text(1.0, 1.085, "alerta", ha="center", va="center", fontsize=6, color=COR["tinta2"])
    a3.text(4.0, 1.085, "sonolento (UTA) e noite", ha="center", va="center", fontsize=6, color=COR["tinta2"])
    fig.legend(*a3.get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.995, 1.0), frameon=False,
               ncol=3, handlelength=0.9, columnspacing=0.9, handletextpad=0.4, fontsize=6.4)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.83, bottom=0.21)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_exp6_noite.{ext}", dpi=300)
    plt.close(fig)


def figura_microssono(V, exemplos):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt = estilo()
    fig = plt.figure(figsize=(7.16, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.6], hspace=0.62, wspace=0.24)
    a1 = fig.add_subplot(gs[:, 0])
    chaves = [("dmd_s5|todos", "DMD s5\natuado"), ("dmd_mov|safe", "DMD\nalerta\nmovim."), ("uta|KSS 0", "UTA\nKSS 0"),
              ("uta|KSS 10", "UTA\nKSS 10"), ("nitymed|Yawning", "NITY.\nbocejo"), ("nitymed|Microsleep", "NITY.\nmicross.")]
    rng = np.random.default_rng(SEED)
    _faixa(a1, V, chaves, "perclos", escala=100.0, rng=rng)
    a1.set_xticklabels([r for _, r in chaves], fontsize=6)
    a1.set_ylabel("PERCLOS do vídeo (% dos quadros com rosto)")
    a1.set_title("A. PERCLOS por vídeo\n(limiar do pipeline; traço = mediana)", loc="left", color=COR["tinta"],
                 fontweight="bold")
    for k, (vid, titulo) in enumerate(exemplos):
        ax = fig.add_subplot(gs[k, 1])
        s = B.carregar_sinal(ROOT / "data" / "signals" / "nitymed" / f"{vid}.npz")
        ev = pd.read_csv(FEAT / "nitymed" / "blinks" / f"{vid}.csv")
        linha = V[(V["dataset"] == "nitymed") & (V["video"] == vid) & (V["recorte"] == "inteiro")].iloc[0]
        e = np.where(s["face"], s["ear"], np.nan)
        for r in ev[ev["dur_s"] >= FECH_S].itertuples():
            ax.axvspan(s["t"][r.inicio], s["t"][r.fim] + s["d"][r.fim], color=COR["s2"], alpha=0.25, lw=0, zorder=1)
        ax.plot(s["t"], e, color=COR["s1"], lw=0.6, zorder=3)
        ax.axhline(linha["limiar_ear"], color=COR["tinta"], lw=0.9, zorder=4)
        ax.axhline(linha["limiar_p90"], color=COR["tinta2"], lw=0.9, ls=(0, (1, 1.5)), zorder=4)
        ax.set_xlim(0, B.duracao_total(s))
        ax.set_ylim(np.nanpercentile(e, 0.2) - 0.02, np.nanpercentile(e, 99.8) + 0.02)
        ax.yaxis.set_major_formatter(virgula(2))
        ax.set_ylabel("EAR")
        cod = vid.split("_")[2]
        ax.set_title(f"{'BC'[k]}. {titulo} ({cod})\nPERCLOS {fpct(linha['perclos'])} (p90: {fpct(linha['perclos_p90'])}); "
                     f"{int(linha['n_fech_05s'])} fech. ≥ 0,5 s; {int(linha['n_fech_longos'])} > 1,5 s",
                     loc="left", color=COR["tinta"], fontweight="bold", fontsize=7.4)
        if k == 0:
            ax.legend(handles=[Line2D([], [], color=COR["tinta"], lw=0.9, label="limiar do pipeline"),
                               Line2D([], [], color=COR["tinta2"], lw=0.9, ls=(0, (1, 1.5)), label="limiar p90"),
                               Patch(color=COR["s2"], alpha=0.25, lw=0, label="fechamento ≥ 0,5 s")],
                      loc="lower left", bbox_to_anchor=(0.0, -0.02), ncol=3, frameon=True, framealpha=0.9,
                      edgecolor="none", fontsize=6.2, handlelength=1.4, borderpad=0.3)
        if k == len(exemplos) - 1:
            ax.set_xlabel("tempo (s)")
    fig.subplots_adjust(left=0.075, right=0.99, top=0.9, bottom=0.11)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"fig_exp6_microssono.{ext}", dpi=300)
    plt.close(fig)


# ======================================================================================
# Relatório
# ======================================================================================
def lista_pt(itens):
    """['a', 'b', 'c'] -> 'a, b e c'."""
    itens = list(itens)
    return itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + " e " + itens[-1] if itens else ""


def _hora(ts):
    """'2019-04-08T10:08:19+02:00' -> '2019-04-08 10:08'."""
    return str(ts)[:16].replace("T", " ")


def tab(df, cols, cab):
    L = ["| " + " | ".join(cab) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(cab))) + "|"]
    for r in df.itertuples(index=False):
        L.append("| " + " | ".join(f(r) for f in cols) + " |")
    return "\n".join(L)


def relatorio(V, W, P, A, Adif, Bq, Bdif, C, D, far, IMP, M, Dd, exemplos, ms, E5, conf, F, lum_ok, meta_mod, CF, S):
    L = []
    ad = lambda s="": L.append(s)  # noqa: E731
    hoje = time.strftime("%d/%m/%Y")
    mm = meta_mod["modelos"]

    def row(df, k):
        r = df[df["chave"] == k]
        return r.iloc[0] if len(r) else None

    def dsel(ds, g, fonte, var=VAR_PRINCIPAL):
        r = D[(D["base"] == ds) & (D["grupo"] == g) & (D["fonte"] == fonte) & (D["variante"].fillna("") == var)]
        return r.iloc[0] if len(r) else None

    def bdif(metrica, a, b):
        r = Bdif[(Bdif["metrica"] == metrica) & (Bdif["a"] == a) & (Bdif["b"] == b)]
        return r.iloc[0] if len(r) else None

    def adif(a, b):
        r = Adif[(Adif["a"] == a) & (Adif["b"] == b)]
        return r.iloc[0] if len(r) else None

    def ic(r, nd=3, pct=False, k="dif"):
        return fic(r[k], r["lo"], r["hi"], nd, pct) if r is not None else "—"

    ny, ut, d5 = row(A, "nitymed|todos"), row(A, "uta|todos"), row(A, "dmd_s5|todos")
    dmv, dpa = row(A, "dmd_mov|safe"), row(A, "dmd_par|safe")
    bn, bu, b5 = row(Bq, "nitymed|todos"), row(Bq, "uta|todos"), row(Bq, "dmd_s5|todos")
    bmv, bpa = row(Bq, "dmd_mov|safe"), row(Bq, "dmd_par|safe")
    wn, wu, wd = W["nitymed"], W["uta"], W["dmd"]
    n_mov = int((wd["grupo_setup"] == "alerta_movimento").sum())
    n_par = int((wd["grupo_setup"] == "alerta_parado").sum())
    vd = V[(V["dataset"] == "dmd") & (V["recorte"] == "inteiro")]
    horas = pd.to_datetime(vd["data_gravacao"].str.slice(0, 19), errors="coerce")
    h_min, h_max = horas.min(), horas.max()
    h_ini = horas.dt.hour * 60 + horas.dt.minute
    anom = vd[((vd["sessao"] == "s1") & (vd["setup"] != MOVIMENTO)) | ((vd["sessao"] == "s2") & (vd["setup"] != PARADO))]
    # ruído rápido (trechos safe_drive) das sessões fora do protocolo × sessões conforme o protocolo (verificação)
    vs_ = V[(V["dataset"] == "dmd") & (V["recorte"] == "safe_drive")]
    conf_mov = vs_[(vs_["sessao"] == "s1") & (vs_["setup"] == MOVIMENTO)]["ruido_rapido"]
    conf_par = vs_[(vs_["sessao"] == "s2") & (vs_["setup"] == PARADO)]["ruido_rapido"]
    rr_anom = vs_.set_index("video")["ruido_rapido"]
    txt_anom = ("ruído rápido do EAR nos trechos de direção segura (`results/exp6_noite_videos.csv`): "
                + "; ".join(f"`{r.sujeito}` {r.sessao} ({r.setup}) {fnum(rr_anom.get(r.video, np.nan), 4)}"
                            for r in anom.itertuples())
                + f" — sessões conforme o protocolo: s1 em movimento mediana {fnum(conf_mov.median(), 4)} "
                  f"(faixa {fnum(conf_mov.min(), 4)}–{fnum(conf_mov.max(), 4)}), s2 parada mediana {fnum(conf_par.median(), 4)} "
                  f"(faixa {fnum(conf_par.min(), 4)}–{fnum(conf_par.max(), 4)}). As faixas se sobrepõem: o ruído dessas sessões "
                  "não confirma nem desmente o `setup` registrado")
    md_, mu_ = mm[f"dmd_{VAR_PRINCIPAL}"], mm[f"uta_{VAR_PRINCIPAL}"]
    do5b = "Exp. 5b final" in md_["origem"]

    ad("# Exp. 6 — Robustez noturna com o NITYMED (PP3) — rodada final")
    ad()
    ad(f"Gerado por `src/experiments/exp6_noite.py` (versão {VERSAO}) em {hoje}; nenhum número digitado à mão: tudo sai de "
       "`results/exp6_noite.csv` (longo), `results/exp6_noite_videos.csv` (por vídeo lógico), `results/exp6_noite_janelas.csv` "
       "(por janela, com as probabilidades) e dos demais CSV citados. Log: `results/logs/exp6_final_noite.log` (luminância: "
       "`results/logs/exp6_final_luminancia.log`). Seed 42. A rodada preliminar (janelas de 30 s, UTA-RLDD só com o Fold 1, "
       "DMD só com a s5) foi descartada; seus arquivos foram removidos em 06/10/2026.")
    ad()
    ad(f"**Dados**: janelas de {fnum(JANELA_S, 0)} s com passo de 5 s (`data/features/final/<base>/windows_10s.csv`). "
       f"NITYMED: {V[(V['dataset'] == 'nitymed') & (V['recorte'] == 'inteiro')].shape[0]} vídeos, {fint(len(wn))} janelas, "
       f"{fint(int(wn['aval'].sum()))} avaliáveis (fora do primeiro terço, que é a referência). UTA-RLDD completa: "
       f"{V[(V['dataset'] == 'uta') & (V['recorte'] == 'inteiro')].shape[0]} vídeos lógicos ({int(V.loc[(V['dataset'] == 'uta') & (V['recorte'] == 'inteiro'), 'n_partes'].sum())} arquivos) "
       f"de {wu['sujeito'].nunique()} sujeitos, {fint(len(wu))} janelas, {fint(int(wu['aval'].sum()))} avaliáveis. DMD: {len(vd)} sessões de "
       f"{vd['sujeito'].nunique()} sujeitos (s5 {int((vd['sessao'] == 's5').sum())}, s1 {int((vd['sessao'] == 's1').sum())}, "
       f"s2 {int((vd['sessao'] == 's2').sum())}, s3 {int((vd['sessao'] == 's3').sum())}), gravadas entre "
       f"{int(h_ini.min() // 60):02d}h{int(h_ini.min() % 60):02d} e {int(h_ini.max() // 60):02d}h{int(h_ini.max() % 60):02d} (hora local do "
       f"`recordTime`, de {h_min:%d/%m/%Y} a {h_max:%d/%m/%Y}) — por isso \"dia\"; janelas: {fint(int((wd['uso'] == 'sonolento').sum()))} sonolentas (s5), "
       f"{fint(int((wd['uso'] == 'alerta').sum()))} alerta da s2 e {fint(int((wd['uso'] == 'alerta_s1').sum()))} da s1 "
       f"({n_par} com o carro parado e {n_mov} em movimento pelo campo `setup`). "
       f"Luminância: {'medida (`results/exp6_luminancia_videos.csv`)' if lum_ok else 'NÃO medida nesta execução'}.")
    ad()
    ad("**O NITYMED não tem classe alerta nem anotação quadro a quadro: nenhuma acurácia, AUC ou F1 foi calculada nele.** "
       "As medidas são de captura (rosto), de sinal (EAR) e de comportamento dos detectores (fração de janelas marcadas). "
       "A comparação noite × dia NÃO isola a noite: as bases também diferem em câmera, distância e tamanho do rosto, "
       "carro em movimento × parado × casa, protocolo (encenado × autorrelato × autenticidade não documentada) e pessoas. "
       "A UTA-RLDD é autogravada (celular/webcam) e não documenta a hora nem a iluminação; o que se compara é a luminância medida. "
       "O DMD em movimento (sessões de distração, direção real) é a única referência diurna com o carro andando: ele separa "
       "parte do efeito do movimento do efeito da noite.")
    ad()
    # ---------------- 1. números-chave
    ad("## 1. Números-chave")
    ad()
    ad(f"- **(a) Detecção de rosto por quadro** (média por vídeo [IC 95%]): NITYMED {fic(ny['media'], ny['lo'], ny['hi'], 4)} "
       f"({int(ny['n_videos'])} vídeos; mín {fnum(ny['minimo'], 4)} em `{ny['id_minimo']}`; {int(ny['n_lt_095'])} vídeos < 0,95; "
       f"{fpct(ny['frac_jan_lt_090'])} das janelas < 0,90); UTA-RLDD {fic(ut['media'], ut['lo'], ut['hi'], 4)} "
       f"({int(ut['n_videos'])} vídeos; mín {fnum(ut['minimo'], 4)} em `{ut['id_minimo']}`; {fpct(ut['frac_jan_lt_090'])} das janelas < 0,90); "
       f"DMD s5 {fic(d5['media'], d5['lo'], d5['hi'], 4)}; DMD em movimento (direção segura) {fic(dmv['media'], dmv['lo'], dmv['hi'], 4)}. "
       "Diferenças NITYMED − " + "; − ".join(f"{NOME_CURTO[b]}: {ic(adif(a, b), 4)}" for a, b in CONTRASTES) + ".")
    pmd = {k: bdif(k, "dmd_mov|safe", "dmd_par|safe") for k in ("ruido_rapido", "snr_rapido")}
    pmd = {k: v for k, v in pmd.items() if v is not None}
    ad(f"- **(b) Qualidade do EAR** (medianas por vídeo): SNR rápida (amplitude das piscadas ÷ ruído quadro a quadro) NITYMED "
       f"{fnum(bn['snr_rapido'], 1)}, UTA-RLDD {fnum(bu['snr_rapido'], 1)}, DMD s5 {fnum(b5['snr_rapido'], 1)}, DMD parado "
       f"{fnum(bpa['snr_rapido'], 1)}, **DMD em movimento {fnum(bmv['snr_rapido'], 1)}**; ruído rápido {fnum(bn['ruido_rapido'], 4)} / "
       f"{fnum(bu['ruido_rapido'], 4)} / {fnum(b5['ruido_rapido'], 4)} / {fnum(bpa['ruido_rapido'], 4)} / {fnum(bmv['ruido_rapido'], 4)}. "
       + (f"No DMD, com os mesmos motoristas, a mesma câmera e o mesmo dia, o carro em movimento muda o ruído rápido em "
          f"{ic(pmd['ruido_rapido'], 4)} e a SNR rápida em "
          f"{ic(pmd['snr_rapido'], 1)} (mediana das diferenças por sujeito, "
          f"{int(pmd['ruido_rapido']['n'])} sujeitos)." if "ruido_rapido" in pmd and "snr_rapido" in pmd else ""))
    txt = []
    for fonte, nomef in (("dmd", "modelo do DMD" + (" (Exp. 5b final)" if do5b else "")), ("uta", "modelo da UTA-RLDD (B3)"),
                         ("regra_dmd", "regra do rótulo do DMD")):
        var = "" if fonte.startswith("regra") else VAR_PRINCIPAL
        rm, rb = dsel("nitymed", "Microsleep", fonte, var), dsel("nitymed", "Yawning", fonte, var)
        r0, rp, rv, r10 = (dsel("uta", "KSS 0", fonte, var), dsel("dmd", "alerta_parado", fonte, var),
                           dsel("dmd", "alerta_movimento", fonte, var), dsel("uta", "KSS 10", fonte, var))
        if rm is None or r0 is None:
            continue
        txt.append(f"{nomef}: microssono {fpct(rm['taxa'])}, bocejo {fpct(rb['taxa'])} × falso alarme no alerta de dia — UTA KSS 0 "
                   f"{fpct(r0['taxa'])}, DMD parado {fpct(rp['taxa'])}, DMD em movimento {fpct(rv['taxa'])} — e disparo na UTA KSS 10 {fpct(r10['taxa'])}")
    ad("- **(c) Taxa de disparo** (janelas de 10 s avaliáveis marcadas \"sonolento\"; características brutas; probabilidade ≥ 0,5): "
       + "; ".join(txt) + ".")
    dms, dby, du0 = row(Dd, "nitymed|Microsleep"), row(Dd, "nitymed|Yawning"), row(Dd, "uta|KSS 0")
    ad(f"- **(d) Microssono**: PERCLOS mediano por vídeo {fpct(dms['perclos'])} (Bocejo {fpct(dby['perclos'])}; UTA KSS 0 {fpct(du0['perclos'])}); "
       f"{fpct(dms['frac_videos_fech05'], 0)} dos {int(dms['n_videos'])} vídeos têm ao menos um fechamento ≥ 0,5 s e "
       f"{fpct(dms['frac_videos_fech_longo'], 0)} ao menos um > 1,5 s ({int(dms['n_fech_longos_total'])} no total). "
       f"Sem referência alerta (nível aberto = p90 do EAR do próprio vídeo): PERCLOS mediano {fpct(dms['perclos_p90'])} e "
       f"{fpct(dms['frac_videos_fech_longo_p90'], 0)} dos vídeos com fechamento > 1,5 s.")
    ad(f"- **(e) Ficha**: `data/nitymed/index.csv` e `results/exp6_nitymed_ficha.md` — {len(F)} vídeos em disco "
       f"({int((F['categoria'] == 'Microsleep').sum())} Microssono, {int((F['categoria'] == 'Yawning').sum())} Bocejo) × 130 declarados; "
       "licença da cópia usada (Kaggle) CC BY-ND 4.0; autenticidade dos comportamentos não documentada" +
       (f"; {int(F['suspeito_pasta'].sum())} vídeos da pasta Bocejo com > {SUSPEITO_DUR_S:.0f} s e nenhum bocejo detectado "
        "(suspeitos de pasta errada; sensibilidade na seção 5.6)." if F["suspeito_pasta"].any() else "."))
    ad()
    # ---------------- 2. decisões
    ad("## 2. Desenho e decisões")
    ad()
    ad("| Item | Decisão | Por quê |")
    ad("|---|---|---|")
    ad(f"| Janela | {fnum(JANELA_S, 0)} s, passo 5 s, nas três bases (grade por arquivo; nas sessões de distração do DMD, janelas ancoradas nos trechos safe_drive) | regra final (notas, item 9): os trechos safe_drive do DMD são curtos |")
    ad(f"| Modelo do DMD | {md_['origem']} | especificação: aplicar o modelo final de transferência do Exp. 5b" +
       ("" if do5b else "; o modelo final do 5b não estava disponível nesta execução, então foi treinado aqui com a mesma receita e os mesmos dados") + " |")
    ad(f"| Treino do DMD | {md_['n_janelas']} janelas de {md_['n_sujeitos']} sujeitos: {md_['n_sonolento']} sonolentas (s5 com fechamento anotado ≥ 0,5 s ou bocejo) × {md_['n_alerta']} alerta (safe_drive da s2); gE_28 (sem s5) fora | desenho do Exp. 5b final (notas, itens 7, 9 e 10) |")
    ad(f"| Modelo intra-UTA (B3) | treinado aqui (o Exp. 5b não salva modelos intra-UTA), mesma receita: {mu_['n_janelas']} janelas de {mu_['n_sujeitos']} sujeitos, KSS 10 = {mu_['n_sonolento']} × KSS 0 = {mu_['n_alerta']} (KSS 5, referência e fronteira fora); `models/exp6/rf_uta_<variante>.joblib` | especificação (B3) |")
    ad(f"| Receita | {RECEITA}; scikit-learn {meta_mod['sklearn']} | a do Exp. 5b |")
    ad("| Características | as 11 da janela (" + ", ".join(f"`{f}`" for f in FEATS11) + "); sensibilidade sem `taxa_rosto` (10) | `taxa_rosto` mede captura, não sonolência |")
    ad("| Variante principal no Exp. 6 | **brutas (`sem_normalizacao`)**; a principal do Exp. 5b (centradas `_c`) entra como sensibilidade | o NITYMED não tem trecho alerta: a referência de `_c` (e do limiar de EAR) nele é o primeiro terço do próprio vídeo sonolento, e centrar nela apaga parte do sinal |")
    ad("| Janelas avaliadas | UTA e NITYMED: `avaliavel` (sem referência nem fronteira); DMD: alerta = janelas de direção segura das sessões s1/s2, agrupadas pelo campo `setup` (carro parado × em movimento), e sonolento = s5 com rótulo | referência nunca avaliada; no NITYMED as janelas que tocam o primeiro terço dividem quadros com a calibração (sensibilidade com todas as janelas na seção 5.6) |")
    ad("| Fora da amostra | UTA pelo modelo da UTA: 5 folds oficiais; DMD pelo modelo do DMD: deixa-um-sujeito-de-fora (clone do modelo usado); o modelo da outra base e o NITYMED: modelo inteiro | nenhum número do próprio domínio vem de janelas vistas no treino |")
    ad(f"| Disparo | probabilidade da classe sonolenta ≥ {fnum(LIMIAR_DISPARO, 1)} (a regra `s >= limiar` do Exp. 5b); secundários: limiar calibrado só com o treino (a regra do 5b: máxima acurácia balanceada nos escores fora da amostra do treino) e limiar com {fpct(FAR_ALVO, 0)} de falso alarme nas janelas alerta da UTA | nenhum ajuste no NITYMED |")
    ad("| Regra sem treino | janela marcada se tem fechamento ≥ 0,5 s (EAR < limiar) ou bocejo (MAR > 0,5 por ≥ 1 s) — a regra que define o rótulo do DMD, aplicada aos NOSSOS sinais | separa o detector de sinais do classificador |")
    ad("| Vídeo lógico | as partes `_p1`/`_p2` (F3_32_10, F5_49_10) são concatenadas em ordem, com tempo contínuo, antes das medidas por vídeo; luminância = mediana das amostras das partes juntas | notas, item 8 |")
    ad("| DMD nas medidas de sinal | sessões de distração: só os quadros dos trechos `safe_drive` (olhando a estrada); a s5 inteira; a sessão inteira da distração só na tabela de captura | as distrações encenadas (celular, rádio, olhar para trás) derrubam rosto e EAR por outro motivo |")
    ad(f"| Carro em movimento × parado | pelo campo `setup` do OpenLABEL (não pela sessão): {', '.join(f'`{r.sujeito}` {r.sessao} ({_hora(r.data_gravacao)}) = {r.setup}' for r in anom.itertuples())} | {txt_anom}; o agrupamento por `setup` segue o registro oficial, e a comparação por sessão (s1 × s2) fica como sensibilidade (as duas levam à mesma conclusão na seção 4) |")
    ad(f"| Olho aberto (b) | quadro com rosto, EAR ≥ limiar e a mais de {fnum(MARGEM_EVENTO_S, 1)} s de qualquer piscada/fechamento (bordas estendidas) | exclui transições |")
    ad(f"| Ruído (b) | (i) DP do EAR nos quadros de olho aberto; (ii) ruído rápido = 1,4826 × MAD do resíduo em torno da mediana móvel de {fnum(JANELA_RUIDO_S, 0)} s | (i) inclui deriva lenta (pose); (ii) isola o tremor quadro a quadro |")
    ad("| Amplitude (b) | média da amplitude **estendida** das piscadas detectadas (`amplitude_ext`) | só existe para piscadas detectadas (viés de seleção) |")
    ad(f"| PERCLOS (d) | fração dos quadros com rosto com EAR < limiar do pipeline (0,75 × mediana da referência: último terço do vídeo 0 na UTA, safe_drive da s3 no DMD, primeiro terço do próprio vídeo no NITYMED); sensibilidade: 0,75 × percentil {P_ABERTO} do EAR do próprio vídeo | no NITYMED não há trecho alerta para calibrar |")
    ad("| IC 95% | bootstrap de agrupamentos, 1000 reamostragens, semente 42: vídeo (NITYMED), sujeito (UTA, DMD); diferenças pareadas: mediana das diferenças por sujeito, bootstrap de sujeitos | vídeos do mesmo motorista do NITYMED não são identificáveis: IC otimista |")
    ad()
    # ---------------- 3. (a)
    ad("## 3. (a) Detecção de rosto por quadro — noite × dia")
    ad()
    ad("Fonte: arrays `face` dos `.npz` de `data/signals/<base>/` (Face Mesh em modo vídeo, confianças 0,5, igual ao app), por vídeo lógico; "
       "janelas de `data/features/final/<base>/windows_10s.csv`.")
    ad()
    Ar = A.set_index("chave").loc[[k for k in LINHAS_A if k in set(A["chave"])]].reset_index()
    ad(tab(Ar, [lambda r: LINHAS[r.chave][0], lambda r: str(r.n_videos), lambda r: str(r.n_agrup), lambda r: fint(r.n_quadros),
                lambda r: fic(r.media, r.lo, r.hi, 4), lambda r: fnum(r.mediana, 4), lambda r: fnum(r.p10, 4),
                lambda r: fnum(r.minimo, 4), lambda r: str(r.n_lt_099), lambda r: str(r.n_lt_095),
                lambda r: fpct(r.frac_jan_lt_090), lambda r: fnum(r.maior_lacuna_mediana, 2), lambda r: fnum(r.maior_lacuna_max, 2)],
           ["Base / grupo", "Vídeos", "Agrup.", "Quadros", "Média por vídeo [IC 95%]", "Mediana", "p10", "Mínimo",
            "Vídeos < 0,99", "Vídeos < 0,95", "Janelas < 0,90", "Maior lacuna (s), mediana", "Maior lacuna (s), máx."]))
    ad()
    ad("Diferenças da média por vídeo (bootstrap independente por agrupamento): " +
       "; ".join(f"NITYMED − {NOME_CURTO[b]} = {ic(adif(a, b), 4)}" for a, b in CONTRASTES) + ". "
       "Pareadas por sujeito no DMD (direção segura): " +
       "; ".join(f"{d_} = {ic(adif(a, b), 4)} ({int(adif(a, b)['n'])} sujeitos)" for a, b, d_ in PAREADOS if adif(a, b) is not None) + ".")
    ad()
    vn = V[(V["dataset"] == "nitymed") & (V["recorte"] == "inteiro")]
    piores = vn.nsmallest(8, "taxa_rosto")
    ad("Os 8 vídeos noturnos com menor detecção: " + "; ".join(
        f"`{r.video}` {fnum(r.taxa_rosto, 3)} (maior lacuna {fnum(r.maior_lacuna_s, 2)} s)" for r in piores.itertuples()) + ".")
    vu = V[(V["dataset"] == "uta") & (V["recorte"] == "inteiro")]
    pu = vu.nsmallest(3, "taxa_rosto")
    ad("Os 3 vídeos da UTA-RLDD com menor detecção: " + "; ".join(
        f"`{r.video}` {fnum(r.taxa_rosto, 3)} ({' + '.join(x.replace('x', '×') for x in str(r.resolucoes).split(';'))} px; maior lacuna {fnum(r.maior_lacuna_s, 1)} s)"
        for r in pu.itertuples()) + ". A pior captura entre todas as bases é de um vídeo da UTA-RLDD com resolução muito baixa, não de um vídeo noturno.")
    ad()
    vi = V[V["recorte"] == "inteiro"]
    conf_rosto = (vi["taxa_rosto"] - vi["taxa_rosto_index_extracao"]).abs().max()
    ad(f"Conferência: diferença máxima entre a taxa recalculada dos `.npz` (por vídeo lógico) e a do `index.csv` da extração "
       f"(ponderada pelos quadros das partes) = {conf_rosto:.2e} ({len(vi)} vídeos lógicos).")
    ad()
    # ---------------- 4. (b)
    ad("## 4. (b) Qualidade do sinal de EAR")
    ad()
    ad("Medianas por vídeo (IC 95% por bootstrap de agrupamentos só nas linhas principais). Fonte: `results/exp6_noite_videos.csv`. "
       "No DMD em movimento/parado, só os quadros dos trechos de direção segura; a luminância é a da sessão inteira.")
    ad()
    Br = Bq.set_index("chave").loc[[k for k in LINHAS_B if k in set(Bq["chave"])]].reset_index()
    principais = {"nitymed|todos", "uta|todos", "dmd_s5|todos", "dmd_mov|safe", "dmd_par|safe"}
    ad(tab(Br, [lambda r: LINHAS[r.chave][0], lambda r: str(r.n_videos), lambda r: fnum(r.ear_aberto_media, 3),
                lambda r: fnum(r.ear_aberto_dp, 4), lambda r: fnum(r.ruido_rapido, 4), lambda r: fnum(r.amp_ext_media, 4),
                lambda r: fnum(r.amp_ext_relativa, 3),
                lambda r: (fic(r.snr, r.snr_lo, r.snr_hi, 2) if r.chave in principais else fnum(r.snr, 2)),
                lambda r: (fic(r.snr_rapido, r.snr_rapido_lo, r.snr_rapido_hi, 1) if r.chave in principais else fnum(r.snr_rapido, 1)),
                lambda r: fnum(r.taxa_piscadas_min, 1), lambda r: fnum(r.olho_largura_px, 1), lambda r: fnum(r.lum_rosto, 1),
                lambda r: fnum(r.contraste_rosto, 1), lambda r: fnum(r.bits_por_pixel, 3)],
           ["Base / grupo", "Vídeos", "EAR aberto (média)", "DP aberto", "Ruído rápido", "Amplitude", "Amplitude ÷ nível",
            "SNR", "SNR rápida", "Piscadas/min", "Olho (px)", "Lum. rosto", "Contraste", "Bits/pixel"]))
    ad()
    ad("Diferenças de mediana NITYMED − referência diurna (bootstrap independente por agrupamento):")
    ad()
    ad("| Medida | " + " | ".join(f"− {NOME_CURTO[b]} [IC 95%]" for _, b in CONTRASTES) + " |")
    ad("|---|" + "---:|" * len(CONTRASTES))
    for k, rot, nd in MET_B:
        rr = [bdif(k, a, b) for a, b in CONTRASTES]
        if all(x is None for x in rr):
            continue
        ad(f"| {rot} | " + " | ".join(ic(x, nd) for x in rr) + " |")
    ad()
    ad("**Efeito do carro em movimento, de dia, no mesmo motorista** (DMD, trechos de direção segura das sessões s1 e s2: mesma câmera, "
       "mesmo carro e, em geral, o mesmo dia, com minutos de diferença). Mediana das diferenças por sujeito (média das sessões de cada "
       "condição) [IC 95%, bootstrap de sujeitos]:")
    ad()
    ad("| Medida | " + " | ".join(f"{d_}" for _, _, d_ in PAREADOS) + " |")
    ad("|---|" + "---:|" * len(PAREADOS))
    rotp = dict((a, b) for a, b, _ in MET_B)
    rotp["taxa_rosto"] = "Taxa de detecção de rosto"
    ndp = dict((a, c) for a, _, c in MET_B)
    ndp["taxa_rosto"] = 4
    for k in MET_PAREADO:
        rr = [bdif(k, a, b) for a, b, _ in PAREADOS]
        if all(x is None for x in rr):
            continue
        ad(f"| {rotp.get(k, k)} | " + " | ".join((ic(x, ndp.get(k, 3)) + f" (n = {int(x['n'])}; {int(x['n_pos'])} ↑, {int(x['n_neg'])} ↓)")
                                                 if x is not None and x["n"] > 0 else "—" for x in rr) + " |")
    ad()
    if len(C):
        ad("Correlações de Spearman entre vídeos da mesma base (a iluminação explica a captura e o sinal?):")
        ad()
        ad("| Base | x | y | ρ | p | vídeos |")
        ad("|---|---|---|---:|---:|---:|")
        for r in C.itertuples():
            ad(f"| {NOME[r.base]} | `{r.x}` | `{r.y}` | {fnum(r.rho, 2)} | {r.p:.2g} | {r.n} |")
        ad()
        ad("Os p-valores tratam vídeos como independentes (no NITYMED há 21 motoristas em 126 vídeos): leia como descrição.")
        ad()
    # ---------------- 5. (c)
    ad("## 5. (c) Taxa de disparo dos detectores nos vídeos noturnos")
    ad()
    ad("Fração das janelas de 10 s (passo 5 s) marcadas \"sonolento\" [IC 95% por agrupamento], só janelas avaliáveis. "
       "ᴸ = fora da amostra no próprio domínio (UTA: 5 folds oficiais; DMD: deixa-um-sujeito-de-fora). Nos vídeos noturnos, uma "
       "janela de microssono não contém necessariamente um microssono (o vídeo também tem fala e olhares ao redor): a fração de "
       "janelas é uma cota inferior da sensibilidade, e a fração de vídeos com alguma janela marcada é o análogo por vídeo. "
       "**Nenhum número desta seção é acurácia no NITYMED.**")
    ad()
    for k_var, var in enumerate(VARIANTES, 1):
        ad(f"### 5.{k_var} {NOME_VAR[var][0].upper() + NOME_VAR[var][1:]}")
        ad()
        ad("| Grupo | Janelas (agrup.) | DMD: janelas | DMD: vídeos (média ≥ 0,5) | DMD: vídeos (alguma) | UTA B3: janelas | UTA B3: vídeos (média ≥ 0,5) | UTA B3: vídeos (alguma) |")
        ad("|---|---:|---:|---:|---:|---:|---:|---:|")
        for (ds, g) in GRUPOS_C_TAB:
            a, b_ = dsel(ds, g, "dmd", var), dsel(ds, g, "uta", var)
            if a is None or b_ is None:
                continue
            ad(f"| {ROTULO_GRUPO[(ds, g)]} | {fint(a['n_janelas'])} ({int(a['n_agrup'])}) | {fic(a['taxa'], a['lo'], a['hi'], 1, True)}"
               f"{' ᴸ' if str(a['amostra']).startswith('fora') else ''} | {fic(a['videos_media'], a['videos_media_lo'], a['videos_media_hi'], 0, True)} | "
               f"{fic(a['videos_alguma'], a['videos_alguma_lo'], a['videos_alguma_hi'], 0, True)} | "
               f"{fic(b_['taxa'], b_['lo'], b_['hi'], 1, True)}{' ᴸ' if str(b_['amostra']).startswith('fora') else ''} | "
               f"{fic(b_['videos_media'], b_['videos_media_lo'], b_['videos_media_hi'], 0, True)} | "
               f"{fic(b_['videos_alguma'], b_['videos_alguma_lo'], b_['videos_alguma_hi'], 0, True)} |")
        ad()
    ad("\"vídeos (média ≥ 0,5)\" = regra por vídeo da Entrega 01 (média dos escores das janelas); \"vídeos (alguma)\" = vídeo com pelo "
       "menos uma janela marcada. No DMD, \"vídeo\" = sessão.")
    ad()
    n5 = len(VARIANTES) + 1
    ad(f"### 5.{n5} Outros limiares (secundários)")
    ad()
    ad("Limiar calibrado só com o treino (a regra do Exp. 5b: máxima acurácia balanceada nos escores fora da amostra das janelas de "
       "treino — DMD: LOSO; UTA: fora do fold) e limiar com 10% de falso alarme nas janelas alerta da UTA (KSS 0), os dois aplicados "
       "sem mudança à noite.")
    ad()
    ad("| Modelo | Variante | Limiar calibrado (origem) | UTA KSS 0 | DMD parado | DMD em mov. | UTA KSS 10 | NITYMED bocejo | NITYMED microssono |")
    ad("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for fonte in ("dmd", "uta"):
        for var in VARIANTES:
            ent = mm[f"{fonte}_{var}"]
            vals = [dsel(ds, g, fonte, var) for ds, g in (("uta", "KSS 0"), ("dmd", "alerta_parado"), ("dmd", "alerta_movimento"),
                                                           ("uta", "KSS 10"), ("nitymed", "Yawning"), ("nitymed", "Microsleep"))]
            orig = ("Exp. 5b" if ent.get("limiar_calibrado_exp5b") is not None else "calculado aqui")
            extra = (f"; aqui {fnum(ent['limiar_calibrado_aqui'], 4)}" if ent.get("limiar_calibrado_exp5b") is not None else "")
            ad(f"| {FONTE_NOME[fonte]} | {var} | {fnum(ent['limiar_calibrado'], 4)} ({orig}{extra}) | " +
               " | ".join(fpct(v["taxa_cal"]) if v is not None else "—" for v in vals) + " |")
    ad()
    ad("| Modelo | Variante | Limiar (FAR 10% na UTA KSS 0) | FAR alcançado | UTA KSS 10 | NITYMED bocejo | NITYMED microssono | Obs. |")
    ad("|---|---|---:|---:|---:|---:|---:|---|")
    for (fonte, var), (tau, fa, emp) in far.items():
        r10, rb, rm = dsel("uta", "KSS 10", fonte, var), dsel("nitymed", "Yawning", fonte, var), dsel("nitymed", "Microsleep", fonte, var)
        obs = (f"FAR de {fpct(FAR_ALVO, 0)} inatingível: {fpct(emp)} das janelas alerta têm probabilidade ≥ {fnum(tau, 3)}"
               if fa < FAR_ALVO / 2 and emp > FAR_ALVO else "")
        ad(f"| {FONTE_NOME[fonte]} | {var} | {fnum(tau, 3)} | {fpct(fa)} | {fpct(r10['taxa_far10'])} | "
           f"{fpct(rb['taxa_far10'])} | {fpct(rm['taxa_far10'])} | {obs} |")
    ad()
    ad("No RF da UTA as probabilidades de dia são fora do fold e as da noite são do modelo inteiro: a transferência de qualquer limiar é aproximada.")
    ad()
    ad(f"### 5.{n5 + 1} Regra do rótulo do DMD aplicada aos nossos sinais (sem treino) e disparo sem evento")
    ad()
    ad("| Grupo | Fechamento ≥ 0,5 s | Bocejo (MAR > 0,5 por ≥ 1 s) | Qualquer um (regra DMD) |")
    ad("|---|---:|---:|---:|")
    for (ds, g) in GRUPOS_C_TAB:
        rr = [dsel(ds, g, n_, "") for n_ in ("regra_fechamento", "regra_bocejo", "regra_dmd")]
        if any(x is None for x in rr):
            continue
        ad(f"| {ROTULO_GRUPO[(ds, g)]} | " + " | ".join(fic(x['taxa'], x['lo'], x['hi'], 1, True) for x in rr) + " |")
    ad()
    ad(f"Disparo dos modelos (variante `{VAR_PRINCIPAL}`) só nas janelas **sem evento detectado** (nenhum fechamento ≥ 0,5 s e nenhum "
       "bocejo nos nossos sinais) — se o modelo marca essas janelas, reage a outra coisa que não fechamento/bocejo (ruído do EAR, piscadas):")
    ad()
    ad("| Grupo | Janelas sem evento (fração do grupo) | Modelo do DMD | Modelo da UTA B3 |")
    ad("|---|---:|---:|---:|")
    for (ds, g) in GRUPOS_C_TAB:
        a, b_ = dsel(ds, g, "dmd"), dsel(ds, g, "uta")
        if a is None or b_ is None or not np.isfinite(a.get("taxa_sem_evento", np.nan)):
            continue
        ad(f"| {ROTULO_GRUPO[(ds, g)]} | {fint(a['n_sem_evento'])} ({fpct(a['frac_sem_evento'])}) | "
           f"{fic(a['taxa_sem_evento'], a['lo_sem_evento'], a['hi_sem_evento'], 1, True)} | "
           f"{fic(b_['taxa_sem_evento'], b_['lo_sem_evento'], b_['hi_sem_evento'], 1, True)} |")
    ad()
    ad(f"### 5.{n5 + 2} Sensibilidades no NITYMED (variante `{VAR_PRINCIPAL}`)")
    ad()
    sus = sorted(F.loc[F["suspeito_pasta"], "id"])
    ad("(i) Todas as janelas (inclui as do primeiro terço, que é a referência, e as de fronteira); (ii) vídeos da pasta Bocejo suspeitos "
       f"de pasta errada (> {SUSPEITO_DUR_S:.0f} s, nenhum bocejo detectado no vídeo inteiro; ver `results/exp6_nitymed_ficha.md`): "
       + (", ".join(f"`{v}`" for v in sus) if sus else "nenhum") + ". Os números principais mantêm as janelas avaliáveis e os rótulos de pasta.")
    ad()
    ad("| Grupo | Conjunto | Vídeos | Janelas | Modelo do DMD | Modelo da UTA B3 | Regra DMD |")
    ad("|---|---|---:|---:|---:|---:|---:|")
    for (g, nome), d in S.groupby(["grupo", "conjunto"], sort=False):
        x = {r.detector: r for r in d.itertuples()}
        r0 = d.iloc[0]
        ad(f"| {'Bocejo' if g == 'Yawning' else 'Microssono'} | {nome} | {fint(r0['n_videos'])} | {fint(r0['n_janelas'])} | " +
           " | ".join(fic(x[k].taxa, x[k].lo, x[k].hi, 1, True) if k in x else "—" for k in ("dmd", "uta", "regra_dmd")) + " |")
    ad()
    ad(f"### 5.{n5 + 3} DMD por sessão do protocolo (sensibilidade do agrupamento por `setup`)")
    ad()
    ad("| Grupo | Janelas (sujeitos) | Modelo do DMD | Modelo da UTA B3 | Regra DMD |")
    ad("|---|---:|---:|---:|---:|")
    for g in ("alerta", "alerta_s1"):
        a, b_, rr = dsel("dmd", g, "dmd"), dsel("dmd", g, "uta"), dsel("dmd", g, "regra_dmd", "")
        if a is None:
            continue
        ad(f"| {ROTULO_GRUPO[('dmd', g)]} | {fint(a['n_janelas'])} ({int(a['n_agrup'])}) | {fic(a['taxa'], a['lo'], a['hi'], 1, True)} ᴸ | "
           f"{fic(b_['taxa'], b_['lo'], b_['hi'], 1, True)} | {fic(rr['taxa'], rr['lo'], rr['hi'], 1, True)} |")
    ad()
    ad(f"### 5.{n5 + 4} Mudança de distribuição: medianas das características por grupo (brutas, janelas avaliadas)")
    ad()
    ad("| Grupo | Janelas | " + " | ".join(f"`{f}`" for f in FEATS11) + " |")
    ad("|---|---:|" + "---:|" * len(FEATS11))
    for r in M.itertuples(index=False):
        rr = r._asdict()
        ad(f"| {ROTULO_GRUPO[(rr['base'], rr['grupo'])]} | {fint(rr['n_janelas'])} | " +
           " | ".join(fnum(rr[f], 3) for f in FEATS11) + " |")
    ad()
    ad(f"Importância por impureza (redução média de Gini) dos modelos inteiros, variante `{VAR_PRINCIPAL}` — o que cada modelo usa:")
    ad()
    ad("| Característica | Modelo do DMD | Modelo da UTA B3 |")
    ad("|---|---:|---:|")
    for f in FEATS11:
        a = IMP[(IMP["fonte"] == "dmd") & (IMP["variante"] == VAR_PRINCIPAL) & (IMP["caracteristica"] == f)]["importancia_impureza"]
        b_ = IMP[(IMP["fonte"] == "uta") & (IMP["variante"] == VAR_PRINCIPAL) & (IMP["caracteristica"] == f)]["importancia_impureza"]
        ad(f"| `{f}` | {fnum(float(a.iloc[0]), 3) if len(a) else '—'} | {fnum(float(b_.iloc[0]), 3) if len(b_) else '—'} |")
    ad()
    if len(CF):
        ad(f"### 5.{n5 + 5} Contrafactual: o que move o disparo noturno")
        ad()
        ad("Disparo nas janelas noturnas avaliáveis (modelos inteiros, variante principal do Exp. 6 = brutas, `sem_normalizacao`) quando UMA característica recebe o valor típico "
           "do dia; última linha: as três de qualidade do sinal de uma vez. Diagnóstico do modelo, não medida de desempenho. Duas "
           "referências de \"dia\": mediana das janelas alerta KSS 0 da UTA-RLDD e mediana das janelas alerta (s2) do DMD — a classe "
           "alerta de treino do modelo do DMD.")
        ad()
        for nome_ref, rot_ref in (("uta_kss0", "Referência: mediana das janelas alerta KSS 0 da UTA-RLDD"),
                                  ("dmd_alerta_s2", "Referência: mediana das janelas alerta da s2 do DMD")):
            cf = CF[CF["referencia"] == nome_ref]
            if cf.empty:
                continue
            orig = cf.groupby(["fonte", "grupo"])["taxa_original"].first()
            ad(f"**{rot_ref}**")
            ad()
            ad("| Característica trocada | Valor do dia | " + " | ".join(
                f"{FONTE_NOME[f]} — {'Microssono' if g == 'Microsleep' else 'Bocejo'} (original {fpct(orig[(f, g)])})"
                for f in ("dmd", "uta") for g in ("Microsleep", "Yawning")) + " |")
            ad("|---|---:|" + "---:|" * 4)
            for troca in cf["troca"].unique():
                sb = cf[cf["troca"] == troca].set_index(["fonte", "grupo"])
                v = sb["valor_dia"].iloc[0]
                ad(f"| `{troca}` | {fnum(v, 3)} | " + " | ".join(
                    f"{fpct(sb.loc[(f, g), 'taxa_contrafactual'])} ({'+' if sb.loc[(f, g), 'delta'] >= 0 else '−'}"
                    f"{fnum(abs(100 * sb.loc[(f, g), 'delta']), 1)} p.p.)" for f in ("dmd", "uta") for g in ("Microsleep", "Yawning")) + " |")
            ad()
    if len(conf):
        ad(f"### 5.{n5 + 6} Conferência com as predições do Exp. 5b final")
        ad()
        ad("Mesma receita e mesmos dados ⇒ mesmas probabilidades. Comparação janela a janela com `results/exp5b_predicoes.csv` (RF; "
           "o CSV do 5b grava o escore com 6 casas decimais, então diferenças da ordem de 1e-07 são arredondamento):")
        ad()
        ad("| Comparação | Variante | Janelas | Sem par | Diferença máxima | Diferença média |")
        ad("|---|---|---:|---:|---:|---:|")
        for r in conf.itertuples():
            ad(f"| {r.comparacao} | {r.variante} | {fint(r.n_janelas)} | {r.n_sem_par} | {r.dif_max:.2e} | {r.dif_media:.2e} |")
        ad()
    if E5:
        ad(f"### 5.{n5 + 7} Todos os modelos finais salvos pelo Exp. 5b (treinados no DMD) aplicados à noite e ao dia")
        ad()
        ad("Modelos inteiros em todas as colunas. Na coluna do DMD em movimento (sessão s1, que não entra no treino) os sujeitos estão no "
           "treino com outras sessões, então ela NÃO é fora da amostra; o número fora da amostra (deixa-um-sujeito-de-fora) está na seção 5.1.")
        ad()
        ad("| Arquivo | Modelo | Variante | UTA KSS 0 (falso alarme) | DMD alerta em movimento (sujeitos do treino) | UTA KSS 10 | NITYMED bocejo | NITYMED microssono |")
        ad("|---|---|---|---:|---:|---:|---:|---:|")
        E5d = pd.DataFrame(E5)
        for arq, g in E5d.groupby("arquivo", sort=True):
            val = {r.grupo: fic(r.taxa, r.lo, r.hi, 1, True) for r in g.itertuples()}
            ad(f"| `{arq}` | {g['modelo'].iloc[0]} | {g['variante'].iloc[0]} | {val.get('KSS 0', '—')} | {val.get('alerta_movimento', '—')} | "
               f"{val.get('KSS 10', '—')} | {val.get('Yawning', '—')} | {val.get('Microsleep', '—')} |")
        ad()
    # ---------------- 6. (d)
    ad("## 6. (d) PERCLOS e fechamentos longos nos vídeos de microssono")
    ad()
    ad("PERCLOS = fração dos quadros com rosto com EAR < limiar. \"Pipeline\" = limiar de `blinks.py` (0,75 × mediana da referência: "
       "primeiro terço do próprio vídeo no NITYMED, último terço do vídeo 0 na UTA, safe_drive da s3 no DMD). \"p90\" = 0,75 × percentil 90 "
       "do EAR do próprio vídeo (não depende de um trecho alerta). Medianas por vídeo [IC 95%]; DMD em movimento/parado: só os trechos de direção segura.")
    ad()
    Dr = Dd.set_index("chave").loc[[k for k in LINHAS_D if k in set(Dd["chave"])]].reset_index()
    ad(tab(Dr, [lambda r: LINHAS[r.chave][0].replace("**", ""), lambda r: str(r.n_videos),
                lambda r: fnum(r.minutos, 1), lambda r: fic(r.perclos, r.perclos_lo, r.perclos_hi, 1, True),
                lambda r: fpct(r.perclos_p90v), lambda r: fpct(r.perclos_janela_p90),
                lambda r: fnum(r.fech_05s_por_min, 2), lambda r: fpct(r.frac_videos_fech05, 0),
                lambda r: fpct(r.frac_videos_fech_longo, 0), lambda r: str(r.n_fech_longos_total),
                lambda r: fic(r.perclos_p90, r.perclos_p90_lo, r.perclos_p90_hi, 1, True),
                lambda r: fpct(r.frac_videos_fech_longo_p90, 0)],
           ["Grupo", "Vídeos", "Minutos", "PERCLOS (pipeline)", "p90 entre vídeos", "p90 entre janelas",
            "Fech. ≥ 0,5 s/min", "Vídeos c/ fech. ≥ 0,5 s", "Vídeos c/ fech. > 1,5 s", "Fech. > 1,5 s (total)",
            "PERCLOS (p90)", "Vídeos c/ fech. > 1,5 s (p90)"]))
    ad()
    ad("Os vídeos de microssono, um a um (ordenados pelo nº de fechamentos ≥ 0,5 s):")
    ad()
    ad(tab(ms.sort_values(["n_fech_05s", "perclos"], ascending=False),
           [lambda r: f"`{r.video}`", lambda r: fnum(r.dur_s, 1), lambda r: fnum(r.taxa_rosto, 3),
            lambda r: fnum(r.limiar_ear, 3), lambda r: fpct(r.perclos), lambda r: str(int(r.n_fech_05s)),
            lambda r: str(int(r.n_fech_1s)), lambda r: str(int(r.n_fech_longos)), lambda r: fnum(r.dur_max_fech_s, 2),
            lambda r: fnum(r.limiar_p90, 3), lambda r: fpct(r.perclos_p90), lambda r: str(int(r.n_fech_05s_p90)),
            lambda r: str(int(r.n_fech_longos_p90)), lambda r: fnum(r.taxa_piscadas_min, 1)],
           ["Vídeo", "Duração (s)", "Rosto", "Limiar", "PERCLOS", "Fech. ≥ 0,5 s", "≥ 1 s", "> 1,5 s", "Maior (s)",
            "Limiar p90", "PERCLOS p90", "Fech. ≥ 0,5 s (p90)", "> 1,5 s (p90)", "Piscadas/min"]))
    ad()
    ad("Exemplos (figura `results/figures/fig_exp6_microssono.pdf`, só séries temporais de EAR — sem imagem): "
       + "; ".join(f"{t} = `{v}`" for v, t in exemplos) +
       ". Regras de escolha: o vídeo de microssono com mais fechamentos ≥ 0,5 s; e, entre os de ≥ 60 s, o com menos fechamentos e menor PERCLOS.")
    ad()

    def ear_resumo(vid):
        z = B.carregar_sinal(ROOT / "data" / "signals" / "nitymed" / f"{vid}.npz")
        e = z["ear"][z["face"]]
        lv = V[(V["dataset"] == "nitymed") & (V["video"] == vid) & (V["recorte"] == "inteiro")].iloc[0]
        return float(np.percentile(e, 1)), float(lv["ear_aberto_mediana"]), float(lv["limiar_ear"]), float(lv["limiar_p90"])
    alinh = RES / "sinais_alinhamento_dmd.csv"
    txt_dmd = ""
    if alinh.exists():
        al = pd.read_csv(alinh)
        if "ear_med_close" in al.columns:
            txt_dmd = (f" No DMD (s5), a mediana do EAR dentro dos intervalos anotados como olho fechado vai de {fnum(al['ear_med_close'].min(), 3)} "
                       f"a {fnum(al['ear_med_close'].max(), 3)} por sessão (`results/sinais_alinhamento_dmd.csv`): o EAR do Face Mesh com olho fechado não vai a 0.")
    frases = []
    for vid, _ in exemplos:
        p1, ab, lim, l90 = ear_resumo(vid)
        frases.append(f"`{vid}`: EAR p1 = {fnum(p1, 3)}, mediana com olho aberto = {fnum(ab, 3)}, limiar do pipeline = {fnum(lim, 3)}, "
                      f"limiar p90 = {fnum(l90, 3)}")
    ad("Números dos exemplos — " + "; ".join(frases) + "." + txt_dmd)
    ad()
    ids_prelim = {"Microsleep_Female_P1043106_720", "Microsleep_Male_P1043122_720"}
    if {v for v, _ in exemplos} == ids_prelim:
        ad("Inspeção visual INTERNA feita na rodada preliminar nestes mesmos dois vídeos (recortes dos olhos vistos pelo analista em "
           "27/09/2026, não publicados e não gravados no repositório; o limiar de EAR do NITYMED não mudou entre as rodadas): nos quadros de "
           "menor EAR de `Microsleep_Female_P1043106_720` os olhos estão fechados; no exemplo de menor PERCLOS, parte dos quadros sorteados "
           "ao acaso (EAR perto da mediana) já mostra pálpebras semicerradas — o trecho de referência (primeiro terço do vídeo) não é alerta, "
           "e o limiar relativo não enxerga o estreitamento sustentado.")
        ad()
    ad("**Calibração sem referência alerta.** No NITYMED o limiar de EAR (e a centragem `_c`) vem do primeiro terço do PRÓPRIO vídeo, "
       "que já pode estar sonolento: o limiar encolhe junto com o olho e os fechamentos sustentados deixam de cruzá-lo. A sensibilidade "
       "\"p90\" (nível de olho aberto tirado dos momentos mais abertos do próprio vídeo) mostra o tamanho do efeito "
       f"(PERCLOS mediano dos microssonos {fpct(dms['perclos'])} → {fpct(dms['perclos_p90'])}; nos vídeos alerta da UTA, a mesma troca vai de "
       f"{fpct(du0['perclos'])} a {fpct(du0['perclos_p90'])}, então o salto não é artefato do p90). Para o app, isso reforça a calibração em "
       "estado alerta no início do uso (a UTA-RLDD tem vídeo alerta; o NITYMED não).")
    ad()
    # ---------------- 7. (e)
    ad("## 7. (e) Ficha do dataset")
    ad()
    ad(f"Ver `results/exp6_nitymed_ficha.md` e `data/nitymed/index.csv` ({len(F)} linhas; SHA-256, codec, fps, resolução, "
       "duração, taxa de bits, taxa de rosto, bocejos detectados e janelas de 10 s por vídeo). Chaves no `paper/refs.bib`: " +
       ", ".join(f"`{k}` ({'presente' if v else 'AUSENTE'})" for k, v in chaves_bib().items()) +
       ". Citar o conjunto de dados e ao menos um dos artigos exigidos pelos autores (notas, item 1).")
    ad()
    # ---------------- 8. leitura
    ad("## 8. Leitura")
    ad()
    du_, d5_, dm_, dp_ = adif("nitymed|todos", "uta|todos"), adif("nitymed|todos", "dmd_s5|todos"), \
        adif("nitymed|todos", "dmd_mov|safe"), adif("nitymed|todos", "dmd_par|safe")
    menor = all(x is not None and x["hi"] < 0 for x in (du_, d5_, dm_))
    peq = all(x is not None and abs(x["dif"]) < 0.02 for x in (du_, d5_, dm_))
    c1 = C[(C["base"] == "nitymed") & (C["x"] == "lum_rosto") & (C["y"] == "taxa_rosto")] if len(C) else C
    ad(f"1. **Captura**: a taxa de detecção de rosto à noite é {'menor' if menor else 'não claramente menor'} que nas referências "
       f"diurnas, e a diferença é {'pequena' if peq else 'grande'} (NITYMED − UTA-RLDD {ic(du_, 4)}; − DMD s5 {ic(d5_, 4)}; − DMD em movimento "
       f"{ic(dm_, 4)}). O que muda à noite é a cauda: {int(ny['n_lt_095'])} dos {int(ny['n_videos'])} vídeos ficam abaixo de 0,95 e "
       f"{fpct(ny['frac_jan_lt_090'])} das janelas abaixo de 0,90 (UTA-RLDD: {int(ut['n_lt_095'])} vídeo(s), {fpct(ut['frac_jan_lt_090'])}; "
       f"DMD s5: {int(d5['n_lt_095'])}, {fpct(d5['frac_jan_lt_090'])}; DMD em movimento: {int(dmv['n_lt_095'])}, {fpct(dmv['frac_jan_lt_090'])}). "
       f"O pior vídeo de todas as bases, porém, não é noturno: é da UTA-RLDD (`{ut['id_minimo']}`, {fnum(ut['minimo'], 3)}; "
       f"{' + '.join(x.replace('x', '×') for x in str(vu.set_index('video').loc[ut['id_minimo'], 'resolucoes']).split(';'))} px) — "
       "enquadramento e resolução pesam mais que a noite nos casos extremos."
       + (f" Dentro do NITYMED, a correlação entre a luminância do rosto e a detecção é fraca (ρ = {fnum(c1.iloc[0]['rho'], 2)}, "
          f"{int(c1.iloc[0]['n'])} vídeos)." if len(c1) else ""))
    rn = lambda k, a, b: bdif(k, a, b)  # noqa: E731
    p_ruido, p_snr = rn("ruido_rapido", "dmd_mov|safe", "dmd_par|safe"), rn("snr_rapido", "dmd_mov|safe", "dmd_par|safe")
    s_ruido, s_snr = rn("ruido_rapido", "dmd_s1|safe", "dmd_s2|safe"), rn("snr_rapido", "dmd_s1|safe", "dmd_s2|safe")
    n_m = rn("snr_rapido", "nitymed|todos", "dmd_mov|safe")
    n_r = rn("ruido_rapido", "nitymed|todos", "dmd_mov|safe")
    mov_piora = p_ruido is not None and p_ruido["lo"] > 0 and p_snr is not None and p_snr["hi"] < 0
    noite_como_mov = n_m is not None and n_m["lo"] <= 0 <= n_m["hi"]
    lum_txt = ""
    if lum_ok and np.isfinite(bn["lum_rosto"]) and np.isfinite(bmv["lum_rosto"]):
        lum_txt = (f" A noite ainda tem menos luz no rosto (luminância mediana {fnum(bn['lum_rosto'], 1)} × {fnum(bmv['lum_rosto'], 1)} no DMD em "
                   f"movimento e {fnum(bu['lum_rosto'], 1)} na UTA-RLDD) e muito mais compressão ({fnum(bn['bits_por_pixel'], 3)} × "
                   f"{fnum(bmv['bits_por_pixel'], 3)} e {fnum(bu['bits_por_pixel'], 3)} bits por pixel); esses fatores não são separáveis aqui.")
    c2 = C[(C["base"] == "nitymed") & (C["x"] == "lum_olhos") & (C["y"] == "ruido_rapido")] if len(C) else C
    ad(f"2. **Sinal**: a SNR rápida do EAR cai de {fnum(bu['snr_rapido'], 1)} (UTA-RLDD), {fnum(b5['snr_rapido'], 1)} (DMD s5) e "
       f"{fnum(bpa['snr_rapido'], 1)} (DMD parado) para {fnum(bn['snr_rapido'], 1)} à noite: o ruído quadro a quadro noturno é "
       f"{fnum(bn['ruido_rapido'] / bu['ruido_rapido'], 1)}× o da UTA-RLDD e {fnum(bn['ruido_rapido'] / bpa['ruido_rapido'], 1)}× o do DMD parado, "
       f"e a amplitude das piscadas é menor ({fnum(bn['amp_ext_media'], 3)} × {fnum(bu['amp_ext_media'], 3)} e {fnum(bpa['amp_ext_media'], 3)}). "
       + (f"**Mas o DMD com o carro em movimento, de dia, já tem ruído rápido {fnum(bmv['ruido_rapido'], 4)} e SNR rápida "
          f"{fnum(bmv['snr_rapido'], 1)}**: no mesmo motorista, com a mesma câmera, o movimento aumenta o ruído rápido em {ic(p_ruido, 4)} e muda a "
          f"SNR rápida em {ic(p_snr, 1)} ({int(p_ruido['n'])} sujeitos, ruído maior em {int(p_ruido['n_pos'])} deles; pela sessão do "
          f"protocolo, s1 − s2: {ic(s_ruido, 4)} e {ic(s_snr, 1)}, "
          f"{int(s_ruido['n'])} sujeitos). " if mov_piora else
          f"No DMD, o efeito do carro em movimento sobre o ruído rápido foi {ic(p_ruido, 4)} e sobre a SNR rápida {ic(p_snr, 1)}. ")
       + (f"A diferença de SNR rápida entre a noite e o DMD em movimento não se distingue de zero ({ic(n_m, 1)}; ruído rápido "
          f"{ic(n_r, 4)}): a perda de qualidade do EAR à noite é do mesmo tamanho da que o carro em movimento produz de dia (vibração, "
          "movimentos de cabeça da direção real e luz que varia) e, com estes dados, não pode ser atribuída à escuridão por si." if noite_como_mov and mov_piora else
          f"Noite − DMD em movimento: SNR rápida {ic(n_m, 1)}; ruído rápido {ic(n_r, 4)}.")
       + lum_txt
       + (f" Dentro do NITYMED, ρ(luminância dos olhos, ruído rápido) = {fnum(c2.iloc[0]['rho'], 2)}." if len(c2) else ""))
    g = lambda ds, gg, f, v=VAR_PRINCIPAL: dsel(ds, gg, f, v)  # noqa: E731
    rd = {k: g(*k[:2], "dmd") for k in (("uta", "KSS 0"), ("dmd", "alerta_parado"), ("dmd", "alerta_movimento"), ("uta", "KSS 10"),
                                         ("nitymed", "Microsleep"), ("nitymed", "Yawning"), ("dmd", "sonolento"))}
    ru = {k: g(*k[:2], "uta") for k in rd}
    rr = {k: dsel(*k[:2], "regra_dmd", "") for k in rd}
    fa_dia_d = [rd[("uta", "KSS 0")]["taxa"], rd[("dmd", "alerta_parado")]["taxa"], rd[("dmd", "alerta_movimento")]["taxa"]]
    ms_d = rd[("nitymed", "Microsleep")]
    ms_no_nivel = any(ms_d["lo"] <= fa <= ms_d["hi"] for fa in fa_dia_d)
    cfu = CF[(CF["referencia"] == "uta_kss0") & (CF["fonte"] == "uta") & (CF["grupo"] == "Microsleep")].set_index("troca") if len(CF) else None
    txt_cf = ""
    if cfu is not None and "ear_dp" in cfu.index:
        g_s = "+".join(GRUPO_SINAL)
        txt_cf = (f"; contrafactual: com o `ear_dp` típico do alerta da UTA, o disparo desse modelo nos microssonos iria de "
                  f"{fpct(cfu.loc['ear_dp', 'taxa_original'])} para {fpct(cfu.loc['ear_dp', 'taxa_contrafactual'])}"
                  + (f", e trocando {', '.join(f'`{x}`' for x in GRUPO_SINAL)}, para {fpct(cfu.loc[g_s, 'taxa_contrafactual'])}" if g_s in cfu.index else ""))
    imp_u = IMP[(IMP["fonte"] == "uta") & (IMP["variante"] == VAR_PRINCIPAL)].sort_values("importancia_impureza", ascending=False)
    top = ", ".join(f"`{x}`" for x in imp_u["caracteristica"].head(3))
    mn = M.set_index(["base", "grupo"])
    ad(f"3. **Detectores** (nenhum número é acurácia: o NITYMED não tem negativos). O modelo do DMD"
       f"{' (Exp. 5b final)' if do5b else ''} marca {fpct(rd[('uta', 'KSS 0')]['taxa'])} das janelas alerta da UTA-RLDD, "
       f"{fpct(rd[('dmd', 'alerta_parado')]['taxa'])} das janelas alerta do DMD com o carro parado (fora da amostra) e "
       f"{fpct(rd[('dmd', 'alerta_movimento')]['taxa'])} com o carro em movimento; nas sonolentas, {fpct(rd[('uta', 'KSS 10')]['taxa'])} (UTA KSS 10) e "
       f"{fpct(rd[('dmd', 'sonolento')]['taxa'])} (DMD s5, fora da amostra). À noite marca {fpct(ms_d['taxa'])} das janelas de microssono e "
       f"{fpct(rd[('nitymed', 'Yawning')]['taxa'])} das de bocejo"
       + (" — nos microssonos, no nível do falso alarme diurno (o IC 95% do microssono contém o falso alarme "
          + lista_pt(n_ for n_, fa in zip(("do alerta da UTA", "do DMD parado", "do DMD em movimento"), fa_dia_d) if ms_d["lo"] <= fa <= ms_d["hi"])
          + "), então esse disparo não informa sensibilidade"
          + ("; nos bocejos, acima dele." if rd[("nitymed", "Yawning")]["lo"] > max(fa_dia_d) else ".") if ms_no_nivel else ".")
       + f" O modelo da UTA-RLDD (B3) tem {fpct(ru[('uta', 'KSS 0')]['taxa'])} de falso alarme no alerta da UTA (fora do fold), "
       f"{fpct(ru[('dmd', 'alerta_parado')]['taxa'])} no DMD parado e {fpct(ru[('dmd', 'alerta_movimento')]['taxa'])} no DMD em movimento, e "
       f"{fpct(ru[('uta', 'KSS 10')]['taxa'])} de disparo na UTA KSS 10; à noite marca {fpct(ru[('nitymed', 'Microsleep')]['taxa'])} (microssono) e "
       f"{fpct(ru[('nitymed', 'Yawning')]['taxa'])} (bocejo), inclusive {fpct(ru[('nitymed', 'Microsleep')]['taxa_sem_evento'])} das janelas de "
       f"microssono sem fechamento nem bocejo detectado (alerta da UTA sem evento: {fpct(ru[('uta', 'KSS 0')]['taxa_sem_evento'])}; DMD em "
       f"movimento sem evento: {fpct(ru[('dmd', 'alerta_movimento')]['taxa_sem_evento'])}). As características mais usadas por esse modelo são {top}, "
       f"e o `ear_dp` mediano das janelas de microssono é {fnum(mn.loc[('nitymed', 'Microsleep'), 'ear_dp'], 3)} contra "
       f"{fnum(mn.loc[('uta', 'KSS 0'), 'ear_dp'], 3)} no alerta da UTA e {fnum(mn.loc[('dmd', 'alerta_movimento'), 'ear_dp'], 3)} no DMD em movimento"
       f" — o ruído do EAR (noturno ou do carro andando) é lido como \"sonolência\"{txt_cf}. A regra sem treino tem falso alarme de "
       f"{fpct(rr[('uta', 'KSS 0')]['taxa'])} no alerta da UTA, {fpct(rr[('dmd', 'alerta_parado')]['taxa'])} no DMD parado e "
       f"{fpct(rr[('dmd', 'alerta_movimento')]['taxa'])} no DMD em movimento; à noite marca {fpct(rr[('nitymed', 'Microsleep')]['taxa'])} das "
       f"janelas de microssono e {fpct(rr[('nitymed', 'Yawning')]['taxa'])} das de bocejo (UTA KSS 10: {fpct(rr[('uta', 'KSS 10')]['taxa'])}; "
       f"DMD s5: {fpct(rr[('dmd', 'sonolento')]['taxa'])})"
       + (" — nos microssonos, menos do que no alerta diurno com o carro em movimento." if rr[("nitymed", "Microsleep")]["taxa"] < rr[("dmd", "alerta_movimento")]["taxa"] else "."))
    ad(f"4. **Microssono**: com o limiar do pipeline, o PERCLOS mediano dos vídeos de microssono é {fpct(dms['perclos'])} "
       f"({'menor' if dms['perclos'] < dby['perclos'] else 'maior'} que o dos vídeos de bocejo, {fpct(dby['perclos'])}) e só "
       f"{fpct(dms['frac_videos_fech_longo'], 0)} deles têm fechamento > 1,5 s. Sem referência alerta (p90 do próprio vídeo), o PERCLOS "
       f"mediano vai a {fpct(dms['perclos_p90'])} e {fpct(dms['frac_videos_fech_longo_p90'], 0)} dos vídeos passam a ter fechamento > 1,5 s: "
       "o limiar relativo calibrado num trecho que já não é alerta esconde os fechamentos (seção 6).")
    ad(f"5. **Resposta à PP3 (noite)**: a perda noturna do pipeline aparece na qualidade do sinal de EAR, não na detecção de rosto; "
       + ("e uma perda do mesmo tamanho aparece de dia no DMD com o carro em movimento, no mesmo motorista — com os dados disponíveis, a "
          "perda de sinal não se concentra na noite: acompanha a condução real (carro andando). " if (noite_como_mov and mov_piora) else
          "a parte dela que vem do carro em movimento foi medida no DMD (item 2). ")
       + "Os modelos de janela disparam muito à noite, mas também disparam no alerta de dia com o carro em movimento: sem classe alerta "
       "noturna, esse disparo não mede sensibilidade à sonolência. A calibração do limiar exige um trecho alerta do próprio motorista.")
    ad()
    # ---------------- 9. limitações
    ad("## 9. Limitações e pendências")
    ad()
    ad("1. **Noite confundida**: câmera, enquadramento, resolução, compressão, protocolo e pessoas mudam junto com a iluminação; o DMD em "
       "movimento separa o fator \"carro andando\" só dentro do DMD (mesmos motoristas e câmera), e a comparação dele com o NITYMED ainda "
       f"troca câmera, compressão, taxa de quadros (NITYMED {fnum(V.loc[(V['dataset'] == 'nitymed') & (V['recorte'] == 'inteiro'), 'fps'].median(), 0)} "
       f"quadros/s × DMD {fnum(V.loc[(V['dataset'] == 'dmd') & (V['recorte'] == 'inteiro'), 'fps'].median(), 1)}; o ruído rápido é medido quadro a quadro) e pessoas.")
    ad("2. **Motoristas não identificados no NITYMED** (21 declarados, 126 vídeos): IC e p-valores por vídeo são otimistas.")
    ad("3. **Autenticidade não documentada** do microssono/bocejo do NITYMED (provavelmente sob instrução): taxa de disparo ≠ sensibilidade "
       "a sonolência real; o NITYMED não entra na PP1.")
    ad("4. **Modelo do DMD**: " + (f"o final do Exp. 5b ({md_['origem']})." if do5b else
       "o modelo final do Exp. 5b não existia quando esta execução rodou; foi treinado aqui com a mesma receita e os mesmos dados "
       "(`models/exp6/rf_dmd_<variante>.joblib`). Rode de novo `python src/experiments/exp6_noite.py` quando `models/exp5/exp5b_final_modelos.json` "
       "existir: o script passa a usar o modelo do 5b sozinho (confere o md5 das janelas) e confere as predições."))
    ad(f"5. **Janelas avaliáveis no NITYMED**: só as que não tocam o primeiro terço (referência). Nos vídeos de bocejo (15–25 s) isso deixa "
       f"1–2 janelas por vídeo; {int((F['n_janelas_10s_avaliaveis'] == 0).sum())} vídeos não têm nenhuma (sensibilidade com todas as janelas na seção 5.6).")
    ad("6. **`setup` do DMD fora do protocolo** em " + ", ".join(f"{r.sujeito} {r.sessao} de {_hora(r.data_gravacao)} ({r.setup})" for r in anom.itertuples()) +
       ": o agrupamento principal usa o `setup` (registro oficial); " + txt_anom + ". A comparação por sessão do protocolo fica como sensibilidade.")
    ad("7. **Luminância**: 24 quadros com rosto por arquivo; no DMD, da sessão inteira (não só dos trechos de direção segura). A UTA-RLDD não "
       "documenta hora nem iluminação: \"dia\" vale só para o DMD (gravado entre 09h e 17h, pelo `recordTime`).")
    ad("8. **Sem classe alerta noturna**: nenhum limiar pode ser calibrado para a noite com estes dados; a PP3 noturna fica respondida no "
       "nível de captura e sinal, não de acurácia.")
    ad()
    # ---------------- 10. arquivos
    ad("## 10. Arquivos")
    ad()
    ad("- Código: `src/experiments/exp6_noite.py` (.venv), `src/experiments/exp6_luminancia.py` (.venv-mp), `src/experiments/run_exp6.sh`.")
    ad("- Resultados: `results/exp6_noite.csv` (longo), `results/exp6_noite_videos.csv` (por vídeo lógico e recorte), `results/exp6_noite_janelas.csv` "
       "(por janela, com as probabilidades), `results/exp6_noite_importancias.csv`, `results/exp6_noite_contrafactual.csv`, "
       "`results/exp6_luminancia_videos.csv`, `results/exp6_nitymed_ficha.md`, `data/nitymed/index.csv`.")
    ad("- Figuras: `results/figures/fig_exp6_noite.{pdf,png}` (A: quadros sem rosto por vídeo; B: SNR do EAR por vídeo; C: taxa de disparo), "
       "`results/figures/fig_exp6_microssono.{pdf,png}` (PERCLOS por vídeo e dois exemplos de EAR). Nenhuma contém rosto.")
    ad("- Modelos: " + "; ".join(f"{k}: {v['origem']}" for k, v in mm.items()) + "; metadados em `models/exp6/modelos.json`.")
    ad("- Logs: `results/logs/exp6_final_noite.log`, `results/logs/exp6_final_luminancia.log`, `results/logs/exp6_final_noite_disparo_resumo.csv`.")
    return "\n".join(L) + "\n"


# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description="Exp. 6 — robustez noturna com o NITYMED (PP3), rodada final.")
    ap.add_argument("--force", action="store_true", help="recalcula as medidas por vídeo (ignora o cache)")
    ap.add_argument("--treinar-dmd", action="store_true",
                    help="não usa os modelos do Exp. 5b final: treina o modelo do DMD aqui (mesma receita)")
    ap.add_argument("--sem-relatorio", action="store_true", help="só CSVs (sem .md e sem figuras)")
    args = ap.parse_args()
    LOGS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_ARQ, mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    t0 = time.time()
    log.info("Exp. 6 final — versão %s; janelas de %s s; seed %d; folds da UTA com .ok: %s", VERSAO, fnum(JANELA_S, 0),
             SEED, B.uta_folds_ok())
    W = carregar_janelas()
    for ds, w in W.items():
        log.info("Janelas %s: %d (avaliadas na seção c: %d; por grupo: %s)", ds, len(w), int(w["aval"].sum()),
                 w[w["aval"]].groupby("grupo").size().to_dict())
    V = tabela_videos(force=args.force)
    lum_ok = (RES / "exp6_luminancia_videos.csv").exists() and V["lum_rosto"].notna().any()
    L = Linhas()
    A, Adif = resumo_a(V, W, L)
    Bq, Bdif = resumo_b(V, L)
    C = correlacoes(V, L)
    P, IMP, meta_mod, inteiros, cand = treinar_e_prever(W, forcar_treino_dmd=args.treinar_dmd)
    D, far = taxas_disparo(P, L, meta_mod)
    CF = contrafactual(P, inteiros, L)
    conf = conferir_exp5b(P, cand)
    for r in conf.itertuples():
        log.info("Conferência com o Exp. 5b — %s (%s): %d janelas, %d sem par, diferença máx. %.2e", r.comparacao,
                 r.variante, r.n_janelas, r.n_sem_par, r.dif_max)
        L.add("c", "", "", "conferencia_exp5b_dif_max", r.dif_max, n_janelas=r.n_janelas, modelo=r.comparacao,
              variante=r.variante, nota=f"{r.n_sem_par} janelas sem par")
    E5 = outros_modelos_exp5(P, cand)
    for r in E5:
        L.add("c", r["base"], r["grupo"], "taxa_disparo_janela", r["taxa"], r["lo"], r["hi"], n_janelas=r["n_janelas"],
              n_agrup=r["n_agrup"], modelo=f"Exp. 5b: {r['arquivo']}", variante=r["variante"],
              nota="modelo final salvo pelo Exp. 5b (treinado no DMD), aplicado sem mudança")
    M = medianas_caracteristicas(P)
    Dd, exemplos, ms = resumo_d(V, W, L)
    F, zinfo = ficha_nitymed(V, W, lum_ok)
    S = sensibilidade_nitymed(P, F, L)
    # ---- CSVs
    L.df().to_csv(RES / "exp6_noite.csv", index=False)
    V.drop(columns=["_chave"], errors="ignore").to_csv(RES / "exp6_noite_videos.csv", index=False)
    CF.to_csv(RES / "exp6_noite_contrafactual.csv", index=False)
    IMP.to_csv(RES / "exp6_noite_importancias.csv", index=False)
    cols_j = ["dataset", "sujeito", "video", "arquivo", "sessao", "setup", "uso", "grupo", "grupo_setup", "cluster", "fold",
              "janela", "t_ini", "t_fim", "parcial", "janela_ref", "sobrepoe_ref", "avaliavel", "aval", "y", "B3"] + FEATS11
    partes = []
    for ds in BASES:
        pj = P[ds][[c for c in cols_j if c in P[ds].columns] + [c for c in P[ds].columns if c.startswith("p_")]].copy()
        fch, boc = _flags_regra(pj)
        pj["regra_fechamento"], pj["regra_bocejo"] = fch, boc
        pj["regra_dmd"] = fch | boc
        partes.append(pj)
    pd.concat(partes, ignore_index=True).to_csv(RES / "exp6_noite_janelas.csv", index=False)
    D.to_csv(LOGS / "exp6_final_noite_disparo_resumo.csv", index=False)
    if not args.sem_relatorio:
        figura_principal(V, D, A, Bq, rot_dmd=("Modelo do DMD (Exp. 5b final)" if "Exp. 5b final" in
                                                meta_mod["modelos"][f"dmd_{VAR_PRINCIPAL}"]["origem"]
                                                else "Modelo do DMD (receita do Exp. 5b)"))
        figura_microssono(V, exemplos)
        (RES / "exp6_nitymed_ficha.md").write_text(escrever_ficha(F, zinfo), encoding="utf-8")
        (RES / "exp6_noite.md").write_text(relatorio(V, W, P, A, Adif, Bq, Bdif, C, D, far, IMP, M, Dd, exemplos, ms,
                                                     E5, conf, F, lum_ok, meta_mod, CF, S), encoding="utf-8")
    log.info("Saídas: results/exp6_noite.{csv,md}, exp6_noite_videos.csv, exp6_noite_janelas.csv, "
             "exp6_noite_importancias.csv, exp6_noite_contrafactual.csv, exp6_nitymed_ficha.md, data/nitymed/index.csv, "
             "figures/fig_exp6_noite.{pdf,png}, figures/fig_exp6_microssono.{pdf,png}, models/exp6/")
    log.info("Tempo total: %.1f s", time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
