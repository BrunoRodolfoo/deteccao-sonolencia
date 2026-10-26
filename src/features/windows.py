"""Características por JANELA (rodada final: 10 s com passo de 5 s; 5 s e 15 s para sensibilidade).

Rodar no .venv, DEPOIS de src/features/blinks.py e src/features/dmd_labels.py:

    python src/features/windows.py                          # janela 10 s, passo 5 s, três bases
    python src/features/windows.py --janela 5  --passo 5
    python src/features/windows.py --janela 15 --passo 5 --dataset dmd

Entradas: data/signals/<base>/<id>.npz; data/features/final/<base>/blinks/<id>.csv e blinks_index.csv (limiar de
          EAR por unidade de referência); no DMD, o OpenLABEL (rótulos da s5 e trechos safe_drive da distração).
Saídas:   data/features/final/<base>/windows_<L>s.csv            todas as janelas: identificação, uso, rótulos,
                                                                 características brutas, _c e _z
          data/features/final/<base>/windows_referencia_<L>s.csv  referência de cada unidade (nº de janelas,
                                                                 tipo, média e desvio por característica)
          results/exp5_janelas_resumo.csv                         contagens por comprimento/base/arquivo e uso
          results/exp5_dmd_balanco_janelas.csv                    DMD: janelas por uso, sujeito, sessão e comprimento
          results/exp5_dmd_grade_safe.csv                         DMD: janelas safe_drive que caberiam numa grade global

JANELAS (tempo = carimbo do contêiner):
  * UTA-RLDD, NITYMED e DMD s5: grade global [t_ini, t_ini + L), t_ini = 0, passo, 2*passo, ... enquanto a janela
    couber inteira no ARQUIVO (as partes _p1/_p2 dos vídeos divididos da UTA são arquivos separados: nenhuma janela
    atravessa o corte; a coluna `video` traz o vídeo lógico). Arquivo mais curto que L gera UMA janela parcial
    [0, duração) se durar >= 10 s (NITYMED).
  * DMD s1/s2/s3 (distração): só janelas INTEIRAMENTE dentro de UM trecho "driver_actions/safe_drive" (trechos
    adjacentes fundidos), ANCORADAS no início de cada trecho (t_ini = início do trecho + k*passo). Os trechos são
    curtos (10-26 s): numa grade global uma janela de 10 s só cabe no trecho quando a grade coincide com ele, o que
    descartaria cerca de metade das janelas alerta sem motivo (contagens das duas grades no relatório).
USO de cada janela (coluna `uso`; `janela_ref` e `sobrepoe_ref` NUNCA entram em treino nem teste):
  * UTA: referência = janelas inteiramente no ÚLTIMO TERÇO (no tempo) do vídeo 0 (como o Preprocessing.py original);
    `exemplo` = demais janelas; as do vídeo 0 que cruzam o corte de 2/3 são `fronteira` (dividem quadros com a
    referência). Alerta avaliado = janelas do vídeo 0 inteiramente nos 2/3 iniciais.
  * DMD: referência = janelas safe_drive das sessões s3 do sujeito (sessão separada); `alerta` = janelas safe_drive
    da s2; se a s3 do sujeito tiver < 3 janelas, referência = último terço (no tempo, por sessão s2) das janelas
    safe_drive da s2, `alerta` = as janelas anteriores que não a tocam, `fronteira` = as que a tocam, e as da s3
    ficam `s3_nao_usada`. s5: `sonolento` se tiver fechamento anotado >= 0,5 s ou bocejo anotado (>= 0,5 s dentro
    da janela), senão `s5_sem_sinal` (fora do principal). s1: `alerta_s1` (sensibilidade).
  * NITYMED: referência = janelas inteiramente no PRIMEIRO terço do vídeo (desvio: não há trecho alerta);
    as que tocam o primeiro terço sem estar nele são `fronteira`; sem nenhuma janela de referência (vídeo < 3 L),
    a média de referência vem do primeiro terço tratado como uma janela só (pseudo-referência).
RÓTULOS: UTA — rotulo_kss, classe (0/1/2), B1, B2, B3 (do vídeo); DMD — y (principal: alerta 0 x sonolento 1),
  y_sens_s1 (alerta_s1 0 x sonolento 1), y_sens_s5_sem_sinal (principal + s5_sem_sinal como 0). Sujeito sem s5
  (gE_28): y = NaN (fora do treino pareado), `uso` preservado. `avaliavel` = entra em treino/teste no principal.
CARACTERÍSTICAS (só quadros COM ROSTO da janela; limiar de EAR = o da unidade, de blinks.py; unidades
independentes do fps):
  perclos              fração dos quadros com EAR < limiar
  taxa_piscadas_min    piscadas (tipo "piscada", fim dentro da janela) por minuto de rosto (ruidosa em 10 s;
                       n_piscadas também sai, como coluna auxiliar)
  dur_media_piscada_s  duração média (s, abaixo do limiar) dessas piscadas (NaN sem piscada)
  dur_max_fechamento_s maior intervalo contínuo com EAR < limiar DENTRO da janela (recortado nas bordas;
                       quadros sem rosto interrompem), em s; 0 se não houver
  n_fechamentos_05s    nº desses intervalos com duração >= 0,5 s
  ear_media, ear_dp    média e desvio-padrão do EAR
  amp_media            amplitude média das piscadas (variante ESTENDIDA de blinks.py: bordas no olho aberto)
  vel_media            velocidade média de abertura das piscadas (variante estendida, EAR/s)
  frac_bocejo          fração dos quadros em intervalos com MAR > 0,5 sustentado >= 1 s (recortado na janela)
  taxa_rosto           fração dos quadros da janela com rosto detectado
NORMALIZAÇÃO por unidade de referência (UTA e DMD = sujeito; NITYMED = vídeo), só com as janelas de referência:
  <f>_c = x - média das janelas de referência                    (principal: definida em todas as bases)
  <f>_z = (x - média) / desvio-padrão das janelas de referência   (só com >= 5 janelas de referência e desvio > 0;
          senão NaN — o desvio é 0 em várias características na referência alerta; ver o relatório)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blinks as B  # noqa: E402
import dmd_labels as L  # noqa: E402

ROOT = B.ROOT
RES = ROOT / "results"
VERSAO = "2.0"   # 2.0: rodada final (janela parametrizável; referência separada da avaliação; rótulos v2 do DMD)
JANELA_S = 10.0
PASSO_S = 5.0
MIN_PARCIAL_S = 10.0
MAR_LIMIAR = 0.5
MAR_SUST_S = 1.0
FECH_S = 0.5
N_MIN_REF_Z = 5
N_MIN_REF_DMD = 3        # s3 com menos janelas de referência -> referência = último terço da s2
EPS = 1e-9

FEATURES = ["perclos", "taxa_piscadas_min", "dur_media_piscada_s", "dur_max_fechamento_s", "n_fechamentos_05s",
            "ear_media", "ear_dp", "amp_media", "vel_media", "frac_bocejo", "taxa_rosto"]
ID_COLS = ["dataset", "sujeito", "video", "arquivo", "parte", "sessao", "data_gravacao", "setup", "unidade_ref",
           "janela", "t_ini", "t_fim", "parcial", "trecho_safe", "uso", "janela_ref", "sobrepoe_ref", "avaliavel",
           "fold", "oculos", "categoria", "genero", "rotulo_video"]
USOS_DMD = ("referencia", "fronteira", "alerta", "sonolento", "s5_sem_sinal", "alerta_s1", "s3_nao_usada")


def rotulo_L(janela):
    """'10s', '5s', '15s' (ou '7p5s' para 7,5 s) — usado nos nomes de arquivo."""
    return (f"{int(janela)}s" if float(janela).is_integer() else f"{janela}s".replace(".", "p"))


# --------------------------------------------------------------------------------------
# Geração das janelas
# --------------------------------------------------------------------------------------
def gerar_janelas(T_end, janela=JANELA_S, passo=PASSO_S, min_parcial=MIN_PARCIAL_S):
    """Grade global: lista de (t_ini, t_fim, parcial). Janelas inteiras; se o arquivo for mais curto que a janela,
    uma janela parcial [0, T_end) quando T_end >= min_parcial."""
    if T_end + EPS >= janela:
        n = int(np.floor((T_end - janela) / passo + EPS)) + 1
        return [(i * passo, i * passo + janela, False) for i in range(n)]
    if T_end >= min_parcial:
        return [(0.0, float(T_end), True)]
    return []


def janelas_em_trechos(trechos_t, janela=JANELA_S, passo=PASSO_S):
    """Janelas ancoradas no início de cada trecho (a, b) em segundos: [a + k*passo, a + k*passo + janela) enquanto
    couberem inteiras no trecho. Devolve lista de (t_ini, t_fim, índice do trecho)."""
    out = []
    for j, (a, b) in enumerate(trechos_t):
        k = 0
        while a + k * passo + janela <= b + EPS:
            out.append((a + k * passo, a + k * passo + janela, j))
            k += 1
    return out


def trechos_em_segundos(sinal, trechos_q):
    """Trechos em quadros (ini, fim inclusivos) -> (t de início do 1º quadro, t de fim do último quadro)."""
    t, d, n = sinal["t"], sinal["d"], sinal["n"]
    out = []
    for s, e in trechos_q:
        s, e = max(int(s), 0), min(int(e), n - 1)
        if e >= s:
            out.append((float(t[s]), float(t[e] + d[e])))
    return out


# --------------------------------------------------------------------------------------
# Características (inalteradas desde a rodada preliminar)
# --------------------------------------------------------------------------------------
def caracteristicas_intervalo(sinal, ev, limiar, i0, i1, mar_limiar=MAR_LIMIAR):
    """Características dos quadros [i0, i1) (ver docstring do módulo)."""
    face = sinal["face"][i0:i1]
    ear = sinal["ear"][i0:i1]
    d = sinal["d"][i0:i1]
    mar = sinal["mar"][i0:i1]
    nq = i1 - i0
    nr = int(face.sum())
    out = dict(n_quadros=nq, n_rosto=nr, taxa_rosto=nr / nq if nq else np.nan)
    if nr == 0:
        out.update({k: np.nan for k in FEATURES if k != "taxa_rosto"})
        out["n_piscadas"] = 0
        return out
    fech = face & (ear < limiar)
    out["perclos"] = float(fech.sum() / nr)
    e = ear[face]
    out["ear_media"] = float(e.mean())
    out["ear_dp"] = float(e.std())
    t_face = float(d[face].sum())
    if len(ev):
        pis = ev[(ev["tipo"] == "piscada") & (ev["fim"] >= i0) & (ev["fim"] < i1)]
    else:
        pis = ev
    out["n_piscadas"] = int(len(pis))
    out["taxa_piscadas_min"] = len(pis) / (t_face / 60.0) if t_face > 0 else np.nan
    out["dur_media_piscada_s"] = float(pis["dur_s"].mean()) if len(pis) else np.nan
    out["amp_media"] = float(pis["amplitude_ext"].mean()) if len(pis) else np.nan
    out["vel_media"] = float(pis["vel_ext_s"].mean()) if len(pis) else np.nan
    s_arr, e_arr = B.runs(fech)
    durs = [float(d[s:e + 1].sum()) for s, e in zip(s_arr, e_arr)]
    out["dur_max_fechamento_s"] = max(durs) if durs else 0.0
    out["n_fechamentos_05s"] = int(sum(x >= FECH_S for x in durs))
    s_arr, e_arr = B.runs(face & (mar > mar_limiar))
    q_boc = sum(e - s + 1 for s, e in zip(s_arr, e_arr) if float(d[s:e + 1].sum()) >= MAR_SUST_S)
    out["frac_bocejo"] = float(q_boc / nr)
    return out


def janelas_arquivo(sinal, ev, limiar, lista, mar_limiar=MAR_LIMIAR):
    """DataFrame de janelas de um arquivo. lista: (t_ini, t_fim, parcial, trecho) por janela."""
    t = sinal["t"]
    rows = []
    for j, (a, b, parcial, trecho) in enumerate(lista):
        i0 = int(np.searchsorted(t, a - EPS, "left"))
        i1 = int(np.searchsorted(t, b - EPS, "left")) if not parcial else sinal["n"]
        r = dict(janela=j, t_ini=a, t_fim=b, parcial=parcial, trecho_safe=trecho, i0=i0, i1=i1)
        r.update(caracteristicas_intervalo(sinal, ev, limiar, i0, i1, mar_limiar))
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Normalização (inalterada; agora com as janelas de referência separadas da avaliação)
# --------------------------------------------------------------------------------------
def normalizar(df, ref_mask, grupos="unidade_ref", pseudo_ref=None, n_min=N_MIN_REF_Z):
    """Acrescenta <f>_c e <f>_z por grupo usando as linhas de referência (ref_mask).

    pseudo_ref: dict grupo -> dict de características do trecho de referência (usado quando o grupo
    não tem nenhuma janela de referência). Devolve (df, tabela de estatísticas de referência)."""
    df = df.copy()
    stats = []
    novas = {}
    for g, sub in df.groupby(grupos, sort=False):
        ref = sub[ref_mask.loc[sub.index]]
        linha = {grupos: g, "n_janelas_ref": len(ref), "pseudo_ref": bool(len(ref) == 0 and pseudo_ref and g in pseudo_ref)}
        for f in FEATURES:
            if len(ref):
                mu = float(np.nanmean(ref[f])) if ref[f].notna().any() else np.nan
                sd = float(np.nanstd(ref[f])) if ref[f].notna().sum() >= n_min else np.nan
            else:
                mu = float(pseudo_ref.get(g, {}).get(f, np.nan)) if pseudo_ref else np.nan
                sd = np.nan
            if not (np.isfinite(sd) and sd > 1e-12) or len(ref) < n_min:
                sd = np.nan
            novas.setdefault(f + "_c", pd.Series(np.nan, index=df.index))
            novas.setdefault(f + "_z", pd.Series(np.nan, index=df.index))
            novas[f + "_c"].loc[sub.index] = sub[f] - mu
            novas[f + "_z"].loc[sub.index] = (sub[f] - mu) / sd if np.isfinite(sd) else np.nan
            linha[f + "_mu"] = mu
            linha[f + "_sd"] = sd
        stats.append(linha)
    for k, v in novas.items():
        df[k] = v
    return df, pd.DataFrame(stats)


# --------------------------------------------------------------------------------------
# Referência e uso por base
# --------------------------------------------------------------------------------------
def marcar_uta(w, T0_por_sujeito):
    """UTA: referência = último terço do vídeo 0; fronteira = janelas do vídeo 0 que cruzam o corte de 2/3."""
    w["janela_ref"] = False
    w["sobrepoe_ref"] = False
    v0 = w["rotulo_video"].astype(str) == "0"
    corte = w["sujeito"].map(lambda s: 2.0 * T0_por_sujeito.get(s, np.nan) / 3.0)
    ref = v0 & (w["t_ini"] >= corte - EPS) & (~w["parcial"].astype(bool))
    fr = v0 & ~ref & (w["t_fim"] > corte + EPS)
    w.loc[ref, "janela_ref"] = True
    w.loc[fr, "sobrepoe_ref"] = True
    w["uso"] = np.where(w["janela_ref"], "referencia", np.where(w["sobrepoe_ref"], "fronteira", "exemplo"))
    return w


def marcar_nitymed(w, T_por_video):
    """NITYMED: referência = janelas inteiramente no primeiro terço; fronteira = as que o tocam sem estar nele."""
    corte = w["arquivo"].map(lambda v: T_por_video[v] / 3.0)
    ref = (w["t_fim"] <= corte + EPS) & (~w["parcial"].astype(bool))
    fr = ~ref & (w["t_ini"] < corte - EPS)
    w["janela_ref"] = ref
    w["sobrepoe_ref"] = fr
    w["uso"] = np.where(ref, "referencia", np.where(fr, "fronteira", "exemplo"))
    return w


def marcar_dmd(w, n_min_ref=N_MIN_REF_DMD):
    """DMD: uso de cada janela conforme a regra final (docstring do módulo). Devolve (w, info por sujeito)."""
    w["janela_ref"] = False
    w["sobrepoe_ref"] = False
    uso = pd.Series("", index=w.index, dtype=object)
    s5 = w["sessao"] == "s5"
    uso[s5] = np.where(w.loc[s5, "rotulo_anotado"] == "sonolento_atuado", "sonolento", "s5_sem_sinal")
    uso[w["sessao"] == "s1"] = "alerta_s1"
    info = {}
    for suj, g in w.groupby("sujeito"):
        s3 = g[g["sessao"] == "s3"]
        s2 = g[g["sessao"] == "s2"]
        if len(s3) >= n_min_ref:
            uso[s3.index] = "referencia"
            uso[s2.index] = "alerta"
            info[suj] = dict(tipo_ref="s3_safe_drive", ref_fallback=False, n_janelas_s3=len(s3))
        else:
            uso[s3.index] = "s3_nao_usada"
            n_ref_tot = 0
            for arq, gs in s2.groupby("arquivo"):
                gs = gs.sort_values("t_ini")
                n_ref = math.ceil(len(gs) / 3.0)
                if n_ref == 0:
                    continue
                ref_idx = gs.index[-n_ref:]
                corte = float(gs.loc[ref_idx, "t_ini"].min())
                uso[ref_idx] = "referencia"
                antes = gs.index[:-n_ref]
                ok = gs.loc[antes, "t_fim"] <= corte + EPS
                uso[antes[ok.to_numpy()]] = "alerta"
                uso[antes[~ok.to_numpy()]] = "fronteira"
                n_ref_tot += n_ref
            info[suj] = dict(tipo_ref="s2_safe_drive_ultimo_terco", ref_fallback=True, n_janelas_s3=len(s3))
    w["uso"] = uso
    w["janela_ref"] = uso == "referencia"
    w["sobrepoe_ref"] = uso == "fronteira"
    tem_s5 = w.groupby("sujeito")["sessao"].transform(lambda x: (x == "s5").any())
    w["tem_s5"] = tem_s5.astype(bool)
    y = pd.Series(np.nan, index=w.index)
    y[uso == "alerta"] = 0.0
    y[uso == "sonolento"] = 1.0
    y[~w["tem_s5"]] = np.nan
    w["y"] = y
    ys1 = pd.Series(np.nan, index=w.index)
    ys1[uso == "alerta_s1"] = 0.0
    ys1[uso == "sonolento"] = 1.0
    ys1[~w["tem_s5"]] = np.nan
    w["y_sens_s1"] = ys1
    ys5 = y.copy()
    ys5[(uso == "s5_sem_sinal") & w["tem_s5"]] = 0.0
    w["y_sens_s5_sem_sinal"] = ys5
    return w, info


# --------------------------------------------------------------------------------------
# Processamento
# --------------------------------------------------------------------------------------
def processar_dataset(ds, janela=JANELA_S, passo=PASSO_S, mar_limiar=MAR_LIMIAR, log=None):
    log = log or logging.getLogger(__name__)
    idx = B.listar_videos(ds)
    if idx.empty:
        log.info("%s: sem sinais", ds)
        return None, None
    bidx_path = B.FEAT / ds / "blinks_index.csv"
    if not bidx_path.exists():
        raise SystemExit(f"Rode antes: python src/features/blinks.py --dataset {ds}")
    bidx = pd.read_csv(bidx_path, dtype={"rotulo": str, "parte": str}).set_index("id")
    faltam = sorted(set(idx["id"]) - set(bidx.index))
    if faltam:
        raise SystemExit(f"{ds}: {len(faltam)} arquivos sem piscadas; rode blinks.py de novo (ex.: {faltam[:3]})")
    partes, pseudo, T_arq = [], {}, {}
    grade_global_safe = 0
    t0 = time.time()
    for _, r in idx.iterrows():
        vid = r["id"]
        b = bidx.loc[vid]
        limiar = float(b["limiar"])
        sinal = B.carregar_sinal(ROOT / r["npz"])
        ev = pd.read_csv(B.FEAT / ds / "blinks" / f"{vid}.csv")
        T = B.duracao_total(sinal)
        T_arq[vid] = T
        ol = None
        if ds == "dmd" and r["sessao"] != "s5":
            ol = L.ler_openlabel(ROOT / r["anotacao"])
            tr = trechos_em_segundos(sinal, L.trechos_safe(ol["acoes"]))
            lista = [(a, b_, False, j) for a, b_, j in janelas_em_trechos(tr, janela, passo)]
            # diagnóstico: quantas caberiam numa grade global (mesmo L e passo)
            grade_global_safe += sum(1 for a, b_, p in gerar_janelas(T, janela, passo)
                                     if not p and any(ta - EPS <= a and b_ <= tb + EPS for ta, tb in tr))
        else:
            lista = [(a, b_, p, np.nan) for a, b_, p in gerar_janelas(T, janela, passo)]
        w = janelas_arquivo(sinal, ev, limiar, lista, mar_limiar)
        if not len(w):
            continue
        if ds == "dmd" and r["sessao"] == "s5":
            ol = L.ler_openlabel(ROOT / r["anotacao"])
            w = pd.concat([w, L.rotular_janelas(w, sinal["t"], ol["acoes"], sinal["fps"])], axis=1)
        w["dataset"] = ds
        w["sujeito"] = r["subject"]
        w["video"] = r["video"]
        w["arquivo"] = vid
        w["parte"] = r["parte"]
        w["sessao"] = r["sessao"]
        w["data_gravacao"] = r["data_gravacao"]
        w["setup"] = r["setup"]
        w["unidade_ref"] = b["unidade_ref"]
        w["fold"] = r.get("fold")
        w["oculos"] = r.get("oculos")
        w["categoria"] = r.get("categoria")
        w["genero"] = r.get("genero")
        w["rotulo_video"] = str(r["label"]) if pd.notna(r["label"]) else ""
        w["limiar_ear"] = limiar
        w["ear_ref_mediana"] = float(b["ear_ref_mediana"])
        if ds == "nitymed":
            i1 = int(np.searchsorted(sinal["t"], T / 3.0 - EPS, "left"))
            pseudo[b["unidade_ref"]] = caracteristicas_intervalo(sinal, ev, limiar, 0, i1, mar_limiar)
        partes.append(w)
    df = pd.concat(partes, ignore_index=True)
    info_dmd = {}
    if ds == "uta":
        T0 = {r["subject"]: T_arq[r["id"]] for _, r in idx.iterrows() if str(r["label"]) == "0"}
        df = marcar_uta(df, T0)
    elif ds == "nitymed":
        df = marcar_nitymed(df, T_arq)
    else:
        df, info_dmd = marcar_dmd(df)
    usar_pseudo = {u: v for u, v in pseudo.items() if not df.loc[df["unidade_ref"] == u, "janela_ref"].any()}
    df, stats = normalizar(df, df["janela_ref"].astype(bool), "unidade_ref", usar_pseudo)
    # rótulos e elegibilidade
    if ds == "uta":
        kss = df["rotulo_video"].astype(int)
        df["rotulo_kss"] = kss
        df["classe"] = kss.map({0: 0, 5: 1, 10: 2})
        df["B1"] = (kss >= 5).astype(int)                         # alerta x (pouca vigilância + sonolento)
        df["B2"] = (kss == 10).astype(int)                        # (alerta + pouca vigilância) x sonolento
        df["B3"] = np.where(kss == 5, np.nan, (kss == 10).astype(float))  # alerta x sonolento (5 descartado)
        df["avaliavel"] = df["uso"] == "exemplo"
    elif ds == "nitymed":
        df["avaliavel"] = df["uso"] == "exemplo"
    else:
        df["avaliavel"] = df["y"].notna()
    # estatísticas de referência por unidade
    tipo_ref = bidx.groupby("unidade_ref")["tipo_ref"].first()
    stats["tipo_ref_limiar"] = stats["unidade_ref"].map(tipo_ref)
    if ds == "dmd":
        stats["tipo_ref_janelas"] = stats["unidade_ref"].map(lambda u: info_dmd.get(u, {}).get("tipo_ref"))
        stats["ref_fallback"] = stats["unidade_ref"].map(lambda u: info_dmd.get(u, {}).get("ref_fallback"))
        stats["n_janelas_s3"] = stats["unidade_ref"].map(lambda u: info_dmd.get(u, {}).get("n_janelas_s3"))
    elif ds == "uta":
        stats["tipo_ref_janelas"] = "ultimo_terco_video0"
        stats["ref_fallback"] = False
    else:
        stats["tipo_ref_janelas"] = np.where(stats["pseudo_ref"], "primeiro_terco_pseudo", "primeiro_terco")
        stats["ref_fallback"] = stats["pseudo_ref"]
    stats.insert(0, "janela_s", janela)
    stats.insert(1, "passo_s", passo)
    extra_cols = [c for c in ["rotulo_kss", "classe", "B1", "B2", "B3", "y", "y_sens_s1", "y_sens_s5_sem_sinal",
                              "tem_s5", "rotulo_anotado", "tem_fechamento_anotado", "tem_bocejo_anotado"]
                  if c in df.columns]
    aux = ["n_quadros", "n_rosto", "n_piscadas", "limiar_ear", "ear_ref_mediana", "i0", "i1"]
    cols = ID_COLS + extra_cols + FEATURES + [f + "_c" for f in FEATURES] + [f + "_z" for f in FEATURES] + aux
    df.insert(0, "janela_s", janela)
    df.insert(1, "passo_s", passo)
    df = df[["janela_s", "passo_s"] + cols]
    rl = rotulo_L(janela)
    out = B.FEAT / ds / f"windows_{rl}.csv"
    tmp = out.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out)
    stats.to_csv(B.FEAT / ds / f"windows_referencia_{rl}.csv", index=False)
    msg = ""
    if ds == "dmd":
        msg = f"; janelas safe_drive (s1/s2/s3): ancoradas {int(df['sessao'].isin(['s1', 's2', 's3']).sum())}, " \
              f"grade global teria {grade_global_safe}"
    log.info("%s [%s]: %d janelas de %d arquivos em %.1f s -> %s%s", ds, rl, len(df), df["arquivo"].nunique(),
             time.time() - t0, out.relative_to(ROOT), msg)
    log.info("%s [%s]: por uso: %s", ds, rl, df["uso"].value_counts().to_dict())
    return df, dict(grade_global_safe=grade_global_safe)


def resumo(df, ds, janela):
    """Contagens por arquivo e uso (linhas de results/exp5_janelas_resumo.csv)."""
    rows = []
    for (suj, arq), g in df.groupby(["sujeito", "arquivo"], sort=True):
        r = dict(janela_s=janela, dataset=ds, sujeito=suj, video=g["video"].iloc[0], arquivo=arq,
                 parte=g["parte"].iloc[0], sessao=g["sessao"].iloc[0], rotulo_video=g["rotulo_video"].iloc[0],
                 n_janelas=len(g), n_parciais=int(g["parcial"].sum()), n_ref=int(g["janela_ref"].sum()),
                 n_fronteira=int(g["sobrepoe_ref"].sum()), n_avaliaveis=int(g["avaliavel"].sum()),
                 frac_nan_dur_media=float(g["dur_media_piscada_s"].isna().mean()),
                 perclos_media=float(g["perclos"].mean()), taxa_piscadas_min_media=float(g["taxa_piscadas_min"].mean()),
                 fold=g["fold"].iloc[0], oculos=g["oculos"].iloc[0], categoria=g["categoria"].iloc[0])
        for u in USOS_DMD + ("exemplo",):
            r[f"n_{u}"] = int((g["uso"] == u).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def gravar_resumos(res_novo, janela):
    """Atualiza results/exp5_janelas_resumo.csv e results/exp5_dmd_balanco_janelas.csv (substitui as linhas do
    comprimento de janela recalculado, mantém os demais)."""
    for path, filtro in ((RES / "exp5_janelas_resumo.csv", None), (RES / "exp5_dmd_balanco_janelas.csv", "dmd")):
        novo = res_novo if filtro is None else res_novo[res_novo["dataset"] == filtro]
        if path.exists():
            antigo = pd.read_csv(path)
            if "janela_s" in antigo.columns:
                manter = antigo[~((antigo["janela_s"] == janela) & antigo["dataset"].isin(novo["dataset"].unique()))]
                novo = pd.concat([manter, novo], ignore_index=True)
        novo = novo.sort_values(["janela_s", "dataset", "sujeito", "arquivo"]).reset_index(drop=True)
        tmp = path.with_suffix(".tmp.csv")
        novo.to_csv(tmp, index=False)
        os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="Características por janela (padrão: 10 s, passo 5 s).")
    ap.add_argument("--dataset", default="todos", choices=list(B.DATASETS) + ["todos"])
    ap.add_argument("--janela", type=float, default=JANELA_S, help="comprimento da janela em s (padrão 10)")
    ap.add_argument("--passo", type=float, default=PASSO_S, help="passo em s (padrão 5)")
    ap.add_argument("--mar", type=float, default=MAR_LIMIAR, help="limiar de MAR para bocejo (padrão 0,5)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    log = logging.getLogger("windows")
    lista = B.DATASETS if args.dataset == "todos" else (args.dataset,)
    if "uta" in lista:
        log.info("Folds da UTA-RLDD com .ok: %s", B.uta_folds_ok())
    resumos, diag = [], {}
    for ds in lista:
        df, info = processar_dataset(ds, janela=args.janela, passo=args.passo, mar_limiar=args.mar, log=log)
        if df is not None:
            resumos.append(resumo(df, ds, args.janela))
            diag[ds] = info
    if resumos:
        gravar_resumos(pd.concat(resumos, ignore_index=True), args.janela)
    if "dmd" in diag:
        p = RES / "exp5_dmd_grade_safe.csv"
        linha = pd.DataFrame([dict(janela_s=args.janela, passo_s=args.passo,
                                   grade_global_safe=diag["dmd"]["grade_global_safe"])])
        if p.exists():
            antigo = pd.read_csv(p)
            linha = pd.concat([antigo[antigo["janela_s"] != args.janela], linha], ignore_index=True)
        linha.sort_values("janela_s").to_csv(p, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
