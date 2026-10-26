"""Detector de piscadas sobre o sinal de EAR (Pipeline A) e características de Ghoddoosian et al. (2019).

Rodar no ambiente .venv (não precisa de MediaPipe; lê os sinais já extraídos):

    python src/features/blinks.py --dataset todos        # ou dmd | nitymed | uta
    python src/features/blinks.py --dataset uta --force  # recalcula tudo

Entrada:  data/signals/<base>/<id>.npz  (EAR/MAR por quadro; ver src/features/README.md)
Saídas (RODADA FINAL, versão 2.x; a rodada preliminar ficou em data/features/<base>/):
          data/features/final/<base>/blinks/<id>.csv   um evento (piscada ou fechamento longo) por linha
          data/features/final/<base>/blinks_index.csv  uma linha por ARQUIVO (limiar, referência, contagens, taxas)
          data/features/final/<base>/blinks/_parametros.json

Algoritmo (decisões documentadas também em src/features/README.md e results/exp5_caracteristicas.md):
  * Limiar adaptativo por UNIDADE DE REFERÊNCIA (sujeito na UTA-RLDD e no DMD; vídeo na NITYMED):
    limiar = K_LIMIAR (0,75) x mediana do EAR nos quadros com rosto do TRECHO DE REFERÊNCIA alerta, que é
    o mesmo trecho que define a normalização das janelas e NUNCA é avaliado (regra final de 27/09/2026):
        UTA-RLDD: ÚLTIMO TERÇO (no tempo) do vídeo 0 do sujeito (como o Preprocessing.py original; antes: vídeo 0
                  inteiro, o que calibrava o limiar também com as janelas alerta avaliadas);
        DMD:      quadros dentro dos trechos "driver_actions/safe_drive" das sessões s3 do sujeito (pacote de
                  distração; carro parado); se o sujeito não tiver s3 com >= 10 s de safe_drive com rosto,
                  o último terço (em tempo acumulado de safe_drive) das sessões s2;
        NITYMED:  o primeiro terço do próprio vídeo (DESVIO: não há vídeo alerta nessa base).
    O limiar vale para TODOS os arquivos da unidade (UTA: vídeos 0, 5, 10 e partes; DMD: s1, s2, s3 e s5).
  * "Olho fechado" no quadro = rosto detectado E EAR < limiar. Piscada = intervalo MAXIMAL de quadros
    fechados; quadros sem rosto interrompem o intervalo (a piscada em curso termina ali e o evento
    fica marcado como "interrompida"). Duração mínima: 2 quadros. Intervalos com duração > 1,5 s
    viram "fechamento_longo" (fora das 4 características e da contagem de piscadas).
  * start = primeiro quadro abaixo do limiar, end = último, bottom = argmin do EAR no intervalo.
  * 4 características de Ghoddoosian (em QUADROS, como no artigo), só para tipo == "piscada":
        Frequency = 100 * (nº de piscadas até i) / (nº de quadros COM ROSTO até end_i)
                    (o código original conta só quadros com rosto detectado — number_of_frames)
        Amplitude = (EAR[start] - 2*EAR[bottom] + EAR[end]) / 2
        Duration  = end - start + 1
        Velocity  = (EAR[end] - EAR[bottom]) / (end - bottom)   (0 quando bottom == end)
    e as versões independentes do fps: freq_hz (piscadas por segundo de rosto), dur_s, vel_s (EAR/s).
    Tempo = carimbo do contêiner (t_pts_s), robusto a vídeos de fps variável (ex.: F1_11_10 da UTA).
    Nos vídeos da UTA divididos em partes (F3_32_10, F5_49_10) as piscadas são detectadas por PARTE (cada parte
    é um arquivo; Frequency conta desde o início da parte); sequences.py concatena as partes com contagem contínua.
  * Variante "estendida" (colunas *_ext): as bordas start/end são deslocadas para o máximo local do EAR
    (suavizado com média móvel de 3 quadros) imediatamente antes/depois do intervalo, até 0,5 s e sem
    invadir o evento vizinho — mais próxima da definição original (bordas no olho aberto), em que a
    amplitude e a velocidade de abertura não ficam truncadas pelo limiar.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SIG = ROOT / "data" / "signals"
FEAT_PRELIMINAR = ROOT / "data" / "features"          # rodada preliminar (30 s, Fold 1, só s5) — não é mais escrita
FEAT = FEAT_PRELIMINAR / "final"                        # rodada final

VERSAO = "2.0"           # 2.0: referência do limiar = trecho de referência da unidade (UTA último terço do vídeo 0;
                         #      DMD safe_drive da s3); saída em data/features/final
K_LIMIAR = 0.75          # limiar = K x mediana do EAR da referência alerta
MIN_QUADROS = 2          # duração mínima de uma piscada (quadros)
FECHAMENTO_LONGO_S = 1.5 # acima disso o evento é "fechamento_longo"
REF_DMD_S = 30.0         # DMD, rodada preliminar: referência = primeiros 30 s da sessão (só diagnóstico agora)
MIN_REF_SAFE_S = 10.0    # DMD: mínimo de safe_drive com rosto na s3 para ela ser a referência do limiar
SAFE_DRIVE = "driver_actions/safe_drive"
EXT_MAX_S = 0.5          # variante estendida: deslocamento máximo de cada borda
DATASETS = ("dmd", "nitymed", "uta")

COLUNAS_EVENTO = [
    "evento", "tipo", "inicio", "fundo", "fim", "t_inicio_s", "t_fim_s", "dur_quadros", "dur_s",
    "ear_inicio", "ear_fundo", "ear_fim", "interrompida", "n_ordem",
    "frequencia", "amplitude", "duracao", "velocidade", "freq_hz", "vel_s",
    "inicio_ext", "fim_ext", "dur_ext_quadros", "dur_ext_s", "amplitude_ext", "velocidade_ext", "vel_ext_s",
]


# --------------------------------------------------------------------------------------
# Sinais
# --------------------------------------------------------------------------------------
def tempo_monotonico(t_pts, t_s):
    """Tempo (s) de cada quadro: carimbo do contêiner, forçado a ser não decrescente e a começar em 0.

    Usa t_s (quadro/fps) se o carimbo for inválido (NaN ou constante)."""
    t_pts = np.asarray(t_pts, dtype=float)
    if len(t_pts) == 0:
        return t_pts
    if not np.all(np.isfinite(t_pts)) or (len(t_pts) > 1 and np.nanmax(t_pts) <= 0):
        return np.asarray(t_s, dtype=float)
    t = np.maximum.accumulate(t_pts)
    return t - t[0]


def duracoes_quadro(t):
    """Duração (s) de cada quadro = intervalo até o próximo; o último recebe a mediana.

    Intervalos maiores que 3x a mediana (quadros perdidos, ex.: lacuna de 2,6 s em F1_06_10) são
    truncados em 3x a mediana, para que a lacuna não conte como tempo observado."""
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return np.full(len(t), np.nan)
    dt = np.diff(t)
    pos = dt[dt > 0]
    med = float(np.median(pos)) if len(pos) else np.nan
    d = np.append(dt, med)
    return np.clip(d, 0.0, 3.0 * med)


def carregar_sinal(npz_path):
    """Lê um .npz de data/signals e devolve dict com ear, mar, face, t, d, fps e meta."""
    with np.load(npz_path, allow_pickle=False) as z:
        ear = z["ear"].astype(float)
        mar = z["mar"].astype(float)
        face = z["face"].astype(bool)
        t = tempo_monotonico(z["t_pts_s"], z["t_s"])
        meta = {k[5:]: z[k].item() for k in z.files if k.startswith("meta_")}
        t_nominal = z["t_s"].astype(float)
    face = face & np.isfinite(ear)
    try:
        extra = json.loads(meta.get("extra_json", "{}") or "{}")
    except json.JSONDecodeError:
        extra = {}
    return dict(ear=ear, mar=mar, face=face, t=t, t_nominal=t_nominal, d=duracoes_quadro(t),
                fps=float(meta.get("fps", np.nan)), meta=meta, extra=extra, n=len(ear))


def duracao_total(sinal):
    """Duração do vídeo (s) = início do último quadro + sua duração."""
    if sinal["n"] == 0:
        return 0.0
    return float(sinal["t"][-1] + sinal["d"][-1])


# --------------------------------------------------------------------------------------
# Utilidades de intervalos
# --------------------------------------------------------------------------------------
def runs(mask):
    """Intervalos maximais de True: devolve (inicios, fins) inclusivos."""
    m = np.concatenate(([False], np.asarray(mask, dtype=bool), [False])).astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def suavizar3(x, valido):
    """Média móvel centrada de 3 quadros, ignorando quadros inválidos (NaN onde não há vizinho válido)."""
    x = np.where(valido, x, np.nan)
    pad = np.concatenate(([np.nan], x, [np.nan]))
    stack = np.vstack([pad[:-2], pad[1:-1], pad[2:]])
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(stack).sum(0)
        s = np.nansum(stack, 0)
    out = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return np.where(valido, out, np.nan)


PROMINENCIA_VALE = 0.06   # vales do EAR relativo à mediana móvel de 3 s (diagnóstico, independente do limiar)


def vales_ear(sinal):
    """Vales do EAR relativo à mediana móvel de 3 s: (quadros, profundidade relativa 1 - EAR/linha de base).

    Diagnóstico independente do limiar: quedas do EAR com proeminência >= 6% da linha de base, >= 3 quadros
    entre si. Usado para medir quantas quedas "tipo piscada" existem e quantas o limiar alcança."""
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks
    e = np.where(sinal["face"], sinal["ear"], np.nan)
    if not np.isfinite(e).any():
        return np.zeros(0, int), np.zeros(0)
    e = np.nan_to_num(e, nan=float(np.nanmedian(e)))
    fps = sinal["fps"] if np.isfinite(sinal["fps"]) and sinal["fps"] > 0 else 30.0
    base = median_filter(e, size=int(3 * fps) + 1)
    rel = e / np.maximum(base, 1e-6)
    pk, _ = find_peaks(-rel, prominence=PROMINENCIA_VALE, distance=3)
    return pk, 1.0 - rel[pk]


def eye_aspect_ratio(p):
    """EAR de um olho a partir de 6 pontos p1..p6 (array 6x2): (|p2-p6| + |p3-p5|) / (2|p1-p4|)."""
    p = np.asarray(p, dtype=float)
    h = np.linalg.norm(p[0] - p[3])
    if h < 1e-9:
        return np.nan
    return (np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2.0 * h)


# --------------------------------------------------------------------------------------
# Limiar e detecção
# --------------------------------------------------------------------------------------
def limiar_adaptativo(ear, face, mask_ref, k=K_LIMIAR):
    """(limiar, mediana) com a mediana do EAR nos quadros com rosto da máscara de referência."""
    sel = np.asarray(mask_ref, bool) & np.asarray(face, bool) & np.isfinite(ear)
    if sel.sum() == 0:
        return np.nan, np.nan
    med = float(np.median(ear[sel]))
    return k * med, med


def _estender(ear_sm, face, t, s, e, lo, hi, ext_max_s):
    """Bordas da variante estendida: sobe o EAR suavizado (estritamente crescente) a partir de s (para trás)
    e de e (para frente) até o máximo local, no máximo ext_max_s e sem invadir os eventos vizinhos."""
    k = s
    while (k - 1 >= lo and face[k - 1] and np.isfinite(ear_sm[k - 1]) and np.isfinite(ear_sm[k])
           and ear_sm[k - 1] > ear_sm[k] and (t[s] - t[k - 1]) <= ext_max_s):
        k -= 1
    s_ext = k
    k = e
    while (k + 1 <= hi and face[k + 1] and np.isfinite(ear_sm[k + 1]) and np.isfinite(ear_sm[k])
           and ear_sm[k + 1] > ear_sm[k] and (t[k + 1] - t[e]) <= ext_max_s):
        k += 1
    return s_ext, k


def detectar_eventos(ear, face, t, d, limiar, min_quadros=MIN_QUADROS, longo_s=FECHAMENTO_LONGO_S,
                     ext_max_s=EXT_MAX_S):
    """Detecta piscadas/fechamentos longos e calcula as características. Devolve DataFrame (COLUNAS_EVENTO).

    ear, face, t, d: arrays por quadro (EAR, rosto detectado, tempo em s, duração do quadro em s).
    """
    ear = np.asarray(ear, dtype=float)
    face = np.asarray(face, dtype=bool) & np.isfinite(ear)
    t = np.asarray(t, dtype=float)
    d = np.asarray(d, dtype=float)
    n = len(ear)
    if n == 0 or not np.isfinite(limiar):
        return pd.DataFrame(columns=COLUNAS_EVENTO)
    fechado = face & (ear < limiar)
    s_arr, e_arr = runs(fechado)
    keep = (e_arr - s_arr + 1) >= min_quadros
    s_arr, e_arr = s_arr[keep], e_arr[keep]
    ear_sm = suavizar3(ear, face)
    face_cum = np.cumsum(face)                           # nº de quadros com rosto em [0, i]
    tface_cum = np.cumsum(np.where(face, d, 0.0))        # tempo com rosto (s) em [0, i]
    rows = []
    n_pisc = 0
    for j, (s, e) in enumerate(zip(s_arr, e_arr)):
        s, e = int(s), int(e)
        b = s + int(np.argmin(ear[s:e + 1]))
        dur_q = e - s + 1
        dur_s = float(t[e] + d[e] - t[s])
        tipo = "fechamento_longo" if dur_s > longo_s else "piscada"
        interrompida = bool((s == 0) or (not face[s - 1]) or (e == n - 1) or (not face[e + 1]))
        amp = (ear[s] - 2.0 * ear[b] + ear[e]) / 2.0
        vel = (ear[e] - ear[b]) / (e - b) if e > b else 0.0
        vel_s = (ear[e] - ear[b]) / (t[e] - t[b]) if (e > b and t[e] > t[b]) else 0.0
        lo = int(e_arr[j - 1]) + 1 if j > 0 else 0
        hi = int(s_arr[j + 1]) - 1 if j + 1 < len(s_arr) else n - 1
        s_x, e_x = _estender(ear_sm, face, t, s, e, lo, hi, ext_max_s)
        amp_x = (ear[s_x] - 2.0 * ear[b] + ear[e_x]) / 2.0
        vel_x = (ear[e_x] - ear[b]) / (e_x - b) if e_x > b else 0.0
        vel_x_s = (ear[e_x] - ear[b]) / (t[e_x] - t[b]) if (e_x > b and t[e_x] > t[b]) else 0.0
        if tipo == "piscada":
            n_pisc += 1
            ordem = n_pisc
            freq = 100.0 * n_pisc / face_cum[e]
            freq_hz = n_pisc / tface_cum[e] if tface_cum[e] > 0 else np.nan
        else:
            ordem, freq, freq_hz = np.nan, np.nan, np.nan
        rows.append(dict(
            evento=j, tipo=tipo, inicio=s, fundo=b, fim=e, t_inicio_s=float(t[s]), t_fim_s=float(t[e] + d[e]),
            dur_quadros=dur_q, dur_s=dur_s, ear_inicio=float(ear[s]), ear_fundo=float(ear[b]),
            ear_fim=float(ear[e]), interrompida=interrompida, n_ordem=ordem,
            frequencia=freq, amplitude=float(amp), duracao=dur_q, velocidade=float(vel),
            freq_hz=freq_hz, vel_s=float(vel_s),
            inicio_ext=s_x, fim_ext=e_x, dur_ext_quadros=e_x - s_x + 1,
            dur_ext_s=float(t[e_x] + d[e_x] - t[s_x]), amplitude_ext=float(amp_x),
            velocidade_ext=float(vel_x), vel_ext_s=float(vel_x_s)))
    return pd.DataFrame(rows, columns=COLUNAS_EVENTO)


# --------------------------------------------------------------------------------------
# Descoberta de vídeos e unidades de referência
# --------------------------------------------------------------------------------------
def uta_folds_ok():
    """Folds da UTA-RLDD prontos (data/uta_rldd_videos/FoldN/.ok)."""
    out = []
    for ok in glob.glob(str(ROOT / "data" / "uta_rldd_videos" / "Fold*" / ".ok")):
        try:
            out.append(int(Path(ok).parent.name.replace("Fold", "")))
        except ValueError:
            pass
    return sorted(out)


def listar_videos(ds):
    """Arquivos com sinal extraído e completo (index.csv da extração). Na UTA, só folds com .ok.

    Colunas acrescentadas: fold, oculos, categoria, genero, video (vídeo LÓGICO: na UTA as partes _p1/_p2 de um
    vídeo dividido compartilham o mesmo; nas outras bases = id), parte, sessao (DMD: s1, s2, s3 ou s5),
    pacote, setup e data_gravacao (DMD)."""
    p = SIG / ds / "index.csv"
    if not p.exists():
        return pd.DataFrame()
    idx = pd.read_csv(p, dtype={"label": str, "session": str})
    if "complete" in idx.columns:
        idx = idx[idx["complete"].astype(str).str.lower() == "true"]
    idx = idx[idx["npz"].map(lambda f: (ROOT / f).exists())].copy()
    extras = idx["extra_json"].fillna("{}").map(json.loads)
    idx["fold"] = extras.map(lambda e: e.get("fold"))
    idx["oculos"] = extras.map(lambda e: e.get("glasses"))
    idx["categoria"] = extras.map(lambda e: e.get("category"))
    idx["genero"] = extras.map(lambda e: e.get("gender"))
    idx["video"] = [e.get("video") or i for e, i in zip(extras, idx["id"])]
    idx["parte"] = extras.map(lambda e: str(e.get("part") or ""))
    idx["sessao"] = extras.map(lambda e: e.get("sessao") or ("s5" if ds == "dmd" else ""))
    idx["pacote"] = extras.map(lambda e: e.get("pacote") or ("drowsiness" if ds == "dmd" else ""))
    idx["setup"] = extras.map(lambda e: e.get("setup") or "")
    idx["data_gravacao"] = extras.map(lambda e: e.get("recordTime") or "")
    idx["anotacao"] = extras.map(lambda e: e.get("annotation") or "")
    if ds == "uta":
        folds = uta_folds_ok()
        idx = idx[idx["fold"].isin(folds)]
    return idx.sort_values("id").reset_index(drop=True)


def fundir_intervalos(iv):
    """Funde intervalos (ini, fim) inclusivos que se sobrepõem ou se tocam (fim + 1 == ini seguinte)."""
    out = []
    for s, e in sorted((int(a), int(b)) for a, b in iv):
        if out and s <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def trechos_acao(anotacao, tipo=SAFE_DRIVE):
    """Trechos (ini, fim) inclusivos, em quadros, de uma ação do OpenLABEL do DMD, fundidos (ver fundir_intervalos).

    O índice da anotação é o índice do quadro do *_rgb_face.mp4 (deslocamento 0; conferido em dmd_labels.py)."""
    import dmd_labels as L   # import local: dmd_labels importa este módulo
    if not anotacao:
        return []
    return fundir_intervalos(L.ler_openlabel(ROOT / anotacao)["acoes"].get(tipo, []))


def plano_referencia(ds, idx):
    """Unidade de referência de cada arquivo e o(s) trecho(s) de referência de cada unidade.

    Devolve (unidade_de, refs): unidade_de[id] = unidade; refs[unidade] = dict(tipo, partes=[(id, tipo_mascara)],
    fallback). Tipos de máscara (ver mascara_referencia): 'ultimo_terco', 'primeiro_terco', 'safe_drive',
    'safe_drive_ultimo_terco'."""
    unidade_de, refs = {}, {}
    if ds == "uta":
        for suj, g in idx.groupby("subject"):
            v0 = g[g["label"].astype(str) == "0"]
            if len(v0):
                for vid in g["id"]:
                    unidade_de[vid] = suj
                refs[suj] = dict(tipo="ultimo_terco_video0", partes=[(v0.iloc[0]["id"], "ultimo_terco")], fallback=False)
            else:   # não ocorre nos 60 sujeitos; mantido por robustez
                for vid in g["id"]:
                    unidade_de[vid] = vid
                    refs[vid] = dict(tipo="primeiro_terco_proprio", partes=[(vid, "primeiro_terco")], fallback=True)
    elif ds == "dmd":
        for suj, g in idx.groupby("subject"):
            for vid in g["id"]:
                unidade_de[vid] = suj
            s3 = g[g["sessao"] == "s3"]
            t_safe = 0.0
            for _, r in s3.iterrows():
                sinal = carregar_sinal(ROOT / r["npz"])
                m = mascara_referencia(sinal, "safe_drive", r["anotacao"])
                t_safe += float(np.sum(np.where(m & sinal["face"], sinal["d"], 0.0)))
            if len(s3) and t_safe >= MIN_REF_SAFE_S:
                refs[suj] = dict(tipo="s3_safe_drive", partes=[(v, "safe_drive") for v in s3["id"]], fallback=False)
            else:
                s2 = g[g["sessao"] == "s2"].sort_values("data_gravacao")
                refs[suj] = dict(tipo="s2_safe_drive_ultimo_terco",
                                 partes=[(v, "safe_drive_ultimo_terco") for v in s2["id"]], fallback=True)
    else:
        for vid in idx["id"]:
            unidade_de[vid] = vid
            refs[vid] = dict(tipo="primeiro_terco_proprio", partes=[(vid, "primeiro_terco")], fallback=False)
    return unidade_de, refs


def mascara_referencia(sinal, tipo, anotacao=None):
    """Máscara de quadros do trecho de referência dentro de um arquivo de referência."""
    t = sinal["t"]
    T = duracao_total(sinal)
    if tipo == "video0":                      # rodada preliminar (UTA): vídeo 0 inteiro
        return np.ones(sinal["n"], bool)
    if tipo == "primeiros30s":                # rodada preliminar (DMD s5): primeiros 30 s da sessão
        return t < REF_DMD_S
    if tipo == "primeiro_terco":
        return t < T / 3.0
    if tipo == "ultimo_terco":
        return t >= 2.0 * T / 3.0
    if tipo in ("safe_drive", "safe_drive_ultimo_terco"):
        m = np.zeros(sinal["n"], bool)
        for s, e in trechos_acao(anotacao or sinal["extra"].get("annotation", "")):
            m[max(s, 0): min(e, sinal["n"] - 1) + 1] = True
        if tipo == "safe_drive_ultimo_terco" and m.any():
            acum = np.cumsum(np.where(m, sinal["d"], 0.0))
            m &= acum > (2.0 / 3.0) * acum[-1]
        return m
    raise ValueError(tipo)


def hash_parametros(k, limiar, tipo_ref):
    s = json.dumps(dict(v=VERSAO, k=k, min_q=MIN_QUADROS, longo=FECHAMENTO_LONGO_S, ext=EXT_MAX_S,
                        ref=tipo_ref, limiar=round(float(limiar), 8)), sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def mediana_preliminar(ds, r, sinal, idx_por_id):
    """Mediana do EAR na referência da RODADA PRELIMINAR (só diagnóstico da mudança de referência):
    UTA = vídeo 0 inteiro do sujeito; DMD = primeiros 30 s da própria sessão; NITYMED = primeiro terço (igual)."""
    if ds == "uta":
        return None   # calculada uma vez por sujeito em processar_dataset
    tipo = "primeiros30s" if ds == "dmd" else "primeiro_terco"
    m = mascara_referencia(sinal, tipo) & sinal["face"] & np.isfinite(sinal["ear"])
    return float(np.median(sinal["ear"][m])) if m.any() else np.nan


def resumo_video(ds, r, sinal, ev, limiar, med, unidade, ref, h, med_prelim):
    pisc = ev[ev["tipo"] == "piscada"]
    tface = float(np.sum(np.where(sinal["face"], sinal["d"], 0.0)))
    return dict(
        dataset=ds, id=r["id"], sujeito=r["subject"], rotulo=str(r["label"]) if pd.notna(r["label"]) else "",
        video=r["video"], parte=r["parte"], sessao=r["sessao"], setup=r["setup"], data_gravacao=r["data_gravacao"],
        fold=r.get("fold"), oculos=r.get("oculos"), categoria=r.get("categoria"), genero=r.get("genero"),
        unidade_ref=unidade, tipo_ref=ref["tipo"], videos_ref=";".join(v for v, _ in ref["partes"]),
        ref_fallback=bool(ref["fallback"]), n_quadros_ref=int(ref.get("n_quadros", 0)),
        fps=sinal["fps"], n_quadros=sinal["n"],
        dur_s=duracao_total(sinal), taxa_rosto=float(sinal["face"].mean()) if sinal["n"] else np.nan,
        ear_ref_mediana=med, limiar=limiar, ear_ref_mediana_prelim=med_prelim, n_piscadas=len(pisc),
        n_fechamentos_longos=int((ev["tipo"] == "fechamento_longo").sum()),
        n_interrompidas=int(ev["interrompida"].sum()) if len(ev) else 0,
        taxa_piscadas_min=len(pisc) / (tface / 60.0) if tface > 0 else np.nan,
        dur_media_piscada_s=float(pisc["dur_s"].mean()) if len(pisc) else np.nan,
        frac_vel_zero=float((pisc["velocidade"] == 0).mean()) if len(pisc) else np.nan,
        frac_fechado=float(np.mean(sinal["face"] & (sinal["ear"] < limiar))) if sinal["n"] else np.nan,
        **diagnostico_video(sinal, limiar), params_hash=h)


def diagnostico_video(sinal, limiar):
    """Tempo (fps variável, lacunas) e vales do EAR (quantos existem e quantos cruzam o limiar)."""
    t, n = sinal["t"], sinal["n"]
    out = dict(desvio_max_tempo_s=np.nan, maior_lacuna_s=np.nan, fps_local_min=np.nan, fps_local_max=np.nan,
               vales_por_min=np.nan, vales12_por_min=np.nan, vales25_por_min=np.nan, prof_rel_p90_vales=np.nan,
               frac_vales_abaixo_limiar=np.nan)
    if n < 2:
        return out
    out["desvio_max_tempo_s"] = float(np.nanmax(np.abs(t - sinal["t_nominal"])))
    out["maior_lacuna_s"] = float(np.max(np.diff(t)))
    blocos = np.floor(t / 60.0).astype(int)                 # fps local em blocos de 60 s (só blocos completos)
    fl = [np.sum(blocos == b) / 60.0 for b in np.unique(blocos)[:-1]]
    if fl:
        out["fps_local_min"], out["fps_local_max"] = float(min(fl)), float(max(fl))
    pk, prof = vales_ear(sinal)
    tface = float(np.sum(np.where(sinal["face"], sinal["d"], 0.0)))
    if tface > 0:
        out["vales_por_min"] = len(pk) / (tface / 60.0)
        out["vales12_por_min"] = float(np.sum(prof >= 0.12)) / (tface / 60.0)   # quedas >= 12% da linha de base
        out["vales25_por_min"] = float(np.sum(prof >= 0.25)) / (tface / 60.0)   # >= 25% (~ o que 0,75 x mediana exige)
    if len(pk):
        out["prof_rel_p90_vales"] = float(np.percentile(prof, 90))
        out["frac_vales_abaixo_limiar"] = float(np.mean(sinal["ear"][pk] < limiar))
    return out


def limiares_por_unidade(ds, idx, refs, k=K_LIMIAR):
    """Limiar de cada unidade = k x mediana do EAR (quadros com rosto) de TODOS os trechos de referência juntos.

    Devolve dict unidade -> (limiar, mediana, n_quadros_ref) e acrescenta n_quadros em refs[unidade]."""
    por_id = {r["id"]: r for _, r in idx.iterrows()}
    out = {}
    for u, ref in refs.items():
        vals = []
        for vid, tipo in ref["partes"]:
            r = por_id[vid]
            s = carregar_sinal(ROOT / r["npz"])
            m = mascara_referencia(s, tipo, r.get("anotacao")) & s["face"] & np.isfinite(s["ear"])
            vals.append(s["ear"][m])
        v = np.concatenate(vals) if vals else np.zeros(0)
        med = float(np.median(v)) if len(v) else np.nan
        ref["n_quadros"] = int(len(v))
        out[u] = (k * med if np.isfinite(med) else np.nan, med, int(len(v)))
    return out


def processar_dataset(ds, k=K_LIMIAR, force=False, log=None):
    log = log or logging.getLogger(__name__)
    idx = listar_videos(ds)
    if idx.empty:
        log.info("%s: nenhum sinal encontrado", ds)
        return pd.DataFrame()
    out_dir = FEAT / ds / "blinks"
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = FEAT / ds / "blinks_index.csv"
    antigo = (pd.read_csv(idx_path, dtype={"params_hash": str}) if idx_path.exists()
              else pd.DataFrame(columns=["id", "params_hash"]))
    antigo_h = dict(zip(antigo["id"], antigo["params_hash"])) if len(antigo) else {}
    unidade_de, refs = plano_referencia(ds, idx)
    lims = limiares_por_unidade(ds, idx, refs, k)
    # diagnóstico: referência da rodada preliminar na UTA (vídeo 0 inteiro)
    med_prelim_uta = {}
    if ds == "uta":
        for u, ref in refs.items():
            vid0 = ref["partes"][0][0]
            s0 = carregar_sinal(ROOT / idx.loc[idx["id"] == vid0, "npz"].iloc[0])
            m = s0["face"] & np.isfinite(s0["ear"])
            med_prelim_uta[u] = float(np.median(s0["ear"][m])) if m.any() else np.nan
    linhas, feitos, pulados = [], 0, 0
    t0 = time.time()
    for _, r in idx.iterrows():
        vid = r["id"]
        unidade = unidade_de[vid]
        ref = refs[unidade]
        limiar, med, _ = lims[unidade]
        h = hash_parametros(k, limiar, ref["tipo"])
        csv = out_dir / f"{vid}.csv"
        npz = ROOT / r["npz"]
        atual = (not force and csv.exists() and antigo_h.get(vid) == h
                 and csv.stat().st_mtime >= npz.stat().st_mtime)
        sinal = carregar_sinal(npz)
        if atual:
            ev = pd.read_csv(csv)
            pulados += 1
        else:
            ev = detectar_eventos(sinal["ear"], sinal["face"], sinal["t"], sinal["d"], limiar)
            tmp = csv.with_suffix(".tmp.csv")
            ev.to_csv(tmp, index=False)
            os.replace(tmp, csv)
            feitos += 1
        mp = med_prelim_uta.get(unidade) if ds == "uta" else mediana_preliminar(ds, r, sinal, None)
        linhas.append(resumo_video(ds, r, sinal, ev, limiar, med, unidade, ref, h, mp))
    res = pd.DataFrame(linhas)
    tmp = idx_path.with_suffix(".tmp.csv")
    res.to_csv(tmp, index=False)
    os.replace(tmp, idx_path)
    json.dump(dict(versao=VERSAO, k_limiar=k, min_quadros=MIN_QUADROS, fechamento_longo_s=FECHAMENTO_LONGO_S,
                   ext_max_s=EXT_MAX_S, min_ref_safe_s=MIN_REF_SAFE_S,
                   referencia={"uta": "último terço (no tempo) do vídeo 0 do sujeito",
                               "dmd": "trechos driver_actions/safe_drive das sessões s3 do sujeito "
                                      "(sem s3: último terço do safe_drive das s2)",
                               "nitymed": "primeiro terço do próprio vídeo (desvio)"}),
              open(out_dir / "_parametros.json", "w"), ensure_ascii=False, indent=1)
    log.info("%s: %d arquivos (%d calculados, %d já prontos) em %.1f s; piscadas=%d, fechamentos longos=%d; "
             "unidades de referência=%d (fallback: %s)", ds, len(res), feitos, pulados, time.time() - t0,
             int(res["n_piscadas"].sum()), int(res["n_fechamentos_longos"].sum()), len(refs),
             sorted(u for u, rf in refs.items() if rf["fallback"]) or "nenhuma")
    return res


def main():
    ap = argparse.ArgumentParser(description="Detecta piscadas no EAR e calcula as características de Ghoddoosian.")
    ap.add_argument("--dataset", default="todos", choices=list(DATASETS) + ["todos"])
    ap.add_argument("--k", type=float, default=K_LIMIAR, help="limiar = k x mediana do EAR da referência")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    log = logging.getLogger("blinks")
    if "uta" in (DATASETS if args.dataset == "todos" else (args.dataset,)):
        log.info("Folds da UTA-RLDD com .ok: %s", uta_folds_ok())
    for ds in (DATASETS if args.dataset == "todos" else (args.dataset,)):
        processar_dataset(ds, k=args.k, force=args.force, log=log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
