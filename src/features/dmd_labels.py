"""Rótulos do DMD (OpenLABEL), alinhamento anotação <-> vídeo e validação do detector de piscadas do Pipeline A.

Rodar no .venv, DEPOIS de src/features/blinks.py:

    python src/features/dmd_labels.py

RODADA FINAL (27/09/2026). O DMD agora tem a sessão s5 (sonolência ATUADA; 16 sessões de 13 sujeitos) e as sessões
do pacote de distração s1 (carro em movimento), s2 e s3 (carro parado) dos mesmos sujeitos + gE_28 (sem s5).
  1. s5: lê o OpenLABEL e monta vetores por quadro: estado do olho (-1 sem anotação, 0 aberto, 1 fechado,
     2 abrindo, 3 fechando), piscada (blinks/blinking), bocejo (com e sem mão). Grava em
     data/features/final/dmd/rotulos_quadros/<id>.npz.
  2. ALINHAMENTO anotação <-> *_rgb_face.mp4:
     s5 (como na rodada preliminar): nº de quadros do vídeo x face_camera.total_frames, fim das ações,
        streams.face_camera.sync.frame_shift e correlação cruzada (lags de -30 a +30 quadros) entre (a) o indicador
        de "eyes_state/close" e -EAR e (b) o indicador anotado de "olho não aberto" e o nosso EAR < limiar;
     s1/s2/s3 (distração): a anotação NÃO tem estado do olho; o sinal ocular disponível é o olhar
        ("gaze_on_road/not_looking_road"). Além da contagem de quadros, correlação cruzada (lags +-30 e +-300 quadros)
        entre o indicador de "não olhando a estrada" e (c) -EAR (olhar para baixo/lado abaixa a pálpebra) e
        (d) o desvio da pose da cabeça (|yaw_proxy| e |pitch_proxy| padronizados pela mediana/IQR da sessão);
        e (e) entre "talking/talking" e a atividade da boca (DP móvel de 0,5 s do MAR). Também a taxa de rosto
        dentro e fora de "não olhando a estrada". lag > 0 = sinal do vídeo atrasado em relação à anotação.
  3. VALIDAÇÃO do detector nas s5 (limiar = o de blinks.py: 0,75 x mediana do EAR no safe_drive da s3 do sujeito):
     - piscadas: nossos eventos (>= 2 quadros abaixo do limiar, incluindo fechamentos longos) x
       intervalos "blinks/blinking" anotados (que incluem fechando+fechado+abrindo e também os
       fechamentos longos); casamento UM-PARA-UM guloso pela maior sobreposição (sobreposição >= 1 quadro);
       precisão, revocação e F1 por sessão;
     - fechamentos >= 0,5 s: nossos eventos com dur_s >= 0,5 x intervalos "eyes_state/close" >= 0,5 s;
     - por quadro (só quadros com rosto): EAR < limiar x eyes_state/close, avaliado nos quadros anotados
       como aberto ou fechado (transições excluídas); e a variante em que abrindo/fechando contam como
       "não aberto";
     - bocejo: MAR > limiar sustentado >= 1 s x bocejo anotado (por evento e por quadro);
     - sensibilidade: k do limiar de EAR em {0,60 ... 0,90} (x a mesma mediana de referência) e limiar de MAR
       em {0,3 ... 0,6}.
Saídas: results/exp5_dmd_alinhamento.csv (s5), results/exp5_dmd_alinhamento_distracao.csv (s1/s2/s3),
        results/exp5_dmd_validacao_detector.csv, results/exp5_dmd_sensibilidade_limiar.csv,
        results/exp5_dmd_bocejo_mar.csv, results/exp5_dmd_bocejos.csv (um bocejo anotado por linha).
Os rótulos por JANELA (usados em windows.py) vêm de rotular_janelas() e trechos_safe() deste módulo.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blinks as B  # noqa: E402

ROOT = B.ROOT
RES = ROOT / "results"
OUT_Q = B.FEAT / "dmd" / "rotulos_quadros"
MAXLAG_AMPLO = 300       # busca ampla (~ +-10 s) no alinhamento das sessões de distração
NAO_OLHA = "gaze_on_road/not_looking_road"
FALANDO = "talking/talking"

ESTADOS = {"eyes_state/open": 0, "eyes_state/close": 1, "eyes_state/opening": 2, "eyes_state/closing": 3}
BOCEJOS = ("yawning/Yawning with hand", "yawning/Yawning without hand")
MAXLAG = 30
FECH_MIN_S = 0.5          # fechamento que torna a janela "sonolento_atuado"
SOBREP_MIN_S = 0.5        # parte do evento anotado que precisa estar DENTRO da janela
MAR_LIMIAR = 0.5          # bocejo: MAR acima disso ...
MAR_SUST_S = 1.0          # ... sustentado por pelo menos 1 s
KS_SENS = (0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
MARS_SENS = (0.3, 0.4, 0.5, 0.6)


# --------------------------------------------------------------------------------------
# Leitura do OpenLABEL
# --------------------------------------------------------------------------------------
def ler_openlabel(path):
    """Devolve dict com 'acoes' (tipo -> lista de (ini, fim) inclusivos), total_frames, frame_shift,
    intervalo do arquivo e atributos do motorista/contexto."""
    d = json.load(open(path))["openlabel"]
    acoes = {}
    for a in d.get("actions", {}).values():
        acoes.setdefault(a["type"], []).extend(
            (int(iv["frame_start"]), int(iv["frame_end"])) for iv in a.get("frame_intervals", []))
    for k in acoes:
        acoes[k].sort()
    fc = d["streams"]["face_camera"]["stream_properties"]
    attrs = {}
    obj = next(iter(d.get("objects", {}).values()), {})
    for kind in ("num", "text", "boolean"):
        for e in obj.get("object_data", {}).get(kind, []):
            attrs[e["name"]] = e["val"]
    ctx = next(iter(d.get("contexts", {}).values()), {})
    for e in ctx.get("context_data", {}).get("text", []):
        attrs[e["name"]] = e["val"]
    fi = d.get("frame_intervals", [{}])
    return dict(acoes=acoes, total_frames=fc.get("total_frames"),
                frame_shift=fc.get("sync", {}).get("frame_shift"),
                arquivo_fim=fi[0].get("frame_end") if fi else None, attrs=attrs)


def indicador(intervalos, n):
    x = np.zeros(n, bool)
    for s, e in intervalos:
        x[max(s, 0): min(e, n - 1) + 1] = True
    return x


def vetores_quadro(acoes, n):
    """Vetores por quadro. estado_olho: -1 sem anotação, 0 aberto, 1 fechado, 2 abrindo, 3 fechando."""
    estado = np.full(n, -1, np.int8)
    conflitos = np.zeros(n, np.int16)
    for nome, cod in ESTADOS.items():
        m = indicador(acoes.get(nome, []), n)
        conflitos += m
        estado[m] = cod
    bm = indicador(acoes.get(BOCEJOS[0], []), n)
    bs = indicador(acoes.get(BOCEJOS[1], []), n)
    return dict(estado_olho=estado, piscada=indicador(acoes.get("blinks/blinking", []), n),
                bocejo=bm | bs, bocejo_com_mao=bm, bocejo_sem_mao=bs,
                n_conflitos_estado=int((conflitos > 1).sum()))


def intervalos_bocejo(acoes):
    return sorted(acoes.get(BOCEJOS[0], []) + acoes.get(BOCEJOS[1], []))


# --------------------------------------------------------------------------------------
# Alinhamento
# --------------------------------------------------------------------------------------
def _corr_lag(a, x, lag):
    """Correlação de Pearson entre a[t] e x[t + lag] (x pode ter NaN)."""
    n = len(a)
    if lag >= 0:
        aa, xx = a[: n - lag], x[lag:]
    else:
        aa, xx = a[-lag:], x[: n + lag]
    m = np.isfinite(xx)
    if m.sum() < 100 or np.std(aa[m]) == 0 or np.std(xx[m]) == 0:
        return np.nan
    return float(np.corrcoef(aa[m].astype(float), xx[m].astype(float))[0, 1])


def melhor_lag(a, x, maxlag=MAXLAG):
    lags = np.arange(-maxlag, maxlag + 1)
    cs = np.array([_corr_lag(a, x, int(L)) for L in lags])
    i = int(np.nanargmax(cs))
    return int(lags[i]), float(cs[i]), float(cs[maxlag])


# --------------------------------------------------------------------------------------
# Casamento de intervalos e métricas
# --------------------------------------------------------------------------------------
def casar_intervalos(det, ann):
    """Casamento um-para-um guloso pela maior sobreposição (em quadros). det/ann: listas de (ini, fim).

    Devolve lista de pares (i_det, j_ann, sobreposicao)."""
    det = list(det)
    ann = list(ann)
    cand = []
    if not det or not ann:
        return []
    starts = np.array([s for s, _ in ann])
    ends = np.array([e for _, e in ann])
    for i, (s, e) in enumerate(det):
        js = np.flatnonzero((starts <= e) & (ends >= s))
        for j in js:
            ov = min(e, ends[j]) - max(s, starts[j]) + 1
            if ov > 0:
                cand.append((ov, i, int(j)))
    cand.sort(key=lambda c: (-c[0], c[1], c[2]))
    usados_d, usados_a, pares = set(), set(), []
    for ov, i, j in cand:
        if i in usados_d or j in usados_a:
            continue
        usados_d.add(i)
        usados_a.add(j)
        pares.append((i, j, ov))
    return pares


def prf(n_det, n_ann, tp):
    p = tp / n_det if n_det else np.nan
    r = tp / n_ann if n_ann else np.nan
    f = 2 * p * r / (p + r) if (n_det and n_ann and (p + r) > 0) else np.nan
    return p, r, f


def metricas_binarias(y, yhat):
    y = np.asarray(y, bool)
    yhat = np.asarray(yhat, bool)
    tp = int((y & yhat).sum()); tn = int((~y & ~yhat).sum())
    fp = int((~y & yhat).sum()); fn = int((y & ~yhat).sum())
    n = tp + tn + fp + fn
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    esp = tn / (tn + fp) if (tn + fp) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else np.nan
    acc = (tp + tn) / n if n else np.nan
    bal = np.nanmean([sens, esp]) if n else np.nan
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (n * n) if n else np.nan
    kappa = (acc - pe) / (1 - pe) if (n and pe < 1) else np.nan
    return dict(n=n, tp=tp, tn=tn, fp=fp, fn=fn, acuracia=acc, acuracia_bal=bal, sensibilidade=sens,
                especificidade=esp, precisao=prec, kappa=kappa)


def runs_sustentados(mask, d, min_s):
    """Intervalos maximais de mask com duração (soma de d) >= min_s."""
    s_arr, e_arr = B.runs(mask)
    out = []
    for s, e in zip(s_arr, e_arr):
        if float(np.sum(d[s:e + 1])) >= min_s:
            out.append((int(s), int(e)))
    return out


# --------------------------------------------------------------------------------------
# Rótulos por janela
# --------------------------------------------------------------------------------------
def rotular_janelas(janelas, t, acoes, fps):
    """Sinais anotados de sonolência em cada janela (DataFrame com t_ini, t_fim) de uma sessão s5 do DMD.

    tem_fechamento_anotado: a janela contém (>= 0,5 s DENTRO da janela) um intervalo eyes_state/close com duração
    >= 0,5 s; tem_bocejo_anotado: >= 0,5 s DENTRO da janela de um bocejo anotado (com ou sem mão).
    rotulo_anotado: 'sonolento_atuado' se houver qualquer um dos dois; 's5_sem_sinal' caso contrário (na rodada
    preliminar este segundo caso se chamava 'alerta'; na final essas janelas saem do principal)."""
    fech = [(s, e) for s, e in acoes.get("eyes_state/close", []) if (e - s + 1) / fps >= FECH_MIN_S]
    boc = intervalos_bocejo(acoes)
    rot, mf, mb = [], [], []
    for _, w in janelas.iterrows():
        i0 = int(np.searchsorted(t, w["t_ini"], "left"))
        i1 = int(np.searchsorted(t, w["t_fim"], "left")) - 1   # último quadro da janela
        tem_f = any((min(e, i1) - max(s, i0) + 1) / fps >= SOBREP_MIN_S for s, e in fech if s <= i1 and e >= i0)
        tem_b = any((min(e, i1) - max(s, i0) + 1) / fps >= SOBREP_MIN_S for s, e in boc if s <= i1 and e >= i0)
        rot.append("sonolento_atuado" if (tem_f or tem_b) else "s5_sem_sinal")
        mf.append(bool(tem_f))
        mb.append(bool(tem_b))
    return pd.DataFrame(dict(rotulo_anotado=rot, tem_fechamento_anotado=mf, tem_bocejo_anotado=mb),
                        index=janelas.index)


def trechos_safe(acoes):
    """Trechos maximais de driver_actions/safe_drive (quadros inclusivos; adjacentes/sobrepostos fundidos)."""
    return B.fundir_intervalos(acoes.get(B.SAFE_DRIVE, []))


# --------------------------------------------------------------------------------------
# Alinhamento das sessões de distração (sem estado do olho na anotação)
# --------------------------------------------------------------------------------------
def _padroniza_robusto(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return np.full(len(x), np.nan)
    med = np.median(x[ok])
    q75, q25 = np.percentile(x[ok], [75, 25])
    iqr = q75 - q25 if q75 > q25 else (np.std(x[ok]) or 1.0)
    return (x - med) / iqr


def desvio_pose(npz_path, face):
    """sqrt(z_yaw^2 + z_pitch^2) com yaw_proxy/pitch_proxy padronizados por mediana/IQR da sessão (NaN sem rosto)."""
    with np.load(npz_path, allow_pickle=False) as z:
        yaw = z["yaw_proxy"].astype(float)
        pit = z["pitch_proxy"].astype(float)
    yaw = np.where(face, yaw, np.nan)
    pit = np.where(face, pit, np.nan)
    return np.sqrt(_padroniza_robusto(yaw) ** 2 + _padroniza_robusto(pit) ** 2)


def atividade_boca(mar, face, janela=15):
    """Desvio-padrão móvel (janela centrada de `janela` quadros, ~0,5 s a 29,76 fps) do MAR; NaN sem rosto."""
    s = pd.Series(np.where(face, mar, np.nan))
    return s.rolling(janela, center=True, min_periods=max(3, janela // 2)).std().to_numpy()


def alinhamento_distracao(sinal, ol, npz_path):
    """Conferência do alinhamento anotação <-> vídeo numa sessão s1/s2/s3 (ver docstring do módulo)."""
    acoes, n, face = ol["acoes"], sinal["n"], sinal["face"]
    max_fim = max(e for v in acoes.values() for _, e in v) if acoes else np.nan
    safe = trechos_safe(acoes)
    out = dict(n_quadros_video=n, face_camera_total_frames=ol["total_frames"], face_frame_shift=ol["frame_shift"],
               fim_intervalo_arquivo=ol["arquivo_fim"], fim_max_acoes=max_fim,
               n_trechos_safe=len(safe), safe_drive_s=float(sum((e - s + 1) for s, e in safe) / sinal["fps"]),
               safe_drive_max_s=float(max(((e - s + 1) for s, e in safe), default=0) / sinal["fps"]))
    nao_olha = indicador(acoes.get(NAO_OLHA, []), n).astype(float)
    fala = indicador(acoes.get(FALANDO, []), n).astype(float)
    out["n_nao_olha"] = len(acoes.get(NAO_OLHA, []))
    out["frac_nao_olha"] = float(nao_olha.mean())
    out["n_falando"] = len(acoes.get(FALANDO, []))
    ear_nan = np.where(face, sinal["ear"], np.nan)
    sinais = {"naoolha_vs_menos_ear": (nao_olha, -ear_nan),
              "naoolha_vs_pose": (nao_olha, desvio_pose(npz_path, face)),
              "falando_vs_boca": (fala, atividade_boca(sinal["mar"], face))}
    for nome, (a, x) in sinais.items():
        if a.std() == 0:
            out.update({f"lag_{nome}": np.nan, f"corr_lag0_{nome}": np.nan, f"corr_melhor_{nome}": np.nan,
                        f"lag_amplo_{nome}": np.nan, f"corr_melhor_amplo_{nome}": np.nan})
            continue
        lag, c, c0 = melhor_lag(a, x, MAXLAG)
        lag2, c2, _ = melhor_lag(a, x, MAXLAG_AMPLO)
        out.update({f"lag_{nome}": lag, f"corr_lag0_{nome}": c0, f"corr_melhor_{nome}": c,
                    f"lag_amplo_{nome}": lag2, f"corr_melhor_amplo_{nome}": c2})
    m = nao_olha.astype(bool)
    out["taxa_rosto_nao_olha"] = float(face[m].mean()) if m.any() else np.nan
    out["taxa_rosto_olha"] = float(face[~m].mean()) if (~m).any() else np.nan
    return out


# --------------------------------------------------------------------------------------
# Validação por sessão
# --------------------------------------------------------------------------------------
def validar_sessao(sinal, ol, limiar, mar_limiar=MAR_LIMIAR):
    """Métricas do detector contra a anotação para um limiar de EAR dado."""
    ear, face, t, d, n = sinal["ear"], sinal["face"], sinal["t"], sinal["d"], sinal["n"]
    fps = sinal["fps"]
    acoes = ol["acoes"]
    ev = B.detectar_eventos(ear, face, t, d, limiar)
    out = {}
    # piscadas (todos os eventos) x blinks/blinking
    det = list(zip(ev["inicio"].astype(int), ev["fim"].astype(int)))
    ann = acoes.get("blinks/blinking", [])
    pares = casar_intervalos(det, ann)
    p, r, f = prf(len(det), len(ann), len(pares))
    out.update(n_piscadas_anot=len(ann), n_eventos_det=len(det), tp_piscadas=len(pares),
               precisao_piscadas=p, revocacao_piscadas=r, f1_piscadas=f)
    # tipos de erro: FP que sobrepõe piscada anotada já casada (fragmentação) x FP sem sobreposição;
    # FN cujo EAR mínimo (quadros com rosto) nunca fica abaixo do limiar ("vale raso") x outros
    cas_d = {i for i, _, _ in pares}
    cas_a = {j for _, j, _ in pares}
    fp_frag = sum(1 for i, (a, b) in enumerate(det) if i not in cas_d and any(sa <= b and ea >= a for sa, ea in ann))
    fn_raso = 0
    for j, (a, b) in enumerate(ann):
        if j in cas_a:
            continue
        seg = np.where(face[a:b + 1], ear[a:b + 1], np.nan)
        if not np.isfinite(seg).any() or np.nanmin(seg) >= limiar:
            fn_raso += 1
    n_fp = len(det) - len(pares)
    n_fn = len(ann) - len(pares)
    out.update(fp_fragmentacao=fp_frag, fp_sem_sobreposicao=n_fp - fp_frag, fn_vale_raso=fn_raso,
               fn_outros=n_fn - fn_raso)
    # só eventos do tipo "piscada" x piscadas anotadas <= 1,5 s (informativo)
    evp = ev[ev["tipo"] == "piscada"]
    detp = list(zip(evp["inicio"].astype(int), evp["fim"].astype(int)))
    annp = [(s, e) for s, e in ann if (e - s + 1) / fps <= B.FECHAMENTO_LONGO_S]
    pares_p = casar_intervalos(detp, annp)
    p2, r2, f2 = prf(len(detp), len(annp), len(pares_p))
    out.update(precisao_piscadas_curtas=p2, revocacao_piscadas_curtas=r2, f1_piscadas_curtas=f2)
    # fechamentos >= 0,5 s
    detf = [(int(s), int(e)) for s, e, ds in zip(ev["inicio"], ev["fim"], ev["dur_s"]) if ds >= FECH_MIN_S]
    annf = [(s, e) for s, e in acoes.get("eyes_state/close", []) if (e - s + 1) / fps >= FECH_MIN_S]
    pares_f = casar_intervalos(detf, annf)
    p3, r3, f3 = prf(len(detf), len(annf), len(pares_f))
    out.update(n_fech05_anot=len(annf), n_fech05_det=len(detf), precisao_fech05=p3, revocacao_fech05=r3,
               f1_fech05=f3)
    # por quadro
    vq = vetores_quadro(acoes, n)
    est = vq["estado_olho"]
    pred = face & (ear < limiar)
    m = face & ((est == 0) | (est == 1))
    mb = metricas_binarias(est[m] == 1, pred[m])
    out.update({f"q_{k}": v for k, v in mb.items()})
    m2 = face & (est >= 0)
    mb2 = metricas_binarias(est[m2] != 0, pred[m2])
    out.update({f"q2_{k}": v for k, v in mb2.items()})
    out.update(ear_mediana_fechado=float(np.median(ear[face & (est == 1)])) if np.any(face & (est == 1)) else np.nan,
               ear_mediana_aberto=float(np.median(ear[face & (est == 0)])) if np.any(face & (est == 0)) else np.nan)
    # bocejo
    boc_ann = intervalos_bocejo(acoes)
    boc_det = runs_sustentados(face & (sinal["mar"] > mar_limiar), d, MAR_SUST_S)
    pares_b = casar_intervalos(boc_det, boc_ann)
    p4, r4, f4 = prf(len(boc_det), len(boc_ann), len(pares_b))
    out.update(n_bocejos_anot=len(boc_ann), n_bocejos_det=len(boc_det), precisao_bocejo=p4,
               revocacao_bocejo=r4, f1_bocejo=f4)
    pred_b = indicador(boc_det, n)
    mbq = metricas_binarias(vq["bocejo"][face], pred_b[face])
    out.update(bocejo_q_sensibilidade=mbq["sensibilidade"], bocejo_q_especificidade=mbq["especificidade"],
               bocejo_q_precisao=mbq["precisao"])
    return out, ev, vq


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    log = logging.getLogger("dmd_labels")
    idx = B.listar_videos("dmd")
    bidx = pd.read_csv(B.FEAT / "dmd" / "blinks_index.csv").set_index("id")
    OUT_Q.mkdir(parents=True, exist_ok=True)
    al_rows, ald_rows, val_rows, sens_rows, mar_rows, boc_rows = [], [], [], [], [], []
    for _, r in idx.iterrows():
        vid = r["id"]
        sinal = B.carregar_sinal(ROOT / r["npz"])
        extra = json.loads(r["extra_json"])
        ol = ler_openlabel(ROOT / extra["annotation"])
        acoes = ol["acoes"]
        n = sinal["n"]
        fps = sinal["fps"]
        limiar = float(bidx.loc[vid, "limiar"])
        med_ref = float(bidx.loc[vid, "ear_ref_mediana"])
        oculos = bool(ol["attrs"].get("glasses"))
        base = dict(id=vid, sujeito=r["subject"], sessao=r["sessao"], setup=r["setup"], data_gravacao=r["data_gravacao"],
                    oculos=oculos)
        if r["sessao"] != "s5":
            # ---- alinhamento das sessões de distração ----
            a = alinhamento_distracao(sinal, ol, ROOT / r["npz"])
            ald_rows.append(dict(**base, **a))
            log.info("%s: quadros %d (total_frames %s, shift %s, fim ações %s) | não olha x -EAR lag %s (r0=%.3f, r*=%.3f) "
                     "| x pose lag %s | falando x boca lag %s", vid, n, a["face_camera_total_frames"], a["face_frame_shift"],
                     a["fim_max_acoes"], a["lag_naoolha_vs_menos_ear"], a["corr_lag0_naoolha_vs_menos_ear"],
                     a["corr_melhor_naoolha_vs_menos_ear"], a["lag_naoolha_vs_pose"], a["lag_falando_vs_boca"])
            continue
        # ---- s5: vetores por quadro ----
        vq = vetores_quadro(acoes, n)
        np.savez_compressed(OUT_Q / f"{vid}.npz", **{k: v for k, v in vq.items() if isinstance(v, np.ndarray)})
        # ---- s5: alinhamento ----
        max_fim = max(e for v in acoes.values() for _, e in v)
        close = indicador(acoes.get("eyes_state/close", []), n).astype(float)
        nao_aberto = (vq["estado_olho"] >= 1).astype(float)
        ear_nan = np.where(sinal["face"], sinal["ear"], np.nan)
        lag_a, c_a, c0_a = melhor_lag(close, -ear_nan)
        nosso = np.where(sinal["face"], (sinal["ear"] < limiar).astype(float), np.nan)
        lag_b, c_b, c0_b = melhor_lag(nao_aberto, nosso)
        blk = acoes.get("blinks/blinking", [])
        cls_ = acoes.get("eyes_state/close", [])
        n_close_em_blink = sum(1 for a, b in cls_ if any(sa <= a and b <= ea for sa, ea in blk))
        for tipo_b in BOCEJOS:
            for a, b in acoes.get(tipo_b, []):
                seg = sinal["mar"][a:b + 1]
                fseg = sinal["face"][a:b + 1]
                boc_rows.append(dict(id=vid, sujeito=r["subject"], oculos=oculos,
                                     tipo="com mão" if "with hand" in tipo_b else "sem mão", inicio=a, fim=b,
                                     dur_s=(b - a + 1) / fps, mar_pico=float(np.nanmax(seg)) if np.isfinite(seg).any() else np.nan,
                                     taxa_rosto=float(fseg.mean()) if len(fseg) else np.nan))
        al_rows.append(dict(**base, n_quadros_video=n,
                            n_close=len(cls_), n_close_dentro_de_blinking=n_close_em_blink,
                            face_camera_total_frames=ol["total_frames"], face_frame_shift=ol["frame_shift"],
                            fim_intervalo_arquivo=ol["arquivo_fim"], fim_max_acoes=max_fim,
                            conflitos_estado=vq["n_conflitos_estado"],
                            lag_close_vs_ear=lag_a, corr_lag0_close_vs_ear=c0_a, corr_melhor_close_vs_ear=c_a,
                            lag_naoaberto_vs_limiar=lag_b, corr_lag0_naoaberto_vs_limiar=c0_b,
                            corr_melhor_naoaberto_vs_limiar=c_b))
        # ---- s5: validação (k = 0,75, o limiar de blinks.py) ----
        met, ev, _ = validar_sessao(sinal, ol, limiar)
        val_rows.append(dict(**base, limiar=limiar, ear_ref_mediana=med_ref,
                             tipo_ref=bidx.loc[vid, "tipo_ref"], taxa_rosto=float(sinal["face"].mean()), **met))
        # ---- s5: sensibilidade ao k do limiar (mesma mediana de referência) ----
        for k in KS_SENS:
            lim_k = k * med_ref
            mk, _, _ = validar_sessao(sinal, ol, lim_k)
            sens_rows.append(dict(id=vid, sujeito=r["subject"], oculos=oculos, k=k, limiar=lim_k,
                                  **{c: mk[c] for c in ("n_piscadas_anot", "n_eventos_det", "tp_piscadas",
                                                        "precisao_piscadas", "revocacao_piscadas", "f1_piscadas",
                                                        "n_fech05_anot", "n_fech05_det", "precisao_fech05",
                                                        "revocacao_fech05", "f1_fech05", "q_acuracia",
                                                        "q_acuracia_bal", "q_sensibilidade", "q_especificidade",
                                                        "q_kappa", "q_tp", "q_tn", "q_fp", "q_fn")}))
        # ---- s5: sensibilidade do limiar de MAR (bocejo) ----
        for ml in MARS_SENS:
            boc_ann = intervalos_bocejo(acoes)
            boc_det = runs_sustentados(sinal["face"] & (sinal["mar"] > ml), sinal["d"], MAR_SUST_S)
            pares_b = casar_intervalos(boc_det, boc_ann)
            p4, r4, f4 = prf(len(boc_det), len(boc_ann), len(pares_b))
            pred_b = indicador(boc_det, n)
            mbq = metricas_binarias(vq["bocejo"][sinal["face"]], pred_b[sinal["face"]])
            mar_rows.append(dict(id=vid, sujeito=r["subject"], oculos=oculos, mar_limiar=ml, sustentado_s=MAR_SUST_S,
                                 n_bocejos_anot=len(boc_ann), n_bocejos_det=len(boc_det), tp=len(pares_b),
                                 precisao=p4, revocacao=r4, f1=f4, q_sensibilidade=mbq["sensibilidade"],
                                 q_especificidade=mbq["especificidade"], q_fp=mbq["fp"]))
        log.info("%s: lag close~EAR=%+d (r0=%.3f, r*=%.3f) | piscadas P=%.3f R=%.3f F1=%.3f | quadro acc=%.3f bal=%.3f",
                 vid, lag_a, c0_a, c_a, met["precisao_piscadas"], met["revocacao_piscadas"], met["f1_piscadas"],
                 met["q_acuracia"], met["q_acuracia_bal"])
    RES.mkdir(exist_ok=True)
    pd.DataFrame(al_rows).to_csv(RES / "exp5_dmd_alinhamento.csv", index=False)
    ald = pd.DataFrame(ald_rows)
    ald.to_csv(RES / "exp5_dmd_alinhamento_distracao.csv", index=False)
    val = pd.DataFrame(val_rows)
    val.to_csv(RES / "exp5_dmd_validacao_detector.csv", index=False)
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(RES / "exp5_dmd_sensibilidade_limiar.csv", index=False)
    pd.DataFrame(mar_rows).to_csv(RES / "exp5_dmd_bocejo_mar.csv", index=False)
    pd.DataFrame(boc_rows).to_csv(RES / "exp5_dmd_bocejos.csv", index=False)
    # resumo no log
    if len(ald):
        ok_q = (ald["n_quadros_video"] == ald["face_camera_total_frames"]).sum()
        log.info("Distração: %d sessões; nº de quadros = total_frames em %d; frame_shift=0 em %d; fim das ações <= n-1 em %d",
                 len(ald), ok_q, int((ald["face_frame_shift"] == 0).sum()),
                 int((ald["fim_max_acoes"] <= ald["n_quadros_video"] - 1).sum()))
        for nome in ("naoolha_vs_menos_ear", "naoolha_vs_pose", "falando_vs_boca"):
            log.info("  %s: melhor lag (+-%d) min %s máx %s | lag amplo (+-%d) min %s máx %s", nome, MAXLAG,
                     ald[f"lag_{nome}"].min(), ald[f"lag_{nome}"].max(), MAXLAG_AMPLO,
                     ald[f"lag_amplo_{nome}"].min(), ald[f"lag_amplo_{nome}"].max())
    cols = ["precisao_piscadas", "revocacao_piscadas", "f1_piscadas", "precisao_fech05", "revocacao_fech05",
            "f1_fech05", "q_acuracia", "q_acuracia_bal", "q_sensibilidade", "q_especificidade", "q_kappa",
            "precisao_bocejo", "revocacao_bocejo"]
    log.info("Média por sessão s5 (k=0,75):\n%s", val[cols].mean().round(4).to_string())
    tp, nd, na = val["tp_piscadas"].sum(), val["n_eventos_det"].sum(), val["n_piscadas_anot"].sum()
    log.info("Agregado (micro) piscadas: P=%.4f R=%.4f (TP=%d, det=%d, anot=%d)", tp / nd, tp / na, tp, nd, na)
    log.info("Erros: FP fragmentação=%d, FP sem sobreposição=%d, FN vale raso=%d, FN outros=%d",
             val["fp_fragmentacao"].sum(), val["fp_sem_sobreposicao"].sum(), val["fn_vale_raso"].sum(),
             val["fn_outros"].sum())
    g = sens.groupby("k")
    tab = pd.DataFrame(dict(P_macro=g["precisao_piscadas"].mean(), R_macro=g["revocacao_piscadas"].mean(),
                            F1_macro=g["f1_piscadas"].mean(), acc_bal_macro=g["q_acuracia_bal"].mean(),
                            F1_fech05_macro=g["f1_fech05"].mean()))
    log.info("Sensibilidade ao k:\n%s", tab.round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
