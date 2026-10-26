"""Testes (asserts puros) da etapa piscadas -> sequências -> janelas -> rótulos (Exp. 5; rodada final).

Rodar no .venv:
    python src/features/test_features.py        # ou: python -m pytest src/features/test_features.py -q
Não lê dados reais (tudo sintético), então roda em segundos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blinks as B  # noqa: E402
import dmd_labels as L  # noqa: E402
import sequences as S  # noqa: E402
import uta_oficial as O  # noqa: E402
import windows as W  # noqa: E402

FPS = 30.0


def _sinal(ear, face=None, fps=FPS, mar=None):
    ear = np.asarray(ear, float)
    n = len(ear)
    face = np.ones(n, bool) if face is None else np.asarray(face, bool)
    t = np.arange(n) / fps
    return dict(ear=np.where(face, ear, np.nan), face=face, t=t, d=B.duracoes_quadro(t), fps=fps, n=n,
                mar=np.zeros(n) if mar is None else np.asarray(mar, float))


# ------------------------------------------------------------------ EAR
def test_ear_sintetico():
    # p1..p6: cantos (0,0) e (4,0); pálpebras a +-1 em x=1 e x=3 -> EAR = (2 + 2) / (2 * 4) = 0,5
    p = np.array([[0, 0], [1, 1], [3, 1], [4, 0], [3, -1], [1, -1]], float)
    assert abs(B.eye_aspect_ratio(p) - 0.5) < 1e-12
    # olho fechado (pálpebras coincidem) -> 0; escala não muda o EAR
    q = p.copy(); q[[1, 2, 4, 5], 1] = 0
    assert abs(B.eye_aspect_ratio(q)) < 1e-12
    assert abs(B.eye_aspect_ratio(p * 7.3) - 0.5) < 1e-12
    # mesma fórmula/índices do extrator (extract_signals.measures_from_points)
    import extract_signals as X
    P = {}
    for idx, off in ((X.LEFT_EYE, 100.0), (X.RIGHT_EYE, 0.0)):
        for i, k in enumerate(idx):
            P[k] = p[i] + np.array([off, 0])
    P.update({61: np.array([0., 10]), 291: np.array([4., 10]), 13: np.array([2., 9]), 14: np.array([2., 11]),
              1: np.array([2., 5]), 152: np.array([2., 20]), 33: P[33], 263: P[263]})
    ear_l, ear_r, mar, *_ = X.measures_from_points(P)
    assert abs(ear_l - 0.5) < 1e-12 and abs(ear_r - 0.5) < 1e-12
    assert abs(mar - 0.5) < 1e-12  # |13-14| / |61-291| = 2 / 4


# ------------------------------------------------------------------ piscadas
def _tres_piscadas():
    ear = np.full(100, 0.30)
    ear[10:16] = [0.25, 0.18, 0.10, 0.15, 0.19, 0.28]  # A: abaixo de 0,2 em 11..14, fundo 12
    ear[40:43] = [0.12, 0.08, 0.11]                    # B: 40..42, fundo 41
    ear[70:72] = [0.15, 0.10]                          # C: 70..71, fundo no fim (velocidade 0)
    ear[90] = 0.10                                     # 1 quadro: ignorado (mínimo 2)
    return ear


def test_tres_piscadas_caracteristicas():
    s = _sinal(_tres_piscadas())
    ev = B.detectar_eventos(s["ear"], s["face"], s["t"], s["d"], limiar=0.2)
    assert list(ev["tipo"]) == ["piscada"] * 3
    assert list(zip(ev["inicio"], ev["fundo"], ev["fim"])) == [(11, 12, 14), (40, 41, 42), (70, 71, 71)]
    assert list(ev["duracao"]) == [4, 3, 2]
    np.testing.assert_allclose(ev["amplitude"], [(0.18 - 0.20 + 0.19) / 2, (0.12 - 0.16 + 0.11) / 2,
                                                 (0.15 - 0.20 + 0.10) / 2], atol=1e-12)
    np.testing.assert_allclose(ev["velocidade"], [(0.19 - 0.10) / 2, (0.11 - 0.08) / 1, 0.0], atol=1e-12)
    np.testing.assert_allclose(ev["dur_s"], np.array([4, 3, 2]) / FPS, atol=1e-9)
    np.testing.assert_allclose(ev["vel_s"], [(0.19 - 0.10) / (2 / FPS), (0.11 - 0.08) / (1 / FPS), 0.0], atol=1e-9)
    # variante estendida da piscada A: máximos locais do EAR suavizado em 8 e 17
    a = ev.iloc[0]
    assert (a["inicio_ext"], a["fim_ext"]) == (8, 17)
    assert abs(a["amplitude_ext"] - (0.30 - 0.20 + 0.30) / 2) < 1e-12
    assert abs(a["velocidade_ext"] - (0.30 - 0.10) / (17 - 12)) < 1e-12


def test_frequencia():
    s = _sinal(_tres_piscadas())
    ev = B.detectar_eventos(s["ear"], s["face"], s["t"], s["d"], limiar=0.2)
    # Frequency = 100 * (nº de piscadas até i) / (nº de quadros com rosto até end_i)
    np.testing.assert_allclose(ev["frequencia"], [100 * 1 / 15, 100 * 2 / 43, 100 * 3 / 72], atol=1e-12)
    np.testing.assert_allclose(ev["freq_hz"], [1 / (15 / FPS), 2 / (43 / FPS), 3 / (72 / FPS)], atol=1e-9)
    # quadros sem rosto antes da piscada não entram no denominador
    face = np.ones(100, bool); face[0:5] = False
    s2 = _sinal(_tres_piscadas(), face)
    ev2 = B.detectar_eventos(s2["ear"], s2["face"], s2["t"], s2["d"], limiar=0.2)
    assert abs(ev2["frequencia"].iloc[0] - 100 * 1 / 10) < 1e-12


def test_sem_rosto_interrompe_piscada():
    ear = np.full(60, 0.3); ear[20:28] = 0.1
    face = np.ones(60, bool); face[23] = False
    s = _sinal(ear, face)
    ev = B.detectar_eventos(s["ear"], s["face"], s["t"], s["d"], limiar=0.2)
    assert list(zip(ev["inicio"], ev["fim"])) == [(20, 22), (24, 27)]
    assert ev["interrompida"].all()
    # sem a falha de rosto é um evento só, não interrompido
    s3 = _sinal(ear)
    ev3 = B.detectar_eventos(s3["ear"], s3["face"], s3["t"], s3["d"], limiar=0.2)
    assert len(ev3) == 1 and not ev3["interrompida"].iloc[0]


def test_fechamento_longo_e_limiar():
    ear = np.full(300, 0.3); ear[50:100] = 0.1   # 50 quadros = 1,67 s > 1,5 s
    s = _sinal(ear)
    ev = B.detectar_eventos(s["ear"], s["face"], s["t"], s["d"], limiar=0.2)
    assert list(ev["tipo"]) == ["fechamento_longo"] and np.isnan(ev["frequencia"].iloc[0])
    ear[50:95] = 0.1; ear[95:100] = 0.3        # 45 quadros = 1,5 s -> ainda piscada (limite é "> 1,5 s")
    s = _sinal(ear)
    ev = B.detectar_eventos(s["ear"], s["face"], s["t"], s["d"], limiar=0.2)
    assert list(ev["tipo"]) == ["piscada"]
    lim, med = B.limiar_adaptativo(np.array([0.2, 0.3, 0.4, np.nan]), np.array([1, 1, 1, 0], bool),
                                   np.ones(4, bool), 0.75)
    assert abs(med - 0.3) < 1e-12 and abs(lim - 0.225) < 1e-12


def test_tempo_vfr():
    t = B.tempo_monotonico(np.array([0.0, 0.1, 0.05, 0.2, 5.0]), np.arange(5) / 10)
    assert np.all(np.diff(t) >= 0)
    d = B.duracoes_quadro(np.array([0.0, 0.1, 0.2, 3.0, 3.1]))
    assert abs(d[2] - 0.3) < 1e-12  # lacuna de 2,8 s truncada em 3x a mediana (0,1 s)


# ------------------------------------------------------------------ normalização
def test_zscore_referencia_alerta():
    A = np.arange(1, 10, dtype=float)[:, None].repeat(4, 1)  # 9 piscadas, valores 1..9
    mu, sd, resto, info = S.referencia_alerta(A, "ultimo")
    assert info["n_referencia"] == 3 and np.allclose(mu, 8.0) and np.allclose(sd, np.std([7, 8, 9]))
    assert list(resto) == list(range(6))
    mu, sd, resto, _ = S.referencia_alerta(A, "primeiro")
    assert np.allclose(mu, 2.0) and list(resto) == list(range(3, 9))
    Z = (A[resto] - mu) / sd
    assert np.allclose(Z[:, 0], (np.arange(4, 10) - 2.0) / np.std([1, 2, 3]))
    C = np.ones((9, 4)); C[:6] = 5.0                          # referência constante -> desvio de todas
    mu, sd, _, info = S.referencia_alerta(C, "ultimo")
    assert np.allclose(sd, np.std(C[:, 0])) and info["sd_zero"] == [0, 1, 2, 3]


def test_normalizacao_janelas():
    df = pd.DataFrame({"unidade_ref": ["a"] * 6 + ["b"] * 6, **{f: np.r_[np.arange(6.), np.arange(6.) * 2] for f in W.FEATURES}})
    ref = pd.Series([True] * 5 + [False] + [True] + [False] * 5)
    out, st = W.normalizar(df, ref, "unidade_ref", n_min=5)
    a = out[out.unidade_ref == "a"]
    mu, sd = np.mean([0, 1, 2, 3, 4]), np.std([0, 1, 2, 3, 4])
    assert np.allclose(a["perclos_c"], np.arange(6) - mu) and np.allclose(a["perclos_z"], (np.arange(6) - mu) / sd)
    b = out[out.unidade_ref == "b"]   # 1 janela de referência (< 5): _c definido, _z NaN
    assert np.allclose(b["perclos_c"], np.arange(6.) * 2 - 0.0) and b["perclos_z"].isna().all()


def _janelas_df(T, janela=10.0, passo=5.0, **cols):
    df = pd.DataFrame([(a, b, p) for a, b, p in W.gerar_janelas(T, janela, passo)], columns=["t_ini", "t_fim", "parcial"])
    for k, v in cols.items():
        df[k] = v
    return df


def test_referencia_uta_separada_da_avaliacao():
    """UTA: só janelas inteiras no ÚLTIMO terço do vídeo 0 são referência; as que cruzam o corte de 2/3 são
    fronteira; nenhuma janela avaliada (exemplo) do vídeo 0 toca a referência; vídeos 5/10 não têm referência."""
    v0 = _janelas_df(600.0, sujeito="S", rotulo_video="0", arquivo="S_0")
    v10 = _janelas_df(600.0, sujeito="S", rotulo_video="10", arquivo="S_10")
    w = W.marcar_uta(pd.concat([v0, v10], ignore_index=True), {"S": 600.0})
    ref = w[w["janela_ref"]]
    assert len(ref) > 0 and (ref["t_ini"] >= 400.0 - 1e-9).all() and (ref["rotulo_video"] == "0").all()
    ex0 = w[(w["uso"] == "exemplo") & (w["rotulo_video"] == "0")]
    assert (ex0["t_fim"] <= 400.0 + 1e-9).all() and len(ex0) > 0
    fr = w[w["uso"] == "fronteira"]
    assert list(zip(fr["t_ini"], fr["t_fim"])) == [(395.0, 405.0)]
    assert (w.loc[w["rotulo_video"] == "10", "uso"] == "exemplo").all()
    # nenhuma janela exemplo compartilha tempo com a referência do mesmo arquivo
    for _, e in ex0.iterrows():
        assert not ((ref["t_ini"] < e["t_fim"] - 1e-9) & (ref["t_fim"] > e["t_ini"] + 1e-9)).any()


def test_referencia_nitymed_primeiro_terco():
    w = _janelas_df(60.0, arquivo="V")
    w = W.marcar_nitymed(w, {"V": 60.0})           # corte = 20 s
    assert list(w.loc[w["janela_ref"], "t_ini"]) == [0.0, 5.0, 10.0]
    assert list(w.loc[w["uso"] == "fronteira", "t_ini"]) == [15.0]
    assert (w.loc[w["uso"] == "exemplo", "t_ini"] >= 20.0 - 1e-9).all()


# ------------------------------------------------------------------ sequências
def test_unroll_igual_ao_original():
    rng = np.random.default_rng(42)
    for n in range(1, 121):
        x = rng.normal(size=(n, 4))
        assert np.array_equal(S.unroll_in_time(x), S.unroll_in_time_ref(x, 30, 2)), n
        assert len(S.unroll_in_time(x)) == S.n_sequencias(n)
    for w, st in ((10, 1), (30, 3), (5, 5)):
        x = rng.normal(size=(47, 4))
        assert np.array_equal(S.unroll_in_time(x, w, st), S.unroll_in_time_ref(x, w, st))
    assert S.unroll_in_time(np.zeros((0, 4))).shape == (0, 30, 4)
    x = np.arange(1, 11, dtype=float)[:, None].repeat(4, 1)   # 10 piscadas -> 20 zeros à esquerda
    u = S.unroll_in_time(x)
    assert u.shape == (1, 30, 4) and np.all(u[0, :20] == 0) and np.array_equal(u[0, 20:], x)


# ------------------------------------------------------------------ janelas
def test_numero_de_janelas():
    # padrão da rodada final: 10 s, passo 5 s
    assert W.JANELA_S == 10.0 and W.PASSO_S == 5.0
    assert len(W.gerar_janelas(10.0)) == 1 and len(W.gerar_janelas(14.99)) == 1 and len(W.gerar_janelas(15.0)) == 2
    assert len(W.gerar_janelas(180.0)) == 35 and len(W.gerar_janelas(600.0)) == 119
    assert W.gerar_janelas(9.0) == [] and W.gerar_janelas(9.99) == []
    assert [(a, b) for a, b, _ in W.gerar_janelas(30.0)] == [(0.0, 10.0), (5.0, 15.0), (10.0, 20.0), (15.0, 25.0), (20.0, 30.0)]
    # 30 s (rodada preliminar) e 5/15 s (sensibilidade) pelos parâmetros
    assert len(W.gerar_janelas(180.0, 30.0, 5.0)) == 31 and W.gerar_janelas(29.0, 30.0, 5.0) == [(0.0, 29.0, True)]
    assert len(W.gerar_janelas(60.0, 5.0, 5.0)) == 12 and len(W.gerar_janelas(60.0, 15.0, 5.0)) == 10
    assert W.gerar_janelas(12.0, 15.0, 5.0) == [(0.0, 12.0, True)]
    s = _sinal(np.full(int(60 * FPS), 0.3))
    lista = [(a, b, p, np.nan) for a, b, p in W.gerar_janelas(60.0)]
    w = W.janelas_arquivo(s, pd.DataFrame(columns=B.COLUNAS_EVENTO), 0.2, lista)
    assert len(w) == 11 and (w["n_quadros"] == 300).all() and (w["perclos"] == 0).all()


def test_janelas_ancoradas_em_trechos_safe_drive():
    """DMD s1/s2/s3: janelas ancoradas no início de cada trecho safe_drive, inteiramente dentro dele."""
    trechos = [(0.5, 16.0), (20.0, 29.9), (40.0, 66.0)]
    js = W.janelas_em_trechos(trechos, 10.0, 5.0)
    assert [(round(a, 6), round(b, 6), j) for a, b, j in js] == [(0.5, 10.5, 0), (5.5, 15.5, 0), (40.0, 50.0, 2),
                                                                  (45.0, 55.0, 2), (50.0, 60.0, 2), (55.0, 65.0, 2)]
    for a, b, j in js:
        assert trechos[j][0] - 1e-9 <= a and b <= trechos[j][1] + 1e-9
    # com o sinal: índices de quadro das janelas ficam dentro do trecho (quadros inclusivos)
    fps = 29.76
    s = _sinal(np.full(int(70 * fps), 0.3), fps=fps)
    tq = [(15, 480), (1190, 1964)]
    tr = W.trechos_em_segundos(s, tq)
    lista = [(a, b, False, j) for a, b, j in W.janelas_em_trechos(tr, 10.0, 5.0)]
    w = W.janelas_arquivo(s, pd.DataFrame(columns=B.COLUNAS_EVENTO), 0.2, lista)
    assert len(w) > 0
    for _, r in w.iterrows():
        a, b = tq[int(r["trecho_safe"])]
        assert r["i0"] >= a and r["i1"] - 1 <= b
    assert B.fundir_intervalos([(10, 20), (21, 30), (40, 50), (45, 60)]) == [(10, 30), (40, 60)]


def _dmd_df(linhas):
    return pd.DataFrame(linhas, columns=["sujeito", "sessao", "arquivo", "t_ini", "t_fim", "rotulo_anotado"])


def test_rotulos_v2_dmd():
    """Regra final: alerta = safe_drive da s2; referência = safe_drive da s3 (>= 3 janelas); sonolento = s5 com
    sinal anotado; s5 sem sinal fora do principal; s1 = alerta de sensibilidade; sem s3 suficiente -> referência =
    último terço (por sessão) das janelas da s2 e as que a tocam são fronteira; sujeito sem s5 -> y NaN."""
    L_ = []
    for k in range(4):                                    # A: s3 com 4 janelas -> referência
        L_.append(("A", "s3", "A_s3", 5.0 * k, 5.0 * k + 10, None))
    for k in range(6):
        L_.append(("A", "s2", "A_s2", 100.0 + 5 * k, 110.0 + 5 * k, None))
    for k in range(3):
        L_.append(("A", "s1", "A_s1", 5.0 * k, 5.0 * k + 10, None))
    L_ += [("A", "s5", "A_s5", 0.0, 10.0, "sonolento_atuado"), ("A", "s5", "A_s5", 5.0, 15.0, "s5_sem_sinal")]
    for k in range(2):                                    # B: s3 com 2 janelas -> fallback na s2
        L_.append(("B", "s3", "B_s3", 5.0 * k, 5.0 * k + 10, None))
    for k in range(6):                                    # 6 janelas seguidas (sobrepostas) na s2
        L_.append(("B", "s2", "B_s2", 5.0 * k, 5.0 * k + 10, None))
    L_ += [("B", "s5", "B_s5", 0.0, 10.0, "sonolento_atuado")]
    L_ += [("C", "s2", "C_s2", 0.0, 10.0, None), ("C", "s3", "C_s3", 0.0, 10.0, None),   # C: sem s5
           ("C", "s3", "C_s3", 20.0, 30.0, None), ("C", "s3", "C_s3", 40.0, 50.0, None)]
    w, info = W.marcar_dmd(_dmd_df(L_))
    A = w[w["sujeito"] == "A"]
    assert (A.loc[A["sessao"] == "s3", "uso"] == "referencia").all() and (A.loc[A["sessao"] == "s2", "uso"] == "alerta").all()
    assert (A.loc[A["sessao"] == "s1", "uso"] == "alerta_s1").all()
    assert list(A.loc[A["sessao"] == "s5", "uso"]) == ["sonolento", "s5_sem_sinal"]
    assert list(A.loc[A["sessao"] == "s5", "y"].fillna(-1)) == [1.0, -1.0]
    assert list(A.loc[A["sessao"] == "s5", "y_sens_s5_sem_sinal"]) == [1.0, 0.0]
    assert (A.loc[A["sessao"] == "s2", "y"] == 0).all() and A.loc[A["sessao"] == "s2", "y_sens_s1"].isna().all()
    assert (A.loc[A["sessao"] == "s1", "y_sens_s1"] == 0).all() and A.loc[A["sessao"] == "s1", "y"].isna().all()
    assert A.loc[A["uso"] == "referencia", ["y", "y_sens_s1", "y_sens_s5_sem_sinal"]].isna().all().all()
    assert info["A"]["tipo_ref"] == "s3_safe_drive" and not info["A"]["ref_fallback"]
    Bb = w[w["sujeito"] == "B"]
    assert info["B"]["ref_fallback"] and (Bb.loc[Bb["sessao"] == "s3", "uso"] == "s3_nao_usada").all()
    s2 = Bb[Bb["sessao"] == "s2"].sort_values("t_ini")
    # 6 janelas -> as 2 últimas (t_ini 20 e 25) são referência; [10, 20) termina no corte (alerta); [15, 25) o cruza
    assert list(s2["uso"]) == ["alerta", "alerta", "alerta", "fronteira", "referencia", "referencia"]
    ref = s2[s2["uso"] == "referencia"]
    for _, e in s2[s2["uso"] == "alerta"].iterrows():      # exemplos não tocam a referência
        assert (e["t_fim"] <= ref["t_ini"].min() + 1e-9)
    C = w[w["sujeito"] == "C"]
    assert not C["tem_s5"].any() and C["y"].isna().all() and (C.loc[C["sessao"] == "s2", "uso"] == "alerta").all()


def test_caracteristicas_janela():
    n = int(40 * FPS)
    ear = np.full(n, 0.3)
    ear[300:330] = 0.1        # fechamento de 1,0 s (30 quadros) começando em t = 10 s
    ear[600:606] = 0.1        # piscada de 6 quadros em t = 20 s
    mar = np.zeros(n); mar[450:495] = 0.8   # MAR > 0,5 por 1,5 s (bocejo)
    face = np.ones(n, bool); face[700:730] = False
    s = _sinal(ear, face, mar=mar)
    ev = B.detectar_eventos(s["ear"], s["face"], s["t"], s["d"], 0.2)
    w = W.janelas_arquivo(s, ev, 0.2, [(a, b, p, np.nan) for a, b, p in W.gerar_janelas(40.0, 30.0, 5.0)])
    j0 = w.iloc[0]                      # [0, 30)
    assert j0["n_quadros"] == 900 and j0["n_rosto"] == 870 and abs(j0["taxa_rosto"] - 870 / 900) < 1e-12
    assert abs(j0["perclos"] - 36 / 870) < 1e-12
    assert j0["n_piscadas"] == 2 and j0["n_fechamentos_05s"] == 1
    assert abs(j0["dur_max_fechamento_s"] - 1.0) < 1e-9
    assert abs(j0["frac_bocejo"] - 45 / 870) < 1e-12
    assert abs(j0["taxa_piscadas_min"] - 2 / (870 / FPS / 60)) < 1e-9
    # janela de 10 s [10, 20): contém o fechamento de 1 s (t = 10-11 s) e o bocejo (t = 15-16,5 s)
    w10 = W.janelas_arquivo(s, ev, 0.2, [(a, b, p, np.nan) for a, b, p in W.gerar_janelas(40.0)])
    j = w10[w10["t_ini"] == 10.0].iloc[0]
    assert j["n_quadros"] == 300 and j["n_fechamentos_05s"] == 1 and abs(j["dur_max_fechamento_s"] - 1.0) < 1e-9
    assert abs(j["frac_bocejo"] - 45 / 300) < 1e-12 and j["n_piscadas"] == 1   # o fechamento de 1 s (< 1,5 s) é piscada


# ------------------------------------------------------------------ DMD
def test_rotulos_dmd_s5_e_casamento():
    fps = 29.76
    t = np.arange(int(120 * fps)) / fps
    jan = pd.DataFrame([(a, b) for a, b, _ in W.gerar_janelas(t[-1] + 1 / fps)], columns=["t_ini", "t_fim"])
    f0 = int(round(70 * fps))
    acoes = {"eyes_state/close": [(f0, f0 + 19)]}           # 20 quadros = 0,67 s em t ~ 70 s
    r = L.rotular_janelas(jan, t, acoes, fps)
    dentro = (jan["t_ini"] <= 70) & (jan["t_fim"] >= 70.8)
    assert dentro.sum() == 2 and (r.loc[dentro, "rotulo_anotado"] == "sonolento_atuado").all()
    assert r.loc[dentro, "tem_fechamento_anotado"].all() and not r["tem_bocejo_anotado"].any()
    assert (r.loc[~dentro, "rotulo_anotado"] == "s5_sem_sinal").all()   # [65, 75) e [70, 80) contêm os 0,67 s
    r2 = L.rotular_janelas(jan, t, {"eyes_state/close": [(f0, f0 + 9)]}, fps)   # 0,34 s: não conta
    assert (r2["rotulo_anotado"] == "s5_sem_sinal").all()
    # bocejo de 1 s que entra só 0,4 s na janela [60, 70): não conta; na [65, 75) conta
    b0 = int(round(69.6 * fps))
    r3 = L.rotular_janelas(jan, t, {"yawning/Yawning without hand": [(b0, b0 + 29)]}, fps)
    assert r3.loc[jan["t_ini"] == 60.0, "rotulo_anotado"].iloc[0] == "s5_sem_sinal"
    assert r3.loc[jan["t_ini"] == 65.0, "tem_bocejo_anotado"].iloc[0]
    assert L.casar_intervalos([(0, 5), (6, 10)], [(2, 8)]) == [(0, 0, 4)]
    p, rr, f = L.prf(10, 8, 6)
    assert abs(p - 0.6) < 1e-12 and abs(rr - 0.75) < 1e-12


def test_partes_uta_concatenadas_com_contagem_continua():
    """Vídeo dividido (_p1/_p2): as piscadas das partes viram uma série só, com Frequency contínua."""
    p1 = _sinal(_tres_piscadas())                 # 100 quadros, 3 piscadas
    ear2 = np.full(50, 0.3); ear2[20:23] = 0.1
    p2 = _sinal(ear2)                             # 50 quadros, 1 piscada (20..22)
    e1 = B.detectar_eventos(p1["ear"], p1["face"], p1["t"], p1["d"], 0.2)
    e2 = B.detectar_eventos(p2["ear"], p2["face"], p2["t"], p2["d"], 0.2)
    c = S.concatenar_partes([e1, e2], [p1, p2])
    assert len(c) == 4 and list(c["fim"]) == [14, 42, 71, 122]
    np.testing.assert_allclose(c["frequencia"], [100 / 15, 200 / 43, 300 / 72, 100 * 4 / (100 + 23)], atol=1e-12)
    np.testing.assert_allclose(c["freq_hz"].iloc[-1], 4 / ((100 + 23) / FPS), atol=1e-9)
    assert list(c["amplitude"]) == list(e1["amplitude"]) + list(e2["amplitude"])   # locais: não mudam
    assert S.concatenar_partes([e1], [p1]).equals(e1.reset_index(drop=True))
    # janelas por parte: nenhuma atravessa o corte (cada parte é um arquivo)
    for sg in (p1, p2):
        T = B.duracao_total(sg)
        for a, b, _ in W.gerar_janelas(T, 1.0, 0.5):
            assert b <= T + 1e-9


# ------------------------------------------------------------------ engenharia reversa das oficiais
def test_recuperacao_frequencia_oficial():
    """Sujeito sintético no formato do código original: F = round(100*T/NF, 4), com extrações agrupadas
    (2 piscadas gravadas juntas, mesmo T e NF) e uma piscada descartada (conta em T, não é gravada)."""
    rng = np.random.default_rng(42)
    series, verdade_nf, verdade_T = [], [], []
    for v in range(3):
        nf_eventos = np.cumsum(rng.integers(40, 160, size=90)) + 30
        linhas_nf, linhas_T, T = [], [], 0
        i = 0
        while i < len(nf_eventos):
            if i in (20, 50):                      # extração agrupada: 2 piscadas no mesmo instante
                T += 2
                linhas_nf += [nf_eventos[i + 1]] * 2; linhas_T += [T] * 2
                i += 2
                continue
            T += 1
            if v == 1 and i == 35:                 # piscada descartada (velocidade <= 0 no original)
                i += 1
                continue
            linhas_nf.append(nf_eventos[i]); linhas_T.append(T)
            i += 1
        nf = np.array(linhas_nf); Tt = np.array(linhas_T)
        series.append(np.round(100.0 * Tt / nf, 4)); verdade_nf.append(nf); verdade_T.append(Tt)
    base = series[0][-30:]                         # "último terço" do vídeo alerta
    mu, sd = base.mean(), base.std()
    zs = [np.c_[(F - mu) / sd, rng.normal(size=(len(F), 3))] for F in series]
    rec = O.recuperar_frequencia(zs)
    assert abs(rec["sigma_F"] - sd) / sd < 1e-6
    for v in range(3):
        nf_rec = np.round(rec["videos"][v]["NF"]).astype(int)
        # coincidências de arredondamento entre piscadas distintas (F igual por acaso) são indistinguíveis
        # de extrações agrupadas quando a tolerância é grande; exigimos >= 97% exatos e os agrupados certos
        assert np.mean(nf_rec == verdade_nf[v]) >= 0.97, (v, np.mean(nf_rec == verdade_nf[v]))
        for g in np.flatnonzero(np.diff(verdade_T[v]) == 0):
            assert nf_rec[g] == nf_rec[g + 1] == verdade_nf[v][g]
    assert rec["videos"][1]["quedas"] >= 1


def test_escore_coerente_e_ordem():
    """Instantes oficiais = vales + 15 quadros (com deriva lenta, como o dlib perdendo quadros) -> escore ~1 e
    defasagens ~15..; vales de outro vídeo -> escore baixo. Atribuição ordenada respeita a ordem."""
    rng = np.random.default_rng(42)
    p = np.sort(rng.choice(np.arange(100, 18000), size=300, replace=False))
    w = p[::2] + 15 - (np.arange(len(p[::2])) // 25) * 40      # defasagem cai 40 quadros por bloco
    fr, lags = O.escore_coerente(w, p)
    assert fr > 0.99 and abs(lags[0] - 15) <= O.TOL_Q and all(b - a == -40 for a, b in zip(lags[:-1], lags[1:]))
    outro = np.sort(rng.choice(np.arange(100, 18000), size=300, replace=False))
    assert O.escore_coerente(w, outro)[0] < 0.5
    real, acaso, _ = O.escore_video_coerente(w, p, 18000)
    assert real - acaso > 0.5
    Sm = np.array([[0.1, 0.9, 0.0, 0.0], [0.0, 0.2, 0.1, 0.8]])
    assert O.atribuicao_ordenada(Sm) == [1, 3]
    assert O.atribuicao_ordenada(np.array([[0.0, 1.0], [1.0, 0.0]])) in ([0, 1],)   # a ordem impede o "cruzamento"


def _rodar_todos():
    fs = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for k, f in fs:
        f()
        print(f"ok  {k}")
    print(f"{len(fs)} testes passaram")


if __name__ == "__main__":
    _rodar_todos()
