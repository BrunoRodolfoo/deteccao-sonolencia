"""Engenharia reversa das características OFICIAIS da UTA-RLDD e mapeamento sujeito-oficial <-> pasta.

Usado por src/features/sequences.py (não precisa ser rodado sozinho).

Problema: BlinksTest_30_Fold<k>.npy traz sequências de 30 piscadas x 4 características já normalizadas
(z-score por sujeito), sem identificação do sujeito nem do vídeo; a ordem dos sujeitos segue os.listdir()
na máquina dos autores. Para comparar com as nossas características é preciso saber qual pasta
(01..12) corresponde a cada sujeito do arquivo — e isso tem de ser INFERIDO.

Como (tudo verificável nos dados):
  1. As fronteiras de vídeo vêm de uta_common.reconstruct_ids (validadas contra os start_indices oficiais).
     Sequências consecutivas de um vídeo se sobrepõem em 28 piscadas (passo 2), então a SÉRIE de piscadas
     do vídeo é reconstruída exatamente (seq[0] + os 2 últimos de cada seq seguinte; o preenchimento com
     zeros à esquerda é removido nos vídeos de uma só sequência).
  2. O código original (blink_video.py) grava cada característica com fmt='%.4f'; logo, dentro de um sujeito,
     os valores normalizados z = (x - mu)/sigma ficam num RETICULADO de passo 1e-4/sigma (Duration: 1/sigma,
     pois a duração é inteira). O passo é estimado das diferenças entre valores distintos (erro ~1e-9).
  3. Frequency = 100 * T / NF (T = piscadas contadas até ali, NF = quadros com rosto até ali). Com sigma
     conhecido, F = (k_i + K) * 1e-4 com k_i conhecido e um único inteiro K desconhecido por sujeito; K é o
     valor que torna NF = 100*T/F inteiro (dentro do erro de arredondamento) para as primeiras piscadas de
     cada vídeo (T = 1, 2, ...; piscadas extraídas juntas têm F idêntico e o mesmo T). Desempate: coerência
     da série inteira (NF inteiro e crescente, poucas piscadas descartadas).
  4. Com NF (instante de gravação de cada piscada oficial, em quadros com rosto do dlib), procura-se no NOSSO
     EAR de cada vídeo candidato quedas (vales do EAR relativo à mediana móvel de 3 s): no vídeo certo, os
     instantes oficiais caem a uma defasagem quase constante dos vales (a defasagem deriva devagar porque o
     dlib perde quadros). Escore por vídeo = fração de piscadas oficiais com vale a +-3 quadros da defasagem
     de cada bloco de 25 piscadas, menos o mesmo escore com os instantes deslocados circularmente (acaso).
     Escore do sujeito = soma dos 3 vídeos; atribuição = algoritmo húngaro (scipy.linear_sum_assignment).

RODADA FINAL (5 folds, 27/09/2026):
  5. Vídeos divididos (F3_32_10, F5_49_10): os vales das partes são concatenados em ordem, com o deslocamento do nº de
     quadros da parte anterior (a série oficial de F3_32_10 passa do fim da parte 1: a contagem do original é contínua).
  6. Escore PRINCIPAL = "coerente": em vez de escolher a defasagem de cada bloco de 25 piscadas de forma independente
     (o que deixa blocos de pastas erradas "acharem" coincidências ao acaso), a trajetória de defasagens é escolhida por
     programação dinâmica com a restrição física de que a defasagem só pode cair devagar (o dlib perde quadros; queda
     <= 300 quadros por bloco) e quase não subir (<= 10 quadros). Mesmo desconto do acaso (instantes deslocados 1/3 e
     2/3 do vídeo, reordenados). O escore por blocos independentes (rodada preliminar) é mantido como sensibilidade.
  7. Checagem pela DURAÇÃO (não usa o EAR): o NF da última piscada da série oficial não pode passar do nº de quadros do
     vídeo candidato (folga de 0,05%); só vale quando a série foi bem recuperada (poucas piscadas "descartadas").
  8. Classes de evidência de cada par adotado: "confirmado" (escore coerente >= LIMIAR_FORTE, máximo da linha e da
     coluna), "provavel" (sem coincidência forte, mas determinado pela ordem de listagem — pastas em ordem crescente,
     como nos pares confirmados do fold — ou pela única pasta restante compatível com a duração) e "indeterminado".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import blinks as B  # noqa: E402

ROOT = B.ROOT
DIR_OFICIAL = ROOT / "data" / "uta_rldd_blinks"
PASSO_ARRED = 1e-4        # fmt='%.4f' do código original
BLOCO = 25                # piscadas oficiais por bloco de defasagem
TOL_Q = 3                 # tolerância (quadros) da coincidência
LAG_MIN, LAG_MAX = -1000, 100   # até ~33 s de quadros perdidos pelo dlib a 30 fps (mapeamento idêntico com -400, -600 e -3000)
# escore coerente (principal na rodada final)
LAGC_MIN, LAGC_MAX = -3000, 100 # defasagem NF_oficial - vale nosso (quadros)
SUBIDA_MAX = 10                 # a defasagem pode subir no máximo isso entre blocos consecutivos
QUEDA_MAX = 300                 # e cair no máximo isso (quadros perdidos pelo dlib em ~25 piscadas)
LIMIAR_FORTE = 0.30             # escore coerente mínimo para "confirmado" (ver distribuição dos pares não adotados)
TOL_NF = 5e-4                   # folga relativa da checagem NF_final <= nº de quadros


# --------------------------------------------------------------------------------------
# 1. Séries oficiais por vídeo
# --------------------------------------------------------------------------------------
def carregar_teste_oficial(fold):
    """(X, y 0/1/2, video_id, subject_id) do arquivo de TESTE oficial do fold, ou None se não existir."""
    p = DIR_OFICIAL / f"BlinksTest_30_Fold{fold}.npy"
    if not p.exists():
        return None
    import uta_common as U  # noqa: E402  (src/experiments)
    _, _, xte, yte = U.load_fold(fold)
    vid, sid = U.reconstruct_ids(yte, fold=fold)
    X = np.load(p, allow_pickle=True).astype(np.float64)  # precisão total (load_fold converte p/ float32)
    return X, yte, vid, sid


def serie_do_video(S):
    """Série de piscadas (n x 4) a partir das sequências (N x 30 x 4) de um vídeo (passo 2)."""
    z = np.concatenate([S[0]] + [S[s][28:] for s in range(1, len(S))])
    if len(S) == 1:  # vídeo com <= 30 piscadas: zeros à esquerda são preenchimento
        nz = np.flatnonzero(np.any(z != 0, axis=1))
        z = z[nz[0]:] if len(nz) else z[:0]
    return z


def verificar_sobreposicao(S):
    return all(np.allclose(S[s][2:], S[s + 1][:28]) for s in range(len(S) - 1))


# --------------------------------------------------------------------------------------
# 2-3. Reticulado e recuperação da Frequency
# --------------------------------------------------------------------------------------
def passo_reticulado(vals):
    """Menor passo delta tal que todos os valores distintos são múltiplos inteiros de delta (a partir do mínimo).

    Devolve (delta, erro máximo em unidades de delta)."""
    u = np.unique(vals)
    d = np.diff(u)
    d = d[d > 1e-9]
    best = (np.nan, np.inf)
    for cand in np.sort(d)[:8]:
        for div in range(1, 8):
            dl = cand / div
            r = (u - u[0]) / dl
            err = float(np.abs(r - np.round(r)).max())
            if err < best[1] - 1e-12:
                best = (dl, err)
    return best


def _grupos(k):
    """Índice de grupo: piscadas consecutivas com o mesmo valor (extraídas juntas) formam um grupo."""
    g = np.zeros(len(k), int)
    for i in range(1, len(k)):
        g[i] = g[i - 1] + (1 if k[i] != k[i - 1] else 0)
    return g


def _rastrear_T(F, k, max_quedas=40):
    """Dado F por piscada, acha T (contagem acumulada) e NF = 100 T / F de cada piscada.

    Piscadas consecutivas com F idêntico podem ser (a) extraídas juntas pelo código original (mesmo T e NF:
    "agrupado") ou (b) piscadas distintas cujo F arredondado coincidiu (T consecutivos: "separado"). Para cada
    corrida de F igual testam-se as duas hipóteses e fica a de menor resíduo de integralidade; q = piscadas
    descartadas pelo original (velocidade <= 0 não é gravada, mas conta em T). Devolve (NF, T, residuo, quedas)."""
    n = len(F)
    NF = np.zeros(n); T = np.zeros(n, int); res = np.zeros(n)
    T_prev, NF_prev, quedas = 0, 0.0, 0

    def resid(nf, f):
        tol = nf * 0.5 * PASSO_ARRED / f + 1e-6
        return abs(nf - round(nf)) / tol

    i = 0
    while i < n:
        j = i
        while j + 1 < n and k[j + 1] == k[i]:
            j += 1
        g = j - i + 1
        best = None
        for q in range(max_quedas):
            hip = [[T_prev + g + q] * g]                                   # agrupado
            if g > 1:
                hip.append([T_prev + q + m for m in range(1, g + 1)])      # separado
            for ts in hip:
                nfs = [100.0 * t_ / F[i] for t_ in ts]
                if nfs[0] <= NF_prev:
                    continue
                r = max(resid(nf, F[i]) for nf in nfs)
                if best is None or r < best[0]:
                    best = (r, ts, nfs, q)
            if best is not None and best[0] <= 1:
                break
        if best is None:
            ts = [T_prev + g] * g
            best = (np.inf, ts, [100.0 * t_ / F[i] for t_ in ts], 0)
        r, ts, nfs, q = best
        NF[i:j + 1], T[i:j + 1], res[i:j + 1] = nfs, ts, r
        T_prev, NF_prev, quedas = ts[-1], nfs[-1], quedas + q
        i = j + 1
    return NF, T, res, quedas


def recuperar_frequencia(series):
    """series: lista (um item por vídeo do sujeito) de arrays n x 4 (z oficiais).

    Devolve dict com K, sigma_F, escore e 2º escore do teste de integralidade e, por vídeo, F, NF, T, quedas."""
    zF = np.concatenate([z[:, 0] for z in series])
    dl, err = passo_reticulado(zF)
    zmin = zF.min()
    ks = [np.round((z[:, 0] - zmin) / dl).astype(np.int64) for z in series]
    Ks = np.arange(1, 80000)
    escore = np.zeros(len(Ks))
    for k in ks:
        grp = _grupos(k)
        for gi in range(6):
            idx = np.flatnonzero(grp == gi)
            if not len(idx):
                continue
            prev = int(np.sum(grp < gi))
            g = len(idx)
            F = (k[idx[0]] + Ks) * PASSO_ARRED
            ok = np.zeros(len(Ks), bool)
            for q in range(3):
                for t_ in {prev + g + q, prev + 1 + q}:   # agrupado ou separado (ver _rastrear_T)
                    nf = 100.0 * t_ / F
                    tol = nf * 0.5 * PASSO_ARRED / F + 1e-6
                    ok |= (np.abs(nf - np.round(nf)) <= tol) & (nf >= 16)
            escore += ok
    ordem = np.argsort(-escore)

    def coerencia(K):
        tot, okc, quedas = 0, 0, 0
        for k in ks:
            F = (k + K) * PASSO_ARRED
            if np.any(F <= 0):
                return -1.0
            _, _, res, q = _rastrear_T(F, k, max_quedas=6)
            tot += len(res)
            okc += int(np.sum(res <= 1))
            quedas += q
        return okc / max(tot, 1) - 0.001 * quedas

    cands = [int(Ks[i]) for i in ordem[:10] if escore[i] >= escore[ordem[0]] - 1]
    coer = {K: coerencia(K) for K in cands}
    K = max(cands, key=lambda c: coer[c])
    out = dict(K=K, sigma_F=PASSO_ARRED / dl, erro_reticulado=err, escore_K=float(escore[ordem[0]]),
               escore_K_2o=float(escore[ordem[1]]), coerencia=coer[K],
               coerencia_2o=max([v for c, v in coer.items() if c != K], default=np.nan), videos=[])
    for k in ks:
        F = (k + K) * PASSO_ARRED
        NF, T, res, quedas = _rastrear_T(F, k)
        out["videos"].append(dict(F=F, NF=NF, T=T, res=res, quedas=quedas, frac_inteiro=float(np.mean(res <= 1))))
    return out


# --------------------------------------------------------------------------------------
# 4. Coincidência com o nosso EAR
# --------------------------------------------------------------------------------------
def vales_ear(sinal):
    """Quadros de vale do EAR relativo à mediana móvel de 3 s (ver blinks.vales_ear)."""
    return B.vales_ear(sinal)[0]


def escore_blocos(w, p, bloco=BLOCO, tol=TOL_Q, lag_min=None):
    """Fração média de instantes oficiais w com evento p a +-tol da defasagem de máxima coincidência do bloco.

    Devolve (fração, lista de defasagens por bloco)."""
    hits, lags = [], []
    if len(p) == 0 or len(w) == 0:
        return 0.0, []
    lmin = LAG_MIN if lag_min is None else lag_min
    bins = np.arange(lmin, LAG_MAX + 1)
    for a in range(0, len(w), bloco):
        ww = w[a:a + bloco]
        dif = (ww[:, None] - p[None, :]).ravel()
        dif = dif[(dif >= lmin) & (dif <= LAG_MAX)]
        if len(dif) == 0:
            hits.append(0.0)
            lags.append(np.nan)
            continue
        h, e = np.histogram(dif, bins=bins)
        hs = np.convolve(h, np.ones(2 * tol + 1), mode="same")
        lag = float(e[int(np.argmax(hs))])
        hit = float(np.mean([np.any(np.abs((wi - p) - lag) <= tol) for wi in ww]))
        hits.append(hit)
        lags.append(lag)
    return float(np.mean(hits)), lags


def escore_video(w, p, n_quadros, lag_min=None):
    """Escore corrigido pelo acaso: coincidência real - média da coincidência com w deslocado 1/3 e 2/3 do vídeo."""
    real, lags = escore_blocos(w, p, lag_min=lag_min)
    acaso = float(np.mean([escore_blocos((w + int(n_quadros * f)) % n_quadros, p, lag_min=lag_min)[0]
                           for f in (1 / 3, 2 / 3)]))
    return real, acaso, lags


def _acertos_por_lag(ww, p, lmin=LAGC_MIN, lmax=LAGC_MAX, tol=TOL_Q):
    """Para cada defasagem L em [lmin, lmax]: nº de instantes de ww com algum vale de p em |w - p - L| <= tol."""
    n_l = lmax - lmin + 1
    tot = np.zeros(n_l)
    for wi in ww:
        a = np.searchsorted(p, wi - lmax - tol, "left")
        b = np.searchsorted(p, wi - lmin + tol, "right")
        if b <= a:
            continue
        d = (wi - p[a:b]) - lmin
        lo = np.clip(d - tol, 0, n_l)
        hi = np.clip(d + tol + 1, 0, n_l)
        marca = np.zeros(n_l + 1, np.int32)
        np.add.at(marca, lo, 1)
        np.add.at(marca, hi, -1)
        tot += np.cumsum(marca[:-1]) > 0
    return tot


def escore_coerente(w, p, bloco=BLOCO):
    """Fração de instantes oficiais w com vale na trajetória de defasagens (uma por bloco) de máxima coincidência,
    sujeita a: L_b - L_(b-1) em [-QUEDA_MAX, +SUBIDA_MAX]. Devolve (fração, defasagens por bloco)."""
    w = np.asarray(w, dtype=np.int64)
    p = np.sort(np.asarray(p, dtype=np.int64))
    if len(w) == 0 or len(p) == 0:
        return 0.0, []
    blocos = [w[a:a + bloco] for a in range(0, len(w), bloco)]
    larg = SUBIDA_MAX + QUEDA_MAX + 1
    prev, trilha = None, []
    for ww in blocos:
        h = _acertos_por_lag(ww, p)
        if prev is None:
            cur = h
            trilha.append(np.full(len(h), -1))
        else:
            pad = np.concatenate([np.full(SUBIDA_MAX, -np.inf), prev, np.full(QUEDA_MAX, -np.inf)])
            jan = np.lib.stride_tricks.sliding_window_view(pad, larg)
            arg = jan.argmax(axis=1)
            cur = h + jan[np.arange(len(h)), arg]
            trilha.append(np.arange(len(h)) + arg - SUBIDA_MAX)
        prev = cur
    k = int(np.argmax(prev))
    lags = [k]
    for tb in trilha[:0:-1]:
        k = int(tb[k])
        lags.append(k)
    lags = [x + LAGC_MIN for x in lags[::-1]]
    return float(prev.max()) / len(w), lags


def escore_video_coerente(w, p, n_quadros):
    """Escore coerente corrigido pelo acaso (instantes deslocados circularmente 1/3 e 2/3 do vídeo, reordenados)."""
    real, lags = escore_coerente(w, p)
    w = np.asarray(w, dtype=np.int64)
    acaso = float(np.mean([escore_coerente(np.sort((w + int(n_quadros * f)) % n_quadros), p)[0]
                           for f in (1 / 3, 2 / 3)]))
    return real, acaso, lags


def videos_logicos(fold):
    """{vídeo lógico -> (lista de arquivos em ordem, nº total de quadros)} do fold, a partir do blinks_index final."""
    bidx = pd.read_csv(B.FEAT / "uta" / "blinks_index.csv", dtype={"rotulo": str, "parte": str})
    bidx = bidx[bidx["fold"] == fold].copy()
    bidx["parte"] = bidx["parte"].fillna("")
    out = {}
    for v, g in bidx.groupby("video"):
        g = g.sort_values("parte")
        out[v] = (list(g["id"]), int(g["n_quadros"].sum()))
    return out, sorted(bidx["sujeito"].unique())


def vales_logico(arquivos, cache):
    """Vales do EAR do vídeo lógico: partes concatenadas em ordem, com deslocamento do nº de quadros anterior."""
    chave = tuple(arquivos)
    if chave not in cache:
        off, pk = 0, []
        for a in arquivos:
            sig = B.carregar_sinal(ROOT / "data" / "signals" / "uta" / f"{a}.npz")
            pk.append(vales_ear(sig) + off)
            off += sig["n"]
        cache[chave] = np.concatenate(pk) if pk else np.zeros(0, int)
    return cache[chave]


def confiavel(rec_video):
    """Série oficial bem recuperada: NF inteiro em todas as piscadas e poucas piscadas 'descartadas'."""
    n = len(rec_video["NF"])
    return bool(n > 0 and rec_video["frac_inteiro"] >= 0.999 and rec_video["quedas"] <= max(5, 0.1 * n))


def atribuicao_ordenada(S):
    """Melhor atribuição que preserva a ordem (sujeito oficial i < j => pasta(i) < pasta(j)), por programação
    dinâmica (listagem em ordem crescente com pastas faltando). Devolve lista de colunas ou None se n_linhas > n_col."""
    nr, nc = S.shape
    if nr > nc:
        return None
    dp = np.full((nr + 1, nc + 1), -np.inf)
    dp[0, :] = 0.0
    esc = np.zeros((nr + 1, nc + 1), bool)
    for i in range(1, nr + 1):
        for j in range(1, nc + 1):
            pula = dp[i, j - 1]
            usa = dp[i - 1, j - 1] + S[i - 1, j - 1]
            if usa >= pula:
                dp[i, j], esc[i, j] = usa, True
            else:
                dp[i, j] = pula
    cols, i, j = [], nr, nc
    while i > 0:
        if esc[i, j]:
            cols.append(j - 1)
            i -= 1
        j -= 1
    return cols[::-1]


def mapear_fold_final(fold, log=None):
    """Mapeamento sujeito-oficial -> pasta do fold com o escore coerente, a checagem de duração e as classes de
    evidência. Devolve dict com: mapa (DataFrame), escores (DataFrame longo), recup (lista), sem_par (lista),
    S (matriz coerente), S_blocos (matriz da rodada preliminar), pastas."""
    dados = carregar_teste_oficial(fold)
    if dados is None:
        return None
    X, y, vid, sid = dados
    vlog, pastas = videos_logicos(fold)
    n_suj = int(sid.max()) + 1
    if len(pastas) < n_suj:
        if log:
            log.warning("fold %d: %d pastas com sinais < %d sujeitos no arquivo oficial; mapeamento pulado",
                        fold, len(pastas), n_suj)
        return None
    cache = {}
    rot_nome = {0: "0", 1: "5", 2: "10"}
    S = np.zeros((n_suj, len(pastas)))
    Sb = np.zeros((n_suj, len(pastas)))
    viol = np.zeros((n_suj, len(pastas)), int)       # vídeos confiáveis com NF_final > nº de quadros
    linhas, recup = [], []
    for s in range(n_suj):
        vids = list(np.unique(vid[sid == s]))
        seqs = [X[vid == v] for v in vids]
        assert all(verificar_sobreposicao(q) for q in seqs), f"fold {fold} sujeito {s}: sobreposição inválida"
        series = [serie_do_video(q) for q in seqs]
        rec = recuperar_frequencia(series)
        rec.update(sujeito_oficial=s, video_ids=vids, series=series, rotulos=[int(y[vid == v][0]) for v in vids])
        # vídeos do mesmo sujeito com sequências IDÊNTICAS no arquivo oficial (defeito do arquivo, não do mapeamento)
        rec["duplicados"] = [f"{rot_nome[rec['rotulos'][a]]}={rot_nome[rec['rotulos'][b]]}"
                             for a in range(len(seqs)) for b in range(a + 1, len(seqs))
                             if seqs[a].shape == seqs[b].shape and np.array_equal(seqs[a], seqs[b])]
        recup.append(rec)
        for vi, v in enumerate(vids):
            rot = rot_nome[rec["rotulos"][vi]]
            rv = rec["videos"][vi]
            w = np.round(rv["NF"]).astype(np.int64)
            conf = confiavel(rv)
            nf_final = float(rv["NF"][-1]) if len(rv["NF"]) else np.nan
            for k, pasta in enumerate(pastas):
                cid = f"{pasta}_{rot}"
                if cid not in vlog:
                    continue
                arqs, nq = vlog[cid]
                pk = vales_logico(arqs, cache)
                real, acaso, lags = escore_video_coerente(w, pk, nq)
                rb, ab, lb = escore_video(rv["NF"], pk, nq)
                S[s, k] += real - acaso
                Sb[s, k] += rb - ab
                v_dur = bool(conf and np.isfinite(nf_final) and nf_final > nq * (1 + TOL_NF))
                viol[s, k] += int(v_dur)
                linhas.append(dict(fold=fold, sujeito_oficial=s, video_oficial=int(v), rotulo=rot, pasta=pasta,
                                   coincidencia=real, acaso=acaso, liquido=real - acaso, liquido_blocos=rb - ab,
                                   defasagens_blocos=";".join(f"{x:.0f}" for x in lags),
                                   n_piscadas_oficiais=len(w), serie_confiavel=conf, NF_final=nf_final,
                                   n_quadros_candidato=nq, razao_NF_quadros=nf_final / nq if nq else np.nan,
                                   viola_duracao=v_dur))
        if log:
            o = np.argsort(-S[s])
            log.info("fold %d, sujeito oficial %d: K=%d (integralidade %d vs %d), melhor pasta %s (%.2f), 2ª %s (%.2f)",
                     fold, s, rec["K"], rec["escore_K"], rec["escore_K_2o"], pastas[o[0]], S[s, o[0]],
                     pastas[o[1]], S[s, o[1]])
    r_h, c_h = linear_sum_assignment(-S)
    hung = dict(zip(r_h, c_h))
    ordem = atribuicao_ordenada(S)
    # pares confirmados: escore forte, máximo da linha e da coluna, e escolhidos pelo húngaro
    forte = {}
    for s in range(n_suj):
        k = hung[s]
        if S[s, k] >= LIMIAR_FORTE and S[s, k] >= S[s].max() - 1e-12 and S[s, k] >= S[:, k].max() - 1e-12:
            forte[s] = k
    # a ordem crescente é compatível com todos os confirmados?
    ordem_ok = ordem is not None and all(ordem[s] == k for s, k in forte.items())
    usados = set(forte.values())
    adot, classe, motivo = {}, {}, {}
    for s, k in forte.items():
        adot[s], classe[s], motivo[s] = k, "confirmado", "coincidência forte com os vales do EAR"
    if ordem_ok:
        for s in range(n_suj):
            if s not in adot:
                adot[s], classe[s], motivo[s] = ordem[s], "provavel", "ordem de listagem (pastas em ordem crescente, como os confirmados do fold)"
    else:
        # cascata: sujeito (com série oficial confiável) com UMA só pasta livre compatível com a duração fica com ela
        confs = {s: any(confiavel(v) for v in recup[s]["videos"]) for s in range(n_suj)}
        mudou = True
        while mudou:
            mudou = False
            livres = [k for k in range(len(pastas)) if k not in set(adot.values())]
            for s in range(n_suj):
                if s in adot or not confs[s]:
                    continue
                comp = [k for k in livres if viol[s, k] == 0]
                if len(comp) == 1:
                    adot[s], classe[s], motivo[s] = comp[0], "provavel", "única pasta restante compatível com a duração"
                    mudou = True
                    break
        livres = [k for k in range(len(pastas)) if k not in set(adot.values())]
        for s in range(n_suj):
            if s in adot:
                continue
            comp = [k for k in livres if viol[s, k] == 0]
            adot[s], classe[s], motivo[s] = None, "indeterminado", (
                "sem coincidência forte; pastas restantes compatíveis com a duração: "
                + (";".join(pastas[k] for k in comp) if confs[s] else "— (série oficial mal recuperada)"))
    # uma pasta "provável" não pode repetir uma já usada
    vistos = {}
    for s in sorted(adot):
        k = adot[s]
        if k is None:
            continue
        if k in vistos:
            for s2 in (s, vistos[k]):
                if classe[s2] == "provavel":
                    adot[s2], classe[s2], motivo[s2] = None, "indeterminado", "conflito entre critérios"
        else:
            vistos[k] = s
    for s in range(n_suj):
        k = adot[s]
        if k is not None and classe[s] == "provavel" and viol[s, k] > 0:
            motivo[s] += (f"; ATENÇÃO: NF_final da série oficial passa do nº de quadros da pasta em {int(viol[s, k])} vídeo(s) "
                          "(cópia dos autores possivelmente mais longa)")
    mapa = []
    for s in range(n_suj):
        k = adot[s]
        linha_S = S[s]
        o = np.argsort(-linha_S)
        rec = recup[s]
        mapa.append(dict(
            fold=fold, sujeito_oficial=s, pasta=pastas[k] if k is not None else "", classe=classe[s], motivo=motivo[s],
            escore=float(linha_S[k]) if k is not None else np.nan, pasta_melhor=pastas[o[0]], escore_melhor=float(linha_S[o[0]]),
            pasta_segunda=pastas[o[1]], segundo_escore=float(linha_S[o[1]]),
            margem=float(linha_S[k] - np.delete(linha_S, k).max()) if k is not None else np.nan,
            pasta_hungaro=pastas[hung[s]], pasta_ordem=pastas[ordem[s]] if ordem is not None else "",
            pasta_hungaro_blocos=None, escore_blocos=float(Sb[s, k]) if k is not None else np.nan,
            viola_duracao_adotada=int(viol[s, k]) if k is not None else np.nan,
            n_pastas_compativeis_duracao=int((viol[s] == 0).sum()),
            series_confiaveis=";".join("1" if confiavel(v) else "0" for v in rec["videos"]),
            videos_identicos_no_oficial=";".join(rec["duplicados"]),
            K=rec["K"], sigma_F=rec["sigma_F"], erro_reticulado_F=rec["erro_reticulado"],
            integralidade=rec["escore_K"], integralidade_2o=rec["escore_K_2o"],
            frac_NF_inteiro=float(np.mean([v["frac_inteiro"] for v in rec["videos"]])),
            n_piscadas_oficiais=";".join(str(len(z)) for z in rec["series"]),
            piscadas_descartadas=";".join(str(v["quedas"]) for v in rec["videos"]),
            NF_final=";".join(f"{v['NF'][-1]:.0f}" if len(v["NF"]) else "" for v in rec["videos"])))
    r_b, c_b = linear_sum_assignment(-Sb)
    hb = dict(zip(r_b, c_b))
    for m in mapa:
        m["pasta_hungaro_blocos"] = pastas[hb[m["sujeito_oficial"]]]
    mapa = pd.DataFrame(mapa)
    usadas = set(mapa.loc[mapa["pasta"] != "", "pasta"])
    sem_par = [p for p in pastas if p not in usadas]
    pasta_de = dict(zip(mapa["sujeito_oficial"], mapa["pasta"]))
    for rec in recup:
        rec["pasta"] = pasta_de.get(rec["sujeito_oficial"]) or None
        rec["classe"] = dict(zip(mapa["sujeito_oficial"], mapa["classe"]))[rec["sujeito_oficial"]]
    return dict(mapa=mapa, escores=pd.DataFrame(linhas), recup=recup, sem_par=sem_par, S=S, S_blocos=Sb,
                pastas=pastas, ordem_compativel=ordem_ok, n_confirmados=len(forte))


def mapear_fold(fold, log=None, lag_min=None):
    """RODADA PRELIMINAR (escore por blocos independentes; não trata vídeos divididos em partes). Mantida só para
    referência; a rodada final usa mapear_fold_final().

    Infere o mapeamento sujeito-oficial (índice local no arquivo de teste) -> pasta do fold.

    Devolve (tabela_mapeamento, tabela_escores, recuperacao) ou None se faltarem dados."""
    dados = carregar_teste_oficial(fold)
    if dados is None:
        return None
    X, y, vid, sid = dados
    bidx = pd.read_csv(B.FEAT / "uta" / "blinks_index.csv", dtype={"rotulo": str})
    bidx = bidx[bidx["fold"] == fold]
    pastas = sorted(bidx["sujeito"].unique())
    n_suj_oficial = int(sid.max()) + 1
    if len(pastas) < n_suj_oficial:
        # fold ainda incompleto (vídeos sem sinal extraído): a atribuição forçada seria enganosa
        if log:
            log.warning("fold %d: %d pastas com sinais < %d sujeitos no arquivo oficial; mapeamento pulado",
                        fold, len(pastas), n_suj_oficial)
        return None
    nq = dict(zip(bidx["id"], bidx["n_quadros"]))
    cache_vales = {}

    def vales(cid):
        if cid not in cache_vales:
            sig = B.carregar_sinal(ROOT / "data" / "signals" / "uta" / f"{cid}.npz")
            cache_vales[cid] = vales_ear(sig)
        return cache_vales[cid]

    rot_nome = {0: "0", 1: "5", 2: "10"}
    n_suj = int(sid.max()) + 1
    S = np.zeros((n_suj, len(pastas)))
    linhas_esc, recup = [], []
    for s in range(n_suj):
        vids = list(np.unique(vid[sid == s]))
        seqs = [X[vid == v] for v in vids]
        assert all(verificar_sobreposicao(q) for q in seqs), f"fold {fold} sujeito {s}: sobreposição inválida"
        series = [serie_do_video(q) for q in seqs]
        rec = recuperar_frequencia(series)
        rec.update(sujeito_oficial=s, video_ids=vids, series=series,
                   rotulos=[int(y[vid == v][0]) for v in vids])
        recup.append(rec)
        for vi, v in enumerate(vids):
            rot = rot_nome[rec["rotulos"][vi]]
            w = rec["videos"][vi]["NF"]
            for k, pasta in enumerate(pastas):
                cid = f"{pasta}_{rot}"
                if cid not in nq:
                    continue
                real, acaso, lags = escore_video(w, vales(cid), int(nq[cid]), lag_min=lag_min)
                S[s, k] += real - acaso
                linhas_esc.append(dict(fold=fold, sujeito_oficial=s, video_oficial=int(v), rotulo=rot, pasta=pasta,
                                       coincidencia=real, acaso=acaso, liquido=real - acaso,
                                       defasagens_blocos=";".join(f"{x:.0f}" for x in lags)))
        if log:
            o = np.argsort(-S[s])
            log.info("fold %d, sujeito oficial %d: K=%d (integralidade %d vs %d), melhor pasta %s (%.2f), 2ª %s (%.2f)",
                     fold, s, rec["K"], rec["escore_K"], rec["escore_K_2o"], pastas[o[0]], S[s, o[0]],
                     pastas[o[1]], S[s, o[1]])
    r, c = linear_sum_assignment(-S)
    mapa = []
    for s, k in zip(r, c):
        outros = np.delete(S[s], k)
        rec = recup[s]
        mapa.append(dict(fold=fold, sujeito_oficial=int(s), pasta=pastas[k], escore=S[s, k],
                         segundo_escore=float(outros.max()) if len(outros) else np.nan,
                         pasta_segunda=pastas[int(np.argsort(-S[s])[1])] if len(pastas) > 1 else "",
                         margem=S[s, k] - (float(outros.max()) if len(outros) else 0.0),
                         K=rec["K"], sigma_F=rec["sigma_F"], erro_reticulado_F=rec["erro_reticulado"],
                         integralidade=rec["escore_K"],
                         integralidade_2o=rec["escore_K_2o"],
                         frac_NF_inteiro=float(np.mean([v["frac_inteiro"] for v in rec["videos"]])),
                         n_piscadas_oficiais=";".join(str(len(z)) for z in rec["series"]),
                         piscadas_descartadas=";".join(str(v["quedas"]) for v in rec["videos"])))
    mapa = pd.DataFrame(mapa)
    sem_par = [p for p in pastas if p not in set(mapa["pasta"])]
    mapa.attrs["sem_par"] = sem_par
    pasta_de = dict(zip(mapa["sujeito_oficial"], mapa["pasta"]))
    for rec in recup:
        rec["pasta"] = pasta_de.get(rec["sujeito_oficial"])
    return mapa, pd.DataFrame(linhas_esc), recup, sem_par


def mapear_por_contagem(recup, n_seq_nosso, pastas):
    """Alternativa ingênua (só para comparação): atribuição húngara com custo = soma, nos 3 vídeos, de
    |log nº de sequências oficiais - log nº de sequências nossas|. Devolve dict sujeito_oficial -> pasta."""
    rot = {0: 0, 1: 5, 2: 10}
    C = np.zeros((len(recup), len(pastas)))
    for i, rec in enumerate(recup):
        for vi in range(len(rec["video_ids"])):
            n_of = max(1, (len(rec["series"][vi]) - 30) // 2 + 1 if len(rec["series"][vi]) > 30 else 1)
            for k, pasta in enumerate(pastas):
                n_no = max(1, n_seq_nosso.get(f"{pasta}_{rot[rec['rotulos'][vi]]}", 1))
                C[i, k] += abs(np.log(n_of) - np.log(n_no))
    r, c = linear_sum_assignment(C)
    return {int(recup[i]["sujeito_oficial"]): pastas[k] for i, k in zip(r, c)}
