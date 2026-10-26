"""Sequências de 30 piscadas (passo 2) no formato de Ghoddoosian et al. (2019) e validação contra as oficiais.

Rodar no .venv, DEPOIS de src/features/blinks.py:

    python src/features/sequences.py                                   # terço de referência = último (código original)
    python src/features/sequences.py --terco primeiro --mapa-existente  # reaproveita o mapeamento já calculado

RODADA FINAL: os 5 folds da UTA-RLDD (60 pastas; características oficiais para 58 sujeitos). Por sujeito, a partir de
data/features/final/uta/blinks/<arquivo>.csv (só eventos do tipo "piscada"):
  * VÍDEOS DIVIDIDOS (F3_32_10 e F5_49_10 em duas partes): as piscadas das partes são CONCATENADAS em ordem, com a
    contagem contínua (Frequency = 100 x piscadas acumuladas desde o início da parte 1 / quadros com rosto acumulados;
    freq_hz idem com o tempo de rosto), como uma única série por vídeo lógico — como os autores fizeram (um arquivo
    de piscadas por vídeo; a série oficial de F3_32_10 passa do fim da parte 1, ou seja, a contagem não recomeçou).
  * normalização por sujeito com UM TERÇO das piscadas do vídeo ALERTA (0): média e desvio-padrão
    (np.std, ddof=0) de cada característica; se o desvio for 0, usa o desvio de todas as piscadas do
    vídeo alerta (como Preprocessing.py). As piscadas usadas na referência NÃO entram nas sequências
    do vídeo alerta. ESCOLHA: último terço (padrão), igual ao código que gerou os .npy oficiais
    (Preprocessing.py: "Using the last bunch_size number of blinks"); o artigo diz primeiro terço
    (opção --terco primeiro).
  * sequências: unroll_in_time (reimplementação vetorizada, testada contra a cópia literal do original em
    dados aleatórios): janela 30, passo 2; <= 30 piscadas -> 1 sequência com zeros à esquerda;
    > 30 -> (n-30)//2 + 1 sequências (as piscadas finais que não completam uma janela ficam de fora).
  * três variantes das 4 características [Frequency, Amplitude, Duration, Velocity]:
        X          "limiar":    frequencia (por 100 quadros), amplitude, duracao (quadros), velocidade (EAR/quadro)
        X_segundos "segundos":  freq_hz, amplitude, dur_s, vel_s (EAR/s)  -> independente do fps
        X_estendida "estendida": frequencia, amplitude_ext, dur_ext_quadros, velocidade_ext (bordas no olho aberto)
Saídas: data/features/final/uta/sequences.npz (X, X_segundos, X_estendida, y 0/1/2, y_kss 0/5/10, sujeito, video
        [vídeo lógico], fold, seq_no_video), data/features/final/uta/sequences_normalizacao.csv,
        results/exp5_sequencias_resumo.csv.
Validação (para cada fold com BlinksTest_30_Fold<k>.npy): mapeamento sujeito-oficial <-> pasta inferido por
engenharia reversa (src/features/uta_oficial.py: escore coerente, checagem de duração, classes de evidência) e
comparação vídeo a vídeo das características (só pares "confirmado" e "provavel"; coluna classe):
results/exp5_uta_mapeamento_oficial.csv, exp5_uta_mapeamento_escores.csv, exp5_uta_mapeamento_sensibilidade.csv,
exp5_uta_validacao_oficial_videos.csv, exp5_uta_validacao_oficial_correlacoes.csv, exp5_uta_validacao_oficial_classes.csv
e data/features/final/uta/oficial_piscadas_fold<k>.csv (piscadas oficiais recuperadas: F, NF, T e z).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import blinks as B  # noqa: E402

ROOT = B.ROOT
RES = ROOT / "results"
JANELA = 30
PASSO = 2
VARIANTES = {
    "limiar": ["frequencia", "amplitude", "duracao", "velocidade"],
    "segundos": ["freq_hz", "amplitude", "dur_s", "vel_s"],
    "estendida": ["frequencia", "amplitude_ext", "dur_ext_quadros", "velocidade_ext"],
}
CHAVE_NPZ = {"limiar": "X", "segundos": "X_segundos", "estendida": "X_estendida"}
NOMES_CARACT = ["frequencia", "amplitude", "duracao", "velocidade"]
KSS_CLASSE = {0: 0, 5: 1, 10: 2}


# --------------------------------------------------------------------------------------
# Janelamento
# --------------------------------------------------------------------------------------
def unroll_in_time_ref(in_data, window_size, stride):
    """CÓPIA LITERAL de Preprocessing.py (Ghoddoosian et al.) — usada só para testar a reimplementação."""
    # in_data is [n,4]            out_data is [N,Window_size,4]
    n = len(in_data)
    if n <= window_size:
        out_data = np.zeros([1, window_size, 4])
        out_data[0, -n:, :] = in_data
        return out_data
    else:
        N = ((n - window_size) // stride) + 1
        out_data = np.zeros([N, window_size, 4])
        for i in range(N):
            if i * stride + window_size <= n:
                out_data[i, :, :] = in_data[i * stride:i * stride + window_size, :]
            else:  # this line should not ever be executed because of the for mula used above N is the exact time the loop is executed
                break

        return out_data


def unroll_in_time(x, janela=JANELA, passo=PASSO):
    """Reimplementação vetorizada de unroll_in_time. Diferença documentada: n == 0 devolve 0 sequências
    (o original falha nesse caso)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x deve ser (n, f)")
    n, f = x.shape
    if n == 0:
        return np.zeros((0, janela, f))
    if n <= janela:
        out = np.zeros((1, janela, f))
        out[0, -n:, :] = x
        return out
    N = (n - janela) // passo + 1
    idx = np.arange(janela)[None, :] + passo * np.arange(N)[:, None]
    return x[idx]


def n_sequencias(n, janela=JANELA, passo=PASSO):
    return 0 if n <= 0 else (1 if n <= janela else (n - janela) // passo + 1)


# --------------------------------------------------------------------------------------
# Normalização por sujeito
# --------------------------------------------------------------------------------------
def referencia_alerta(A, terco="ultimo"):
    """A: (n, f) características das piscadas do vídeo alerta, em ordem temporal.

    Devolve (mu, sd, indices_restantes, info). Regras de Preprocessing.py: bunch = n // 3; desvio 0 -> desvio
    de todas as piscadas do vídeo alerta. Se bunch == 0 (n < 3), usa todas as piscadas (sinalizado)."""
    A = np.asarray(A, dtype=float)
    n = len(A)
    bunch = n // 3
    info = dict(n_alerta=n, n_referencia=bunch, fallback_todas=False, sd_zero=[])
    if bunch == 0:
        base, resto = A, np.arange(n)
        info["fallback_todas"] = True
    elif terco == "ultimo":
        base, resto = A[-bunch:], np.arange(n - bunch)
    elif terco == "primeiro":
        base, resto = A[:bunch], np.arange(bunch, n)
    else:
        raise ValueError(terco)
    with np.errstate(invalid="ignore"):
        mu = np.nanmean(base, axis=0) if len(base) else np.full(A.shape[1], np.nan)
        sd = np.nanstd(base, axis=0) if len(base) else np.full(A.shape[1], np.nan)
        sd_all = np.nanstd(A, axis=0) if n else np.full(A.shape[1], np.nan)
    for j in range(A.shape[1]):
        if not (np.isfinite(sd[j]) and sd[j] > 0):
            info["sd_zero"].append(j)
            sd[j] = sd_all[j] if (np.isfinite(sd_all[j]) and sd_all[j] > 0) else 1.0
    return mu, sd, resto, info


def concatenar_partes(evs, sinais):
    """Concatena as piscadas (DataFrames já filtrados em tipo == "piscada", em ordem) das partes de um vídeo, com a
    contagem contínua: frequencia = 100 x piscadas acumuladas / quadros com rosto acumulados desde o início da
    parte 1; freq_hz idem com o tempo de rosto; índices de quadro deslocados pelo nº de quadros das partes
    anteriores. As demais características são locais à piscada e não mudam. Com uma parte só, devolve-a intacta."""
    if len(evs) == 1:
        return evs[0].reset_index(drop=True)
    blocos, T_off, NF_off, TF_off, Q_off = [], 0, 0, 0.0, 0
    for ev, sig in zip(evs, sinais):
        ev = ev.copy().reset_index(drop=True)
        face_cum = np.cumsum(sig["face"])
        tface_cum = np.cumsum(np.where(sig["face"], sig["d"], 0.0))
        if len(ev):
            fim = ev["fim"].astype(int).to_numpy()
            T = np.arange(1, len(ev) + 1) + T_off
            ev["frequencia"] = 100.0 * T / (face_cum[fim] + NF_off)
            ev["freq_hz"] = T / (tface_cum[fim] + TF_off)
            ev["n_ordem"] = T
            for c in ("inicio", "fundo", "fim", "inicio_ext", "fim_ext"):
                ev[c] = ev[c] + Q_off
        T_off += len(ev)
        NF_off += int(face_cum[-1]) if len(face_cum) else 0
        TF_off += float(tface_cum[-1]) if len(tface_cum) else 0.0
        Q_off += sig["n"]
        blocos.append(ev)
    return pd.concat(blocos, ignore_index=True)


def carregar_piscadas(arquivos):
    """Piscadas (tipo "piscada") de um vídeo LÓGICO, em ordem (partes concatenadas por concatenar_partes)."""
    if isinstance(arquivos, str):
        arquivos = [arquivos]
    evs, sinais = [], []
    for a in arquivos:
        ev = pd.read_csv(B.FEAT / "uta" / "blinks" / f"{a}.csv")
        ev = ev[ev["tipo"] == "piscada"].sort_values("inicio").reset_index(drop=True)
        ev["arquivo"] = a
        evs.append(ev)
        if len(arquivos) > 1:
            sinais.append(B.carregar_sinal(ROOT / "data" / "signals" / "uta" / f"{a}.npz"))
    return concatenar_partes(evs, sinais)


def montar(terco="ultimo", log=None):
    """Monta as sequências de todos os sujeitos disponíveis. Devolve (dict de arrays, normalização, resumo,
    séries normalizadas por vídeo para a validação)."""
    log = log or logging.getLogger(__name__)
    bidx = pd.read_csv(B.FEAT / "uta" / "blinks_index.csv", dtype={"rotulo": str, "parte": str})
    bidx["parte"] = bidx["parte"].fillna("")
    folds_ok = B.uta_folds_ok()
    bidx = bidx[bidx["fold"].isin(folds_ok)]
    arr = {k: [] for k in ["X", "X_segundos", "X_estendida", "y", "y_kss", "sujeito", "video", "fold",
                           "seq_no_video"]}
    norm_rows, resumo_rows, series = [], [], {}
    for suj, g in bidx.groupby("sujeito", sort=True):
        vids, arqs = {}, {}
        for rot, gv in g.groupby("rotulo"):
            gv = gv.sort_values("parte")
            vids[str(rot)] = gv["video"].iloc[0]
            arqs[str(rot)] = list(gv["id"])
        if "0" not in vids:
            log.warning("%s: sem vídeo alerta (0); sujeito ignorado nas sequências", suj)
            continue
        fold = int(g["fold"].iloc[0])
        ev = {r: carregar_piscadas(a) for r, a in arqs.items()}
        blocos = {}
        for var, cols in VARIANTES.items():
            A = ev["0"][cols].to_numpy(float)
            mu, sd, resto, info = referencia_alerta(A, terco)
            norm_rows.append(dict(sujeito=suj, fold=fold, variante=var, terco=terco, **{f"mu_{c}": m for c, m in zip(NOMES_CARACT, mu)},
                                  **{f"sd_{c}": s for c, s in zip(NOMES_CARACT, sd)},
                                  n_alerta=info["n_alerta"], n_referencia=info["n_referencia"],
                                  fallback_todas=info["fallback_todas"], sd_zero=";".join(map(str, info["sd_zero"]))))
            for r in ("0", "5", "10"):
                if r not in ev:
                    continue
                Z = (ev[r][cols].to_numpy(float) - mu) / sd
                if r == "0":
                    Z = Z[resto]
                blocos.setdefault(r, {})[var] = Z
                series[(vids[r], var)] = Z
        for r in ("0", "5", "10"):
            if r not in blocos:
                continue
            seqs = {var: unroll_in_time(blocos[r][var]) for var in VARIANTES}
            N = len(seqs["limiar"])
            assert all(len(s) == N for s in seqs.values())
            for var in VARIANTES:
                arr[CHAVE_NPZ[var]].append(seqs[var])
            kss = int(r)
            arr["y"].append(np.full(N, KSS_CLASSE[kss]))
            arr["y_kss"].append(np.full(N, kss))
            arr["sujeito"].append(np.full(N, suj))
            arr["video"].append(np.full(N, vids[r]))
            arr["fold"].append(np.full(N, fold))
            arr["seq_no_video"].append(np.arange(N))
            resumo_rows.append(dict(fold=fold, sujeito=suj, video=vids[r], rotulo=kss, n_arquivos=len(arqs[r]),
                                    arquivos=";".join(arqs[r]),
                                    n_piscadas=len(ev[r]), n_piscadas_nas_sequencias=len(blocos[r]["limiar"]),
                                    n_sequencias=N))
    out = {}
    for k, v in arr.items():
        if not v:
            out[k] = np.zeros((0, JANELA, 4)) if k.startswith("X") else np.zeros(0)
        else:
            out[k] = np.concatenate(v)
    return out, pd.DataFrame(norm_rows), pd.DataFrame(resumo_rows), series


# --------------------------------------------------------------------------------------
# Validação contra as características oficiais
# --------------------------------------------------------------------------------------
def validar_fold(fold, series_nossas, resumo, log, mapa_existente=None):
    """Mapeamento (uta_oficial.mapear_fold_final) + piscadas oficiais recuperadas + comparação vídeo a vídeo.

    mapa_existente: DataFrame do mapeamento já calculado (results/exp5_uta_mapeamento_oficial.csv) para não refazer
    os escores (usado na variante --terco primeiro; o mapeamento não depende da nossa normalização)."""
    import uta_oficial as O
    sens = pd.DataFrame()
    esc = pd.DataFrame()
    if mapa_existente is None:
        res = O.mapear_fold_final(fold, log=log)
        if res is None:
            log.info("fold %d: sem arquivo oficial de teste ou sem vídeos; validação pulada", fold)
            return None
        mapa, esc, recup = res["mapa"], res["escores"], res["recup"]
        log.info("fold %d: %d confirmados; ordem crescente compatível com os confirmados: %s; pastas sem par: %s", fold,
                 res["n_confirmados"], res["ordem_compativel"], res["sem_par"])
        log.info("fold %d: mapeamento:\n%s", fold, mapa[["sujeito_oficial", "pasta", "classe", "escore", "margem",
                                                           "pasta_hungaro_blocos", "viola_duracao_adotada"]].round(3).to_string(index=False))
        # sensibilidade aos parâmetros do escore coerente (pares confirmados/prováveis iguais?)
        linhas = []
        for lmin, queda in ((-1500, 150), (-6000, 600)):
            O.LAGC_MIN, O.QUEDA_MAX = lmin, queda
            r2 = O.mapear_fold_final(fold)
            O.LAGC_MIN, O.QUEDA_MAX = -3000, 300
            m2 = r2["mapa"]
            a = dict(zip(mapa["sujeito_oficial"], zip(mapa["pasta"], mapa["classe"])))
            b = dict(zip(m2["sujeito_oficial"], zip(m2["pasta"], m2["classe"])))
            linhas.append(dict(fold=fold, lag_min=lmin, queda_max=queda, mapeamento_igual=a == b,
                               n_confirmados=int((m2["classe"] == "confirmado").sum()),
                               n_provaveis=int((m2["classe"] == "provavel").sum()),
                               n_indeterminados=int((m2["classe"] == "indeterminado").sum()),
                               pastas_iguais=int(sum(a[k][0] == b[k][0] for k in a)),
                               margem_min_confirmados=float(m2.loc[m2["classe"] == "confirmado", "margem"].min()),
                               sem_par=";".join(r2["sem_par"])))
        sens = pd.DataFrame(linhas)
        log.info("fold %d: sensibilidade do mapeamento aos parâmetros do escore coerente:\n%s", fold,
                 sens.to_string(index=False))
    else:
        mapa = mapa_existente[mapa_existente["fold"] == fold].copy()
        if mapa.empty:
            return None
        X, y, vid, sid = O.carregar_teste_oficial(fold)
        recup = []
        pasta_de = dict(zip(mapa["sujeito_oficial"], mapa["pasta"].fillna("")))
        classe_de = dict(zip(mapa["sujeito_oficial"], mapa["classe"]))
        for s_ in range(int(sid.max()) + 1):
            vids = list(np.unique(vid[sid == s_]))
            series = [O.serie_do_video(X[vid == v]) for v in vids]
            rec = O.recuperar_frequencia(series)
            rec.update(sujeito_oficial=s_, video_ids=vids, series=series, rotulos=[int(y[vid == v][0]) for v in vids],
                       pasta=pasta_de.get(s_) or None, classe=classe_de.get(s_))
            recup.append(rec)
    # piscadas oficiais recuperadas
    linhas = []
    for rec in recup:
        for vi, v in enumerate(rec["video_ids"]):
            z = rec["series"][vi]
            vv = rec["videos"][vi]
            for i in range(len(z)):
                linhas.append(dict(fold=fold, sujeito_oficial=rec["sujeito_oficial"], pasta=rec["pasta"] or "",
                                   classe=rec["classe"], video_oficial=int(v), rotulo=[0, 5, 10][rec["rotulos"][vi]],
                                   piscada=i, F=vv["F"][i], NF=vv["NF"][i], T=int(vv["T"][i]), NF_inteiro=bool(vv["res"][i] <= 1),
                                   **{f"z_{c}": z[i, j] for j, c in enumerate(NOMES_CARACT)}))
    pd.DataFrame(linhas).to_csv(B.FEAT / "uta" / f"oficial_piscadas_fold{fold}.csv", index=False)
    # comparação vídeo a vídeo (só pares com pasta)
    vrows = []
    for rec in recup:
        if not rec["pasta"]:
            continue
        for vi in range(len(rec["video_ids"])):
            kss = [0, 5, 10][rec["rotulos"][vi]]
            vid_ = f"{rec['pasta']}_{kss}"
            z_of = rec["series"][vi]
            r = dict(fold=fold, sujeito_oficial=rec["sujeito_oficial"], pasta=rec["pasta"], classe=rec["classe"],
                     video=vid_, rotulo=kss, n_piscadas_oficial_seq=len(z_of),
                     T_final_oficial=int(rec["videos"][vi]["T"][-1]) if len(z_of) else 0,
                     NF_final_oficial=float(rec["videos"][vi]["NF"][-1]) if len(z_of) else np.nan)
            rs = resumo[resumo["video"] == vid_]
            r["n_piscadas_nossa_seq"] = int(rs["n_piscadas_nas_sequencias"].iloc[0]) if len(rs) else np.nan
            r["n_piscadas_nossa_total"] = int(rs["n_piscadas"].iloc[0]) if len(rs) else np.nan
            for j, c in enumerate(NOMES_CARACT):
                r[f"oficial_media_{c}"] = float(np.mean(z_of[:, j])) if len(z_of) else np.nan
                r[f"oficial_mediana_{c}"] = float(np.median(z_of[:, j])) if len(z_of) else np.nan
            for var in VARIANTES:
                Z = series_nossas.get((vid_, var))
                for j, c in enumerate(NOMES_CARACT):
                    r[f"{var}_media_{c}"] = float(np.nanmean(Z[:, j])) if Z is not None and len(Z) else np.nan
                    r[f"{var}_mediana_{c}"] = float(np.nanmedian(Z[:, j])) if Z is not None and len(Z) else np.nan
            vrows.append(r)
    vdf = pd.DataFrame(vrows)
    return mapa, esc, vdf, sens


def correlacoes(vdf):
    """Correlação (Pearson e Spearman) entre a média oficial e a nossa, por variante e característica,
    sobre os vídeos casados (todos e só os vídeos 5/10, em que a referência não entra)."""
    rows = []
    conf = (vdf["classe"] == "confirmado").to_numpy() if "classe" in vdf.columns else np.ones(len(vdf), bool)
    subconj = (("todos", np.ones(len(vdf), bool)), ("so_5_e_10", vdf["rotulo"].isin([5, 10]).to_numpy()),
               ("confirmados", conf), ("confirmados_so_5_e_10", conf & vdf["rotulo"].isin([5, 10]).to_numpy()))
    for subset, m in subconj:
        for var in VARIANTES:
            for c in NOMES_CARACT:
                for est in ("media", "mediana"):
                    a = vdf.loc[m, f"oficial_{est}_{c}"].to_numpy(float)
                    b = vdf.loc[m, f"{var}_{est}_{c}"].to_numpy(float)
                    ok = np.isfinite(a) & np.isfinite(b)
                    if ok.sum() >= 5:
                        pr = pearsonr(a[ok], b[ok])
                        sr = spearmanr(a[ok], b[ok])
                        rows.append(dict(videos=subset, variante=var, caracteristica=c, estatistica=est, n=int(ok.sum()),
                                         pearson=pr.statistic, p_pearson=pr.pvalue, spearman=sr.statistic,
                                         p_spearman=sr.pvalue))
    return pd.DataFrame(rows)


def medias_por_classe(vdf):
    rows = []
    for kss, g in vdf.groupby("rotulo"):
        r = dict(rotulo=kss, n_videos=len(g))
        for c in NOMES_CARACT:
            r[f"oficial_{c}"] = float(g[f"oficial_mediana_{c}"].median())
            for var in VARIANTES:
                r[f"{var}_{c}"] = float(g[f"{var}_mediana_{c}"].median())
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Sequências de 30 piscadas (Ghoddoosian) e validação contra as oficiais.")
    ap.add_argument("--terco", default="ultimo", choices=["ultimo", "primeiro"])
    ap.add_argument("--sem-validacao", action="store_true")
    ap.add_argument("--mapa-existente", action="store_true",
                    help="reaproveita results/exp5_uta_mapeamento_oficial.csv (não recalcula os escores)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
    log = logging.getLogger("sequences")
    folds = B.uta_folds_ok()
    log.info("Folds da UTA-RLDD com .ok: %s", folds)
    # teste rápido da reimplementação (o teste completo está em test_features.py)
    rng = np.random.default_rng(42)
    for n in (1, 7, 29, 30, 31, 32, 33, 61, 100):
        x = rng.normal(size=(n, 4))
        assert np.array_equal(unroll_in_time(x), unroll_in_time_ref(x, 30, 2)), n
    out, norm, resumo, series = montar(args.terco, log)
    sufixo = "" if args.terco == "ultimo" else "_primeiro_terco"
    p = B.FEAT / "uta" / f"sequences{sufixo}.npz"
    np.savez_compressed(p, **{k: (v.astype(str) if v.dtype == object else v) for k, v in out.items()},
                        caracteristicas=np.array(NOMES_CARACT), terco=np.array(args.terco))
    norm.to_csv(B.FEAT / "uta" / f"sequences_normalizacao{sufixo}.csv", index=False)
    resumo.to_csv(RES / f"exp5_sequencias_resumo{sufixo}.csv", index=False)
    log.info("sequências: %d (X %s), %d vídeos lógicos com sequência (de %d), %d sujeitos -> %s", len(out["y"]), out["X"].shape,
             int((resumo["n_sequencias"] > 0).sum()) if len(resumo) else 0, resumo["video"].nunique() if len(resumo) else 0,
             resumo["sujeito"].nunique() if len(resumo) else 0, p.relative_to(ROOT))
    if len(resumo) and (resumo["n_sequencias"] == 0).any():
        log.warning("vídeos sem nenhuma piscada (sem sequência): %s",
                    ", ".join(resumo.loc[resumo["n_sequencias"] == 0, "video"]))
    if len(out["y"]):
        log.info("por classe (0/1/2): %s", np.bincount(out["y"].astype(int), minlength=3).tolist())
        for f in sorted(set(out["fold"].tolist())):
            m = out["fold"] == f
            log.info("  fold %d: %d sequências, %d sujeitos, por classe %s", f, int(m.sum()), len(set(out["sujeito"][m])),
                     np.bincount(out["y"][m].astype(int), minlength=3).tolist())
    if args.sem_validacao:
        return 0
    mapa_ex = None
    if args.mapa_existente:
        mp = RES / "exp5_uta_mapeamento_oficial.csv"
        if not mp.exists():
            raise SystemExit("rode antes sem --mapa-existente")
        mapa_ex = pd.read_csv(mp, dtype={"pasta": str})
    mapas, escs, vdfs, senss = [], [], [], []
    for fold in folds:
        r = validar_fold(fold, series, resumo, log, mapa_ex)
        if r is None:
            continue
        mapa, esc, vdf, sens = r
        mapas.append(mapa); escs.append(esc); vdfs.append(vdf); senss.append(sens)
    if not mapas:
        return 0
    if mapa_ex is None:
        mapa = pd.concat(mapas, ignore_index=True)
        mapa.to_csv(RES / "exp5_uta_mapeamento_oficial.csv", index=False)
        pd.concat(escs, ignore_index=True).to_csv(RES / "exp5_uta_mapeamento_escores.csv", index=False)
        pd.concat(senss, ignore_index=True).to_csv(RES / "exp5_uta_mapeamento_sensibilidade.csv", index=False)
        log.info("Mapeamento (5 folds): %s", mapa["classe"].value_counts().to_dict())
    vdf = pd.concat(vdfs, ignore_index=True)
    vdf.to_csv(RES / f"exp5_uta_validacao_oficial_videos{sufixo}.csv", index=False)
    cor = correlacoes(vdf)
    cor.to_csv(RES / f"exp5_uta_validacao_oficial_correlacoes{sufixo}.csv", index=False)
    cls = medias_por_classe(vdf[vdf["classe"] == "confirmado"])
    cls.to_csv(RES / f"exp5_uta_validacao_oficial_classes{sufixo}.csv", index=False)
    sub = cor[(cor["estatistica"] == "mediana")]
    log.info("Correlação vídeo a vídeo (mediana por vídeo), oficial x nossa (Spearman):\n%s",
             sub.pivot_table(index=["videos", "caracteristica"], columns="variante", values="spearman").round(3).to_string())
    log.info("Medianas por classe (mediana dos vídeos, só confirmados):\n%s", cls.round(3).T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
