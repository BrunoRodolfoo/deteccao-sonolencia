#!/usr/bin/env python3
"""Exp. 5b: robustez da queda B − C (revisão final do artigo, 26/10/2026).

Rodar no .venv, DEPOIS de `bash src/experiments/run_exp5b.sh`. Usa as mesmas janelas de 10 s, a mesma receita de
modelos (novo_modelo), os mesmos modelos finais de C (models/exp5/) e a mesma reamostragem de sujeitos da UTA-RLDD
do Exp. 5b, importados de exp5_transfer.py:

    .venv/bin/python src/experiments/exp5b_robustez.py            # ~15 min (o bootstrap em dois níveis domina)
    .venv/bin/python src/experiments/exp5b_robustez.py --rapido   # 100 reamostragens e 5 sorteios (só para testar)
    .venv/bin/python src/experiments/exp5b_robustez.py --so-md    # refaz só o .md a partir do CSV

Análises (todas no esquema principal B3, alerta × sonolento, janelas de 10 s):
  (0) réplica da rodada principal: B (5 folds oficiais) e C (modelo final do DMD) têm de reproduzir a AUC e o IC
      da queda gravados em results/exp5b_transferencia.csv; senão o script para.
  (1) referência alternativa na UTA-RLDD: a centragem usa a média das janelas do vídeo de pouca vigilância do sujeito
      (código 5, KSS 6 e 7), que não entra no B3,
      no lugar do último terço do vídeo alerta. Assim nem as janelas alertas nem as sonolentas avaliadas vêm do
      vídeo da referência. O DMD não muda (referência = s3), e o limiar de EAR do detector de piscadas continua o do
      vídeo alerta (as características brutas são as mesmas).
  (2) bootstrap em dois níveis da queda: em cada reamostragem, os sujeitos do DMD são sorteados com reposição e C é
      treinado de novo; os sujeitos da UTA-RLDD seguem a mesma reamostragem pareada do Exp. 5b. O IC passa a incluir
      a variância do treino de C.
  (3) volume do treino: B treinado com só 13 sujeitos da UTA-RLDD por fold (o número de motoristas do DMD), com todas
      as janelas deles ou com 510 janelas (o tamanho do treino de C), em sorteios repetidos.
  (4) vazamento no mesmo cenário: folds oficiais por sujeito × partição aleatória das janelas em 5 partes (sementes 42,
      43 e 44, como no Exp. 2), com e sem centragem.
  (5) ponto de operação da regra do rótulo do DMD (fechamento >= 0,5 s ou bocejo): sensibilidade de cada modelo no
      mesmo falso alarme da regra nas janelas alertas. O limiar usa o rótulo do teste: é diagnóstico, não desempenho.

Saídas: results/exp5b_robustez.csv (uma linha por número) e results/exp5b_robustez.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp5_transfer as E  # noqa: E402  (mesma receita, mesmos dados e mesma reamostragem do Exp. 5b)

ROOT = E.ROOT
CSV_OUT = E.RES / "exp5b_robustez.csv"
MD_OUT = E.RES / "exp5b_robustez.md"
N_SORTEIOS_TREZE = 20
SEMENTES_VAZAMENTO = (42, 43, 44)
N_SUJ_DMD = 13


def v1(x: float) -> str:
    """Uma casa decimal com vírgula (formato dos números no artigo)."""
    return f"{x:.1f}".replace(".", ",").replace("-", "−")


def ic_txt(lo: float, hi: float) -> str:
    return f"[{v1(lo)}; {v1(hi)}]"


def escores_b(X, y, folds, suj, nome):
    s, _ = E.validacao_cruzada(X, y, folds, suj, nome)
    return s


def queda_pareada(y, sB, sC, suj, ream):
    """AUC de B, de C e queda B − C (pp) com IC pelo bootstrap pareado de sujeitos da UTA-RLDD."""
    rB, rC = E.avaliar(y, sB, suj, ream, auc_por_sujeito=False), E.avaliar(y, sC, suj, ream, auc_por_sujeito=False)
    q = 100 * (rB["boot"]["auc"] - rC["boot"]["auc"])
    lo, hi, _ = E.ic(q)
    return dict(auc_B=100 * rB["ponto"]["auc"], auc_C=100 * rC["ponto"]["auc"],
                queda=100 * (rB["ponto"]["auc"] - rC["ponto"]["auc"]), ic=(lo, hi),
                ic_B=tuple(100 * np.array(E.ic(rB["boot"]["auc"])[:2])),
                ic_C=tuple(100 * np.array(E.ic(rC["boot"]["auc"])[:2])))


def queda_video(u, sB, sC, ream):
    vB, vC = E.por_video(u, sB), E.por_video(u, sC)
    rB, rC = E.avaliar_video(vB, ream), E.avaliar_video(vC, ream)
    q = 100 * (rB["boot"]["auc"] - rC["boot"]["auc"])
    lo, hi, _ = E.ic(q)
    return dict(auc_B=100 * rB["ponto"]["auc"], auc_C=100 * rC["ponto"]["auc"],
                queda=100 * (rB["ponto"]["auc"] - rC["ponto"]["auc"]), ic=(lo, hi))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rapido", action="store_true", help="100 reamostragens e 5 sorteios (teste; não grava)")
    ap.add_argument("--so-md", action="store_true", help="refaz só o .md a partir do CSV já gravado")
    args = ap.parse_args()
    if args.so_md:
        df = pd.read_csv(CSV_OUT)
        escrever_md(df, str(df["execucao"].iloc[0]))
        print(f"-> {MD_OUT.relative_to(ROOT)}")
        return 0
    n_boot2 = 100 if args.rapido else E.N_BOOT
    n_treze = 5 if args.rapido else N_SORTEIOS_TREZE
    t0 = time.time()
    execucao = datetime.now().strftime("%Y-%m-%d %H:%M")
    linhas = []

    def reg(analise, variante, modelo, nivel, metrica, valor, ic=(np.nan, np.nan), n_rep=np.nan, unidade="", desc=""):
        linhas.append(dict(analise=analise, variante=variante, modelo=modelo, nivel=nivel, metrica=metrica,
                           valor=float(valor), ic95_inf=float(ic[0]), ic95_sup=float(ic[1]), n_rep=n_rep,
                           unidade=unidade, descricao=desc, execucao=execucao))

    # ---------------- dados (os mesmos do Exp. 5b) ----------------
    uta, _ = E.ler_janelas("uta", 10)
    dmd, _ = E.ler_janelas("dmd", 10)
    cfg = E.VARIANTES["principal"]
    d = E.conjunto_dmd(dmd, cfg["rotulo"], cfg["excluir"])
    u = E.conjunto_uta(uta, "B3")
    cols_c, cols_b = E.colunas("c"), E.colunas("bruta")
    y, suj, folds = u["y"].to_numpy(), u["sujeito"].to_numpy(), u["fold"].to_numpy()
    suj_uta = sorted(uta["sujeito"].unique())
    suj_dmd = sorted(d["sujeito"].unique())
    assert len(suj_uta) == 60 and len(suj_dmd) == N_SUJ_DMD, (len(suj_uta), len(suj_dmd))
    ream_uta = E.Reamostragem(suj_uta, E.N_BOOT, np.random.default_rng(E.SEED))   # a mesma do Exp. 5b (sorteada primeiro)
    print(f"UTA B3: {len(u)} janelas ({int(y.sum())} sonolentas), {len(np.unique(suj))} sujeitos | "
          f"DMD: {len(d)} janelas ({int(d['y'].sum())} sonolentas), {len(suj_dmd)} sujeitos", flush=True)

    # referência alternativa: média das janelas inteiras do vídeo KSS 5 de cada sujeito (como normalizar() do windows.py)
    v5 = uta[(uta["rotulo_kss"] == 5) & ~uta["parcial"]]
    mu5 = v5.groupby("sujeito")[E.FEATURES].mean()
    assert set(u["sujeito"]) <= set(mu5.index), "sujeito sem vídeo KSS 5"
    X_alt = (u[E.FEATURES] - mu5.loc[u["sujeito"]].to_numpy()).to_numpy(float)

    csv_5b = pd.read_csv(E.CSV_RES, low_memory=False)
    final = {(var, nome): joblib.load(E.MODELS / f"{E.PREFIXO_MODELO}{var}_{nome}.joblib")
             for var in ("principal", "sem_normalizacao") for nome in E.MODELOS}

    sB_c, sC_c = {}, {}
    for nome in E.MODELOS:
        # (0) réplica da rodada principal
        sB_c[nome] = escores_b(u[cols_c].to_numpy(float), y, folds, suj, nome)
        sC_c[nome] = final[("principal", nome)].predict_proba(u[cols_c].to_numpy(float))[:, 1]
        r = queda_pareada(y, sB_c[nome], sC_c[nome], suj, ream_uta)
        ref = csv_5b[(csv_5b.experimento == "queda_B_menos_C") & (csv_5b.variante == "principal") &
                     (csv_5b.modelo == nome) & (csv_5b.esquema == "B3") & (csv_5b.nivel == "janela") &
                     (csv_5b.estrato == "todos") & (csv_5b.limiar_tipo == "fixo_0.5") & (csv_5b.metrica == "auc")]
        assert len(ref) == 1
        for a, b in ((r["queda"], ref.valor.iloc[0]), (r["ic"][0], ref.ic95_inf.iloc[0]), (r["ic"][1], ref.ic95_sup.iloc[0])):
            if abs(a - b) > 1e-6:
                raise SystemExit(f"{nome}: a réplica não reproduz o Exp. 5b ({a:.6f} x {b:.6f}); rode run_exp5b.sh antes")
        reg("replica_principal", "ref_video_alerta", nome, "janela", "queda_auc", r["queda"], r["ic"], E.N_BOOT, "pp",
            "réplica do Exp. 5b: queda B − C de AUC por janela (confere com exp5b_transferencia.csv)")
        print(f"(0) {nome}: réplica ok, queda {v1(r['queda'])} {ic_txt(*r['ic'])}", flush=True)

        # (1) referência alternativa (vídeo KSS 5)
        sB_alt = escores_b(X_alt, y, folds, suj, nome)
        sC_alt = final[("principal", nome)].predict_proba(X_alt)[:, 1]
        r = queda_pareada(y, sB_alt, sC_alt, suj, ream_uta)
        rv = queda_video(u, sB_alt, sC_alt, ream_uta)
        reg("referencia_kss5", "ref_video_kss5", nome, "janela", "auc_B", r["auc_B"], r["ic_B"], E.N_BOOT, "AUC x 100",
            "B (UTA→UTA) com a centragem pelo vídeo de pouca vigilância (código 5)")
        reg("referencia_kss5", "ref_video_kss5", nome, "janela", "auc_C", r["auc_C"], r["ic_C"], E.N_BOOT, "AUC x 100",
            "C (DMD→UTA) com a centragem pelo vídeo de pouca vigilância (código 5)")
        reg("referencia_kss5", "ref_video_kss5", nome, "janela", "queda_auc", r["queda"], r["ic"], E.N_BOOT, "pp",
            "queda B − C de AUC por janela com a centragem pelo vídeo de pouca vigilância (bootstrap pareado)")
        reg("referencia_kss5", "ref_video_kss5", nome, "video", "auc_B", rv["auc_B"], unidade="AUC x 100",
            desc="B, AUC por vídeo, centragem pelo vídeo de pouca vigilância")
        reg("referencia_kss5", "ref_video_kss5", nome, "video", "auc_C", rv["auc_C"], unidade="AUC x 100",
            desc="C, AUC por vídeo, centragem pelo vídeo de pouca vigilância")
        reg("referencia_kss5", "ref_video_kss5", nome, "video", "queda_auc", rv["queda"], rv["ic"], E.N_BOOT, "pp",
            "queda B − C de AUC por vídeo com a centragem pelo vídeo de pouca vigilância")
        print(f"(1) {nome}: B {v1(r['auc_B'])}, C {v1(r['auc_C'])}, queda {v1(r['queda'])} {ic_txt(*r['ic'])}; "
              f"vídeo {v1(rv['queda'])} {ic_txt(*rv['ic'])}", flush=True)

        # (2) bootstrap em dois níveis (retreina C com os sujeitos do DMD sorteados)
        rng_dmd = np.random.default_rng(E.SEED + 1)
        idx_dmd = {s: np.flatnonzero(d["sujeito"].to_numpy() == s) for s in suj_dmd}
        Xd, yd = d[cols_c].to_numpy(float), d["y"].to_numpy()
        Xu = u[cols_c].to_numpy(float)
        rB = E.avaliar(y, sB_c[nome], suj, ream_uta, auc_por_sujeito=False)
        quedas = np.full(n_boot2, np.nan)
        for b in range(n_boot2):
            sorteio = rng_dmd.choice(suj_dmd, len(suj_dmd), replace=True)
            ii = np.concatenate([idx_dmd[s] for s in sorteio])
            if len(np.unique(yd[ii])) < 2:
                continue
            sC_b = E.novo_modelo(nome).fit(Xd[ii], yd[ii]).predict_proba(Xu)[:, 1]
            W = ream_uta.pesos(suj, slice(b, b + 1))
            quedas[b] = 100 * (rB["boot"]["auc"][b] - E.auc_ponderada(y, sC_b, W)[0])
        lo, hi, nb = E.ic(quedas)
        r0 = queda_pareada(y, sB_c[nome], sC_c[nome], suj, ream_uta)
        reg("bootstrap_dois_niveis", "ref_video_alerta", nome, "janela", "queda_auc", r0["queda"], (lo, hi), nb, "pp",
            "queda B − C de AUC por janela; IC com sujeitos do DMD (retreino de C) e da UTA-RLDD reamostrados")
        print(f"(2) {nome}: queda {v1(r0['queda'])}, IC em dois níveis {ic_txt(lo, hi)} ({nb} reamostragens)", flush=True)

        # (3) volume do treino: B com 13 sujeitos (e com 510 janelas)
        rng13 = np.random.default_rng(E.SEED)
        Xc = u[cols_c].to_numpy(float)
        for rotulo, n_jan in (("13_sujeitos", None), ("13_sujeitos_510_janelas", len(d))):
            aucs = []
            for _ in range(n_treze):
                s = np.full(len(y), np.nan)
                for k in np.unique(folds):
                    tr_suj = rng13.choice(np.unique(suj[folds != k]), N_SUJ_DMD, replace=False)
                    tr = np.flatnonzero(np.isin(suj, tr_suj))
                    if n_jan:
                        tr = rng13.choice(tr, n_jan, replace=False)
                    te = folds == k
                    s[te] = E.novo_modelo(nome).fit(Xc[tr], y[tr]).predict_proba(Xc[te])[:, 1]
                aucs.append(100 * roc_auc_score(y, s))
            aucs = np.array(aucs)
            faixa = tuple(np.percentile(aucs, [2.5, 97.5]))
            reg("volume_treino", rotulo, nome, "janela", "auc_B", float(np.median(aucs)), faixa, n_treze, "AUC x 100",
                f"B treinado com {N_SUJ_DMD} sujeitos sorteados por fold" + (f" e {n_jan} janelas" if n_jan else "") +
                "; valor = mediana dos sorteios, faixa = percentis 2,5 e 97,5 dos sorteios (não é IC)")
            print(f"(3) {nome} {rotulo}: AUC B mediana {v1(np.median(aucs))} {ic_txt(*faixa)}", flush=True)

        # (4) vazamento no mesmo cenário: folds oficiais × partição aleatória de janelas
        for norm, cols in (("centrada", cols_c), ("bruta", cols_b)):
            X = u[cols].to_numpy(float)
            s_suj = sB_c[nome] if norm == "centrada" else escores_b(X, y, folds, suj, nome)
            ba_s, auc_s = balanced_accuracy_score(y, s_suj >= E.LIMIAR), roc_auc_score(y, s_suj)
            ba_r, auc_r = [], []
            for semente in SEMENTES_VAZAMENTO:
                s = np.full(len(y), np.nan)
                for tr, te in KFold(5, shuffle=True, random_state=semente).split(X):
                    s[te] = E.novo_modelo(nome).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                ba_r.append(balanced_accuracy_score(y, s >= E.LIMIAR))
                auc_r.append(roc_auc_score(y, s))
            for met, sv, rv_ in (("acuracia_balanceada", ba_s, np.mean(ba_r)), ("auc", auc_s, np.mean(auc_r))):
                reg("vazamento_mesmo_cenario", norm, nome, "janela", f"{met}_por_sujeito", 100 * sv, unidade="%",
                    desc=f"B3, janelas de 10 s, folds oficiais por sujeito ({norm})")
                reg("vazamento_mesmo_cenario", norm, nome, "janela", f"{met}_aleatoria", 100 * rv_,
                    n_rep=len(SEMENTES_VAZAMENTO), unidade="%",
                    desc=f"B3, janelas de 10 s, partição aleatória de janelas em 5 partes, média das sementes ({norm})")
                reg("vazamento_mesmo_cenario", norm, nome, "janela", f"{met}_inflacao", 100 * (rv_ - sv),
                    n_rep=len(SEMENTES_VAZAMENTO), unidade="pp",
                    desc=f"inflação pela partição aleatória de janelas: aleatória − por sujeito ({norm})")
            print(f"(4) {nome} {norm}: acur. bal. {v1(100 * ba_s)} → {v1(100 * np.mean(ba_r))} "
                  f"(+{v1(100 * (np.mean(ba_r) - ba_s))} pp); AUC +{v1(100 * (np.mean(auc_r) - auc_s))}", flush=True)

    # (5) ponto de operação da regra (diagnóstico)
    j = pd.read_csv(E.RES / "exp6_noite_janelas.csv", low_memory=False)
    j = j[(j["dataset"] == "uta")].copy()
    j["chave"] = j["arquivo"].astype(str) + "|" + j["janela"].astype(str)
    regra = j.set_index("chave")["regra_dmd"].map(lambda v: str(v).strip().lower() in ("true", "1", "1.0"))
    chave_u = u["id_sinal"].astype(str) + "|" + u["janela"].astype(str)
    assert chave_u.isin(regra.index).all(), "janela do B3 sem a regra no Exp. 6"
    rg = regra.loc[chave_u].to_numpy()
    fa_regra, sens_regra = rg[y == 0].mean(), rg[y == 1].mean()
    reg("ponto_operacao_regra", "regra_dmd", "regra", "janela", "falso_alarme", 100 * fa_regra, unidade="%",
        desc="regra do rótulo do DMD nas janelas KSS 0 do B3")
    reg("ponto_operacao_regra", "regra_dmd", "regra", "janela", "sensibilidade", 100 * sens_regra, unidade="%",
        desc="regra do rótulo do DMD nas janelas KSS 10 do B3")
    escores = {("C", "centrada", n): sC_c[n] for n in E.MODELOS}
    escores.update({("B", "centrada", n): sB_c[n] for n in E.MODELOS})
    for n in E.MODELOS:
        escores[("C", "bruta", n)] = final[("sem_normalizacao", n)].predict_proba(u[cols_b].to_numpy(float))[:, 1]
    for (cen, norm, n), s in escores.items():
        t = np.quantile(s[y == 0], 1 - fa_regra)
        reg("ponto_operacao_regra", f"{cen}_{norm}", n, "janela", "sensibilidade_no_fa_da_regra",
            100 * (s[y == 1] >= t).mean(), unidade="%",
            desc=f"{cen} ({norm}): sensibilidade com o limiar que dá o falso alarme da regra (diagnóstico)")
    print(f"(5) regra: falso alarme {v1(100 * fa_regra)}%, sensibilidade {v1(100 * sens_regra)}%", flush=True)

    if args.rapido:
        print(f"--rapido: nada gravado ({time.time() - t0:.0f} s)")
        return 0
    df = pd.DataFrame(linhas)
    df.to_csv(CSV_OUT, index=False)
    escrever_md(df, execucao)
    print(f"-> {CSV_OUT.relative_to(ROOT)}, {MD_OUT.relative_to(ROOT)} ({time.time() - t0:.0f} s)")
    return 0


def escrever_md(df: pd.DataFrame, execucao: str) -> None:
    def g(analise, variante, modelo, nivel, metrica):
        r = df[(df.analise == analise) & (df.variante == variante) & (df.modelo == modelo) & (df.nivel == nivel) &
               (df.metrica == metrica)]
        assert len(r) == 1, (analise, variante, modelo, nivel, metrica)
        return r.iloc[0]

    def com_ic(r):
        return f"{v1(r.valor)} {ic_txt(r.ic95_inf, r.ic95_sup)}" if np.isfinite(r.ic95_inf) else v1(r.valor)

    nomes = {"RF": "RF", "LR": "RL"}
    L = ["# Exp. 5b: robustez da queda B − C", "",
         f"Execução: {execucao}. Gerado por `src/experiments/exp5b_robustez.py` (mesmos dados, receita de modelos, "
         "modelos finais de C e reamostragem de sujeitos do Exp. 5b). Esquema B3 (alerta × sonolento), janelas de 10 s. "
         "AUC × 100; quedas em pontos de AUC; IC 95% por bootstrap percentil de sujeitos (1.000 reamostragens).", "",
         "## 1. Referência alternativa na UTA-RLDD (vídeo de pouca vigilância)", "",
         "Na rodada principal, a referência da centragem na UTA-RLDD é o último terço do vídeo alerta, e as janelas "
         "alertas avaliadas saem do mesmo vídeo. Aqui a referência é a média das janelas do vídeo de pouca vigilância do "
         "sujeito (código 5, KSS 6 e 7), que não entra no B3: nem as janelas alertas nem as sonolentas vêm do vídeo da referência. O DMD não muda.", "",
         "| Modelo | Referência | AUC B | AUC C | Queda B − C (janela) | Queda B − C (vídeo) |", "|---|---|---|---|---|---|"]
    csv_5b = pd.read_csv(E.CSV_RES, low_memory=False)

    def principal(nome, exp, nivel, metrica="auc"):
        r = csv_5b[(csv_5b.experimento == exp) & (csv_5b.variante == "principal") & (csv_5b.modelo == nome) &
                   (csv_5b.esquema == "B3") & (csv_5b.nivel == nivel) & (csv_5b.estrato == "todos") &
                   (csv_5b.limiar_tipo == "fixo_0.5") & (csv_5b.metrica == metrica)].iloc[0]
        return r

    for nome in E.MODELOS:
        qj, qv = principal(nome, "queda_B_menos_C", "janela"), principal(nome, "queda_B_menos_C", "video")
        L.append(f"| {nomes[nome]} | último terço do vídeo alerta (principal) | {v1(100 * principal(nome, 'B_uta_intra', 'janela').valor)} | "
                 f"{v1(100 * principal(nome, 'C_dmd_para_uta', 'janela').valor)} | {v1(qj.valor)} {ic_txt(qj.ic95_inf, qj.ic95_sup)} | "
                 f"{v1(qv.valor)} {ic_txt(qv.ic95_inf, qv.ic95_sup)} |")
        L.append(f"| {nomes[nome]} | vídeo de pouca vigilância | {v1(g('referencia_kss5', 'ref_video_kss5', nome, 'janela', 'auc_B').valor)} | "
                 f"{v1(g('referencia_kss5', 'ref_video_kss5', nome, 'janela', 'auc_C').valor)} | "
                 f"{com_ic(g('referencia_kss5', 'ref_video_kss5', nome, 'janela', 'queda_auc'))} | "
                 f"{com_ic(g('referencia_kss5', 'ref_video_kss5', nome, 'video', 'queda_auc'))} |")
    L += ["", "Leitura: com o RF, a queda depende da referência; com a RL, ela se mantém. O RF treinado na UTA-RLDD "
          "aproveita a semelhança entre as janelas alertas e a referência tirada do mesmo vídeo; o modelo do DMD não "
          "tem como aprender essa estrutura. Ressalva: o limiar de EAR do detector de piscadas continua o do vídeo alerta.",
          "", "## 2. Bootstrap em dois níveis", "",
          "Em cada reamostragem, os 13 sujeitos do DMD são sorteados com reposição e C é treinado de novo; os sujeitos da "
          "UTA-RLDD seguem a reamostragem pareada do Exp. 5b.", "",
          "| Modelo | Queda B − C (janela) | IC só UTA-RLDD (Exp. 5b) | IC em dois níveis |", "|---|---|---|---|"]
    for nome in E.MODELOS:
        qj = principal(nome, "queda_B_menos_C", "janela")
        b2 = g("bootstrap_dois_niveis", "ref_video_alerta", nome, "janela", "queda_auc")
        L.append(f"| {nomes[nome]} | {v1(qj.valor)} | {ic_txt(qj.ic95_inf, qj.ic95_sup)} | {ic_txt(b2.ic95_inf, b2.ic95_sup)} |")
    L += ["", "## 3. Volume do treino de B", "",
          f"B treinado com só {N_SUJ_DMD} sujeitos da UTA-RLDD por fold (sorteados), com todas as janelas deles ou com "
          "510 janelas (o tamanho do treino de C), em "
          f"{int(g('volume_treino', '13_sujeitos', 'RF', 'janela', 'auc_B').n_rep)} sorteios. Valor = mediana dos sorteios; "
          "entre colchetes, os percentis 2,5 e 97,5 dos sorteios (variação do sorteio, não IC).", "",
          "| Modelo | B completo | B com 13 sujeitos | B com 13 sujeitos e 510 janelas | C |", "|---|---|---|---|---|"]
    for nome in E.MODELOS:
        L.append(f"| {nomes[nome]} | {v1(100 * principal(nome, 'B_uta_intra', 'janela').valor)} | "
                 f"{com_ic(g('volume_treino', '13_sujeitos', nome, 'janela', 'auc_B'))} | "
                 f"{com_ic(g('volume_treino', '13_sujeitos_510_janelas', nome, 'janela', 'auc_B'))} | "
                 f"{v1(100 * principal(nome, 'C_dmd_para_uta', 'janela').valor)} |")
    L += ["", "## 4. Vazamento no mesmo cenário", "",
          "Folds oficiais por sujeito contra partição aleatória das janelas em 5 partes (média das sementes 42, 43 e 44). "
          "Janelas vizinhas do mesmo vídeo (passo de 5 s, sobreposição de 5 s) caem nos dois lados.", "",
          "| Modelo | Características | Acur. bal. por sujeito | Acur. bal. aleatória | Inflação (pp) | AUC por sujeito | AUC aleatória | Inflação da AUC |",
          "|---|---|---|---|---|---|---|---|"]
    for nome in E.MODELOS:
        for norm in ("centrada", "bruta"):
            def m(met):
                return v1(g("vazamento_mesmo_cenario", norm, nome, "janela", met).valor)
            L.append(f"| {nomes[nome]} | {norm} | {m('acuracia_balanceada_por_sujeito')} | {m('acuracia_balanceada_aleatoria')} | "
                     f"{m('acuracia_balanceada_inflacao')} | {m('auc_por_sujeito')} | {m('auc_aleatoria')} | {m('auc_inflacao')} |")
    qb = principal("RF", "queda_B_menos_C", "janela", "acuracia_balanceada")
    L += ["", f"Para comparar: a queda B − C da acurácia balanceada do RF (limiar 0,5) é {v1(qb.valor)} "
          f"{ic_txt(qb.ic95_inf, qb.ic95_sup)} pp.", "",
          "## 5. Ponto de operação da regra (diagnóstico)", "",
          "Regra do rótulo do DMD aplicada aos nossos sinais (fechamento de 0,5 s ou mais ou bocejo). Para cada modelo, o "
          "limiar é o que dá o mesmo falso alarme da regra nas janelas alertas (KSS 0) do B3. Esse limiar usa o rótulo do "
          "teste: a tabela só diz onde a regra fica em relação às curvas ROC dos modelos.", "",
          f"Regra: falso alarme {v1(g('ponto_operacao_regra', 'regra_dmd', 'regra', 'janela', 'falso_alarme').valor)}%, "
          f"sensibilidade {v1(g('ponto_operacao_regra', 'regra_dmd', 'regra', 'janela', 'sensibilidade').valor)}%.", "",
          "| Modelo | Sensibilidade no falso alarme da regra (%) |", "|---|---|"]
    for cen, norm, rot in (("C", "bruta", "C (DMD→UTA), características brutas"),
                           ("C", "centrada", "C (DMD→UTA), centradas"),
                           ("B", "centrada", "B (UTA→UTA), centradas")):
        for nome in E.MODELOS:
            r = g("ponto_operacao_regra", f"{cen}_{norm}", nome, "janela", "sensibilidade_no_fa_da_regra")
            L.append(f"| {nomes[nome]}, {rot} | {v1(r.valor)} |")
    MD_OUT.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
