#!/usr/bin/env python3
"""Números do artigo e do pôster gerados a partir de results/ (nenhum número digitado à mão).

Lê os CSV/JSON/MD FINAIS de results/ (e models/*.json) e escreve:

  paper/numeros.tex          \\newcommand de cada número (vírgula decimal, IC e ± quando houver);
                             incluir no artigo com \\input{numeros}
  poster/numeros.json        os mesmos valores (texto Unicode) para poster/build_poster.py, com a
                             especificação da origem de cada um (usada pela conferência)
  results/numeros_resumo.md  tabela nome -> valor -> arquivo de origem -> coluna/filtro; a seção
                             "Conferência automática" é preenchida por src/experiments/conferir_numeros.py

Convenções dos macros (nomes só com letras; números por extenso: Um, Dois, Tres, ...):
  \\Nome        valor (percentuais SEM o símbolo %, em pontos percentuais quando é diferença)
  \\NomeIC      "[inf; sup]" (IC 95% do arquivo de origem: bootstrap por sujeito/vídeo)
  \\NomeComIC   "valor [inf; sup]"
  \\NomeDP      desvio-padrão entre folds (mesmo ddof do .md de origem)
  \\NomeMDP     "média ± dp"
  Negativos saem com o sinal de menos tipográfico ($-$ no LaTeX, "−" no JSON).
  Aparelhos: \\CelularIPhone... (iPhone 17 Pro medido); \\CelularSamsung... só se houver dados do
  Galaxy S23 Ultra em results/exp4_phone_latency.csv; senão \\SamsungPendente = "medição prevista".
  \\ifSamsungMedido ... \\else ... \\fi existe sempre (para o texto do artigo não quebrar).

Uso:  .venv/bin/python src/experiments/make_paper_numbers.py
      (depois: .venv/bin/python src/experiments/conferir_numeros.py)
Antes, se houver benchmark novo do celular: python tools/import_phone_benchmark.py results/phone
(ou tudo de uma vez: bash tools/atualizar_celular.sh).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT_TEX = ROOT / "paper" / "numeros.tex"
OUT_JSON = ROOT / "poster" / "numeros.json"
OUT_MD = RES / "numeros_resumo.md"

ENTRIES: list[dict] = []
_CACHE: dict[str, pd.DataFrame] = {}


# ----------------------------------------------------------------------------------------------
# Formatação pt-BR
# ----------------------------------------------------------------------------------------------
def fmt_abs(x: float, casas: int) -> str:
    s = f"{abs(x):,.{casas}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_num(x: float, casas: int, tex: bool) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        raise ValueError("valor ausente")
    r = round(float(x), casas)
    neg = r < 0 and fmt_abs(r, casas).strip("0,.") != ""
    s = fmt_abs(r, casas)
    if neg:
        return ("$-$" if tex else "−") + s
    return s


def tex_escape(s: str) -> str:
    rep = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
           "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\^{}", "–": "--", "−": "$-$"}
    return "".join(rep.get(c, c) for c in s)


# ----------------------------------------------------------------------------------------------
# Leitura com seleção estrita (exatamente as linhas pedidas)
# ----------------------------------------------------------------------------------------------
def csv(rel: str) -> pd.DataFrame:
    if rel not in _CACHE:
        _CACHE[rel] = pd.read_csv(ROOT / rel, dtype=str, keep_default_na=False)
    return _CACHE[rel]


def select(rel: str, filtros: dict) -> pd.DataFrame:
    df = csv(rel)
    m = pd.Series(True, index=df.index)
    for col, val in filtros.items():
        if col not in df.columns:
            raise KeyError(f"{rel}: coluna {col!r} não existe")
        if isinstance(val, (list, tuple)):
            m &= df[col].isin([str(v) for v in val])
        else:
            m &= df[col] == str(val)
    return df[m]


def num(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="raise").to_numpy(float)


AGG = {
    None: None,
    "mean": lambda v: float(np.mean(v)),
    "median": lambda v: float(np.median(v)),
    "std0": lambda v: float(np.std(v, ddof=0)),
    "std1": lambda v: float(np.std(v, ddof=1)),
    "min": lambda v: float(np.min(v)),
    "max": lambda v: float(np.max(v)),
    "sum": lambda v: float(np.sum(v)),
    "count": lambda v: float(len(v)),
}


def eval_csv(spec: dict):
    sel = select(spec["arquivo"], spec["filtros"])
    n_esp = spec.get("n_linhas")
    if spec.get("agg") is None and len(sel) != 1:
        raise ValueError(f"{spec['arquivo']} {spec['filtros']}: {len(sel)} linhas (esperada 1)")
    if n_esp is not None and len(sel) != n_esp:
        raise ValueError(f"{spec['arquivo']} {spec['filtros']}: {len(sel)} linhas (esperadas {n_esp})")
    if len(sel) == 0:
        raise ValueError(f"{spec['arquivo']} {spec['filtros']}: nenhuma linha")
    if spec.get("agg") == "count":
        val = float(len(sel))
    else:
        v = num(sel[spec["coluna"]])
        val = float(v[0]) if spec.get("agg") is None else AGG[spec["agg"]](v)
    ic = None
    if spec.get("ic"):
        lo, hi = spec["ic"]
        ic = (float(num(sel[lo])[0]), float(num(sel[hi])[0]))
    dp = AGG[spec["dp"]](num(sel[spec["coluna"]])) if spec.get("dp") else None
    return val, ic, dp


def eval_json(spec: dict):
    d = json.loads((ROOT / spec["arquivo"]).read_text(encoding="utf-8"))
    for k in spec["chave"].split("."):
        d = d[k]
    return (d if isinstance(d, str) else float(d)), None, None


def eval_md(spec: dict):
    txt = (ROOT / spec["arquivo"]).read_text(encoding="utf-8")
    m = re.search(spec["regex"], txt)
    if not m:
        raise ValueError(f"{spec['arquivo']}: regex não encontrada: {spec['regex']}")
    g = m.group(spec.get("grupo", 1))
    if spec.get("texto"):
        return g, None, None
    return float(g.replace(".", "").replace(",", ".") if spec.get("decimal_virgula") else g), None, None


def by_name(nome: str) -> dict:
    for e in ENTRIES:
        if e["nome"] == nome:
            return e
    raise KeyError(nome)


def eval_derivado(spec: dict):
    env = {n.replace("-", "_"): by_name(n)["bruto"] for n in spec["deps"]}
    return float(eval(spec["expr"], {"__builtins__": {}, "abs": abs, "max": max, "min": min}, env)), None, None


def eval_csv_texto(spec: dict):
    sel = select(spec["arquivo"], spec["filtros"])
    if len(sel) == 0:
        raise ValueError(f"{spec['arquivo']} {spec['filtros']}: nenhuma linha")
    return ", ".join(sorted(set(sel[spec["coluna"]]))), None, None


EVAL = {"csv_texto": eval_csv_texto, "csv": eval_csv, "json": eval_json, "md": eval_md, "derivado": eval_derivado}


def describe(spec: dict) -> tuple[str, str]:
    """(arquivo, coluna/filtro) legíveis para o resumo."""
    t = spec["tipo"]
    if t == "csv":
        f = ", ".join(f"{k}={v if not isinstance(v, (list, tuple)) else '{' + ','.join(map(str, v)) + '}'}"
                      for k, v in spec["filtros"].items())
        agg = spec.get("agg")
        col = spec.get("coluna", "(linhas)")
        s = f"`{col}`" + (f" [{agg}]" if agg else "") + (f" onde {f}" if f else "")
        if spec.get("ic"):
            s += f"; IC `{spec['ic'][0]}`/`{spec['ic'][1]}`"
        if spec.get("dp"):
            s += f"; dp {spec['dp']}"
        return spec["arquivo"], s
    if t == "csv_texto":
        f = ", ".join(f"{k}={v}" for k, v in spec["filtros"].items())
        return spec["arquivo"], f"`{spec['coluna']}` (valores distintos) onde {f}"
    if t == "json":
        return spec["arquivo"], f"chave `{spec['chave']}`"
    if t == "md":
        return spec["arquivo"], f"regex `{spec['regex']}` (grupo {spec.get('grupo', 1)})"
    return "(derivado)", f"`{spec['expr']}` com " + ", ".join(spec["deps"])


def add(nome: str, spec: dict, *, casas: int = 1, escala: float = 1.0, unidade: str = "",
        descricao: str = "", evidencia: str | None = None, texto: bool = False) -> dict:
    if not re.fullmatch(r"[A-Za-z]+", nome):
        raise ValueError(f"nome de macro inválido (só letras): {nome}")
    if any(e["nome"] == nome for e in ENTRIES):
        raise ValueError(f"macro repetido: {nome}")
    val, ic, dp = EVAL[spec["tipo"]](spec)
    e = {"nome": nome, "spec": spec, "casas": casas, "escala": escala, "unidade": unidade,
         "descricao": descricao, "evidencia": evidencia, "texto": texto or isinstance(val, str),
         "bruto": val, "ic_bruto": list(ic) if ic else None, "dp_bruto": dp}
    ENTRIES.append(e)
    return e


def add_texto(nome: str, valor: str, fonte: str, descricao: str, evidencia: str | None = None):
    """Texto fixo que é FATO de um arquivo (ex.: nome do aparelho); conferido por regex no arquivo."""
    spec = {"tipo": "md", "arquivo": fonte, "regex": "(" + re.escape(valor) + ")", "texto": True}
    return add(nome, spec, descricao=descricao, evidencia=evidencia or fonte, texto=True)


# ----------------------------------------------------------------------------------------------
# Especificações
# ----------------------------------------------------------------------------------------------
PCT = dict(escala=100.0, unidade="%")


def bases():
    b5 = "results/exp5b_transferencia.csv"
    fB = dict(experimento="B_uta_intra", variante="principal", modelo="RF", esquema="B1", nivel="video",
              estrato="todos", limiar_tipo="fixo_0.5", metrica="acuracia")
    add("BaseUTASujeitos", {"tipo": "csv", "arquivo": b5, "filtros": fB, "coluna": "n_sujeitos"}, casas=0,
        descricao="UTA-RLDD: sujeitos (todos os vídeos em disco)", evidencia="results/exp5b_transferencia.md")
    add("BaseUTAVideos", {"tipo": "csv", "arquivo": b5, "filtros": fB, "coluna": "n_unidades"}, casas=0,
        descricao="UTA-RLDD: vídeos lógicos", evidencia="results/exp5b_transferencia.md")
    add("BaseUTAArquivos", {"tipo": "md", "arquivo": "results/exp5b_transferencia.md",
                            "regex": r"180 vídeos lógicos \((\d+) arquivos\)"}, casas=0,
        descricao="UTA-RLDD: arquivos de vídeo (dois vídeos sonolentos vêm em 2 partes)")
    rx = r"Pool: (\d+) sequências de teste dos 5 folds oficiais, (\d+) vídeos, (\d+) sujeitos"
    add("BaseUTASequencias", {"tipo": "md", "arquivo": "results/exp2_leakage.md", "regex": rx, "grupo": 1}, casas=0,
        descricao="UTA-RLDD: sequências de 30 piscadas (características oficiais, 5 folds)")
    add("BaseUTASujeitosOficiais", {"tipo": "md", "arquivo": "results/exp2_leakage.md", "regex": rx, "grupo": 3},
        casas=0, descricao="UTA-RLDD: sujeitos nos .npy oficiais")
    add("BaseUTAVideosOficiais", {"tipo": "md", "arquivo": "results/exp2_leakage.md", "regex": rx, "grupo": 2},
        casas=0, descricao="UTA-RLDD: vídeos nos .npy oficiais")
    fBj = dict(fB, esquema="B3", nivel="janela")
    add("BaseUTAJanelasBTres", {"tipo": "csv", "arquivo": b5, "filtros": fBj, "coluna": "n_unidades"}, casas=0,
        descricao="UTA-RLDD: janelas de 10 s avaliáveis, B3", evidencia="results/exp5b_transferencia.md")
    add("BaseUTAJanelasSonolentas", {"tipo": "csv", "arquivo": b5, "filtros": fBj, "coluna": "n_pos"}, casas=0,
        descricao="UTA-RLDD: janelas KSS 10", evidencia="results/exp5b_transferencia.md")
    add("BaseUTAJanelasAlerta", {"tipo": "csv", "arquivo": b5, "filtros": fBj, "coluna": "n_neg"}, casas=0,
        descricao="UTA-RLDD: janelas KSS 0 avaliáveis", evidencia="results/exp5b_transferencia.md")
    stats = "results/exp3_dataset_stats.md"
    add("BaseMRLImagens", {"tipo": "md", "arquivo": stats, "regex": r"Imagens: \*\*(\d+)\*\*"}, casas=0,
        descricao="MRL Eye: imagens")
    add("BaseMRLSujeitos", {"tipo": "md", "arquivo": stats, "regex": r"sujeitos: \*\*(\d+)\*\*"}, casas=0,
        descricao="MRL Eye: sujeitos")
    add("BaseCEWImagens", {"tipo": "csv", "arquivo": "results/exp3_eye_state.csv",
                           "filtros": dict(protocol="B_final_mrl_to_cew", test_on="CEW_patch", fold="final", stratum="todos", metric="n"),
                           "coluna": "value"}, casas=0, descricao="CEW: recortes de olho",
        evidencia="results/exp3_eye_state.md")
    fA = dict(experimento="A_dmd_intra", variante="principal", modelo="RF", esquema="DMD", nivel="janela",
              estrato="todos", limiar_tipo="fixo_0.5", metrica="acuracia")
    add("BaseDMDSujeitos", {"tipo": "csv", "arquivo": b5, "filtros": fA, "coluna": "n_sujeitos"}, casas=0,
        descricao="DMD: sujeitos com sessão de sonolência (s5)", evidencia="results/exp5b_transferencia.md")
    add("BaseDMDJanelas", {"tipo": "csv", "arquivo": b5, "filtros": fA, "coluna": "n_unidades"}, casas=0,
        descricao="DMD: janelas de 10 s do Exp. 5b (principal)", evidencia="results/exp5b_transferencia.md")
    add("BaseDMDJanelasSonolentas", {"tipo": "csv", "arquivo": b5, "filtros": fA, "coluna": "n_pos"}, casas=0,
        descricao="DMD: janelas sonolentas (s5 com fechamento >= 0,5 s ou bocejo)",
        evidencia="results/exp5b_transferencia.md")
    add("BaseDMDJanelasAlerta", {"tipo": "csv", "arquivo": b5, "filtros": fA, "coluna": "n_neg"}, casas=0,
        descricao="DMD: janelas alerta (safe_drive da s2)", evidencia="results/exp5b_transferencia.md")
    rx6 = r"DMD: (\d+) sessões de (\d+) sujeitos \(s5 (\d+), s1 (\d+), s2 (\d+), s3 (\d+)\)"
    add("BaseDMDSessoesSonolencia", {"tipo": "md", "arquivo": "results/exp6_noite.md", "regex": rx6, "grupo": 3},
        casas=0, descricao="DMD: sessões de sonolência s5")
    add("BaseDMDSessoesDirecaoReal", {"tipo": "md", "arquivo": "results/exp6_noite.md", "regex": rx6, "grupo": 4},
        casas=0, descricao="DMD: sessões s1 (carro em movimento, direção real)")
    add("BaseDMDSessoes", {"tipo": "md", "arquivo": "results/exp6_noite.md", "regex": rx6, "grupo": 1},
        casas=0, descricao="DMD: sessões usadas (s1, s2, s3, s5)")
    add("BaseDMDSujeitosTodos", {"tipo": "md", "arquivo": "results/exp6_noite.md", "regex": rx6, "grupo": 2},
        casas=0, descricao="DMD: sujeitos baixados (13 com s5 + gE_28)")
    e6 = "results/exp6_noite.csv"
    for nome, grupo, d in [("BaseNITYMEDVideos", "nitymed|todos", "NITYMED: vídeos em disco"),
                           ("BaseNITYMEDMicrossono", "nitymed|Microsleep", "NITYMED: vídeos de microssono"),
                           ("BaseNITYMEDBocejo", "nitymed|Yawning", "NITYMED: vídeos de bocejo")]:
        add(nome, {"tipo": "csv", "arquivo": e6, "filtros": dict(secao="a", grupo=grupo,
                                                                   metrica="taxa_rosto_media_por_video"),
                   "coluna": "n_videos"}, casas=0, descricao=d, evidencia="results/exp6_noite.md")
    add("BaseNITYMEDDeclarados", {"tipo": "md", "arquivo": "results/exp6_nitymed_ficha.md",
                                  "regex": r"Conteúdo declarado \| (\d+) vídeos"}, casas=0,
        descricao="NITYMED: vídeos declarados na página oficial")
    rxn = r"NITYMED: (\d+) vídeos, (\d+) janelas, (\d+) avaliáveis"
    add("BaseNITYMEDJanelas", {"tipo": "md", "arquivo": "results/exp6_noite.md", "regex": rxn, "grupo": 2},
        casas=0, descricao="NITYMED: janelas de 10 s")
    add("BaseNITYMEDJanelasAvaliaveis", {"tipo": "md", "arquivo": "results/exp6_noite.md", "regex": rxn, "grupo": 3},
        casas=0, descricao="NITYMED: janelas avaliáveis (fora do 1º terço)")


def exp1():
    f = "results/exp1_intra.csv"
    md = "results/exp1_intra.md"
    modelos = [("RandomForest", "RF"), ("LSTM", "LSTM"), ("CNN1D", "CNN"), ("SVM-RBF", "SVM"), ("LogReg", "LR")]
    metricas = [("bsa", "BSA"), ("bal_acc", "AcBal"), ("f1_macro", "FMacro"), ("auc_macro", "AUC"), ("va", "VA")]
    for scheme, tag in [("3c", ""), ("B3", "BTres"), ("B1", "BUm"), ("B2", "BDois")]:
        for model, mt in modelos:
            for metric, nt in metricas:
                spec = {"tipo": "csv", "arquivo": f, "coluna": "value", "agg": "mean", "dp": "std0", "n_linhas": 5,
                        "filtros": dict(model=model, scheme=scheme, metric=metric, fold=[1, 2, 3, 4, 5])}
                add(f"ExpUm{mt}{tag}{nt}", spec, **PCT, evidencia=md,
                    descricao=f"Exp. 1 {model}, esquema {scheme}, {metric} (média ± dp, ddof=0, 5 folds)")
        for metric, nt in [("bsa", "BSA"), ("va", "VA")]:
            spec = {"tipo": "csv", "arquivo": f, "coluna": "value", "agg": "mean", "dp": "std0", "n_linhas": 5,
                    "filtros": dict(model="Chance(majoritaria)", scheme=scheme, metric=metric, fold=[1, 2, 3, 4, 5])}
            add(f"ExpUmChance{tag}{nt}", spec, **PCT, evidencia=md,
                descricao=f"Exp. 1 chance (classe majoritária), esquema {scheme}, {metric}")
    rx = (r"HM-LSTM: BSA ([\d.]+)% / VA ([\d.]+)%; LSTM: BSA ([\d.]+)% / VA ([\d.]+)%; "
          r"Camadas densas: BSA ([\d.]+)% / VA ([\d.]+)%; Humanos: BSA — / VA ([\d.]+)%")
    for nome, g, d in [("ExpUmHMLSTMBSA", 1, "HM-LSTM (Ghoddoosian et al., 2019), BSA"),
                       ("ExpUmHMLSTMVA", 2, "HM-LSTM (Ghoddoosian et al., 2019), VA"),
                       ("ExpUmRefLSTMBSA", 3, "LSTM do artigo original, BSA"),
                       ("ExpUmRefLSTMVA", 4, "LSTM do artigo original, VA"),
                       ("ExpUmRefDensasBSA", 5, "camadas densas do artigo original, BSA"),
                       ("ExpUmRefDensasVA", 6, "camadas densas do artigo original, VA"),
                       ("ExpUmHumanosVA", 7, "julgamento humano (artigo original), VA")]:
        add(nome, {"tipo": "md", "arquivo": md, "regex": rx, "grupo": g}, casas=1, unidade="%", descricao=d)
    add("ExpUmDifHMLSTMRFVA", {"tipo": "derivado", "expr": "ExpUmHMLSTMVA - ExpUmRFVA*100",
                               "deps": ["ExpUmHMLSTMVA", "ExpUmRFVA"]}, casas=1, unidade="pp",
        descricao="HM-LSTM − nosso RF, VA 3 classes (pp)")
    add("ExpUmDifRFHumanosVA", {"tipo": "derivado", "expr": "ExpUmRFVA*100 - ExpUmHumanosVA",
                                "deps": ["ExpUmHumanosVA", "ExpUmRFVA"]}, casas=1, unidade="pp",
        descricao="nosso RF − humanos, VA 3 classes (pp)")
    bj = "models/blink_temporal.json"
    add("ExpUmLSTMParametros", {"tipo": "json", "arquivo": bj, "chave": "treino.parametros"}, casas=0,
        descricao="LSTM exportado: parâmetros", evidencia=md)
    add("ExpUmLSTMKB", {"tipo": "json", "arquivo": bj, "chave": "tamanho_kb"}, casas=1, unidade="KB",
        descricao="LSTM .tflite: tamanho", evidencia=md)
    add("ExpUmLSTMConcordancia", {"tipo": "json", "arquivo": bj, "chave": "validacao_tflite_vs_keras.concordancia_argmax"},
        casas=1, **PCT, descricao="LSTM TFLite x Keras: concordância do argmax")
    add("ExpUmLSTMPcinquentaMac", {"tipo": "json", "arquivo": bj, "chave": "latencia_mac_cpu_ms.p50"}, casas=3,
        unidade="ms", descricao="LSTM TFLite no Mac (CPU, lote 1): p50", evidencia=md)
    add("ExpUmLSTMPnoventaecincoMac", {"tipo": "json", "arquivo": bj, "chave": "latencia_mac_cpu_ms.p95"}, casas=3,
        unidade="ms", descricao="LSTM TFLite no Mac (CPU, lote 1): p95", evidencia=md)


def exp2():
    f = "results/exp2_leakage_plot.csv"
    md = "results/exp2_leakage.md"
    for model, mt in [("RandomForest", "RF"), ("LSTM", "LSTM")]:
        for proto, pt in [("janela", "Janela"), ("video", "Video"), ("sujeito", "Sujeito")]:
            for metric, nt in [("bsa", "BSA"), ("bal_acc", "AcBal"), ("auc_macro", "AUC"), ("va", "VA")]:
                if proto == "janela" and metric == "va":
                    continue
                spec = {"tipo": "csv", "arquivo": f, "coluna": "mean",
                        "filtros": dict(protocol=proto, model=model, metric=metric)}
                add(f"ExpDois{mt}{pt}{nt}", spec, **PCT, evidencia=md,
                    descricao=f"Exp. 2 {model}, partição por {proto}, {metric} (média)")
        for proto, pt in [("janela", "Janela"), ("video", "Video")]:
            for metric, nt in [("bsa", "BSA"), ("bal_acc", "AcBal"), ("auc_macro", "AUC"), ("va", "VA")]:
                if proto == "janela" and metric == "va":
                    continue
                a, b = f"ExpDois{mt}{pt}{nt}", f"ExpDois{mt}Sujeito{nt}"
                add(f"ExpDois{mt}Inflacao{pt}{nt}", {"tipo": "derivado", "expr": f"({a} - {b})*100", "deps": [a, b]},
                    casas=1, unidade="pp", evidencia=md,
                    descricao=f"Exp. 2 {model}: {proto} − sujeito, {metric} (pp; positivo = inflado)")
    add("ExpDoisVideosVistosJanela", {"tipo": "csv", "arquivo": f, "coluna": "mean",
                                      "filtros": dict(protocol="janela", model="RandomForest", metric="leak_videos")},
        **PCT, evidencia=md, descricao="Exp. 2 partição por janela: vídeos do teste vistos no treino")


def exp3():
    f = "results/exp3_eye_state.csv"
    s = "results/exp3_strat_conditions.csv"
    md = "results/exp3_eye_state.md"
    folds = [1, 2, 3, 4, 5]

    def fold_mean(nome, protocol, test_on, metric, d):
        spec = {"tipo": "csv", "arquivo": f, "coluna": "value", "agg": "mean", "dp": "std1", "n_linhas": 5,
                "filtros": dict(protocol=protocol, test_on=test_on, metric=metric, fold=folds, stratum="todos")}
        add(nome, spec, **PCT, evidencia=md, descricao=d + " (média ± dp, ddof=1, 5 folds)")

    for metric, nt in [("acc", "Ac"), ("bal_acc", "AcBal"), ("auc", "AUC")]:
        fold_mean(f"ExpTresIntraMRL{nt}", "A_intra_mrl", "MRL", metric, f"Exp. 3 A intra-MRL, {metric}")
        fold_mean(f"ExpTresMRLCEWPatch{nt}", "B_folds_mrl_to_cew", "CEW_patch", metric,
                  f"Exp. 3 B MRL→CEW patch (modelos dos folds), {metric}")
        fold_mean(f"ExpTresMRLCEWCrop{nt}", "B_folds_mrl_to_cew", "CEW_crop", metric,
                  f"Exp. 3 B MRL→CEW crop (modelos dos folds), {metric}")
        fold_mean(f"ExpTresCEWPatchMRL{nt}", "C_cew_patch_to_mrl", "MRL", metric, f"Exp. 3 C CEW patch→MRL, {metric}")
        fold_mean(f"ExpTresCEWCropMRL{nt}", "C_cew_crop_to_mrl", "MRL", metric, f"Exp. 3 C CEW crop→MRL, {metric}")
        fold_mean(f"ExpTresIntraCEWPatch{nt}", "D_intra_cew_patch", "CEW_patch", metric,
                  f"Exp. 3 D intra-CEW patch, {metric}")
        fold_mean(f"ExpTresIntraCEWCrop{nt}", "D_intra_cew_crop", "CEW_crop", metric,
                  f"Exp. 3 D intra-CEW crop, {metric}")
        for test_on, tt in [("CEW_patch", "Patch"), ("CEW_crop", "Crop")]:
            add(f"ExpTresFinalMRLCEW{tt}{nt}",
                {"tipo": "csv", "arquivo": f, "coluna": "value",
                 "filtros": dict(protocol="B_final_mrl_to_cew", test_on=test_on, fold="final", stratum="todos", metric=metric)},
                **PCT, evidencia=md, descricao=f"Exp. 3 B modelo final (37 sujeitos) → {test_on}, {metric}")
    add("ExpTresQuedaMRLCEWPatch", {"tipo": "derivado", "expr": "(ExpTresIntraMRLAc - ExpTresMRLCEWPatchAc)*100",
                                    "deps": ["ExpTresIntraMRLAc", "ExpTresMRLCEWPatchAc"]}, casas=1, unidade="pp",
        descricao="Exp. 3 queda de acurácia intra-MRL − MRL→CEW patch (pp)")
    add("ExpTresQuedaCEWMRLPatch", {"tipo": "derivado", "expr": "(ExpTresIntraCEWPatchAc - ExpTresCEWPatchMRLAc)*100",
                                    "deps": ["ExpTresIntraCEWPatchAc", "ExpTresCEWPatchMRLAc"]}, casas=1, unidade="pp",
        descricao="Exp. 3 queda de acurácia intra-CEW patch − CEW patch→MRL (pp)")
    # óculos
    for lvl, lt in [("com óculos", "ComOculos"), ("sem óculos", "SemOculos")]:
        for metric, nt in [("acc", "Ac"), ("bal_acc", "AcBal"), ("auc", "AUC")]:
            add(f"ExpTresIntraMRL{lt}{nt}",
                {"tipo": "csv", "arquivo": s, "coluna": "value",
                 "filtros": dict(protocol="A_intra_mrl", fold="oof", stratum="glasses", level=lvl, metric=metric)},
                **PCT, evidencia=md, descricao=f"Exp. 3 A intra-MRL {lvl} (out-of-fold agregado), {metric}")
            add(f"ExpTresCEWPatchMRL{lt}{nt}",
                {"tipo": "csv", "arquivo": s, "coluna": "value", "agg": "mean", "dp": "std1", "n_linhas": 5,
                 "filtros": dict(protocol="C_cew_patch_to_mrl", fold=folds, stratum="glasses", level=lvl, metric=metric)},
                **PCT, evidencia=md, descricao=f"Exp. 3 C CEW patch→MRL {lvl}, {metric} (média ± dp, ddof=1)")
    add("ExpTresParametros", {"tipo": "json", "arquivo": "results/exp3_meta.json", "chave": "eyenet_params"}, casas=0,
        descricao="EyeNet: parâmetros", evidencia=md)


def exp4():
    lat = "results/exp4_desktop_latency.csv"
    md = "results/exp3_eye_state.md"
    for modelo, mt in [("int8", "Int"), ("fp32", "Float")]:
        for thr, tt in [("1", ""), ("4", "QuatroThreads")]:
            for col, ct in [("p50_ms", "Pcinquenta"), ("p95_ms", "Pnoventaecinco")]:
                add(f"ExpQuatroCNN{mt}{tt}{ct}", {"tipo": "csv", "arquivo": lat, "coluna": col,
                                                  "filtros": dict(modelo=modelo, threads=thr)},
                    casas=3, unidade="ms", evidencia=md,
                    descricao=f"Exp. 4 CNN do olho {modelo}, {thr} thread(s), desktop, {col}")
        add(f"ExpQuatroCNN{mt}KB", {"tipo": "csv", "arquivo": lat, "coluna": "tamanho_kb",
                                    "filtros": dict(modelo=modelo, threads="1")}, casas=1, unidade="KB",
            evidencia=md, descricao=f"Exp. 4 CNN do olho {modelo}: tamanho do .tflite")
    add("ExpQuatroModelosKB", {"tipo": "derivado", "expr": "ExpQuatroCNNIntKB + ExpUmLSTMKB",
                               "deps": ["ExpQuatroCNNIntKB", "ExpUmLSTMKB"]}, casas=1, unidade="KB",
        descricao="Exp. 4 CNN int8 + LSTM: tamanho total dos modelos aprendidos")
    tf = "results/exp4_tflite_eval.csv"
    names = []
    for ds, dt_ in [("CEW_patch", "CEWPatch"), ("CEW_crop", "CEWCrop"), ("MRL_completo", "MRLCompleto")]:
        for modelo, mt in [("keras", "Keras"), ("int", "Int")]:
            m = "int8" if modelo == "int" else modelo
            nm = f"ExpQuatroAc{dt_}{mt}"
            add(nm, {"tipo": "csv", "arquivo": tf, "coluna": "value", "filtros": dict(dataset=ds, modelo=m, metric="acc")},
                **PCT, evidencia=md, descricao=f"Exp. 4 acurácia {m} em {ds}")
            names.append(nm)
    add("ExpQuatroQuedaQuantizacaoMax",
        {"tipo": "derivado",
         "expr": "max(abs(ExpQuatroAcCEWPatchKeras-ExpQuatroAcCEWPatchInt), abs(ExpQuatroAcCEWCropKeras-ExpQuatroAcCEWCropInt), "
                 "abs(ExpQuatroAcMRLCompletoKeras-ExpQuatroAcMRLCompletoInt))*100", "deps": names},
        casas=2, unidade="pp", evidencia=md, descricao="Exp. 4 maior queda de acurácia int8 − Keras (pp)")
    mac = "results/pwa_headless_mac_bench_20260920-190012.json"
    for chave, nm, d in [("latencia_ms.total_por_quadro.p50", "MacPWATotalPcinquenta", "total por quadro p50"),
                         ("latencia_ms.total_por_quadro.p95", "MacPWATotalPnoventaecinco", "total por quadro p95"),
                         ("fps_medio", "MacPWAFPS", "FPS médio")]:
        add(nm, {"tipo": "json", "arquivo": mac, "chave": chave}, casas=1, unidade="ms" if "lat" in chave else "",
            descricao=f"PWA v1.0 no Chrome headless do Mac (20/09, vídeo 320x240): {d}")


DEVICE_SLUGS = [("Galaxy S23 Ultra", "Samsung"), ("iPhone 17 Pro", "IPhone")]


def device_slug(nome: str) -> str:
    for trecho, slug in DEVICE_SLUGS:
        if trecho.lower() in nome.lower():
            return slug
    return "".join(w.capitalize() for w in re.findall(r"[A-Za-z]+", nome)) or "Aparelho"


def exp4_celular():
    f = "results/exp4_phone_latency.csv"
    if not (ROOT / f).exists():
        raise SystemExit("results/exp4_phone_latency.csv não existe: rode tools/import_phone_benchmark.py results/phone")
    df = csv(f)
    aparelhos = sorted(set(df["aparelho"]), key=lambda a: min(df[df.aparelho == a]["arquivo"]))
    tem_samsung = False

    def chave_versao(v: str):
        return tuple(int(x) for x in re.findall(r"\d+", v)) or (0,)

    for ap in aparelhos:
        slug = device_slug(ap)
        tem_samsung |= slug == "Samsung"
        pre = f"Celular{slug}"
        # só as execuções da versão mais nova do app em cada aparelho (as antigas ficam no CSV, fora das médias)
        versao = max(set(df[df.aparelho == ap]["versao_app"]), key=chave_versao)
        base = dict(aparelho=ap, versao_app=versao)
        add(f"{pre}Nome", {"tipo": "csv_texto", "arquivo": f, "filtros": base, "coluna": "aparelho"}, texto=True,
            evidencia="results/exp4_phone_latency.md", descricao=f"Exp. 4 nome do aparelho ({ap})")
        add(f"{pre}Execucoes", {"tipo": "csv", "arquivo": f, "filtros": base, "agg": "count"}, casas=0,
            descricao=f"Exp. 4 {ap}: número de execuções de 60 s")
        sub = df[(df.aparelho == ap) & (df.versao_app == versao)]
        for col, ct in [("versao_app", "VersaoApp"), ("sistema", "Sistema"), ("navegador", "Navegador"),
                        ("delegate_landmarker", "Delegate")]:
            add(f"{pre}{ct}", {"tipo": "csv_texto", "arquivo": f, "filtros": base, "coluna": col}, texto=True,
                evidencia="results/exp4_phone_latency.md",
                descricao=f"Exp. 4 {ap}: {col} (valores distintos entre as execuções, em ordem)")
        for st, stt in [("total_ms", "Total"), ("landmarker_ms", "Landmarker"), ("cnn_ms", "CNN"),
                        ("engine_ms", "LSTM"), ("motor_ms", "Motor")]:
            for q, qt in [("p50", "Pcinquenta"), ("p95", "Pnoventaecinco")]:
                col = f"{st}_{q}"
                if col not in sub.columns or (sub[col] == "").all():
                    continue
                for agg, at in [("median", ""), ("min", "Min"), ("max", "Max")]:
                    add(f"{pre}{stt}{qt}{at}", {"tipo": "csv", "arquivo": f, "filtros": base, "coluna": col, "agg": agg},
                        casas=1, unidade="ms", evidencia="results/exp4_phone_latency.md" if agg != "median" else None,
                        descricao=f"Exp. 4 {ap}: {st} {q} por quadro — "
                                  + {"median": "mediana entre as execuções", "min": "menor execução",
                                     "max": "maior execução"}[agg])
        for agg, at in [("median", ""), ("min", "Min"), ("max", "Max")]:
            add(f"{pre}FPS{at}", {"tipo": "csv", "arquivo": f, "filtros": base, "coluna": "fps_medio", "agg": agg},
                casas=1, descricao=f"Exp. 4 {ap}: quadros por segundo ({agg} entre execuções)",
                evidencia="results/exp4_phone_latency.md" if agg != "median" else None)
        add(f"{pre}Quadros", {"tipo": "csv", "arquivo": f, "filtros": base, "coluna": "quadros", "agg": "sum"}, casas=0,
            descricao=f"Exp. 4 {ap}: quadros somados das execuções")
        add(f"{pre}QuadrosComRosto", {"tipo": "csv", "arquivo": f, "filtros": base, "coluna": "quadros_com_rosto",
                                      "agg": "sum"}, casas=0, descricao=f"Exp. 4 {ap}: quadros com rosto")
        add(f"{pre}TaxaRosto", {"tipo": "derivado", "expr": f"{pre}QuadrosComRosto / {pre}Quadros * 100",
                                "deps": [f"{pre}QuadrosComRosto", f"{pre}Quadros"]}, casas=1, unidade="%",
            descricao=f"Exp. 4 {ap}: fração de quadros com rosto detectado")
    # Lição da v1 (execução de 22:41Z no iPhone: falso alarme por PERCLOS com o usuário acordado)
    j = "results/phone/benchmark_pwa_2026-09-27T22-41-35-365Z.json"
    if (ROOT / j).exists():
        add("CelularVUmFalsoAlarmePERCLOS", {"tipo": "json", "arquivo": j, "chave": "perclos_final"}, **PCT,
            descricao="app v1.0 no iPhone (22:41Z, usuário acordado): PERCLOS no fim do benchmark")
        add("CelularVUmFalsoAlarmeQuadros", {"tipo": "json", "arquivo": j, "chave": "quadros_em_alerta"}, casas=0,
            descricao="app v1.0 no iPhone (22:41Z): quadros com alarme ligado")
        add("CelularVUmFalsoAlarmeQuadrosTotal", {"tipo": "json", "arquivo": j, "chave": "quadros"}, casas=0,
            descricao="app v1.0 no iPhone (22:41Z): quadros do benchmark")
        add("CelularVUmFalsoAlarmeFracao",
            {"tipo": "derivado", "expr": "CelularVUmFalsoAlarmeQuadros / CelularVUmFalsoAlarmeQuadrosTotal * 100",
             "deps": ["CelularVUmFalsoAlarmeQuadros", "CelularVUmFalsoAlarmeQuadrosTotal"]}, casas=1, unidade="%",
            descricao="app v1.0 no iPhone (22:41Z): % dos quadros com alarme ligado (falso alarme)")
    return tem_samsung


def exp5a():
    f = "results/exp5a_validacao_custo.csv"
    md = "results/exp5a_validacao.md"
    for secao, st in [("treino_oficial_outros_folds", "Custo"), ("mesmo_extrator_5_folds", "Mesmo"),
                      ("mesmo_extrator_loso", "MesmoLOSO")]:
        for esquema, et in [("3c", ""), ("B3", "BTres")]:
            for metric, mt in [("bal_acc", "AcBal"), ("va", "VA")]:
                filt = dict(secao=secao, conjunto_sujeitos="confirmado", fold="todos", esquema=esquema,
                            variante="estendida", metrica=metric)
                pre = f"ExpCincoA{st}{et}{mt}"
                add(f"{pre}Dif", {"tipo": "csv", "arquivo": f, "filtros": filt, "coluna": "diferenca",
                                  "ic": ["boot_dif_ic_inf", "boot_dif_ic_sup"]}, casas=1, escala=100.0, unidade="pp",
                    evidencia=md, descricao=f"Exp. 5a {secao} {esquema} {metric}: oficial − nosso (pp, IC bootstrap por sujeito)")
                add(f"{pre}Oficial", {"tipo": "csv", "arquivo": f, "filtros": filt, "coluna": "oficial"}, **PCT,
                    evidencia=md, descricao=f"Exp. 5a {secao} {esquema} {metric}: características oficiais")
                add(f"{pre}Nosso", {"tipo": "csv", "arquivo": f, "filtros": filt, "coluna": "nosso"}, **PCT,
                    evidencia=md, descricao=f"Exp. 5a {secao} {esquema} {metric}: nosso extrator (MediaPipe)")
                add(f"{pre}P", {"tipo": "csv", "arquivo": f, "filtros": filt, "coluna": "perm_p"}, casas=3,
                    evidencia=md, descricao=f"Exp. 5a {secao} {esquema} {metric}: p da permutação pareada")
    filt = dict(secao="treino_oficial_outros_folds", conjunto_sujeitos="confirmado", fold="todos", esquema="3c",
                variante="estendida", metrica="bal_acc")
    add("ExpCincoASujeitos", {"tipo": "csv", "arquivo": f, "filtros": filt, "coluna": "n_sujeitos"}, casas=0,
        evidencia=md, descricao="Exp. 5a sujeitos com par confirmado")
    add("ExpCincoAVideos", {"tipo": "csv", "arquivo": f, "filtros": filt, "coluna": "n_videos"}, casas=0,
        evidencia=md, descricao="Exp. 5a vídeos (3 classes)")
    add("ExpCincoAVideosBTres", {"tipo": "csv", "arquivo": f, "filtros": dict(filt, esquema="B3"), "coluna": "n_videos"},
        casas=0, evidencia=md, descricao="Exp. 5a vídeos (B3)")


def exp5b():
    f = "results/exp5b_transferencia.csv"
    md = "results/exp5b_transferencia.md"
    for modelo, mt in [("RF", ""), ("LR", "LR")]:
        pre = f"ExpCincoB{mt}"

        def row(nome, exp, esquema, nivel, lim, metrica, d, escala=100.0, unidade="%"):
            filt = dict(experimento=exp, variante="principal", modelo=modelo, esquema=esquema, nivel=nivel,
                        estrato="todos", limiar_tipo=lim, metrica=metrica)
            add(nome, {"tipo": "csv", "arquivo": f, "filtros": filt, "coluna": "valor", "ic": ["ic95_inf", "ic95_sup"]},
                casas=1, escala=escala, unidade=unidade, evidencia=md, descricao=f"Exp. 5b {modelo}: {d}")

        for lim, lt in [("fixo_0.5", ""), ("calibrado_treino", "Cal")]:
            row(f"{pre}AcBalJanelaA{lt}", "A_dmd_intra", "DMD", "janela", lim, "acuracia_balanceada",
                f"(A) DMD→DMD, acur. bal. por janela, limiar {lim}")
            row(f"{pre}AcBalJanelaB{lt}", "B_uta_intra", "B3", "janela", lim, "acuracia_balanceada",
                f"(B) UTA→UTA, acur. bal. por janela, limiar {lim}")
            row(f"{pre}AcBalJanelaC{lt}", "C_dmd_para_uta", "B3", "janela", lim, "acuracia_balanceada",
                f"(C) DMD→UTA, acur. bal. por janela, limiar {lim}")
            row(f"{pre}QuedaAcBalJanela{lt}", "queda_B_menos_C", "B3", "janela", lim, "acuracia_balanceada",
                f"queda B − C, acur. bal. por janela, limiar {lim} (pp)", escala=1.0, unidade="pp")
            row(f"{pre}ChoqueAcBalJanela{lt}", "choque_A_menos_C", "B3", "janela", lim, "acuracia_balanceada",
                f"choque de realidade A − C, acur. bal. por janela, limiar {lim} (pp)", escala=1.0, unidade="pp")
            row(f"{pre}AcVideoB{lt}", "B_uta_intra", "B3", "video", lim, "acuracia",
                f"(B) UTA→UTA, acurácia por vídeo, limiar {lim}")
            row(f"{pre}AcVideoC{lt}", "C_dmd_para_uta", "B3", "video", lim, "acuracia",
                f"(C) DMD→UTA, acurácia por vídeo, limiar {lim}")
            row(f"{pre}QuedaAcVideo{lt}", "queda_B_menos_C", "B3", "video", lim, "acuracia",
                f"queda B − C, acurácia por vídeo, limiar {lim} (pp)", escala=1.0, unidade="pp")
            for sm, stt in [("sensibilidade", "Sens"), ("especificidade", "Espec")]:
                row(f"{pre}C{stt}Janela{lt}", "C_dmd_para_uta", "B3", "janela", lim, sm,
                    f"(C) DMD→UTA, {sm} por janela (sonolentas/alertas reconhecidas), limiar {lim}")
                row(f"{pre}C{stt}Video{lt}", "C_dmd_para_uta", "B3", "video", lim, sm,
                    f"(C) DMD→UTA, {sm} por vídeo, limiar {lim}")
        row(f"{pre}AUCJanelaA", "A_dmd_intra", "DMD", "janela", "fixo_0.5", "auc", "(A) DMD→DMD, AUC por janela")
        row(f"{pre}AUCJanelaB", "B_uta_intra", "B3", "janela", "fixo_0.5", "auc", "(B) UTA→UTA, AUC por janela")
        row(f"{pre}AUCJanelaC", "C_dmd_para_uta", "B3", "janela", "fixo_0.5", "auc", "(C) DMD→UTA, AUC por janela")
        row(f"{pre}QuedaAUCJanela", "queda_B_menos_C", "B3", "janela", "fixo_0.5", "auc",
            "queda B − C, AUC por janela (pontos de AUC×100)", escala=1.0, unidade="pp")
        row(f"{pre}ChoqueAUCJanela", "choque_A_menos_C", "B3", "janela", "fixo_0.5", "auc",
            "choque de realidade A − C, AUC por janela", escala=1.0, unidade="pp")
        row(f"{pre}LabAUCJanela", "lab_A_menos_B", "B3", "janela", "fixo_0.5", "auc",
            "laboratório − real com treino real, A − B, AUC por janela", escala=1.0, unidade="pp")
        row(f"{pre}AUCVideoB", "B_uta_intra", "B3", "video", "fixo_0.5", "auc", "(B) UTA→UTA, AUC por vídeo")
        row(f"{pre}AUCVideoC", "C_dmd_para_uta", "B3", "video", "fixo_0.5", "auc", "(C) DMD→UTA, AUC por vídeo")
        row(f"{pre}QuedaAUCVideo", "queda_B_menos_C", "B3", "video", "fixo_0.5", "auc",
            "queda B − C, AUC por vídeo", escala=1.0, unidade="pp")
        filt = dict(experimento="C_dmd_para_uta", variante="principal", modelo=modelo, esquema="B3", nivel="janela",
                    estrato="todos", limiar_tipo="calibrado_treino", metrica="acuracia_balanceada")
        add(f"{pre}LimiarCal", {"tipo": "csv", "arquivo": f, "filtros": filt, "coluna": "limiar_valor"}, casas=3,
            evidencia=md, descricao=f"Exp. 5b {modelo}: limiar calibrado no LOSO do DMD (aplicado à UTA)")


def exp6():
    f = "results/exp6_noite.csv"
    md = "results/exp6_noite.md"
    for grupo, gt in [("nitymed|todos", "Noite"), ("nitymed|Microsleep", "NoiteMicrossono"),
                      ("nitymed|Yawning", "NoiteBocejo"), ("uta|todos", "UTA"), ("uta|KSS 0", "UTAAlerta"),
                      ("dmd_s5|todos", "DMDSonolencia"), ("dmd_par|safe", "DMDParado"), ("dmd_mov|safe", "DMDMovimento")]:
        add(f"ExpSeisRosto{gt}", {"tipo": "csv", "arquivo": f, "coluna": "valor", "ic": ["ic95_inf", "ic95_sup"],
                                  "filtros": dict(secao="a", grupo=grupo, metrica="taxa_rosto_media_por_video")},
            casas=2, **{"escala": 100.0, "unidade": "%"}, evidencia=md,
            descricao=f"Exp. 6 detecção de rosto por quadro (média por vídeo), {grupo}")
        add(f"ExpSeisSNR{gt}", {"tipo": "csv", "arquivo": f, "coluna": "valor", "ic": ["ic95_inf", "ic95_sup"],
                                "filtros": dict(secao="b", grupo=grupo, metrica="snr_rapido_mediana_por_video")},
            casas=1, evidencia=md, descricao=f"Exp. 6 SNR rápida do EAR (mediana por vídeo), {grupo}")
        add(f"ExpSeisRuido{gt}", {"tipo": "csv", "arquivo": f, "coluna": "valor", "ic": ["ic95_inf", "ic95_sup"],
                                  "filtros": dict(secao="b", grupo=grupo, metrica="ruido_rapido_mediana_por_video")},
            casas=4, evidencia=md, descricao=f"Exp. 6 ruído rápido do EAR (mediana por vídeo), {grupo}")
    for alvo, at in [("uta|todos", "UTA"), ("dmd_s5|todos", "DMDSonolencia"), ("dmd_par|safe", "DMDParado"),
                     ("dmd_mov|safe", "DMDMovimento")]:
        add(f"ExpSeisDifRostoNoite{at}", {"tipo": "csv", "arquivo": f, "coluna": "valor", "ic": ["ic95_inf", "ic95_sup"],
                                          "filtros": dict(secao="a", metrica=f"dif_taxa_rosto_media_nitymed|todos_menos_{alvo}")},
            casas=2, escala=100.0, unidade="pp", evidencia=md,
            descricao=f"Exp. 6 detecção de rosto NITYMED − {alvo} (pp)")
        add(f"ExpSeisDifSNRNoite{at}", {"tipo": "csv", "arquivo": f, "coluna": "valor", "ic": ["ic95_inf", "ic95_sup"],
                                        "filtros": dict(secao="b", metrica=f"dif_mediana_snr_rapido_nitymed|todos_menos_{alvo}")},
            casas=1, evidencia=md, descricao=f"Exp. 6 SNR rápida NITYMED − {alvo} (diferença de medianas)")
    add("ExpSeisDifSNRMovimentoParado",
        {"tipo": "csv", "arquivo": f, "coluna": "valor", "ic": ["ic95_inf", "ic95_sup"],
         "filtros": dict(secao="b", metrica="dif_pareada_snr_rapido_dmd_mov|safe_menos_dmd_par|safe")},
        casas=1, evidencia=md, descricao="Exp. 6 DMD mesmos motoristas: SNR rápida em movimento − parado (mediana pareada)")
    add("ExpSeisDifRuidoMovimentoParado",
        {"tipo": "csv", "arquivo": f, "coluna": "valor", "ic": ["ic95_inf", "ic95_sup"],
         "filtros": dict(secao="b", metrica="dif_pareada_ruido_rapido_dmd_mov|safe_menos_dmd_par|safe")},
        casas=4, evidencia=md, descricao="Exp. 6 DMD mesmos motoristas: ruído rápido em movimento − parado")
    add("ExpSeisSujeitosPareadosMovimento",
        {"tipo": "csv", "arquivo": f, "coluna": "n_agrupamentos",
         "filtros": dict(secao="b", metrica="dif_pareada_snr_rapido_dmd_mov|safe_menos_dmd_par|safe")},
        casas=0, evidencia=md, descricao="Exp. 6 sujeitos na comparação pareada movimento × parado")
    modelos = [("Modelo do DMD", "sem_normalizacao", "ModeloDMD"), ("Modelo da UTA-RLDD (B3)", "sem_normalizacao", "ModeloUTA"),
               ("regra_dmd", "", "Regra")]
    grupos = [("Microsleep", "Microssono"), ("Yawning", "Bocejo"), ("KSS 0", "UTAAlerta"), ("KSS 5", "UTAIntermediario"),
              ("KSS 10", "UTASonolento"), ("alerta_parado", "DMDParado"), ("alerta_movimento", "DMDMovimento"),
              ("sonolento", "DMDSonolento")]
    for modelo, var, mt in modelos:
        for grupo, gt in grupos:
            add(f"ExpSeisDisparo{mt}{gt}",
                {"tipo": "csv", "arquivo": f, "coluna": "valor", "ic": ["ic95_inf", "ic95_sup"],
                 "filtros": dict(secao="c", grupo=grupo, modelo=modelo, variante=var, metrica="taxa_disparo_janela")},
                **PCT, evidencia=md, descricao=f"Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), {modelo}, {grupo}")


def app():
    v = "results/app_validacao.csv"
    af = "results/app_validacao_dmd_ajuste_final.csv"
    md_nav = "results/app_validacao.md"
    md_off = "results/app_validacao_dmd.md"
    ver = "2.1 (ajuste final)"
    add("AppVersao", {"tipo": "md", "arquivo": "app/pwa/app.js", "regex": r'const APP_VERSION = "([\d.]+)"',
                      "texto": True}, texto=True, descricao="versão do app (PWA) descrita e validada")
    rd = "app/pwa/README.md"
    add("AppLimiarAmarelo", {"tipo": "md", "arquivo": rd, "regex": r"\| ATENÇÃO \(amarelo\) \| nível >= (\d+) \|"},
        casas=0, descricao="app: nível de entrada no amarelo (escolha de projeto)")
    add("AppLimiarVermelho", {"tipo": "md", "arquivo": rd, "regex": r"\| SONOLENTO \(vermelho\) \| nível >= (\d+),"},
        casas=0, descricao="app: nível de entrada no vermelho (escolha de projeto)")
    add("AppVolumePadrao", {"tipo": "md", "arquivo": rd, "regex": r"\*\*Padrão: (\d+) %\*\*"}, casas=0, unidade="%",
        descricao="app: volume padrão do alarme")
    # navegador (câmera falsa), UTA vídeo 0 (acordados)
    uta0 = ["uta_F1_09_v0", "uta_F3_31_v0", "uta_F5_51_v0"]
    uta10 = ["uta_F1_09_v10", "uta_F3_31_v10", "uta_F5_51_v10"]
    add("AppUTAAlertaVideos", {"tipo": "csv", "arquivo": v, "filtros": dict(versao_app=ver, metrica="pct_tempo_alerta",
                                                                            base="UTA-RLDD 0 (alerta)"), "agg": "count"},
        casas=0, descricao="app no navegador: vídeos UTA-RLDD alerta (vídeo 0) testados", evidencia=md_nav)
    add("AppUTAAlertaVerdeMin", {"tipo": "csv", "arquivo": v, "coluna": "valor", "agg": "min", "n_linhas": 3,
                                 "filtros": dict(versao_app=ver, video=uta0, metrica="pct_tempo_alerta")},
        casas=1, unidade="%", descricao="app no navegador: menor % do tempo em verde nos vídeos alerta da UTA",
        evidencia=md_nav)
    add("AppUTAAlertaAlarmes", {"tipo": "csv", "arquivo": v, "coluna": "valor", "agg": "sum", "n_linhas": 3,
                                "filtros": dict(versao_app=ver, video=uta0, metrica="n_alarme")},
        casas=0, descricao="app no navegador: alarmes nos vídeos alerta da UTA (soma)", evidencia=md_nav)
    add("AppUTAAlertaAvisos", {"tipo": "csv", "arquivo": v, "coluna": "valor", "agg": "sum", "n_linhas": 3,
                               "filtros": dict(versao_app=ver, video=uta0, metrica="n_aviso")},
        casas=0, descricao="app no navegador: avisos (amarelo) nos vídeos alerta da UTA (soma)", evidencia=md_nav)
    add("AppUTASonolentoVideos", {"tipo": "csv", "arquivo": v, "agg": "count", "n_linhas": 3,
                                  "filtros": dict(versao_app=ver, video=uta10, metrica="n_alarme")},
        casas=0, descricao="app no navegador: vídeos UTA-RLDD sonolentos (vídeo 10) testados", evidencia=md_nav)
    add("AppUTASonolentoComVermelho", {"tipo": "csv", "arquivo": v, "agg": "count",
                                       "filtros": dict(versao_app=ver, video=uta10, metrica="n_alarme", valor="1")},
        casas=0, descricao="app no navegador: vídeos UTA sonolentos que chegaram ao vermelho (n_alarme = 1)",
        evidencia=md_nav)
    rxt = r"\| \*\*Total\*\* \| \| \| ([\d,]+) / ([\d,]+) / ([\d,]+) \|"
    add("AppKappaNavegador", {"tipo": "md", "arquivo": md_nav, "regex": rxt, "grupo": 3, "decimal_virgula": True},
        casas=3, descricao="app no navegador (4 sessões DMD s5): kappa do olho fechado por quadro (linha Total da v2.1)")
    for metric, nt in [("bocejos_vp_anotados", "BocejosNavegadorAcertos"), ("bocejos_anotados", "BocejosNavegadorAnotados"),
                       ("bocejos_anotados_sem_mao", "BocejosNavegadorAnotadosSemMao")]:
        add(f"App{nt}", {"tipo": "csv", "arquivo": v, "coluna": "valor", "agg": "sum", "n_linhas": 4,
                         "filtros": dict(versao_app=ver, base="DMD s5 (encenada)", metrica=metric)},
            casas=0, descricao=f"app no navegador (4 sessões DMD s5): {metric} (soma)", evidencia=md_nav)
    # motor offline no DMD (mesmo engine.js, Face Landmarker da PWA)
    fin = dict(alternativa="final")
    for col, nm, d, casas, esc in [
            ("s1_min", "AppDMDMinutosDirecaoReal", "minutos ativos de direção real (s1)", 1, 1.0),
            ("s1_pct_vermelho", "AppDMDVermelhoDirecaoReal", "% do tempo em vermelho na direção real (s1)", 1, 1.0),
            ("s1_pct_amarelo", "AppDMDAmareloDirecaoReal", "% do tempo em amarelo na direção real (s1)", 1, 1.0),
            ("s1_alarmes", "AppDMDAlarmesDirecaoReal", "alarmes (entradas no vermelho) na direção real", 0, 1.0),
            ("s1_alarmes_h", "AppDMDAlarmesHoraDirecaoReal", "alarmes por hora na direção real", 1, 1.0),
            ("s1_bocejos", "AppDMDBocejosFalsosDirecaoReal", "bocejos detectados na direção real (falsos)", 0, 1.0),
            ("s5_min", "AppDMDMinutosSonolencia", "minutos ativos na sonolência encenada (s5)", 1, 1.0),
            ("s5_pct_vermelho", "AppDMDVermelhoSonolencia", "% do tempo em vermelho na sonolência encenada (s5)", 1, 1.0),
            ("s5_kappa", "AppKappaOffline", "kappa do olho fechado por quadro (média das 16 sessões s5)", 3, 1.0),
            ("s5_revoc_fech1", "AppFechUmSRevocacao", "revocação de fechamentos >= 1 s (s5)", 3, 1.0),
            ("s5_prec_fech1", "AppFechUmSPrecisao", "precisão de fechamentos >= 1 s (s5)", 3, 1.0),
            ("s5_revoc_fech05", "AppFechMeioSRevocacao", "revocação de fechamentos >= 0,5 s (s5)", 3, 1.0),
            ("s5_prec_fech05", "AppFechMeioSPrecisao", "precisão de fechamentos >= 0,5 s (s5)", 3, 1.0)]:
        add(nm, {"tipo": "csv", "arquivo": af, "filtros": fin, "coluna": col}, casas=casas, escala=esc,
            unidade="%" if "pct" in col else "", evidencia=md_off, descricao=f"motor do app offline no DMD: {d}")
    add("AppDMDVermelhoVUmDirecaoReal", {"tipo": "csv", "arquivo": "results/app_validacao_dmd_variantes_olhos.csv",
                                         "filtros": dict(variante="v1_app_anterior (EAR + PERCLOS>15% + olho fechado>=1 s; sem LSTM)"), "coluna": "pct_vermelho_s1"},
        casas=1, unidade="%", evidencia=md_off,
        descricao="lógica da v1 (EAR + PERCLOS > 15% + olho fechado >= 1 s) no mesmo DMD s1: % do tempo em vermelho")
    ses = "results/app_validacao_dmd_sessoes.csv"
    add("AppDMDSessoesDirecaoReal", {"tipo": "csv", "arquivo": ses, "filtros": dict(sessao="s1"), "agg": "count"},
        casas=0, descricao="motor offline: sessões de direção real (s1)", evidencia=md_off)
    add("AppDMDSessoesSemAlarme", {"tipo": "csv", "arquivo": ses, "filtros": dict(sessao="s1", alarmes="0"),
                                   "agg": "count"}, casas=0, descricao="motor offline: sessões s1 sem nenhum alarme")
    add("AppGACincoVermelho", {"tipo": "csv", "arquivo": ses, "filtros": dict(sessao="s1", sujeito="gA_5"),
                               "coluna": "frac_sonolento"}, **PCT, evidencia=md_off,
        descricao="motor offline: % do tempo em vermelho do motorista gA_5 na direção real (falso alarme conhecido)")
    boc = "results/app_validacao_dmd_bocejo.csv"
    fb = dict(detector="app (jawOpen>=0,4 e >= repouso+0,3, ou MAR>=repouso+0,4; por 1,5 s)")
    for col, nm, d in [("tp_sem_mao", "AppBocejoAcertosSemMao", "bocejos sem a mão detectados"),
                       ("n_sem_mao", "AppBocejoAnotadosSemMao", "bocejos anotados sem a mão"),
                       ("tp_com_mao", "AppBocejoAcertosComMao", "bocejos com a mão detectados"),
                       ("n_com_mao", "AppBocejoAnotadosComMao", "bocejos anotados com a mão"),
                       ("fp", "AppBocejoFalsos", "bocejos detectados sem anotação (s5)")]:
        add(nm, {"tipo": "csv", "arquivo": boc, "filtros": fb, "coluna": col, "agg": "sum", "n_linhas": 16}, casas=0,
            descricao=f"motor offline (16 sessões s5): {d}", evidencia=md_off)
    add("AppBocejoRevocacaoSemMao", {"tipo": "derivado", "expr": "AppBocejoAcertosSemMao / AppBocejoAnotadosSemMao",
                                     "deps": ["AppBocejoAcertosSemMao", "AppBocejoAnotadosSemMao"]}, casas=3,
        evidencia=md_off, descricao="motor offline: revocação de bocejos sem a mão na boca")
    add("AppBocejoRevocacaoComMao", {"tipo": "derivado", "expr": "AppBocejoAcertosComMao / AppBocejoAnotadosComMao",
                                     "deps": ["AppBocejoAcertosComMao", "AppBocejoAnotadosComMao"]}, casas=3,
        evidencia=md_off, descricao="motor offline: revocação de bocejos com a mão na boca")


def artigo_extra():
    """Números acrescentados para a versão final do artigo (28/09/2026): validação do detector de piscadas,
    horas/fps das bases, estratos de iluminação e reflexo do Exp. 3, sensibilidades e controles do Exp. 5b,
    detector no motorista de óculos do DMD e alguns números do app. Mesmas regras: só leitura de results/."""
    # --- detector de piscadas contra a anotação quadro a quadro do DMD (s5, limiar final; média das sessões)
    det = "results/exp5_dmd_validacao_detector.csv"
    mdc = "results/exp5_caracteristicas.md"
    for col, nm, d in [("f1_piscadas", "SinaisDetectorFUmPiscadas", "F1 de piscadas"),
                       ("precisao_piscadas", "SinaisDetectorPrecisaoPiscadas", "precisão de piscadas"),
                       ("revocacao_piscadas", "SinaisDetectorRevocacaoPiscadas", "revocação de piscadas"),
                       ("q_acuracia_bal", "SinaisDetectorAcBalQuadro", "acurácia balanceada olho fechado por quadro"),
                       ("q_kappa", "SinaisDetectorKappaQuadro", "kappa olho fechado por quadro")]:
        add(nm, {"tipo": "csv", "arquivo": det, "filtros": dict(sessao="s5"), "coluna": col, "agg": "mean",
                 "n_linhas": 16}, casas=3, evidencia=mdc,
            descricao=f"detector de piscadas (limiar final) x anotação do DMD s5: {d} (média das 16 sessões)")
    add("SinaisDetectorSessoes", {"tipo": "csv", "arquivo": det, "filtros": dict(sessao="s5"), "agg": "count"},
        casas=0, evidencia=mdc, descricao="detector de piscadas: sessões s5 do DMD avaliadas")
    rxs = r"os trechos safe_drive do DMD têm (\d+)–(\d+) s"
    add("SinaisSafeDriveMin", {"tipo": "md", "arquivo": mdc, "regex": rxs, "grupo": 1}, casas=0, unidade="s",
        descricao="DMD: duração mínima dos trechos safe_drive (s)")
    add("SinaisSafeDriveMax", {"tipo": "md", "arquivo": mdc, "regex": rxs, "grupo": 2}, casas=0, unidade="s",
        descricao="DMD: duração máxima dos trechos safe_drive (s; nenhum cabe numa janela de 30 s)")
    # --- horas e fps dos sinais extraídos (tabela 'Sinais disponíveis' do relatório de características)
    rx_tot = r"\| {base} \| \*\*total\*\* \| (\d+) \| (\d+) \| (\d+) \| ([\d.]+) \| ([\d,]+) \| ([\d,]+) \| ([\d,]+) \| ([\d,–]+) \|"
    for base, bt in [("DMD", "DMD"), ("NITYMED", "NITYMED"), ("UTA-RLDD", "UTA")]:
        rx = rx_tot.replace("{base}", re.escape(base))
        add(f"Base{bt}Horas", {"tipo": "md", "arquivo": mdc, "regex": rx, "grupo": 5, "decimal_virgula": True},
            casas=2, unidade="h", descricao=f"{base}: horas de vídeo com sinais extraídos (todas as sessões usadas)")
        add(f"Base{bt}FPSFaixa", {"tipo": "md", "arquivo": mdc, "regex": rx, "grupo": 8, "texto": True}, texto=True,
            descricao=f"{base}: quadros por segundo do contêiner (valor ou faixa)")
    # --- Exp. 3: estratos de iluminação e reflexo do MRL (intra out-of-fold e CEW patch -> MRL)
    s = "results/exp3_strat_conditions.csv"
    md3 = "results/exp3_eye_state.md"
    for stratum, lvl, lt in [("lighting", "iluminação ruim", "IlumRuim"), ("lighting", "iluminação boa", "IlumBoa"),
                             ("reflections", "reflexo grande", "ReflexoGrande"), ("reflections", "sem reflexo", "SemReflexo")]:
        add(f"ExpTresIntraMRL{lt}AcBal",
            {"tipo": "csv", "arquivo": s, "coluna": "value",
             "filtros": dict(protocol="A_intra_mrl", fold="oof", stratum=stratum, level=lvl, metric="bal_acc")},
            **PCT, evidencia=md3, descricao=f"Exp. 3 A intra-MRL {lvl} (out-of-fold agregado), acurácia balanceada")
        for metric, nt in [("bal_acc", "AcBal"), ("auc", "AUC")]:
            add(f"ExpTresCEWPatchMRL{lt}{nt}",
                {"tipo": "csv", "arquivo": s, "coluna": "value", "agg": "mean", "dp": "std1", "n_linhas": 5,
                 "filtros": dict(protocol="C_cew_patch_to_mrl", fold=[1, 2, 3, 4, 5], stratum=stratum, level=lvl,
                                 metric=metric)},
                **PCT, evidencia=md3, descricao=f"Exp. 3 C CEW patch→MRL {lvl}, {metric} (média ± dp, ddof=1)")
    # --- Exp. 5b: sensibilidades e controle do vídeo de referência (RF)
    f5 = "results/exp5b_transferencia.csv"
    md5 = "results/exp5b_transferencia.md"

    def rf(nome, exp, var, esquema, d, escala=100.0, unidade="%"):
        filt = dict(experimento=exp, variante=var, modelo="RF", esquema=esquema, nivel="janela", estrato="todos",
                    limiar_tipo="fixo_0.5", metrica="auc")
        add(nome, {"tipo": "csv", "arquivo": f5, "filtros": filt, "coluna": "valor", "ic": ["ic95_inf", "ic95_sup"]},
            casas=1, escala=escala, unidade=unidade, evidencia=md5, descricao=f"Exp. 5b RF: {d}")

    rf("ExpCincoBSemNormAUCJanelaB", "B_uta_intra", "sem_normalizacao", "B3", "(B) UTA→UTA sem normalização, AUC por janela")
    rf("ExpCincoBSemNormAUCJanelaC", "C_dmd_para_uta", "sem_normalizacao", "B3", "(C) DMD→UTA sem normalização, AUC por janela")
    rf("ExpCincoBSemNormQuedaAUCJanela", "queda_B_menos_C", "sem_normalizacao", "B3",
       "queda B − C sem normalização, AUC por janela (pp)", escala=1.0, unidade="pp")
    rf("ExpCincoBAlertaSUmAUCJanelaC", "C_dmd_para_uta", "alerta_s1", "B3",
       "(C) DMD→UTA com o alerta do DMD tirado da s1 (carro em movimento), AUC por janela")
    rf("ExpCincoBControleAUCCincoDezB", "B_uta_intra", "principal", "K10v5", "(B) AUC KSS 10 × KSS 5 (controle)")
    rf("ExpCincoBControleAUCCincoDezC", "C_dmd_para_uta", "principal", "K10v5", "(C) AUC KSS 10 × KSS 5 (controle)")
    rf("ExpCincoBControleQuedaCincoDez", "queda_B_menos_C", "principal", "K10v5",
       "queda B − C no contraste KSS 10 × 5 (pp de AUC)", escala=1.0, unidade="pp")
    rf("ExpCincoBControleQuedaZeroCinco", "queda_B_menos_C", "principal", "K5v0",
       "queda B − C no contraste KSS 5 × 0 (pp de AUC)", escala=1.0, unidade="pp")
    imp = "results/exp5b_importancia.csv"
    for base, bt, esquema in [("dmd", "DMD", "DMD"), ("uta", "UTA", "B3")]:
        for car, ct, casas in [("dur_max_fechamento_s", "FechMax", 2), ("perclos", "PERCLOS", 3)]:
            add(f"ExpCincoBMediana{ct}Sonolento{bt}",
                {"tipo": "csv", "arquivo": imp, "coluna": "mediana_sonolento",
                 "filtros": dict(tipo="auc_univariada", experimento=base, variante="sem_normalizacao", esquema=esquema,
                                 caracteristica=car)},
                casas=casas, evidencia=md5,
                descricao=f"Exp. 5b: mediana de {car} (bruto) nas janelas sonolentas, {bt}")
        add(f"ExpCincoBAUCUnivBocejo{bt}",
            {"tipo": "csv", "arquivo": imp, "coluna": "valor",
             "filtros": dict(tipo="auc_univariada", experimento=base, variante="principal", esquema=esquema,
                             caracteristica="frac_bocejo")},
            **PCT, evidencia=md5, descricao=f"Exp. 5b: AUC univariada da fração de bocejo, {bt}")
    oc = "results/exp5b_oculos_ear_dmd.csv"
    add("ExpCincoBMotoristasOculos", {"tipo": "csv", "arquivo": oc, "filtros": dict(oculos="True"), "agg": "count"},
        casas=0, evidencia=md5, descricao="DMD s5: sessões (= motoristas) com óculos")
    add("ExpCincoBOculosFalhaFUm", {"tipo": "csv", "arquivo": oc, "filtros": dict(sujeito="gE_29"),
                                    "coluna": "det_f1_piscadas"}, casas=3, evidencia=md5,
        descricao="DMD s5, motorista de óculos gE_29: F1 de piscadas do detector")
    add("ExpCincoBOculosFalhaQuedaEAR", {"tipo": "csv", "arquivo": oc, "filtros": dict(sujeito="gE_29"),
                                         "coluna": "queda_relativa"}, **PCT, evidencia=md5,
        descricao="DMD s5, gE_29: queda relativa do EAR mediano de aberto para fechado")
    add("ExpCincoBOculosFalhaAUCEAR", {"tipo": "csv", "arquivo": oc, "filtros": dict(sujeito="gE_29"),
                                       "coluna": "auc_menos_ear"}, **PCT, evidencia=md5,
        descricao="DMD s5, gE_29: AUC de −EAR para separar quadros fechados de abertos")
    for agg, at in [("min", "Min"), ("max", "Max")]:
        add(f"ExpCincoBSemOculosQuedaEAR{at}", {"tipo": "csv", "arquivo": oc, "filtros": dict(oculos="False"),
                                                 "coluna": "queda_relativa", "agg": agg}, **PCT, evidencia=md5,
            descricao=f"DMD s5, motoristas sem óculos: queda relativa do EAR ({agg} entre sessões)")
    add("ExpCincoBQuedaAUCIntraSujeito",
        {"tipo": "csv", "arquivo": f5, "coluna": "valor", "ic": ["ic95_inf", "ic95_sup"],
         "filtros": dict(experimento="queda_B_menos_C", variante="principal", modelo="RF", esquema="B3", nivel="janela",
                         estrato="todos", limiar_tipo="fixo_0.5", metrica="auc_media_intra_sujeito")},
        casas=1, escala=1.0, unidade="pp", evidencia=md5,
        descricao="Exp. 5b RF: queda B − C na AUC média intra-sujeito (pp de AUC)")
    # --- Exp. 6 (d): PERCLOS com o limiar do pipeline x limiar pelo p90 do próprio vídeo
    f6 = "results/exp6_noite.csv"
    md6 = "results/exp6_noite.md"
    # dmd_mov|safe acrescentado na verificação do artigo (28/09): o p90 também infla o PERCLOS de motoristas
    # ACORDADOS com o carro em movimento, ressalva necessária à leitura do microssono noturno.
    for grupo, gt in [("nitymed|Microsleep", "Microssono"), ("uta|KSS 0", "UTAAlerta"), ("dmd_mov|safe", "DMDMovimento")]:
        for met, mt in [("perclos_mediana_por_video", ""), ("perclos_p90_mediana_por_video", "PNoventa")]:
            add(f"ExpSeisPERCLOS{gt}{mt}", {"tipo": "csv", "arquivo": f6, "coluna": "valor",
                                             "filtros": dict(secao="d", grupo=grupo, metrica=met)},
                **PCT, evidencia=md6,
                descricao=f"Exp. 6 PERCLOS mediano por vídeo, {grupo}, " +
                          ("limiar do pipeline (referência)" if not mt else "limiar pelo p90 do EAR do próprio vídeo"))
    # --- PWA v1.0 no Chrome headless do Mac (20/09): latência por estágio
    mac = "results/pwa_headless_mac_bench_20260920-190012.json"
    for est, et in [("landmarker", "Landmarker"), ("cnn_olhos_2olhos", "CNN"), ("temporal", "LSTM")]:
        for q, qt in [("p50", "Pcinquenta"), ("p95", "Pnoventaecinco")]:
            add(f"MacPWA{et}{qt}", {"tipo": "json", "arquivo": mac, "chave": f"latencia_ms.{est}.{q}"}, casas=1,
                unidade="ms", descricao=f"PWA v1.0 no Chrome headless do Mac (20/09, vídeo 320x240): {est} {q}")
    # --- app: EAR sozinho (regra dos olhos da v1) e contagens da validação no navegador
    vo = "results/app_validacao_dmd_variantes_olhos.csv"
    md_off = "results/app_validacao_dmd.md"
    add("AppKappaEARSozinho", {"tipo": "csv", "arquivo": vo, "filtros": dict(variante="ear_v1"), "coluna": "kappa_s5"},
        casas=3, evidencia=md_off, descricao="motor offline: kappa por quadro com o EAR sozinho (regra da v1)")
    add("AppDMDVermelhoEARSozinho", {"tipo": "csv", "arquivo": vo, "filtros": dict(variante="ear_v1"),
                                     "coluna": "pct_vermelho_s1"}, casas=1, unidade="%", evidencia=md_off,
        descricao="motor offline: % do tempo em vermelho na direção real com o EAR sozinho nos olhos")
    v = "results/app_validacao.csv"
    ver = "2.1 (ajuste final)"
    add("AppNavegadorSessoesDMD", {"tipo": "csv", "arquivo": v, "agg": "count",
                                   "filtros": dict(versao_app=ver, base="DMD s5 (encenada)", metrica="n_alarme")},
        casas=0, evidencia="results/app_validacao.md", descricao="app no navegador: sessões do DMD s5 testadas")
    add("AppNavegadorVideosNITYMED", {"tipo": "csv", "arquivo": v, "agg": "count",
                                      "filtros": dict(versao_app=ver, base=["NITYMED bocejo (noite)",
                                                                            "NITYMED microssono (noite)"],
                                                      metrica="n_alarme")},
        casas=0, evidencia="results/app_validacao.md", descricao="app no navegador: vídeos do NITYMED testados")


def revisao_pares():
    """Números acrescentados pela revisão por pares simulada do ARTIGO (28/09/2026): (i) disparo do RF do
    DMD com o limiar calibrado no treino e da RL do DMD, para comparar a regra com os modelos num ponto de
    operação justo (não só no limiar 0,5); (ii) queda da RL sem centragem e robustez da queda do RF nas
    sensibilidades do Exp. 5b; (iii) sujeitos de treino de B por fold (60 sujeitos, 5 folds oficiais de 12).
    Mesmas regras: só leitura de results/."""
    f6 = "results/exp6_noite.csv"
    md6 = "results/exp6_noite.md"
    for grupo, gt in [("KSS 0", "UTAAlerta"), ("KSS 10", "UTASonolento"), ("alerta_movimento", "DMDMovimento")]:
        add(f"ExpSeisDisparoModeloDMDCal{gt}",
            {"tipo": "csv", "arquivo": f6, "coluna": "valor", "ic": ["ic95_inf", "ic95_sup"],
             "filtros": dict(secao="c", grupo=grupo, modelo="Modelo do DMD", variante="sem_normalizacao",
                             metrica="taxa_disparo_janela_limiar_calibrado")},
            **PCT, evidencia=md6,
            descricao=f"Exp. 6 taxa de disparo do RF do DMD (brutas) com o limiar calibrado só no treino (LOSO do DMD), {grupo}")
    for grupo, gt in [("KSS 0", "UTAAlerta"), ("KSS 10", "UTASonolento")]:
        add(f"ExpSeisDisparoModeloDMDLR{gt}",
            {"tipo": "csv", "arquivo": f6, "coluna": "valor", "ic": ["ic95_inf", "ic95_sup"],
             "filtros": dict(secao="c", grupo=grupo, modelo="Exp. 5b: exp5b_final_C_sem_normalizacao_LR.joblib",
                             variante="sem_normalizacao", metrica="taxa_disparo_janela")},
            **PCT, evidencia=md6,
            descricao=f"Exp. 6 taxa de disparo da RL do DMD (brutas, limiar 0,5; modelo final do Exp. 5b), {grupo}")
    f5 = "results/exp5b_transferencia.csv"
    md5 = "results/exp5b_transferencia.md"
    add("ExpCincoBLRSemNormQuedaAUCJanela",
        {"tipo": "csv", "arquivo": f5, "coluna": "valor", "ic": ["ic95_inf", "ic95_sup"],
         "filtros": dict(experimento="queda_B_menos_C", variante="sem_normalizacao", modelo="LR", esquema="B3",
                         nivel="janela", estrato="todos", limiar_tipo="fixo_0.5", metrica="auc")},
        casas=1, escala=1.0, unidade="pp", evidencia=md5,
        descricao="Exp. 5b RL: queda B − C sem normalização por sujeito, AUC por janela (pp de AUC)")
    rx = (r"RF: AUC de C entre [\d,]+ \(`alerta_s1`\) e [\d,]+ \(`ref_s2_espelho`\); "
          r"queda B − C com IC acima de 0 em (\d+) de (\d+) variantes")
    add("ExpCincoBSensVariantesICPositivo", {"tipo": "md", "arquivo": md5, "regex": rx, "grupo": 1}, casas=0,
        descricao="Exp. 5b RF: variantes (incluindo a principal) em que o IC da queda B − C de AUC fica acima de 0")
    add("ExpCincoBSensVariantesTotal", {"tipo": "md", "arquivo": md5, "regex": rx, "grupo": 2}, casas=0,
        descricao="Exp. 5b RF: total de variantes analisadas (principal + sensibilidades)")
    add("ExpCincoBSujeitosTreinoB", {"tipo": "derivado", "deps": ["BaseUTASujeitos"], "expr": "BaseUTASujeitos*4/5"},
        casas=0, descricao="Exp. 5b: sujeitos da UTA-RLDD no treino de B a cada fold (4 dos 5 folds oficiais de 12)")


def robustez():
    """Números da revisão final (26/10/2026): robustez da queda B − C (src/experiments/exp5b_robustez.py).
    (i) referência alternativa na UTA-RLDD (vídeo KSS 5); (ii) IC em dois níveis (retreino de C); (iii) B com
    13 sujeitos; (iv) vazamento no mesmo cenário do Exp. 5b; (v) sensibilidade no falso alarme da regra.
    Os valores do CSV já estão em AUC x 100, % ou pp."""
    f = "results/exp5b_robustez.csv"
    md = "results/exp5b_robustez.md"

    def rob(nome, analise, variante, modelo, nivel, metrica, d, unidade="", com_ic=True):
        spec = {"tipo": "csv", "arquivo": f, "coluna": "valor",
                "filtros": dict(analise=analise, variante=variante, modelo=modelo, nivel=nivel, metrica=metrica)}
        if com_ic:
            spec["ic"] = ["ic95_inf", "ic95_sup"]
        add(nome, spec, casas=1, unidade=unidade, evidencia=md, descricao=f"Exp. 5b robustez: {d}")

    for mod, mt, mn in [("RF", "", "RF"), ("LR", "LR", "RL")]:
        rob(f"ExpCincoB{mt}RefAltAUCJanelaB", "referencia_kss5", "ref_video_kss5", mod, "janela", "auc_B",
            f"{mn}, (B) UTA→UTA com a centragem pelo vídeo KSS 5, AUC por janela")
        rob(f"ExpCincoB{mt}RefAltAUCJanelaC", "referencia_kss5", "ref_video_kss5", mod, "janela", "auc_C",
            f"{mn}, (C) DMD→UTA com a centragem pelo vídeo KSS 5, AUC por janela")
        rob(f"ExpCincoB{mt}RefAltQuedaAUCJanela", "referencia_kss5", "ref_video_kss5", mod, "janela", "queda_auc",
            f"{mn}, queda B − C com a centragem pelo vídeo KSS 5, AUC por janela (pp)", "pp")
        rob(f"ExpCincoB{mt}RefAltQuedaAUCVideo", "referencia_kss5", "ref_video_kss5", mod, "video", "queda_auc",
            f"{mn}, queda B − C com a centragem pelo vídeo KSS 5, AUC por vídeo (pp)", "pp")
        rob(f"ExpCincoB{mt}BootDuploQuedaAUCJanela", "bootstrap_dois_niveis", "ref_video_alerta", mod, "janela",
            "queda_auc", f"{mn}, queda B − C por janela com IC em dois níveis (retreino de C)", "pp")
        rob(f"ExpCincoB{mt}TrezeAUCJanelaB", "volume_treino", "13_sujeitos", mod, "janela", "auc_B",
            f"{mn}, B treinado com 13 sujeitos por fold (mediana dos sorteios; faixa = percentis 2,5 e 97,5)")
        rob(f"ExpCincoB{mt}TrezeQuinhentasAUCJanelaB", "volume_treino", "13_sujeitos_510_janelas", mod, "janela",
            "auc_B", f"{mn}, B treinado com 13 sujeitos e 510 janelas por fold (mediana dos sorteios)")
        for met, mtt in [("acuracia_balanceada", "AcBal"), ("auc", "AUC")]:
            rob(f"ExpCincoB{mt}Vazamento{mtt}", "vazamento_mesmo_cenario", "centrada", mod, "janela", f"{met}_inflacao",
                f"{mn}, inflação pela partição aleatória de janelas no cenário do Exp. 5b ({met}, pp)", "pp", False)
            rob(f"ExpCincoB{mt}Vazamento{mtt}Sujeito", "vazamento_mesmo_cenario", "centrada", mod, "janela",
                f"{met}_por_sujeito", f"{mn}, {met} com os folds oficiais por sujeito (cenário do Exp. 5b)", "%", False)
            rob(f"ExpCincoB{mt}Vazamento{mtt}Aleatoria", "vazamento_mesmo_cenario", "centrada", mod, "janela",
                f"{met}_aleatoria", f"{mn}, {met} com a partição aleatória de janelas (média de 3 sementes)", "%", False)
        for cen, ct in [("C_bruta", "CBruta"), ("C_centrada", "CCentrada"), ("B_centrada", "BCentrada")]:
            rob(f"ExpCincoB{mt}OpRegraSens{ct}", "ponto_operacao_regra", cen, mod, "janela",
                "sensibilidade_no_fa_da_regra",
                f"{mn}, {cen}: sensibilidade no falso alarme da regra nas janelas KSS 0 (diagnóstico)", "%", False)
        add(f"ExpCincoB{mt}TrezeQuedaAUCJanela",
            {"tipo": "derivado", "deps": [f"ExpCincoB{mt}TrezeAUCJanelaB", f"ExpCincoB{mt}AUCJanelaC"],
             "expr": f"ExpCincoB{mt}TrezeAUCJanelaB - 100*ExpCincoB{mt}AUCJanelaC"}, casas=1, unidade="pp",
            descricao=f"Exp. 5b {mn}: queda B − C com B treinado em 13 sujeitos (mediana dos sorteios − C), pp de AUC")
    # controle KSS 10 × 5 da RL (o do RF está em artigo_extra)
    f5 = "results/exp5b_transferencia.csv"
    add("ExpCincoBLRControleQuedaCincoDez",
        {"tipo": "csv", "arquivo": f5, "coluna": "valor", "ic": ["ic95_inf", "ic95_sup"],
         "filtros": dict(experimento="queda_B_menos_C", variante="principal", modelo="LR", esquema="K10v5",
                         nivel="janela", estrato="todos", limiar_tipo="fixo_0.5", metrica="auc")},
        casas=1, unidade="pp", evidencia="results/exp5b_transferencia.md",
        descricao="Exp. 5b RL: queda B − C no contraste KSS 10 × 5 (pp de AUC)")
    # referência do DMD na s2 (variante ref_s2_espelho) e quedas por vídeo sem centragem, nos dois modelos
    for mod, mt in [("RF", ""), ("LR", "LR")]:
        for var, vt, nivel, nt in [("ref_s2_espelho", "RefSDois", "janela", "Janela"),
                                   ("sem_normalizacao", "SemNorm", "video", "Video")]:
            add(f"ExpCincoB{mt}{vt}QuedaAUC{nt}",
                {"tipo": "csv", "arquivo": f5, "coluna": "valor", "ic": ["ic95_inf", "ic95_sup"],
                 "filtros": dict(experimento="queda_B_menos_C", variante=var, modelo=mod, esquema="B3", nivel=nivel,
                                 estrato="todos", limiar_tipo="fixo_0.5", metrica="auc")},
                casas=1, unidade="pp", evidencia="results/exp5b_transferencia.md",
                descricao=f"Exp. 5b {mod}: queda B − C de AUC por {nivel}, variante {var} (pp)")
    # medianas das janelas ALERTAS (as das sonolentas estão em artigo_extra), características brutas
    imp = "results/exp5b_importancia.csv"
    for base, bt, esquema in [("dmd", "DMD", "DMD"), ("uta", "UTA", "B3")]:
        for car, ct, casas in [("dur_max_fechamento_s", "FechMax", 2), ("perclos", "PERCLOS", 3)]:
            add(f"ExpCincoBMediana{ct}Alerta{bt}",
                {"tipo": "csv", "arquivo": imp, "coluna": "mediana_alerta",
                 "filtros": dict(tipo="auc_univariada", experimento=base, variante="sem_normalizacao", esquema=esquema,
                                 caracteristica=car)},
                casas=casas, evidencia="results/exp5b_transferencia.md",
                descricao=f"Exp. 5b: mediana de {car} (bruto) nas janelas alertas, {bt}")
    # faixas usadas no texto: queda da RL por janela nas quatro normalizações e do RF fora da normalização principal
    quedas_lr = ["ExpCincoBLRQuedaAUCJanela", "ExpCincoBLRSemNormQuedaAUCJanela", "ExpCincoBLRRefAltQuedaAUCJanela",
                 "ExpCincoBLRRefSDoisQuedaAUCJanela"]
    quedas_rf = ["ExpCincoBSemNormQuedaAUCJanela", "ExpCincoBRefAltQuedaAUCJanela"]
    for nome, fn, deps, d in [("ExpCincoBLRQuedaNormMin", "min", quedas_lr, "menor queda da RL entre as quatro normalizações"),
                              ("ExpCincoBLRQuedaNormMax", "max", quedas_lr, "maior queda da RL entre as quatro normalizações"),
                              ("ExpCincoBQuedaOutrasNormMin", "min", quedas_rf, "menor queda do RF sem a referência principal"),
                              ("ExpCincoBQuedaOutrasNormMax", "max", quedas_rf, "maior queda do RF sem a referência principal")]:
        add(nome, {"tipo": "derivado", "deps": deps, "expr": f"{fn}({', '.join(deps)})"}, casas=1, unidade="pp",
            descricao=f"Exp. 5b: {d} (pp de AUC por janela)")


# ----------------------------------------------------------------------------------------------
# Saídas
# ----------------------------------------------------------------------------------------------
def formatted(e: dict, tex: bool) -> dict:
    """Macros de uma entrada: {sufixo: texto}."""
    if e["texto"]:
        s = str(e["bruto"])
        return {"": tex_escape(s) if tex else s}
    k, c = e["escala"], e["casas"]
    out = {"": fmt_num(e["bruto"] * k, c, tex)}
    if e["ic_bruto"]:
        lo, hi = (fmt_num(x * k, c, tex) for x in e["ic_bruto"])
        out["IC"] = f"[{lo}; {hi}]"
        out["ComIC"] = f"{out['']} [{lo}; {hi}]"
    if e["dp_bruto"] is not None:
        out["DP"] = fmt_num(e["dp_bruto"] * k, c, tex)
        pm = r"\,$\pm$\," if tex else " ± "
        out["MDP"] = f"{out['']}{pm}{out['DP']}"
    return out


def write_tex(tem_samsung: bool) -> None:
    now = dt.datetime.now().strftime("%d/%m/%Y %H:%M")
    lines = [f"% paper/numeros.tex — GERADO por src/experiments/make_paper_numbers.py em {now}.",
             "% NÃO EDITE À MÃO: rode `bash tools/atualizar_celular.sh` (ou o script acima).",
             "% Origem de cada número: results/numeros_resumo.md (arquivo, coluna e filtro).",
             "% Percentuais sem o símbolo %; diferenças em pontos percentuais (pp); IC 95% entre colchetes.",
             r"\newif\ifSamsungMedido"]
    lines.append(r"\SamsungMedidotrue" if tem_samsung else r"\SamsungMedidofalse")
    if not tem_samsung:
        lines.append(r"\newcommand{\SamsungPendente}{medição prevista}")
    grupo = None
    for e in ENTRIES:
        g = re.match(r"(Base|ExpUm|ExpDois|ExpTres|ExpQuatro|MacPWA|Celular|ExpCincoA|ExpCincoB|ExpSeis|App)", e["nome"])
        g = g.group(1) if g else "Outros"
        if g != grupo:
            lines.append(f"% --- {g} ---")
            grupo = g
        for suf, val in formatted(e, tex=True).items():
            lines.append(f"\\newcommand{{\\{e['nome']}{suf}}}{{{val}}}")
    OUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(tem_samsung: bool) -> None:
    data = {"gerado_em": dt.datetime.now().isoformat(timespec="seconds"),
            "gerado_por": "src/experiments/make_paper_numbers.py",
            "convencoes": "percentuais sem %; diferenças em pp; IC 95% entre colchetes; texto com vírgula decimal",
            "samsung_medido": tem_samsung,
            "valores": {}, "entradas": []}
    if not tem_samsung:
        data["valores"]["SamsungPendente"] = "medição prevista"
    for e in ENTRIES:
        f = formatted(e, tex=False)
        for suf, val in f.items():
            data["valores"][e["nome"] + suf] = val
        data["entradas"].append({
            "nome": e["nome"], "valor": f[""], "macros": {e["nome"] + s: v for s, v in f.items()},
            "bruto": e["bruto"], "ic_bruto": e["ic_bruto"], "dp_bruto": e["dp_bruto"],
            "casas": e["casas"], "escala": e["escala"], "unidade": e["unidade"], "texto": e["texto"],
            "descricao": e["descricao"], "evidencia": e["evidencia"], "spec": e["spec"]})
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def write_md(tem_samsung: bool) -> None:
    now = dt.datetime.now().strftime("%d/%m/%Y %H:%M")
    out = ["# Números do artigo e do pôster (gerados de results/)", "",
           f"Gerado em {now} por `src/experiments/make_paper_numbers.py`. Cada linha é um macro de "
           "`paper/numeros.tex` (o mesmo valor está em `poster/numeros.json`). Nenhum número foi digitado: "
           "todos saem do arquivo e da coluna/filtro indicados. Macros derivados (`IC`, `ComIC`, `DP`, `MDP`) "
           "saem da mesma linha. Percentuais sem o símbolo %; diferenças em pontos percentuais (pp).", "",
           "Galaxy S23 Ultra: " + ("medido (macros `\\CelularSamsung...`; `\\ifSamsungMedido` verdadeiro)." if tem_samsung
                                   else "sem dados; macro `\\SamsungPendente` = \"medição prevista\" e `\\ifSamsungMedido` falso."),
           "", f"Total: {len(ENTRIES)} números ({sum(len(formatted(e, False)) for e in ENTRIES)} macros).", "",
           "| Macro | Valor | Unidade | O que é | Arquivo de origem | Coluna / filtro |", "|---|---|---|---|---|---|"]
    for e in ENTRIES:
        f = formatted(e, tex=False)
        val = f.get("ComIC") or f.get("MDP") or f[""]
        arq, filt = describe(e["spec"])
        esc = "" if e["escala"] == 1 else f" (×{e['escala']:g})"
        out.append(f"| `\\{e['nome']}` | {val} | {e['unidade']}{esc} | {e['descricao']} | `{arq}` | "
                   f"{filt.replace('|', '&#124;')} |")
    out += ["", "## Conferência automática", "",
            "(ainda não executada: rode `.venv/bin/python src/experiments/conferir_numeros.py`)", ""]
    OUT_MD.write_text("\n".join(out), encoding="utf-8")


def main() -> int:
    bases()
    exp1()
    exp2()
    exp3()
    exp4()
    tem_samsung = exp4_celular()
    exp5a()
    exp5b()
    exp6()
    app()
    artigo_extra()
    revisao_pares()
    robustez()
    write_tex(tem_samsung)
    write_json(tem_samsung)
    write_md(tem_samsung)
    n_macros = sum(len(formatted(e, False)) for e in ENTRIES)
    rel = lambda p: p.relative_to(ROOT) if p.is_relative_to(ROOT) else p  # noqa: E731
    print(f"{len(ENTRIES)} números, {n_macros} macros -> {rel(OUT_TEX)}, {rel(OUT_JSON)}, {rel(OUT_MD)}; "
          f"Samsung medido: {tem_samsung}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
