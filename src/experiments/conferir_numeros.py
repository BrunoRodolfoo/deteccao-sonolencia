#!/usr/bin/env python3
"""Conferência automática de paper/numeros.tex e poster/numeros.json contra os arquivos de origem.

Implementação INDEPENDENTE da de make_paper_numbers.py (outro leitor de CSV/JSON/MD, outro formatador).
Para cada número:
  1. recalcula o valor a partir da especificação gravada em poster/numeros.json (arquivo, filtros,
     coluna, agregação) lendo o arquivo de origem de novo, e compara o texto formatado;
  2. confere que cada macro de paper/numeros.tex tem o MESMO texto do JSON (e vice-versa);
  3. procura o número no .md de resultados correspondente (os .md são gerados pelos próprios
     experimentos): "no .md" = o valor (ou o IC) aparece lá, com a mesma precisão;
  4. (opcional, padrão ligado) compila com tectonic um documento que imprime TODOS os macros, para
     garantir que numeros.tex é LaTeX válido.
Escreve o relatório na seção "## Conferência automática" de results/numeros_resumo.md e sai com
código 1 se houver qualquer divergência nos passos 1, 2 ou 4 (o passo 3 é informativo: números
derivados ou agregados de celular não aparecem literalmente em nenhum .md).

Uso:  .venv/bin/python src/experiments/conferir_numeros.py [--sem-latex]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEX = ROOT / "paper" / "numeros.tex"
JSN = ROOT / "poster" / "numeros.json"
MD = ROOT / "results" / "numeros_resumo.md"
TECTONIC = os.environ.get("TECTONIC", str(Path.home() / ".local" / "bin" / "tectonic"))


# ------------------------------- recálculo independente ------------------------------------
def br(x: float, casas: int) -> str:
    x = round(x, casas)
    inteiro, _, dec = f"{abs(x):.{casas}f}".partition(".")
    grupos = []
    while len(inteiro) > 3:
        grupos.insert(0, inteiro[-3:])
        inteiro = inteiro[:-3]
    grupos.insert(0, inteiro)
    s = ".".join(grupos) + ("," + dec if casas > 0 else "")
    zero = all(ch in "0.," for ch in s)
    return ("−" + s) if (x < 0 and not zero) else s


def ler_csv(arq: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / arq, dtype=str, keep_default_na=False)


def filtrar(df: pd.DataFrame, filtros: dict) -> pd.DataFrame:
    for k, v in filtros.items():
        df = df[df[k].isin([str(x) for x in v])] if isinstance(v, list) else df[df[k] == str(v)]
    return df


def recalcular(spec: dict, entradas: dict):
    t = spec["tipo"]
    if t == "csv":
        sel = filtrar(ler_csv(spec["arquivo"]), spec["filtros"])
        agg = spec.get("agg")
        if agg is None:
            assert len(sel) == 1, f"{len(sel)} linhas"
        if spec.get("n_linhas") is not None:
            assert len(sel) == spec["n_linhas"], f"{len(sel)} linhas (esperadas {spec['n_linhas']})"
        if agg == "count":
            return float(len(sel)), None, None
        s = sel[spec["coluna"]].astype(float)
        val = {None: lambda: s.iloc[0], "mean": s.mean, "median": s.median, "min": s.min, "max": s.max,
               "sum": s.sum, "std0": lambda: s.std(ddof=0), "std1": lambda: s.std(ddof=1)}[agg]()
        ic = None
        if spec.get("ic"):
            ic = (float(sel[spec["ic"][0]].astype(float).iloc[0]), float(sel[spec["ic"][1]].astype(float).iloc[0]))
        dp = float(s.std(ddof=0 if spec["dp"] == "std0" else 1)) if spec.get("dp") else None
        return float(val), ic, dp
    if t == "csv_texto":
        sel = filtrar(ler_csv(spec["arquivo"]), spec["filtros"])
        return ", ".join(sorted(sel[spec["coluna"]].unique())), None, None
    if t == "json":
        d = json.loads((ROOT / spec["arquivo"]).read_text(encoding="utf-8"))
        for k in spec["chave"].split("."):
            d = d[k]
        return (d if isinstance(d, str) else float(d)), None, None
    if t == "md":
        m = re.search(spec["regex"], (ROOT / spec["arquivo"]).read_text(encoding="utf-8"))
        assert m, "regex não encontrada"
        g = m.group(spec.get("grupo", 1))
        if spec.get("texto"):
            return g, None, None
        return float(g.replace(".", "").replace(",", ".") if spec.get("decimal_virgula") else g), None, None
    if t == "derivado":
        env = {n: entradas[n]["_recalc"] for n in spec["deps"]}
        return float(eval(spec["expr"], {"__builtins__": {}, "abs": abs, "max": max, "min": min}, env)), None, None
    raise ValueError(t)


def esperado(e: dict, val, ic, dp) -> dict:
    if e["texto"]:
        return {e["nome"]: str(val)}
    k, c = e["escala"], e["casas"]
    out = {e["nome"]: br(val * k, c)}
    if ic:
        lo, hi = br(ic[0] * k, c), br(ic[1] * k, c)
        out[e["nome"] + "IC"] = f"[{lo}; {hi}]"
        out[e["nome"] + "ComIC"] = f"{out[e['nome']]} [{lo}; {hi}]"
    if dp is not None:
        out[e["nome"] + "DP"] = br(dp * k, c)
        out[e["nome"] + "MDP"] = f"{out[e['nome']]} ± {out[e['nome'] + 'DP']}"
    return out


# ------------------------------- tex -> texto ------------------------------------------------
def tex_para_texto(s: str) -> str:
    s = s.replace(r"\,$\pm$\,", " ± ").replace("$-$", "−").replace("--", "–")
    for a, b in [(r"\_", "_"), (r"\%", "%"), (r"\&", "&"), (r"\#", "#"), (r"\$", "$"), (r"\{", "{"), (r"\}", "}")]:
        s = s.replace(a, b)
    return s


def ler_tex() -> dict:
    out = {}
    for ln in TEX.read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", ln)
        if m:
            assert m.group(1) not in out, f"macro duplicado {m.group(1)}"
            out[m.group(1)] = m.group(2)
    return out


# ------------------------------- evidência nos .md -------------------------------------------
_TOK = re.compile(r"[−-]?\d+(?:[.,]\d+)*")


def numeros_md(path: str) -> list[float]:
    txt = (ROOT / path).read_text(encoding="utf-8")
    vals = []
    for t in _TOK.findall(txt):
        neg = t[0] in "−-"
        t = t.lstrip("−-")
        cands = set()
        if "," in t:                                  # pt-BR: vírgula decimal, ponto de milhar
            cands.add(t.replace(".", "").replace(",", "."))
        else:
            cands.add(t)                              # ponto decimal (ou inteiro)
            if re.fullmatch(r"\d{1,3}(\.\d{3})+", t):  # 8.661 = oito mil...
                cands.add(t.replace(".", ""))
        for c in cands:
            try:
                v = float(c)
            except ValueError:
                continue
            vals.append(-v if neg else v)
            vals.append(v)                            # o .md pode trazer o módulo (ex.: "+2,0 pp" / "queda de 7,0")
    return vals


def no_md(e: dict, macros: dict, cache: dict) -> str:
    if not e.get("evidencia"):
        return "—"
    path = e["evidencia"]
    if e["texto"]:
        return "sim" if e["valor"] in (ROOT / path).read_text(encoding="utf-8") else "NÃO"
    if path not in cache:
        cache[path] = numeros_md(path)
    toks = cache[path]
    c, k = e["casas"], e["escala"]

    def achou(v_exib: float) -> bool:
        tol = 0.5 * 10 ** (-c) + 1e-9
        return any(abs(t - v_exib) <= tol or abs(t * k - v_exib) <= tol for t in toks)

    alvo = [round(e["bruto"] * k, c)]
    if e.get("ic_bruto"):
        alvo += [round(x * k, c) for x in e["ic_bruto"]]
    ok = [achou(v) for v in alvo]
    return "sim" if all(ok) else ("parcial" if any(ok) else "não")


# ------------------------------- LaTeX -------------------------------------------------------
def compila(macros: list[str]) -> tuple[bool, str]:
    if not Path(TECTONIC).exists():
        return False, f"tectonic não encontrado em {TECTONIC}"
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(TEX, Path(tmp) / "numeros.tex")
        corpo = "\n".join(f"\\texttt{{{m}}}: \\csname {m}\\endcsname\\par" for m in macros)
        doc = ("\\documentclass[conference]{IEEEtran}\n\\usepackage[T1]{fontenc}\n\\usepackage[utf8]{inputenc}\n"
               "\\usepackage[brazil]{babel}\n\\input{numeros}\n\\begin{document}\n"
               "\\ifSamsungMedido Samsung medido.\\else Samsung: \\SamsungPendente.\\fi\\par\n"
               f"{corpo}\n\\end{{document}}\n")
        (Path(tmp) / "teste.tex").write_text(doc, encoding="utf-8")
        r = subprocess.run([TECTONIC, "--keep-logs", "teste.tex"], cwd=tmp, capture_output=True, text=True)
        log = (Path(tmp) / "teste.log").read_text(encoding="utf-8", errors="replace") if (Path(tmp) / "teste.log").exists() else ""
        ok = r.returncode == 0 and (Path(tmp) / "teste.pdf").exists()
        und = re.findall(r"Undefined control sequence", log)
        return ok and not und, (f"tectonic rc={r.returncode}; {len(und)} 'Undefined control sequence'"
                                + ("" if ok else "\n" + r.stderr[-1500:]))


# ------------------------------- principal ---------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-latex", action="store_true", help="não compila o teste de LaTeX")
    args = ap.parse_args()
    data = json.loads(JSN.read_text(encoding="utf-8"))
    ents = {e["nome"]: e for e in data["entradas"]}
    tex = ler_tex()
    erros, linhas = [], []
    cache: dict = {}
    todos_macros_json = dict(data["valores"])
    for nome, e in ents.items():                          # ordem de geração: dependências antes
        try:
            val, ic, dp = recalcular(e["spec"], ents)
            e["_recalc"] = val
            esp = esperado(e, val, ic, dp)
            status = "ok"
            for mac, txt in esp.items():
                if e["macros"].get(mac) != txt:
                    status = "DIVERGE"
                    erros.append(f"{mac}: JSON {e['macros'].get(mac)!r} × recalculado {txt!r}")
                if mac not in tex:
                    status = "DIVERGE"
                    erros.append(f"{mac}: ausente de numeros.tex")
                elif tex_para_texto(tex[mac]) != txt:
                    status = "DIVERGE"
                    erros.append(f"{mac}: TeX {tex[mac]!r} × recalculado {txt!r}")
            if set(e["macros"]) != set(esp):
                status = "DIVERGE"
                erros.append(f"{nome}: conjunto de macros {sorted(e['macros'])} × {sorted(esp)}")
        except Exception as exc:  # noqa: BLE001
            status = "ERRO"
            erros.append(f"{nome}: {exc}")
            esp = {}
        linhas.append((nome, e["valor"], status, no_md(e, e["macros"], cache), e.get("evidencia") or ""))
    extras_tex = set(tex) - set(todos_macros_json)
    extras_json = set(todos_macros_json) - set(tex)
    for m in sorted(extras_tex):
        erros.append(f"{m}: está em numeros.tex e não no JSON")
    for m in sorted(extras_json):
        erros.append(f"{m}: está no JSON e não em numeros.tex")
    if "SamsungPendente" in tex and tex_para_texto(tex["SamsungPendente"]) != "medição prevista":
        erros.append("SamsungPendente com texto inesperado")
    latex_ok, latex_msg = (True, "não executada (--sem-latex)") if args.sem_latex else compila(sorted(tex))
    if not latex_ok:
        erros.append("compilação de teste: " + latex_msg)

    n_ok = sum(1 for l in linhas if l[2] == "ok")
    n_md = sum(1 for l in linhas if l[3] == "sim")
    n_md_parcial = sum(1 for l in linhas if l[3] == "parcial")
    n_md_nao = [l for l in linhas if l[3] in ("não", "NÃO")]
    now = dt.datetime.now().strftime("%d/%m/%Y %H:%M")
    rel = ["## Conferência automática", "",
           f"Executada em {now} por `src/experiments/conferir_numeros.py` (leitura e formatação independentes das "
           "do gerador).", "",
           f"- **Recalculados a partir do arquivo de origem e idênticos no JSON e no TeX: {n_ok} de {len(linhas)}** números "
           f"({len(tex)} macros em `paper/numeros.tex`, {len(todos_macros_json)} no JSON; "
           f"sobrando só no TeX: {len(extras_tex)}; só no JSON: {len(extras_json)}).",
           f"- Encontrados no `.md` de resultados correspondente (valor e IC com a mesma precisão): {n_md}; "
           f"parcialmente (valor ou só parte do IC): {n_md_parcial}; não encontrados: {len(n_md_nao)}; "
           f"sem `.md` de referência (derivados, agregados entre execuções do celular, regex no próprio .md): "
           f"{sum(1 for l in linhas if l[3] == '—')}.",
           f"- Compilação de teste (IEEEtran + babel brazil, todos os macros impressos): "
           f"{'ok' if latex_ok else 'FALHOU'} — {latex_msg}.",
           f"- Divergências: {len(erros)}.", ""]
    if erros:
        rel += ["### Divergências", ""] + [f"- {x}" for x in erros] + [""]
    if n_md_nao:
        rel += ["### Não encontrados literalmente no .md (conferir à mão; recálculo do CSV está ok)", ""]
        rel += [f"- `\\{l[0]}` = {l[1]} (em `{l[4]}`)" for l in n_md_nao] + [""]
    rel += ["### Por número", "", "| Macro | Valor | Recalculado | No .md |", "|---|---|---|---|"]
    rel += [f"| `\\{a}` | {b} | {c} | {d} |" for a, b, c, d, _ in linhas]
    txt = MD.read_text(encoding="utf-8")
    i = txt.find("## Conferência automática")
    txt = (txt[:i] if i >= 0 else txt + "\n") + "\n".join(rel) + "\n"
    MD.write_text(txt, encoding="utf-8")
    print(f"conferência: {n_ok}/{len(linhas)} ok; no .md {n_md} (+{n_md_parcial} parcial, {len(n_md_nao)} não); "
          f"LaTeX {'ok' if latex_ok else 'FALHOU'}; divergências {len(erros)}")
    for x in erros[:30]:
        print("  -", x)
    return 1 if erros else 0


if __name__ == "__main__":
    sys.exit(main())
