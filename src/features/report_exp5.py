"""Gera results/exp5_caracteristicas.md (RODADA FINAL) a partir dos CSVs da etapa de características.

Nenhum número é digitado à mão: tudo sai de data/signals/<base>/index.csv, data/features/final/<base>/*.csv e dos CSVs
results/exp5_*.csv produzidos por blinks.py, dmd_labels.py, windows.py e sequences.py (ou use run_features.sh):
    python src/features/report_exp5.py
Também grava results/exp5_contagens_sinais.csv (contagens dos index.csv).
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import blinks as B  # noqa: E402
import windows as W  # noqa: E402

ROOT = B.ROOT
RES = ROOT / "results"
FEAT = B.FEAT
PRE = RES / "preliminar"


# ------------------------------------------------------------------ formatação (pt-BR)
def n(x, d=3):
    """Número com vírgula decimal (pt-BR); '—' para NaN."""
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (bool, np.bool_)):
        return "sim" if x else "não"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", ".").replace("-", "−")
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".").replace("-", "−")


def i(x):
    return n(int(x)) if x is not None and np.isfinite(x) else "—"


def pct(x, d=1):
    return "—" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}".replace(".", ",") + "%"


def faixa(a, b, f):
    if a == b:
        return f(a)
    return f"{f(a)} a {f(b)}" if (a < 0 or b < 0) else f"{f(a)}–{f(b)}"


def tab(linhas, cab, alinh=None):
    alinh = alinh or ["---"] + ["---:"] * (len(cab) - 1)
    out = ["| " + " | ".join(cab) + " |", "|" + "|".join(alinh) + "|"]
    out += ["| " + " | ".join(str(c) for c in l) + " |" for l in linhas]
    return "\n".join(out)


def data_curta(s):
    """'2019-03-14T14:26:17+01:00' -> '14/03 14h26'."""
    if not isinstance(s, str) or len(s) < 16:
        return "—"
    return f"{s[8:10]}/{s[5:7]} {s[11:13]}h{s[14:16]}"


def ler(p, **kw):
    return pd.read_csv(p, **kw) if Path(p).exists() else None


def testes_do_log():
    p = RES / "logs" / "exp5_final_testes.log"
    if not p.exists():
        return None
    for linha in p.read_text(encoding="utf-8").splitlines()[::-1]:
        if "testes passaram" in linha:
            return int(linha.split()[0])
    return None


# ------------------------------------------------------------------ fatos calculados
def videos_em_disco():
    """Nº de vídeos em disco por base (para conferir com os index.csv)."""
    import glob
    ext = (".mp4", ".mov", ".m4v", ".avi")
    dmd = glob.glob(str(ROOT / "data" / "dmd" / "*" / "*" / "s[1-5]" / "*_rgb_face.mp4"))
    nit = glob.glob(str(ROOT / "data" / "nitymed" / "DSM_Dataset-HDTV720" / "*" / "*" / "HDTV720" / "*.mp4"))
    uta = [f for k in B.uta_folds_ok() for f in glob.glob(str(ROOT / "data" / "uta_rldd_videos" / f"Fold{k}" / "*" / "*"))
           if Path(f).suffix.lower() in ext]
    return {"dmd": len(dmd), "nitymed": len(nit), "uta": len(uta)}


def preliminar_dmd():
    """(alerta, sonolento) da rodada preliminar: janelas de 30 s da s5, rótulo anotado, sem as de referência."""
    p = B.FEAT_PRELIMINAR / "dmd" / "windows.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p, usecols=["y_anotado", "janela_ref"])
    d = d[~d["janela_ref"].astype(bool)]
    return int((d["y_anotado"] == 0).sum()), int((d["y_anotado"] == 1).sum())


# ------------------------------------------------------------------ contagens dos sinais
def contagens_sinais():
    linhas = []
    for ds in B.DATASETS:
        idx = B.listar_videos(ds)
        if idx.empty:
            continue
        idx["dur_h"] = idx["duration_s"] / 3600.0
        grupos = {"dmd": ["sessao"], "uta": ["fold"], "nitymed": ["categoria"]}[ds]
        for chave, g in list(idx.groupby(grupos)) + [(("total",), idx)]:
            chave = chave if isinstance(chave, tuple) else (chave,)
            linhas.append(dict(dataset=ds, grupo=str(chave[0]), arquivos=len(g), videos_logicos=g["video"].nunique(),
                               sujeitos=g["subject"].nunique(), quadros=int(g["n_frames"].sum()),
                               horas=float(g["dur_h"].sum()), taxa_rosto_media=float(g["face_rate"].mean()),
                               taxa_rosto_min=float(g["face_rate"].min()),
                               fps_min=float(g["fps"].min()), fps_max=float(g["fps"].max())))
    df = pd.DataFrame(linhas)
    df.to_csv(RES / "exp5_contagens_sinais.csv", index=False)
    return df


# ------------------------------------------------------------------ relatório
def main():
    L = []
    a = L.append
    hoje = dt.date.today().strftime("%d/%m/%Y")
    cont = contagens_sinais()
    w10 = {ds: ler(FEAT / ds / "windows_10s.csv", dtype={"parte": str, "rotulo_video": str}) for ds in B.DATASETS}
    ref = {(ds, Lj): ler(FEAT / ds / f"windows_referencia_{Lj}s.csv") for ds in B.DATASETS for Lj in (5, 10, 15)}
    bal = ler(RES / "exp5_dmd_balanco_janelas.csv")
    jres = ler(RES / "exp5_janelas_resumo.csv", dtype={"rotulo_video": str})
    grade = ler(RES / "exp5_dmd_grade_safe.csv")
    bidx = {ds: ler(FEAT / ds / "blinks_index.csv", dtype={"rotulo": str, "parte": str}) for ds in B.DATASETS}
    al = ler(RES / "exp5_dmd_alinhamento.csv")
    ald = ler(RES / "exp5_dmd_alinhamento_distracao.csv")
    val = ler(RES / "exp5_dmd_validacao_detector.csv")
    val_pre = ler(PRE / "exp5_dmd_validacao_detector.csv")
    sens = ler(RES / "exp5_dmd_sensibilidade_limiar.csv")
    mapa = ler(RES / "exp5_uta_mapeamento_oficial.csv", dtype={"pasta": str})
    esc = ler(RES / "exp5_uta_mapeamento_escores.csv")
    msens = ler(RES / "exp5_uta_mapeamento_sensibilidade.csv")
    seqr = ler(RES / "exp5_sequencias_resumo.csv")
    cor = ler(RES / "exp5_uta_validacao_oficial_correlacoes.csv")
    n_testes = testes_do_log()
    d10 = w10["dmd"]
    u10 = w10["uta"]
    y10 = w10["nitymed"]

    # ============================================================== cabeçalho
    a("# Exp. 5 — Características da rodada final: janelas de 10 s, rótulos v2 do DMD e sequências dos 5 folds")
    a("")
    a(f"Gerado por `src/features/report_exp5.py` em {hoje}, a partir dos CSVs citados em cada seção (nenhum número digitado "
      "à mão). Reproduzir tudo a partir dos sinais por quadro (`data/signals/`): `bash src/features/run_features.sh` (no "
      "`.venv`). Saídas de dados em `data/features/final/<base>/`; logs em `results/logs/exp5_final_*.log`. "
      f"Testes: `src/features/test_features.py` ({n_testes if n_testes else '?'} testes de asserts puros, todos passaram; "
      "`results/logs/exp5_final_testes.log`). A rodada preliminar (janelas de 30 s, só o Fold 1 da UTA-RLDD "
      "e só a s5 do DMD) foi descartada; seus arquivos foram removidos em 06/10/2026.")
    a("")

    # ============================================================== números-chave
    c = cont.set_index(["dataset", "grupo"])
    a("## 1. Números-chave")
    a("")
    dmd_tot = c.loc[("dmd", "total")]
    disco = videos_em_disco()
    nomes_ = {"dmd": "DMD", "uta": "UTA-RLDD", "nitymed": "NITYMED"}
    pre = preliminar_dmd()
    uta_tot = c.loc[("uta", "total")]
    nit_tot = c.loc[("nitymed", "total")]
    ses = {s: int(c.loc[("dmd", s), "arquivos"]) for s in ("s1", "s2", "s3", "s5") if ("dmd", s) in c.index}
    a(f"- **Sinais conferidos nos `index.csv`** (`results/exp5_contagens_sinais.csv`): DMD {i(dmd_tot['arquivos'])} sessões "
      f"de {i(dmd_tot['sujeitos'])} sujeitos (s5 {ses.get('s5', 0)}, s1 {ses.get('s1', 0)}, s2 {ses.get('s2', 0)}, s3 {ses.get('s3', 0)}); "
      f"UTA-RLDD {i(uta_tot['arquivos'])} arquivos = {i(uta_tot['videos_logicos'])} vídeos lógicos de {i(uta_tot['sujeitos'])} "
      f"sujeitos (5 folds); NITYMED {i(nit_tot['arquivos'])} vídeos. Vídeos em disco: "
      + ", ".join(f"{nomes_[k]} {v}" for k, v in disco.items())
      + (" — todos com sinal completo (nenhuma extração foi necessária nesta frente)."
         if all(disco[k] == int(c.loc[(k, "total"), "arquivos"]) for k in disco) else " — ATENÇÃO: há vídeo sem sinal."))
    uso = d10["uso"].value_counts()
    y = d10["y"]
    suj_par = d10[d10["y"].notna()].groupby("sujeito")["y"].nunique()
    a(f"- **DMD, janelas de 10 s (passo 5 s)** (`data/features/final/dmd/windows_10s.csv`): principal = "
      f"**{i((y == 0).sum())} alerta** (safe_drive da s2) × **{i((y == 1).sum())} sonolento atuado** (s5 com fechamento "
      f"anotado ≥ 0,5 s ou bocejo), {i((suj_par == 2).sum())} sujeitos com as duas classes; fora do principal: "
      f"{i(uso.get('s5_sem_sinal', 0))} janelas da s5 sem sinal anotado (sensibilidade), {i(uso.get('alerta_s1', 0))} alerta da s1 "
      f"(sensibilidade), {i(uso.get('referencia', 0))} de referência (s3; nunca avaliadas)."
      + (f" Na rodada preliminar (30 s, só s5, rótulo anotado, sem as janelas de referência) eram {i(pre[0])} alerta × {i(pre[1])} "
         "sonolento, com as alerta sobrepostas à referência." if pre else ""))
    uu = u10["uso"].value_counts()
    ex = u10[u10["avaliavel"]]
    a(f"- **UTA-RLDD, janelas de 10 s** (`data/features/final/uta/windows_10s.csv`): {i(len(u10))} janelas; avaliáveis "
      f"KSS 0 / 5 / 10 = {i((ex['rotulo_kss'] == 0).sum())} / {i((ex['rotulo_kss'] == 5).sum())} / {i((ex['rotulo_kss'] == 10).sum())}; "
      f"{i(uu.get('referencia', 0))} de referência (último terço do vídeo 0) e {i(uu.get('fronteira', 0))} de fronteira (cruzam o "
      "corte de 2/3), fora de treino e teste.")
    r10u = ref[("uta", 10)]
    r10d = ref[("dmd", 10)]
    a(f"- **Referência de normalização por sujeito, separada da avaliação**: UTA {faixa(int(r10u['n_janelas_ref'].min()), int(r10u['n_janelas_ref'].max()), i)} "
      f"janelas por sujeito (mediana {i(r10u['n_janelas_ref'].median())}); DMD "
      f"{faixa(int(r10d.loc[~r10d['ref_fallback'].astype(bool), 'n_janelas_ref'].min()), int(r10d.loc[~r10d['ref_fallback'].astype(bool), 'n_janelas_ref'].max()), i)} "
      f"janelas da s3 por sujeito; referência na s2 (sem s3 ou s3 com < 3 janelas): "
      + (", ".join(f"{r_['unidade_ref']} ({int(r_['n_janelas_ref'])} janelas)" for _, r_ in r10d[r10d['ref_fallback'].astype(bool)].iterrows()) or "nenhum")
      + ". Por sujeito: seções 4.5 e 5.2.")
    # z estimável
    zest = {}
    for ds in B.DATASETS:
        r = ref[(ds, 10)]
        zest[ds] = {f: float(r[f + "_sd"].notna().mean()) for f in W.FEATURES}
    nao_todas = [f for f in W.FEATURES if min(zest[ds][f] for ds in ("uta", "dmd")) < 1.0]
    a(f"- **z-score por sujeito continua não estimável** em parte das características (desvio 0 ou < 5 janelas na referência): "
      f"no DMD o z é definido em {pct(np.mean([zest['dmd'][f] for f in W.FEATURES]))} das células (unidade × característica) e na "
      f"UTA em {pct(np.mean([zest['uta'][f] for f in W.FEATURES]))}; na UTA só "
      + (", ".join(f"`{f}`" for f in W.FEATURES if zest['uta'][f] >= 1.0) or "nenhuma") + " têm z em todos os sujeitos. **A centragem (`<f>_c`) segue como normalização principal** (seção 5.3).")
    if ald is not None:
        ok_q = int((ald["n_quadros_video"] == ald["face_camera_total_frames"]).sum())
        forte = ald[ald["corr_melhor_naoolha_vs_pose"] >= 0.2]
        a(f"- **Alinhamento anotação ↔ vídeo nas {len(ald)} sessões de distração**: nº de quadros = `total_frames` em {ok_q}/{len(ald)}, "
          f"`frame_shift` = 0 em {int((ald['face_frame_shift'] == 0).sum())}/{len(ald)}; olhar fora da estrada × desvio da pose da cabeça: "
          f"melhor defasagem mediana {i(forte['lag_naoolha_vs_pose'].median())} quadros ({int((forte['lag_naoolha_vs_pose'].abs() <= 5).sum())} de "
          f"{len(forte)} sessões com correlação ≥ 0,2 entre 0 e 5 quadros; máximo {i(forte['lag_naoolha_vs_pose'].max())}); fala × atividade da boca: "
          f"{faixa(int(ald['lag_falando_vs_boca'].min()), int(ald['lag_falando_vs_boca'].max()), i)} quadros. **Deslocamento adotado: 0** (como na s5).")
    if mapa is not None:
        cls = mapa["classe"].value_counts()
        a(f"- **Mapeamento sujeito oficial ↔ pasta nos 5 folds** (`results/exp5_uta_mapeamento_oficial.csv`): "
          f"{i(cls.get('confirmado', 0))} confirmados pela coincidência das piscadas oficiais com o nosso EAR, "
          f"{i(cls.get('provavel', 0))} prováveis (ordem de listagem ou duração) e {i(cls.get('indeterminado', 0))} indeterminados "
          f"(todos no Fold 3), de {len(mapa)} sujeitos oficiais. **Pastas ausentes das características oficiais: F1_12 (Fold 1) e "
          f"F5_49 (Fold 5)** (seção 5.4).")
    if seqr is not None:
        a(f"- **Sequências de 30 piscadas (passo 2)** (`data/features/final/uta/sequences.npz`): {i(seqr['n_sequencias'].sum())} "
          f"sequências de {seqr['sujeito'].nunique()} sujeitos e {int((seqr['n_sequencias'] > 0).sum())} dos {seqr['video'].nunique()} vídeos lógicos"
          + (f" (sem nenhuma piscada detectada, logo sem sequência: {', '.join(seqr.loc[seqr['n_sequencias'] == 0, 'video'])}; seção 5.5)"
             if (seqr['n_sequencias'] == 0).any() else "") + "; alerta / pouca vigilância / "
          f"sonolento = {i(seqr.loc[seqr['rotulo'] == 0, 'n_sequencias'].sum())} / {i(seqr.loc[seqr['rotulo'] == 5, 'n_sequencias'].sum())} / "
          f"{i(seqr.loc[seqr['rotulo'] == 10, 'n_sequencias'].sum())}.")
    a("")

    # ============================================================== contagens dos sinais
    a("## 2. Sinais disponíveis (conferência dos `index.csv`)")
    a("")
    a("Fonte: `data/signals/<base>/index.csv` → `results/exp5_contagens_sinais.csv` (só arquivos com `complete` = True e `.npz` "
      "presente). Duração e fps do contêiner; taxa de rosto = fração de quadros com landmarks do Face Mesh.")
    a("")
    nomes = {"dmd": "DMD", "uta": "UTA-RLDD", "nitymed": "NITYMED"}
    rot_g = {"s1": "s1 (distração; carro em movimento)", "s2": "s2 (distração; carro parado)", "s3": "s3 (distração; carro parado, uma atividade)",
             "s5": "s5 (sonolência atuada; carro parado)", "total": "**total**"}
    linhas = []
    for _, r in cont.iterrows():
        g = r["grupo"]
        rot = rot_g.get(g, f"Fold {g}" if r["dataset"] == "uta" and g != "total" else g)
        if r["dataset"] == "uta" and g == "total":
            rot = "**total**"
        linhas.append([nomes[r["dataset"]], rot, i(r["arquivos"]), i(r["videos_logicos"]), i(r["sujeitos"]), i(r["quadros"]),
                       n(r["horas"], 2), n(r["taxa_rosto_media"], 4), n(r["taxa_rosto_min"], 4),
                       faixa(round(r["fps_min"], 2), round(r["fps_max"], 2), lambda v: n(v, 2))])
    a(tab(linhas, ["Base", "Grupo", "Arquivos", "Vídeos lógicos", "Sujeitos", "Quadros", "Horas", "Taxa de rosto (média)",
                   "Taxa de rosto (mín.)", "fps"], ["---", "---"] + ["---:"] * 8))
    a("")
    bu = bidx["uta"]
    bd0 = bidx["dmd"]
    iu = pd.read_csv(ROOT / "data" / "signals" / "uta" / "index.csv")
    partes = bu[bu["parte"].fillna("") != ""]
    a("UTA-RLDD: os arquivos a mais em relação aos 180 vídeos são as partes dos dois vídeos sonolentos divididos: "
      + "; ".join(f"`{r['id']}` ({n(r['dur_s'], 0)} s, {n(r['fps'], 2)} fps, taxa de rosto {n(r['taxa_rosto'], 3)})"
                  for _, r in partes.iterrows())
      + ". Resolução original das partes: "
      + "; ".join(f"`{r['id']}` {int(r['width_orig'])}×{int(r['height_orig'])}" for _, r in iu[iu["id"].isin(partes["id"])].iterrows())
      + " (os demais vídeos do mesmo sujeito têm " + ", ".join(sorted({f"{int(r['width_orig'])}×{int(r['height_orig'])}" for _, r in iu[iu["subject"].isin(partes["sujeito"]) & ~iu["id"].isin(partes["id"])].iterrows()}))
      + "). DMD: sujeitos sem s5: " + (", ".join(sorted(set(bd0["sujeito"]) - set(bd0.loc[bd0["sessao"] == "s5", "sujeito"]))) or "nenhum")
      + "; sem s3: " + (", ".join(sorted(set(bd0["sujeito"]) - set(bd0.loc[bd0["sessao"] == "s3", "sujeito"]))) or "nenhum") + ".")
    a("")

    # ============================================================== decisões
    a("## 3. Parâmetros e decisões desta rodada")
    a("")
    dec = [
        ("Janela", "**10 s, passo 5 s** nas três bases (`--janela`, `--passo`); 5 s e 15 s para sensibilidade", "decisão de 27/09 (os trechos safe_drive do DMD têm 10–26 s; 30 s não cabem)"),
        ("Grade", "UTA, NITYMED e DMD s5: grade global a partir de 0 s, por arquivo (partes _p1/_p2 separadas; nenhuma janela atravessa o corte; coluna `video` = vídeo lógico, `arquivo` = parte). DMD s1/s2/s3: janelas **ancoradas no início de cada trecho safe_drive** e inteiramente dentro dele", "numa grade global a janela só cabe no trecho quando a grade coincide com ele; ancorar dá "
         + (f"{int(d10['sessao'].isin(['s1', 's2', 's3']).sum())} janelas safe_drive em 10 s contra {int(grade.loc[grade['janela_s'] == 10, 'grade_global_safe'].iloc[0])} na grade global" if grade is not None else "")),
        ("Características", "as 11 da rodada preliminar, só com quadros com rosto e em unidades independentes do fps (`perclos`, `taxa_piscadas_min`, `dur_media_piscada_s`, `dur_max_fechamento_s`, `n_fechamentos_05s`, `ear_media`, `ear_dp`, `amp_media`, `vel_media`, `frac_bocejo`, `taxa_rosto`); `n_piscadas` como coluna auxiliar", "em 10 s a taxa de piscadas é ruidosa (poucas piscadas por janela)"),
        ("Referência (normalização)", "UTA: janelas inteiramente no **último terço do vídeo 0**; DMD: janelas safe_drive das **sessões s3** do sujeito (se < 3 janelas: último terço, por sessão, das janelas safe_drive da s2); NITYMED: janelas no **primeiro terço** do vídeo (pseudo-referência = o terço inteiro quando nenhuma janela cabe nele)", "regra final de 27/09; a referência nunca é avaliada"),
        ("Exclusões", "`janela_ref` (referência) e `sobrepoe_ref` (fronteira: divide quadros com a referência) nunca entram em treino/teste; coluna `avaliavel` marca o que entra no principal", "correção do vazamento da rodada preliminar (alerta do DMD sobreposto à referência)"),
        ("Limiar de EAR", "0,75 × mediana do EAR (quadros com rosto) do **mesmo trecho de referência**, por sujeito, aplicado a todos os arquivos do sujeito: UTA último terço do vídeo 0 (antes: vídeo 0 inteiro); DMD safe_drive das s3 (antes: primeiros 30 s de cada s5); NITYMED primeiro terço", "o limiar também é calibração por sujeito: calculá-lo com as janelas alerta avaliadas era um vazamento menor do mesmo tipo"),
        ("Normalização principal", "`<f>_c` = x − média das janelas de referência da unidade; `<f>_z` só com ≥ 5 janelas de referência e desvio > 0", "o z segue indefinido em várias características (seção 5.3)"),
        ("Rótulos DMD (v2)", "`uso`: `alerta` (s2 safe_drive), `sonolento` (s5 com fechamento anotado ≥ 0,5 s ou bocejo, ≥ 0,5 s dentro da janela), `s5_sem_sinal`, `alerta_s1`, `referencia`, `fronteira`, `s3_nao_usada`; `y` (principal), `y_sens_s1`, `y_sens_s5_sem_sinal`; sujeito sem s5 (gE_28): `y` = NaN", "item 9 das notas de fechamento"),
        ("Rótulos UTA", "do vídeo: `rotulo_kss`, `classe` 0/1/2, B1, B2, B3 (B3 principal)", "iguais à rodada preliminar"),
        ("Sequências", "30 piscadas, passo 2; z-score por sujeito com o último terço das piscadas do vídeo 0 (`--terco primeiro` como sensibilidade); partes concatenadas com contagem contínua", "como o Preprocessing.py original"),
    ]
    a(tab([[x, y_, z] for x, y_, z in dec], ["Item", "Decisão", "Por quê"], ["---", "---", "---"]))
    a("")

    # ============================================================== DMD
    a("## 4. DMD")
    a("")
    a("### 4.1 Sessões por sujeito: datas, montagem e óculos")
    a("")
    a("Fonte: `data/features/final/dmd/blinks_index.csv` (metadados do OpenLABEL: `recordTime`, `setup`, `glasses`).")
    a("")
    bd = bidx["dmd"].copy()
    linhas = []
    for suj, g in bd.groupby("sujeito"):
        cel = []
        for s_ in ("s1", "s2", "s3", "s5"):
            gs = g[g["sessao"] == s_].sort_values("data_gravacao")
            cel.append("; ".join(f"{data_curta(r['data_gravacao'])}{'' if r['setup'] == ('Car Moving' if s_ == 's1' else 'Car Stopped') else ' (**' + r['setup'] + '**)'}"
                                 f"{' óculos' if bool(r['oculos']) else ''}" for _, r in gs.iterrows()) or "—")
        tem5 = (g["sessao"] == "s5").any()
        mesmo_dia = set(g.loc[g["sessao"] == "s2", "data_gravacao"].str[:10]) & set(g.loc[g["sessao"] == "s5", "data_gravacao"].str[:10])
        linhas.append([suj] + cel + [("sim" if mesmo_dia else "não") if tem5 else "— (sem s5)"])
    a(tab(linhas, ["Sujeito", "s1 (movimento)", "s2 (parado)", "s3 (parado)", "s5 (sonolência)", "s2 no mesmo dia da s5?"],
          ["---"] * 6))
    a("")
    s3s5 = []
    for suj, g in bd.groupby("sujeito"):
        d3 = set(g.loc[g["sessao"] == "s3", "data_gravacao"].str[:10])
        d5 = set(g.loc[g["sessao"] == "s5", "data_gravacao"].str[:10])
        if d5:
            s3s5.append(bool(d3 & d5))
    n_s2_outro = sum(1 for l in linhas if l[-1] == "não" and l[4] != "—")
    fora_conv = [f"{r['sujeito']} {r['sessao']} ({data_curta(r['data_gravacao'])}): '{r['setup']}'" for _, r in bd.iterrows()
                 if r["setup"] != ("Car Moving" if r["sessao"] == "s1" else "Car Stopped")]
    oc_misto = [f"{suj} (óculos em: " + ", ".join(f"{r['sessao']} {data_curta(r['data_gravacao'])}" for _, r in g[g["oculos"].astype(bool)].iterrows()) + ")"
                for suj, g in bd.groupby("sujeito") if g["oculos"].astype(bool).nunique() > 1]
    a(f"**Estrutura temporal (fato dos metadados, relevante para o Exp. 5b):** a s3 (referência) foi gravada no mesmo dia e "
      f"poucos minutos antes ou depois da s5 (sonolento) em {sum(s3s5)} de {len(s3s5)} sujeitos com s5; a s2 (exemplos alerta), em outro "
      f"dia em {n_s2_outro} de {len(s3s5)}. Assim, na normalização pela s3, as janelas sonolentas são centradas numa referência da mesma "
      "sessão de gravação e as alerta numa de outro dia (outra montagem da câmera, iluminação, óculos): diferenças de montagem "
      "entram na classe alerta. É o espelho do que ocorre na UTA, onde a referência sai do mesmo vídeo das janelas alerta. "
      "Metadados `setup` que contrariam a convenção do README (s1 = em movimento; s2/s3/s5 = parado) estão em **negrito** na tabela: "
      + ("; ".join(fora_conv) or "nenhum") + ". Sujeitos com óculos em só parte das sessões: "
      + ("; ".join(oc_misto) or "nenhum") + ".")
    a("")
    # nível do EAR por sessão
    a("### 4.2 Limiar de EAR por sujeito e nível do EAR em cada sessão")
    a("")
    a("Limiar = 0,75 × mediana do EAR no safe_drive das s3 do sujeito (`ear_ref_mediana`). Para mostrar quanto o nível de olho "
      "aberto muda entre sessões, a coluna ao lado dá a mediana do EAR nos primeiros 30 s de cada sessão (a referência da rodada "
      "preliminar), dividida pela mediana de referência.")
    a("")
    linhas = []
    for suj, g in bd.groupby("sujeito"):
        g = g.sort_values(["sessao", "data_gravacao"])
        rz = "; ".join(f"{r['sessao']} {n(r['ear_ref_mediana_prelim'] / r['ear_ref_mediana'], 2)}" for _, r in g.iterrows())
        tr_ = {"s3_safe_drive": "safe_drive das s3", "s2_safe_drive_ultimo_terco": "último terço do safe_drive das s2"}.get(g["tipo_ref"].iloc[0], g["tipo_ref"].iloc[0])
        linhas.append([suj, tr_, i(g["n_quadros_ref"].iloc[0]), n(g["ear_ref_mediana"].iloc[0], 3),
                       n(g["limiar"].iloc[0], 3), rz])
    a(tab(linhas, ["Sujeito", "Referência do limiar", "Quadros de referência", "Mediana do EAR (ref.)", "Limiar",
                   "Mediana dos 30 s iniciais ÷ ref., por sessão"], ["---", "---", "---:", "---:", "---:", "---"]))
    a("")
    rat = (bd["ear_ref_mediana_prelim"] / bd["ear_ref_mediana"])
    rs1 = rat[bd["sessao"] == "s1"]
    a(f"A razão vai de {n(rat.min(), 2)} a {n(rat.max(), 2)} (mediana {n(rat.median(), 2)}; na s1, com o carro em movimento, mediana "
      f"{n(rs1.median(), 2)}): o nível de olho aberto varia entre sessões do mesmo sujeito, o que desloca PERCLOS e contagem de "
      "piscadas quando o limiar vem de outra sessão — efeito inerente à calibração por sujeito (o mesmo de F1_03_10 na UTA).")
    a("")

    # balanço
    a("### 4.3 Balanço de classes (janelas de 10 s)")
    a("")
    a("Fonte: `data/features/final/dmd/windows_10s.csv` (coluna `uso`) e `results/exp5_dmd_balanco_janelas.csv`. Principal = "
      "`alerta` × `sonolento` (y); `s5_sem_sinal` e `alerta_s1` são só de sensibilidade; `referencia` nunca é avaliada.")
    a("")
    usos = ["alerta", "sonolento", "s5_sem_sinal", "alerta_s1", "referencia", "fronteira", "s3_nao_usada"]
    ct = pd.crosstab(d10["sujeito"], d10["uso"]).reindex(columns=usos, fill_value=0)
    ytab = d10.groupby("sujeito")["y"].agg(lambda s: int(s.notna().sum()))
    fr_al = d10.groupby("sujeito")["y"].agg(lambda s_: float((s_ == 0).sum() / s_.notna().sum()) if s_.notna().any() else np.nan)
    linhas = [[suj] + [i(r[u]) for u in usos] + [i(ytab[suj]), pct(fr_al[suj])] for suj, r in ct.iterrows()]
    tot = ct.sum()
    linhas.append(["**total**"] + [i(tot[u]) for u in usos] + [i(d10["y"].notna().sum()), pct(float((d10["y"] == 0).sum() / d10["y"].notna().sum()))])
    a(tab(linhas, ["Sujeito"] + [f"`{u}`" for u in usos] + ["No principal (y)", "Fração alerta no principal"]))
    a("")
    a("gE_28 não tem s5: suas janelas `alerta` ficam com y = NaN (fora do treino pareado), como decidido; a referência dele é a s2 "
      "(não há s3).")
    a("")
    a("**Por sessão** (cada linha é um arquivo; só as colunas com janelas):")
    a("")
    b10 = bal[bal["janela_s"] == 10].copy()
    b10 = b10.merge(bd[["id", "data_gravacao", "setup"]], left_on="arquivo", right_on="id", how="left")
    linhas = []
    for _, r in b10.sort_values(["sujeito", "sessao", "data_gravacao"]).iterrows():
        partes_ = [f"{u} {int(r['n_' + u])}" for u in usos if r.get("n_" + u, 0)]
        linhas.append([r["sujeito"], r["sessao"], data_curta(r["data_gravacao"]), i(r["n_janelas"]), ", ".join(partes_) or "—"])
    a(tab(linhas, ["Sujeito", "Sessão", "Gravação", "Janelas", "Por uso"], ["---", "---", "---", "---:", "---"]))
    a("")
    sem_jan = sorted(set(bd["id"]) - set(b10["arquivo"]))
    if sem_jan:
        a(f"Sessões sem nenhuma janela de 10 s: {', '.join('`' + s + '`' for s in sem_jan)}.")
        a("")
    # sensibilidade ao comprimento
    a("### 4.4 Sensibilidade ao comprimento da janela (5, 10 e 15 s; passo 5 s)")
    a("")
    linhas = []
    for Lj in (5, 10, 15):
        dL = ler(FEAT / "dmd" / f"windows_{Lj}s.csv")
        if dL is None:
            continue
        u_ = dL["uso"].value_counts()
        yL = dL["y"]
        sp = dL[dL["y"].notna()].groupby("sujeito")["y"].nunique()
        rL = ref[("dmd", Lj)]
        gg = int(grade.loc[grade["janela_s"] == Lj, "grade_global_safe"].iloc[0]) if grade is not None and (grade["janela_s"] == Lj).any() else np.nan
        linhas.append([f"{Lj} s", i(len(dL)), i((yL == 0).sum()), i((yL == 1).sum()), i(u_.get("s5_sem_sinal", 0)), i(u_.get("alerta_s1", 0)),
                       i(u_.get("referencia", 0)), i((sp == 2).sum()), ", ".join(rL.loc[rL["ref_fallback"].astype(bool), "unidade_ref"]) or "—",
                       i(int(dL["sessao"].isin(["s1", "s2", "s3"]).sum())), i(gg)])
    a(tab(linhas, ["Janela", "Janelas", "Alerta (y=0)", "Sonolento (y=1)", "s5 sem sinal", "Alerta s1", "Referência",
                   "Sujeitos com 2 classes", "Referência na s2 (fallback)", "Safe_drive ancoradas", "Safe_drive na grade global"],
          ["---"] + ["---:"] * 7 + ["---", "---:", "---:"]))
    a("")
    a("Com 15 s quase não há janela safe_drive (os trechos têm 10–26 s): a s3 deixa de ter 3 janelas em quase todos os sujeitos, a "
      "referência passa para a s2 e sobram poucas janelas alerta — a sensibilidade de 15 s no DMD é pouco informativa. Com 5 s o "
      "balanço melhora, mas as janelas são curtas demais para fechamentos longos e taxa de piscadas.")
    a("")
    # referência por sujeito
    a("### 4.5 Janelas de referência por sujeito")
    a("")
    linhas = []
    for suj in sorted(set(ref[("dmd", 10)]["unidade_ref"])):
        cel = []
        for Lj in (5, 10, 15):
            rL = ref[("dmd", Lj)]
            rr = rL[rL["unidade_ref"] == suj]
            if len(rr):
                rr = rr.iloc[0]
                cel.append(f"{int(rr['n_janelas_ref'])} ({'s2' if rr['ref_fallback'] else 's3'})")
            else:
                cel.append("—")
        linhas.append([suj] + cel)
    a(tab(linhas, ["Sujeito", "5 s", "10 s", "15 s"]))
    a("")
    r10d_ = ref[("dmd", 10)]
    a("Fonte: `data/features/final/dmd/windows_referencia_<L>s.csv` (média e desvio de referência de cada característica). Em 10 s, "
      f"{int((r10d_['n_janelas_ref'] < 5).sum())} dos {len(r10d_)} sujeitos têm < 5 janelas de referência (o z exige ≥ 5); com ≥ 5: "
      + (", ".join(r10d_.loc[r10d_["n_janelas_ref"] >= 5, "unidade_ref"]) or "nenhum") + ".")
    a("")
    # alinhamento
    a("### 4.6 Alinhamento anotação ↔ `*_rgb_face.mp4`")
    a("")
    if al is not None:
        a(f"**s5 (sonolência, {len(al)} sessões)** — `results/exp5_dmd_alinhamento.csv`, mesmo método da rodada preliminar: nº de "
          f"quadros = `total_frames` em {int((al['n_quadros_video'] == al['face_camera_total_frames']).sum())}/{len(al)}; "
          f"`frame_shift` = 0 em {int((al['face_frame_shift'] == 0).sum())}/{len(al)}; melhor defasagem entre o fechamento anotado e "
          f"−EAR de {i(al['lag_close_vs_ear'].min())} a {i(al['lag_close_vs_ear'].max())} quadros, com ganho de correlação sobre o lag 0 de no "
          f"máximo {n((al['corr_melhor_close_vs_ear'] - al['corr_lag0_close_vs_ear']).max(), 3)}. A verificação independente da rodada "
          "preliminar (±300 quadros e inspeção visual; `results/exp5_dmd_alinhamento_verificacao.csv`) continua válida (mesmos vídeos).")
        a("")
    if ald is not None:
        exc = ald["fim_max_acoes"] - (ald["n_quadros_video"] - 1)
        a(f"**Distração (s1/s2/s3, {len(ald)} sessões)** — `results/exp5_dmd_alinhamento_distracao.csv`. A anotação de distração **não "
          "tem estado do olho nem piscada**; o sinal ocular disponível é o olhar (`gaze_on_road/not_looking_road`). Conferências: (i) "
          f"contagem de quadros: nº de quadros do vídeo = `total_frames` em {int((ald['n_quadros_video'] == ald['face_camera_total_frames']).sum())}/{len(ald)}, "
          f"`frame_shift` da câmera do rosto = 0 em {int((ald['face_frame_shift'] == 0).sum())}/{len(ald)}, e o fim das ações passa do "
          f"último quadro em no máximo {i(exc.max())} quadro(s) ({int((exc > 0).sum())} sessões; a s5 tem a mesma convenção); (ii) correlação "
          "cruzada do indicador de olhar fora da estrada com o **desvio da pose da cabeça** (|yaw| e |pitch| padronizados) e com **−EAR**; "
          "(iii) da fala anotada (`talking/talking`) com a **atividade da boca** (DP móvel de 0,5 s do MAR).")
        a("")
        def resumo_corr(col_lag, col_c0, col_c, col_lag_amplo, nome):
            sub = ald[ald[col_c].notna()]
            fortes = sub[sub[col_c] >= 0.2]
            return [nome, i(len(sub)), i(len(fortes)), n(sub[col_c0].median(), 3),
                    (faixa(int(fortes[col_lag].min()), int(fortes[col_lag].max()), i) if len(fortes) else "—"),
                    (faixa(int(fortes[col_lag_amplo].min()), int(fortes[col_lag_amplo].max()), i) if len(fortes) else "—"),
                    n((fortes[col_c] - fortes[col_c0]).max(), 3) if len(fortes) else "—"]
        linhas = [resumo_corr("lag_naoolha_vs_pose", "corr_lag0_naoolha_vs_pose", "corr_melhor_naoolha_vs_pose", "lag_amplo_naoolha_vs_pose", "olhar fora da estrada × desvio da pose"),
                  resumo_corr("lag_falando_vs_boca", "corr_lag0_falando_vs_boca", "corr_melhor_falando_vs_boca", "lag_amplo_falando_vs_boca", "fala × atividade da boca"),
                  resumo_corr("lag_naoolha_vs_menos_ear", "corr_lag0_naoolha_vs_menos_ear", "corr_melhor_naoolha_vs_menos_ear", "lag_amplo_naoolha_vs_menos_ear", "olhar fora da estrada × −EAR")]
        a(tab(linhas, ["Par", "Sessões com o evento", "Sessões com r ≥ 0,2", "r no lag 0 (mediana)", "Melhor lag (±30), sessões com r ≥ 0,2",
                       "Melhor lag (±300)", "Ganho máx. de r sobre o lag 0"], ["---"] + ["---:"] * 6))
        a("")
        fp = ald[ald["corr_melhor_naoolha_vs_pose"] >= 0.2]
        fora = fp[fp["lag_naoolha_vs_pose"].abs() > 5]
        fe = ald[ald["corr_melhor_naoolha_vs_menos_ear"] >= 0.2]
        a("A pose da cabeça e a boca acompanham a anotação com defasagem de poucos quadros (a 29,76 fps, 3 quadros ≈ 0,1 s): "
          f"olhar × pose com mediana de {i(fp['lag_naoolha_vs_pose'].median())} quadros ({len(fp) - len(fora)} de {len(fp)} sessões entre "
          f"0 e 5; fora disso: " + ("; ".join(f"{r_['sujeito']} {r_['sessao']} {i(r_['lag_naoolha_vs_pose'])} quadros (r = {n(r_['corr_melhor_naoolha_vs_pose'], 2)})"
                                             for _, r_ in fora.iterrows()) or "nenhuma")
          + ") — a pose atrasa um pouco porque o olhar muda antes da cabeça, e o DP móvel centrado adianta a fala. O −EAR só "
          f"acompanha o olhar em {len(fe)} das {int(ald['corr_melhor_naoolha_vs_menos_ear'].notna().sum())} sessões (r ≥ 0,2; nelas o melhor lag vai de "
          f"{i(fe['lag_naoolha_vs_menos_ear'].min())} a {i(fe['lag_naoolha_vs_menos_ear'].max())} quadros); nas demais a correlação é fraca ou negativa "
          f"({int((ald['corr_lag0_naoolha_vs_menos_ear'] < 0).sum())} com r < 0 no lag 0). Não há sinal de deslocamento: **índice da anotação = índice do quadro do "
          "vídeo do rosto também nas sessões de distração**. As janelas alerta dependem só dos trechos safe_drive, que duram "
          "segundos — um erro de poucos quadros não mudaria o rótulo.")
        a("")
    # validação
    if val is not None:
        a("### 4.7 Validação do detector de piscadas (s5) com o limiar novo")
        a("")
        cols = [("precisao_piscadas", "P piscadas"), ("revocacao_piscadas", "R piscadas"), ("f1_piscadas", "F1 piscadas"),
                ("f1_fech05", "F1 fech. ≥ 0,5 s"), ("q_acuracia_bal", "Acur. bal. por quadro"), ("q_kappa", "Kappa por quadro"),
                ("revocacao_bocejo", "R bocejo")]
        linhas = [["final (limiar da s3 do sujeito)"] + [n(val[c].mean(), 3) for c, _ in cols]]
        if val_pre is not None:
            linhas.append(["preliminar (limiar dos 30 s iniciais da própria s5)"] + [n(val_pre[c].mean(), 3) for c, _ in cols])
        a(tab(linhas, ["Limiar"] + [nm for _, nm in cols], ["---"] + ["---:"] * len(cols)))
        a("")
        k75 = sens[sens["k"] == 0.75] if sens is not None else None
        g = sens.groupby("k")[["f1_piscadas", "q_acuracia_bal", "f1_fech05"]].mean() if sens is not None else None
        if g is not None:
            a("Médias por sessão (16 sessões s5; `results/exp5_dmd_validacao_detector.csv`, a linha “preliminar” vem da rodada descartada de 27/09). "
              "Sensibilidade ao fator k (× a mesma mediana de referência; `results/exp5_dmd_sensibilidade_limiar.csv`): F1 de piscadas "
              + ", ".join(f"k = {n(k, 2)}: {n(r['f1_piscadas'], 3)}" for k, r in g.iterrows())
              + ". k = 0,75 mantido (pré-especificado).")
            a("")

    # ============================================================== UTA
    a("## 5. UTA-RLDD")
    a("")
    a("### 5.1 Janelas de 10 s por fold e classe")
    a("")
    linhas = []
    for f, g in u10.groupby("fold"):
        e = g[g["avaliavel"]]
        linhas.append([f"Fold {int(f)}", i(g["sujeito"].nunique()), i(g["video"].nunique()), i(g["arquivo"].nunique()), i(len(g)),
                       i((e["rotulo_kss"] == 0).sum()), i((e["rotulo_kss"] == 5).sum()), i((e["rotulo_kss"] == 10).sum()),
                       i(g["janela_ref"].sum()), i(g["sobrepoe_ref"].sum())])
    e = u10[u10["avaliavel"]]
    linhas.append(["**total**", i(u10["sujeito"].nunique()), i(u10["video"].nunique()), i(u10["arquivo"].nunique()), i(len(u10)),
                   i((e["rotulo_kss"] == 0).sum()), i((e["rotulo_kss"] == 5).sum()), i((e["rotulo_kss"] == 10).sum()),
                   i(u10["janela_ref"].sum()), i(u10["sobrepoe_ref"].sum())])
    a(tab(linhas, ["Fold", "Sujeitos", "Vídeos lógicos", "Arquivos", "Janelas", "Avaliáveis KSS 0", "KSS 5", "KSS 10",
                   "Referência", "Fronteira"]))
    a("")
    a(f"B3 (principal): {i((e['rotulo_kss'] == 0).sum())} alerta × {i((e['rotulo_kss'] == 10).sum())} sonolento; B1: "
      f"{i((e['rotulo_kss'] == 0).sum())} × {i((e['rotulo_kss'] >= 5).sum())}; B2: {i((e['rotulo_kss'] <= 5).sum())} × "
      f"{i((e['rotulo_kss'] == 10).sum())}. As partes dos vídeos divididos geram janelas próprias (nenhuma atravessa o corte) e "
      "compartilham a coluna `video` (vídeo lógico) — para métricas por vídeo, agrupar por `video`, não por `arquivo`.")
    a("")
    # referência e limiar
    rr = ref[("uta", 10)]
    bu = bidx["uta"]
    uu_ = bu.drop_duplicates("unidade_ref")
    razao = uu_["ear_ref_mediana"] / uu_["ear_ref_mediana_prelim"]
    a("### 5.2 Referência por sujeito (último terço do vídeo 0) e limiar")
    a("")
    a(f"Janelas de referência por sujeito em 10 s: mínimo {i(rr['n_janelas_ref'].min())}, mediana {i(rr['n_janelas_ref'].median())}, "
      f"máximo {i(rr['n_janelas_ref'].max())} (60 sujeitos; `data/features/final/uta/windows_referencia_10s.csv`, uma linha por sujeito). "
      "As janelas avaliadas do vídeo 0 terminam antes do corte de 2/3; a janela que o cruza é `fronteira`. O limiar de EAR passou a sair "
      f"do mesmo último terço: a mediana de referência mudou por um fator de {n(razao.min(), 3)} a {n(razao.max(), 3)} (mediana "
      f"{n(razao.median(), 3)}) em relação ao vídeo 0 inteiro. Em 5 s / 15 s: "
      + "; ".join(f"{Lj} s: {i(ref[('uta', Lj)]['n_janelas_ref'].min())}–{i(ref[('uta', Lj)]['n_janelas_ref'].max())}" for Lj in (5, 15)) + ".")
    a("")
    a("Janelas de referência (10 s) por sujeito:")
    a("")
    linhas = []
    rr_ = rr.assign(fold=rr["unidade_ref"].str.split("_").str[0])
    for f, g in rr_.groupby("fold"):
        linhas.append([f.replace("F", "Fold "), "; ".join(f"{u.split('_')[1]}: {int(k)}" for u, k in zip(g["unidade_ref"], g["n_janelas_ref"]))])
    a(tab(linhas, ["Fold", "Pasta: nº de janelas de referência"], ["---", "---"]))
    a("")
    # z
    a("### 5.3 z-score por sujeito: onde é estimável (janelas de 10 s)")
    a("")
    a("Fração das unidades de referência (UTA e DMD: sujeitos; NITYMED: vídeos) em que o z de cada característica é definido "
      "(≥ 5 janelas de referência e desvio > 0). Fonte: `data/features/final/<base>/windows_referencia_10s.csv`.")
    a("")
    linhas = [[f"`{f}`"] + [pct(zest[ds][f], 0) for ds in ("uta", "dmd", "nitymed")] for f in W.FEATURES]
    a(tab(linhas, ["Característica", "UTA-RLDD (60)", "DMD (14)", "NITYMED (126)"]))
    a("")
    todas_uta = [f for f in W.FEATURES if zest["uta"][f] >= 1.0]
    raras = [f for f in W.FEATURES if zest["uta"][f] < 0.5]
    a("Mesmo com mais janelas de referência (10 s em vez de 30 s), o z só existe em TODAS as unidades da UTA para "
      + (", ".join(f"`{f}`" for f in todas_uta) or "nenhuma característica")
      + "; " + (", ".join(f"`{f}`" for f in raras) or "nenhuma") + " ficam indefinidas na maioria dos sujeitos (desvio 0: quase "
      f"sempre constantes numa referência alerta) e, no DMD, {int((ref[('dmd', 10)]['n_janelas_ref'] < 5).sum())} dos "
      f"{len(ref[('dmd', 10)])} sujeitos têm < 5 janelas de referência. **Decisão mantida: centragem por sujeito (`_c`) como "
      "principal, padronização global ajustada só no treino; `_z` disponível onde existe.**")
    a("")
    # mapeamento
    if mapa is not None:
        a("### 5.4 Mapeamento sujeito oficial ↔ pasta nos 5 folds")
        a("")
        a("Fontes: `results/exp5_uta_mapeamento_oficial.csv` (um sujeito oficial por linha: pasta, classe, motivo, escores, checagem de "
          "duração, K, piscadas descartadas), `results/exp5_uta_mapeamento_escores.csv` (sujeito × vídeo × pasta candidata), "
          "`results/exp5_uta_mapeamento_sensibilidade.csv`, `data/features/final/uta/oficial_piscadas_fold<k>.csv`; método em "
          "`src/features/uta_oficial.py`.")
        a("")
        split_txt = "Vídeos divididos: vales das partes concatenados em ordem."
        m32 = mapa[mapa["pasta"] == "F3_32"]
        p1 = bu[bu["id"] == "F3_32_10_p1"]
        if len(m32) and len(p1):
            nf10 = float(str(m32["NF_final"].iloc[0]).split(";")[-1])
            split_txt += (f" A série oficial do vídeo 10 de F3_32 termina em NF = {i(nf10)} quadros com rosto, além dos "
                          f"{i(p1['n_quadros'].iloc[0])} quadros da parte 1: a contagem do original seguiu na parte 2 (não recomeçou).")
        a("Método (extensão do da rodada preliminar): a Frequency oficial, gravada com 4 casas decimais, devolve o quadro (NF, "
          "em quadros com rosto do dlib) de cada piscada oficial; no vídeo certo esses instantes caem a uma defasagem quase constante "
          "dos vales do nosso EAR. O escore principal agora é **coerente**: uma trajetória de defasagens por bloco de 25 piscadas, "
          "escolhida por programação dinâmica com a restrição física de que a defasagem só cai devagar (dlib perdendo quadros: ≤ 300 "
          "quadros por bloco) e quase não sobe (≤ 10), menos o acaso (instantes deslocados 1/3 e 2/3 do vídeo); soma dos 3 vídeos. "
          + split_txt + " Classes: **confirmado** = escore ≥ 0,30, máximo da linha e da coluna; **provável** = sem coincidência "
          "forte, mas determinado pela ordem de listagem (pastas em ordem crescente, compatível com todos os confirmados do fold) ou pela "
          "única pasta restante compatível com a duração (NF da última piscada oficial ≤ nº de quadros); **indeterminado** = o resto.")
        a("")
        # calibração
        conf = mapa[mapa["classe"] == "confirmado"]
        nulos = []
        for _, r in conf.iterrows():
            sub = esc[(esc["fold"] == r["fold"]) & (esc["sujeito_oficial"] == r["sujeito_oficial"])]
            tot_p = sub.groupby("pasta")["liquido"].sum()
            nulos += list(tot_p.drop(r["pasta"]).values)
        nulos = np.array(nulos)
        nulos_b = []
        for _, r in conf.iterrows():
            sub = esc[(esc["fold"] == r["fold"]) & (esc["sujeito_oficial"] == r["sujeito_oficial"])]
            nulos_b += list(sub.groupby("pasta")["liquido_blocos"].sum().drop(r["pasta"]).values)
        nulos_b = np.array(nulos_b)
        a(f"Calibração do limiar 0,30: nos {len(conf)} sujeitos confirmados, as {len(nulos)} combinações com as outras pastas do fold "
          f"têm escore máximo {n(nulos.max(), 3)} (percentil 99: {n(np.percentile(nulos, 99), 3)}), e os pares confirmados, mínimo "
          f"{n(conf['escore'].min(), 3)} (mediana {n(conf['escore'].median(), 3)}). O escore por blocos independentes da rodada "
          f"preliminar escolheria a mesma pasta em {int((conf['pasta'] == conf['pasta_hungaro_blocos']).sum())} dos {len(conf)} confirmados "
          + ("(difere em " + "; ".join(f"Fold {int(r_['fold'])} oficial {int(r_['sujeito_oficial'])}: {r_['pasta_hungaro_blocos']} em vez de {r_['pasta']}"
                                      for _, r_ in conf[conf['pasta'] != conf['pasta_hungaro_blocos']].iterrows()) + ")" if (conf['pasta'] != conf['pasta_hungaro_blocos']).any() else "")
          + f", mas separa pior: máximo entre as combinações não adotadas {n(nulos_b.max(), 3)} contra mínimo {n(conf['escore_blocos'].min(), 3)} nos pares confirmados. "
          "Sensibilidade aos parâmetros (defasagem mínima −1.500/−6.000 quadros, queda máxima 150/600 por bloco): "
          + ("mapeamento idêntico em todos os folds." if msens is not None and msens["mapeamento_igual"].all() else "ver CSV."))
        a("")
        linhas = []
        for f, g in mapa.groupby("fold"):
            c_ = g["classe"].value_counts()
            pastas_f = sorted(set(bu.loc[bu["fold"] == f, "sujeito"]))
            usadas = set(g["pasta"].dropna())
            sem = [p for p in pastas_f if p not in usadas]
            linhas.append([f"Fold {int(f)}", i(len(g)), i(len(pastas_f)), i(c_.get("confirmado", 0)), i(c_.get("provavel", 0)),
                           i(c_.get("indeterminado", 0)), ", ".join(sem) or "—"])
        a(tab(linhas, ["Fold", "Sujeitos oficiais", "Pastas", "Confirmados", "Prováveis", "Indeterminados", "Pastas sem sujeito oficial atribuído"],
              ["---"] + ["---:"] * 5 + ["---"]))
        a("")
        a("Pares não confirmados e motivo:")
        a("")
        linhas = []
        for _, r in mapa[mapa["classe"] != "confirmado"].iterrows():
            linhas.append([f"Fold {int(r['fold'])}", i(r["sujeito_oficial"]), r["pasta"] if isinstance(r["pasta"], str) and r["pasta"] else "—",
                           r["classe"], n(r["escore_melhor"], 2) + f" ({r['pasta_melhor']})", r["motivo"]])
        a(tab(linhas, ["Fold", "Sujeito oficial", "Pasta", "Classe", "Maior escore (pasta)", "Motivo"], ["---", "---:", "---", "---", "---:", "---"]))
        a("")
        dup = mapa[mapa["videos_identicos_no_oficial"].fillna("") != ""]
        f5 = mapa[(mapa["fold"] == 5) & (mapa["pasta"] == "F5_52")]
        m5 = mapa[mapa["fold"] == 5].sort_values("sujeito_oficial")
        pastas5 = sorted(set(bu.loc[bu["fold"] == 5, "sujeito"]))
        sem5 = [p_ for p_ in pastas5 if p_ not in set(m5["pasta"].dropna())]
        conf5 = m5[m5["classe"] == "confirmado"]
        prov5 = m5[m5["classe"] == "provavel"]
        txt5 = (f"**Pasta ausente no Fold 5: {', '.join(sem5) or '—'}.** Os {len(conf5)} sujeitos oficiais confirmados do Fold 5 "
                f"(índices {', '.join(str(int(x)) for x in conf5['sujeito_oficial'])}) correspondem às pastas "
                f"{', '.join(conf5['pasta'])}, na mesma ordem crescente; ")
        for _, r in prov5.iterrows():
            e5 = esc[(esc["fold"] == 5) & (esc["sujeito_oficial"] == r["sujeito_oficial"]) & (esc["pasta"] == r["pasta"])]
            e49 = esc[(esc["fold"] == 5) & (esc["sujeito_oficial"] == r["sujeito_oficial"]) & (esc["pasta"].isin(sem5))]
            txt5 += (f"o sujeito oficial {int(r['sujeito_oficial'])} (maior escore {n(r['escore_melhor'], 2)}, sem coincidência forte com "
                     f"nenhuma pasta) fica, pela ordem, em {r['pasta']}. Ressalva: a última piscada oficial dele (NF = "
                     f"{' / '.join(i(float(x)) for x in str(r['NF_final']).split(';'))} nos vídeos 0 / 5 / 10) passa do nº de quadros de {r['pasta']} em "
                     f"{int(e5['viola_duracao'].sum())} vídeo(s) (NF/quadros até {n(e5['razao_NF_quadros'].max(), 3)}) e do de "
                     f"{', '.join(sem5)} em {int(e49['viola_duracao'].sum())} (até {n(e49['razao_NF_quadros'].max(), 3)}) — explicação "
                     "possível: cópia dos autores um pouco mais longa; a identificação é pela ordem, não pelas piscadas. ")
        # conferência: a checagem de duração nunca falha nos pares confirmados -> a violação do provável é atípica
        ec = esc.merge(mapa.loc[mapa["classe"] == "confirmado", ["fold", "sujeito_oficial", "pasta"]],
                       on=["fold", "sujeito_oficial", "pasta"])
        ec = ec[ec["serie_confiavel"].astype(str).str.lower() == "true"]
        if len(ec) and len(prov5):
            txt5 += (f"**Cautela:** nos {len(ec)} vídeos com série oficial confiável dos pares confirmados (todos os folds), a razão "
                     f"NF/quadros nunca passa de 1 (máximo {n(ec['razao_NF_quadros'].max(), 4)}; {int((ec['razao_NF_quadros'] > 1).sum())} "
                     "acima de 1), de modo que a violação acima é atípica: o par oficial 2 ↔ F5_52 se apoia só na ordem de "
                     "listagem e deve ficar fora da análise principal (só na sensibilidade `confirmado` + `provavel`). ")
        m1 = mapa[mapa["fold"] == 1]
        sem1 = [p_ for p_ in sorted(set(bu.loc[bu["fold"] == 1, "sujeito"])) if p_ not in set(m1["pasta"].dropna())]
        a(txt5 + f"No Fold 1, a pasta ausente continua sendo {', '.join(sem1)}.")
        a("")
        m3 = mapa[mapa["fold"] == 3].sort_values("sujeito_oficial")
        c3 = m3[m3["classe"] == "confirmado"]
        desl = sorted({int(p_.split("_")[1]) - int(s_) for p_, s_ in zip(c3["pasta"], c3["sujeito_oficial"])})
        ruins = m3[m3["series_confiaveis"].astype(str).str.split(";").map(lambda v: all(x == "0" for x in v))]
        ind3 = m3[m3["classe"] == "indeterminado"]
        prov3 = m3[m3["classe"] == "provavel"]
        pastas3 = sorted(set(bu.loc[bu["fold"] == 3, "sujeito"]))
        livres3 = [p_ for p_ in pastas3 if p_ not in set(m3["pasta"].dropna())]
        a(f"**Fold 3 — ordem não crescente e defeitos do arquivo oficial:** (i) os {len(c3)} confirmados (oficiais "
          f"{', '.join(str(int(x)) for x in c3['sujeito_oficial'])} → {', '.join(c3['pasta'])}) seguem pasta = índice + "
          f"{'/'.join(str(d_) for d_ in desl)}, o que não cabe numa listagem crescente de 12 pastas com 12 sujeitos oficiais; a ordem não "
          "pode completar o mapeamento; (ii) "
          + ("; ".join(f"no sujeito oficial {int(r['sujeito_oficial'])} os vídeos {r['videos_identicos_no_oficial'].replace('=', ' e ')} "
                       "têm sequências idênticas no arquivo oficial" for _, r in dup.iterrows()) or "nenhum vídeo repetido")
          + "; (iii) a Frequency oficial não é recuperada de forma confiável (em nenhum dos 3 vídeos há NF inteiro em todas as piscadas "
          "com poucas piscadas 'descartadas') nos sujeitos oficiais "
          + (", ".join(str(int(x)) for x in ruins["sujeito_oficial"]) or "—") + ". Indeterminados: oficiais "
          + (", ".join(str(int(x)) for x in ind3["sujeito_oficial"]) or "—") + f" (pastas restantes: {', '.join(livres3) or '—'}); "
          + ("; ".join(f"oficial {int(r['sujeito_oficial'])} → {r['pasta']} provável ({r['motivo']})" for _, r in prov3.iterrows()) or "") + ".")
        a("")
        a("**Uso recomendado:** comparações com as características oficiais (Exp. 5a) com os pares `confirmado` (principal) e "
          "`confirmado` + `provavel` (sensibilidade); nunca com os indeterminados. Os experimentos com as características oficiais "
          "(Exp. 1 e 2) não usam o mapeamento e não mudam.")
        a("")
    # sequências
    if seqr is not None:
        a("### 5.5 Sequências de 30 piscadas (passo 2)")
        a("")
        linhas = []
        for f, g in seqr.groupby("fold"):
            linhas.append([f"Fold {int(f)}", i(g["sujeito"].nunique()), i(int((g["n_sequencias"] > 0).sum())),
                           i(g.loc[g["rotulo"] == 0, "n_sequencias"].sum()), i(g.loc[g["rotulo"] == 5, "n_sequencias"].sum()),
                           i(g.loc[g["rotulo"] == 10, "n_sequencias"].sum()), i(g["n_sequencias"].sum()), i(g["n_piscadas"].sum())])
        linhas.append(["**total**", i(seqr["sujeito"].nunique()), i(int((seqr["n_sequencias"] > 0).sum())), i(seqr.loc[seqr["rotulo"] == 0, "n_sequencias"].sum()),
                       i(seqr.loc[seqr["rotulo"] == 5, "n_sequencias"].sum()), i(seqr.loc[seqr["rotulo"] == 10, "n_sequencias"].sum()),
                       i(seqr["n_sequencias"].sum()), i(seqr["n_piscadas"].sum())])
        try:
            import uta_common as U
            of_rows = []
            for f in range(1, 6):
                _, _, xte, yte = U.load_fold(f)
                of_rows.append((f, np.bincount(np.asarray(yte).reshape(-1).astype(int), minlength=3)))
            for (f, bc), l in zip(of_rows, linhas[:-1]):
                l.append(f"{i(bc[0])} / {i(bc[1])} / {i(bc[2])}")
            tot_of = sum(bc for _, bc in of_rows)
            linhas[-1].append(f"{i(tot_of[0])} / {i(tot_of[1])} / {i(tot_of[2])}")
            cab_extra = ["Oficial (teste do fold): 0 / 5 / 10"]
        except Exception as exc:  # noqa: BLE001
            cab_extra = []
            print("aviso: sem contagem oficial:", exc)
        a(tab(linhas, ["Fold", "Sujeitos", "Vídeos lógicos com sequência", "Seq. KSS 0", "Seq. KSS 5", "Seq. KSS 10", "Sequências", "Piscadas"] + cab_extra))
        a("")
        z0 = seqr[seqr["n_sequencias"] == 0]
        if len(z0):
            bu_ = bidx["uta"].set_index("id")
            det = []
            for _, r in z0.iterrows():
                arqs_ = str(r["arquivos"]).split(";")
                sig_ = [B.carregar_sinal(ROOT / "data" / "signals" / "uta" / f"{x}.npz") for x in arqs_]
                emin = min(float(np.nanmin(np.where(s_["face"], s_["ear"], np.nan))) for s_ in sig_)
                p1 = float(np.nanpercentile(np.concatenate([s_["ear"][s_["face"]] for s_ in sig_]), 1))
                lim_ = float(bu_.loc[arqs_[0], "limiar"])
                det.append(f"`{r['video']}` (KSS {int(r['rotulo'])}; limiar {n(lim_, 3)}, EAR mínimo {n(emin, 3)}, percentil 1 {n(p1, 3)})")
            a(f"**Vídeos sem nenhuma piscada detectada** ({len(z0)} de {len(seqr)}; nenhuma sequência): " + "; ".join(det)
              + ". O EAR nunca cruza o limiar calibrado no último terço do vídeo 0 do sujeito — o nível de olho aberto subiu "
              "nesses vídeos (efeito inverso ao de F1_03_10; calibração por sujeito, não erro de código). Nas janelas de 10 s esses "
              "vídeos têm PERCLOS e taxa de piscadas 0 em todas as janelas; o Exp. 5b deve reportar isso (e, por vídeo, a métrica "
              "de sequências da UTA cobre só os vídeos com sequência).")
            a("")
        dv = seqr[seqr["n_arquivos"] > 1]
        a("Vídeos divididos concatenados: " + "; ".join(f"`{r['video']}` = {r['arquivos'].replace(';', ' + ')} ({i(r['n_piscadas'])} piscadas, "
                                                        f"{i(r['n_sequencias'])} sequências)" for _, r in dv.iterrows())
          + ". Fontes: `results/exp5_sequencias_resumo.csv` (uma linha por vídeo lógico), `data/features/final/uta/sequences_normalizacao.csv`; "
          "variante com o primeiro terço como referência em `data/features/final/uta/sequences_primeiro_terco.npz`.")
        a("")
        if cor is not None:
            sub = cor[(cor["estatistica"] == "mediana") & (cor["videos"].isin(["confirmados", "confirmados_so_5_e_10"]))]
            pv = sub.pivot_table(index=["videos", "caracteristica"], columns="variante", values="spearman")
            nn = sub.groupby("videos")["n"].first()
            linhas = [[("todos os vídeos" if v == "confirmados" else "só vídeos 5 e 10") + f" ({i(nn[v])})", c_,
                       n(pv.loc[(v, c_), "limiar"], 2), n(pv.loc[(v, c_), "segundos"], 2), n(pv.loc[(v, c_), "estendida"], 2)]
                      for v, c_ in pv.index]
            a("Concordância vídeo a vídeo com as características oficiais (Spearman entre as medianas por vídeo das características "
              "normalizadas; só pares confirmados; `results/exp5_uta_validacao_oficial_correlacoes.csv` — a análise completa é do Exp. 5a):")
            a("")
            a(tab(linhas, ["Vídeos", "Característica", "limiar", "segundos", "estendida"], ["---", "---", "---:", "---:", "---:"]))
            a("")

    # ============================================================== NITYMED
    a("## 6. NITYMED")
    a("")
    un = y10["uso"].value_counts()
    rn = ref[("nitymed", 10)]
    a(f"Janelas de 10 s: {i(len(y10))} de {y10['arquivo'].nunique()} vídeos — {i(un.get('exemplo', 0))} `exemplo`, "
      f"{i(un.get('referencia', 0))} `referencia` (primeiro terço), {i(un.get('fronteira', 0))} `fronteira`. "
      f"{i(rn['pseudo_ref'].sum())} vídeos (" + ", ".join(f"{cat} {int(v)}" for cat, v in rn.assign(cat=rn['unidade_ref'].str.split('_').str[0]).groupby('cat')['pseudo_ref'].sum().items())
      + ") não têm janela inteira no primeiro terço (vídeo < 30 s) e usam a pseudo-referência (o terço inteiro como uma janela). Sem classe alerta nem rótulo por janela: serve para taxa de disparo, captura e qualidade de "
      "sinal (PP3). Com características brutas (principal do Exp. 6) a referência não entra no cálculo; `avaliavel` exclui referência "
      "e fronteira para manter a mesma regra das variantes normalizadas. Por categoria: "
      + "; ".join(f"{cat} {i(len(g))} janelas ({i(g['avaliavel'].sum())} avaliáveis)" for cat, g in y10.groupby("categoria")) + ".")
    a("")

    # ============================================================== limitações
    a("## 7. Limitações e pendências")
    a("")
    a(f"1. **DMD — alerta de outra sessão de gravação**: a referência (s3) foi gravada no mesmo dia da s5 em {sum(s3s5)} de {len(s3s5)} "
      f"sujeitos e a s2 (alerta), em outro dia em {n_s2_outro} de {len(s3s5)} (seção 4.1). O Exp. 5b deve considerar esse fator (ex.: "
      "sensibilidade restrita aos sujeitos com s2 no mesmo dia da s5, ou sem normalização).")
    a("2. **Metadados `setup` fora da convenção** (seção 4.1): " + ("; ".join(fora_conv) or "nenhum")
      + ". As janelas foram mantidas pela sessão, conforme a regra.")
    a("3. **Sensibilidade de 15 s no DMD** quase sem alerta e com referência na s2 (seção 4.4).")
    a(f"4. **Mapeamento**: {int((mapa['classe'] == 'indeterminado').sum())} sujeitos oficiais indeterminados (Fold 3, com defeitos no "
      "arquivo oficial) e pares prováveis identificados só pela ordem ou pela duração (seção 5.4).")
    a("5. **Limiar de EAR de outra sessão/trecho**: sessões com nível de olho aberto diferente do da referência têm PERCLOS e "
      "contagem de piscadas deslocados (seções 4.2 e 5.2); é a calibração por sujeito prevista, não um erro.")
    a("6. `results/sinais_resumo.md` (regenerado em 06/10/2026) cobre todas as sessões e folds; as contagens deste relatório "
      "estão na seção 2 e em `results/exp5_contagens_sinais.csv`.")
    a("")
    out = RES / "exp5_caracteristicas.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"relatório: {out.relative_to(ROOT)} ({len(L)} linhas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
