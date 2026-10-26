"""Resumo da extração de sinais por base -> results/sinais_resumo.md e results/sinais_resumo*.csv.

Lê somente data/signals/<base>/index.csv e os .npz (nenhum número digitado à mão).
Rodar no .venv:  python src/features/summarize_signals.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SIG = ROOT / "data" / "signals"
NOMES = {"dmd": "DMD (sessões s1, s2, s3 e s5; dia)",
         "nitymed": "NITYMED (noite, carro em movimento)",
         "uta": "UTA-RLDD (real, celular/webcam)"}


def br(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}".replace(".", ",")


def per_video_quality(df):
    """Medidas extras por vídeo a partir do .npz: maior sequência sem rosto e fração de quadros
    com cabeça muito girada (|yaw_proxy| > 0,3), só como descrição."""
    extra = []
    for _, r in df.iterrows():
        z = np.load(ROOT / r["npz"])
        f = z["face"]
        run = mx = 0
        for v in ~f:
            run = run + 1 if v else 0
            mx = max(mx, run)
        yaw = z["yaw_proxy"]
        fin = np.isfinite(yaw)
        extra.append(dict(id=r["id"], max_gap_s=mx / r["fps"] if r["fps"] > 0 else np.nan,
                          frac_yaw_gt_0_3=float((np.abs(yaw[fin]) > 0.3).mean()) if fin.any() else np.nan,
                          ear_p10=float(np.nanpercentile(z["ear"], 10)) if np.isfinite(z["ear"]).any() else np.nan,
                          ear_p90=float(np.nanpercentile(z["ear"], 90)) if np.isfinite(z["ear"]).any() else np.nan))
    return df.merge(pd.DataFrame(extra), on="id")


def main():
    frames, lines = [], []
    for ds in ["dmd", "nitymed", "uta"]:
        p = SIG / ds / "index.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df = df[df["complete"] == True].copy()  # noqa: E712
        df["dataset"] = ds
        df = per_video_quality(df)
        frames.append(df)
    allv = pd.concat(frames, ignore_index=True)
    cols = ["dataset", "id", "subject", "label", "fps", "n_frames", "duration_s", "width_orig", "height_orig",
            "rotation_meta", "width_proc", "height_proc", "face_rate", "max_gap_s", "ear_median", "ear_p10",
            "ear_p90", "mar_median", "frac_yaw_gt_0_3", "proc_fps", "extra_json"]
    allv[cols].to_csv(ROOT / "results" / "sinais_resumo_videos.csv", index=False)

    agg = []
    for ds, g in allv.groupby("dataset", sort=False):
        res = g.apply(lambda r: f"{int(r.width_orig)}x{int(r.height_orig)}", axis=1).value_counts()
        agg.append(dict(dataset=ds, n_videos=len(g), n_sujeitos=g["subject"].nunique(),
                        quadros=int(g["n_frames"].sum()), horas=g["duration_s"].sum() / 3600,
                        fps_min=g["fps"].min(), fps_mediana=g["fps"].median(), fps_max=g["fps"].max(),
                        resolucoes="; ".join(f"{k} ({v})" for k, v in res.items()),
                        n_reduzidos=int((g["width_proc"] < g["width_orig"]).sum()),
                        n_rotacionados=int((g["rotation_meta"] != 0).sum()),
                        face_media=g["face_rate"].mean(),
                        face_ponderada=(g["face_rate"] * g["n_frames"]).sum() / g["n_frames"].sum(),
                        face_min=g["face_rate"].min(), face_min_id=g.loc[g["face_rate"].idxmin(), "id"],
                        n_videos_face_lt_0_95=int((g["face_rate"] < 0.95).sum()),
                        maior_lacuna_s=g["max_gap_s"].max(),
                        ear_mediana=g["ear_median"].median(),
                        proc_fps_mediana=g["proc_fps"].median(),
                        proc_fps_min=g["proc_fps"].min(), proc_fps_max=g["proc_fps"].max()))
    A = pd.DataFrame(agg)
    A.to_csv(ROOT / "results" / "sinais_resumo.csv", index=False)

    # Estratos para a PP3
    strata = []
    for _, r in allv.iterrows():
        ex = json.loads(r["extra_json"])
        if r["dataset"] == "dmd":
            strata.append(("dmd", "óculos" if ex.get("glasses") else "sem óculos", r))
        elif r["dataset"] == "nitymed":
            strata.append(("nitymed", f"{ex.get('category')}/{ex.get('gender')}", r))
        else:
            strata.append(("uta", f"rótulo {r['label']}", r))
    S = pd.DataFrame([dict(dataset=d, estrato=s, id=r["id"], face_rate=r["face_rate"], n_frames=r["n_frames"],
                           ear_median=r["ear_median"]) for d, s, r in strata])
    SG = S.groupby(["dataset", "estrato"]).agg(n_videos=("id", "count"), face_media=("face_rate", "mean"),
                                                face_min=("face_rate", "min"),
                                                ear_mediana=("ear_median", "median")).reset_index()
    SG.to_csv(ROOT / "results" / "sinais_resumo_estratos.csv", index=False)

    L = lines
    L.append("# Extração de sinais por quadro (EAR/MAR/pose) — resumo\n")
    L.append("Gerado por `src/features/summarize_signals.py` a partir de `data/signals/<base>/index.csv` e dos `.npz` "
             "(extração: `src/features/extract_signals.py`, MediaPipe Face Mesh 0.10.21 legado, "
             "`refine_landmarks=False`, confiança de detecção/rastreamento 0,5, modo vídeo). "
             "CSVs: `results/sinais_resumo.csv` (por base), `results/sinais_resumo_videos.csv` (por vídeo), "
             "`results/sinais_resumo_estratos.csv` (estratos). Logs: `results/logs/extract_signals_<base>.log`.\n")
    L.append("| Base | Vídeos | Sujeitos | Quadros | Horas | fps (mín / mediana / máx) | Resoluções originais (nº vídeos) | Reduzidos p/ lado ≤1280 | Com rotação no metadado |")
    L.append("|---|---:|---:|---:|---:|---|---|---:|---:|")
    for _, a in A.iterrows():
        L.append(f"| {NOMES[a.dataset]} | {a.n_videos} | {a.n_sujeitos} | " + f"{a.quadros:,}".replace(",", ".") +
                 f" | {br(a.horas, 2)} | {br(a.fps_min, 2)} / {br(a.fps_mediana, 2)} / {br(a.fps_max, 2)} | "
                 f"{a.resolucoes} | {a.n_reduzidos} | {a.n_rotacionados} |")
    dmd_v = allv[allv.dataset == "dmd"]
    ses = dmd_v["extra_json"].map(lambda t: json.loads(t).get("sessao") or "s5").value_counts()
    uta_v = allv[allv.dataset == "uta"]
    folds = sorted(uta_v["id"].astype(str).str.extract(r"^(F\d+)_")[0].dropna().unique())
    logicos = uta_v["id"].astype(str).str.replace(r"_p\d+$", "", regex=True).nunique()
    L.append("\nNITYMED não tem identificador de motorista: cada vídeo conta como um \"sujeito\". "
             "DMD: " + ", ".join(f"{int(ses.get(k, 0))} sessões {k}" for k in ("s5", "s1", "s2", "s3")) +
             " (s5 = sonolência atuada com o carro parado; s1 = direção com o carro em movimento; s2 e s3 = carro parado). "
             f"UTA-RLDD: {len(folds)} folds ({', '.join(folds)}), {len(uta_v)} arquivos de vídeo e {logicos} vídeos lógicos "
             "(vídeos divididos em partes contam uma vez); a rotação do metadado foi aplicada "
             "(`CAP_PROP_ORIENTATION_AUTO=1`). Sujeitos da UTA com fps diferente entre os próprios vídeos "
             "(máx/mín > 1,05): "
             + "; ".join(f"`{sj}` (" + ", ".join(f"{r.label}: {br(r.fps, 2)}" for _, r in g.sort_values("label").iterrows()) + " fps)"
                         for sj, g in allv[allv.dataset == "uta"].groupby("subject") if g.fps.max() / g.fps.min() > 1.05)
             + " — Duration/Frequency em quadros dependem do fps; o z-score por sujeito só cancela o fator se o fps "
             "for igual nos vídeos do sujeito, o que NÃO vale para esses: usar duração em segundos).\n")
    L.append("\n## Taxa de detecção de rosto (fração de quadros em que o Face Mesh devolveu landmarks)\n")
    L.append("| Base | Média por vídeo | Ponderada por quadros | Mínimo por vídeo (vídeo) | Vídeos < 0,95 | Maior lacuna contínua sem rosto (s) | Mediana do EAR (mediana dos vídeos) |")
    L.append("|---|---:|---:|---|---:|---:|---:|")
    for _, a in A.iterrows():
        L.append(f"| {NOMES[a.dataset]} | {br(a.face_media, 4)} | {br(a.face_ponderada, 4)} | {br(a.face_min, 4)} (`{a.face_min_id}`) | "
                 f"{a.n_videos_face_lt_0_95} | {br(a.maior_lacuna_s, 1)} | {br(a.ear_mediana, 3)} |")
    L.append("\n### Por estrato\n")
    L.append("| Base | Estrato | Vídeos | Detecção média | Detecção mínima | Mediana do EAR |")
    L.append("|---|---|---:|---:|---:|---:|")
    for _, s in SG.iterrows():
        L.append(f"| {s.dataset} | {s.estrato} | {s.n_videos} | {br(s.face_media, 4)} | {br(s.face_min, 4)} | {br(s.ear_mediana, 3)} |")
    nit = A[A.dataset == "nitymed"].iloc[0] if (A.dataset == "nitymed").any() else None
    L.append("\n### Leitura para a PP3 (com cautela)\n")
    if nit is not None:
        dia = A[A.dataset != "nitymed"]
        mn = A.loc[A["face_media"].idxmin()]
        abertura = (f"- A NITYMED (noite) tem a menor taxa média de detecção ({br(nit.face_media, 4)})" if mn.dataset == "nitymed"
                    else f"- A NITYMED (noite) tem taxa média de detecção de {br(nit.face_media, 4)}, acima da do "
                         f"{mn.dataset.upper()} ({br(mn.face_media, 4)})")
        L.append(abertura + f"; o pior vídeo dela tem {br(nit.face_min, 4)}, e "
                 f"{nit.n_videos_face_lt_0_95} de {nit.n_videos} vídeos ficam abaixo de 0,95, "
                 f"contra {int(dia.n_videos_face_lt_0_95.sum())} de {int(dia.n_videos.sum())} no DMD e na UTA-RLDD juntos. A diferença absoluta é pequena: a taxa ponderada é "
                 + " / ".join(f"{a.dataset} {br(a.face_ponderada, 4)}" for _, a in A.iterrows()) + ".")
    L.append("- **Não dá para atribuir a diferença só à noite**: as bases diferem também em câmera, distância "
             "(na NITYMED a câmera fica atrás do volante e o rosto é pequeno), carro em movimento e protocolo "
             "(vídeos de bocejo). Na inspeção dos 4 vídeos NITYMED de menor taxa, os quadros sem rosto eram "
             "cabeça de perfil, mão na frente do rosto e boca muito aberta durante o bocejo — a perda coincide "
             "com os eventos de interesse, o que enviesa MAR/EAR justamente neles.")
    L.append("- \"Detectou rosto\" ≠ \"landmarks corretos\": em modo vídeo o Face Mesh rastreia entre quadros e "
             "pode manter landmarks imprecisos. A qualidade do EAR precisa ser avaliada pelos eventos "
             "(no DMD, pela concordância com a anotação: `results/sinais_alinhamento_dmd.csv`).")
    oc = SG[SG.dataset == "dmd"].set_index("estrato")
    if {"óculos", "sem óculos"} <= set(oc.index):
        L.append(f"- No DMD, a detecção média foi {br(oc.loc['óculos', 'face_media'], 4)} nas "
                 f"{int(oc.loc['óculos', 'n_videos'])} sessões com óculos e {br(oc.loc['sem óculos', 'face_media'], 4)} nas "
                 f"{int(oc.loc['sem óculos', 'n_videos'])} sem óculos (poucos motoristas de óculos: leitura descritiva).\n")
    L.append("\n## Velocidade de processamento (quadros/s por processo, em CPU, com vários processos em paralelo no Mac)\n")
    L.append("| Base | Mediana | Mín | Máx |")
    L.append("|---|---:|---:|---:|")
    for _, a in A.iterrows():
        L.append(f"| {NOMES[a.dataset]} | {br(a.proc_fps_mediana, 0)} | {br(a.proc_fps_min, 0)} | {br(a.proc_fps_max, 0)} |")
    L.append("\nA velocidade inclui decodificação, rotação e redução do quadro; vídeos 1080p/1440p HEVC de "
             "iPhone são os mais lentos. Não é medida de latência no celular (isso é o Exp. 4/PP2).\n")
    (ROOT / "results" / "sinais_resumo.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
