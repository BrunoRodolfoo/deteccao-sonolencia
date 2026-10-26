#!/usr/bin/env python3
"""Importa os benchmarks gravados pelo app Android (Exp. 4: custo embarcado no Galaxy S23 Ultra).

Lê um ou mais pares bench_*.csv / bench_*.json (exportados pelo botão "Compartilhar" do app
ou copiados de Android/data/br.pucpr.sonolencia/files/benchmarks/) e gera:

  results/exp4_phone_latency.csv  — uma linha por benchmark (recalculado a partir do CSV bruto)
  results/exp4_phone_latency.md   — tabela em Markdown para o artigo/poster
  results/logs/exp4_phone_import_<data>.log — log da importação

Uso:
  python app/android/tools/import_phone_benchmark.py <arquivo.csv | pasta> [...] [--out results]

As estatísticas (p50/p95/média) são recomputadas com numpy a partir das linhas por quadro;
o JSON é usado só para os metadados (aparelho, Android, delegate, tamanho dos modelos).
Nenhum número é inventado: sem arquivos de entrada o script termina sem escrever resultados.
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGES = ["landmarker_ms", "cnn_ms", "engine_ms", "total_ms"]


def find_pairs(paths):
    csvs = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            csvs += sorted(p.glob("bench_*.csv"))
        elif p.suffix.lower() == ".csv":
            csvs.append(p)
    return csvs


def summarize(csv_path: Path):
    df = pd.read_csv(csv_path)
    js_path = csv_path.with_suffix(".json")
    meta = json.loads(js_path.read_text()) if js_path.exists() else {}
    face = df[df["face"] == 1]
    dur_s = (df["t_ms"].max() - df["t_ms"].min()) / 1000.0 if len(df) > 1 else 0.0
    row = {
        "arquivo": csv_path.name,
        "aparelho": f"{meta.get('device', {}).get('manufacturer', '')} {meta.get('device', {}).get('model', '')}".strip() or "n/d",
        "soc": meta.get("device", {}).get("soc_model", "n/d"),
        "android": meta.get("android", {}).get("release", "n/d"),
        "delegate_landmarker": meta.get("landmarker_delegate", "n/d"),
        "modelo_olho": meta.get("eye_model", "n/d"),
        "modelo_temporal": meta.get("temporal_model", "n/d"),
        "quadros": int(len(df)),
        "quadros_com_rosto": int(len(face)),
        "duracao_s": round(dur_s, 2),
        "fps_medio": round(len(df) / dur_s, 2) if dur_s > 0 else float("nan"),
    }
    for st in STAGES:
        col = df[st] if st in ("landmarker_ms", "total_ms") else face[st]
        col = col.dropna()
        if len(col) == 0:
            row[f"{st}_p50"] = row[f"{st}_p95"] = row[f"{st}_media"] = float("nan")
            continue
        row[f"{st}_p50"] = round(float(np.percentile(col, 50)), 2)
        row[f"{st}_p95"] = round(float(np.percentile(col, 95)), 2)
        row[f"{st}_media"] = round(float(col.mean()), 2)
    sizes = meta.get("model_sizes_bytes", {})
    for name, key in [("face_landmarker.task", "tam_landmarker_kb"), ("blink_temporal.tflite", "tam_temporal_kb"),
                      ("eye_state_int8.tflite", "tam_olho_int8_kb"), ("eye_state_fp32.tflite", "tam_olho_fp32_kb")]:
        row[key] = round(sizes[name] / 1024.0, 1) if name in sizes else float("nan")
    return row


def to_markdown(rows):
    cols = [("arquivo", "Arquivo"), ("aparelho", "Aparelho"), ("soc", "SoC"), ("android", "Android"),
            ("delegate_landmarker", "Delegate"), ("modelo_olho", "CNN olho"), ("fps_medio", "FPS"),
            ("landmarker_ms_p50", "Landmarker p50 (ms)"), ("landmarker_ms_p95", "p95"),
            ("cnn_ms_p50", "CNN p50 (ms)"), ("cnn_ms_p95", "p95"),
            ("total_ms_p50", "Total p50 (ms)"), ("total_ms_p95", "p95"), ("quadros", "Quadros")]
    out = ["# Exp. 4 — Latência no smartphone (app Android nativo)", "",
           f"Gerado em {dt.datetime.now():%Y-%m-%d %H:%M} por `app/android/tools/import_phone_benchmark.py` "
           "a partir dos CSV brutos gravados pelo app (uma linha por quadro; p50/p95 recalculados com numpy).", "",
           "| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k, "")) for k, _ in cols) + " |")
    out += ["", "Latência do landmarker = tempo entre `detectAsync` e o callback (inclui fila do LIVE_STREAM); "
            "CNN = dois recortes 48x48 (olho E/D) por quadro; total = do envio do quadro ao fim do motor.", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="arquivos bench_*.csv ou pastas que os contenham")
    ap.add_argument("--out", default="results", help="pasta de saída (padrão: results)")
    args = ap.parse_args()

    csvs = find_pairs(args.inputs)
    if not csvs:
        print("Nenhum bench_*.csv encontrado; nada foi escrito.", file=sys.stderr)
        return 1
    rows = [summarize(c) for c in csvs]
    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_out = out / "exp4_phone_latency.csv"
    md_out = out / "exp4_phone_latency.md"
    df.to_csv(csv_out, index=False)
    md_out.write_text(to_markdown(rows), encoding="utf-8")
    log = out / "logs" / f"exp4_phone_import_{dt.datetime.now():%Y%m%d_%H%M%S}.log"
    log.write_text("entradas:\n" + "\n".join(str(c.resolve()) for c in csvs) + "\n\n" + df.to_string() + "\n",
                   encoding="utf-8")
    print(df.to_string())
    print(f"\nEscrito: {csv_out}, {md_out}, {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
