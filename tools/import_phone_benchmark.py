#!/usr/bin/env python3
"""Importa os benchmarks de 60 s medidos no smartphone (Exp. 4, custo embarcado: Galaxy S23 Ultra)
e gera a tabela do artigo/pôster. Aceita os DOIS formatos gerados pelo projeto:

  * app Android nativo (app/android): pares  bench_<data>.csv  +  bench_<data>.json
    (botão "Compartilhar" ou pasta Android/data/br.pucpr.sonolencia/files/benchmarks/);
  * PWA (app/pwa): pares  benchmark_pwa_<data>_frames.csv  +  benchmark_pwa_<data>.json
    (downloads do Chrome ao fim do "Benchmark 60 s").

Saídas (em --out, padrão results/):
  exp4_phone_latency.csv               uma linha por benchmark (p50/p95/média recalculados do CSV bruto)
  exp4_phone_latency.md                tabela em Markdown
  logs/exp4_phone_import_<data>.log    log da importação

Uso:
  python tools/import_phone_benchmark.py results/phone            # importa tudo que houver na pasta
  python tools/import_phone_benchmark.py ~/Downloads/bench_2026*.csv ~/Downloads/benchmark_pwa_*_frames.csv

Colunas produzidas (compatíveis com src/experiments/make_figures.py, painel (c) da fig_exp4_latencia):
  origem (android|pwa), aparelho, soc, android, delegate_landmarker, modelo_olho, modelo_temporal,
  quadros, quadros_com_rosto, duracao_s, fps_medio,
  landmarker_ms_{p50,p95,media}, cnn_ms_{...}, engine_ms_{...}, total_ms_{...}, tam_*_kb.
  No Android "engine_ms" = motor completo (piscadas + LSTM); na PWA = só a inferência do LSTM 30x4
  (a PWA não mede o motor separadamente). Nada é inventado: sem arquivos de entrada, nada é escrito.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ANDROID_TOOL = ROOT / "app" / "android" / "tools" / "import_phone_benchmark.py"

STAGES = ["landmarker_ms", "cnn_ms", "engine_ms", "motor_ms", "total_ms"]
# motor_ms = motor de sinais da PWA v2.x (olhos + bocejo + cabeça + nível); não existe na v1.0 (fica NaN).
PWA_COLS = {"lat_landmarker_ms": "landmarker_ms", "lat_eye_cnn_ms": "cnn_ms",
            "lat_temporal_ms": "engine_ms", "lat_motor_ms": "motor_ms", "lat_total_ms": "total_ms"}


def _android_module():
    spec = importlib.util.spec_from_file_location("android_import", ANDROID_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find_inputs(paths):
    android, pwa = [], []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            android += sorted(p.glob("bench_*.csv"))
            pwa += sorted(p.glob("benchmark_pwa_*_frames.csv"))
        elif p.suffix.lower() == ".csv":
            (pwa if p.name.endswith("_frames.csv") else android).append(p)
    return android, pwa


# Códigos de modelo Samsung (User-Agent Client Hints do Chrome no Android) -> nome comercial.
SAMSUNG = {"SM-S918": "Samsung Galaxy S23 Ultra", "SM-S916": "Samsung Galaxy S23+", "SM-S911": "Samsung Galaxy S23"}
MAPA_APARELHOS = ROOT / "results" / "phone" / "aparelhos.json"


def _mapa_aparelhos():
    """results/phone/aparelhos.json: [{"contem": "<trecho do user-agent ou do nome do arquivo>", "aparelho": "<nome>"}].

    Necessário para iPhone: o iOS não expõe o modelo ao navegador de jeito nenhum.
    """
    try:
        return json.loads(MAPA_APARELHOS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def identify_device(ua: str, dev: dict, fname: str):
    """(aparelho, sistema, navegador) a partir do user-agent, dos Client Hints e do mapa de nomes.

    O Chrome no Android reduz o user-agent (modelo vira "K"); o modelo real vem dos Client Hints
    (dev["modelo_ua_ch"], coletado pela PWA). No iOS nenhum dos dois informa o modelo: usa-se o mapa.
    """
    ua = ua or ""
    aparelho = (dev.get("modelo_informado") or "").strip()
    code = (dev.get("modelo_ua_ch") or "").strip()
    if not aparelho and code:
        aparelho = next((name for pre, name in SAMSUNG.items() if code.upper().startswith(pre)), code)
    if not aparelho:
        aparelho = next((m["aparelho"] for m in _mapa_aparelhos()
                         if m.get("contem") and (m["contem"] in ua or m["contem"] in fname)), "")
    if not aparelho:
        aparelho = ("iPhone (modelo não informado)" if "iPhone" in ua
                    else "Android (modelo não informado)" if "Android" in ua else "n/d")
    m_ios = re.search(r"iPhone OS (\d+)_(\d+)(?:_(\d+))?", ua)
    m_and = re.search(r"Android ([\d.]+)", ua)
    if m_ios:
        sistema = "iOS " + ".".join(x for x in m_ios.groups() if x)
    elif dev.get("versao_plataforma") and "Android" in ua:
        sistema = f"Android {dev['versao_plataforma']}"
    elif m_and:
        sistema = f"Android {m_and.group(1)}"
    else:
        sistema = dev.get("platform", "n/d")
    m_crios, m_chrome = re.search(r"CriOS/(\d+)", ua), re.search(r"Chrome/(\d+)", ua)
    if m_crios:
        navegador = f"Chrome {m_crios.group(1)} (iOS, motor WebKit)"
    elif m_chrome:
        navegador = f"Chrome {m_chrome.group(1)} (Android)" if "Android" in ua else f"Chrome {m_chrome.group(1)}"
    elif "Safari" in ua:
        navegador = "Safari"
    else:
        navegador = "n/d"
    return aparelho, sistema, navegador


def summarize_pwa(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path).rename(columns=PWA_COLS)
    js_path = csv_path.with_name(csv_path.name.replace("_frames.csv", ".json"))
    meta = json.loads(js_path.read_text(encoding="utf-8")) if js_path.exists() else {}
    dev = meta.get("aparelho", {})
    ua = dev.get("userAgent", "")
    aparelho, sistema, navegador = identify_device(ua, dev, csv_path.name)
    face = df[df["face"] == 1]
    dur_s = (df["t_ms"].max() - df["t_ms"].min()) / 1000.0 if len(df) > 1 else 0.0
    row = {
        "arquivo": csv_path.name,
        "origem": "pwa",
        "versao_app": str(meta.get("versao_app", "n/d")),
        "data": str(meta.get("data", "")),
        "aparelho": aparelho,
        "soc": "n/d (PWA)",
        "sistema": sistema,
        "navegador": navegador,
        "delegate_landmarker": meta.get("delegate_landmarker", dev.get("delegate", "n/d")),
        "modelo_olho": meta.get("modelos", {}).get("eye_state", "n/d"),
        "modelo_temporal": meta.get("modelos", {}).get("blink_temporal", "n/d"),
        "quadros": int(len(df)),
        "quadros_com_rosto": int(len(face)),
        "duracao_s": round(dur_s, 2),
        "fps_medio": round(len(df) / dur_s, 2) if dur_s > 0 else float("nan"),
    }
    for st in STAGES:
        if st not in df.columns:
            row[f"{st}_p50"] = row[f"{st}_p95"] = row[f"{st}_media"] = float("nan")
            continue
        col = df[st] if st in ("landmarker_ms", "total_ms") else face[st]
        col = pd.to_numeric(col, errors="coerce").dropna()
        col = col[col > 0] if st == "engine_ms" else col   # a PWA grava 0 quando o LSTM não rodou no quadro
        if len(col) == 0:
            row[f"{st}_p50"] = row[f"{st}_p95"] = row[f"{st}_media"] = float("nan")
            continue
        row[f"{st}_p50"] = round(float(np.percentile(col, 50)), 2)
        row[f"{st}_p95"] = round(float(np.percentile(col, 95)), 2)
        row[f"{st}_media"] = round(float(col.mean()), 2)
    for key in ("tam_landmarker_kb", "tam_temporal_kb", "tam_olho_int8_kb", "tam_olho_fp32_kb"):
        row[key] = float("nan")
    return row


def to_markdown(rows) -> str:
    cols = [("origem", "Origem"), ("arquivo", "Arquivo"), ("versao_app", "Versão do app"), ("aparelho", "Aparelho"), ("sistema", "Sistema"), ("navegador", "Navegador"),
            ("delegate_landmarker", "Delegate"), ("modelo_olho", "CNN olho"), ("fps_medio", "FPS"),
            ("landmarker_ms_p50", "Landmarker p50 (ms)"), ("landmarker_ms_p95", "p95"),
            ("cnn_ms_p50", "CNN p50 (ms)"), ("cnn_ms_p95", "p95"),
            ("engine_ms_p50", "Motor/LSTM p50 (ms)"), ("engine_ms_p95", "p95"),
            ("motor_ms_p50", "Motor v2 p50 (ms)"), ("motor_ms_p95", "p95"),
            ("total_ms_p50", "Total p50 (ms)"), ("total_ms_p95", "p95"), ("quadros", "Quadros")]
    out = ["# Exp. 4 — Latência no smartphone (benchmarks de 60 s)", "",
           f"Gerado em {dt.datetime.now():%Y-%m-%d %H:%M} por `tools/import_phone_benchmark.py` a partir dos "
           "CSV brutos por quadro gravados no aparelho (p50/p95/média recalculados com numpy.percentile). "
           "Origem `android` = app nativo (TFLite nativo + MediaPipe Tasks); `pwa` = Chrome (tfjs-tflite WASM + "
           "MediaPipe WebAssembly/WebGL).", "",
           "| " + " | ".join(h for _, h in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k, "")) for k, _ in cols) + " |")
    out += ["", "Landmarker = detecção dos 478 marcos (no Android inclui a fila do modo LIVE_STREAM); "
            "CNN = dois recortes 48x48 (olho E/D) por quadro; Motor/LSTM = no Android o motor completo "
            "(piscadas + janela + LSTM), na PWA só a inferência do LSTM 30x4 (quadros em que ele rodou); "
            "Motor v2 = motor de sinais da PWA v2.x (olhos, bocejo, cabeça e nível 0-100; vazio na v1.0); "
            "Total = do recebimento do quadro ao fim do processamento.", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="arquivos CSV (Android bench_*.csv ou PWA *_frames.csv) ou pastas")
    ap.add_argument("--out", default="results", help="pasta de saída (padrão: results)")
    args = ap.parse_args()

    android_csvs, pwa_csvs = find_inputs(args.inputs)
    if not android_csvs and not pwa_csvs:
        print("Nenhum bench_*.csv (Android) nem benchmark_pwa_*_frames.csv (PWA) encontrado; nada foi escrito.",
              file=sys.stderr)
        return 1
    rows = []
    if android_csvs:
        android = _android_module()
        for c in android_csvs:
            r = android.summarize(c)
            r["origem"] = "android"
            if "android" in r:
                r["sistema"] = f"Android {r.pop('android')}"
            r.setdefault("navegador", "app nativo")
            rows.append(r)
    rows += [summarize_pwa(c) for c in pwa_csvs]

    out = Path(args.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for r in rows:
        r.setdefault("versao_app", "n/d")
        r.setdefault("data", "")
    first = ["origem", "arquivo", "versao_app", "aparelho"]
    df = df[first + [c for c in df.columns if c not in first]]
    csv_out, md_out = out / "exp4_phone_latency.csv", out / "exp4_phone_latency.md"
    df.to_csv(csv_out, index=False)
    md_out.write_text(to_markdown(rows), encoding="utf-8")
    log = out / "logs" / f"exp4_phone_import_{dt.datetime.now():%Y%m%d_%H%M%S}.log"
    log.write_text("entradas:\n" + "\n".join(str(c.resolve()) for c in android_csvs + pwa_csvs)
                   + "\n\n" + df.to_string() + "\n", encoding="utf-8")
    print(df.to_string())
    print(f"\nEscrito: {csv_out}, {md_out}, {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
