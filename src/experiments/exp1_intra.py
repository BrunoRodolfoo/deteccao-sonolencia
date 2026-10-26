"""Experimento 1 — avaliação intra-conjunto no UTA-RLDD (protocolo oficial: 5 folds por sujeito).

Modelos: LogReg, SVM-RBF, RandomForest sobre estatísticas 24d (hiperparâmetros do
baseline) e dois modelos temporais em Keras sobre a sequência bruta (30, 4): LSTM e
1D-CNN (ver ``blink_models.py``). Cada modelo é treinado separadamente para cada
esquema de rótulos: 3 classes, B1, B2 e B3 (harmonizações binárias, ver
``uta_common.SCHEMES``); no B3 a classe intermediária é descartada no treino e no teste.

Métricas por fold e média ± desvio: BSA, acurácia balanceada, F1 macro, AUC macro
OvR e VA (acurácia por vídeo, voto majoritário). Comparação com Ghoddoosian et al.
(2019), Tabelas 2 e 3.

Saídas:
  results/exp1_intra.csv                 (linhas: modelo x fold x esquema x métrica)
  results/exp1_intra_confusion.csv       (matrizes de confusão somadas nos 5 folds)
  results/exp1_intra.md                  (tabelas legíveis)
  results/logs/exp1_<timestamp>.log
  models/exp1/<modelo>_fold<k>.(pkl|keras)
  models/blink_temporal.tflite + models/blink_temporal.json (melhor modelo temporal, 3 classes)
  models/exp1_best_temporal.txt          (nome do melhor modelo temporal, usado pelo Exp. 2)

Uso: python src/experiments/exp1_intra.py [--epochs 50] [--schemes 3c,B1,B2,B3]
     python src/experiments/exp1_intra.py --epochs 15 --tag _e15 --no-save   (sensibilidade a épocas)
     python src/experiments/exp1_intra.py --regen-md [--epochs N --tag T]     (só reescreve o .md)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uta_common import (CLASS_NAMES, FEATURE_NAMES, N_FOLDS, PAPER_TABLE2,  # noqa: E402
                        PAPER_TABLE3_VA, ROOT, SCHEMES, SEED, apply_scheme,
                        chance_levels, fmt_pct, load_fold_with_ids, mean_std,
                        n_classes, sequence_metrics, sequences_to_features,
                        video_accuracy)
from blink_models import (CLASSICAL, EPOCHS, HPARAMS, TEMPORAL, build_classical,  # noqa: E402
                          build_temporal, fit_temporal, predict_proba_any)

warnings.filterwarnings("ignore", category=FutureWarning)
RESULTS = ROOT / "results"
MODELS = ROOT / "models" / "exp1"
METRICS = ["bsa", "bal_acc", "f1_macro", "auc_macro", "va"]
METRIC_LABEL = {"bsa": "BSA", "bal_acc": "Acur. bal.", "f1_macro": "F1 macro",
                "auc_macro": "AUC macro", "va": "VA"}


class Logger:
    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.f.write(msg + "\n"); self.f.flush()


def run_fold(k, scheme, epochs, log, rows, confs, save_models=True, tag=""):
    xtr, ytr, xte, yte, vid, _ = load_fold_with_ids(k)
    ytr_s, mtr = apply_scheme(ytr, scheme)
    yte_s, mte = apply_scheme(yte, scheme)
    xtr_s, xte_s, vid_s = xtr[mtr], xte[mte], vid[mte]
    K = n_classes(scheme)
    ch_seq, ch_vid = chance_levels(yte_s, vid_s)
    log(f"[{scheme}] Fold {k}: treino {xtr_s.shape} teste {xte_s.shape} "
        f"classes teste {np.bincount(yte_s)} | chance seq {ch_seq:.3f} vídeo {ch_vid:.3f}")
    rows.append(dict(model="Chance(majoritaria)", family="referencia", fold=k, scheme=scheme,
                     metric="bsa", value=ch_seq))
    rows.append(dict(model="Chance(majoritaria)", family="referencia", fold=k, scheme=scheme,
                     metric="va", value=ch_vid))
    ftr, fte = sequences_to_features(xtr_s), sequences_to_features(xte_s)
    for name in CLASSICAL + TEMPORAL:
        t0 = time.time()
        is_temporal = name in TEMPORAL
        if is_temporal:
            model = build_temporal(name, K, xtr_s, seed=SEED)
            hist = fit_temporal(model, xtr_s, ytr_s, epochs=epochs, seed=SEED)
            proba = predict_proba_any(model, xte_s, True)
            pred = proba.argmax(axis=1)
            extra = f" loss_treino_final={hist['loss'][-1]:.3f} acc_treino={hist['accuracy'][-1]:.3f}"
            if save_models:
                model.save(MODELS / f"{name}_{scheme}_fold{k}{tag}.keras")
        else:
            model = build_classical(name, seed=SEED)
            model.fit(ftr, ytr_s)
            proba = predict_proba_any(model, fte, False)
            pred = model.predict(fte)
            extra = ""
            if save_models:
                joblib.dump(model, MODELS / f"{name}_{scheme}_fold{k}{tag}.pkl", compress=3)
        m = sequence_metrics(yte_s, pred, proba)
        va, _, _ = video_accuracy(yte_s, pred, proba, vid_s)
        m["va"] = va
        confs.append((name, scheme, k, m["confusion"]))
        for met in METRICS:
            rows.append(dict(model=name, family="temporal" if is_temporal else "classico",
                             fold=k, scheme=scheme, metric=met, value=float(m[met])))
        log(f"  {name:13s} BSA={m['bsa']:.3f} bal={m['bal_acc']:.3f} F1={m['f1_macro']:.3f} "
            f"AUC={m['auc_macro']:.3f} VA={va:.3f} ({time.time() - t0:.0f}s){extra}")
        if is_temporal:
            model_cache[(name, scheme, k)] = (model, va)
    log()


model_cache = {}


def summary_table(df, scheme):
    """Tabela markdown: modelo x métricas (média ± desvio nos 5 folds)."""
    d = df[df.scheme == scheme]
    models = ["Chance(majoritaria)"] + CLASSICAL + TEMPORAL
    lines = ["| Modelo | " + " | ".join(METRIC_LABEL[m] for m in METRICS) + " |",
             "|---|" + "---|" * len(METRICS)]
    for mod in models:
        cells = []
        for met in METRICS:
            v = d[(d.model == mod) & (d.metric == met)].value.values
            cells.append("—" if len(v) == 0 or np.all(np.isnan(v)) else fmt_pct(*mean_std(v)))
        lines.append(f"| {mod} | " + " | ".join(cells) + " |")
    return lines


def per_fold_table(df, scheme, metric):
    d = df[(df.scheme == scheme) & (df.metric == metric)]
    models = CLASSICAL + TEMPORAL
    lines = [f"| Modelo | " + " | ".join(f"Fold {k}" for k in range(1, N_FOLDS + 1)) + " | Média |",
             "|---|" + "---|" * (N_FOLDS + 1)]
    for mod in models:
        vals = [d[(d.model == mod) & (d.fold == k)].value.values for k in range(1, N_FOLDS + 1)]
        vals = [v[0] if len(v) else np.nan for v in vals]
        lines.append(f"| {mod} | " + " | ".join(fmt_pct(v) for v in vals) +
                     f" | {fmt_pct(np.nanmean(vals))} |")
    if scheme == "3c" and metric == "va":
        for ref, vals in PAPER_TABLE3_VA.items():
            lines.append(f"| {ref} (Tabela 3 do artigo) | " + " | ".join(fmt_pct(v) for v in vals) +
                         f" | {fmt_pct(np.mean(vals))} |")
    return lines


def export_tflite(best_name, epochs, log):
    """Exporta o melhor modelo temporal (3 classes) para TFLite com ops builtin e valida."""
    import keras
    import tensorflow as tf

    # Fold representativo: aquele cuja VA está mais próxima da VA média do modelo.
    cands = {k: va for (n, s, k), (_, va) in model_cache.items() if n == best_name and s == "3c"}
    mean_va = float(np.mean(list(cands.values())))
    fold = min(cands, key=lambda k: abs(cands[k] - mean_va))
    model, _ = model_cache[(best_name, "3c", fold)]
    log(f"Exportando {best_name} (fold {fold}, VA={cands[fold]:.3f}, VA média={mean_va:.3f}) para TFLite")
    out = ROOT / "models" / "blink_temporal.tflite"
    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        blob = conv.convert()
        how = "from_keras_model"
    except Exception as e:  # noqa: BLE001
        log(f"  from_keras_model falhou ({type(e).__name__}: {e}); tentando SavedModel")
        sm = ROOT / "models" / "exp1" / f"{best_name}_3c_fold{fold}_savedmodel"
        model.export(str(sm))
        conv = tf.lite.TFLiteConverter.from_saved_model(str(sm))
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        blob = conv.convert()
        how = "from_saved_model"
    out.write_bytes(blob)
    size_kb = out.stat().st_size / 1024
    # validação numérica e latência
    _, _, xte, yte, vid, _ = load_fold_with_ids(fold)
    interp = tf.lite.Interpreter(model_path=str(out))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]; outd = interp.get_output_details()[0]
    # "DELEGATE" é o delegado XNNPACK aplicado pelo interpretador, não um op do modelo
    ops = (sorted({d["op_name"] for d in interp._get_ops_details()} - {"DELEGATE"})
           if hasattr(interp, "_get_ops_details") else [])
    ref = model.predict(xte, batch_size=512, verbose=0)
    tfl = np.zeros_like(ref)
    for i in range(len(xte)):
        interp.set_tensor(inp["index"], xte[i:i + 1].astype(np.float32))
        interp.invoke()
        tfl[i] = interp.get_tensor(outd["index"])[0]
    max_diff = float(np.abs(ref - tfl).max())
    agree = float((ref.argmax(1) == tfl.argmax(1)).mean())
    times = []
    x0 = xte[:1].astype(np.float32)
    for _ in range(50):
        interp.set_tensor(inp["index"], x0); interp.invoke()
    for _ in range(500):
        t0 = time.perf_counter(); interp.set_tensor(inp["index"], x0); interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    p50, p95 = float(np.percentile(times, 50)), float(np.percentile(times, 95))
    va_tfl, _, _ = video_accuracy(yte, tfl.argmax(1), tfl, vid)
    m_tfl = sequence_metrics(yte, tfl.argmax(1), tfl)
    log(f"  TFLite: {size_kb:.1f} KB via {how}; ops={ops}; max|dif|={max_diff:.2e}; "
        f"concordância argmax={agree:.4f}; latência p50={p50:.3f} ms p95={p95:.3f} ms (Mac, CPU)")
    log(f"  TFLite no fold {fold}: BSA={m_tfl['bsa']:.3f} VA={va_tfl:.3f}")
    norm = model.layers[1]
    meta = {
        "modelo": best_name, "arquitetura": HPARAMS[best_name], "fold_origem": fold,
        "conversao": how, "ops_tflite": ops,
        "forma_entrada": [1, 30, 4], "dtype": "float32",
        "ordem_caracteristicas": FEATURE_NAMES,
        "definicoes": {
            "frequencia": "100 * (n. de piscadas até a piscada i) / (n. de quadros até end_i)",
            "amplitude": "(EAR[start] - 2*EAR[bottom] + EAR[end]) / 2",
            "duracao": "end - start + 1 (quadros)",
            "velocidade_abertura": "(EAR[end] - EAR[bottom]) / (end - bottom)",
            "EAR": "(||p2-p6|| + ||p3-p5||) / ||p1-p4||"},
        "normalizacao_por_sujeito": ("z-score de cada característica com média e desvio das "
                                     "piscadas de um terço do vídeo em estado alerta do próprio "
                                     "sujeito (calibração); as demais piscadas são normalizadas "
                                     "com esses parâmetros (Ghoddoosian et al., 2019)"),
        "normalizacao_embutida_no_modelo": {
            "tipo": "camada Normalization (z-score por característica, ajustada no treino do fold)",
            "mean": [float(v) for v in np.asarray(norm.mean).reshape(-1)],
            "variance": [float(v) for v in np.asarray(norm.variance).reshape(-1)]},
        "janela": {"tamanho": 30, "passo": 2, "unidade": "piscadas"},
        "classes": {i: n for i, n in enumerate(CLASS_NAMES)},
        "rotulos_originais": {"0": 0, "5": 1, "10": 2},
        "decisao_por_video": "voto majoritário das janelas; empate -> maior probabilidade média",
        "treino": {"epocas": epochs, "seed": SEED, "batch": 64, "otimizador": "Adam", "lr": 1e-3,
                   "parametros": int(model.count_params())},
        "metricas_fold_origem_keras": {"va": cands[fold]},
        "metricas_fold_origem_tflite": {"bsa": m_tfl["bsa"], "bal_acc": m_tfl["bal_acc"],
                                        "f1_macro": m_tfl["f1_macro"], "va": va_tfl},
        "validacao_tflite_vs_keras": {"max_abs_diff": max_diff, "concordancia_argmax": agree,
                                      "n_sequencias": int(len(xte))},
        "latencia_mac_cpu_ms": {"p50": p50, "p95": p95, "n_execucoes": 500},
        "tamanho_kb": round(size_kb, 1),
    }
    (ROOT / "models" / "blink_temporal.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return meta


def write_md(df, epochs, ts, tfl_meta, log_path, tag=""):
    L = ["# Experimento 1 — UTA-RLDD intra-conjunto (5 folds oficiais, divisão por sujeito)", "",
         f"Execução {ts}. Seed {SEED}. Modelos temporais treinados por {epochs} épocas fixas "
         "(sem early stopping; o teste nunca é consultado). Valores em % (média ± desvio populacional, ddof=0, nos 5 folds). "
         "Dados brutos: `results/exp1_intra" + tag + ".csv`; log: `results/logs/" + log_path.name + "`.", "",
         "Modelos clássicos: estatísticas 24d por sequência (média, desvio, mín., máx., mediana, IQR de "
         "cada característica) com os hiperparâmetros do baseline. Modelos temporais (Keras) sobre a "
         "sequência bruta (30 piscadas x 4): " +
         "; ".join(f"**{n}**: {json.dumps({**HPARAMS[n], 'epochs': epochs}, ensure_ascii=False)}"
                   for n in TEMPORAL) + ".", "",
         "VA = acurácia por vídeo, voto majoritário das janelas do vídeo; empate resolvido pela maior "
         "probabilidade média entre as classes empatadas. Chance = classe majoritária do teste.", ""]
    for scheme, desc in SCHEMES.items():
        if scheme not in df.scheme.unique():
            continue
        L += [f"## Esquema {scheme}: {desc}", ""] + summary_table(df, scheme) + [""]
        if scheme == "3c":
            L += ["Referências (Ghoddoosian et al., 2019, Tabela 2, média dos 5 folds): " +
                  "; ".join(f"{k}: BSA {fmt_pct(v[0]) if v[0] is not None else '—'} / VA {fmt_pct(v[1])}"
                            for k, v in PAPER_TABLE2.items()) + ".", "",
                  "### VA por fold (3 classes) — comparação com a Tabela 3 do artigo", ""]
            L += per_fold_table(df, "3c", "va") + ["", "### BSA por fold (3 classes)", ""]
            L += per_fold_table(df, "3c", "bsa") + [""]
    if tfl_meta:
        L += ["## Exportação TFLite (melhor modelo temporal por VA média, 3 classes)", "",
              f"- Modelo: **{tfl_meta['modelo']}** (fold de origem {tfl_meta['fold_origem']}, "
              f"conversão `{tfl_meta['conversao']}`, ops: {', '.join(tfl_meta['ops_tflite'])})",
              f"- Tamanho: {tfl_meta['tamanho_kb']} KB; parâmetros: {tfl_meta['treino']['parametros']}",
              f"- Validação vs. Keras ({tfl_meta['validacao_tflite_vs_keras']['n_sequencias']} sequências): "
              f"max |dif| = {tfl_meta['validacao_tflite_vs_keras']['max_abs_diff']:.2e}, "
              f"concordância do argmax = {100 * tfl_meta['validacao_tflite_vs_keras']['concordancia_argmax']:.2f}%",
              f"- Latência no Mac (CPU, 500 execuções, lote 1): p50 = {tfl_meta['latencia_mac_cpu_ms']['p50']:.3f} ms, "
              f"p95 = {tfl_meta['latencia_mac_cpu_ms']['p95']:.3f} ms",
              f"- Metadados completos: `models/blink_temporal.json`", ""]
    if not tag:
        L += sensitivity_section(df, epochs)
    (RESULTS / f"exp1_intra{tag}.md").write_text("\n".join(L) + "\n")


def sensitivity_section(df_main, epochs_main, tag_alt="_e15", epochs_alt=15):
    """Tabela comparando os modelos temporais (3 classes) entre a execução principal e a
    execução alternativa ``exp1_intra<tag_alt>.csv`` (gerada com ``--epochs 15 --tag _e15
    --no-save``). Retorna [] se o CSV alternativo não existir."""
    alt_csv = RESULTS / f"exp1_intra{tag_alt}.csv"
    if not alt_csv.exists():
        return []
    df_alt = pd.read_csv(alt_csv)
    logs = sorted(p for p in (RESULTS / "logs").glob(f"exp1{tag_alt}_*.log")
                  if re.fullmatch(rf"exp1{re.escape(tag_alt)}_\d{{8}}-\d{{6}}\.log", p.name))
    log_name = logs[-1].name if logs else "(log não encontrado)"
    L = ["## Sensibilidade ao número de épocas (modelos temporais, 3 classes)", "",
         f"Execução adicional com {epochs_alt} épocas (`results/exp1_intra{tag_alt}.csv`, log "
         f"`results/logs/{log_name}`), mesmos hiperparâmetros e seed; sem salvar modelos nem alterar "
         f"a exportação TFLite (que usa o LSTM de {epochs_main} épocas).", "",
         "| Modelo | Épocas | " + " | ".join(METRIC_LABEL[m] for m in METRICS) + " |",
         "|---|---|" + "---|" * len(METRICS)]
    for name in TEMPORAL:
        for ep, d in [(epochs_main, df_main), (epochs_alt, df_alt)]:
            cells = [fmt_pct(*mean_std(d[(d.model == name) & (d.scheme == "3c") &
                                          (d.metric == m)].value.values)) for m in METRICS]
            L.append(f"| {name} | {ep} | " + " | ".join(cells) + " |")
    L += ["", f"Com {epochs_main} épocas a acurácia de treino chega a ~0,90–0,93 (ver log) enquanto a BSA em "
          "sujeitos não vistos fica em ~0,50: os modelos temporais pequenos memorizam padrões dos "
          "sujeitos de treino sem ganho claro de generalização, e nenhum deles supera o RandomForest "
          "sobre estatísticas 24d.", ""]
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--schemes", default="3c,B1,B2,B3")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--tag", default="", help="sufixo dos arquivos de saída (ex.: _e15); com tag, não exporta TFLite")
    ap.add_argument("--regen-md", action="store_true",
                    help="só reescreve results/exp1_intra<tag>.md a partir do CSV e do JSON existentes (sem treinar)")
    args = ap.parse_args()
    tag = args.tag
    if args.regen_md:
        # só logs desta tag: exp1<tag>_AAAAMMDD-HHMMSS.log (o glob "exp1_*" pegaria também "exp1_e15_*")
        logs = sorted(p for p in (RESULTS / "logs").glob(f"exp1{tag}_*.log")
                      if re.fullmatch(rf"exp1{re.escape(tag)}_\d{{8}}-\d{{6}}\.log", p.name))
        if not logs:
            raise SystemExit("nenhum log exp1 encontrado para regenerar o .md")
        log_path = logs[-1]
        ts = log_path.stem.split("_")[-1]
        df = pd.read_csv(RESULTS / f"exp1_intra{tag}.csv")
        tfl_json = ROOT / "models" / "blink_temporal.json"
        tfl_meta = json.loads(tfl_json.read_text()) if (not tag and tfl_json.exists()) else None
        write_md(df, args.epochs, ts, tfl_meta, log_path, tag)
        print(f"{RESULTS / f'exp1_intra{tag}.md'} regenerado a partir de {RESULTS / f'exp1_intra{tag}.csv'} (log {log_path.name})")
        return
    ts = time.strftime("%Y%m%d-%H%M%S")
    (RESULTS / "logs").mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS / "logs" / f"exp1{tag}_{ts}.log"
    log = Logger(log_path)
    log(f"Exp. 1 intra-conjunto UTA-RLDD — seed {SEED}, épocas {args.epochs}, esquemas {args.schemes}")
    rows, confs, tfl_meta = [], [], None
    for scheme in args.schemes.split(","):
        for k in range(1, N_FOLDS + 1):
            run_fold(k, scheme, args.epochs, log, rows, confs, save_models=not args.no_save, tag=tag)
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS / f"exp1_intra{tag}.csv", index=False)
        if scheme == "3c" and not tag:
            va = {n: df[(df.model == n) & (df.scheme == "3c") & (df.metric == "va")].value.mean()
                  for n in TEMPORAL}
            best = max(va, key=va.get)
            log(f"VA média 3 classes dos temporais: {va} -> melhor: {best}")
            (ROOT / "models" / "exp1_best_temporal.txt").write_text(best + "\n")
            tfl_meta = export_tflite(best, args.epochs, log)
            log()
        write_md(df, args.epochs, ts, tfl_meta, log_path, tag)
    # matrizes de confusão somadas nos folds
    crows = []
    for (name, scheme, k, c) in confs:
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                crows.append(dict(model=name, scheme=scheme, fold=k, true=i, pred=j, count=int(c[i, j])))
    pd.DataFrame(crows).to_csv(RESULTS / f"exp1_intra_confusion{tag}.csv", index=False)
    df = pd.DataFrame(rows)
    log("\n" + "\n".join(summary_table(df, "3c")))
    log(f"\nResultados em {RESULTS / f'exp1_intra{tag}.csv'} e {RESULTS / f'exp1_intra{tag}.md'}; log {log_path}")


if __name__ == "__main__":
    main()
