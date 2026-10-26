"""Exp. 4 (parte desktop) — exportação do EyeNet final (treinado no MRL) para
TFLite e medição de custo.

Gera:
  models/eye_state_fp32.tflite   pesos/ativações float32
  models/eye_state_int8.tflite   quantização inteira completa (pesos e
                                 ativações int8) com conjunto representativo de
                                 --n-rep imagens do MRL; entrada/saída uint8
                                 quando o conversor aceita, senão int8, senão
                                 E/S float (o que foi usado fica em eye_state.json)
  models/eye_state.json          contrato de entrada/saída para o app/PWA
  results/exp4_tflite_eval.csv   acurácia Keras vs fp32 vs int8 (MRL val, MRL
                                 completo, CEW patch, CEW crop)
  results/exp4_desktop_latency.csv  latência batch 1 neste Mac (p50/p95, 1 e 4
                                 threads, 50 execuções de aquecimento + 500)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from pipeline_b import datasets as D  # noqa: E402
from pipeline_b.train_eye_state import metrics  # noqa: E402

MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SEED = 42


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def convert(saved_dir: str, x_rep: np.ndarray | None, io_type=None):
    import tensorflow as tf
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    if x_rep is not None:
        def rep():
            for i in range(len(x_rep)):
                yield [x_rep[i:i + 1].astype(np.float32)]
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = rep
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if io_type is not None:
            conv.inference_input_type = io_type
            conv.inference_output_type = io_type
    return conv.convert()


class TFLiteModel:
    def __init__(self, path: Path, threads: int = 1):
        import tensorflow as tf
        self.interp = tf.lite.Interpreter(model_path=str(path), num_threads=threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def _prep(self, x_float: np.ndarray) -> np.ndarray:
        dt = self.inp["dtype"]
        if dt == np.float32:
            return x_float.astype(np.float32)
        scale, zp = self.inp["quantization"]
        q = np.round(x_float / scale + zp)
        info = np.iinfo(dt)
        return np.clip(q, info.min, info.max).astype(dt)

    def _post(self, y: np.ndarray) -> np.ndarray:
        if self.out["dtype"] == np.float32:
            return y.astype(np.float32)
        scale, zp = self.out["quantization"]
        return (y.astype(np.float32) - zp) * scale

    def predict(self, x_float: np.ndarray) -> np.ndarray:
        out = np.empty(len(x_float), dtype=np.float32)
        xq = self._prep(x_float)
        for i in range(len(x_float)):
            self.interp.set_tensor(self.inp["index"], xq[i:i + 1])
            self.interp.invoke()
            out[i] = self._post(self.interp.get_tensor(self.out["index"])).reshape(-1)[0]
        return out

    def latency(self, x_float: np.ndarray, warmup=50, runs=500):
        xq = self._prep(x_float[:1])
        for _ in range(warmup):
            self.interp.set_tensor(self.inp["index"], xq)
            self.interp.invoke()
        ts = []
        for _ in range(runs):
            t0 = time.perf_counter()
            self.interp.set_tensor(self.inp["index"], xq)
            self.interp.invoke()
            ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts)
        return {"p50_ms": float(np.percentile(ts, 50)), "p95_ms": float(np.percentile(ts, 95)),
                "mean_ms": float(ts.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "exp3" / "eyenet_mrl_final.keras"))
    ap.add_argument("--n-rep", type=int, default=500)
    ap.add_argument("--n-eval-mrl", type=int, default=0, help="0 = MRL completo")
    args = ap.parse_args()
    import keras
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(8)
    rng = np.random.default_rng(SEED)

    model = keras.models.load_model(args.model)
    log(f"modelo {args.model}: {model.count_params()} parâmetros")
    x_mrl, mrl = D.load_mrl()
    y_mrl = mrl["state"].values.astype(int)
    meta = json.loads((RESULTS / "exp3_meta.json").read_text(encoding="utf-8"))
    val_subj = meta["final_model"]["val_subjects"]
    val_mask = mrl["subject"].isin(val_subj).values
    rep_idx = rng.choice(len(x_mrl), args.n_rep, replace=False)
    x_rep = D.to_float(x_mrl[rep_idx])

    with tempfile.TemporaryDirectory() as td:
        sd = os.path.join(td, "sm")
        model.export(sd)
        fp32 = convert(sd, None)
        (MODELS / "eye_state_fp32.tflite").write_bytes(fp32)
        io_used = None
        int8 = None
        for io_name, io_type in [("uint8", tf.uint8), ("int8", tf.int8), ("float32", None)]:
            try:
                int8 = convert(sd, x_rep, io_type)
                io_used = io_name
                log(f"int8 convertido com E/S {io_name}")
                break
            except Exception as e:  # noqa: BLE001
                log(f"falha na conversão int8 com E/S {io_name}: {str(e)[:200]}")
        (MODELS / "eye_state_int8.tflite").write_bytes(int8)
    sizes = {"fp32": len(fp32) / 1024, "int8": len(int8) / 1024,
             "keras": os.path.getsize(args.model) / 1024}
    log(f"tamanhos KB: {sizes}")

    # ---------- avaliação ----------
    eval_sets = [("MRL_val_subjects", x_mrl[val_mask], y_mrl[val_mask])]
    if args.n_eval_mrl:
        sub = rng.choice(len(x_mrl), args.n_eval_mrl, replace=False)
        eval_sets.append((f"MRL_amostra{args.n_eval_mrl}", x_mrl[sub], y_mrl[sub]))
    else:
        eval_sets.append(("MRL_completo", x_mrl, y_mrl))
    for kind in ("patch", "crop"):
        xc, dfc = D.load_cew(kind)
        eval_sets.append((f"CEW_{kind}", xc, dfc["state"].values.astype(int)))
    tfl = {"fp32": TFLiteModel(MODELS / "eye_state_fp32.tflite", 4),
           "int8": TFLiteModel(MODELS / "eye_state_int8.tflite", 4)}
    rows = []
    for name, xx, yy in eval_sets:
        xf = D.to_float(xx)
        p_keras = model.predict(xf, batch_size=1024, verbose=0).reshape(-1)
        res = {"keras": metrics(yy, p_keras)}
        for k, m in tfl.items():
            t0 = time.time()
            res[k] = metrics(yy, m.predict(xf))
            log(f"{name} {k}: {len(xx)} imagens em {time.time() - t0:.0f}s")
        for k, mm in res.items():
            for mk, mv in mm.items():
                rows.append({"dataset": name, "modelo": k, "metric": mk, "value": float(mv)})
        log(f"{name}: keras acc={res['keras']['acc']:.4f} fp32 acc={res['fp32']['acc']:.4f} "
            f"int8 acc={res['int8']['acc']:.4f} (n={len(xx)})")
    pd.DataFrame(rows).to_csv(RESULTS / "exp4_tflite_eval.csv", index=False)

    # ---------- latência ----------
    lat_rows = []
    x1 = D.to_float(x_mrl[:1])
    for k in ("fp32", "int8"):
        for th in (1, 4):
            m = TFLiteModel(MODELS / f"eye_state_{k}.tflite", th)
            r = m.latency(x1)
            lat_rows.append({"modelo": k, "threads": th, "arquivo": f"eye_state_{k}.tflite",
                             "tamanho_kb": round(sizes[k], 1), **{kk: round(v, 3) for kk, v in r.items()},
                             "warmup": 50, "runs": 500, "batch": 1,
                             "maquina": f"{platform.machine()} / {platform.platform()}",
                             "tf_version": tf.__version__})
            log(f"latência {k} {th} thread(s): p50={r['p50_ms']:.3f} ms p95={r['p95_ms']:.3f} ms")
    pd.DataFrame(lat_rows).to_csv(RESULTS / "exp4_desktop_latency.csv", index=False)

    # ---------- contrato ----------
    tfl_int8 = tfl["int8"]
    ev = pd.DataFrame(rows)

    def get(ds, mo, me):
        return float(ev[(ev.dataset == ds) & (ev.modelo == mo) & (ev.metric == me)].value.iloc[0])

    contract = {
        "modelo": "EyeNet (src/pipeline_b/model.py), treinado no MRL Eye (37 sujeitos), seed 42",
        "parametros": int(model.count_params()),
        "entrada": {"forma": [1, 48, 48, 1], "descricao": "recorte quadrado do olho, escala de cinza, "
                    "redimensionado para 48x48 (INTER_AREA ao reduzir, INTER_CUBIC ao ampliar), valores em [0,1] = pixel/255"},
        "saida": {"forma": [1, 1], "descricao": "P(olho ABERTO); classe 1 = aberto, 0 = fechado; limiar 0,5"},
        "fp32": {"arquivo": "eye_state_fp32.tflite", "tamanho_kb": round(sizes["fp32"], 1),
                 "entrada_dtype": "float32", "saida_dtype": "float32"},
        "int8": {"arquivo": "eye_state_int8.tflite", "tamanho_kb": round(sizes["int8"], 1),
                 "io": io_used,
                 "entrada_dtype": str(np.dtype(tfl_int8.inp["dtype"])),
                 "entrada_quantizacao": {"scale": float(tfl_int8.inp["quantization"][0]),
                                          "zero_point": int(tfl_int8.inp["quantization"][1]),
                                          "formula": "q = round(x/scale) + zero_point, x em [0,1]"},
                 "saida_dtype": str(np.dtype(tfl_int8.out["dtype"])),
                 "saida_quantizacao": {"scale": float(tfl_int8.out["quantization"][0]),
                                        "zero_point": int(tfl_int8.out["quantization"][1]),
                                        "formula": "p = (q - zero_point) * scale"},
                 "representative_dataset": f"{args.n_rep} imagens MRL, seed {SEED}"},
        "metricas": {ds: {mo: {"acc": get(ds, mo, "acc"), "auc": get(ds, mo, "auc"), "n": int(get(ds, mo, "n"))}
                          for mo in ("keras", "fp32", "int8")} for ds in ev.dataset.unique()},
        "latencia_desktop": lat_rows,
    }
    (MODELS / "eye_state.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    log("contrato salvo em models/eye_state.json")


if __name__ == "__main__":
    main()
