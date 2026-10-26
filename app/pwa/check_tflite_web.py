#!/usr/bin/env python3
"""Verifica se um .tflite é compatível com o tfjs-tflite 0.0.1-alpha.10 usado na PWA.

Achado (20/09/2026, teste em Chrome headless): modelos exportados com dimensão de
lote DINÂMICA (shape_signature [-1, ...]) e Flatten/Reshape geram um RESHAPE cuja
forma-alvo é calculada em tempo de execução (SHAPE -> STRIDED_SLICE -> PACK ->
RESHAPE) e fazem o runtime WASM abortar ("Aborted()"). Um LSTM com lote dinâmico
(que usa SHAPE/PACK/FILL só para o estado inicial) CARREGA normalmente.
Exportar com lote fixo 1 (tf.keras.Input(batch_shape=(1, ...))) elimina o risco;
float32, int8 completo e LSTM com unroll=True funcionam.

Critério: PROBLEMA = RESHAPE com forma-alvo não constante; AVISO = lote dinâmico
sem esse padrão (carregou nos testes, mas prefira lote fixo).

Uso:
    python app/pwa/check_tflite_web.py models/eye_state_fp32.tflite [outros.tflite ...]

Também imprime dtype/quantização de entrada e saída (bloco "int8" do eye_state.json).
"""

import json
import sys

from tensorflow.lite.python import schema_py_generated as schema  # noqa: E402

RISKY = {77: "SHAPE", 45: "STRIDED_SLICE", 83: "PACK"}


def builtin_names():
    names = {}
    for k, v in vars(schema.BuiltinOperator).items():
        if isinstance(v, int) and not k.startswith("_"):
            names[v] = k
    return names


def check(path):
    buf = open(path, "rb").read()
    m = schema.Model.GetRootAsModel(buf, 0)
    names = builtin_names()
    ops = []
    for i in range(m.OperatorCodesLength()):
        oc = m.OperatorCodes(i)
        code = oc.BuiltinCode() or oc.DeprecatedBuiltinCode()
        ops.append((code, names.get(code, str(code)), oc.Version()))
    sg = m.Subgraphs(0)
    problems = []
    warnings = []
    # Tensores produzidos por operadores (não constantes)
    produced = set()
    for k in range(sg.OperatorsLength()):
        op = sg.Operators(k)
        for j in range(op.OutputsLength()):
            produced.add(op.Outputs(j))
    for k in range(sg.OperatorsLength()):
        op = sg.Operators(k)
        code = ops[op.OpcodeIndex()][0]
        if code == 22 and op.InputsLength() >= 2 and op.Inputs(1) in produced:  # RESHAPE com forma dinâmica
            problems.append(f"RESHAPE #{k} com forma-alvo calculada em tempo de execução (Flatten/Reshape com lote dinâmico)")
    for code, name, _ in ops:
        if code in RISKY:
            warnings.append(name)

    def tensor_info(idx):
        t = sg.Tensors(idx)
        shape = [t.Shape(j) for j in range(t.ShapeLength())]
        sig = [t.ShapeSignature(j) for j in range(t.ShapeSignatureLength())] if t.ShapeSignatureLength() else None
        q = t.Quantization()
        quant = None
        if q is not None and q.ScaleLength():
            quant = {"scale": float(q.Scale(0)), "zero_point": int(q.ZeroPoint(0)) if q.ZeroPointLength() else 0}
        dtype = {0: "float32", 1: "float16", 2: "int32", 3: "uint8", 4: "int64", 7: "int16", 9: "int8"}.get(t.Type(), str(t.Type()))
        return {"name": t.Name().decode(), "shape": shape, "shape_signature": sig, "dtype": dtype, "quant": quant}

    inputs = [tensor_info(sg.Inputs(j)) for j in range(sg.InputsLength())]
    outputs = [tensor_info(sg.Outputs(j)) for j in range(sg.OutputsLength())]
    for t in inputs:
        if t["shape_signature"] and -1 in t["shape_signature"]:
            warnings.append(f"entrada com lote dinâmico {t['shape_signature']}")

    print(f"== {path}")
    print("  ops:", ", ".join(f"{n}v{v}" for _, n, v in ops))
    for t in inputs:
        print(f"  entrada: {t['dtype']} {t['shape']} sig={t['shape_signature']} quant={t['quant']}")
    for t in outputs:
        print(f"  saída:   {t['dtype']} {t['shape']} sig={t['shape_signature']} quant={t['quant']}")
    if inputs[0]["dtype"] in ("int8", "uint8"):
        block = {"input": {"dtype": inputs[0]["dtype"], **(inputs[0]["quant"] or {})},
                 "output": {"dtype": outputs[0]["dtype"], **(outputs[0]["quant"] or {})}}
        print('  bloco "int8" para eye_state.json:', json.dumps(block))
    if problems:
        print("  PROBLEMAS (o navegador aborta ao carregar): " + "; ".join(problems))
        print("  -> reexporte com tf.keras.Input(batch_shape=(1, ...)) e sem select_tf_ops.")
        return False
    if warnings:
        print(f"  AVISO: {', '.join(warnings)} — carregou nos testes, mas prefira lote fixo 1.")
    print("  OK: sem RESHAPE de forma dinâmica; deve carregar na PWA.")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    ok = all([check(p) for p in sys.argv[1:]])
    sys.exit(0 if ok else 1)
