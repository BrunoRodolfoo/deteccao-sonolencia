/**
 * tflite_io.js — Entrada/saída dos modelos TFLite no navegador (tfjs-tflite 0.0.1-alpha.10).
 *
 * Módulo sem DOM, compartilhado por app.js e pelos testes em Chrome headless.
 *
 * Fatos verificados no código-fonte do pacote (dist/tf-tflite.js, TFLiteModel):
 *   - `model.inputs[0].shape` / `model.outputs[0].shape` são arrays de números
 *     (o runner interno devolve a forma como texto "1,48,48,1" e o pacote a converte);
 *   - `model.modelRunner.getInputs()[i].dataType` é o tipo TFLite real
 *     ("float32", "uint8", "int8", ...); `model.inputs[i].dtype` já vem mapeado para
 *     o tipo tfjs ("float32" ou "int32"), por isso o dtype real é lido no runner;
 *   - entradas int8/uint8 aceitam um tf.tensor "int32" (o pacote converte com
 *     Int8Array.from / Uint8Array.from; um tensor float32 lança "Data type mismatch");
 *   - saídas uint8 viram tensor "int32" com os valores crus 0..255 (tfjs-core 4.22,
 *     inferDtype: Uint8Array -> "int32"); saídas int8 são convertidas para Int32Array.
 *     Logo, a desquantização (q - zero_point) * scale precisa ser feita aqui.
 */
import { quantize, dequantize } from "./engine.js";

/**
 * Converte o `models/eye_state.json` gerado pela exportação
 * (chaves em português: `fp32.arquivo`, `int8.arquivo`, `int8.entrada_quantizacao`, ...)
 * para o contrato que a PWA usa (`files`, `int8.input/output`, `input_size`, ...).
 * Um JSON que já esteja no contrato da PWA passa sem alteração.
 */
export function adaptEyeMeta(meta) {
  if (!meta || typeof meta !== "object") return meta || null;
  const out = { ...meta };
  // arquivos
  const files = { ...(meta.files || {}) };
  if (!files.fp32 && meta.fp32?.arquivo) files.fp32 = meta.fp32.arquivo;
  if (!files.int8 && meta.int8?.arquivo) files.int8 = meta.int8.arquivo;
  if (Object.keys(files).length) out.files = files;
  // tamanho da entrada
  if (!out.input_size) {
    const forma = meta.entrada?.forma;
    if (Array.isArray(forma) && forma.length === 4) {
      out.input_size = forma[1];
      out.channels = forma[3];
    }
  }
  // quantização do int8 (bloco "int8" com input/output)
  const q = meta.int8 || {};
  if (!q.input && q.entrada_quantizacao) {
    out.int8 = {
      ...q,
      input: { dtype: q.entrada_dtype || q.io || "uint8", ...q.entrada_quantizacao },
      output: { dtype: q.saida_dtype || q.io || "uint8", ...(q.saida_quantizacao || {}) },
    };
  }
  // saída: 1 neurônio sigmoid = P(olho aberto)
  if (!out.output && /aberto/i.test(meta.saida?.descricao || "")) out.output = "prob_open";
  if (!out.normalize && /\[0,1\]|pixel\/255/i.test(meta.entrada?.descricao || "")) out.normalize = "0-1";
  return out;
}

/**
 * Descobre dtype real de entrada/saída do modelo TFLite e monta parâmetros de (des)quantização.
 * `meta[key]` (ex.: meta.int8) pode trazer {input:{dtype,scale,zero_point}, output:{...}}.
 */
export function tfliteIO(model, meta, key) {
  let inType = "float32";
  let outType = "float32";
  try {
    const ins = model.modelRunner?.getInputs?.();
    const outs = model.modelRunner?.getOutputs?.();
    if (ins?.length) inType = ins[0].dataType || inType;
    if (outs?.length) outType = outs[0].dataType || outType;
  } catch {
    /* usa metadados */
  }
  const q = meta?.[key] || meta?.quant || {};
  if (inType === "float32" && q.input?.dtype) inType = q.input.dtype;
  if (outType === "float32" && q.output?.dtype) outType = q.output.dtype;
  const quantized = inType !== "float32";
  if (quantized && !(q.input?.scale > 0)) {
    throw new Error(`modelo com entrada ${inType} sem parâmetros de quantização no JSON (bloco "${key}")`);
  }
  return {
    inDtype: inType,
    outDtype: outType,
    inQuant: q.input || null,
    outQuant: q.output || null,
    inShape: model.inputs?.[0]?.shape || null,
    outShape: model.outputs?.[0]?.shape || null,
  };
}

/**
 * Executa um modelo TFLite com entrada Float32Array e devolve Float32Array já desquantizado.
 *  - float32: via `model.predict(tf.tensor(x, shape, "float32"))` (API pública do tfjs-tflite);
 *  - int8/uint8: escreve direto no buffer de entrada do runner e lê o de saída
 *    (`model.modelRunner.getInputs()[0].data()` / `infer()` / `getOutputs()[0].data()`), que é
 *    exatamente o que `predict` faz por dentro, mas sem alocar tensores e sem o
 *    `console.warn("converting 'int32' to 'uint8'")` que o predict emite a CADA chamada
 *    (3.594 avisos em 60 s no benchmark headless — custo desnecessário no celular).
 */
export function runTflite(tf, model, io, inputF32, shape) {
  if (io.inDtype !== "float32") {
    const q = quantize(inputF32, io.inQuant, io.inDtype);
    const runner = model.modelRunner;
    const inBuf = runner.getInputs()[0].data();
    if (inBuf.length !== q.length) throw new Error(`tamanho da entrada ${q.length} != esperado ${inBuf.length}`);
    inBuf.set(q);
    if (!runner.infer()) throw new Error("Failed running inference");
    const raw = runner.getOutputs()[0].data();
    const data = io.outDtype !== "float32" && io.outQuant ? dequantize(raw, io.outQuant) : Float32Array.from(raw);
    return data;
  }
  const input = tf.tensor(inputF32, shape, "float32");
  let out = model.predict(input);
  if (!(out instanceof tf.Tensor)) out = Object.values(out)[0];
  let data = out.dataSync();
  input.dispose();
  out.dispose();
  if (io.outDtype !== "float32" && io.outQuant) data = dequantize(data, io.outQuant);
  return Float32Array.from(data);
}
