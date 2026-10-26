# app/pwa/models — modelos TFLite usados pela PWA

A PWA procura os modelos TensorFlow Lite **nesta pasta** (`./models/` relativo ao
`index.html`). Eles são gerados pelos scripts de treino e exportação em `models/` na
raiz do repositório e **já foram copiados para cá** (20/09/2026): `blink_temporal.tflite`
+ `.json`, `eye_state_fp32.tflite`, `eye_state_int8.tflite` + `eye_state.json`. Se algum
arquivo não existir (HTTP 404) o app continua em **modo degradado** (somente landmarks:
EAR, MAR, PERCLOS, piscadas e blendshapes) e mostra um aviso amarelo — nada quebra.
Se os modelos da raiz forem regenerados, repita o `cp` abaixo.

## Arquivos esperados

| Arquivo | Origem (raiz do repo) | Uso |
|---|---|---|
| `blink_temporal.tflite` | `models/blink_temporal.tflite` | modelo temporal, entrada `[1, 30, 4]` float32 (z-score de [Frequency, Amplitude, Duration, Velocity]), saída 3 classes (alerta, pouca vigilância, sonolento) |
| `blink_temporal.json` (opcional) | `models/blink_temporal.json` | metadados do modelo temporal |
| `eye_state_fp32.tflite` | `models/eye_state_fp32.tflite` | CNN de estado do olho, entrada `[1, 48, 48, 1]` float32 |
| `eye_state_int8.tflite` (opcional) | `models/eye_state_int8.tflite` | versão quantizada (entrada/saída `uint8`); usada primeiro **se** `eye_state.json` trouxer os parâmetros de quantização (o JSON real traz) |
| `eye_state.json` | `models/eye_state.json` | metadados da CNN (normalização, ordem das classes, arquivos) |

Comando para copiar (a partir da raiz do repositório):

```bash
cp models/blink_temporal.tflite models/blink_temporal.json \
   models/eye_state_fp32.tflite models/eye_state_int8.tflite models/eye_state.json app/pwa/models/
```

## Formato de `eye_state.json` (todos os campos são opcionais; valores padrão entre parênteses)

```json
{
  "input_size": 48,                 // lado do recorte quadrado em pixels (48)
  "channels": 1,                    // 1 = cinza (padrão); 3 = o cinza é replicado nos 3 canais
  "normalize": "0-1",               // "0-1" (x/255), "-1-1" (x/127.5-1), "mean_std" ((x/255-mean)/std), "none" (0..255)
  "mean": 0.5, "std": 0.5,          // usados só com "mean_std"
  "class_order": ["closed", "open"],// ordem das saídas quando há 2 neurônios (padrão: índice 1 = aberto)
  "output": "prob_open",            // "prob_open" (1 neurônio sigmoid), "logit_open", "softmax", "logits", "prob_closed"
  "files": { "fp32": "eye_state_fp32.tflite", "int8": "eye_state_int8.tflite" },
  "int8": {                         // OBRIGATÓRIO para usar o int8 no navegador (o tfjs-tflite não expõe os
    "input":  { "dtype": "uint8", "scale": 0.003921, "zero_point": 0 },     //   parâmetros de quantização);
    "output": { "dtype": "uint8", "scale": 0.003906, "zero_point": 0 }      //   dtype "int8" ou "uint8"
  }
}
```

Como obter `scale`/`zero_point` em Python:

```python
import tensorflow as tf
it = tf.lite.Interpreter(model_path="models/eye_state_int8.tflite")
i, o = it.get_input_details()[0], it.get_output_details()[0]
print(i["dtype"], i["quantization"], o["dtype"], o["quantization"])  # (scale, zero_point)
```

Sem o bloco `int8`, o app usa automaticamente o `fp32`.

## Compatibilidade com os JSON gerados pela exportação

`tflite_io.js` (`adaptEyeMeta`) aceita o `models/eye_state.json` real, com chaves em português
(`entrada.forma`, `fp32.arquivo`, `int8.arquivo`, `int8.entrada_dtype`, `int8.entrada_quantizacao`,
`int8.saida_quantizacao`, `saida.descricao`), convertendo-o para o contrato acima. Com esse JSON o
app carrega **primeiro o `eye_state_int8.tflite`** (entrada/saída `uint8`, escala 1/255 e 1/256,
zero_point 0) e cai para o `fp32` se falhar. Verificado em Chrome headless (20/09/2026,
`results/logs/pwa_tflite_numerics_*.log`): as saídas de `fp32` e `int8` no navegador coincidem com
as do `tf.lite.Interpreter` em Python para as mesmas imagens do MRL (diferença < 1e-6).

O `models/blink_temporal.json` (chaves `forma_entrada`, `classes`, `ordem_caracteristicas`,
`normalizacao_embutida_no_modelo`, ...) também é aceito: o app só lê `window`, `n_features`, `output`
e `files` quando existem e usa os padrões (30, 4, softmax, `blink_temporal.tflite`) caso contrário.
Como esse modelo tem uma camada `Normalization` embutida ajustada sobre as *blink features* já
normalizadas por sujeito (dados do UTA-RLDD), o app continua alimentando o **z-score por sujeito**
(calibração de 20 piscadas).

**Atenção (verificado):** no tfjs-tflite 0.0.1-alpha.10 a **primeira** inferência do
`blink_temporal.tflite` (LSTM com lote dinâmico, ops SHAPE/PACK/FILL para o estado inicial)
devolve `[0, 0, 0]`; da segunda em diante o resultado coincide com o Python (< 1e-6). Por isso o app
faz duas inferências de aquecimento e valida a segunda (`warmupTemporal` em `app.js`). Os modelos
`eye_state_*` não apresentam esse comportamento.

## Formato de `blink_temporal.json` (opcional)

```json
{
  "window": 30, "n_features": 4,
  "feature_order": ["Frequency", "Amplitude", "Duration", "Velocity"],
  "classes": ["alerta", "pouca vigilância", "sonolento"],
  "output": "softmax",              // ou "logits" (o app aplica softmax se a saída não somar 1)
  "files": { "fp32": "blink_temporal.tflite" }
}
```

## REQUISITO DE EXPORTAÇÃO (verificado em Chrome headless, 20/09/2026)

O runtime `tfjs-tflite@0.0.1-alpha.10` **aborta** (`RuntimeError: Aborted()`) ao carregar modelos
exportados com dimensão de lote **dinâmica** (`shape_signature [-1, ...]`) que contenham
`Flatten`/`Reshape`: o conversor gera um `RESHAPE` cuja forma-alvo é calculada em tempo de execução
(`SHAPE -> STRIDED_SLICE -> PACK -> RESHAPE`), e é esse padrão que derruba o runtime. Um LSTM com
lote dinâmico (usa `SHAPE/PACK/FILL` só para o estado inicial) carregou normalmente — o
`models/blink_temporal.tflite` atual (LSTM 32, unroll) foi testado no Chrome e **funciona**
(aquecimento 4,4 ms). Resultado dos testes (`python app/pwa/check_tflite_web.py`):

| Variante | Resultado no navegador |
|---|---|
| `Input((48,48,1))` + Flatten (lote dinâmico) | **ABORTA** |
| `Input((48,48,1))` + Reshape explícito (lote dinâmico) | **ABORTA** |
| `Input((48,48,1))` + GlobalAveragePooling2D (lote dinâmico) | OK |
| `Input(batch_shape=(1,48,48,1))` + Flatten, float32 | OK (0,7 ms) |
| `Input(batch_shape=(1,48,48,1))` + Flatten, **int8 completo** (entrada/saída int8) | OK (1,1 ms) |
| CNN realista (BatchNorm, Dropout, 2 Conv, softmax 2 classes), lote fixo 1 | OK |
| LSTM(16, `unroll=True`) sobre `Input(batch_shape=(1,30,4))` | OK (0,5 ms) |
| LSTM sem `unroll` | **falha já na conversão** (TensorList; exigiria SELECT_TF_OPS, que o navegador não tem) |
| `models/blink_temporal.tflite` real (LSTM 32 unroll, lote dinâmico, Normalization embutida) | OK (4,4 ms) |

Portanto, ao exportar a **CNN de olhos** (que tem Flatten/Dense) para a PWA use obrigatoriamente lote fixo;
para o temporal é recomendado:

```python
inp = tf.keras.Input(batch_shape=(1, 48, 48, 1))      # lote FIXO = 1
# ...
inp_t = tf.keras.Input(batch_shape=(1, 30, 4)); x = tf.keras.layers.LSTM(64, unroll=True)(inp_t)
converter = tf.lite.TFLiteConverter.from_keras_model(model)   # só TFLITE_BUILTINS
```

e confira com `python app/pwa/check_tflite_web.py models/*.tflite` (o script também imprime o bloco
`"int8"` pronto para o `eye_state.json`; os três modelos atuais passam com AVISO de lote dinâmico e carregam). O mesmo arquivo com lote fixo funciona igualmente no
Android nativo, então não há custo em adotar essa regra para todos os exports.

## Observações

- Modelos com ops não suportadas pelo XNNPACK (ex.: LSTM com `SELECT_TF_OPS`) podem falhar no
  navegador; o app mostra a mensagem de erro na seção "Modelos". Prefira exportar o modelo
  temporal com `TFLITE_BUILTINS` apenas (`unroll=True` no LSTM/GRU do Keras ou uma 1D-CNN).
- Esta pasta não é versionada com binários grandes por padrão; adicione os `.tflite` ao commit
  final da Entrega 02 (são pequenos, < 1 MB cada) ou publique-os junto do GitHub Pages.
