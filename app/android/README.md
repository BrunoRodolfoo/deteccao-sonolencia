# App Android nativo — Detecção de Sonolência (Exp. 4: custo embarcado no Galaxy S23 Ultra)

> **Situação em 27/09/2026 (leia antes):** este app mantém a **lógica ANTERIOR (v1)** — alertas por olho fechado,
> PERCLOS e classe do LSTM — que foi **substituída** na PWA v2.1 (nível de sonolência 0-100 por olhos, bocejo e
> cabeça, com o aprendizado de máquina só como estimativa; `app/pwa/README.md`). A v1 deu falso alarme por PERCLOS com
> uma pessoa acordada (PWA v1 no iPhone, 27/09), e a regra v1 da PWA (EAR + PERCLOS > 15 % + olho fechado >= 1 s, sem
> o LSTM) ficou 55,6 % do tempo em vermelho na direção real do DMD (`results/app_validacao_dmd.md`); este app Kotlin
> não foi avaliado nesses dados. O projeto **nunca foi compilado**. A **versão de demonstração é a PWA**; este app
> serve, no máximo, como caminho alternativo para medir latência de TFLite nativo. O aparelho medido até agora é o
> iPhone 17 Pro (PWA); o Galaxy S23 Ultra é medição prevista (`docs/GUIA_EQUIPE.md`).

Aplicativo Kotlin que roda **totalmente no aparelho** o pipeline do TCC:

```
CameraX (frontal, 640x480) -> MediaPipe FaceLandmarker (LIVE_STREAM, GPU/CPU)
  -> EAR / MAR / PERCLOS (mesmos índices do protótipo src/realtime/metrics.py)
  -> detector de piscadas (limiar adaptativo) -> 4 características por piscada
     (Frequency, Amplitude, Duration, Velocity; Ghoddoosian et al., CVPRW 2019)
  -> z-score por sujeito (calibração em estado alerta) -> janela das últimas 30 piscadas
  -> blink_temporal.tflite [1,30,4] -> {alerta, pouca vigilância, sonolento}
  -> recorte dos olhos 48x48 cinza -> eye_state_{int8,fp32}.tflite -> P(olho aberto)
  -> alertas (beep + vibração + borda vermelha) e benchmark de 60 s (CSV + JSON)
```

**Privacidade:** nenhum quadro é salvo. Os bitmaps ficam só em memória durante o
processamento; o benchmark grava apenas números (latências, EAR, MAR, PERCLOS,
probabilidades, classe) em CSV/JSON na pasta privada do app.

> Este projeto foi escrito em uma máquina sem JDK/Android SDK, portanto **ainda não foi
> compilado**. As versões de todas as dependências foram verificadas nos repositórios Maven
> em 20/09/2026 (tabela no fim). Se o Android Studio sugerir atualizar algo, aceite.

## Requisitos

| Item | Versão |
|---|---|
| Android Studio | Narwhal 3 Feature Drop (2025.1.3) ou mais recente (AGP 8.13 exige essa versão ou superior) |
| JDK | 17 (embutido no Android Studio) |
| Android SDK | Platform 36 (compileSdk/targetSdk = 36), Build-Tools 35.0.0+ |
| Gradle | 8.14.5 (baixado automaticamente pelo wrapper) |
| Aparelho | Galaxy S23 Ultra (Android 14+, arm64-v8a); qualquer Android 8.0+ arm64 funciona |

O projeto restringe o ABI a `arm64-v8a` (`ndk.abiFilters`) para reduzir o APK. Em um
emulador x86_64, remova essa linha de `app/build.gradle.kts`.

## Passo a passo

1. **Gerar/obter os modelos** (na raiz do repositório):
   - `models/blink_temporal.tflite` (+ `blink_temporal.json`): classificador temporal
     [1,30,4] float32 -> [1,3] softmax (frentes de treino/exportação);
   - `models/eye_state_int8.tflite` e `models/eye_state_fp32.tflite` (+ contrato único
     `models/eye_state.json`, gerado por `src/pipeline_b/export_tflite.py`): CNN de
     estado do olho, entrada [1,48,48,1] cinza em [0,1] (fp32: float32; int8: E/S uint8,
     escala 1/255 e 1/256, ponto zero 0 — lidos do próprio tensor), saída [1,1] sigmoide
     P(aberto). `fetch_assets.sh` copia `eye_state.json` como `eye_state_int8.json` e
     `eye_state_fp32.json`; como esse contrato não traz `input_range`/`mean`/`std`, o app
     usa os padrões (pixel/255, sem centralização), que é exatamente o pré-processamento do treino.
   O app lê shape, dtype e parâmetros de quantização direto do interpretador; o JSON só
   ajusta a normalização (`input_range`, `mean`, `std`) e os nomes das classes
   (`class_names` / `open_index`). Sem os `.tflite`, o app roda em **modo degradado**
   (só EAR/MAR/PERCLOS e alerta de olhos fechados).
2. **Baixar os assets** (Face Landmarker float16 + cópia dos modelos):
   ```bash
   bash app/android/tools/fetch_assets.sh
   ```
   (Se preferir manualmente: baixe
   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
   e coloque em `app/android/app/src/main/assets/` junto com os `.tflite`/`.json`.)
3. **Abrir `app/android/` no Android Studio** (File > Open), aguardar o Gradle Sync.
4. **Testes unitários (JVM, sem aparelho):** botão direito em `app/src/test` > *Run Tests*,
   ou `./gradlew :app:testDebugUnitTest` no terminal (o wrapper `gradlew` + `gradle-wrapper.jar`
   da versão 8.14.5 já está incluído; exige JDK 17 no PATH ou `JAVA_HOME`).
   Os testes cobrem `DrowsinessEngine`, `BlinkDetector`, as fórmulas das 4
   características, `FaceMetrics` (EAR/MAR/índices), `PerclosWindow` e `LatencyStats`.
5. **Instalar no S23 Ultra:** ativar *Opções do desenvolvedor > Depuração USB*, conectar
   o cabo, selecionar o aparelho e *Run* (▶). Aceitar a permissão de câmera.
   APK de release: *Build > Generate Signed App Bundle/APK* (ou `./gradlew :app:assembleRelease`).

## Usando o app

- **Iniciar**: liga o motor. Fase 1 (≈3 s): calibração do limiar de EAR — mantenha os
  olhos abertos (limiar = 0,75 × mediana). Fase 2: calibração das piscadas em estado
  **alerta** (as primeiras 20 piscadas ou 2 min, mínimo 5) => média/desvio por
  característica (z-score por sujeito, como o terço de calibração do UTA-RLDD). Fase 3:
  monitoramento; a cada piscada a janela de 30 é reclassificada.
- **Calibrar**: refaz só a fase 2 (mantém o limiar).
- **Benchmark 60 s**: grava uma linha por quadro em
  `Android/data/br.pucpr.sonolencia/files/benchmarks/bench_<data>.csv` e um resumo
  `bench_<data>.json` (aparelho `Build.MODEL`/`Build.SOC_MODEL`, Android, delegate,
  tamanho dos modelos, p50/p95/média por estágio, FPS médio).
- **Compartilhar**: envia o último par CSV/JSON (FileProvider) por e-mail/Drive/etc.
- Toque no texto "Modelos:" do HUD para alternar a CNN entre `eye_state_int8` e
  `eye_state_fp32` (quando ambos estão nos assets) e repita o benchmark para comparar.
- HUD: FPS; latência do último quadro e p50/p95 das últimas 300 amostras por estágio
  (landmarker = `detectAsync` → callback; CNN = 2 recortes; total = envio do quadro →
  fim do motor); EAR (e limiar), MAR, PERCLOS (60 s); P(aberto) por olho; piscadas e
  tamanho da sequência; classe e probabilidades suavizadas (α = 0,3).
- Alertas (beep `ToneGenerator` + vibração + borda vermelha): olhos fechados ≥ 1,0 s;
  PERCLOS > 0,15 (após ≥ 10 s de janela); classe *sonolento* em 3 janelas consecutivas.

### Protocolo sugerido para o Exp. 4 (S23 Ultra)

1. Ambiente iluminado, aparelho em um suporte a ~40 cm do rosto, bateria > 50 %,
   modo avião opcional (evita interferência de notificações).
2. Iniciar → esperar a calibração do limiar → **Benchmark 60 s** (com CNN int8).
3. Tocar em "Modelos:" para trocar para fp32 → novo **Benchmark 60 s**.
4. Repetir 3 vezes cada configuração (a importação lista todas as execuções).
5. Compartilhar os arquivos para o computador e importar:
   ```bash
   python app/android/tools/import_phone_benchmark.py ~/Downloads/bench_*.csv --out results
   ```
   Isso gera `results/exp4_phone_latency.csv`, `results/exp4_phone_latency.md` e um log em
   `results/logs/`. Os p50/p95 são recalculados a partir do CSV bruto (numpy.percentile).

## Estrutura

```
app/android/
├── build.gradle.kts, settings.gradle.kts, gradle.properties, gradlew(.bat), gradle/wrapper/
├── tools/fetch_assets.sh            # baixa .task e copia models/*.tflite(+json) para assets
├── tools/import_phone_benchmark.py  # CSV/JSON do celular -> results/exp4_phone_latency.{csv,md}
└── app/
    ├── build.gradle.kts, proguard-rules.pro
    └── src/main/java/br/pucpr/sonolencia/
        ├── MainActivity.kt          # CameraX + permissões + HUD + botões + benchmark
        ├── FaceLandmarkerHelper.kt  # MediaPipe Tasks (LIVE_STREAM, GPU->CPU)
        ├── FaceMetrics.kt           # EAR/MAR/caixa do olho (lógica pura)
        ├── BlinkDetector.kt         # piscadas, 4 características, z-score (lógica pura)
        ├── PerclosWindow.kt, LatencyStats.kt
        ├── DrowsinessEngine.kt      # máquina de estados, alertas, janela de 30 (lógica pura)
        ├── TfliteModels.kt          # Interpreter TFLite: temporal e estado do olho (quantização lida do tensor)
        ├── ModelMetadata.kt, EyeCropper.kt, BenchmarkRecorder.kt, Alerter.kt, AlertOverlayView.kt
    └── src/test/java/br/pucpr/sonolencia/   # JUnit 4 (JVM puro)
```

## Decisões de implementação (para o artigo)

- EAR = (|p2−p6| + |p3−p5|) / (2|p1−p4|), com p1..p6 = [362,385,387,263,373,380] (olho
  esquerdo) e [33,160,158,133,153,144] (direito), como no protótipo e em `blink_video.py`
  do UTA-RLDD. (O artigo escreve a fórmula sem o fator 2; após o z-score por sujeito a
  escala é irrelevante para Amplitude e Velocity.)
- Piscada: EAR < limiar; start = primeiro quadro abaixo, bottom = argmin, end = último
  abaixo; mínimo 2 quadros; fechamentos > 1,5 s não viram piscada (vão para o alerta de
  olhos fechados). Duration = end − start + 1; Amplitude = (EAR_start − 2·EAR_bottom +
  EAR_end)/2; Velocity = (EAR_end − EAR_bottom)/(end − bottom) (piso de 1 quadro);
  Frequency = 100 · (piscadas até i)/(quadros até end_i). Ordem do vetor: F, A, D, V.
- Duration e Frequency dependem da taxa de quadros; o z-score por sujeito (média/desvio
  das piscadas da calibração) compensa deslocamento e escala, como no protocolo original.
- Janela com menos de 30 piscadas é preenchida com zeros à esquerda (igual a
  `unroll_in_time` em Preprocessing.py).
- Latência do landmarker inclui a fila do modo LIVE_STREAM (no máximo 2 quadros em voo;
  `STRATEGY_KEEP_ONLY_LATEST` no CameraX).
- Recorte do olho para a CNN: caixa quadrada de 1,6 x |p1−p4| centrada na média dos 6
  pontos, deslocada para dentro da imagem quando encosta na borda (continua quadrada),
  cinza 0,299R+0,587G+0,114B, redimensionada para 48x48 com filtro bilinear. No treino, os
  recortes "crop" do CEW usaram caixa de 0,35 x largura da face (36 px em faces de 100 px,
  ~1,6 x a largura do olho) e INTER_AREA; os do MRL são os recortes originais do dataset.
- Z-score por sujeito: desvio populacional (ddof=0, como `numpy.std` em Preprocessing.py)
  com piso de 1e-6; se uma característica não variar nas piscadas de calibração (ex.:
  duração sempre igual em poucas piscadas), o z-score dessa característica explode. Por
  isso o padrão é 20 piscadas de calibração (mínimo 5 só após 2 min).

## Versões verificadas (20/09/2026)

| Dependência | Versão | Fonte da verificação |
|---|---|---|
| Android Gradle Plugin | 8.13.2 | dl.google.com/dl/android/maven2 (últimas 8.x: 8.13.0/1/2; 9.4.1 é a mais nova, mas muda o DSL) |
| Gradle | 8.14.5 | services.gradle.org/versions/all (AGP 8.13 exige ≥ 8.13) |
| Kotlin | 2.2.21 | repo1.maven.org (kotlin-gradle-plugin) |
| androidx.core:core-ktx | 1.17.0 | minCompileSdk=36 (aar-metadata); 1.19.0 exige compileSdk 37 e AGP 9.1 |
| androidx.appcompat | 1.7.1 | minCompileSdk=34 |
| androidx.activity:activity-ktx | 1.12.4 | minCompileSdk=36 |
| androidx.constraintlayout | 2.2.1 | minCompileSdk=34 |
| androidx.lifecycle:lifecycle-runtime-ktx | 2.9.4 | Maven do Google |
| androidx.camera:* | 1.6.2 | estável 26/08/2026; minCompileSdk=36, AGP ≥ 8.9.1 |
| com.google.mediapipe:tasks-vision | 0.10.35 | Maven do Google (também existe 1.0.0 com a mesma API) |
| org.tensorflow:tensorflow-lite | 2.17.0 | Maven Central; depende de com.google.ai.edge.litert:litert:1.0.1 (pacote org.tensorflow.lite) |
| kotlinx-coroutines-android | 1.10.2 | Maven Central |
| junit | 4.13.2 | Maven Central |

APIs usadas (conferidas nos JARs 0.10.35 / litert 1.0.1): `FaceLandmarker.createFromOptions`,
`FaceLandmarkerOptions.builder().setBaseOptions/.setRunningMode(RunningMode.LIVE_STREAM)/.setNumFaces/
.setResultListener/.setErrorListener`, `BaseOptions.builder().setModelAssetPath/.setDelegate(Delegate.GPU|CPU)`,
`detectAsync(MPImage, long)`, `BitmapImageBuilder(bitmap).build()`, `FaceLandmarkerResult.faceLandmarks()/timestampMs()`,
`NormalizedLandmark.x()/y()`; `org.tensorflow.lite.Interpreter(ByteBuffer, Options)`, `Interpreter.Options.setNumThreads`,
`run(Object, Object)`, `getInputTensor/getOutputTensor`, `Tensor.shape()/dataType()/quantizationParams()`.

## O que só um build real confirma

Revisão estática (20/09/2026) conferiu versões (Maven do Google/Central, services.gradle.org),
nomes de classes/métodos nos JARs reais de `tasks-vision:0.10.35`, `tasks-core:0.10.35`,
`litert:1.0.1`/`litert-api:1.0.1` e `camera-core:1.6.2`, o contrato dos três `.tflite`
(carregados com o interpretador Python) e a integridade do wrapper (`gradle-wrapper.jar` e
`gradlew` idênticos ao tag v8.14.5). Ficam para o primeiro build no Android Studio:

1. Sintaxe do Gradle Kotlin DSL nas versões exatas (`androidResources { noCompress += ... }`,
   `ndk { abiFilters += ... }`, `kotlin { compilerOptions { jvmTarget } }`).
2. Resolução de dependências: `listenablefuture` (CameraX 1.0 vs. Guava 27.0.1-android do
   MediaPipe, que traz a versão `9999.0-empty-to-avoid-conflict-with-guava`) e o `lint`
   de release (`MissingDefaultResource` para o ícone só em `mipmap-anydpi-v26`, coberto por
   `minSdk 26`).
3. Conversão SAM das lambdas Kotlin para `OutputHandler.ResultListener`/`ErrorListener`
   (o exemplo oficial usa referências de método `this::fn`; se o compilador reclamar,
   troque por `setResultListener(::onResult)`).
4. Inferência dos tipos `Float?`/`Integer` nos setters do `FaceLandmarkerOptions.Builder`
   (recebem tipos encaixotados; `0.5f` e `1` são convertidos automaticamente).
5. Comportamento em tempo de execução: delegate GPU do FaceLandmarker no S23 Ultra (há
   fallback para CPU), orientação/espelhamento da câmera frontal (mesma matriz do exemplo
   oficial `mediapipe-samples/face_landmarker/android`) e latências reais.
6. Testes JVM (`./gradlew :app:testDebugUnitTest`): os valores esperados foram conferidos à
   mão, mas ainda não executados.

## Solução de problemas

- *"face_landmarker.task não encontrado"*: rode `tools/fetch_assets.sh`.
- Delegate GPU falha ao criar: o app cai para CPU automaticamente (HUD mostra o delegate).
- Latência alta/FPS baixo: feche outros apps; confira se `abiFilters` inclui `arm64-v8a`.
- `Cannot access class ListenableFuture`: use CameraX ≥ 1.6.1 (já configurado).
- Resultado do MediaPipe fora de ordem: os timestamps enviados são `uptimeMillis` e
  estritamente crescentes (garantido em `analyzeFrame`).
