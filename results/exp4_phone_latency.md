# Exp. 4 — Latência no smartphone (benchmarks de 60 s)

Gerado em 2026-10-06 16:15 por `tools/import_phone_benchmark.py` a partir dos CSV brutos por quadro gravados no aparelho (p50/p95/média recalculados com numpy.percentile). Origem `android` = app nativo (TFLite nativo + MediaPipe Tasks); `pwa` = Chrome (tfjs-tflite WASM + MediaPipe WebAssembly/WebGL).

| Origem | Arquivo | Versão do app | Aparelho | Sistema | Navegador | Delegate | CNN olho | FPS | Landmarker p50 (ms) | p95 | CNN p50 (ms) | p95 | Motor/LSTM p50 (ms) | p95 | Motor v2 p50 (ms) | p95 | Total p50 (ms) | p95 | Quadros |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pwa | benchmark_pwa_2026-09-27T21-58-22-872Z_frames.csv | 1.0 | iPhone 17 Pro (512 GB) | iOS 26.6.1 | Chrome 154 (iOS, motor WebKit) | GPU | eye_state_int8.tflite | 40.51 | 19.0 | 26.0 | 3.0 | 6.0 | 1.0 | 1.0 | nan | nan | 23.0 | 30.0 | 2431 |
| pwa | benchmark_pwa_2026-09-27T22-41-35-365Z_frames.csv | 1.0 | iPhone 17 Pro (512 GB) | iOS 26.6.1 | Chrome 154 (iOS, motor WebKit) | GPU | eye_state_int8.tflite | 59.99 | 9.0 | 11.0 | 2.0 | 3.0 | 1.0 | 1.0 | nan | nan | 11.0 | 14.0 | 3600 |

Landmarker = detecção dos 478 marcos (no Android inclui a fila do modo LIVE_STREAM); CNN = dois recortes 48x48 (olho E/D) por quadro; Motor/LSTM = no Android o motor completo (piscadas + janela + LSTM), na PWA só a inferência do LSTM 30x4 (quadros em que ele rodou); Motor v2 = motor de sinais da PWA v2.x (olhos, bocejo, cabeça e nível 0-100; vazio na v1.0); Total = do recebimento do quadro ao fim do processamento.
