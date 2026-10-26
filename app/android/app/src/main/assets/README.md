# Assets (não versionados)

Rode `tools/fetch_assets.sh` (na pasta `app/android`) para colocar aqui:

- `face_landmarker.task` — MediaPipe Face Landmarker (float16), baixado de
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
- `blink_temporal.tflite` (+ `blink_temporal.json`) — classificador temporal [1,30,4] -> 3 classes (gerado em `models/`).
- `eye_state_int8.tflite`, `eye_state_fp32.tflite` (+ `eye_state_int8.json`/`eye_state_fp32.json`, cópias do contrato único `models/eye_state.json`) — CNN de estado do olho 48x48 cinza em [0,1], saída P(aberto) (gerado em `models/` por `src/pipeline_b/export_tflite.py`).

Sem o `.task` o app não inicia a detecção; sem os `.tflite` ele roda em modo degradado (só EAR/MAR/PERCLOS).
