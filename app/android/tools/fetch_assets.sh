#!/usr/bin/env bash
# Baixa o modelo do MediaPipe Face Landmarker e copia os modelos TFLite gerados em
# <repo>/models/ para app/src/main/assets/. Rode a partir de qualquer pasta:
#   bash app/android/tools/fetch_assets.sh
# Os arquivos de assets não são versionados (ver app/android/.gitignore).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # app/android/tools
ANDROID_DIR="$(cd "$HERE/.." && pwd)"                          # app/android
REPO_DIR="$(cd "$ANDROID_DIR/../.." && pwd)"                   # raiz do repositório
ASSETS="$ANDROID_DIR/app/src/main/assets"
MODELS="${MODELS_DIR:-$REPO_DIR/models}"
mkdir -p "$ASSETS"

# 1) Face Landmarker (float16, versão 1) — ~3,7 MB
TASK_URL="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
if [ ! -s "$ASSETS/face_landmarker.task" ]; then
  echo "[fetch_assets] baixando face_landmarker.task ..."
  curl -fL --retry 3 -o "$ASSETS/face_landmarker.task" "$TASK_URL"
else
  echo "[fetch_assets] face_landmarker.task já existe (pulando download)"
fi
ls -la "$ASSETS/face_landmarker.task"

# 2) Modelos TFLite do projeto (gerados por src/*/export_tflite*.py em models/)
missing=0
for f in blink_temporal.tflite eye_state_int8.tflite eye_state_fp32.tflite; do
  if [ -s "$MODELS/$f" ]; then
    cp -f "$MODELS/$f" "$ASSETS/$f"
    # metadados opcionais (mesmo nome, extensão .json). A CNN de olho tem um único
    # contrato compartilhado, models/eye_state.json (gerado por src/pipeline_b/export_tflite.py):
    # ele é copiado com o nome de cada variante para o app achá-lo ao lado do .tflite.
    j="${f%.tflite}.json"
    if [ -s "$MODELS/$j" ]; then
      cp -f "$MODELS/$j" "$ASSETS/$j"
    elif [[ "$f" == eye_state_* ]] && [ -s "$MODELS/eye_state.json" ]; then
      cp -f "$MODELS/eye_state.json" "$ASSETS/$j"
    fi
    echo "[fetch_assets] copiado $f ($(wc -c < "$ASSETS/$f") bytes)"
  else
    echo "[fetch_assets] AVISO: $MODELS/$f não encontrado — o app rodará em modo degradado para este modelo" >&2
    missing=$((missing+1))
  fi
done

echo "[fetch_assets] concluído. Conteúdo de $ASSETS:"
ls -la "$ASSETS"
[ "$missing" -gt 0 ] && echo "[fetch_assets] $missing modelo(s) ausente(s); gere-os antes do benchmark completo." >&2
exit 0
