#!/usr/bin/env bash
# Baixa para app/pwa/vendor/ as dependências da PWA que, sem esta pasta, vêm da internet (CDN), nas MESMAS
# versões fixas do app/pwa/app.js: MediaPipe Tasks Vision 1.0.1 (módulo + WASM), o modelo face_landmarker.task
# (float16, versão 1), TensorFlow.js 4.22.0 (core + backend CPU) e tfjs-tflite 0.0.1-alpha.10 (com os módulos WASM).
# Com a pasta completa, o app servido pelo serve_https.py funciona sem internet (celular sem dados, ligado ao
# computador pelo Acesso Pessoal); sem ela, o app usa a CDN. A pasta não é versionada (.gitignore); ~36 MB.
#
#   bash tools/baixar_dependencias_app.sh               # baixa o que falta e confere tudo
#   bash tools/baixar_dependencias_app.sh --conferir    # só confere, sem rede (saída 1 se faltar algo)
#   bash tools/baixar_dependencias_app.sh --pasta DIR   # outra pasta vendor/ (ex.: cópia servida com --dir)
#
# Cada arquivo só entra na pasta depois de conferido pelo SHA-256 abaixo: o mesmo que a CDN publica para o pacote
# e, no modelo, o arquivo cujo MD5 está em tools/app_validacao/extrair_sinais_app.py. Para trocar uma versão, mude
# o app.js e esta lista juntos. Em cada pacote, o arquivo que o app carrega primeiro vem por último na lista: uma
# interrupção no meio nunca deixa o módulo sem os WASM dele.
set -euo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$RAIZ/app/pwa/vendor"
CONFERIR=0
while [ $# -gt 0 ]; do
  case "$1" in
    --conferir) CONFERIR=1 ;;
    --pasta) DEST="$2"; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "opção desconhecida: $1 (use --help)" >&2; exit 2 ;;
  esac
  shift
done

NPM="https://cdn.jsdelivr.net/npm"
# arquivo em vendor/ | URL | SHA-256
ARQUIVOS="
face_landmarker/float16/1/face_landmarker.task|https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task|64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
tasks-vision@1.0.1/wasm/vision_wasm_internal.wasm|$NPM/@mediapipe/tasks-vision@1.0.1/wasm/vision_wasm_internal.wasm|8da277a733926eacd0474b8704b36742d6ec3231c57a860c5b889dff8f1df886
tasks-vision@1.0.1/wasm/vision_wasm_internal.js|$NPM/@mediapipe/tasks-vision@1.0.1/wasm/vision_wasm_internal.js|e170ee67dd4e16c1a6fcd8840a206687e5a59b22c20e4a902bc445b095454d73
tasks-vision@1.0.1/wasm/vision_wasm_nosimd_internal.wasm|$NPM/@mediapipe/tasks-vision@1.0.1/wasm/vision_wasm_nosimd_internal.wasm|a28483cd42e74e855bf5ebdb6b40d9b66a5b49e35e95020bc97669e6822a3192
tasks-vision@1.0.1/wasm/vision_wasm_nosimd_internal.js|$NPM/@mediapipe/tasks-vision@1.0.1/wasm/vision_wasm_nosimd_internal.js|e81d715a3d42cc3373602eb2f7aff795d164934db680e32496b65dab537f9658
tasks-vision@1.0.1/vision_bundle.mjs|$NPM/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs|d885630c297c0b20b1fe86096cb06291c4c8080876f27852e724f24ac603713f
tfjs-core@4.22.0/dist/tf-core.min.js|$NPM/@tensorflow/tfjs-core@4.22.0/dist/tf-core.min.js|0924438788ff7da57b14d0f7c45a05d81611a5a4b52653da6d065e86d348757c
tfjs-backend-cpu@4.22.0/dist/tf-backend-cpu.min.js|$NPM/@tensorflow/tfjs-backend-cpu@4.22.0/dist/tf-backend-cpu.min.js|4f232e90455e04cc5f69a691295d07a881a4ad93d7f229598459b9b533259d83
tfjs-tflite@0.0.1-alpha.10/wasm/tflite_web_api_cc_simd.wasm|$NPM/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/tflite_web_api_cc_simd.wasm|40f49cc2d458ec09bb43db160e9e55f4b3c790a4c1d1fa280f8a61796296cf8e
tfjs-tflite@0.0.1-alpha.10/wasm/tflite_web_api_cc_simd.js|$NPM/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/tflite_web_api_cc_simd.js|d5c71816525140d7a35b59a908d3b07d42741576630e647d246c19c6fa72d63d
tfjs-tflite@0.0.1-alpha.10/wasm/tflite_web_api_cc.wasm|$NPM/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/tflite_web_api_cc.wasm|f1eb42f96ae91e5ef34e2f833f011373ca1c40fdc9d4062ec21eb07b886ea54f
tfjs-tflite@0.0.1-alpha.10/wasm/tflite_web_api_cc.js|$NPM/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/tflite_web_api_cc.js|08e963fcde11dc041feeac97ff711b3a157614edc98ff8bfa309557189b7368d
tfjs-tflite@0.0.1-alpha.10/dist/tf-tflite.min.js|$NPM/@tensorflow/tfjs-tflite@0.0.1-alpha.10/dist/tf-tflite.min.js|2fb09bf069728bf4b63c51c72a3a6bcf415f3732c28e85624a0fd9bb8edabd2e
"

if command -v shasum >/dev/null 2>&1; then sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
else sha() { sha256sum "$1" | cut -d' ' -f1; }; fi

TMP=""
ok=0; baixados=0; faltam=0
while IFS='|' read -r rel url hash; do
  [ -n "$rel" ] || continue
  alvo="$DEST/$rel"
  if [ -f "$alvo" ] && [ "$(sha "$alvo")" = "$hash" ]; then ok=$((ok + 1)); continue; fi
  if [ "$CONFERIR" -eq 1 ]; then echo "falta ou difere: vendor/$rel"; faltam=$((faltam + 1)); continue; fi
  if [ -z "$TMP" ]; then TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT; fi
  echo "baixando $url"
  if ! curl -fL --retry 3 --connect-timeout 15 -sS -o "$TMP/arquivo" "$url"; then
    echo "ERRO: não baixou $url (sem internet?). O app continua usando a CDN para o que faltar." >&2
    exit 1
  fi
  if [ "$(sha "$TMP/arquivo")" != "$hash" ]; then
    echo "ERRO: SHA-256 diferente do esperado em $url; o arquivo NÃO foi copiado para vendor/." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$alvo")"
  mv -f "$TMP/arquivo" "$alvo"
  baixados=$((baixados + 1))
done <<< "$ARQUIVOS"

if [ "$CONFERIR" -eq 1 ]; then
  if [ "$faltam" -gt 0 ]; then echo "vendor/ incompleta: $ok arquivos conferidos, $faltam faltando ou diferentes ($DEST)"; exit 1; fi
  echo "vendor/ completa: $ok arquivos conferidos ($DEST)"
  exit 0
fi
echo "vendor/ completa: $((ok + baixados)) arquivos conferidos ($baixados baixados agora), $(du -sh "$DEST" | cut -f1 | tr -d " ") em $DEST"
echo "O app servido por app/pwa/serve_https.py agora funciona sem internet."
