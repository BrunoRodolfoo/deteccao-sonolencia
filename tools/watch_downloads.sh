#!/usr/bin/env bash
# Completa a UTA-RLDD em segundo plano, pegando cada zip da fonte que chegar primeiro:
#   1. ~/Downloads (download feito no navegador, com conta Google logada);
#   2. download anônimo gentil do Drive: um pedido por vez, pausa de 5 min quando há cota.
# Extrai cada fold assim que as duas partes estão completas, confere e apaga os zips.
# O Chrome só renomeia para .zip ao terminar (antes é .crdownload): nunca extrai pela metade.
cd "$(dirname "$0")/.." || exit 1
source .venv/bin/activate
while true; do
  python tools/prepare_data.py --uta
  n=$(ls data/uta_rldd_videos/Fold*/.ok 2>/dev/null | wc -l | tr -d ' ')
  [ "$n" -ge 5 ] && { echo "TODOS OS 5 FOLDS PRONTOS"; break; }
  sleep 120
done
