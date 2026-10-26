#!/usr/bin/env bash
# Exp. 6 (rodada final) — robustez noturna com o NITYMED (PP3). Incremental; pré-requisitos: sinais extraídos
# (src/features/extract_signals.py, .venv-mp), piscadas/janelas finais (bash src/features/run_features.sh, .venv) e,
# de preferência, o Exp. 5b final (models/exp5/exp5b_final_modelos.json; sem ele o modelo do DMD é treinado aqui com a
# mesma receita, e o relatório registra isso).
#   bash src/experiments/run_exp6.sh            # luminância (.venv-mp, <= 4 processos) + análise (.venv)
#   FORCE=1 bash src/experiments/run_exp6.sh    # recalcula tudo
# Logs: results/logs/exp6_final_luminancia.log e results/logs/exp6_final_noite.log.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
FLAG=""
if [ "${FORCE:-0}" = "1" ]; then FLAG="--force"; fi
# 1) iluminação medida nos vídeos (precisa do OpenCV do .venv-mp)
source .venv-mp/bin/activate
python src/experiments/exp6_luminancia.py --workers 4 $FLAG > results/logs/exp6_final_luminancia.log 2>&1
deactivate
# 2) análise, modelos, figuras e relatórios (.venv); o próprio script grava results/logs/exp6_final_noite.log
source .venv/bin/activate
python src/experiments/exp6_noite.py $FLAG > /dev/null
deactivate
echo "Exp. 6 concluído: results/exp6_noite.md"
