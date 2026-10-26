#!/usr/bin/env bash
# Exp. 5a (rodada final) — validação do nosso extrator contra as características oficiais da UTA-RLDD, nos 5 folds.
# Pré-requisito: bash src/features/run_features.sh (sequências, mapeamento oficial e índices da rodada final).
#   bash src/experiments/run_exp5a.sh                 # tudo (opções extras vão para exp5a_validacao.py)
#   bash src/experiments/run_exp5a.sh --sem-loso      # sem o LOSO (teste rápido)
# Etapas (cada script grava o próprio log em results/logs/):
#   1. detector de rosto do dlib com o pré-processamento dos autores (.venv-mp; incremental; <= 4 processos)
#   2. verificação independente do mapeamento sujeito oficial <-> pasta (.venv)
#   3. análise, figuras e relatório (.venv) -> results/exp5a_validacao.md
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
mkdir -p results/logs
source .venv-mp/bin/activate
python src/experiments/exp5a_dlib_rosto.py --workers 4 > /dev/null
deactivate
source .venv/bin/activate
python src/experiments/exp5a_mapeamento_indep.py > /dev/null
python src/experiments/exp5a_validacao.py "$@" > /dev/null
deactivate
echo "Exp. 5a concluído: results/exp5a_validacao.md (logs em results/logs/exp5a_*)"
