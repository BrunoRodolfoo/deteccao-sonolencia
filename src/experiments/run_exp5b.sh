#!/usr/bin/env bash
# Exp. 5b (RODADA FINAL) — transferência DMD (sonolência atuada) -> UTA-RLDD (sonolência real), PP1.
# Pré-requisito: características da rodada final em data/features/final/ (bash src/features/run_features.sh).
#   bash src/experiments/run_exp5b.sh                 # análises + CSV + relatório + figura (~20 min)
#   bash src/experiments/run_exp5b.sh --so-relatorio  # refaz só o .md e a figura a partir dos CSV
# O próprio script grava results/logs/exp5b_transferencia.log (análises); a saída do terminal vai para
# results/logs/exp5b_transferencia_stdout.log (com --so-relatorio: results/logs/exp5b_transferencia_relatorio.log).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
mkdir -p results/logs
LOGN=results/logs/exp5b_transferencia_stdout.log
case " $* " in *" --so-relatorio "*) LOGN=results/logs/exp5b_transferencia_relatorio.log ;; esac
python src/experiments/exp5_transfer.py "$@" 2>&1 | tee "$LOGN"
