#!/usr/bin/env bash
# RODADA FINAL — etapa de características (Exp. 5): refaz toda esta etapa a partir dos sinais por quadro já
# extraídos em data/signals/<base>/ (a extração em si, extract_signals.py, roda no .venv-mp e não é refeita aqui).
#
#   bash src/features/run_features.sh            # incremental nas piscadas (pula arquivo com hash igual)
#   FORCE=1 bash src/features/run_features.sh    # recalcula as piscadas de todos os arquivos
#
# Etapas (logs em results/logs/exp5_final_*.log):
#   1. testes (asserts puros, dados sintéticos)
#   2. piscadas + limiar por unidade de referência          -> data/features/final/<base>/blinks*
#   3. DMD: alinhamento (s5 e distração), validação do detector, bocejos -> results/exp5_dmd_*.csv
#   4. janelas de 10 s (principal), 5 s e 15 s (passo 5 s)  -> data/features/final/<base>/windows_<L>s.csv
#   5. sequências de 30 piscadas + mapeamento oficial dos 5 folds (último terço; depois primeiro terço)
#   6. relatório                                             -> results/exp5_caracteristicas.md
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
mkdir -p results/logs
F=""; [ "${FORCE:-0}" = "1" ] && F="--force"
t0=$(date +%s)
echo "[$(date +%H:%M:%S)] 1/6 testes"
python src/features/test_features.py 2>&1 | tee results/logs/exp5_final_testes.log
echo "[$(date +%H:%M:%S)] 2/6 piscadas"
python src/features/blinks.py --dataset todos $F > results/logs/exp5_final_blinks.log 2>&1
tail -3 results/logs/exp5_final_blinks.log
echo "[$(date +%H:%M:%S)] 3/6 DMD: alinhamento e validação do detector"
python src/features/dmd_labels.py > results/logs/exp5_final_dmd_labels.log 2>&1
grep -E "Distração|Agregado" results/logs/exp5_final_dmd_labels.log || true
echo "[$(date +%H:%M:%S)] 4/6 janelas (10 s principal; 5 s e 15 s sensibilidade)"
for L in 10 5 15; do
  python src/features/windows.py --janela "$L" --passo 5 > "results/logs/exp5_final_windows_${L}s.log" 2>&1
  grep -E "por uso" "results/logs/exp5_final_windows_${L}s.log" || true
done
echo "[$(date +%H:%M:%S)] 5/6 sequências e mapeamento oficial (5 folds)"
python src/features/sequences.py > results/logs/exp5_final_sequences.log 2>&1
python src/features/sequences.py --terco primeiro --mapa-existente > results/logs/exp5_final_sequences_primeiro_terco.log 2>&1
grep -E "sequências:|Mapeamento \(5" results/logs/exp5_final_sequences.log || true
echo "[$(date +%H:%M:%S)] 6/6 relatório"
python src/features/report_exp5.py 2>&1 | tee results/logs/exp5_final_relatorio.log
echo "[$(date +%H:%M:%S)] fim ($(( $(date +%s) - t0 )) s)"
