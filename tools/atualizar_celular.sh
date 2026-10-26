#!/usr/bin/env bash
# Comando ÚNICO para depois de um benchmark novo no celular (ex.: Galaxy S23 Ultra na PUC).
#
#   bash tools/atualizar_celular.sh            # tudo
#   bash tools/atualizar_celular.sh --sem-poster
#   bash tools/atualizar_celular.sh --sem-artigo
#
# Pré-requisito: os pares benchmark_pwa_<data>.json + benchmark_pwa_<data>_frames.csv (PWA) ou
# bench_<data>.csv + .json (app Android nativo) em results/phone/. A PWA servida por
# app/pwa/serve_https.py já grava lá sozinha (POST /api/benchmark); downloads manuais: copie para
# results/phone/. O nome do aparelho vem dos Client Hints do Chrome Android (SM-S918* = Galaxy S23
# Ultra) ou de results/phone/aparelhos.json (iPhone).
#
# Passos (para no primeiro erro):
#   1. importa results/phone            -> results/exp4_phone_latency.{csv,md}
#   2. refaz a figura do Exp. 4         -> results/figures/fig_exp4_latencia.{pdf,png} (+ cópia em paper/figures/)
#   3. refaz os números                 -> paper/numeros.tex, poster/numeros.json, results/numeros_resumo.md
#      e confere cada um contra o arquivo de origem (src/experiments/conferir_numeros.py)
#   4. recompila o artigo               -> paper/artigo.pdf (tectonic; confere páginas e macros indefinidos)
#   5. gera o pôster                    -> poster/poster_final.{pptx,html,pdf} + poster_final_preview.png
#
# Nada é enviado para fora do computador; nenhum commit é feito.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$REPO/.venv/bin/python"
TECTONIC="${TECTONIC:-$HOME/.local/bin/tectonic}"
FAZ_ARTIGO=1
FAZ_POSTER=1
for arg in "$@"; do
  case "$arg" in
    --sem-artigo) FAZ_ARTIGO=0 ;;
    --sem-poster) FAZ_POSTER=0 ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "opção desconhecida: $arg" >&2; exit 2 ;;
  esac
done

cd "$REPO"
LOG_DIR="$REPO/results/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/atualizar_celular_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
etapa() { printf '\n==> %s\n' "$*"; }
falha() { echo "ERRO: $*" >&2; echo "log: $LOG" >&2; exit 1; }

[ -x "$PY" ] || falha "ambiente .venv não encontrado em $REPO/.venv"

etapa "1/5 Importando os benchmarks do celular (results/phone)"
"$PY" tools/import_phone_benchmark.py results/phone
"$PY" - <<'EOF'
import pandas as pd
d = pd.read_csv("results/exp4_phone_latency.csv")
print("\nExecuções por aparelho (versão do app):")
for ap, g in d.groupby("aparelho", sort=False):
    print(f"  - {ap}: {len(g)} execução(ões); app {', '.join(sorted(set(map(str, g.versao_app))))}; "
          f"total por quadro p50 {', '.join(f'{v:g}' for v in g.total_ms_p50)} ms")
if not d.aparelho.str.contains("Galaxy S23 Ultra").any():
    print("  (Galaxy S23 Ultra ainda sem benchmark: o artigo e o pôster continuam com 'medição prevista')")
EOF

etapa "2/5 Figura do Exp. 4 (agrupada por aparelho)"
"$PY" src/experiments/make_figures.py --only fig_exp4_latencia
cp results/figures/fig_exp4_latencia.pdf results/figures/fig_exp4_latencia.png paper/figures/

etapa "3/5 Números do artigo e do pôster + conferência automática"
"$PY" src/experiments/make_paper_numbers.py
"$PY" src/experiments/conferir_numeros.py || falha "a conferência dos números encontrou divergências (results/numeros_resumo.md)"

if [ "$FAZ_ARTIGO" = 1 ]; then
  etapa "4/5 Recompilando o artigo (paper/artigo.tex)"
  [ -x "$TECTONIC" ] || falha "tectonic não encontrado em $TECTONIC"
  if ! grep -q 'input{numeros' paper/artigo.tex; then
    echo "AVISO: paper/artigo.tex não faz \\input{numeros}: os números do texto NÃO são atualizados automaticamente."
  fi
  ( cd paper && "$TECTONIC" --keep-logs artigo.tex ) || falha "a compilação do artigo falhou (paper/artigo.log)"
  "$PY" - <<'EOF'
import re, sys
import pymupdf as fitz
n = len(fitz.open("paper/artigo.pdf"))
log = open("paper/artigo.log", encoding="utf-8", errors="replace").read()
und = len(re.findall(r"Undefined control sequence", log))
refs = len(re.findall(r"(Reference|Citation) `[^']+' on page \d+ undefined", log))
over = len(re.findall(r"Overfull \\hbox", log))
print(f"artigo.pdf: {n} páginas (limite 6-8); macros indefinidos: {und}; referências/citações indefinidas: {refs}; "
      f"Overfull hbox: {over}")
if und:
    sys.exit("há macros indefinidos no artigo (ex.: um \\CelularSamsung... usado sem dados?)")
if not 6 <= n <= 8:
    print("AVISO: número de páginas fora do limite 6-8")
EOF
else
  etapa "4/5 Artigo: pulado (--sem-artigo)"
fi

if [ "$FAZ_POSTER" = 1 ]; then
  etapa "5/5 Pôster (pptx + html -> pdf)"
  "$PY" poster/make_assets.py
  # --pdf imprime o HTML no Chrome headless (mudo), grava título e autores nos metadados e refaz a prévia
  "$PY" poster/build_poster.py --pdf
else
  etapa "5/5 Pôster: pulado (--sem-poster)"
fi

etapa "Concluído"
echo "Arquivos: results/exp4_phone_latency.{csv,md}, results/figures/fig_exp4_latencia.{pdf,png},"
echo "          paper/numeros.tex, poster/numeros.json, results/numeros_resumo.md,"
[ "$FAZ_ARTIGO" = 1 ] && echo "          paper/artigo.pdf,"
[ "$FAZ_POSTER" = 1 ] && echo "          poster/poster_final.{pptx,html,pdf}, poster/poster_final_preview.png"
echo "Log: $LOG"
