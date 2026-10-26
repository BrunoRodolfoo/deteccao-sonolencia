#!/usr/bin/env bash
# =============================================================================
# run_all.sh — reproduz o trabalho inteiro, na ordem final (27/09/2026):
#
#   etapa         ambiente  o que faz                                                        tempo (Mac M5)
#   dados         .venv     baixa as blink features da UTA-RLDD; confere os vídeos em data/    ~1 min
#                           (com --preparar-videos: extrai zips/tar.gz já baixados)
#   sinais        .venv-mp  vídeo -> EAR/MAR/pose por quadro (MediaPipe 0.10.21), 3 bases       ~2,5 h
#                           (DMD s5 2,6 min + distração 24 min; NITYMED 2,2 min; UTA 60 sujeitos
#                           ~2 h); incremental; depois alinhamento DMD e resumo (.venv)
#   caracteristicas .venv   piscadas, janelas de 10 s (5/15 s), sequências de 30 piscadas      ~3 min
#   exp1          .venv     Exp. 1 intra UTA-RLDD (características oficiais, 5 folds)           ~10 min
#   exp2          .venv     Exp. 2 vazamento de identidade                                     ~13 min
#   exp3          .venv     Exp. 3 EyeNet MRL <-> CEW + Exp. 4 desktop (TFLite fp32/int8)       ~16 min
#   exp5a         ambos     Exp. 5a nosso extrator x características oficiais (dlib no .venv-mp) ~5 min
#                           (+ ~1 h na primeira vez: detector do dlib em 182 vídeos; incremental)
#   exp5b         .venv     Exp. 5b transferência DMD (atuado) -> UTA-RLDD (real), PP1           ~11-20 min
#                           + robustez (exp5b_robustez.py -> results/exp5b_robustez.{csv,md})   ~15 min
#   exp6          ambos     Exp. 6 noite (NITYMED), PP3 (luminância no .venv-mp ~40 min + 1 min) ~40 min
#   figuras       .venv     amostras do MRL Eye + figuras de resultados                        ~1 min
#   documentos    .venv     tools/atualizar_celular.sh: importa results/phone (Exp. 4 celular),  ~1 min
#                           figura do Exp. 4, paper/numeros.tex + poster/numeros.json (conferidos),
#                           recompila o artigo (tectonic) e gera o pôster (Chrome headless, mudo)
#
# Uso (na raiz do repositório; o script chama .venv/bin/python e .venv-mp/bin/python diretamente):
#   bash run_all.sh                          # tudo (~4 h com os vídeos já em data/)
#   bash run_all.sh --so exp5b,exp6,documentos   # só estas etapas (nomes separados por vírgula)
#   bash run_all.sh --pular sinais,exp5a     # tudo, menos estas etapas
#   bash run_all.sh --desde exp5b            # desta etapa até o fim
#   bash run_all.sh --sem-artigo --sem-poster   # a etapa "documentos" não recompila artigo/pôster
#   bash run_all.sh --preparar-videos        # etapa "dados" também extrai os pacotes baixados (UTA/DMD)
#   bash run_all.sh --with-optional          # exp3 inclui MobileNetV3 (fold 1, ~7 min) e ablação da BN (~3 min)
#   bash run_all.sh --smoke                  # teste rápido de exp1/exp3 (2 épocas; só arquivos *_smoke)
#   bash run_all.sh --dry-run                # só imprime os comandos
#   bash run_all.sh --lista                  # lista as etapas
# Atalhos antigos: --skip-blink (= --pular exp1,exp2), --skip-eye (= --pular exp3).
#
# Pré-requisitos por etapa: data/README.md (como obter cada base). Etapas cujos dados faltam são
# puladas com aviso. Nada sai do computador (exceto os downloads da etapa "dados"); nenhum commit é feito.
# Semente 42 em todos os experimentos. Logs: results/logs/ (um por execução de cada script).
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
REPO="$(pwd)"
export PATH="$HOME/.local/bin:$PATH"

ETAPAS=(dados sinais caracteristicas exp1 exp2 exp3 exp5a exp5b exp6 figuras documentos)
SO=""; PULAR=""; DESDE=""; OPTIONAL=0; SMOKE=0; DRY=0; PREP=0; DOC_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --so) SO="$2"; shift ;;
    --pular) PULAR="$PULAR,$2"; shift ;;
    --desde) DESDE="$2"; shift ;;
    --skip-blink) PULAR="$PULAR,exp1,exp2" ;;
    --skip-eye) PULAR="$PULAR,exp3" ;;
    --sem-artigo) DOC_ARGS+=(--sem-artigo) ;;
    --sem-poster) DOC_ARGS+=(--sem-poster) ;;
    --preparar-videos) PREP=1 ;;
    --with-optional) OPTIONAL=1 ;;
    --smoke) SMOKE=1 ;;
    --dry-run) DRY=1 ;;
    --lista) printf '%s\n' "${ETAPAS[@]}"; exit 0 ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "opção desconhecida: $1 (use --help)" >&2; exit 2 ;;
  esac
  shift
done
for lista in "$SO" "$PULAR" "$DESDE"; do
  for e in ${lista//,/ }; do
    case " ${ETAPAS[*]} " in *" $e "*) ;; *) echo "etapa desconhecida: $e (use --lista)" >&2; exit 2 ;; esac
  done
done

# --smoke sem --so: só as etapas rápidas de treino (exp1/exp2/exp3), que gravam arquivos *_smoke
[ "$SMOKE" -eq 1 ] && [ -z "$SO" ] && SO="exp1,exp2,exp3"
PY="${PYTHON:-$REPO/.venv/bin/python}"
PYMP="${PYTHON_MP:-$REPO/.venv-mp/bin/python}"
if [ "$DRY" -eq 0 ]; then
  "$PY" -c "import numpy, sklearn, tensorflow" 2>/dev/null \
    || { echo "ERRO: ambiente .venv ausente ou incompleto (uv pip install -r requirements.txt)" >&2; exit 1; }
fi
mkdir -p results/logs results/figures models data

ATIVA_DESDE=0; [ -z "$DESDE" ] && ATIVA_DESDE=1
quer() {  # quer ETAPA -> 0 se a etapa deve rodar
  local e="$1"
  [ "$e" = "$DESDE" ] && ATIVA_DESDE=1
  [ "$ATIVA_DESDE" -eq 1 ] || return 1
  if [ -n "$SO" ]; then case ",$SO," in *",$e,"*) ;; *) return 1 ;; esac; fi
  case ",$PULAR," in *",$e,"*) return 1 ;; esac
  return 0
}
T_ALL=$SECONDS
run() {  # run "descrição" comando...
  local desc="$1"; shift
  echo; echo "==================== $desc ===================="; echo "+ $*"
  [ "$DRY" -eq 1 ] && return 0
  local t0=$SECONDS
  "$@"
  echo "---- concluído: $desc em $(( (SECONDS - t0) / 60 )) min $(( (SECONDS - t0) % 60 )) s"
}
com_log() {  # com_log LOG comando...: roda o comando e grava a saída (stdout + stderr) também em LOG
  local log="$1"; shift
  PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "$log"
}
aviso() { echo; echo "AVISO: $*"; }
tem_mp() { [ "$DRY" -eq 1 ] || "$PYMP" -c "import mediapipe, cv2" 2>/dev/null; }
tem_uta() { ls data/uta_rldd_videos/Fold*/.ok >/dev/null 2>&1; }
tem_dmd() { ls data/dmd/g*/*/s5/*_rgb_face.mp4 >/dev/null 2>&1; }
tem_nity() { ls data/nitymed/DSM_Dataset-HDTV720/*/*/HDTV720/*.mp4 >/dev/null 2>&1; }
tem_features() { [ -f data/features/final/uta/windows_10s.csv ] && [ -f data/features/final/dmd/windows_10s.csv ]; }

# ---------- dados ----------
if quer dados; then
  run "dados: blink features da UTA-RLDD (20 .npy, 40 MB, repositório dos autores)" "$PY" src/baseline/download_data.py
  if [ "$PREP" -eq 1 ]; then
    run "dados: extrai os pacotes do DMD já baixados (tools/download_dmd.py baixa a partir do e-mail)" "$PY" tools/prepare_data.py --dmd
    run "dados: extrai os folds da UTA-RLDD cujos zips estão em ~/Downloads" "$PY" tools/prepare_data.py --uta
  fi
  echo; echo "Vídeos presentes em data/ (obtenção manual: data/README.md):"
  echo "  UTA-RLDD: $(ls data/uta_rldd_videos/Fold*/.ok 2>/dev/null | wc -l | tr -d ' ') de 5 folds"
  echo "  DMD: $(ls data/dmd/g*/*/s5/*_rgb_face.mp4 2>/dev/null | wc -l | tr -d ' ') sessões s5; $(ls data/dmd/g*/*/s[123]/*_rgb_face.mp4 2>/dev/null | wc -l | tr -d ' ') de distração (s1-s3)"
  echo "  NITYMED: $(ls data/nitymed/DSM_Dataset-HDTV720/*/*/HDTV720/*.mp4 2>/dev/null | wc -l | tr -d ' ') vídeos"
  echo "  MRL Eye: $([ -d data/mrl_eye/mrlEyes_2018_01 ] && echo sim || echo NÃO) | CEW: $([ -d data/cew/cew_eyes_24x24 ] && echo sim || echo NÃO)"
fi

# ---------- sinais (.venv-mp) ----------
if quer sinais; then
  if ! tem_mp; then
    aviso ".venv-mp ausente (uv venv --python 3.11 .venv-mp && uv pip install --python .venv-mp/bin/python -r requirements-mediapipe.txt): etapa 'sinais' pulada."
  else
    if tem_dmd; then run "sinais: DMD (s5 + sessões de distração s1-s3)" "$PYMP" src/features/extract_signals.py --dataset dmd --workers 6
    else aviso "DMD ausente em data/dmd/: sinais do DMD pulados"; fi
    if tem_nity; then run "sinais: NITYMED (126 vídeos)" "$PYMP" src/features/extract_signals.py --dataset nitymed --workers 6
    else aviso "NITYMED ausente em data/nitymed/: sinais pulados"; fi
    if tem_uta; then run "sinais: UTA-RLDD (folds com .ok; incremental)" "$PYMP" src/features/extract_signals.py --dataset uta --workers 6
    else aviso "UTA-RLDD sem folds prontos em data/uta_rldd_videos/: sinais pulados"; fi
    run "sinais: alinhamento anotação DMD x vídeo (.venv)" "$PY" src/features/check_dmd_alignment.py
    run "sinais: resumo por base (results/sinais_resumo.md)" "$PY" src/features/summarize_signals.py
  fi
fi

# ---------- características ----------
if quer caracteristicas; then
  if [ "$DRY" -eq 0 ] && [ ! -f data/signals/uta/index.csv ]; then
    aviso "data/signals/ vazio: rode a etapa 'sinais' antes. Etapa 'caracteristicas' pulada."
  else
    run "características: piscadas, janelas 10 s (5/15 s), sequências, mapeamento oficial" bash src/features/run_features.sh
  fi
fi

# ---------- Exp. 1 e 2 (blink features oficiais da UTA-RLDD) ----------
if quer exp1; then
  if [ "$SMOKE" -eq 1 ]; then
    run "exp1 (SMOKE: 2 épocas, sem salvar modelos, saída *_smoke)" "$PY" src/experiments/exp1_intra.py --epochs 2 --tag _smoke --no-save
  else
    run "exp1: intra UTA-RLDD (5 folds; 3c + B1/B2/B3; LogReg/SVM/RF/LSTM/CNN1D) + TFLite do temporal" "$PY" src/experiments/exp1_intra.py
    run "exp1: sensibilidade a 15 épocas (só temporais)" "$PY" src/experiments/exp1_intra.py --epochs 15 --tag _e15 --no-save
    run "exp1: reescreve results/exp1_intra.md com a sensibilidade" "$PY" src/experiments/exp1_intra.py --regen-md
  fi
fi
if quer exp2; then
  if [ "$SMOKE" -eq 1 ]; then echo "(smoke) exp2 pulado: sobrescreveria results/exp2_leakage.*"
  else run "exp2: vazamento de identidade (janela / vídeo / sujeito; seeds 42-44)" "$PY" src/experiments/exp2_leakage.py; fi
fi

# ---------- Exp. 3 e Exp. 4 desktop (Pipeline B) ----------
if quer exp3; then
  if [ "$DRY" -eq 0 ] && { [ ! -d data/mrl_eye/mrlEyes_2018_01 ] || [ ! -d data/cew/cew_eyes_24x24 ]; }; then
    aviso "MRL Eye e/ou CEW ausentes (data/README.md): etapa 'exp3' pulada."
  else
    # Os scripts do Pipeline B só imprimem: o log de cada um vai para results/logs/ por com_log (tee).
    run "exp3: cache 48x48 (MRL Eye + CEW) e estatísticas" com_log results/logs/exp3_cache_build.log "$PY" src/pipeline_b/datasets.py
    if [ "$SMOKE" -eq 1 ]; then
      run "exp3 (SMOKE: 2 épocas, 3 mil imagens, saída *_smoke)" \
          com_log results/logs/exp3_eye_state_train_smoke.log "$PY" src/pipeline_b/train_eye_state.py --smoke
    else
      run "exp3: EyeNet, protocolos A (MRL), B (MRL->CEW), C (CEW->MRL), D (CEW)" \
          com_log results/logs/exp3_eye_state_train.log "$PY" src/pipeline_b/train_eye_state.py --epochs 12 --max-train 25000 --threads 8
      if [ "$OPTIONAL" -eq 1 ]; then
        run "exp3 (opcional): MobileNetV3-Small no fold 1" com_log results/logs/exp3_mobilenetv3.log "$PY" src/pipeline_b/train_mobilenet.py
        run "exp3 (opcional): ablação do momento da BatchNorm" \
            com_log results/logs/exp3_bn_momentum_ablation.log "$PY" src/pipeline_b/ablation_bn_momentum.py
      fi
      run "exp4 desktop: TFLite fp32/int8 do EyeNet + avaliação + latência" com_log results/logs/exp4_export_tflite.log "$PY" src/pipeline_b/export_tflite.py
      run "exp3: relatório results/exp3_eye_state.md" com_log results/logs/exp3_relatorio.log "$PY" src/pipeline_b/report_exp3.py
    fi
  fi
fi

# ---------- Exp. 5a, 5b, 6 (vídeos: DMD, UTA-RLDD, NITYMED) ----------
if [ "$SMOKE" -eq 1 ]; then
  echo; echo "(smoke) exp5a, exp5b, exp6, figuras e documentos pulados (usariam os resultados reais)"
else
  if quer exp5a; then
    if [ "$DRY" -eq 0 ] && ! tem_features; then aviso "características finais ausentes: etapa 'exp5a' pulada."
    elif ! tem_mp; then aviso ".venv-mp ausente (dlib): etapa 'exp5a' pulada."
    else run "exp5a: nosso extrator x características oficiais (5 folds)" bash src/experiments/run_exp5a.sh; fi
  fi
  if quer exp5b; then
    if [ "$DRY" -eq 0 ] && ! tem_features; then aviso "características finais ausentes: etapa 'exp5b' pulada."
    else
      run "exp5b: transferência DMD (atuado) -> UTA-RLDD (real), PP1" bash src/experiments/run_exp5b.sh
      run "exp5b: robustez (results/exp5b_robustez.{csv,md})" "$PY" src/experiments/exp5b_robustez.py
    fi
  fi
  if quer exp6; then
    if [ "$DRY" -eq 0 ] && ! tem_features; then aviso "características finais ausentes: etapa 'exp6' pulada."
    elif ! tem_mp; then aviso ".venv-mp ausente (luminância): etapa 'exp6' pulada."
    else run "exp6: noite (NITYMED), PP3" bash src/experiments/run_exp6.sh; fi
  fi

  # ---------- figuras ----------
  if quer figuras; then
    if [ -f data/cache/mrl_48.npz ] || [ "$DRY" -eq 1 ]; then
      run "figuras: amostras do MRL Eye (recortes de olho; o CEW não pode ser redistribuído)" "$PY" src/pipeline_b/make_figures.py
    fi
    run "figuras: resultados (results/figures/fig_exp*.pdf|png)" "$PY" src/experiments/make_figures.py
  fi

  # ---------- números, Exp. 4 no celular, artigo, pôster ----------
  if quer documentos; then
    run "documentos: celular (results/phone) -> números -> artigo -> pôster" bash tools/atualizar_celular.sh ${DOC_ARGS[@]+"${DOC_ARGS[@]}"}
  fi
fi

echo; echo "TOTAL: $(( (SECONDS - T_ALL) / 60 )) min $(( (SECONDS - T_ALL) % 60 )) s"
echo "Resultados: results/*.md | números: results/numeros_resumo.md | figuras: results/figures/ | artigo: paper/artigo.pdf | pôster: poster/poster_final.pdf"
