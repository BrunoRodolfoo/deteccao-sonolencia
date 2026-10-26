# `src/features` — dos vídeos às características (Exp. 5a, 5b, 6 e validação do app)

Duas etapas, em ambientes diferentes:

1. **Sinais por quadro** (`extract_signals.py`, **`.venv-mp`**): cada vídeo vira séries temporais (EAR, MAR, pose
   aproximada da cabeça) com o **MediaPipe Face Mesh** 0.10.21. Bases: DMD (s5 e sessões de distração s1-s3),
   NITYMED e UTA-RLDD completa (5 folds, 182 arquivos = 180 vídeos lógicos).
2. **Características** (`run_features.sh`, **`.venv`**): piscadas, janelas de **10 s** (passo 5 s; 5 s e 15 s como
   sensibilidade), rótulos do DMD, sequências de 30 piscadas no formato de Ghoddoosian et al. (2019) e o
   mapeamento sujeito oficial ↔ pasta da UTA-RLDD. Usadas no treino atuado (DMD) → teste real (UTA-RLDD), PP1,
   e na comparação noite × dia, PP3.

Ordem completa (com os experimentos): `bash run_all.sh --so sinais,caracteristicas` (ver `README.md`, seção 6).

## Ambiente

Roda **somente no `.venv-mp`** (mediapipe 0.10.21, opencv 4.11, numpy 1.26). O mediapipe 1.x aborta
neste Mac (serviço Metal do detector de rosto) e o 0.10.21 exige protobuf < 5, incompatível com o
TensorFlow 2.21 do `.venv`. Os scripts de verificação/resumo (que usam matplotlib/pandas) rodam no `.venv`.

```bash
# na raiz do repositório:
.venv-mp/bin/python src/features/extract_signals.py --dataset dmd      --workers 6   # s5 2,6 min + distração 24 min
.venv-mp/bin/python src/features/extract_signals.py --dataset nitymed  --workers 6   # 2,2 min
.venv-mp/bin/python src/features/extract_signals.py --dataset uta      --workers 6   # Fold 1 21 min; folds 2-5 96 min
# teste rápido (300 quadros de 1 vídeo; saída separada em data/signals/_teste/<base>/):
.venv-mp/bin/python src/features/extract_signals.py --dataset uta --limit 300 --max-videos 1 --workers 1
.venv/bin/python src/features/check_dmd_alignment.py   # alinhamento anotação DMD x vídeo + figura
.venv/bin/python src/features/summarize_signals.py     # results/sinais_resumo.md e CSVs
bash src/features/run_features.sh                      # etapa 2 (usa o .venv sozinho), ~3 min
```

Opções: `--limit N` (só N quadros por vídeo, grava em `data/signals/_teste/`), `--max-videos K`,
`--only TEXTO` (filtra ids), `--force` (reprocessa). **Incremental**: um `.npz` que abre, tem todas as
chaves e `meta_complete=True` é pulado; a escrita é atômica (`.tmp.npz` + rename), então um processo
interrompido não deixa arquivo "válido" pela metade. O `index.csv` é reconstruído a cada execução a
partir de todos os `.npz` da pasta. **UTA-RLDD**: os folds são descobertos por `data/uta_rldd_videos/Fold*/.ok`;
ao chegar um fold novo, basta rodar de novo `--dataset uta` (só os vídeos novos são processados; foi assim que os folds 2-5 entraram).

## Saídas

`data/signals/<base>/<id>.npz` (comprimido) e `data/signals/<base>/index.csv`.

| Base | id | sujeito | rótulo |
|---|---|---|---|
| DMD | s5: `<grupo>_<sujeito>_<timestamp>` (ex.: `gA_1_2019-03-14T14;26;17+01;00`); distração (s1/s2/s3): `<grupo>_<sujeito>_<sN>_<timestamp>` | `<grupo>_<sujeito>` | vazio (rótulos por quadro no JSON OpenLABEL; caminho em `extra_json.annotation`; `extra_json.sessao`, `extra_json.pacote`) |
| NITYMED | `<Categoria>_<Gênero>_<arquivo>` | o próprio vídeo (não há ID de motorista) | `Microsleep` ou `Yawning` (pasta) |
| UTA-RLDD | `F<n>_<sujeito>_<rótulo>` (ex.: `F1_03_10`); partes de vídeo dividido: `F3_32_10_p1`, `_p2` (e `F5_49_10_p1/_p2`), com `extra_json.video` = vídeo lógico | `F<n>_<sujeito>` (pastas 01..60) | `0`, `5`, `10` |

Arrays por quadro (comprimento = nº de quadros efetivamente decodificados):

| Chave | Tipo | Conteúdo |
|---|---|---|
| `frame` | int32 | índice do quadro 0..n-1 (ordem de decodificação) |
| `t_s` | float64 | `frame / fps` |
| `t_pts_s` | float64 | carimbo do contêiner (`CAP_PROP_POS_MSEC`/1000) — para vídeos de fps variável (celulares da UTA) |
| `face` | bool | Face Mesh devolveu landmarks |
| `ear_l`, `ear_r`, `ear` | float64 | EAR de cada olho e média; NaN sem rosto |
| `mar` | float64 | `|13-14| / |61-291|`; NaN sem rosto |
| `roll_deg` | float64 | ângulo (graus) da reta canto externo 33 → 263 (inclinação lateral) |
| `yaw_proxy` | float64 | projeção (ponta do nariz 1 − ponto médio dos cantos externos) no eixo olho-olho ÷ distância interocular; ≈0 de frente |
| `pitch_proxy` | float64 | distância olhos→nariz ÷ olhos→queixo (152), no eixo perpendicular; cresce ao abaixar a cabeça |
| `lm` | float32 (n×21×2) | pixels (no quadro processado) dos pontos `lm_idx` = 12 dos olhos, 4 da boca, 1, 152, 10, 234, 454 — permite recalcular EAR/MAR sem rodar o Face Mesh de novo |

Metadados `meta_*`: `id, dataset, subject, session, label, path, fps, n_frames_meta` (do contêiner),
`n_frames` (lidos), `width_orig, height_orig` (já na orientação de exibição), `rotation_meta`,
`width_proc, height_proc, scale, face_rate, proc_fps, proc_seconds, extra_json, complete`.
`extra_json` traz, no DMD, idade, gênero, experiência, frequência de direção, **óculos**, clima, `setup`
e os campos de sincronização da anotação; na NITYMED, categoria e gênero; na UTA, fold, pasta e arquivo.

## Decisões (e por quê)

- **EAR/MAR em pixels**: landmarks normalizados × largura/altura do quadro processado, para que a razão
  não seja distorcida pela proporção 16:9. Índices idênticos a `src/realtime/metrics.py` (mesmo pipeline do app).
- **Face Mesh em modo vídeo** (`static_image_mode=False`, rastreamento entre quadros), `max_num_faces=1`,
  `refine_landmarks=False`, confianças 0,5 — igual ao app. O rastreamento faz a detecção por quadro depender
  do quadro anterior; é o comportamento real do celular.
- **Redução de resolução**: o quadro é reduzido (INTER_AREA) para que o **lado maior** fique ≤ 1280 px.
  A especificação pedia "largura > 1280 → 1280"; usamos o lado maior porque há vídeos em retrato
  (1080×1920, 1440×2560) em que a largura ≤ 1280 mas a altura não. EAR e MAR são razões e não mudam com a escala.
- **Rotação**: nesta build do OpenCV 4.11 `CAP_PROP_ORIENTATION_AUTO` vinha **desligado (0)** — os quadros
  saíam deitados nos .MOV/.mp4 com metadado 90°/270°. O script liga explicitamente
  (`cap.set(CAP_PROP_ORIENTATION_AUTO, 1)`); verificado visualmente nos 36 vídeos do Fold 1 (todos em pé).
  `width_orig/height_orig` já saem na orientação corrigida.
- **Quadros sem rosto** ficam com NaN em todas as medidas (nenhuma interpolação aqui; isso é decisão da etapa de piscadas).
- **Processos**: `multiprocessing` (spawn), 1 vídeo por processo, `maxtasksperchild=1`, `cv2.setNumThreads(1)`,
  no máximo 6 processos (Mac de 10 núcleos).

## Alinhamento anotação DMD ↔ `*_rgb_face.mp4` (verificado)

`check_dmd_alignment.py` → `results/sinais_alinhamento_dmd.csv`, `results/figures/diag_sinal_dmd.png`,
log em `results/logs/check_dmd_alignment.log`.

- Nas 16 sessões: nº de quadros do vídeo = `streams.face_camera.total_frames`; o maior `frame_end` das ações
  = n−1; `streams.face_camera.sync.frame_shift = 0` (as câmeras de corpo e mãos têm deslocamentos de 1 a 74
  quadros — a câmera do rosto é a **referência** do mosaico usado na anotação).
- O deslocamento que maximiza a correlação entre o indicador `eyes_state/close` e −EAR ficou entre −3 e +2
  quadros (≤ 0,1 s) em todas as sessões, com ganho de correlação desprezível em relação ao lag 0 — compatível
  com a convenção de bordas (os estados `closing`/`opening` são rotulados à parte), não com um deslocamento.
  Inspeção visual (quadros de gA_1 no início, meio e fim de 4 intervalos `close`, e 20 quadros antes/depois)
  confirma olhos fechados dentro e abertos fora. **Conclusão: índice da anotação = índice do quadro do
  face.mp4, sem correção.**
- `frame_intervals` do arquivo às vezes termina em n (um além do último quadro) em 4 sessões; as ações nunca passam de n−1.

## Limitações conhecidas

- NITYMED: câmera longe (rosto pequeno), perfil e mão/bocejo muito aberto derrubam a detecção
  (inspecionado nos 4 vídeos de menor taxa). A perda **não é aleatória**: tende a ocorrer justamente nos
  eventos (bocejo), o que enviesa o MAR — tratar na etapa de eventos.
- No DMD, a mediana do EAR **dentro** de `eyes_state/close` ficou entre 0,118 e 0,247 conforme a sessão
  (fora: 0,216–0,298; `results/sinais_alinhamento_dmd.csv`) — o EAR do Face Mesh com olho fechado não vai
  a ~0 e a separação varia muito por sujeito (pior: gE_29, com óculos, 0,247 × 0,282). Limiares fixos de EAR
  são inadequados; normalização por sujeito é necessária.
- `yaw_proxy`/`pitch_proxy` são proxies geométricos 2D, não ângulos calibrados.
- UTA-RLDD (conferido no Fold 1): o fps **muda entre os vídeos do mesmo sujeito** em F1_04 (25/30/25), F1_06 (24,96/29,91/23,82)
  e F1_11 (29,94/15,03/17,16) — ver `results/sinais_resumo.md`. Nesses sujeitos o z-score por sujeito NÃO cancela
  o fator do fps em Duration/Frequency medidos em quadros; usar as versões em segundos/Hz.

## Execuções e onde estão os números

- Primeira execução (27/09/2026, manhã): DMD s5, NITYMED e UTA-RLDD Fold 1 → `results/sinais_alinhamento_dmd.csv` e
  `results/figures/diag_sinal_dmd.png`. O resumo `results/sinais_resumo.md` (+ `sinais_resumo.csv`, `sinais_resumo_videos.csv`,
  `sinais_resumo_estratos.csv`) foi regenerado em 06/10/2026 e cobre todas as sessões e folds.
- Execução final (27/09/2026, tarde): sessões de distração do DMD (65 sessões no total; 24,3 min) e UTA-RLDD folds 2-5
  (182 arquivos no total; 95,5 min), logs `results/logs/extract_signals_{dmd,dmd_distracao,nitymed,uta,uta_full}.log`.
  Contagens atualizadas: `results/exp5_caracteristicas.md` (seção 2) e `results/exp6_noite.md` (detecção de rosto e
  qualidade do EAR nas três bases).
- Tempo de parede com 6 processos: DMD s5 2,6 min, NITYMED 2,2 min, UTA Fold 1 20,8 min.
- A figura `results/figures/exp5_dmd_verificacao_visual.jpg` (conferência visual do alinhamento, com **rostos** do DMD)
  só existe se `verificar_alinhamento_dmd.py` for rodado de novo; fica no `.gitignore` e não deve ser publicada
  (a cópia antiga foi apagada em 06/10/2026).

---

# Exp. 5 — piscadas, janelas, rótulos e sequências: RODADA FINAL (a partir de `data/signals/`)

Roda no **`.venv`** (não precisa de MediaPipe). Tudo de uma vez (~3 min neste Mac), com logs em `results/logs/exp5_final_*.log`:

```bash
bash src/features/run_features.sh            # piscadas incrementais (hash dos parâmetros)
FORCE=1 bash src/features/run_features.sh    # recalcula tudo
```

Saídas em **`data/features/final/<base>/`** (a rodada preliminar — janelas de 30 s, só Fold 1 e só s5 — foi descartada; seus arquivos foram removidos em 06/10/2026).
Relatório: **`results/exp5_caracteristicas.md`**.

| Script | Faz | Saídas |
|---|---|---|
| `blinks.py` | limiar por unidade de referência (0,75 × mediana do EAR do trecho de referência), piscadas e fechamentos > 1,5 s, 4 características de Ghoddoosian (quadros) + versões em s/Hz + variante estendida | `data/features/final/<base>/blinks/<id>.csv`, `blinks_index.csv` (uma linha por arquivo: `video` lógico, `parte`, `sessao`, `tipo_ref`, limiar) |
| `dmd_labels.py` | s5: vetores por quadro, alinhamento, validação do detector contra a anotação (limiar da s3), bocejos; distração (s1/s2/s3): alinhamento por contagem de quadros e correlação com olhar × pose e fala × boca | `data/features/final/dmd/rotulos_quadros/`, `results/exp5_dmd_alinhamento*.csv`, `exp5_dmd_validacao_detector.csv`, `exp5_dmd_sensibilidade_limiar.csv`, `exp5_dmd_bocejo*.csv` |
| `windows.py` | janelas `--janela` (10 s) / `--passo` (5 s), `uso` de cada janela, rótulos, 11 características brutas + `_c` + `_z` | `data/features/final/<base>/windows_<L>s.csv`, `windows_referencia_<L>s.csv`, `results/exp5_janelas_resumo.csv`, `exp5_dmd_balanco_janelas.csv`, `exp5_dmd_grade_safe.csv` |
| `sequences.py` | sequências de 30 piscadas (passo 2), partes concatenadas com contagem contínua; mapeamento oficial dos 5 folds e comparação com as características oficiais | `data/features/final/uta/sequences.npz` (+ `_primeiro_terco`), `results/exp5_sequencias_resumo.csv`, `results/exp5_uta_*.csv`, `data/features/final/uta/oficial_piscadas_fold<k>.csv` |
| `uta_oficial.py` | engenharia reversa do `.npy` oficial (Frequency → quadro de cada piscada) e mapeamento sujeito-oficial ↔ pasta: escore coerente, checagem de duração, classes `confirmado`/`provavel`/`indeterminado` | usado por `sequences.py` |
| `report_exp5.py` | relatório a partir dos CSVs | `results/exp5_caracteristicas.md`, `results/exp5_contagens_sinais.csv` |
| `test_features.py` | 19 testes de asserts puros (inclui janela de 10 s, janelas ancoradas no safe_drive, rótulos v2, referência separada, partes divididas, escore coerente) | `results/logs/exp5_final_testes.log` |

## Regras da rodada final (detalhes e números em `results/exp5_caracteristicas.md`)

- **Janela 10 s, passo 5 s** nas três bases (5 s e 15 s como sensibilidade). UTA, NITYMED e DMD s5: grade global por
  **arquivo** (partes `_p1`/`_p2` separadas; nenhuma janela atravessa o corte; `video` = vídeo lógico — agrupe métricas por
  vídeo por essa coluna). DMD s1/s2/s3: só janelas inteiramente dentro de UM trecho `driver_actions/safe_drive`, ancoradas
  no início do trecho.
- **`uso` de cada janela**: UTA/NITYMED `exemplo`/`referencia`/`fronteira`; DMD `alerta` (s2), `sonolento` (s5 com fechamento
  anotado ≥ 0,5 s ou bocejo), `s5_sem_sinal`, `alerta_s1`, `referencia`, `fronteira`, `s3_nao_usada`. **`janela_ref` e
  `sobrepoe_ref` nunca entram em treino/teste**; `avaliavel` marca o que entra no principal. Rótulos: UTA `rotulo_kss`,
  `classe`, `B1`, `B2`, `B3`; DMD `y` (principal), `y_sens_s1`, `y_sens_s5_sem_sinal` (sujeito sem s5 — gE_28 — tem `y` NaN).
- **Referência por sujeito, separada da avaliação** (define o limiar de EAR e a normalização): UTA = último terço do vídeo 0
  (os 2/3 iniciais são o alerta avaliado); DMD = janelas safe_drive das sessões s3 (s3 com < 3 janelas → último terço, por
  sessão, das janelas safe_drive da s2, e só as anteriores que não a tocam viram alerta); NITYMED = primeiro terço (desvio).
- **Normalização principal = centragem** `<f>_c` (x − média da referência); `<f>_z` só com ≥ 5 janelas de referência e desvio > 0
  (continua indefinido em várias características). Padronização global: ajustar só no treino.
- **Alinhamento DMD**: deslocamento 0 na s5 e nas sessões de distração.
- **Mapeamento oficial (5 folds)**: pastas ausentes das características oficiais = **F1_12** e **F5_49**; no Fold 3 a ordem do
  arquivo não é crescente e há defeitos no próprio arquivo (4 sujeitos indeterminados). Use os pares `confirmado`.
