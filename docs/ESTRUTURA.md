# Estrutura de pastas

Levantamento de 26/10/2026 (noite), depois da limpeza do dia. Os tamanhos vêm de `du -sh`.
Na coluna "Git", **sim** quer dizer versionado (ou versionado no próximo commit) e **não** quer dizer
ignorado pelo `.gitignore`. O repositório `github.com/BrunoRodolfoo/deteccao-sonolencia` é **público**:
dados das bases, rostos e chaves privadas nunca entram no Git.

## 1. Pasta raiz local (`files (3)/`, na Mesa)

```
files (3)/
├── LEIA-ME.md                  mapa das entregas (artigo, pôster, Entrega 02, app, guia de estudo)
└── tcc/
    ├── deteccao-sonolencia/    REPOSITÓRIO do projeto (seção 2), cerca de 146 GB em disco, quase tudo em data/
    ├── guia_estudo/            guia de estudo (20 páginas), resumo curto para a equipe (4 páginas)
    │                           e os geradores em fonte/, que leem os números conferidos deste repositório
    └── material_disciplina/    material do professor (6,0 MB): TemplatePoster.pptx (usado por
                                poster/build_poster.py), template do pôster em PDF e slides
```

## 2. Repositório (`tcc/deteccao-sonolencia/`)

| Pasta | Tamanho | Git | Papel |
|---|---:|---|---|
| `src/` | 1,5 MB | sim | código dos experimentos (detalhe abaixo) |
| `tools/` | 0,2 MB | sim | obtenção das bases, dependências do app, importação do benchmark do celular, validação do app |
| `app/pwa/` | 0,9 MB | sim | app de demonstração (PWA 2.1), a versão usada na banca; ver `app/pwa/README.md` |
| `app/pwa/vendor/` | cerca de 36 MB | não | dependências locais do app (MediaPipe com WASM, `face_landmarker.task`, TensorFlow.js e tfjs-tflite; 13 arquivos). Com ela, o app funciona sem internet. Baixada e conferida por `bash tools/baixar_dependencias_app.sh` |
| `app/android/` | 4,3 MB | parcial | app Android nativo: mantém a lógica 1.0 e nunca foi compilado. Ignorados: `*.tflite` e `face_landmarker.task` em `assets/` (baixados por `app/android/tools/fetch_assets.sh`) e as pastas de build. `app/android/tools/import_phone_benchmark.py` continua, porque o importador de `tools/` o usa |
| `app/.pwa-cert/` | 8 KB | não | certificado autoassinado e chave privada da PWA (gerados por `serve_https.py`); mantidos para o iPhone não precisar confiar num certificado novo |
| `models/` | 162 MB | parcial | ver a seção 2.3 |
| `results/` | 50 MB | sim | fonte da verdade dos números: `.md` e `.csv` de cada experimento, `figures/`, `logs/`, `phone/` |
| `paper/` | 0,7 MB | sim (sem `.log` e `.blg`) | artigo IEEE (`artigo.tex`, `artigo.pdf`, `refs.bib`, `numeros.tex` gerado) e `figures/` |
| `poster/` | 4,4 MB | sim | `build_poster.py` gera `poster_final.{pptx,html,pdf}`; `assets/`; `numeros.json` |
| `docs/` | 0,9 MB (o PDF do relatório tem 0,8 MB) | sim | relatório da Entrega 02 (`.md` e `.pdf`), guia da equipe, troca de bases, esta estrutura |
| `data/` | 142 GB | não (só `data/README.md`) | bases, sinais e características (seção 2.2). Caro de refazer: não apagar |
| `.venv/` | 2,8 GB | não | Python 3.11: TensorFlow 2.21, sklearn, treino e figuras (`requirements.txt`, `requirements-lock.txt`) |
| `.venv-mp/` | 730 MB | não | MediaPipe 0.10.21 + OpenCV, só para extrair sinais de vídeo e rodar a demo da webcam (`requirements-mediapipe*.txt`) |
| `.git/` | 24 MB | não se aplica | histórico; último push em 26/08 (HEAD `d8154ae`); o trabalho novo ainda está sem commit |

Arquivos soltos na raiz do repositório: `README.md`, `run_all.sh` (pipeline completo; `--lista` mostra as
11 etapas), `requirements.txt`, `requirements-lock.txt`, `requirements-mediapipe.txt`,
`requirements-mediapipe-lock.txt`, `LICENSE` e `.gitignore`.

### 2.1 Código e resultados

```
deteccao-sonolencia/
├── src/
│   ├── baseline/          baseline de agosto (RF/SVM/LogReg nos .npy oficiais da UTA-RLDD)
│   ├── experiments/       Exp. 1, 2, 5a, 5b e 6 (exp*.py, run_exp*.sh), exp5b_robustez.py (controles da
│   │                      queda do Exp. 5b), make_figures.py, make_paper_numbers.py (gera paper/numeros.tex
│   │                      e poster/numeros.json) e conferir_numeros.py
│   ├── features/          sinais por quadro (extract_signals.py, MediaPipe), piscadas, janelas de 10 s,
│   │                      sequências de 30 piscadas, rótulos do DMD, relatório do Exp. 5, test_features.py
│   ├── pipeline_b/        Exp. 3 (EyeNet, MRL Eye <-> CEW), exportação TFLite do Exp. 4 e a figura de
│   │                      amostras do MRL (make_figures.py)
│   └── realtime/          demo_webcam.py (roda no .venv-mp) e métricas
├── tools/
│   ├── prepare_data.py, watch_downloads.sh, download_dmd.py     obtenção das bases
│   ├── baixar_dependencias_app.sh                               app/pwa/vendor/ (app sem internet)
│   ├── import_phone_benchmark.py, atualizar_celular.sh          celular -> figura, números, artigo e pôster
│   └── app_validacao/     validação do app com vídeos reais (câmera falsa do Chrome e motor offline no DMD)
├── app/
│   ├── pwa/               index.html, app.js, engine.js, sw.js, models/ (TFLite + JSON), serve_https.py,
│   │                      test_engine.mjs, README.md e vendor/ (fora do Git)
│   └── android/           projeto Gradle (Kotlin), lógica 1.0, não compilado
├── models/                (seção 2.3)
├── results/
│   ├── exp1_* a exp6_*, app_validacao*, sinais_*   relatórios .md e tabelas .csv finais
│   │                      (inclui exp5b_robustez.{csv,md}, de 26/10)
│   ├── figures/           figuras de resultados (.pdf/.png), inclusive amostras_mrl (só recortes do MRL Eye)
│   ├── phone/             benchmarks do celular (JSON e CSV por quadro, só números) + aparelhos.json
│   └── logs/              logs das execuções citadas nos relatórios e da última execução de cada etapa
├── paper/
│   ├── artigo.tex, artigo.pdf, refs.bib, numeros.tex (gerado)
│   ├── figures/           fig_pipeline.tex/.pdf (TikZ), as figuras dos Exp. 4, 5b e 6 e render.py
│   └── dataset_facts.md, related_work_notes.md, fontes_brasil.md, REVISAO_SIMULADA.md, README.md
├── poster/                build_poster.py, make_assets.py, layout.py, assets/, poster_final.*
└── docs/                  ENTREGA02_relatorio.md e .pdf, GUIA_EQUIPE.md, TROCA_DATASETS.md, ESTRUTURA.md
```

### 2.2 Dados (`data/`, fora do Git)

| Pasta | Tamanho | Conteúdo |
|---|---:|---|
| `uta_rldd_videos/` | 112 GB | vídeos da UTA-RLDD, 5 folds (de 20 a 25 GB cada) |
| `dmd/` | 27 GB | DMD: 14 sujeitos, sessões s1, s2 e s3 (distração) e s5 (sonolência), READMEs oficiais |
| `nitymed/` | 1,3 GB | NITYMED 720p (126 vídeos extraídos) + `nitymed_720p.zip` (lido por `exp6_noite.py` na ficha de integridade) + `index.csv` |
| `signals/` | 768 MB | sinais por quadro (EAR, MAR, pose) de `uta/`, `dmd/`, `nitymed/`; `app_headless/` e `app_tasks/` são da validação do app |
| `mrl_eye/` | 460 MB | MRL Eye extraído (84.898 imagens) + `index.csv` |
| `cache/` | 128 MB | caches do Exp. 3 (`*_48.npz`) e o modelo `face_landmarker` do MediaPipe |
| `features/final/` | 64 MB | piscadas, janelas, sequências, luminância e vídeos do Exp. 6 (rodada final) |
| `uta_rldd_blinks/` | 40 MB | `.npy` oficiais (sequências de 30 piscadas) dos Exp. 1, 2 e 5a |
| `cew/` | 32 MB | CEW extraído: recortes de olho 24×24 e faces 100×100 (só pesquisa; nunca vai para figuras) |

### 2.3 Modelos (`models/`)

O que cada grupo de arquivos faz está em `docs/ENTREGA02_relatorio.md`, seção "Checkpoints".

| Item | Tamanho | Git | Observação |
|---|---:|---|---|
| `*.pkl` na raiz (RF, SVM, LogReg, 5 folds) | ~25 MB | sim (desde agosto) | baseline da Entrega 01 |
| `*.tflite`, `*.json` na raiz | 0,4 MB | sim | modelos embarcados do Exp. 4 (cópias idênticas em `app/pwa/models/` e nos assets do Android) |
| `exp1/` | 62 MB | não | 100 modelos do Exp. 1, refeitos em ~10 min por `exp1_intra.py` |
| `exp3/` | 20 MB | sim (`eyenet_*`), não (`mobilenetv3_*`) | EyeNet por fold e o modelo final |
| `exp5/` | 5,1 MB | sim | modelos finais do Exp. 5b (`exp5b_final_*.joblib`), usados pelo Exp. 6 |
| `exp6/` | 51 MB | não | RF do Exp. 6, refeitos em ~1 min por `run_exp6.sh` |

## 3. Limpezas

Tudo o que sai vai para a Lixeira do Mac, numa pasta datada (com um `LEIA-ME.txt` que lista cada item, o
caminho original e o motivo). Para recuperar algo, basta arrastar de volta.

### 26/10/2026

Pasta da Lixeira: `TCC_limpeza_2026-10-26`.

| Saiu | Motivo |
|---|---|
| `results/figures/amostras_mrl_cew.{png,pdf}` | tinha recortes do CEW, cuja licença proíbe redistribuição (o repositório é público); substituída por `results/figures/amostras_mrl.{png,pdf}`, só com o MRL Eye |
| `paper/figures/fig_desenho_v2.{tex,pdf,png}` | desenho experimental feito para os slides, que já tinham sido descartados; artigo, pôster e guia não usam |
| `paper/figures/fig_blink_features.{tex,pdf,png}` e `paper/figures/fig_leakage_protocols.{tex,pdf,png}` | figuras de reserva fora da versão final do artigo; nada no projeto as usa |
| `paper/figures/preamble.tex` | só os três diagramas de reserva acima o incluíam; `fig_pipeline.tex` tem preâmbulo próprio |
| `paper/figures/fig_pipeline.png`, `fig_exp5b_transferencia.png`, `fig_exp6_noite.png` | cópias PNG sem uso: o artigo inclui os PDFs, e os PNG atuais ficam em `results/figures/` |
| `src/pipeline_b/run_all_eye.sh` | ninguém chamava; o `run_all.sh` (etapa exp3) faz o mesmo e grava os mesmos logs em `results/logs/` |
| `poster/assets/fig_poster_disparo.png` | gráfico de alarme em acordados que saiu do pôster; `make_assets.py` não o gera mais |
| `poster/__pycache__/` (duas vezes) | cache do Python criado ao gerar e ao testar o pôster; a geração normal não cria mais |

### 06/10/2026

Pasta da Lixeira: `TCC_limpeza_2026-10-06`.

| Saiu | Motivo |
|---|---|
| Rodada preliminar de 27/09 (janelas de 30 s, só o Fold 1): `results/preliminar/`, `src/experiments/preliminar/`, `src/features/preliminar/`, `models/exp5/preliminar/`, `models/exp6/preliminar/`, `data/features/{uta,dmd,nitymed}/` e os logs dela | substituída pela rodada final; nada dela é citado. Os relatórios finais dizem que ela foi descartada |
| 52 logs de execuções repetidas, testes antigos do app e tarefas operacionais (downloads, servidor) | ficaram os logs citados nos relatórios e a última execução de cada etapa |
| `src/experiments/run_all_blink.sh`, `src/features/run_exp5.sh` | atalhos antigos; o `run_all.sh` faz o mesmo |
| Cópias de figuras não usadas em `paper/figures/` e `poster/assets/` | os originais ficam em `results/figures/` |
| Zips do MRL e do CEW e os rostos brutos do CEW | o conteúdo extraído fica; nenhum código lia os rostos brutos |
| `results/figures/exp5_dmd_verificacao_visual.jpg` | figura interna com rostos do DMD, que não pode ser publicada |
| `paper/artigo.log`, `paper/artigo.blg`, `.DS_Store`, caches de Python | restos de compilação, recriados sozinhos |
| Fora do repositório: histórico da Entrega 01, pasta do pedido do NTHU-DDD, arquivo de contexto antigo e sobras do TCC em Downloads | pedido da equipe: deixar só o necessário para as entregas |

`results/sinais_resumo.md` (e os CSVs dele), que ainda descrevia só a primeira extração, foi regenerado com todas
as sessões e folds.

### 28/09/2026

Saíram 7 pastas `__pycache__/`, `.pytest_cache/` e `data/signals/_teste/` (saída do teste rápido de extração).

## 4. O que o próximo commit leva

Em 26/10/2026 (noite), `git status --short -uall` lista 363 caminhos (5 modificados e 358 novos), com
79,4 MB, já contando o PDF do relatório:

| Pasta | Tamanho |
|---|---:|
| `results/` | 52,2 MB |
| `models/` | 19,2 MB |
| `poster/` | 4,2 MB |
| `src/` | 1,1 MB |
| `app/` | 0,9 MB |
| `docs/` | 0,9 MB |
| `paper/` | 0,7 MB |
| `tools/` e raiz | 0,2 MB |

Depois do commit, a árvore versionada fica com cerca de 105 MB em 386 arquivos. O maior arquivo é
`results/exp5b_predicoes.csv` (32 MB), abaixo do aviso de 50 MB do GitHub. A lista não inclui `data/`
(só `data/README.md`), `.venv*`, caches, certificados, chaves, `app/pwa/vendor/`, `models/exp1/` nem `models/exp6/`.
Esses números mudam a cada arquivo novo; refaça a conta antes do commit (comando abaixo).

Conferências feitas na lista:
- Nenhum e-mail pessoal, token ou URL assinada nos arquivos que seriam versionados. Os únicos e-mails
  são os institucionais dos autores no artigo e os contatos públicos das bases em `paper/dataset_facts.md`.
- Nenhuma imagem com rosto. As figuras são de gráficos; `results/figures/amostras_mrl.pdf` mostra só recortes
  de olho do MRL Eye.

Para refazer a conferência antes do commit:
`git status --short -uall | awk '{print $2}' | xargs -I{} du -k "{}" | sort -rn | head`
(ver também o topo do `.gitignore`).
