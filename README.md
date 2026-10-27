# Detecção de Sonolência em Dispositivos Móveis

**Avaliação da Generalização entre Sonolência Atuada e Real**

Projeto Transformador II (Experiência Criativa), Bacharelado em Ciência da Computação, PUCPR. Turma 8ºA, Equipe 14.
André Luís Oliveira Ferreira da Silva · Bruno Rodolfo da Silveira · Bruno Sandoval Ribeiro · Paulo Henrique Perin.
Orientador: Prof. Júlio César Nievola.

> **Estado em 27/10/2026.** Os seis experimentos terminaram e foram conferidos. Em outubro entraram os controles
> da queda do Exp. 5b (`src/experiments/exp5b_robustez.py`), e o artigo foi reescrito depois de uma banca simulada
> (8 páginas). O app de demonstração (PWA 2.1) foi validado com vídeos reais e agora funciona sem internet.
> - Relatório da Entrega 02: [`docs/ENTREGA02_relatorio.md`](docs/ENTREGA02_relatorio.md) (PDF para o Canvas: `docs/ENTREGA02_relatorio.pdf`).
> - Troca de bases (aprovada pelo orientador): [`docs/TROCA_DATASETS.md`](docs/TROCA_DATASETS.md).
> - Passo a passo da equipe (celular, demonstração, pôster, revisão por pares, GitHub): [`docs/GUIA_EQUIPE.md`](docs/GUIA_EQUIPE.md).
> - Árvore de pastas comentada (tamanhos, o que é versionado, limpezas de 28/09, 06/10 e 26/10): [`docs/ESTRUTURA.md`](docs/ESTRUTURA.md).
>
> **Celular:** até agora, só o app 1.0 foi medido, no iPhone 17 Pro (2 execuções de 27/09; mediana de 17 ms por quadro).
> O app 2.1 ainda vai ser medido no iPhone e no Galaxy S23 Ultra, até 09/11 (`results/exp4_phone_latency.md`).

## 1. Visão geral

Em bases de sonolência **atuada**, a literatura de detecção por vídeo relata acurácias acima de 90%.
Na UTA-RLDD, base pública de sonolência **real** com protocolo por sujeito, os números são bem mais modestos.
O HM-LSTM dos autores acerta 54% por sequência de piscadas e 65,2% por vídeo, e o julgamento humano acerta 57,8%.

O projeto mede de onde vem essa diferença:
- quanto é a passagem do sono encenado para o sono real;
- quanto o protocolo pode inflar os números (vazamento entre treino e teste: janelas do mesmo vídeo nos dois lados);
- quanto é a condição de captura (óculos, noite, carro em movimento);
- quanto custa rodar tudo localmente num celular.

O processamento do vídeo é todo local, sem nuvem (veja a nota sobre telemetria na seção 9). O produto final é um
**app de demonstração** que detecta sinais visíveis de sonolência (olhos, bocejo e cabeça) e explica cada alerta.

## 2. Perguntas de pesquisa e experimentos

| | Pergunta | Experimentos |
|---|---|---|
| **PP1** | Quanto o desempenho cai ao treinar em sonolência atuada e testar em real? | Exp. 5b (DMD atuado → UTA-RLDD real, com controles da normalização); Exp. 2 (inflação por vazamento entre treino e teste); Exp. 1 (teto intra-UTA); Exp. 5a (validação do extrator) |
| **PP2** | Qual o custo acurácia × latência num celular? | Exp. 4 (TFLite fp32/int8; app 1.0 no iPhone 17 Pro; app 2.1 no iPhone e no Galaxy S23 Ultra a medir) |
| **PP3** | A queda se concentra em óculos, iluminação ruim ou noite? | Exp. 3 (estado ocular MRL Eye ↔ CEW por condição); Exp. 6 (noite real, NITYMED, com o DMD em movimento como controle) |

| Exp. | Treino → teste | Unidade | Resultado em |
|---|---|---|---|
| 1 | UTA-RLDD → UTA-RLDD (5 folds oficiais por sujeito, características oficiais) | sequência de 30 piscadas; vídeo | `results/exp1_intra.md` |
| 2 | UTA-RLDD, partição por janela × vídeo × sujeito | idem | `results/exp2_leakage.md` |
| 3 | MRL Eye ↔ CEW (EyeNet, 62.865 parâmetros) | recorte do olho | `results/exp3_eye_state.md` |
| 4 | modelos TFLite no desktop e no celular | quadro | `results/exp4_*.csv`, `results/exp4_phone_latency.md` |
| 5a | características oficiais (dlib) × nosso extrator (MediaPipe), mesmos vídeos | sequência; vídeo | `results/exp5a_validacao.md` |
| 5b | DMD (atuado) → UTA-RLDD (real), janelas de 10 s, e os controles da queda | janela; vídeo | `results/exp5b_transferencia.md`, `results/exp5b_robustez.md` |
| 6 | NITYMED (noite real) × UTA-RLDD × DMD parado e em movimento | vídeo; janela | `results/exp6_noite.md` |

Duas cadeias de processamento:
- **Pipeline A:** MediaPipe Face Mesh/Landmarker → EAR, MAR e pose por quadro → piscadas → características da janela ou da sequência → Random Forest, RL ou LSTM.
- **Pipeline B:** recorte do olho 48×48 → CNN "EyeNet" → aberto ou fechado.

## 3. As cinco bases: papel, obtenção e citação obrigatória

Nenhum dado é versionado (`.gitignore`). Pastas e comandos: [`data/README.md`](data/README.md).

| Base | Papel no trabalho | Como obter | Citação obrigatória (chave em `paper/refs.bib`) | Licença e restrições |
|---|---|---|---|---|
| **UTA-RLDD**: 60 sujeitos × 3 vídeos (KSS 0/5/10), sonolência **real** autorrelatada, celular ou webcam, ~112 GB | teste real (Exp. 1, 2, 5a, 5b, 6); as características oficiais (20 `.npy`, 40 MB) servem aos Exp. 1, 2 e 5a | características: `python src/baseline/download_data.py`. Vídeos: Google Drive oficial ([página](https://sites.google.com/view/utarldd/home)); baixe pelo navegador com conta Google para `~/Downloads` e deixe `bash tools/watch_downloads.sh` extrair (usa `tools/prepare_data.py --uta`). **Aviso:** o Drive aplica cota por rajada a downloads anônimos ("Quota exceeded"). Espere e retome; o script é reinicializável. Não usamos cópias do Kaggle (resultados só por canais oficiais) | Ghoddoosian, Galib e Athitsos, CVPRW 2019 (`ghoddoosian2019`) | uso acadêmico; rostos publicáveis só dos 36 sujeitos que autorizaram. **Nenhum quadro no artigo ou no pôster** |
| **DMD** (Vicomtech): 13 sujeitos com sessão de sonolência s5 (16 sessões), mais as sessões de distração s1, s2 e s3; sonolência **encenada** ("These states were acted"), carro parado na s5 | treino atuado (Exp. 5b); controle "carro em movimento" (Exp. 6, s1); validação offline do app | formulários em [dmd.vicomtech.org](https://dmd.vicomtech.org/): pacote de sonolência e pacote de distração. O e-mail traz links assinados, válidos por 48 h. Salve o e-mail como `.eml` (Outlook: "..." > Baixar) e rode `python tools/download_dmd.py [arquivo.eml]`, que baixa, confere e extrai só o vídeo do rosto e a anotação | Ortega et al., ECCV Workshops 2020 (`ortega2020`) | **CC BY-NC-ND 4.0**: sem quadros no artigo ou no pôster (NoDerivatives e rostos) |
| **NITYMED**: 126 vídeos noturnos baixados (19 de microssono + 107 de bocejo; a ficha oficial diz 130), carros reais em movimento, 25 fps | condição **noite** da PP3 (Exp. 6): captura, qualidade do sinal, taxa de disparo. **Não** é evidência de sono real: a autenticidade dos comportamentos não é documentada (3 bocejos por vídeo sugerem instrução) | Kaggle [`nikospetrellis/nitymed`](https://www.kaggle.com/datasets/nikospetrellis/nitymed) (versão 720p); descompactar em `data/nitymed/` | o dataset (`petrellis2022` ou `petrellis2022kaggle`) **e** ao menos um artigo dos autores (`petrellis2021dsd` ou `petrellis2021vlsisoc`), como exigido na página | cópia do Kaggle: **CC BY-ND 4.0**; sem quadros publicados sem decisão da equipe |
| **MRL Eye**: 84.898 recortes infravermelhos de 37 sujeitos, com metadados de óculos, iluminação, reflexo e sensor | Exp. 3 (treino e teste); EyeNet exportada para o app | download livre em http://mrl.cs.vsb.cz/eyedataset; descompactar `mrlEyes_2018_01.zip` em `data/mrl_eye/mrlEyes_2018_01/` | Fusek, ISVC 2018, LNCS 11241 (`fusek2018`) | uso acadêmico; recortes de olho podem ir para figuras (`results/figures/amostras_mrl.{png,pdf}`) |
| **CEW** (Closed Eyes in the Wild): 2.423 faces RGB, 4.846 olhos | Exp. 3 (teste e treino cruzado) | página do grupo PARNEC/NUAA (`ClosedEyeDatabases.html`); descompactar em `data/cew/` | Song et al., Pattern Recognition 47(9), 2014 (`song2014`) | **só pesquisa, redistribuição proibida**: nenhum recorte do CEW aparece em figura |

O NTHU-DDD, previsto na Entrega 01, **não foi obtido**. O termo de licença exige assinatura de chefia institucional, e o pedido não foi atendido até a data-limite. A troca por DMD + NITYMED, com justificativa e impacto, está em [`docs/TROCA_DATASETS.md`](docs/TROCA_DATASETS.md).

Modelos de terceiros usados nos apps: MediaPipe Face Landmarker (`face_landmarker.task`, Lugaresi et al., 2019, `lugaresi2019`).

## 4. Estrutura do repositório

```
run_all.sh                 reproduz tudo, na ordem final (seção 6)
requirements*.txt          dependências dos DOIS ambientes (seção 5) + congelamentos completos (*-lock.txt)
src/baseline/              Entrega 01: download das características oficiais, folds, classificadores clássicos
src/experiments/           Exp. 1, 2, 5a, 5b (+ exp5b_robustez.py, controles da queda), 6; figuras;
                           números do artigo (make_paper_numbers.py + conferir_numeros.py)
src/pipeline_b/            Exp. 3 (EyeNet MRL <-> CEW), Exp. 4 desktop (TFLite fp32/int8), amostras do MRL
src/features/              vídeo -> sinais por quadro (.venv-mp) -> piscadas, janelas de 10 s, sequências (src/features/README.md)
src/realtime/              demo de webcam (Pipeline A, .venv-mp)
app/pwa/                   APP DE DEMONSTRAÇÃO (PWA 2.1: olhos, bocejo, cabeça, nível 0-100, alarme) + serve_https.py;
                           vendor/ (fora do Git) deixa o app funcionar sem internet
app/android/               app Android nativo (Kotlin), lógica ANTERIOR (1.0), nunca compilado
tools/                     obtenção de dados, dependências do app (baixar_dependencias_app.sh), importação do
                           benchmark do celular, atualizar_celular.sh, validação do app
paper/                     artigo IEEE (artigo.tex, refs.bib, numeros.tex GERADO, figuras)
poster/                    pôster da sessão de 16/11 (build_poster.py no template do professor)
docs/                      relatório da Entrega 02 (.md e .pdf), troca de bases, guia da equipe, estrutura de pastas
results/                   CSV/MD de cada experimento (fonte da verdade), figures/, logs/, phone/ (benchmarks)
models/                    .tflite + contratos .json, EyeNet (.keras), modelos finais do Exp. 5b (.joblib)
data/                      dados locais, NÃO versionados (data/README.md)
```

**Histórico do repositório.** O trabalho foi feito no computador que tem as bases (142 GB, fora do Git). As datas de
cada etapa estão nos logs de `results/logs/` e no cronograma do relatório da Entrega 02 (seção 9). O envio ao GitHub
foi feito em 26/10, em commits separados por assunto. Daqui em diante, cada integrante faz o commit da parte que fizer.

## 5. Instalação: dois ambientes Python

Usamos Python 3.11 (testado em 3.11.16, macOS arm64) e [uv](https://docs.astral.sh/uv/).
A resolução dos quatro arquivos de dependências foi conferida em 28/09 (`uv pip compile`) para macOS 14+ arm64, Linux x86_64 e Windows x86_64; no macOS, a roda do `dlib-bin` exige macOS 14 ou mais novo.
São **dois** ambientes porque as dependências são incompatíveis:
- o mediapipe 1.x **aborta** neste Mac (serviço Metal do detector de rosto);
- o mediapipe 0.10.21, que funciona, exige `protobuf < 5` e `numpy < 2`, e isso quebra o TensorFlow 2.21.

| Ambiente | Para quê | Instalação |
|---|---|---|
| `.venv` (principal) | treino, Exp. 1 a 6 (análises), figuras, números, artigo, pôster | `uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt` |
| `.venv-mp` (vídeo) | extração de sinais dos vídeos (`src/features/extract_signals.py`), **demo de webcam**, detector do dlib (Exp. 5a), luminância (Exp. 6), sinais do app (`tools/app_validacao/extrair_sinais_app.py`) | `uv venv --python 3.11 .venv-mp && uv pip install --python .venv-mp/bin/python -r requirements-mediapipe.txt` |

Para reproduzir exatamente os ambientes usados, instale `requirements-lock.txt` e `requirements-mediapipe-lock.txt`, que são os congelamentos completos de 27/09/2026.

Ferramentas opcionais:
- `node` ≥ 18: testes da PWA e validação do app;
- `openssl`: certificado da PWA (já vem no macOS);
- `curl`: dependências do app (`tools/baixar_dependencias_app.sh`);
- [tectonic](https://tectonic-typesetting.github.io/): artigo;
- Google Chrome: pôster em PDF e testes headless, sempre com `--mute-audio`;
- Android Studio com JDK 17: app nativo.

## 6. Como reproduzir

Tudo, na ordem final (dados → sinais no `.venv-mp` → características → Exp. 1 a 6 → figuras → números → artigo → pôster).
O script chama `.venv/bin/python` e `.venv-mp/bin/python` diretamente, sem precisar ativar os ambientes:

```bash
bash run_all.sh --lista                 # etapas: dados sinais caracteristicas exp1 exp2 exp3 exp5a exp5b exp6 figuras documentos
bash run_all.sh --dry-run               # só mostra os comandos
bash run_all.sh                         # tudo (~4 h com os vídeos em data/; a extração da UTA-RLDD leva ~2 h)
bash run_all.sh --so exp5b,exp6,documentos
bash run_all.sh --desde figuras         # só figuras, números, artigo e pôster
bash run_all.sh --pular sinais,exp5a --sem-poster
```

O pôster precisa do template do professor (`TemplatePoster.pptx`), que não está no repositório. O `poster/build_poster.py`
procura o arquivo em `material_disciplina/`, ao lado da pasta do repositório, ou no caminho da variável `POSTER_TEMPLATE`.
Sem o template, rode com `--sem-poster` (vale para o `run_all.sh` e para o `tools/atualizar_celular.sh`).

Para conferir o repositório em poucos minutos, sem refazer os experimentos, veja a seção "Como verificar em 5 minutos"
de [`docs/ENTREGA02_relatorio.md`](docs/ENTREGA02_relatorio.md).

Cada etapa, isolada. Os tempos são de um Mac M5 com no máximo 6 processos. Os logs vão para `results/logs/`, e a semente é 42 em todos os experimentos.

| Etapa | Ambiente | Comando | Tempo | Saídas principais |
|---|---|---|---|---|
| Características oficiais UTA | .venv | `.venv/bin/python src/baseline/download_data.py` | 1 min | `data/uta_rldd_blinks/*.npy` |
| Sinais por quadro | **.venv-mp** | `.venv-mp/bin/python src/features/extract_signals.py --dataset {dmd,nitymed,uta} --workers 6` | DMD 2,6 min (s5) + 24 min (distração); NITYMED 2,2 min; UTA ~2 h | `data/signals/<base>/*.npz` |
| Resumo e alinhamento | .venv | `src/features/check_dmd_alignment.py`; `src/features/summarize_signals.py` | 1 min | `results/sinais_*` |
| Características (piscadas, janelas de 10 s, sequências) | .venv | `bash src/features/run_features.sh` | 3 min | `data/features/final/`, `results/exp5_caracteristicas.md` |
| **Exp. 1** | .venv | `.venv/bin/python src/experiments/exp1_intra.py` (+ `--epochs 15 --tag _e15 --no-save`, `--regen-md`) | 10 min | `results/exp1_intra.{csv,md}`, `models/blink_temporal.tflite` |
| **Exp. 2** | .venv | `.venv/bin/python src/experiments/exp2_leakage.py` | 13 min | `results/exp2_leakage.{csv,md}` |
| **Exp. 3** | .venv | `src/pipeline_b/datasets.py`; `src/pipeline_b/train_eye_state.py --epochs 12 --max-train 25000 --threads 8`; `src/pipeline_b/report_exp3.py` (no `run_all.sh`, os logs vão para `results/logs/exp3_*.log`) | 16 min | `results/exp3_*` |
| **Exp. 4** desktop | .venv | `.venv/bin/python src/pipeline_b/export_tflite.py` | 30 s | `models/eye_state_{fp32,int8}.tflite`, `results/exp4_{tflite_eval,desktop_latency}.csv` |
| **Exp. 5a** | ambos | `bash src/experiments/run_exp5a.sh` | 5 min (+ ~1 h de dlib na 1ª vez) | `results/exp5a_validacao.md` |
| **Exp. 5b** | .venv | `bash src/experiments/run_exp5b.sh` (`--so-relatorio` refaz só o .md) | de 11 a 20 min | `results/exp5b_transferencia.md`, `models/exp5/` |
| **Exp. 5b**, controles da queda | .venv | `.venv/bin/python src/experiments/exp5b_robustez.py` (o `run_all.sh` roda junto, na etapa exp5b) | 4 min (log de 26/10: 233 s) | `results/exp5b_robustez.{csv,md}`, log em `results/logs/exp5b_robustez_*.log` |
| **Exp. 6** | ambos | `bash src/experiments/run_exp6.sh` | ~40 min (luminância) + 1 min | `results/exp6_noite.md` |
| **Exp. 4** celular | .venv | benchmark no aparelho (seção 8) → `.venv/bin/python tools/import_phone_benchmark.py results/phone` | segundos | `results/exp4_phone_latency.{csv,md}` |
| App: sensibilidade dos pesos | node | `node tools/app_validacao/sensibilidade_pesos.mjs` (até 6 threads; `--workers=1` dá o mesmo resultado). Usa os sinais do DMD de `data/signals/app_tasks/dmd/` | 92 s (26/10) | `results/app_sensibilidade_pesos.{csv,md}` |
| App: teste independente na UTA-RLDD | **.venv-mp** e node | `.venv-mp/bin/python tools/app_validacao/extrair_sinais_app.py --uta-folds 2 4 --uta-rotulos 0 10 --proc 6`; depois `node tools/app_validacao/validar_motor_uta.mjs` | cerca de 40 min de extração (48 vídeos) + segundos | `data/signals/app_tasks/uta/`, `results/app_validacao_uta.{csv,md}` |
| App × modelos nas mesmas pessoas | .venv | `.venv/bin/python src/experiments/comparar_app_uta.py` (depois do teste do app e dos Exp. 5b e 6) | segundos | `results/app_comparacao_uta.{csv,md}` |
| Figuras | .venv | `src/pipeline_b/make_figures.py` (amostras do MRL Eye); `src/experiments/make_figures.py` | 1 min | `results/figures/*.{pdf,png}` (inclui `amostras_mrl`) |
| Números, artigo, pôster | .venv | `bash tools/atualizar_celular.sh` (importa o celular, refaz a figura do Exp. 4, `paper/numeros.tex` + `poster/numeros.json` **conferidos contra a origem**, compila o artigo, gera o pôster) | 1 min | `results/numeros_resumo.md`, `paper/artigo.pdf`, `poster/poster_final.pdf` |

Nenhum número do artigo ou do pôster é digitado à mão. `src/experiments/make_paper_numbers.py` gera cada valor a partir do arquivo e da coluna indicados em `results/numeros_resumo.md`, e `src/experiments/conferir_numeros.py` recalcula todos eles (658 números, 0 divergências em 27/10/2026).

## 7. Resultados (estado de 27/10/2026)

Todos os números abaixo foram copiados dos arquivos indicados, que são gerados pelos scripts, e coincidem com as macros de `paper/numeros.tex`.
Os valores estão em %, salvo indicação. Os intervalos entre colchetes são IC 95% por bootstrap sobre **sujeitos**. "pp" = pontos percentuais.
A rodada preliminar de 27/09 (janelas de 30 s, só o Fold 1) foi descartada; nada dela é citado.

| Exp. | Resultado principal | Origem |
|---|---|---|
| 1 | UTA-RLDD, 3 classes, 5 folds por sujeito: Random Forest com BSA 54,5 ± 9,3 e VA 57,3 ± 7,2. Referências: HM-LSTM 54,0 / 65,2; julgamento humano (VA) 57,8 | `results/exp1_intra.md` |
| 2 | Os mesmos dados com partição aleatória por janela: RF 99,5, contra 54,2 por sujeito, ou seja, inflação de +45,3 pp (LSTM: +38,1 pp). A causa são janelas do mesmo vídeo nos dois lados: sequências vizinhas dividem 28 das 30 piscadas. A partição por vídeo, em que a pessoa aparece no treino com os outros vídeos, fica *abaixo* da por sujeito (47,2): nesta base, o que infla é o mesmo vídeo (e com ele o mesmo rótulo) nos dois lados, não a mesma pessoa. No cenário do Exp. 5b (janelas de 10 s, duas classes), a partição aleatória infla a acurácia balanceada do RF em 12,6 pp e a da RL em só 1,2 pp | `results/exp2_leakage.md`, `results/exp5b_robustez.md` (seção 4) |
| 3 | EyeNet: MRL → MRL 94,1 ± 3,4; MRL → CEW 79,0 ± 8,5 (modelo final 85,7/90,0); CEW → MRL 68,1 ± 1,9. Com óculos, no sentido CEW → MRL, a AUC cai para 65,1, contra 93,4 sem óculos | `results/exp3_eye_state.md` |
| 4 | EyeNet int8 com 72,1 KB (fp32: 248,2 KB) e queda de no máximo 0,31 pp. No iPhone 17 Pro, o app 1.0 levou 11 e 23 ms por quadro (p50) em 2 execuções, mediana de 17 ms; o app 2.1 ainda vai ser medido no iPhone e no Galaxy S23 Ultra | `results/exp4_tflite_eval.csv`, `results/exp4_phone_latency.md` |
| 5a | Trocar o extrator (dlib → MediaPipe) num RF treinado nas características oficiais custa +2,0 pp [−4,8; +8,5] (3 classes, 47 sujeitos), com IC que inclui zero. Com cada extrator no próprio treino e teste, o nosso é melhor em B3 nos 5 folds (oficial − nosso = −6,7 pp [−12,9; −1,8], p = 0,029), mas, deixando um sujeito de fora por vez, a diferença não se distingue de zero (−4,5 [−11,0; 0,6]) | `results/exp5a_validacao.md` |
| 5b | PP1. AUC por janela (B3, RF): DMD → DMD 82,4, UTA → UTA 80,1, DMD → UTA 69,2. A RL perde de 5,4 a 9,8 pontos de AUC por janela nas quatro formas testadas de normalizar (principal: 9,8 [3,7; 16,0]), sempre com IC acima de zero. O RF perde 10,9 [6,7; 15,7] na normalização principal, mas, com a referência no vídeo de pouca vigilância ou sem centragem, perde de 2,1 a 2,3, sem diferença detectável: na UTA-RLDD, as janelas alertas saem do mesmo vídeo da referência, e o RF treinado nela aproveita essa semelhança. O achado mais firme: na UTA real, o modelo treinado em sono encenado reconhece 82,0% das janelas alertas, mas só 48,4% das sonolentas (limiar 0,5); a sonolência atuada é mais extrema que a real (maior fechamento mediano de 0,97 s no DMD e 0,17 s na UTA-RLDD). O volume do treino explica pouco: com B treinado em só 13 sujeitos, a queda fica em 9,3 (RF) e 7,0 (RL) | `results/exp5b_transferencia.md`, `results/exp5b_robustez.md` |
| 6 | PP3. SNR do EAR: noite (NITYMED) 7,2, UTA-RLDD 21,5, DMD parado 13,6, DMD em movimento 8,7. Noite − movimento = −1,6 [−3,7; +1,1], com IC que inclui zero: só o carro andar já leva a SNR para perto do nível noturno. Em pessoas acordadas (UTA KSS 0), no limiar 0,5, os RF com características brutas disparam em 26,0% (modelo do DMD) e 43,6% (modelo da UTA) das janelas, e a regra por sinais visíveis em 2,6%. Esse limiar é arbitrário: com o mesmo falso alarme da regra, a RL do DMD centrada acha tantas janelas sonolentas quanto ela (24,7% contra 24,1%) | `results/exp6_noite.md`, `results/exp5b_robustez.md` (seção 5) |
| App 2.1 | Falso alarme em pessoas acordadas: DMD em direção real (s1, 17 sessões, 63,3 min) ficou 3,3% do tempo em vermelho, e 16 de 17 sessões não tiveram alarme. O caso restante é o motorista gA_5, que olha para baixo e é lido como olho fechado. A lógica 1.0 ficava 55,6% do tempo em vermelho. UTA-RLDD acordado: 3 vídeos 100% verdes, 0 alarmes. Detecção: estado do olho por quadro com kappa 0,938 (offline) e 0,902 (navegador); fechamentos de 1 s ou mais com revocação 0,920; bocejo com revocação 0,638 sem a mão e quase nula com a mão na boca; UTA sonolento: só 1 de 3 pessoas chegou ao vermelho. Cabeça: sem revocação medida. Teste independente em sono real (24 pessoas da UTA-RLDD dos folds 2 e 4, fora do ajuste, sentadas e sem dirigir; o rótulo é o sono declarado): alarme em 2 dos 24 vídeos acordados (0,46% do tempo em vermelho); 15 dos 24 sonolentos chegaram ao vermelho, e 7 ficaram verdes o vídeo inteiro; AUC sonolento × acordado de 0,812 [0,705; 0,918]. Sensibilidade dos pesos: em 300 combinações sorteadas (cada peso, teto e decaimento de 0,7 a 1,3 vez; limiares 40 e 70 deslocados em até 10 pontos), o falso alarme na direção real ficou de 1,7% a 4,9% do tempo (percentis 5 e 95; a configuração final dá 3,3%). Nas mesmas 24 pessoas do teste independente, a AUC por vídeo do RF treinado no DMD foi 0,811, a da regra do rótulo 0,853 e a do RF treinado na própria UTA-RLDD 0,953 com a centragem pelo vídeo acordado e 0,806 sem ela: o app (0,812), sem treino, fica no nível deles | `results/app_validacao_dmd.md`, `results/app_validacao.md`, `results/app_validacao_uta.md`, `results/app_sensibilidade_pesos.md`, `results/app_comparacao_uta.md` |

Leitura em uma linha:
- misturar janelas do mesmo vídeo no treino e no teste infla muito a acurácia (de 54,2% para 99,5% nas sequências de piscadas), mas isso não explica os 90% do NTHU-DDD da literatura, que foram medidos com outras pessoas no teste (Exp. 2 e 5b);
- treinar em sono encenado custa de 5,4 a 9,8 pontos de AUC por janela na RL; no RF, a perda depende da normalização; e o modelo do DMD reconhece só 48,4% das janelas sonolentas reais (Exp. 5b);
- parte do que se atribui à noite é movimento do carro (Exp. 6);
- a regra por sinais visíveis não é mais precisa que os modelos; o app usa a regra porque cada alarme tem explicação e ela não precisa de treino, e o aprendizado de máquina fica só como estimativa;
- com sono real declarado, em 24 pessoas fora do ajuste (sentadas, sem dirigir), o app alarmou 15 dos 24 vídeos sonolentos e 2 dos 24 acordados: ele reage a sinais visíveis, não ao sono declarado.

## 8. Demonstrações e apps

**Demo de webcam (Pipeline A, desktop). Roda no `.venv-mp`**, porque no `.venv` o MediaPipe não funciona.
Ela calibra o limiar de EAR nos primeiros segundos (olhos abertos); `q` sai. O alerta é um bipe do terminal.

```bash
.venv-mp/bin/python src/realtime/demo_webcam.py [--camera 0] [--closed-s 1.0] [--perclos-thresh 0.15]
```

**PWA, o app de demonstração (2.1).** O guia completo, com a lógica do nível de 0 a 100, a validação e os limites, está em [`app/pwa/README.md`](app/pwa/README.md).
Celular e computador ficam na mesma rede. Na PUC, a rede isola os aparelhos: use o Acesso Pessoal do iPhone e conecte o Mac e o celular nele.

O app funciona sem internet quando a pasta `app/pwa/vendor/` está completa. Ela tem as dependências que, sem
ela, viriam da internet: o MediaPipe com os arquivos WASM, o modelo `face_landmarker.task`, o TensorFlow.js e o
tfjs-tflite, nas mesmas versões fixas do `app.js` (13 arquivos, cerca de 36 MB, fora do Git). Baixe uma vez, com internet:

```bash
bash tools/baixar_dependencias_app.sh             # baixa o que falta para app/pwa/vendor/ e confere cada arquivo pelo SHA-256
bash tools/baixar_dependencias_app.sh --conferir  # só confere, sem rede (sai com erro se faltar algo)
python3 app/pwa/serve_https.py      # imprime https://<IP-do-Mac>:8443/ ; confere vendor/ ao subir e oferece baixar o que faltar
node app/pwa/test_engine.mjs        # testes do motor (217 asserções)
```

No celular:
1. Abra a URL, aceite o certificado e toque em **Iniciar câmera**. Em "Detalhes técnicos > Modelos > Bibliotecas", deve aparecer "local (./vendor/)".
2. Faça a calibração com os olhos abertos.
3. Em "Detalhes técnicos", toque em **Benchmark 60 s**. Ao terminar, o app envia o resultado sozinho ao Mac (`POST /api/benchmark`), e os arquivos vão para `results/phone/`. Os botões de download ficam como reserva.
4. Depois, rode `bash tools/atualizar_celular.sh`. Ele atualiza a tabela e a figura do Exp. 4, os números, o artigo e o pôster.

O nome do aparelho vem dos Client Hints do Chrome no Android (`SM-S918*` = Galaxy S23 Ultra). No iPhone, que não informa o modelo, ele vem de `results/phone/aparelhos.json`.

O volume do alarme começa em 20% e sobe para 100% só na banca. Todo teste automático roda mudo.

**App Android nativo** (`app/android/`, Kotlin, CameraX + MediaPipe Tasks + TFLite).
Ele mantém a lógica anterior (1.0: alarme por PERCLOS/LSTM), que foi substituída na PWA 2.1, e nunca foi compilado: esta máquina não tem JDK nem Android SDK.
Serve como caminho alternativo de medição de latência, não como app de demonstração. Instruções em [`app/android/README.md`](app/android/README.md):

```bash
bash app/android/tools/fetch_assets.sh     # face_landmarker.task + modelos .tflite para os assets
# abrir app/android/ no Android Studio (JDK 17, SDK 36) -> Run -> Benchmark 60 s -> copiar para results/phone/
```

## 9. Ética e LGPD

- **Não é dispositivo médico nem veicular.** É um protótipo de pesquisa acadêmica: não diagnostica distúrbios do sono, não é certificado e não substitui o descanso previsto na Lei 13.103/2015. Ele detecta **sinais visíveis** (olhos, bocejo, cabeça); não lê o estado interno da pessoa.
- **Processamento local.** Apps e demo processam o vídeo no próprio aparelho. Nenhum quadro de vídeo é armazenado nem transmitido. Os benchmarks e as validações gravam apenas números (latências, EAR, estados, eventos).
- **Telemetria do MediaPipe.** No app, com internet, a biblioteca do MediaPipe envia telemetria de uso para um servidor do Google (`odml.pa.googleapis.com`) depois de cerca de 60 s de uso, sem quadros de vídeo. Sem internet (com `app/pwa/vendor/` completa, o app funciona assim), nada sai do aparelho.
- **Imagens e licenças.** As bases são usadas nos termos das licenças da seção 3. Nenhum quadro com rosto do DMD, da UTA-RLDD ou do NITYMED entra no artigo, no pôster ou no repositório: a figura interna de verificação do DMD está no `.gitignore`. Nas figuras, só aparecem recortes de olho do MRL Eye (`results/figures/amostras_mrl`); o CEW, cuja licença proíbe redistribuição, não aparece em figura.
- **Pessoas filmadas.** Ninguém foi filmado para o projeto além dos próprios integrantes, em testes voluntários dos apps, cujos benchmarks contêm só números.
- **Uso de assistentes de IA.** Assistentes de IA foram usados como apoio na programação e na revisão de texto. Cada número foi conferido contra `results/`.

## 10. Licença e como citar

O código está sob licença MIT (`LICENSE`). Os dados **não** estão incluídos e seguem as licenças próprias (seção 3). Ao usar este repositório, cite:
- as bases: `ghoddoosian2019`, `ortega2020`, `petrellis2022` + `petrellis2021dsd`, `fusek2018`, `song2014`;
- o MediaPipe: `lugaresi2019`;
- as características de piscada de Ghoddoosian et al. (2019).

A bibliografia completa e verificada está em `paper/refs.bib`. Não cite as entradas marcadas `UNVERIFIED`.
