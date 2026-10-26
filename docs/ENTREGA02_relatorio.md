# Entrega 02: Codificação, Metodologia Implementada e Resultados

**Detecção de Sonolência em Dispositivos Móveis: Avaliação da Generalização entre Sonolência Atuada e Real**

Projeto Transformador II, Ciência da Computação, PUCPR. Turma 8ºA, Equipe 14.
André Luís O. F. da Silva, Bruno Rodolfo da Silveira, Bruno Sandoval Ribeiro, Paulo Henrique Perin. Orientador: Prof. Júlio César Nievola.

Versão de 26/10/2026 (prazo da entrega: 31/10/2026). Repositório: https://github.com/BrunoRodolfoo/deteccao-sonolencia

> Todos os números deste relatório foram copiados dos arquivos de `results/` gerados pelos scripts. O arquivo de cada tabela aparece entre parênteses, e os mesmos valores estão em `paper/numeros.tex`, conferidos por `src/experiments/conferir_numeros.py` (615 números, 0 divergências em 26/10/2026). Nenhum valor foi digitado à mão ou estimado.
>
> Os números do celular (aparelhos, execuções, versão do app e tempos) estão em `results/exp4_phone_latency.md`.
>
> A rodada preliminar dos Exp. 5 e 6 (janelas de 30 s, só o Fold 1) foi descartada e não é citada.

## 1. Resumo executivo

O projeto implementou, executou e documentou seis experimentos e um app de demonstração.
- Os Exp. 1 a 4 ficaram prontos em 20/09.
- Os Exp. 5a, 5b e 6 foram feitos em 27/09, depois da troca do NTHU-DDD, que não foi obtido, por DMD (sonolência encenada) e NITYMED (condução noturna real). A justificativa está em `docs/TROCA_DATASETS.md`.
- Em 26/10 entraram os controles da queda do Exp. 5b (`src/experiments/exp5b_robustez.py`), que mudaram a leitura da PP1.
- A base de sonolência real continua sendo a UTA-RLDD, com os vídeos dos 60 sujeitos processados.

Achados centrais:

- PP1: sono encenado → sono real (Exp. 5b). Com o mesmo extrator e as mesmas características, os modelos treinados no DMD reconhecem o sono real pior que os treinados na própria UTA-RLDD.
  - A regressão logística (RL) perde de 5,4 a 9,8 pontos de AUC por janela nas quatro formas testadas de normalizar os dados de cada pessoa, sempre com IC acima de zero.
  - O Random Forest (RF) perde 10,9 [6,7; 15,7] na normalização principal. Com a referência no vídeo de pouca vigilância ou sem centragem, perde de 2,1 a 2,3, sem diferença detectável: na UTA-RLDD, as janelas alertas saem do mesmo vídeo da referência, e o RF treinado nela aproveita essa semelhança.
  - O achado mais firme: o modelo do DMD reconhece 82,0% das janelas alertas reais, mas só 48,4% das sonolentas, e a sonolência atuada se mostra mais extrema que a real. O volume do treino explica pouco (B treinado com 13 sujeitos).
  - O extrator não explica a perda: trocar o extrator oficial (dlib) pelo nosso (MediaPipe) custa +2,0 pp [−4,8; +8,5] (Exp. 5a).
- Vazamento entre treino e teste. Com partição aleatória por janela, os mesmos dados do Exp. 1 vão de 54,2% a 99,5% de BSA (+45,3 pp; Exp. 2): sequências vizinhas dividem 28 das 30 piscadas, e janelas do mesmo vídeo caem nos dois lados. No cenário do Exp. 5b, a inflação é de 12,6 pp no RF e de só 1,2 pp na RL. Isso não explica os 90% do NTHU-DDD da literatura, que foram medidos com outras pessoas no teste.
- PP2: custo no celular. A EyeNet int8 tem 72,1 KB, com queda de no máximo 0,31 pp de acurácia. Tempo por quadro do app no celular: ver `results/exp4_phone_latency.md`.
- PP3: condições adversas.
  - Óculos é o que mais derruba a transferência do estado ocular (CEW → MRL: AUC 65,1 com óculos, 93,4 sem; Exp. 3).
  - À noite (NITYMED), a SNR do EAR cai para 7,2, contra 21,5 na UTA-RLDD. Mas o DMD com os mesmos motoristas e o carro em movimento, de dia, já chega a 8,7, e a diferença noite − movimento (−1,6 [−3,7; +1,1]) não exclui zero: parte da degradação atribuída à noite vem do movimento (Exp. 6).
- App de demonstração (PWA 2.1). O app alarma por sinais visíveis (olhos, bocejo e cabeça), num nível de 0 a 100 que explica cada ponto. A regra não é mais precisa que os modelos: com o mesmo falso alarme, a RL do DMD acha tantas janelas sonolentas quanto ela. A vantagem da regra é ser explicável e não precisar de treino.
  - Na direção real do DMD, com motoristas acordados, o app ficou 3,3% do tempo em vermelho, e 16 de 17 sessões não tiveram alarme.
  - O estado do olho por quadro tem kappa 0,938 (offline) e 0,902 (navegador).
  - Desde 26/10, o app funciona sem internet (`app/pwa/vendor/`, baixada por `tools/baixar_dependencias_app.sh`).

## 2. O que foi feito em outubro

| Data | O que foi feito | Onde ver |
|---|---|---|
| 04/10 | O orientador autorizou a troca de bases, verbalmente | `docs/TROCA_DATASETS.md` |
| 06/10 | Artigo enxugado para 7 páginas, sem travessões e com a Fig. 1 refeita em preto e branco; números, artigo e pôster refeitos e conferidos | `results/logs/atualizar_celular_20261006_161528.log` |
| 06/10 | Limpeza da pasta: rodada preliminar, logs repetidos e cópias sem uso foram para a Lixeira | `docs/ESTRUTURA.md`, seção 3 |
| 26/10 | Controles da queda do Exp. 5b: referência no vídeo de pouca vigilância, IC em dois níveis (motoristas do DMD reamostrados e C treinado de novo), B treinado com 13 sujeitos, vazamento no cenário do Exp. 5b e ponto de operação da regra | `results/exp5b_robustez.md`; log `results/logs/exp5b_robustez_20261026-2051.log` |
| 26/10 | Banca simulada e reescrita do artigo, com a leitura nova da PP1, do vazamento e da regra × modelos: 8 páginas, 30 referências, Tabelas I a V e 3 figuras; números refeitos e conferidos (615, 0 divergências) | `paper/README.md`, `results/numeros_resumo.md` |
| 26/10 | App 2.1 sem internet: dependências locais em `app/pwa/vendor/` (13 arquivos, cerca de 36 MB, fora do Git), baixadas e conferidas pelo SHA-256 por `tools/baixar_dependencias_app.sh`; o `serve_https.py` confere a pasta ao subir e oferece baixar o que faltar | `app/pwa/README.md`, seção 7 |
| 26/10 | Medição do app 2.1 no iPhone 17 Pro e no Galaxy S23 Ultra | `results/exp4_phone_latency.md` |
| 26/10 | Figura de amostras refeita só com o MRL Eye (a anterior tinha recortes do CEW, cuja licença proíbe redistribuição); figuras de reserva do artigo e `src/pipeline_b/run_all_eye.sh` foram para a Lixeira; o `run_all.sh` passou a gravar os logs do Exp. 3 e a rodar os controles do Exp. 5b | `docs/ESTRUTURA.md`, seção 3 |

## 3. Como verificar em 5 minutos

Na raiz do repositório, com o `.venv` instalado (`README.md`, seção 5) e o Node 18 ou mais novo. Nenhum destes comandos precisa dos dados de `data/` nem refaz experimentos.

```bash
node app/pwa/test_engine.mjs                                  # motor do app
.venv/bin/python -m pytest -q src/features/test_features.py   # piscadas, janelas, sequências e rótulos
bash run_all.sh --dry-run                                     # o pipeline inteiro, só os comandos
.venv/bin/python src/experiments/conferir_numeros.py          # todos os números do artigo e do pôster
```

| Comando | O que confere | Resultado esperado |
|---|---|---|
| `test_engine.mjs` | o motor do app (`engine.js`): EAR, piscadas, PERCLOS, bocejo, cabeça, nível e histerese | "217 passaram, 0 falharam", em menos de 1 s |
| `test_features.py` | detector de piscadas, janelas de 10 s, sequências de 30 piscadas e rótulos do DMD | "19 passed", em cerca de 1 s |
| `run_all.sh --dry-run` | a ordem das 11 etapas (`--lista`) e o comando de cada uma, inclusive `exp5b_robustez.py` na etapa exp5b | lista os comandos e termina sem rodar nada |
| `conferir_numeros.py` | recalcula cada número de `paper/numeros.tex` e `poster/numeros.json` a partir dos CSV de `results/`, com leitura e formatação próprias | "615/615 ok" e "divergências 0", em poucos segundos |

O `conferir_numeros.py` reescreve a seção "Conferência automática" de `results/numeros_resumo.md` e compila um documento de teste com o tectonic; sem o tectonic instalado, use `--sem-latex`. Para ir além, escolha um número do artigo (por exemplo, 48,4) e procure-o em `results/numeros_resumo.md`, que diz o arquivo e a coluna de origem.

## 4. O que foi implementado

| Módulo | Conteúdo | Testes e validações |
|---|---|---|
| `src/baseline/` (Entrega 01) | download das características oficiais da UTA-RLDD, folds, LogReg, SVM e RF | segunda execução reprodutível (24/08 e 26/08) |
| `src/experiments/uta_common.py`, `blink_models.py` | carga dos 5 folds, reconstrução de vídeo e sujeito (58 sujeitos, 8.661 sequências), métricas BSA e VA, esquemas 3c, B1, B2 e B3; LogReg, SVM, RF, LSTM, CNN1D | `assert` contra os índices oficiais |
| `src/experiments/exp1_intra.py`, `exp2_leakage.py` | Exp. 1 e 2 + exportação TFLite do LSTM | TFLite × Keras: max\|dif\| 6,5e-6 |
| `src/pipeline_b/` | Exp. 3 (EyeNet, protocolos A, B, C e D por condição), Exp. 4 desktop (TFLite fp32 e int8) e a figura de amostras do MRL Eye | fp32 idêntico ao Keras; ablação da BatchNorm |
| `src/features/extract_signals.py` (.venv-mp) | vídeo → EAR, MAR e pose por quadro (MediaPipe Face Mesh 0.10.21): DMD (65 sessões), NITYMED (126), UTA-RLDD (182 arquivos); incremental, escrita atômica | alinhamento anotação × vídeo do DMD verificado (deslocamento 0) |
| `src/features/` (`blinks.py`, `dmd_labels.py`, `windows.py`, `sequences.py`, `uta_oficial.py`) | piscadas, janelas de 10 s (5 e 15 s), rótulos do DMD, sequências de 30 piscadas, mapeamento sujeito oficial ↔ pasta | 19 testes (`test_features.py`); validação do detector contra a anotação do DMD |
| `src/experiments/exp5a_*.py` | Exp. 5a: nosso extrator × características oficiais (5 folds, 47 sujeitos confirmados); detector do dlib (.venv-mp) | bootstrap por sujeito, permutação pareada, McNemar; 3 checagens independentes do mapeamento |
| `src/experiments/exp5_transfer.py` | Exp. 5b: DMD → UTA-RLDD (A, B e C), 8 sensibilidades, controle do vídeo de referência, importância por permutação | referência de normalização separada da avaliação (0 janelas sobrepostas) |
| `src/experiments/exp5b_robustez.py` (26/10) | controles da queda B − C: referência no vídeo de pouca vigilância, bootstrap em dois níveis com C treinado de novo, B com 13 sujeitos (20 sorteios), vazamento no cenário do Exp. 5b, ponto de operação da regra | antes das análises, refaz a queda principal e confere com o Exp. 5b (10,9 [6,7; 15,7] no RF e 9,8 [3,7; 16,0] na RL) |
| `src/experiments/exp6_*.py` | Exp. 6: detecção de rosto, qualidade do EAR, taxa de disparo, luminância medida | DMD em movimento × parado como controle, pareado por motorista |
| `src/experiments/make_paper_numbers.py`, `conferir_numeros.py` | todos os números do artigo e do pôster (615 em 26/10) → `paper/numeros.tex`, `poster/numeros.json`, `results/numeros_resumo.md` | recálculo independente: 0 divergências |
| `app/pwa/` (demonstração, 2.1) | câmera → MediaPipe Face Landmarker (WASM/WebGL) → olhos (EAR + blendshape), bocejo (jawOpen/MAR), cabeça (matriz de pose) → nível de 0 a 100 com histerese → alarme sonoro; CNN e LSTM como estimativa; benchmark de 60 s enviado ao Mac; funciona sem internet com `vendor/` | 217 asserções em Node; validação offline no DMD e no navegador com 14 vídeos reais (`results/app_validacao*.md`) |
| `app/android/` (Kotlin) | lógica anterior (1.0): PERCLOS e LSTM; benchmark nativo | nunca compilado (sem JDK e SDK); JUnit escrito, não executado |
| `tools/` | obtenção de dados (UTA-RLDD, DMD), dependências do app (`baixar_dependencias_app.sh`, com conferência por SHA-256), importação do benchmark do celular, `atualizar_celular.sh` (um comando: importa → figura → números → artigo → pôster), validação do app | testado de ponta a ponta com os dados atuais |
| `paper/`, `poster/` | artigo IEEE (`artigo.tex`, `refs.bib` verificado), pôster no template do professor | compilação com tectonic; PDF do pôster pelo Chrome headless, mudo |
| `run_all.sh`, `requirements*.txt` | reprodução completa em 11 etapas (a etapa exp5b roda também os controles); dois ambientes (`.venv`, `.venv-mp`) com congelamento completo | `--dry-run`, `--smoke` |

## 5. Checkpoints (`models/`)

Os modelos treinados ficam em `models/`. Os pequenos e os usados pelo app ou por outros experimentos são versionados; os grandes, que um script refaz em minutos, ficam fora do Git.

| Arquivos | O que são | Para que servem | Git |
|---|---|---|---|
| `LogReg_fold{1..5}.pkl`, `SVM-RBF_fold{1..5}.pkl`, `RandomForest_fold{1..5}.pkl` (raiz) | baseline da Entrega 01: classificadores clássicos sobre 24 estatísticas das sequências oficiais, um por fold | reproduzir a baseline (`src/baseline/train.py`; `results/baseline_uta_rldd.md`) | sim |
| `exp1/` (100 arquivos) | Exp. 1: LogReg, SVM-RBF e RF (`.pkl`), LSTM e CNN1D (`.keras`), por fold (5) e por esquema (3c, B1, B2, B3) | refazer as tabelas do Exp. 1 sem treinar de novo; o `exp1_intra.py` os refaz em cerca de 10 min | não |
| `exp1_best_temporal.txt` | o melhor modelo temporal do Exp. 1 (`LSTM`) | escolhe o modelo exportado para o app | sim |
| `blink_temporal.tflite` + `blink_temporal.json` | LSTM de piscadas do fold 3 em TFLite (101,8 KB, 5.324 parâmetros), com o contrato de entrada (30 × 4, ordem das características, normalização) | estimativa experimental no app (soma no máximo 10 pontos) e latência do Exp. 4 | sim |
| `exp3/eyenet_mrl_fold{1..5}.keras`, `exp3/eyenet_mrl_final.keras` | EyeNet treinada no MRL Eye: 5 folds por sujeito e o modelo final com os 37 sujeitos | Exp. 3 (protocolos A e B); o modelo final é o que vira TFLite | sim |
| `exp3/eyenet_cew_{patch,crop}_fold{1..5}.keras` | EyeNet treinada no CEW, com o recorte oficial 24×24 e com o recorte da face | Exp. 3 (protocolos C e D) | sim |
| `exp3/mobilenetv3_mrl_fold1.keras` | MobileNetV3-Small no fold 1 | referência opcional do Exp. 3 | não |
| `eye_state_fp32.tflite`, `eye_state_int8.tflite` + `eye_state.json` | EyeNet final em TFLite (248,2 KB e 72,1 KB), com o contrato de entrada e saída e as métricas da quantização | Exp. 4 (quantização e latência) e a CNN do olho que o app mostra | sim |
| `exp5/exp5b_final_C_{principal,sem_normalizacao}_{RF,LR}.joblib` + `exp5b_final_modelos.json` | os quatro modelos de transferência do cenário C (treino em todo o DMD: 510 janelas de 13 motoristas), com as características, a normalização e os limiares calibrados no DMD | Exp. 5b (cenário C) e Exp. 6 (taxa de disparo do modelo do DMD) | sim |
| `exp6/rf_uta_*.joblib`, `exp6/rf_dmd_sem_taxa_rosto.joblib` + `exp6/modelos.json` | RF treinados em todas as janelas B3 da UTA-RLDD e variantes sem a taxa de rosto | Exp. 6 (disparo do modelo da UTA e controles); o `run_exp6.sh` os refaz em cerca de 1 min | não |

Os `.tflite` e `.json` da raiz têm cópias idênticas em `app/pwa/models/`, que é o que o app carrega.

## 6. Metodologia implementada

### 6.1 Dados

| Base | Natureza | Uso | Sujeitos e vídeos usados |
|---|---|---|---|
| UTA-RLDD (Ghoddoosian et al., 2019) | sonolência real autorrelatada (KSS 0, 5 e 10), celular ou webcam | teste real; características oficiais nos Exp. 1, 2 e 5a | 60 sujeitos, 180 vídeos lógicos (182 arquivos); os `.npy` oficiais têm 58 sujeitos |
| DMD (Ortega et al., 2020) | sonolência encenada, carro parado (s5); direção segura com o carro parado (s2 e s3) e em movimento (s1) | treino atuado (Exp. 5b); controle de movimento (Exp. 6); validação do app | 13 sujeitos com s5 (16 sessões) + sessões de distração (65 sessões no total, 14 sujeitos) |
| NITYMED (Petrellis et al., 2022) | condução noturna real; bocejos e microssonos de autenticidade não documentada | condição noite (Exp. 6), sem acurácia | 126 vídeos (19 de microssono, 107 de bocejo; a ficha oficial diz 130) |
| MRL Eye (Fusek, 2018) | recortes infravermelhos com metadados | Exp. 3 | 84.898 recortes, 37 sujeitos |
| CEW (Song et al., 2014) | faces RGB *in the wild* | Exp. 3 | 2.423 faces, 4.846 olhos |

Detalhes de obtenção, licença e citação obrigatória: `README.md` (seção 3) e `data/README.md`.

### 6.2 Exp. 1: intra-conjunto (teto na UTA-RLDD)

Protocolo oficial: treino nos 4 folds restantes, teste no fold em questão, 5 vezes. Modelos clássicos
sobre 24 estatísticas por sequência (média, desvio, mín., máx., mediana e IQR de cada característica) e
modelos temporais sobre a sequência bruta 30×4. Métricas por fold e média ± desvio populacional: BSA
(acurácia por sequência), acurácia balanceada, F1 macro, AUC macro OvR e VA (acurácia por vídeo,
voto majoritário das janelas do vídeo; empate → maior probabilidade média). Esquemas: 3 classes e três
harmonizações binárias (B1 alerta × resto; B2 sonolento × resto; B3 alerta × sonolento).
Sensibilidade a épocas (15 × 50) para os temporais.

### 6.3 Exp. 2: vazamento entre treino e teste

Pool: as 8.661 sequências de teste dos 5 folds (cada sequência aparece uma vez), com `video_id` e
`subject_id` reconstruídos. Três partições em validação cruzada 5-fold: (i) por janela aleatória
(janelas sobrepostas do mesmo vídeo em treino e teste; sementes 42, 43 e 44), (ii) por vídeo (nenhum
vídeo dos dois lados, mas os outros dois vídeos do sujeito no treino), (iii) por sujeito (folds
oficiais). Mesmos modelos e hiperparâmetros do Exp. 1 (RF e o melhor temporal, LSTM). A inflação é a
diferença (i) − (iii) e (ii) − (iii) em cada métrica. Em 26/10, a mesma comparação foi refeita no cenário do
Exp. 5b (janelas de 10 s, duas classes; `exp5b_robustez.py`, seção 4).

### 6.4 Exp. 3: estado ocular intra e entre bases

EyeNet: entrada 48×48×1 em [0,1]; Conv(16)-BN-ReLU-Pool / Conv(32)-BN-ReLU-Pool /
Conv(64)-BN-ReLU-Conv(64)-BN-ReLU-Pool / GAP / Dense(32) / Dropout(0,3) / sigmoide; 62.865 parâmetros;
Adam 1e-3, lote 128, no máximo 12 épocas, EarlyStopping (val_loss, paciência 3), ReduceLROnPlateau; aumento de dados
(espelhamento, rotação ±10°, translação ±8 %, zoom ±10 %, brilho ±15 %, contraste ±20 %); BatchNorm com momento 0,9
(ablação com 0,99 em `results/exp3_bn_momentum_ablation.csv`: resultados comparáveis). Protocolos:
A: GroupKFold de 5 folds por sujeito no MRL (treino subamostrado a 25.000 imagens estratificadas por
sujeito × classe; 10 % dos sujeitos de treino para early stopping; teste em todas as imagens dos sujeitos
de teste), com predições fora da amostra estratificadas por condição; B: MRL → CEW (os 5 modelos de A e
um modelo final treinado nos 37 sujeitos); C: CEW → MRL (5 folds por face + ensemble), por condição
do MRL; D: CEW → CEW (faces não vistas). Referência opcional: MobileNetV3-Small 0,75 no fold 1.

### 6.5 Exp. 4: custo no celular

Exportação do EyeNet final para TFLite fp32 e int8 (quantização pós-treino completa, E/S uint8,
conjunto representativo de 500 imagens do MRL) e do LSTM de piscadas (fp32, `unroll=True`, só operações builtin).
Métricas: queda de acurácia e AUC dos `.tflite` em relação ao Keras (MRL e CEW), tamanho em KB e latência
em lote 1 (50 aquecimentos + 500 execuções, 1 e 4 threads) no desktop. No celular: latência p50 e p95
por estágio (landmarker, CNN dos dois olhos, motor ou LSTM, total por quadro) e quadros por segundo em benchmarks de 60 s,
medidos pela PWA (MediaPipe WebAssembly/WebGL + tfjs-tflite), que envia o resultado sozinha ao Mac; o app
Android nativo (TFLite + MediaPipe Tasks) fica como caminho alternativo, ainda não compilado. Protocolo em
`app/pwa/README.md` e `docs/GUIA_EQUIPE.md`, seção 1.

### 6.6 Extração de sinais e características (base dos Exp. 5a, 5b e 6)

**Sinais.** MediaPipe Face Mesh em modo vídeo, como no app. EAR e MAR em pixels; aproximações da pose. Quadros sem rosto ficam como NaN. O lado maior do quadro é reduzido para no máximo 1280 px, e a rotação do metadado é corrigida.

**Piscadas.** Limiar de 0,75 × a mediana do EAR num trecho de referência do próprio sujeito. A referência é separada da avaliação:
- UTA-RLDD: último terço do vídeo alerta. Os dois terços iniciais são o alerta avaliado.
- DMD: trechos de direção segura da sessão s3.
- NITYMED: primeiro terço do vídeo, um desvio inevitável, porque a base não tem trecho alerta.

As janelas de referência e as que tocam nela nunca entram em treino nem em teste.

**Janelas.** 10 s com passo de 5 s nas três bases. Os trechos de direção segura do DMD duram de 10 a 26 s, então as janelas de 30 s do planejamento não cabem neles.

Cada janela tem 11 características: PERCLOS, taxa de piscadas, duração média, maior fechamento, fechamentos de 0,5 s ou mais, EAR médio e desvio, amplitude, velocidade, fração de bocejo e taxa de rosto. A normalização principal é a centragem pelo sujeito: x − média da referência.

**Rótulos.**
- DMD: alerta = janelas inteiras dentro de direção segura da s2 (carro parado, como a s5); sonolento = janelas da s5 com fechamento anotado de 0,5 s ou mais ou bocejo; as janelas da s5 sem sinal ficam de fora.
- UTA-RLDD: KSS do vídeo. B3 (0 × 10) é o esquema principal.

**Vídeos divididos.** Em dois sujeitos (F3_32, F5_49), as partes do vídeo sonolento são tratadas como um vídeo lógico.

### 6.7 Exp. 5a: validação do nosso extrator

O RF do Exp. 1, treinado nas características oficiais dos outros 4 folds, é aplicado às características oficiais e às nossas dos mesmos vídeos.

Trabalhamos com os 47 sujeitos cujo par entre sujeito oficial e pasta de vídeo foi confirmado. O mapeamento foi reconstruído por engenharia reversa dos `.npy` e verificado por três checagens independentes.

A diferença mede o custo de trocar o extrator. Complemento: cada extrator treinado e testado nas próprias características (5 folds e LOSO).

Incerteza:
- IC 95% por bootstrap sobre sujeitos (1.000 reamostragens);
- permutação pareada (10.000 sorteios);
- McNemar na VA.

### 6.8 Exp. 5b: transferência atuado → real (PP1)

Três avaliações com as mesmas características e o mesmo extrator:
- (A) DMD → DMD, deixando um sujeito de fora;
- (B) UTA → UTA, nos 5 folds oficiais;
- (C) DMD → UTA, com um modelo treinado em todo o DMD.

Modelos: Random Forest (300 árvores) e regressão logística, ambos balanceados, com hiperparâmetros fixos.

Limiares:
- 0,5, fixado antes;
- calibrado só no treino (em C, no DMD, sem olhar a UTA).

Queda da PP1 = B − C, pareada por sujeito da UTA. Métricas por janela e por vídeo (média dos escores das janelas do vídeo lógico). IC 95% por bootstrap sobre sujeitos.

Sensibilidades: sem normalização; alerta da s1; s5 sem sinal como alerta; janelas de 5 s e 15 s; referência espelhada; sem o motorista gE_29.

Controle do vídeo de referência: AUC entre os vídeos de código 5 e 10 (KSS 6 e 7 contra 8 e 9), dois vídeos que não são a referência.

Controles acrescentados em 26/10 (`src/experiments/exp5b_robustez.py`, mesmos dados e mesma receita de modelos):
- referência no vídeo de pouca vigilância: a centragem da UTA-RLDD passa a usar a média das janelas do vídeo de pouca vigilância do sujeito (código 5, KSS 6 e 7), que não entra no B3; assim, nem as janelas alertas nem as sonolentas vêm do vídeo da referência;
- bootstrap em dois níveis: em cada reamostragem, os 13 motoristas do DMD também são sorteados com reposição, e C é treinado de novo;
- volume do treino: B treinado com só 13 sujeitos da UTA-RLDD por fold (todas as janelas deles ou 510 janelas, o tamanho do treino de C), em 20 sorteios;
- vazamento no mesmo cenário: folds oficiais por sujeito contra partição aleatória das janelas (sementes 42, 43 e 44);
- ponto de operação da regra: para cada modelo, o limiar que dá o mesmo falso alarme da regra nas janelas alertas. Esse limiar usa o rótulo do teste: é só um diagnóstico de onde a regra fica nas curvas ROC.

### 6.9 Exp. 6: noite (PP3)

Três medidas nas três bases:
- (a) detecção de rosto por quadro;
- (b) qualidade do EAR, em que a SNR rápida é a amplitude das piscadas dividida pelo ruído quadro a quadro (MAD do resíduo da mediana móvel de 1 s);
- (c) taxa de disparo: janelas marcadas como sonolentas pelo modelo do DMD (Exp. 5b), pelo modelo da UTA (B3) e por uma regra sem treino (fechamento de 0,5 s ou mais ou bocejo).

O DMD em movimento (s1) × parado (s2), pareado por motorista, separa o efeito do movimento.

O NITYMED não tem classe alerta, e nenhuma acurácia é calculada nele. Também medimos a luminância do rosto e dos olhos e a compressão dos vídeos.

### 6.10 App de demonstração (PWA 2.1) e a validação dele

**Detector.**
- Olhos: EAR calibrado com o blendshape eyeBlink, com travas para "olhar para baixo". O PERCLOS de 60 s conta só fechamentos de 0,5 s ou mais. Fechamentos longos e microssonos (1 s ou mais) também entram.
- Bocejo: jawOpen ou MAR relativo sustentado por 1,5 s ou mais.
- Cabeça: queda com os olhos fechados ou cabeceio. Cabeça baixa com os olhos abertos conta como distração e vale 0 ponto.
- LSTM: soma no máximo 10 pontos, e só quando os sinais visíveis já passam de 40.

**Nível.** Os sinais formam um nível de 0 a 100, com decaimento e histerese: amarelo a partir de 40, vermelho a partir de 70. O microssono e a cabeça caindo com os olhos fechados levam ao vermelho na hora. O alarme sonoro começa com volume de 20%. Os pesos do nível são escolhas de projeto documentadas, não ajustadas em sonolência real; a regra dos olhos e os ajustes finais foram escolhidos nos dados da validação, por isso os números dela são de desenvolvimento.

**Sem internet.** As dependências que viriam da internet (MediaPipe com os WASM, o `face_landmarker.task`, TensorFlow.js e tfjs-tflite) ficam em `app/pwa/vendor/`, nas versões fixas do `app.js`. O app usa essa cópia primeiro e a CDN só como reserva.

**Validação.**
- Offline: o mesmo `engine.js`, nas 16 sessões s5 do DMD (detecção, com anotação) e nas 17 sessões s1 (falso alarme em motoristas acordados, direção real).
- No navegador: o app, rodando no Chrome headless mudo, recebe 14 trechos reais pela câmera falsa: DMD s5, UTA-RLDD alerta e sonolento, e NITYMED.

## 7. Resultados

### 7.1 Exp. 1 (`results/exp1_intra.md`, `results/exp1_intra.csv`; log `results/logs/exp1_20260920-182858.log`)

3 classes, % (média ± dp populacional nos 5 folds):

| Modelo | BSA | Acur. bal. | F1 macro | AUC | VA |
|---|---|---|---|---|---|
| Acaso (classe majoritária) | 44,4 ± 2,4 | n/a | n/a | n/a | 33,3 |
| LogReg | 49,4 ± 4,3 | 49,7 ± 3,0 | 48,5 ± 3,6 | 68,9 ± 5,5 | 50,7 ± 7,6 |
| SVM-RBF | 53,0 ± 3,8 | 56,2 ± 4,5 | 54,2 ± 3,9 | 69,9 ± 5,1 | 54,0 ± 4,0 |
| RandomForest | 54,5 ± 9,3 | 57,8 ± 7,9 | 56,0 ± 9,1 | 73,5 ± 6,8 | 57,3 ± 7,2 |
| LSTM (50 ép.) | 50,8 ± 8,9 | 53,1 ± 6,7 | 51,8 ± 7,8 | 68,6 ± 8,2 | 57,2 ± 7,5 |
| CNN1D (50 ép.) | 48,0 ± 6,8 | 51,4 ± 5,2 | 50,0 ± 6,4 | 68,4 ± 6,1 | 55,0 ± 8,2 |
| Ref. HM-LSTM (Ghoddoosian et al., 2019, Tab. 2) | 54,0 | n/a | n/a | n/a | 65,2 |
| Ref. LSTM (idem) | 52,8 | n/a | n/a | n/a | 61,4 |
| Ref. humanos (idem) | n/a | n/a | n/a | n/a | 57,8 |

VA por fold (RF: 54,5 / 69,4 / 50,0 / 61,1 / 51,5; LSTM: 45,5 / 63,9 / 55,6 / 66,7 / 54,5; HM-LSTM:
64 / 61 / 70 / 64 / 67) mostra variância entre folds da ordem de 15 a 20 pp, própria de 58 sujeitos.
Esquemas binários (BSA / VA do RF): B1 87,4 / 82,2 (acaso por sequência 81,4); B2 66,5 / 70,1 (LogReg
VA 73,6); B3 86,9 / 85,5 (CNN1D BSA 85,2, AUC 93,8). Sensibilidade a épocas: LSTM com 15 épocas, BSA 50,6 / VA
52,7; CNN1D com 15 épocas, 49,8 / 55,1. Com 50 épocas, a acurácia de treino chega a cerca de 0,90 a 0,93, enquanto a BSA
em sujeitos não vistos fica em cerca de 0,50 (sobreajuste aos sujeitos do treino). Modelo exportado: LSTM do fold 3
(`models/blink_temporal.tflite`, 101,8 KB, 5.324 parâmetros; latência no Mac, p50 de 0,033 ms).

**Leitura.** O RF empata com o HM-LSTM na unidade por sequência (54,5 × 54,0), mas fica 7,9 pp abaixo por
vídeo (57,3 × 65,2), no nível do julgamento humano (57,8). O artigo original usa uma rede treinada
de ponta a ponta com regressão dos níveis KSS e votação diferente; a diferença por vídeo é coerente
com a literatura de sonolência real e com a instabilidade dos modelos temporais pequenos.

### 7.2 Exp. 2 (`results/exp2_leakage.md`, `results/exp2_leakage_plot.csv`; log `results/logs/exp2_20260920-183128.log`)

3 classes, % (média ± dp entre 3 sementes em (i) e (ii) e entre os 5 folds em (iii)):

| Partição | RF BSA | RF acur. bal. | RF AUC | RF VA | LSTM BSA | LSTM VA | vídeos / sujeitos do teste vistos no treino |
|---|---|---|---|---|---|---|---|
| (i) por janela | 99,5 ± 0,0 | 99,6 ± 0,0 | 100,0 | n/a | 87,3 ± 0,7 | n/a | 97,1 % / 100 % |
| (ii) por vídeo | 47,2 ± 1,4 | 52,5 ± 0,7 | 67,9 | 52,8 ± 1,2 | 45,1 ± 0,6 | 46,2 ± 2,3 | 0 % / 98,3 % |
| (iii) por sujeito | 54,2 ± 8,7 | 57,4 ± 7,5 | 73,4 | 55,7 ± 6,6 | 49,2 ± 8,8 | 50,8 ± 13,7 | 0 % / 0 % |
| Inflação (i) − (iii) | +45,3 pp | +42,1 pp | +26,6 pp | n/a | +38,1 pp | n/a | |
| (ii) − (iii) | −7,0 pp | −5,0 pp | −5,5 pp | −2,8 pp | −4,1 pp | −4,6 pp | |

No cenário do Exp. 5b (`results/exp5b_robustez.md`, seção 4; janelas de 10 s, B3), com as mesmas características:

| Modelo | Características | Acur. bal. por sujeito | Acur. bal. com partição aleatória | Inflação |
|---|---|---|---|---|
| RF | centradas | 74,2 | 86,8 | +12,6 pp |
| RF | brutas | 62,9 | 77,5 | +14,7 pp |
| RL | centradas | 72,0 | 73,1 | +1,2 pp |
| RL | brutas | 68,1 | 69,0 | +0,9 pp |

**Leitura.**
- O que infla é o mesmo vídeo nos dois lados. Na partição por janela, sequências vizinhas dividem 28 das 30 piscadas, então o teste vê quase as mesmas sequências do treino, e o RF vai de 54,2% a 99,5%.
- Reconhecer a pessoa não infla nada aqui. Na partição por vídeo, a pessoa aparece no treino com os outros dois vídeos, e o resultado fica *abaixo* da partição por sujeito (−7,0 pp). Uma explicação provável, que não isolamos, é que o modelo associa a aparência da pessoa a rótulos que não são os do vídeo de teste (cada sujeito tem exatamente um vídeo por classe).
- O vazamento pesa nos modelos flexíveis. No cenário do Exp. 5b, a partição aleatória infla o RF em 12,6 pp, efeito da mesma ordem da troca do treino real pelo atuado (9,0 pp de acurácia balanceada), e quase não muda a RL (1,2 pp).
- Não é a explicação dos 90% da literatura. Os 90% do NTHU-DDD foram medidos com outras pessoas no teste; a distância até a UTA-RLDD não se explica por esse vazamento. A parte atuado → real é medida no Exp. 5b (seção 7.6).

### 7.3 Exp. 3 (`results/exp3_eye_state.md`, `results/exp3_eye_state.csv`, `results/exp3_strat_conditions.csv`; log `results/logs/exp3_eye_state_train.log`)

Acurácia média ± dp (ddof=1) de 5 folds, %:

| Treino → Teste | Acurácia | Acur. bal. | AUC | Acaso |
|---|---|---|---|---|
| A: MRL → MRL (sujeitos não vistos) | 94,1 ± 3,4 | 94,2 ± 3,5 | 98,6 ± 1,3 | 60,3 |
| B: MRL → CEW patch (5 folds) | 79,0 ± 8,5 | 78,7 ± 8,6 | 93,2 ± 4,7 | 50,8 |
| B: MRL → CEW crop (5 folds) | 80,6 ± 10,3 | 80,3 ± 10,5 | 93,5 ± 4,9 | 50,8 |
| B: modelo final (37 sujeitos) → CEW patch / crop | 85,7 / 90,0 | 85,5 / 89,9 | 96,8 / 97,6 | 50,8 |
| C: CEW patch → MRL (5 folds) | 68,1 ± 1,9 | 68,4 ± 1,9 | 84,2 ± 2,8 | 50,6 |
| C: CEW crop → MRL (5 folds) | 67,1 ± 1,8 | 67,4 ± 1,8 | 83,6 ± 3,1 | 50,6 |
| D: CEW → CEW patch / crop (faces não vistas) | 94,7 ± 1,1 / 94,5 ± 1,3 | 94,7 / 94,4 | 98,9 / 98,8 | 50,8 |

Por condição (PP3), acurácia / acurácia balanceada:

| Condição do MRL (n) | A: MRL → MRL (fora da amostra) | C: CEW patch → MRL (média dos 5 modelos) |
|---|---|---|
| com óculos (24.001) | 88,8 / 88,5 | 65,2 / 61,1 (AUC 65,1) |
| sem óculos (60.897) | 96,1 / 96,0 | 69,3 / 71,8 (AUC 93,4) |
| iluminação ruim (53.630) | 94,4 / 94,3 | 67,1 / 63,8 |
| iluminação boa (31.268) | 93,5 / 92,6 | 69,9 / 75,1 |
| reflexo grande (12.709) | 86,7 / 84,2 | 62,5 / 65,7 |
| sem reflexo (66.227) | 95,4 / 95,5 | 67,7 / 64,8 |
| sensor RealSense / IDS / Aptina | 93,5 / 96,4 / 98,3 (bal. 93,5 / 91,9 / 93,7) | bal. 60,9 / 84,3 / 97,1 |

Referência (só o fold 1, `results/exp3_mobilenetv3.csv`): MobileNetV3-Small 0,75 (583.593 parâmetros)
95,9 % / AUC 99,2, contra EyeNet 91,2 % / 97,3 nos mesmos sujeitos de teste.

**Leitura.** Dentro de cada base, a tarefa é quase resolvida nos dois domínios (de 94 % a 95 %). A transferência é
assimétrica: MRL → CEW perde de 4 a 15 pp (e o modelo com todos os 37 sujeitos recupera parte: 90,0 % no
crop), enquanto CEW → MRL perde cerca de 27 pp: 2.423 faces RGB de baixa resolução não cobrem a variedade de
sensores infravermelhos, óculos e reflexos do MRL. Óculos é a condição dominante: no sentido CEW → MRL, a
AUC despenca de 93,4 (sem) para 65,1 (com). O fold 3 de A é o pior tanto dentro do MRL (90,1 %) quanto
MRL → CEW (66,0 / 64,7 %), o que explica o desvio de 8 a 10 pp no sentido MRL → CEW. Os estratos IDS e Aptina
têm 95,2 % e 86,4 % de olhos abertos; por isso a acurácia balanceada é a métrica reportada neles.

### 7.4 Exp. 4: custo no celular (`results/exp4_tflite_eval.csv`, `results/exp4_desktop_latency.csv`, `results/exp4_phone_latency.md`)

Queda causada pela quantização (o conjunto MRL não é teste cego para o modelo final; serve só para a diferença):

| Conjunto | n | Keras acc | fp32 acc | int8 acc | int8 − Keras | Keras AUC | int8 AUC |
|---|---|---|---|---|---|---|---|
| MRL completo | 84.898 | 98,5 | 98,5 | 98,5 | −0,03 pp | 99,9 | 99,9 |
| CEW patch (cego) | 4.846 | 85,7 | 85,7 | 85,6 | −0,10 pp | 96,8 | 96,3 |
| CEW crop (cego) | 4.846 | 90,0 | 90,0 | 89,7 | −0,31 pp | 97,6 | 97,2 |

Tamanho e latência em lote 1 no desktop (`results/exp4_desktop_latency.csv`; Mac arm64, TF 2.21, 1 thread, p50 / p95):

| Modelo | Tamanho | p50 / p95 |
|---|---|---|
| EyeNet fp32 | 248,2 KB | 0,067 / 0,070 ms |
| EyeNet int8 (E/S uint8) | 72,1 KB | 0,060 / 0,064 ms |
| LSTM de piscadas fp32 | 101,8 KB | 0,033 / 0,045 ms |

Com 4 threads, a latência *aumenta*: os modelos são tão pequenos que o custo de chamada domina. O número do desktop não deve ser extrapolado para o celular.

**Celular.** A tabela por execução (aparelho, sistema, navegador, versão do app, quadros por segundo e p50 e p95 de cada estágio) é gerada por `tools/import_phone_benchmark.py` a partir dos CSV brutos por quadro de `results/phone/`: ver `results/exp4_phone_latency.md`.

A versão 1.0 deu falso alarme. Numa execução de 27/09, com um dos autores acordado diante da câmera, a lógica antiga acusou PERCLOS de 25,8% e manteve o alarme ligado em 63,4% dos quadros (`results/phone/benchmark_pwa_2026-09-27T22-41-35-365Z.json`). Isso motivou a reescrita do detector (seção 7.8).

### 7.5 Exp. 5a: validação do extrator (`results/exp5a_validacao.md`)

47 sujeitos com par confirmado e 141 vídeos. Diferença oficial − nosso, com IC 95% por bootstrap sobre sujeitos; positivo = o nosso perde:

| Comparação | Oficial | Nosso | Diferença (pp) | p (permutação) |
|---|---|---|---|---|
| Custo de trocar o extrator, 3 classes, acur. bal. | 58,9 | 57,0 | +2,0 [−4,8; +8,5] | 0,555 |
| Idem, VA | 58,9 | 56,7 | +2,1 [−6,4; +9,9] | 0,734 |
| Idem, B3, acur. bal. | 85,5 | 85,0 | +0,5 [−6,4; +7,2] | 0,904 |
| Mesmo extrator no treino e no teste (5 folds), B3, acur. bal. | 83,1 | 89,8 | −6,7 [−12,9; −1,8] | 0,029 |
| Idem, 3 classes, acur. bal. | 54,6 | 56,3 | −1,7 [−10,4; +5,7] | 0,712 |

**Leitura.** O custo de trocar o extrator não se distingue de zero. Com cada extrator no próprio treino e teste, o nosso é melhor em B3 nos 5 folds, mas, deixando um sujeito de fora por vez, a diferença não se distingue de zero (−4,5 [−11,0; 0,6]); em 3 classes, ela também não se distingue de zero. O IC largo não prova equivalência, mas não há perda detectável ao trocar de extrator.

As variantes com bordas da piscada no limiar custam mais (+9,3 pp), o que confirma a escolha da variante "estendida", fixada antes do experimento.

Duas pastas da UTA-RLDD não têm sujeito oficial (F1_12, F5_49). Por isso os `.npy` têm 58 sujeitos.

### 7.6 Exp. 5b: transferência DMD → UTA-RLDD, PP1 (`results/exp5b_transferencia.md`, `results/exp5b_robustez.md`)

Esquema B3 (KSS 0 × 10), variante principal. AUC × 100 e acurácias em %, com IC 95% por bootstrap sobre sujeitos:

| Métrica | Modelo | (A) DMD → DMD | (B) UTA → UTA | (C) DMD → UTA | Queda PP1 (B − C), pp |
|---|---|---|---|---|---|
| AUC por janela | RF | 82,4 [73,6; 89,7] | 80,1 [76,3; 83,8] | 69,2 [63,7; 74,3] | 10,9 [6,7; 15,7] |
| AUC por janela | RL | 83,5 [76,1; 88,9] | 74,4 [69,7; 78,4] | 64,5 [58,1; 70,9] | 9,8 [3,7; 16,0] |
| Acur. bal. por janela (0,5) | RF | 76,4 [68,3; 83,7] | 74,2 [70,9; 77,6] | 65,2 [61,7; 68,6] | 9,0 [5,8; 12,2] |
| AUC por vídeo | RF | n/a | 92,4 [87,9; 96,2] | 80,5 [73,0; 87,2] | 11,9 [5,6; 18,8] |
| AUC por vídeo | RL | n/a | 89,0 [82,2; 94,9] | 76,0 [67,8; 84,7] | 13,0 [5,8; 20,8] |
| Acurácia por vídeo (0,5) | RF | n/a | 85,8 [80,0; 91,7] | 73,3 [67,5; 79,2] | 12,5 [5,0; 19,2] |

Amostra:
- DMD: 510 janelas (87 alertas, 423 sonolentas) de 13 sujeitos;
- UTA B3: 12.487 janelas avaliáveis e 120 vídeos lógicos de 60 sujeitos.

Controles da queda B − C da AUC por janela (Tabela II do artigo; pontos de AUC, IC 95%). Na configuração principal, a referência da UTA-RLDD é o último terço do vídeo alerta:

| Variação | RF | RL |
|---|---|---|
| Principal | 10,9 [6,7; 15,7] | 9,8 [3,7; 16,0] |
| IC com os motoristas do DMD reamostrados (C treinado de novo) | [5,7; 19,5] | [1,9; 18,1] |
| Referência no vídeo de pouca vigilância | 2,3 [−2,5; 6,8] | 7,6 [3,0; 12,5] |
| Sem centragem | 2,1 [−1,2; 5,8] | 7,3 [3,0; 11,7] |
| Contraste KSS 8 e 9 × 6 e 7 (dois vídeos fora da referência) | 1,2 [−3,0; 5,8] | 4,6 [−0,2; 9,7] |
| B treinado com 13 sujeitos (mediana de 20 sorteios, sem IC) | 9,3 | 7,0 |

**Leitura.**
- A RL perde em todas as formas de normalizar: de 5,4 a 9,8 pontos de AUC por janela, sempre com IC acima de zero.
- No RF, a queda depende da normalização. Na UTA-RLDD, as janelas alertas saem do mesmo vídeo usado como referência, e o RF treinado nela aproveita essa semelhança. Com a referência no vídeo de pouca vigilância ou sem centragem, a AUC de B no RF cai de 80,1 para 68,4 e 69,4, e a queda fica de 2,1 a 2,3, sem diferença detectável. Num app, a calibração e a sonolência acontecem na mesma sessão, e essa vantagem de B não se repetiria.
- O achado mais firme é a intensidade. O modelo do DMD erra sobretudo ao chamar de alerta a sonolência real: reconhece 82,0% das janelas alertas e só 48,4% das sonolentas no limiar 0,5; no limiar calibrado no DMD (0,662), 88,7% e 37,0%. Nas janelas sonolentas, o maior fechamento mediano é de 0,97 s no DMD e de 0,17 s na UTA-RLDD, e o PERCLOS mediano, de 0,208 contra 0,040. A sonolência atuada é mais extrema, em parte por construção: as janelas sonolentas do DMD são as que contêm um evento anotado.
- O volume do treino explica pouco. Com B treinado em só 13 sujeitos da UTA-RLDD por fold, como no DMD, a AUC de B fica em 78,6 (RF) e 71,5 (RL), e a queda continua (9,3 e 7,0).
- Reamostrar também os motoristas do DMD alarga o IC da queda principal, que continua acima de zero.
- Tirar o alerta do DMD da direção em movimento (s1) piora a transferência (AUC de C de 69,2 para 58,6).
- Sensibilidades. Nas variações de janela (5 e 15 s), de alerta e de referência no DMD e na exclusão do motorista em que o detector falhou, o IC da queda do RF fica acima de zero; contando a principal e a sem centragem, são 7 de 8 variantes.
- O bocejo não transfere: discrimina no DMD e quase não aparece na UTA-RLDD.

### 7.7 Exp. 6: noite, PP3 (`results/exp6_noite.md`)

| Base / grupo | Detecção de rosto, média por vídeo [IC 95%] | SNR rápida do EAR, mediana [IC 95%] | Ruído rápido |
|---|---|---|---|
| NITYMED (noite, carro em movimento) | 0,9915 [0,9875; 0,9948] | 7,2 [6,4; 7,9] | 0,0084 |
| UTA-RLDD | 0,9977 [0,9940; 0,9996] | 21,5 [18,3; 23,5] | 0,0037 |
| DMD s5 (dia, parado, encenado) | 0,9978 [0,9958; 0,9995] | 23,1 [16,1; 27,9] | 0,0046 |
| DMD, direção segura, carro parado | 0,9989 [0,9977; 0,9997] | 13,6 [12,0; 16,9] | 0,0059 |
| DMD, direção segura, carro em movimento | 0,9965 [0,9935; 0,9987] | 8,7 [6,2; 10,8] | 0,0097 |

Comparações da SNR rápida:
- Noite − DMD em movimento: −1,6 [−3,7; +1,1], com IC que inclui zero.
- No mesmo motorista do DMD, o movimento muda a SNR rápida em −5,4 [−6,6; −3,5] (11 sujeitos).

Taxa de disparo no limiar 0,5, com características brutas: % das janelas de 10 s marcadas como sonolentas, com IC 95%.

| Detector | UTA KSS 0 (acordados) | DMD parado (alerta) | DMD em movimento (alerta) | UTA KSS 10 | NITYMED microssono | NITYMED bocejo |
|---|---|---|---|---|---|---|
| RF do DMD (Exp. 5b) | 26,0 [21,1; 31,1] | 45,7 | 42,9 | 52,2 | 44,7 | 73,0 |
| RF da UTA-RLDD (B3) | 43,6 [38,7; 48,8] | 69,1 | 77,1 | 69,4 | 83,5 | 89,5 |
| Regra por sinais (fechamento de 0,5 s ou mais ou bocejo) | 2,6 | 14,9 | 23,8 | 24,1 | 15,0 | 61,7 |

Regra × modelos no mesmo ponto de operação (`results/exp5b_robustez.md`, seção 5). Para cada modelo, o limiar é o que dá o mesmo falso alarme da regra (2,6%) nas janelas alertas do B3; esse limiar usa o rótulo do teste, então a tabela é só um diagnóstico:

| Detector | Janelas sonolentas encontradas (%) |
|---|---|
| Regra (fechamento de 0,5 s ou mais ou bocejo) | 24,1 |
| RL do DMD (C), características centradas | 24,7 |
| RF do DMD (C), características centradas | 16,4 |
| RL do DMD (C), características brutas | 19,2 |
| RF do DMD (C), características brutas | 15,3 |
| RL da UTA-RLDD (B), características centradas | 27,3 |
| RF da UTA-RLDD (B), características centradas | 24,2 |

**Leitura.**
- A perda noturna está no sinal, não na captura. A queda na detecção de rosto é pequena; o que cai é a qualidade do sinal do EAR. Uma perda do mesmo tamanho já aparece de dia com o carro em movimento, no mesmo motorista. Com estes dados, a perda de sinal acompanha a condução real e não pode ser atribuída à escuridão por si só.
- Os modelos leem ruído como sonolência. O modelo da UTA lê o ruído do EAR como sonolência: se o `ear_dp` fosse o típico do alerta, o disparo nos microssonos cairia de 83,5% para 64,5%.
- A regra não é mais precisa que os modelos. No limiar 0,5, os RF disparam em muitas janelas de pessoas acordadas, mas esse limiar é arbitrário. Com o mesmo falso alarme da regra, a RL do DMD centrada acha tantas janelas sonolentas quanto ela; os modelos com características brutas acham menos. A regra fica na mesma curva, sem treino e com cada alarme explicável.
- Nenhum dos dois resolve o carro em movimento. Lá, com o limiar calibrado no treino, o RF do DMD dispara em 21,9% das janelas de motoristas acordados, e a regra, em 23,8%.
- Sem classe alerta noturna, não há acurácia. Nenhum limiar pode ser calibrado para a noite.
- O NITYMED não entra na PP1. A autenticidade dos comportamentos dele não é documentada.

### 7.8 App de demonstração, PWA 2.1 (`results/app_validacao_dmd.md`, `results/app_validacao.md`, `app/pwa/README.md`)

| Medida | Resultado | Onde |
|---|---|---|
| Falso alarme, direção real com motoristas acordados (DMD s1, 17 sessões, 63,3 min) | 3,3% do tempo em vermelho; 16 de 17 sessões sem alarme. O caso restante é gA_5 (55,8% em vermelho): olhar para baixo é lido como olho fechado | motor offline |
| Mesma medida com a lógica 1.0 (PERCLOS; sem LSTM) | 55,6% do tempo em vermelho | motor offline |
| UTA-RLDD, vídeos alertas (3 pessoas, 3 min cada) | 100% do tempo em verde, 0 alarmes, 0 avisos | navegador |
| Estado do olho por quadro (DMD s5) | kappa 0,938 (offline, 16 sessões) e 0,902 (navegador, 4 sessões) | ambos |
| Fechamentos de 1 s ou mais (DMD s5) | revocação 0,920 e precisão 0,748 (offline); 31 de 34 no navegador | ambos |
| Bocejo (DMD s5) | revocação 0,638 sem a mão e 0,026 com a mão na boca (offline); 6 de 24 no navegador, nenhum com a mão | ambos |
| UTA-RLDD, vídeos sonolentos (3 pessoas) | só 1 de 3 chegou ao vermelho (Fold3/31: 93,5% do tempo); os outros dois ficaram 100% verdes | navegador |
| Cabeça | sem revocação medida: o DMD não anota cabeceio. Há só o teste sintético (0 eventos falsos com a cabeça parada) e casos isolados | navegador |

**Leitura.**
- O app mede sinais visíveis, não o estado interno. Quem relata sono sem mostrar sinais não é acusado.
- A regra foi escolhida por ser explicável. No mesmo falso alarme, ela acha tantas janelas sonolentas quanto a RL do DMD (seção 7.7), sem treino e com o porquê de cada alarme.
- Os pesos do nível são escolhas de projeto, não ajustes em dados de sonolência real, e a validação usa os mesmos dados em que as regras foram escolhidas.
- A amostra do navegador é pequena (cerca de 40 min).
- O app funciona sem internet com `app/pwa/vendor/` completa; com internet, a biblioteca do MediaPipe envia telemetria de uso a um servidor do Google depois de cerca de 60 s, sem quadros de vídeo (`README.md`, seção 9).
- Limites conhecidos: `app/pwa/README.md`, seção 10.

## 8. Respostas às perguntas de pesquisa

- PP1. Sim, há queda, e ela não vem do extrator (Exp. 5a). Treinar em sonolência encenada custa de 5,4 a 9,8 pontos de AUC por janela na RL em todas as formas de normalizar; no RF, a perda de 10,9 da configuração principal depende da referência da normalização e some com outra referência ou sem centragem (Exp. 5b).
  - O achado mais firme é a intensidade: o modelo encenado reconhece a maioria das janelas alertas reais, mas só 48,4% das sonolentas. A causa mais provável é a intensidade: ele aprende com eventos extremos, e a sonolência real tem fechamentos curtos e PERCLOS baixo.
  - O volume do treino explica pouco (B com 13 sujeitos mantém a queda). Com um único par de bases, a queda ainda se mistura com a câmera, o cenário e a natureza do rótulo.
  - O vazamento por janelas do mesmo vídeo infla muito a acurácia (+45,3 pp no Exp. 2; +12,6 pp no RF no cenário do Exp. 5b), mas isso não explica os 90% do NTHU-DDD, medidos com outras pessoas no teste.
- PP2. O custo cabe num celular atual. Os modelos próprios são pequenos (72,1 KB a CNN int8 e 101,8 KB o LSTM), e o int8 custa no máximo 0,31 pp de acurácia. Tempo por quadro no celular: ver `results/exp4_phone_latency.md`. Não medimos bateria, aquecimento em uso longo nem aparelhos de entrada.
- PP3.
  - Óculos: é a condição que mais derruba o estado ocular entre domínios (Exp. 3).
  - Noite: a perda aparece no sinal do EAR, e uma perda do mesmo tamanho aparece de dia com o carro em movimento (Exp. 6). A noite não pode ser isolada com as bases públicas disponíveis.
  - Óculos escuros ficam sem cobertura em dado real.

## 9. Cronograma: planejado × realizado e o que mudou

As datas da coluna "Quando" saem dos logs de `results/logs/` (o nome de cada log traz a data e a hora da execução) e, para o celular, dos arquivos de `results/phone/`.

| Fase (plano da Entrega 01) | Previsto | Realizado | Quando | Situação |
|---|---|---|---|---|
| F0 Planejamento e baseline | 03/08 a 05/09 | plano, pôster, repositório com a baseline (RF, BSA 54,7 ± 9,0) e demo de webcam | baseline em 24/08, repetida em 26/08 (`baseline_20260824-212550.log`, `baseline_20260826-090734.log`); último push em 26/08 | concluída |
| F1 Preparação dos dados | 31/08 a 27/09 | UTA-RLDD completa (características oficiais e 182 vídeos), DMD (sonolência e distração), NITYMED, MRL Eye, CEW; sinais por quadro das três bases de vídeo | caches do MRL e do CEW em 20/09 (`exp3_cache_build.log`); sinais e características em 27/09 (`extract_signals_*.log`, `exp5_final_run_features.log`) | concluída no prazo, com troca de base |
| F2 Pipeline A e Exp. 1 | 14/09 a 11/10 | Exp. 1 (5 modelos, 4 esquemas, BSA e VA); extrator próprio validado (Exp. 5a); janelas de 10 s | Exp. 1 em 20/09 (`exp1_20260920-182858.log`); Exp. 5a em 27/09 (`exp5a_validacao_20260927-221239.log`) | concluída antes do prazo |
| F3 Pipeline B | 28/09 a 25/10 | EyeNet (62.865 parâmetros) no MRL e no CEW, por óculos, iluminação, reflexo e sensor; MobileNetV3 como referência | 20/09 (`exp3_eye_state_train.log`, `exp3_mobilenetv3.log`) | concluída antes do prazo |
| F4 Exp. 2 e 3 do plano (treino atuado, teste real, por condição) | 12 a 25/10 | Exp. 5b (DMD → UTA-RLDD) com 8 sensibilidades; Exp. 6 (noite); Exp. 2 (vazamento) e Exp. 3 (MRL ↔ CEW) como complementares; controles da queda do Exp. 5b | Exp. 2 em 20/09 (`exp2_20260920-183128.log`); Exp. 5b e 6 em 27/09 (`exp5b_transferencia.log`, `exp6_final_noite.log`); controles em 26/10 (`exp5b_robustez_20261026-2051.log`) | concluída em 27/09; os controles, pedidos pelas revisões simuladas, entraram em 26/10 |
| F5 Exp. 4 e consolidação | 12/10 a 08/11 | int8 e desktop; PWA 2.1 validada; app sem internet; medição do app 2.1 nos celulares; números conferidos | Exp. 4 desktop em 20/09 (`exp4_export_tflite.log`); validação do app em 27/09 (`app_validacao_dmd_20260927-225227.log`, `app_validacao_20260927-232519.log`, `pwa_test_engine_20260927-231003.log`); app 1.0 no iPhone em 27/09 (`results/phone/`); números refeitos em 28/09 e 06/10 (`atualizar_celular_*.log`) e em 26/10 (`results/numeros_resumo.md`); app 2.1 nos celulares em 26/10 (`results/exp4_phone_latency.md`) | dentro do prazo |
| F6 Comunicação | 28/09 a 16/11 | artigo e pôster reescritos para DMD, NITYMED, Exp. 5a, 5b e 6 e app 2.1; artigo com 7 páginas em 06/10 e reescrito depois da banca simulada em 26/10 (8 páginas); pôster de 06/10, refeito com os números novos por `bash tools/atualizar_celular.sh` | `atualizar_celular_20260928_014152.log`, `atualizar_celular_20261006_161528.log`; `paper/numeros.tex` de 26/10 | em andamento: falta a leitura final da equipe e do orientador |

**O que mudou em relação ao planejamento** (justificativa completa em `docs/TROCA_DATASETS.md`):

1. Base atuada: NTHU-DDD → DMD, e noite pelo NITYMED. O NTHU-DDD não foi obtido: o termo exige assinatura de chefia institucional, e o prazo era 20/09. A PP1 passou a ter experimento direto (Exp. 5b), além do contraste com a literatura e da inflação medida no Exp. 2.
2. Janela: 30 s → 10 s (passo de 5 s mantido). Os trechos de direção segura do DMD duram de 10 a 26 s. A predição por vídeo continua sendo a média dos escores das janelas, como no plano.
3. Aparelho: além do Galaxy S23 Ultra, o iPhone 17 Pro. O Galaxy é de um colega e só pode ser medido na PUC; o iPhone é de um dos autores. Execuções e versão do app: ver `results/exp4_phone_latency.md`.
4. Exp. 3 = estado ocular MRL ↔ CEW, e não mais o NTHU-DDD por condição. A CNN própria substituiu a MobileNetV3 como modelo principal, por ser 9× menor e exportável para int8.
5. App: a lógica de alarme mudou. A 1.0 (PERCLOS e LSTM) deu falso alarme com uma pessoa acordada e ficou 55,6% do tempo em vermelho na direção real do DMD. A 2.1 alarma por sinais visíveis e funciona sem internet. O app Android nativo mantém a 1.0 e não foi compilado.
6. Ambiente: dois ambientes Python (`.venv` e `.venv-mp`). O MediaPipe que funciona neste Mac é incompatível com o TensorFlow.
7. Leitura da PP1 (26/10). Os controles do Exp. 5b mostraram que a queda do RF depende da referência da normalização. O resultado principal passou a ser a perda da RL em todas as normalizações e a diferença de intensidade entre o sono encenado e o real.

## 10. Riscos e próximos passos até 16/11

| Risco | Impacto | Mitigação |
|---|---|---|
| O orientador não aprovar a troca de bases | artigo e pôster precisariam de outro enquadramento | resolvido: o orientador autorizou a troca verbalmente (registrado em 04/10/2026; `docs/TROCA_DATASETS.md`) |
| Medição curta no celular (execuções de 60 s, aparelhos de topo) | conclusão restrita na PP2 | reportar mediana e extremos entre as execuções, sem extrapolar; o artigo diz que não medimos bateria, aquecimento em uso longo nem aparelhos de entrada |
| Rede da PUC isola os aparelhos ou não tem internet | demonstração sem app | Acesso Pessoal do iPhone com o Mac e o celular; app sem internet com `app/pwa/vendor/` (conferir na véspera com `bash tools/baixar_dependencias_app.sh --conferir`) |
| Artigo passar de 8 páginas | fora da regra | hoje está com 8: qualquer acréscimo precisa de um corte; conferir as páginas depois de cada compilação |
| Artigo ou pôster voltarem a divergir dos resultados depois de alguma nova rodada | inconsistência visível | números só por macros; depois de qualquer mudança em `results/`, rodar `bash tools/atualizar_celular.sh` (refaz números, artigo e pôster e confere cada número) |
| Rostos das bases no repositório público | violação de licença | `.gitignore` exclui dados e a figura interna com rostos do DMD; a figura de amostras só tem o MRL Eye; conferir `git status` antes do push |
| `results/exp5b_predicoes.csv` (32 MB) no GitHub | repositório pesado | abaixo do limite de 100 MB por arquivo; é a fonte dos IC do Exp. 5b. Mantido versionado |
| Som do alarme no iPhone (política de áudio do iOS) | demonstração sem som | "Testar som" antes, volume de mídia no máximo (`docs/GUIA_EQUIPE.md`, seção 8) |
| Telemetria do MediaPipe com internet | dúvida sobre privacidade | nota no `README.md` (seção 9): sem quadros de vídeo; sem internet, nada sai do aparelho |

Próximos passos, com responsável sugerido conforme a divisão do plano:

1. Feito (04/10). O orientador autorizou a troca de bases verbalmente; registro em `docs/TROCA_DATASETS.md`.
2. Feito (26/10). Controles do Exp. 5b, reescrita do artigo depois da banca simulada e app sem internet.
3. 26/10. Medição do app 2.1 no iPhone 17 Pro e no Galaxy S23 Ultra; depois, `bash tools/atualizar_celular.sh` refaz tabela, figura, números, artigo e pôster (Bruno Rodolfo, com o André). Passo a passo: `docs/GUIA_EQUIPE.md`, seção 1.
4. Até 31/10 (Entrega 02). Este relatório lido pelos quatro e o PDF (`docs/ENTREGA02_relatorio.pdf`) enviado no Canvas; commit, tag `entrega02` e push, depois do "ok" do André (`docs/GUIA_EQUIPE.md`, seção 6) (Bruno Rodolfo).
5. Até 09/11 (artigo e pôster). Leitura final do artigo pelos quatro e pelo orientador; conferir as 8 páginas e as citações obrigatórias do NITYMED e do DMD depois de cada edição; pôster final com teste de legibilidade impresso (Paulo, com revisão de todos).
6. De 09 a 12/11. Revisão por pares de 2 artigos de outros grupos, em duplas, com o modelo de parecer de `docs/GUIA_EQUIPE.md`, seção 5; críticas ao nosso artigo incorporadas até 14/11 (todos).
7. Até 15/11. Dois ensaios da demonstração sem internet e do pitch de 60 a 90 s (`docs/GUIA_EQUIPE.md`, seções 2 e 3), com o volume em 100% só na hora (André e Bruno Sandoval).
8. 16/11. Sessão de pôsteres.
9. Trabalho futuro (registrado no artigo): repetir a transferência com outra base de sonolência real, como o DROZY, e com a referência gravada na mesma sessão; validar o app em campo, com voluntários, KSS, óculos e noite; medir a revocação da queda da cabeça. Fora do artigo: anotar óculos nos vídeos da UTA-RLDD (21 de 180 têm óculos, sem lista) e compilar o app Android com a lógica 2.1.
