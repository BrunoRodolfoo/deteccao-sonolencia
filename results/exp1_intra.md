# Experimento 1 — UTA-RLDD intra-conjunto (5 folds oficiais, divisão por sujeito)

Execução 20260920-182858. Seed 42. Modelos temporais treinados por 50 épocas fixas (sem early stopping; o teste nunca é consultado). Valores em % (média ± desvio populacional, ddof=0, nos 5 folds). Dados brutos: `results/exp1_intra.csv`; log: `results/logs/exp1_20260920-182858.log`.

Modelos clássicos: estatísticas 24d por sequência (média, desvio, mín., máx., mediana, IQR de cada característica) com os hiperparâmetros do baseline. Modelos temporais (Keras) sobre a sequência bruta (30 piscadas x 4): **LSTM**: {"units": 32, "dense": 16, "dropout": 0.2, "unroll": true, "optimizer": "Adam", "lr": 0.001, "batch": 64, "epochs": 50, "input_norm": "z-score (treino)"}; **CNN1D**: {"conv": [32, 64], "kernel": 3, "pool": "GAP", "dense": 16, "dropout": 0.2, "optimizer": "Adam", "lr": 0.001, "batch": 64, "epochs": 50, "input_norm": "z-score (treino)"}.

VA = acurácia por vídeo, voto majoritário das janelas do vídeo; empate resolvido pela maior probabilidade média entre as classes empatadas. Chance = classe majoritária do teste.

## Esquema 3c: 3 classes: alerta / pouca vigilância / sonolento

| Modelo | BSA | Acur. bal. | F1 macro | AUC macro | VA |
|---|---|---|---|---|---|
| Chance(majoritaria) | 44.4 ± 2.4 | — | — | — | 33.3 ± 0.0 |
| LogReg | 49.4 ± 4.3 | 49.7 ± 3.0 | 48.5 ± 3.6 | 68.9 ± 5.5 | 50.7 ± 7.6 |
| SVM-RBF | 53.0 ± 3.8 | 56.2 ± 4.5 | 54.2 ± 3.9 | 69.9 ± 5.1 | 54.0 ± 4.0 |
| RandomForest | 54.5 ± 9.3 | 57.8 ± 7.9 | 56.0 ± 9.1 | 73.5 ± 6.8 | 57.3 ± 7.2 |
| LSTM | 50.8 ± 8.9 | 53.1 ± 6.7 | 51.8 ± 7.8 | 68.6 ± 8.2 | 57.2 ± 7.5 |
| CNN1D | 48.0 ± 6.8 | 51.4 ± 5.2 | 50.0 ± 6.4 | 68.4 ± 6.1 | 55.0 ± 8.2 |

Referências (Ghoddoosian et al., 2019, Tabela 2, média dos 5 folds): HM-LSTM: BSA 54.0% / VA 65.2%; LSTM: BSA 52.8% / VA 61.4%; Camadas densas: BSA 52.0% / VA 57.0%; Humanos: BSA — / VA 57.8%.

### VA por fold (3 classes) — comparação com a Tabela 3 do artigo

| Modelo | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Média |
|---|---|---|---|---|---|---|
| LogReg | 57.6% | 36.1% | 55.6% | 52.8% | 51.5% | 50.7% |
| SVM-RBF | 57.6% | 55.6% | 50.0% | 58.3% | 48.5% | 54.0% |
| RandomForest | 54.5% | 69.4% | 50.0% | 61.1% | 51.5% | 57.3% |
| LSTM | 45.5% | 63.9% | 55.6% | 66.7% | 54.5% | 57.2% |
| CNN1D | 42.4% | 66.7% | 50.0% | 58.3% | 57.6% | 55.0% |
| HM-LSTM (Tabela 3 do artigo) | 64.0% | 61.0% | 70.0% | 64.0% | 67.0% | 65.2% |
| Humanos (Tabela 3 do artigo) | 62.0% | 59.0% | 60.0% | 53.0% | 55.0% | 57.8% |

### BSA por fold (3 classes)

| Modelo | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Média |
|---|---|---|---|---|---|---|
| LogReg | 44.8% | 46.1% | 48.2% | 56.8% | 50.8% | 49.4% |
| SVM-RBF | 55.5% | 58.4% | 53.2% | 50.8% | 47.3% | 53.0% |
| RandomForest | 46.7% | 72.6% | 51.9% | 49.0% | 52.3% | 54.5% |
| LSTM | 34.7% | 59.6% | 48.2% | 54.3% | 57.1% | 50.8% |
| CNN1D | 39.5% | 60.3% | 47.2% | 46.8% | 46.3% | 48.0% |

## Esquema B1: alerta {0} vs. não alerta {1,2}

| Modelo | BSA | Acur. bal. | F1 macro | AUC macro | VA |
|---|---|---|---|---|---|
| Chance(majoritaria) | 81.4 ± 2.8 | — | — | — | 66.7 ± 0.0 |
| LogReg | 78.5 ± 6.7 | 60.1 ± 5.5 | 60.3 ± 5.8 | 80.3 ± 6.5 | 70.9 ± 4.5 |
| SVM-RBF | 85.1 ± 2.9 | 78.0 ± 1.7 | 76.6 ± 2.4 | 89.9 ± 2.2 | 81.8 ± 6.4 |
| RandomForest | 87.4 ± 3.2 | 79.8 ± 4.3 | 79.4 ± 2.9 | 88.7 ± 5.1 | 82.2 ± 3.3 |
| LSTM | 81.1 ± 9.4 | 73.8 ± 4.9 | 72.2 ± 7.7 | 85.2 ± 7.9 | 77.8 ± 8.3 |
| CNN1D | 83.2 ± 8.5 | 75.6 ± 6.7 | 74.2 ± 7.4 | 87.5 ± 6.5 | 81.3 ± 6.8 |

## Esquema B2: não sonolento {0,1} vs. sonolento {2}

| Modelo | BSA | Acur. bal. | F1 macro | AUC macro | VA |
|---|---|---|---|---|---|
| Chance(majoritaria) | 59.2 ± 4.5 | — | — | — | 66.7 ± 0.0 |
| LogReg | 66.8 ± 6.5 | 63.4 ± 7.2 | 63.0 ± 8.1 | 66.9 ± 10.3 | 73.6 ± 2.3 |
| SVM-RBF | 66.2 ± 5.9 | 63.9 ± 7.1 | 63.7 ± 7.0 | 66.2 ± 12.1 | 69.4 ± 3.1 |
| RandomForest | 66.5 ± 9.2 | 63.4 ± 10.2 | 62.9 ± 10.5 | 70.8 ± 11.6 | 70.1 ± 4.1 |
| LSTM | 60.9 ± 8.6 | 59.3 ± 9.9 | 58.5 ± 9.5 | 63.7 ± 11.8 | 67.0 ± 7.7 |
| CNN1D | 61.2 ± 9.9 | 60.8 ± 11.1 | 59.7 ± 10.4 | 65.2 ± 12.9 | 66.5 ± 6.7 |

## Esquema B3: alerta {0} vs. sonolento {2}, descartando a classe intermediária

| Modelo | BSA | Acur. bal. | F1 macro | AUC macro | VA |
|---|---|---|---|---|---|
| Chance(majoritaria) | 68.6 ± 4.8 | — | — | — | 50.0 ± 0.0 |
| LogReg | 74.0 ± 9.1 | 72.2 ± 7.9 | 71.0 ± 9.0 | 81.8 ± 7.4 | 77.5 ± 9.8 |
| SVM-RBF | 85.5 ± 4.9 | 85.0 ± 3.0 | 83.6 ± 5.1 | 92.5 ± 3.5 | 82.0 ± 5.3 |
| RandomForest | 86.9 ± 3.9 | 84.8 ± 3.7 | 84.6 ± 4.1 | 92.8 ± 3.3 | 85.5 ± 6.6 |
| LSTM | 83.0 ± 6.2 | 80.8 ± 2.3 | 80.3 ± 5.1 | 91.1 ± 4.1 | 81.2 ± 7.7 |
| CNN1D | 85.2 ± 5.1 | 83.8 ± 1.3 | 83.1 ± 4.4 | 93.8 ± 2.5 | 84.5 ± 5.9 |

## Exportação TFLite (melhor modelo temporal por VA média, 3 classes)

- Modelo: **LSTM** (fold de origem 3, conversão `from_keras_model`, ops: ADD, FILL, FULLY_CONNECTED, LOGISTIC, MUL, PACK, SHAPE, SOFTMAX, SPLIT, STRIDED_SLICE, SUB, TANH, TRANSPOSE, UNPACK)
- Tamanho: 101.8 KB; parâmetros: 5324
- Validação vs. Keras (2196 sequências): max |dif| = 6.47e-06, concordância do argmax = 100.00%
- Latência no Mac (CPU, 500 execuções, lote 1): p50 = 0.033 ms, p95 = 0.045 ms
- Metadados completos: `models/blink_temporal.json`

## Sensibilidade ao número de épocas (modelos temporais, 3 classes)

Execução adicional com 15 épocas (`results/exp1_intra_e15.csv`, log `results/logs/exp1_e15_20260920-183128.log`), mesmos hiperparâmetros e seed; sem salvar modelos nem alterar a exportação TFLite (que usa o LSTM de 50 épocas).

| Modelo | Épocas | BSA | Acur. bal. | F1 macro | AUC macro | VA |
|---|---|---|---|---|---|---|
| LSTM | 50 | 50.8 ± 8.9 | 53.1 ± 6.7 | 51.8 ± 7.8 | 68.6 ± 8.2 | 57.2 ± 7.5 |
| LSTM | 15 | 50.6 ± 9.5 | 53.7 ± 5.5 | 51.5 ± 8.0 | 71.4 ± 7.0 | 52.7 ± 6.0 |
| CNN1D | 50 | 48.0 ± 6.8 | 51.4 ± 5.2 | 50.0 ± 6.4 | 68.4 ± 6.1 | 55.0 ± 8.2 |
| CNN1D | 15 | 49.8 ± 6.9 | 53.7 ± 5.4 | 51.3 ± 5.6 | 69.7 ± 5.5 | 55.1 ± 3.6 |

Com 50 épocas a acurácia de treino chega a ~0,90–0,93 (ver log) enquanto a BSA em sujeitos não vistos fica em ~0,50: os modelos temporais pequenos memorizam padrões dos sujeitos de treino sem ganho claro de generalização, e nenhum deles supera o RandomForest sobre estatísticas 24d.

