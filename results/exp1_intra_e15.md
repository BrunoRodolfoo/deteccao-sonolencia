# Experimento 1 — UTA-RLDD intra-conjunto (5 folds oficiais, divisão por sujeito)

Execução 20260920-183128. Seed 42. Modelos temporais treinados por 15 épocas fixas (sem early stopping; o teste nunca é consultado). Valores em % (média ± desvio populacional, ddof=0, nos 5 folds). Dados brutos: `results/exp1_intra_e15.csv`; log: `results/logs/exp1_e15_20260920-183128.log`.

Modelos clássicos: estatísticas 24d por sequência (média, desvio, mín., máx., mediana, IQR de cada característica) com os hiperparâmetros do baseline. Modelos temporais (Keras) sobre a sequência bruta (30 piscadas x 4): **LSTM**: {"units": 32, "dense": 16, "dropout": 0.2, "unroll": true, "optimizer": "Adam", "lr": 0.001, "batch": 64, "epochs": 15, "input_norm": "z-score (treino)"}; **CNN1D**: {"conv": [32, 64], "kernel": 3, "pool": "GAP", "dense": 16, "dropout": 0.2, "optimizer": "Adam", "lr": 0.001, "batch": 64, "epochs": 15, "input_norm": "z-score (treino)"}.

VA = acurácia por vídeo, voto majoritário das janelas do vídeo; empate resolvido pela maior probabilidade média entre as classes empatadas. Chance = classe majoritária do teste.

## Esquema 3c: 3 classes: alerta / pouca vigilância / sonolento

| Modelo | BSA | Acur. bal. | F1 macro | AUC macro | VA |
|---|---|---|---|---|---|
| Chance(majoritaria) | 44.4 ± 2.4 | — | — | — | 33.3 ± 0.0 |
| LogReg | 49.4 ± 4.3 | 49.7 ± 3.0 | 48.5 ± 3.6 | 68.9 ± 5.5 | 50.7 ± 7.6 |
| SVM-RBF | 53.0 ± 3.8 | 56.2 ± 4.5 | 54.2 ± 3.9 | 69.9 ± 5.1 | 54.0 ± 4.0 |
| RandomForest | 54.5 ± 9.3 | 57.8 ± 7.9 | 56.0 ± 9.1 | 73.5 ± 6.8 | 57.3 ± 7.2 |
| LSTM | 50.6 ± 9.5 | 53.7 ± 5.5 | 51.5 ± 8.0 | 71.4 ± 7.0 | 52.7 ± 6.0 |
| CNN1D | 49.8 ± 6.9 | 53.7 ± 5.4 | 51.3 ± 5.6 | 69.7 ± 5.5 | 55.1 ± 3.6 |

Referências (Ghoddoosian et al., 2019, Tabela 2, média dos 5 folds): HM-LSTM: BSA 54.0% / VA 65.2%; LSTM: BSA 52.8% / VA 61.4%; Camadas densas: BSA 52.0% / VA 57.0%; Humanos: BSA — / VA 57.8%.

### VA por fold (3 classes) — comparação com a Tabela 3 do artigo

| Modelo | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Média |
|---|---|---|---|---|---|---|
| LogReg | 57.6% | 36.1% | 55.6% | 52.8% | 51.5% | 50.7% |
| SVM-RBF | 57.6% | 55.6% | 50.0% | 58.3% | 48.5% | 54.0% |
| RandomForest | 54.5% | 69.4% | 50.0% | 61.1% | 51.5% | 57.3% |
| LSTM | 42.4% | 58.3% | 50.0% | 58.3% | 54.5% | 52.7% |
| CNN1D | 48.5% | 58.3% | 55.6% | 58.3% | 54.5% | 55.1% |
| HM-LSTM (Tabela 3 do artigo) | 64.0% | 61.0% | 70.0% | 64.0% | 67.0% | 65.2% |
| Humanos (Tabela 3 do artigo) | 62.0% | 59.0% | 60.0% | 53.0% | 55.0% | 57.8% |

### BSA por fold (3 classes)

| Modelo | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Média |
|---|---|---|---|---|---|---|
| LogReg | 44.8% | 46.1% | 48.2% | 56.8% | 50.8% | 49.4% |
| SVM-RBF | 55.5% | 58.4% | 53.2% | 50.8% | 47.3% | 53.0% |
| RandomForest | 46.7% | 72.6% | 51.9% | 49.0% | 52.3% | 54.5% |
| LSTM | 32.2% | 55.6% | 52.6% | 53.3% | 59.4% | 50.6% |
| CNN1D | 42.7% | 61.8% | 51.9% | 49.0% | 43.8% | 49.8% |

## Esquema B1: alerta {0} vs. não alerta {1,2}

| Modelo | BSA | Acur. bal. | F1 macro | AUC macro | VA |
|---|---|---|---|---|---|
| Chance(majoritaria) | 81.4 ± 2.8 | — | — | — | 66.7 ± 0.0 |
| LogReg | 78.5 ± 6.7 | 60.1 ± 5.5 | 60.3 ± 5.8 | 80.3 ± 6.5 | 70.9 ± 4.5 |
| SVM-RBF | 85.1 ± 2.9 | 78.0 ± 1.7 | 76.6 ± 2.4 | 89.9 ± 2.2 | 81.8 ± 6.4 |
| RandomForest | 87.4 ± 3.2 | 79.8 ± 4.3 | 79.4 ± 2.9 | 88.7 ± 5.1 | 82.2 ± 3.3 |
| LSTM | 83.1 ± 9.8 | 77.3 ± 6.3 | 75.3 ± 8.7 | 87.1 ± 8.4 | 80.0 ± 7.5 |
| CNN1D | 83.3 ± 9.0 | 75.3 ± 6.5 | 74.3 ± 7.7 | 89.2 ± 6.2 | 77.1 ± 5.4 |

## Esquema B2: não sonolento {0,1} vs. sonolento {2}

| Modelo | BSA | Acur. bal. | F1 macro | AUC macro | VA |
|---|---|---|---|---|---|
| Chance(majoritaria) | 59.2 ± 4.5 | — | — | — | 66.7 ± 0.0 |
| LogReg | 66.8 ± 6.5 | 63.4 ± 7.2 | 63.0 ± 8.1 | 66.9 ± 10.3 | 73.6 ± 2.3 |
| SVM-RBF | 66.2 ± 5.9 | 63.9 ± 7.1 | 63.7 ± 7.0 | 66.2 ± 12.1 | 69.4 ± 3.1 |
| RandomForest | 66.5 ± 9.2 | 63.4 ± 10.2 | 62.9 ± 10.5 | 70.8 ± 11.6 | 70.1 ± 4.1 |
| LSTM | 62.9 ± 7.9 | 60.8 ± 8.7 | 59.8 ± 8.7 | 65.1 ± 8.9 | 69.5 ± 2.6 |
| CNN1D | 61.1 ± 8.0 | 61.1 ± 9.2 | 59.6 ± 7.9 | 65.0 ± 10.7 | 69.5 ± 6.0 |

## Esquema B3: alerta {0} vs. sonolento {2}, descartando a classe intermediária

| Modelo | BSA | Acur. bal. | F1 macro | AUC macro | VA |
|---|---|---|---|---|---|
| Chance(majoritaria) | 68.6 ± 4.8 | — | — | — | 50.0 ± 0.0 |
| LogReg | 74.0 ± 9.1 | 72.2 ± 7.9 | 71.0 ± 9.0 | 81.8 ± 7.4 | 77.5 ± 9.8 |
| SVM-RBF | 85.5 ± 4.9 | 85.0 ± 3.0 | 83.6 ± 5.1 | 92.5 ± 3.5 | 82.0 ± 5.3 |
| RandomForest | 86.9 ± 3.9 | 84.8 ± 3.7 | 84.6 ± 4.1 | 92.8 ± 3.3 | 85.5 ± 6.6 |
| LSTM | 86.1 ± 4.8 | 84.8 ± 3.1 | 84.0 ± 4.6 | 93.2 ± 3.4 | 83.9 ± 6.9 |
| CNN1D | 86.4 ± 4.3 | 85.1 ± 2.0 | 84.4 ± 3.6 | 94.8 ± 2.3 | 85.5 ± 7.3 |

