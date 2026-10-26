# Experimento 2 — vazamento de identidade no UTA-RLDD (blink features)

Execução 20260920-183128. Pool: 8661 sequências de teste dos 5 folds oficiais, 174 vídeos, 58 sujeitos; 3 classes. Seeds [42, 43, 44] para (i) e (ii); (iii) usa os folds oficiais. Modelos temporais: 50 épocas. Dados brutos: `results/exp2_leakage.csv`; agregado: `results/exp2_leakage_plot.csv`; log: `results/logs/exp2_20260920-183128.log`.

Valores em % (média ± desvio populacional, ddof=0, entre seeds para (i)/(ii); entre folds para (iii)). Chance (classe majoritária) por sequência no pool: 40.6%; por vídeo: 33.3%.

## RandomForest — {'n_estimators': 300}

| Protocolo | BSA | Acur. bal. | F1 macro | AUC macro | VA | vídeos do teste vistos no treino | sujeitos do teste vistos no treino |
|---|---|---|---|---|---|---|---|
| (i) aleatória por janela | 99.5 ± 0.0 | 99.6 ± 0.0 | 99.5 ± 0.1 | 100.0 ± 0.0 | n/a | 97.1 ± 0.1 | 100.0 ± 0.0 |
| (ii) por vídeo | 47.2 ± 1.4 | 52.5 ± 0.7 | 49.7 ± 1.2 | 67.9 ± 0.5 | 52.8 ± 1.2 | 0.0 ± 0.0 | 98.3 ± 0.9 |
| (iii) por sujeito (folds oficiais) | 54.2 ± 8.7 | 57.4 ± 7.5 | 55.8 ± 8.3 | 73.4 ± 6.8 | 55.7 ± 6.6 | 0.0 ± 0.0 | 0.0 ± 0.0 |

### Inflação (pontos percentuais em relação ao protocolo por sujeito)

| Comparação | BSA | Acur. bal. | F1 macro | AUC macro | VA |
|---|---|---|---|---|---|
| (i) janela − (iii) sujeito | +45.3 pp | +42.1 pp | +43.7 pp | +26.6 pp | n/a |
| (ii) vídeo − (iii) sujeito | -7.0 pp | -5.0 pp | -6.1 pp | -5.5 pp | -2.8 pp |

## LSTM — {'units': 32, 'dense': 16, 'dropout': 0.2, 'unroll': True, 'optimizer': 'Adam', 'lr': 0.001, 'batch': 64, 'epochs': 50, 'input_norm': 'z-score (treino)'}

| Protocolo | BSA | Acur. bal. | F1 macro | AUC macro | VA | vídeos do teste vistos no treino | sujeitos do teste vistos no treino |
|---|---|---|---|---|---|---|---|
| (i) aleatória por janela | 87.3 ± 0.7 | 87.9 ± 0.5 | 87.0 ± 0.7 | 97.1 ± 0.3 | n/a | 97.1 ± 0.1 | 100.0 ± 0.0 |
| (ii) por vídeo | 45.1 ± 0.6 | 48.9 ± 1.3 | 46.5 ± 0.5 | 62.3 ± 1.0 | 46.2 ± 2.3 | 0.0 ± 0.0 | 98.3 ± 0.9 |
| (iii) por sujeito (folds oficiais) | 49.2 ± 8.8 | 51.0 ± 6.6 | 49.7 ± 7.5 | 67.6 ± 9.0 | 50.8 ± 13.7 | 0.0 ± 0.0 | 0.0 ± 0.0 |

### Inflação (pontos percentuais em relação ao protocolo por sujeito)

| Comparação | BSA | Acur. bal. | F1 macro | AUC macro | VA |
|---|---|---|---|---|---|
| (i) janela − (iii) sujeito | +38.1 pp | +36.9 pp | +37.4 pp | +29.6 pp | n/a |
| (ii) vídeo − (iii) sujeito | -4.1 pp | -2.1 pp | -3.1 pp | -5.3 pp | -4.6 pp |

## Leitura

- (i) mede o que se obtém quando janelas sobrepostas do mesmo vídeo aparecem em treino e teste (vazamento de vídeo e de identidade); (ii) elimina o vazamento de vídeo, mas mantém o de identidade (os outros dois vídeos do mesmo sujeito ficam no treino); (iii) é o protocolo oficial, sem nenhum vazamento.
- A diferença (i)−(iii) e (ii)−(iii) quantifica quanto uma partição aleatória (comum na literatura de sonolência atuada) infla os resultados em relação a sujeitos nunca vistos.
- Em (ii) o resultado fica ABAIXO de (iii), e não acima: no UTA-RLDD cada sujeito contribui com exatamente um vídeo por classe, logo, quando um vídeo está no teste, o mesmo sujeito aparece no treino apenas com os OUTROS dois rótulos. A identidade vira uma pista contrária (transferência negativa), o que mostra que o vazamento de identidade só infla resultados quando o rótulo do sujeito é o mesmo dos dois lados — exatamente o caso das partições aleatórias por quadro/janela de bases de sonolência atuada.
- Em (iii) o treino de cada fold é o pool menos o fold de teste (ex.: 7392 sequências no fold 1), e não o arquivo oficial de treino (7379 no fold 1): as 13 sequências de diferença são as do último sujeito do fold 3 (três vídeos com 1, 6 e 6 sequências), ausente de todos os arquivos oficiais de treino. Além disso, a ordem das sequências de treino é outra. Para o RandomForest a diferença em relação ao Exp. 1 é pequena (BSA 54,2 vs. 54,5; VA 55,7 vs. 57,3), mas o LSTM é sensível a essa perturbação (BSA 49,2 vs. 50,8; VA 50,8 vs. 57,2), o que reforça a instabilidade dos modelos temporais pequenos com 58 sujeitos.

