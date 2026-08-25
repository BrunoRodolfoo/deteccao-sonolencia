# Baseline — UTA-RLDD (3 classes, 5 folds oficiais, divisão por sujeito)

Classificadores clássicos sobre blink features (30 piscadas x 4 caract.). Seed 42.
Referências (Ghoddoosian et al., 2019): HM-LSTM 65.2% | humanos 57.8%.

| Representação | Modelo | Acurácia média | Acur. balanceada | F1 macro |
|---|---|---|---|---|
| estatisticas(24d) | LogReg | 49.4% ± 4.2% | 49.7% | 48.5% |
| estatisticas(24d) | SVM-RBF | 53.0% ± 3.8% | 56.2% | 54.2% |
| estatisticas(24d) | RandomForest | 54.7% ± 9.0% | 58.0% | 56.2% |
| achatado(120d) | LogReg | 47.6% ± 6.1% | 47.7% | 46.6% |
| achatado(120d) | SVM-RBF | 48.1% ± 7.3% | 51.9% | 49.4% |
| achatado(120d) | RandomForest | 49.6% ± 9.5% | 54.4% | 51.5% |

**Melhor:** RandomForest com estatisticas(24d) — acurácia média 54.7% (vs. 65.2% do HM-LSTM e 57.8% de humanos).
