# Exp. 3 — Estado ocular (aberto/fechado): intra-dataset e generalização cross-dataset MRL Eye ↔ CEW

Todos os números abaixo vêm de `results/exp3_eye_state.csv` / `results/exp3_strat_conditions.csv` (gerado por `src/pipeline_b/report_exp3.py`; log em `results/logs/exp3_eye_state_train.log`). Rótulo 1 = aberto, 0 = fechado. Valores em %. Chance = proporção da classe majoritária no conjunto de teste.

## Configuração

- Modelo: EyeNet (`src/pipeline_b/model.py`), 62865 parâmetros, entrada 48x48x1 em [0,1], Conv(16)-BN-ReLU-Pool / Conv(32)-BN-ReLU-Pool / Conv(64)-BN-ReLU-Conv(64)-BN-ReLU-Pool / GAP / Dense(32) / Dropout(0,3) / sigmoide. BatchNorm com momento 0,9 em vez do padrão 0,99 (ver ablação ao final: no treino completo os dois valores dão resultados comparáveis; 0,9 é mantido por ser o usado em todos os resultados reportados).
- Treino: Adam lr 1e-3, batch 128, até 12 épocas, EarlyStopping (val_loss, paciência 3, melhores pesos), ReduceLROnPlateau (0,5, paciência 2), entropia cruzada binária, seed 42.
- Augmentation (só no treino): flip horizontal, rotação ±10°, translação ±8%, zoom ±10%, brilho ±15%, contraste ±20%.
- Protocolo A: GroupKFold 5 folds sobre os 37 sujeitos do MRL; 10% dos sujeitos de treino separados para early stopping; treino subamostrado para ≤25000 imagens (estratificado por sujeito e classe); teste em TODAS as imagens dos sujeitos de teste.
- Protocolo B: modelo final treinado com a mesma receita em todos os 37 sujeitos (sujeitos de validação: [13, 14, 15, 25], 25002 imagens de treino, melhor época 11/12) avaliado em todo o CEW; também reportamos os 5 modelos dos folds de A no CEW.
- CEW: dois recortes — *patch* = 24x24 oficiais ampliados para 48x48; *crop* = caixa quadrada de lado 0.35·largura da face (36 px nas faces 100x100) centrada na coordenada do olho, redimensionada para 48x48.
- Protocolos C/D: GroupKFold 5 folds no CEW agrupado por face de origem (olhos E/D da mesma face no mesmo lado); 10% das faces de treino para early stopping; cada modelo é avaliado no fold de teste do CEW (D) e em todo o MRL (C).

## Protocolo A — intra-MRL, separação por sujeito

| Fold | n | Acurácia | Acur. balanceada | F1 (macro) | AUC | Chance |
|---|---|---|---|---|---|---|
| 1 | 16975 | 91.2 | 90.8 | 89.8 | 97.3 | 69.7 |
| 2 | 16973 | 97.6 | 97.6 | 97.6 | 99.8 | 52.3 |
| 3 | 16886 | 90.1 | 90.5 | 90.1 | 97.2 | 54.5 |
| 4 | 17167 | 94.3 | 94.5 | 94.3 | 99.0 | 54.6 |
| 5 | 16897 | 97.1 | 97.6 | 96.6 | 99.8 | 70.4 |
| **média ± dp** | | **94.1 ± 3.4** | **94.2 ± 3.5** | **93.7 ± 3.6** | **98.6 ± 1.3** | **60.3 ± 8.9** |
| agregado out-of-fold | 84898 | 94.1 | 94.1 | 94.1 | 98.7 | 50.6 |

### A — por condição (predições out-of-fold agregadas dos 5 folds, todo o MRL)

| Condição | n | Acurácia | Acur. balanceada | F1 (macro) | AUC | Chance |
|---|---|---|---|---|---|---|
| com óculos | 24001 | 88.8 | 88.5 | 88.5 | 95.7 | 59.4 |
| sem óculos | 60897 | 96.1 | 96.0 | 96.1 | 99.4 | 54.5 |
| iluminação boa | 31268 | 93.5 | 92.6 | 93.1 | 98.8 | 61.5 |
| iluminação ruim | 53630 | 94.4 | 94.3 | 94.3 | 98.6 | 55.8 |
| reflexo grande | 12709 | 86.7 | 84.2 | 84.5 | 95.0 | 68.2 |
| reflexo pequeno | 5962 | 94.8 | 93.4 | 93.8 | 98.3 | 69.7 |
| sem reflexo | 66227 | 95.4 | 95.5 | 95.4 | 99.1 | 54.5 |
| Aptina | 2764 | 98.3 | 93.7 | 96.1 | 99.6 | 86.4 |
| IDS | 11992 | 96.4 | 91.9 | 84.0 | 98.6 | 95.2 |
| RealSense | 70142 | 93.5 | 93.5 | 93.3 | 98.5 | 58.4 |

## Protocolo B — MRL → CEW (cross-dataset, IR de laboratório → RGB in-the-wild)

### B — avaliado em CEW_patch

Modelos dos 5 folds de A:

| Fold | n | Acurácia | Acur. balanceada | F1 (macro) | AUC | Chance |
|---|---|---|---|---|---|---|
| 1 | 4846 | 77.9 | 77.5 | 76.8 | 94.6 | 50.8 |
| 2 | 4846 | 80.6 | 80.3 | 79.9 | 95.9 | 50.8 |
| 3 | 4846 | 66.0 | 65.5 | 61.6 | 85.1 | 50.8 |
| 4 | 4846 | 80.9 | 80.6 | 80.3 | 93.9 | 50.8 |
| 5 | 4846 | 89.5 | 89.4 | 89.4 | 96.8 | 50.8 |
| **média ± dp** | | **79.0 ± 8.5** | **78.7 ± 8.6** | **77.6 ± 10.1** | **93.2 ± 4.7** | **50.8 ± 0.0** |

Modelo final (37 sujeitos):

| Fold | n | Acurácia | Acur. balanceada | F1 (macro) | AUC | Chance |
|---|---|---|---|---|---|---|
| modelo final | 4846 | 85.7 | 85.5 | 85.4 | 96.8 | 50.8 |

### B — avaliado em CEW_crop

Modelos dos 5 folds de A:

| Fold | n | Acurácia | Acur. balanceada | F1 (macro) | AUC | Chance |
|---|---|---|---|---|---|---|
| 1 | 4846 | 86.3 | 86.2 | 86.2 | 95.7 | 50.8 |
| 2 | 4846 | 78.6 | 78.2 | 77.5 | 95.5 | 50.8 |
| 3 | 4846 | 64.7 | 64.1 | 59.3 | 85.1 | 50.8 |
| 4 | 4846 | 81.4 | 81.2 | 80.9 | 93.3 | 50.8 |
| 5 | 4846 | 92.0 | 91.9 | 92.0 | 97.7 | 50.8 |
| **média ± dp** | | **80.6 ± 10.3** | **80.3 ± 10.5** | **79.2 ± 12.4** | **93.5 ± 4.9** | **50.8 ± 0.0** |

Modelo final (37 sujeitos):

| Fold | n | Acurácia | Acur. balanceada | F1 (macro) | AUC | Chance |
|---|---|---|---|---|---|---|
| modelo final | 4846 | 90.0 | 89.9 | 89.9 | 97.6 | 50.8 |

Modelo final por lado do olho (CEW):

| Conjunto | Lado | n | Acurácia | AUC |
|---|---|---|---|---|
| CEW_crop | L | 2423 | 90.0 | 97.8 |
| CEW_crop | R | 2423 | 90.0 | 97.4 |
| CEW_patch | L | 2423 | 86.0 | 96.6 |
| CEW_patch | R | 2423 | 85.3 | 97.0 |

## Protocolo C — CEW → MRL (cross-dataset, RGB in-the-wild → IR de laboratório)

### C — treinado em CEW (patch), avaliado em todo o MRL

| Fold | n | Acurácia | Acur. balanceada | F1 (macro) | AUC | Chance |
|---|---|---|---|---|---|---|
| 1 | 84898 | 71.0 | 71.1 | 70.5 | 79.8 | 50.6 |
| 2 | 84898 | 69.2 | 69.6 | 66.6 | 86.5 | 50.6 |
| 3 | 84898 | 67.4 | 67.8 | 64.7 | 86.8 | 50.6 |
| 4 | 84898 | 66.3 | 66.6 | 63.1 | 84.7 | 50.6 |
| 5 | 84898 | 66.8 | 67.1 | 64.5 | 83.1 | 50.6 |
| **média ± dp** | | **68.1 ± 1.9** | **68.4 ± 1.9** | **65.9 ± 2.9** | **84.2 ± 2.8** | **50.6 ± 0.0** |
| ensemble (média) dos 5 | 84898 | 68.5 | 68.9 | 66.2 | 85.1 | 50.6 |

Por condição do MRL (média ± dp dos 5 modelos de fold):

| Condição | n | Acurácia | Acur. balanceada | F1 (macro) | AUC | Chance |
|---|---|---|---|---|---|---|
| com óculos | 24001 | 65.2 ± 5.1 | 61.1 ± 2.8 | 60.2 ± 3.5 | 65.1 ± 4.3 | 59.4 ± 0.0 |
| sem óculos | 60897 | 69.3 ± 4.4 | 71.8 ± 4.0 | 67.7 ± 5.2 | 93.4 ± 2.3 | 54.5 ± 0.0 |
| iluminação boa | 31268 | 69.9 ± 3.2 | 75.1 ± 2.5 | 69.8 ± 3.3 | 89.6 ± 1.6 | 61.5 ± 0.0 |
| iluminação ruim | 53630 | 67.1 ± 2.1 | 63.8 ± 3.1 | 61.6 ± 4.4 | 81.1 ± 3.2 | 55.8 ± 0.0 |
| reflexo grande | 12709 | 62.5 ± 4.0 | 65.7 ± 3.6 | 61.2 ± 3.4 | 69.2 ± 3.5 | 68.2 ± 0.0 |
| reflexo pequeno | 5962 | 85.2 ± 5.3 | 85.3 ± 9.0 | 83.0 ± 7.2 | 93.2 ± 3.7 | 69.7 ± 0.0 |
| sem reflexo | 66227 | 67.7 ± 2.7 | 64.8 ± 3.4 | 62.0 ± 5.0 | 86.9 ± 3.5 | 54.5 ± 0.0 |
| Aptina | 2764 | 97.9 ± 0.4 | 97.1 ± 0.5 | 95.7 ± 0.8 | 99.3 ± 0.2 | 86.4 ± 0.0 |
| IDS | 11992 | 70.9 ± 5.2 | 84.3 ± 2.5 | 53.5 ± 3.5 | 91.3 ± 2.5 | 95.2 ± 0.0 |
| RealSense | 70142 | 66.5 ± 2.0 | 60.9 ± 3.2 | 58.3 ± 4.8 | 81.6 ± 3.9 | 58.4 ± 0.0 |

### C — treinado em CEW (crop), avaliado em todo o MRL

| Fold | n | Acurácia | Acur. balanceada | F1 (macro) | AUC | Chance |
|---|---|---|---|---|---|---|
| 1 | 84898 | 66.4 | 66.7 | 63.7 | 83.1 | 50.6 |
| 2 | 84898 | 67.4 | 67.7 | 64.6 | 85.4 | 50.6 |
| 3 | 84898 | 69.8 | 70.1 | 68.2 | 86.3 | 50.6 |
| 4 | 84898 | 66.9 | 67.3 | 64.1 | 84.6 | 50.6 |
| 5 | 84898 | 64.8 | 65.1 | 61.4 | 78.4 | 50.6 |
| **média ± dp** | | **67.1 ± 1.8** | **67.4 ± 1.8** | **64.4 ± 2.4** | **83.6 ± 3.1** | **50.6 ± 0.0** |
| ensemble (média) dos 5 | 84898 | 67.2 | 67.6 | 64.5 | 84.5 | 50.6 |

Por condição do MRL (média ± dp dos 5 modelos de fold):

| Condição | n | Acurácia | Acur. balanceada | F1 (macro) | AUC | Chance |
|---|---|---|---|---|---|---|
| com óculos | 24001 | 67.0 ± 2.1 | 62.2 ± 2.5 | 61.7 ± 3.1 | 67.7 ± 5.5 | 59.4 ± 0.0 |
| sem óculos | 60897 | 67.1 ± 2.2 | 69.8 ± 2.0 | 65.1 ± 2.7 | 92.0 ± 2.7 | 54.5 ± 0.0 |
| iluminação boa | 31268 | 70.8 ± 3.6 | 75.7 ± 2.8 | 70.8 ± 3.7 | 89.9 ± 2.5 | 61.5 ± 0.0 |
| iluminação ruim | 53630 | 64.9 ± 1.4 | 60.9 ± 1.7 | 57.6 ± 2.8 | 79.2 ± 3.8 | 55.8 ± 0.0 |
| reflexo grande | 12709 | 65.1 ± 3.9 | 64.9 ± 3.2 | 62.9 ± 3.4 | 70.6 ± 4.9 | 68.2 ± 0.0 |
| reflexo pequeno | 5962 | 86.9 ± 3.0 | 85.1 ± 6.0 | 84.5 ± 4.3 | 93.4 ± 1.2 | 69.7 ± 0.0 |
| sem reflexo | 66227 | 65.6 ± 2.1 | 62.3 ± 2.3 | 57.8 ± 3.4 | 87.0 ± 4.0 | 54.5 ± 0.0 |
| Aptina | 2764 | 92.8 ± 3.5 | 94.8 ± 1.8 | 87.5 ± 4.9 | 99.4 ± 0.3 | 86.4 ± 0.0 |
| IDS | 11992 | 74.9 ± 4.5 | 85.6 ± 1.8 | 56.2 ± 3.3 | 93.0 ± 1.2 | 95.2 ± 0.0 |
| RealSense | 70142 | 64.7 ± 1.5 | 58.5 ± 1.9 | 54.8 ± 3.0 | 79.3 ± 4.1 | 58.4 ± 0.0 |

## Protocolo D — intra-CEW (referência), 5 folds por face

### D — CEW (patch)

| Fold | n | Acurácia | Acur. balanceada | F1 (macro) | AUC | Chance |
|---|---|---|---|---|---|---|
| 1 | 970 | 94.1 | 94.1 | 94.1 | 99.2 | 50.9 |
| 2 | 970 | 94.2 | 94.3 | 94.2 | 98.1 | 50.7 |
| 3 | 970 | 93.8 | 93.8 | 93.8 | 98.6 | 50.7 |
| 4 | 968 | 94.6 | 94.7 | 94.6 | 99.1 | 50.8 |
| 5 | 968 | 96.5 | 96.5 | 96.5 | 99.4 | 50.8 |
| **média ± dp** | | **94.7 ± 1.1** | **94.7 ± 1.1** | **94.7 ± 1.1** | **98.9 ± 0.5** | **50.8 ± 0.1** |
| agregado out-of-fold | 4846 | 94.7 | 94.7 | 94.7 | 98.8 | 50.8 |

### D — CEW (crop)

| Fold | n | Acurácia | Acur. balanceada | F1 (macro) | AUC | Chance |
|---|---|---|---|---|---|---|
| 1 | 970 | 95.9 | 95.9 | 95.9 | 99.2 | 50.9 |
| 2 | 970 | 94.5 | 94.5 | 94.5 | 98.5 | 50.7 |
| 3 | 970 | 92.6 | 92.5 | 92.6 | 98.5 | 50.7 |
| 4 | 968 | 95.6 | 95.5 | 95.5 | 99.0 | 50.8 |
| 5 | 968 | 93.8 | 93.8 | 93.8 | 98.7 | 50.8 |
| **média ± dp** | | **94.5 ± 1.3** | **94.4 ± 1.4** | **94.5 ± 1.3** | **98.8 ± 0.3** | **50.8 ± 0.1** |
| agregado out-of-fold | 4846 | 94.5 | 94.4 | 94.5 | 98.7 | 50.8 |

## Resumo (acurácia média ± dp dos 5 folds)

| Treino → Teste | Acurácia | AUC | Chance |
|---|---|---|---|
| MRL → MRL (sujeitos não vistos) | 94.1 ± 3.4 | 98.6 ± 1.3 | 60.3 |
| MRL → CEW patch | 79.0 ± 8.5 | 93.2 ± 4.7 | 50.8 |
| MRL → CEW crop | 80.6 ± 10.3 | 93.5 ± 4.9 | 50.8 |
| CEW patch → MRL | 68.1 ± 1.9 | 84.2 ± 2.8 | 50.6 |
| CEW crop → MRL | 67.1 ± 1.8 | 83.6 ± 3.1 | 50.6 |
| CEW patch → CEW patch (faces não vistas) | 94.7 ± 1.1 | 98.9 ± 0.5 | 50.8 |
| CEW crop → CEW crop (faces não vistas) | 94.5 ± 1.3 | 98.8 ± 0.3 | 50.8 |

## MobileNetV3-Small (opcional, só fold 1 do protocolo A)

Fonte: `results/exp3_mobilenetv3.csv` (`src/pipeline_b/train_mobilenet.py`; alpha 0,75, pesos ImageNet, entrada 96x96x3 com o cinza replicado, batch 64, guarda de 15 min, mesma partição e subamostragem do fold 1).

| Modelo (fold 1, mesmos sujeitos de teste) | Parâmetros | n | Acurácia | Acur. balanceada | F1 (macro) | AUC | Chance | Épocas | Treino (s) |
|---|---|---|---|---|---|---|---|---|---|
| EyeNet | 62865 | 16975 | 91.2 | 90.8 | 89.8 | 97.3 | 69.7 | 12 | 148 |
| MobileNetV3-Small 0,75 | 583593 | 16975 | 95.9 | 96.1 | 95.2 | 99.2 | 69.7 | 4 | 386 |

A MobileNetV3 executou 4 épocas com melhor época 1 (val_loss): parou por EarlyStopping (paciência 3), não pela guarda de tempo. Parâmetros ≈ 9x os do EyeNet; comparação nos 5 folds e exportação TFLite ficam como trabalho futuro.

## Ablação: momento da BatchNormalization (fold 1 do protocolo A)

Fonte: `results/exp3_bn_momentum_ablation.csv` (`src/pipeline_b/ablation_bn_momentum.py`, mesma partição e subamostragem do fold 1; acurácia em modo de inferência). A hipótese original (0,99 faria o modelo prever classe constante em modo de inferência) NÃO se confirma no treino completo — só havia sido observada em execuções curtas de teste; com 0,99 o fold 1 fica até um pouco melhor. O valor 0,9 foi mantido por ser o usado em todos os resultados reportados.

| BN momentum | Conjunto | n | Acurácia | Acur. balanceada | AUC | Épocas | Melhor época |
|---|---|---|---|---|---|---|---|
| 0.99 | treino (mesmas imagens, modo inferência) | 25001 | 97.4 | 97.6 | 99.8 | 10 | 7 |
| 0.99 | teste fold 1 | 16975 | 93.2 | 91.8 | 97.9 | 10 | 7 |
| 0.9 | treino (mesmas imagens, modo inferência) | 25001 | 98.3 | 98.3 | 99.9 | 12 | 11 |
| 0.9 | teste fold 1 | 16975 | 91.2 | 90.8 | 97.3 | 12 | 11 |

## Exp. 4 (desktop) — exportação TFLite do modelo final MRL

Atenção: o modelo final foi treinado com todos os sujeitos do MRL (exceto os de validação, usados só para early stopping); por isso os conjuntos `MRL_*` abaixo NÃO são teste cego (o número cego intra-MRL é o do protocolo A). Eles servem para medir a QUEDA causada pela quantização, que é a coluna Δ. O CEW é totalmente cego.

Fonte: `results/exp4_tflite_eval.csv`, `results/exp4_desktop_latency.csv`, `models/eye_state.json`.

| Conjunto | n | Keras acc | fp32 acc | int8 acc | Δ int8−Keras (p.p.) | Keras AUC | int8 AUC |
|---|---|---|---|---|---|---|---|
| MRL_val_subjects | 15123 | 98.5 | 98.5 | 98.5 | -0.05 | 99.9 | 99.9 |
| MRL_completo | 84898 | 98.5 | 98.5 | 98.5 | -0.03 | 99.9 | 99.9 |
| CEW_patch | 4846 | 85.7 | 85.7 | 85.6 | -0.10 | 96.8 | 96.3 |
| CEW_crop | 4846 | 90.0 | 90.0 | 89.7 | -0.31 | 97.6 | 97.2 |

Latência batch 1 (arm64 / macOS-26.5.1-arm64-arm-64bit, TF 2.21.0, 50 aquecimento + 500 execuções):

| Modelo | Tamanho (KB) | Threads | p50 (ms) | p95 (ms) |
|---|---|---|---|---|
| fp32 | 248.2 | 1 | 0.067 | 0.070 |
| fp32 | 248.2 | 4 | 0.082 | 0.099 |
| int8 | 72.1 | 1 | 0.060 | 0.064 |
| int8 | 72.1 | 4 | 0.070 | 0.087 |

Modelo int8: E/S `uint8` (entrada uint8, scale=0.00392157, zero_point=0; saída uint8, scale=0.00390625, zero_point=0).

