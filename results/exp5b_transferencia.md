# Exp. 5b (final) — Transferência de sonolência atuada (DMD) para real (UTA-RLDD): resposta direta da PP1

Execução 20260927-211326. Seed 42. Gerado por `src/experiments/exp5_transfer.py` a partir de `results/exp5b_transferencia.csv` (métricas, quedas e diagnósticos), `results/exp5b_limiares.csv`, `results/exp5b_importancia.csv`, `results/exp5b_por_sujeito.csv`, `results/exp5b_comparacao_exp1.csv`, `results/exp5b_predicoes.csv`, `results/exp5b_predicoes_video.csv` e `results/exp5b_amostra.csv`; log em `results/logs/exp5b_transferencia.log`; modelos de transferência em `models/exp5/exp5b_final_C_*.joblib` (metadados em `models/exp5/exp5b_final_modelos.json`); figura `results/figures/fig_exp5b_transferencia.{pdf,png}`. Nenhum número deste arquivo foi digitado à mão. A rodada preliminar (janelas de 30 s, só o Fold 1, DMD só com a s5) foi descartada; seus arquivos foram removidos em 06/10/2026.

**Dados.** UTA-RLDD completa: 60 sujeitos, 180 vídeos lógicos (182 arquivos), 5 folds oficiais. DMD: 13 sujeitos com sessão de sonolência (s5) e as sessões de distração s1/s2/s3 dos mesmos sujeitos. Janelas de 10 s com passo de 5 s de `data/features/final/` (`results/exp5_caracteristicas.md`).

Valores em %, com IC 95% entre colchetes: bootstrap percentil sobre **sujeitos** (1000 reamostragens, seed 42) das predições de teste — os modelos não são retreinados na reamostragem. AUC × 100. Acaso = 50 em acurácia balanceada e AUC. Quedas em pontos percentuais (pp; para AUC, pontos de AUC × 100); valor positivo = o segundo termo é pior.

## 1. Resposta curta (variante principal, esquema B3: KSS 0 × KSS 10)

| Métrica | Modelo | (A) DMD → DMD (atuado, LOSO) | (B) UTA → UTA (real, 5 folds) | (C) DMD → UTA (transferência) | Queda da PP1 (B − C), pp | Choque de realidade (A − C), pp |
|---|---|---|---|---|---|---|
| AUC por janela | RF | 82,4 [73,6; 89,7] | 80,1 [76,3; 83,8] | 69,2 [63,7; 74,3] | 10,9 [6,7; 15,7] | 13,1 [3,5; 22,6] |
| AUC por janela | LR | 83,5 [76,1; 88,9] | 74,4 [69,7; 78,4] | 64,5 [58,1; 70,9] | 9,8 [3,7; 16,0] | 18,9 [9,7; 27,8] |
| Acur. bal. por janela, limiar 0,5 | RF | 76,4 [68,3; 83,7] | 74,2 [70,9; 77,6] | 65,2 [61,7; 68,6] | 9,0 [5,8; 12,2] | 11,3 [2,2; 19,8] |
| Acur. bal. por janela, limiar 0,5 | LR | 78,6 [70,1; 85,3] | 72,0 [68,9; 74,9] | 64,0 [60,7; 67,4] | 7,9 [4,8; 10,9] | 14,6 [5,6; 22,0] |
| Acur. bal. por janela, limiar calibrado no treino ¹ | RF | 78,6 [72,6; 84,2] | 74,0 [70,7; 77,4] | 62,9 [60,0; 66,0] | 11,1 [8,0; 14,2] | 15,8 [8,8; 22,1] |
| Acur. bal. por janela, limiar calibrado no treino ¹ | LR | 78,7 [72,1; 83,8] | 72,2 [69,1; 75,2] | 63,3 [60,2; 66,7] | 8,9 [5,6; 11,9] | 15,4 [7,8; 21,4] |
| AUC por vídeo | RF | n/a ² | 92,4 [87,9; 96,2] | 80,5 [73,0; 87,2] | 11,9 [5,6; 18,8] | n/a |
| AUC por vídeo | LR | n/a ² | 89,0 [82,2; 94,9] | 76,0 [67,8; 84,7] | 13,0 [5,8; 20,8] | n/a |
| Acurácia por vídeo, limiar 0,5 | RF | n/a ² | 85,8 [80,0; 91,7] | 73,3 [67,5; 79,2] | 12,5 [5,0; 19,2] | n/a |
| Acurácia por vídeo, limiar 0,5 | LR | n/a ² | 85,8 [80,0; 90,8] | 65,8 [60,0; 71,7] | 20,0 [12,5; 26,7] | n/a |
| Acurácia por vídeo, limiar calibrado no treino ¹ | RF | n/a ² | 82,5 [76,7; 88,3] | 60,8 [55,8; 65,8] | 21,7 [14,2; 29,2] | n/a |
| Acurácia por vídeo, limiar calibrado no treino ¹ | LR | n/a ² | 85,0 [79,2; 90,0] | 64,2 [58,3; 70,0] | 20,8 [12,5; 28,3] | n/a |

¹ Limiar escolhido só com o treino de cada modelo, pela máxima acurácia balanceada nos escores fora da amostra: C usa o limiar do LOSO do DMD (RF 0,662; LR 0,535), sem olhar a UTA; A usa um LOSO aninhado (o sujeito de teste fica fora da escolha); B, uma validação interna nos 4 folds de treino de cada fold (seção 6). ² Uma sessão do DMD mistura janelas alerta e sonolentas (e a classe alerta vem de outra sessão, a s2): não há rótulo por vídeo no DMD.

Leitura automática (regras fixas sobre os IC; a interpretação está na seção 14):

- **Random Forest**: AUC da transferência acima do acaso; queda de AUC B − C com IC que exclui 0; queda de acurácia balanceada com IC que exclui 0 no limiar 0,5 e que exclui 0 com os limiares calibrados no treino; na UTA o modelo do DMD acerta 48,4% das janelas sonolentas e 82,0% das alertas no limiar 0,5, e 37,0% / 88,7% no limiar calibrado no DMD (0,662).
- **Regressão logística**: AUC da transferência acima do acaso; queda de AUC B − C com IC que exclui 0; queda de acurácia balanceada com IC que exclui 0 no limiar 0,5 e que exclui 0 com os limiares calibrados no treino; na UTA o modelo do DMD acerta 33,7% das janelas sonolentas e 94,4% das alertas no limiar 0,5, e 30,8% / 95,8% no limiar calibrado no DMD (0,535).

## 2. Desenho e decisões

| Item | Decisão | Por quê |
|---|---|---|
| Unidade de análise | janela de **10 s, passo 5 s** (`data/features/final/<base>/windows_10s.csv`) | os trechos `driver_actions/safe_drive` do DMD são curtos demais para as janelas de 30 s da Entrega 01 (item 9 das notas de fechamento; `results/exp5_caracteristicas.md`); 5 s e 15 s como sensibilidade |
| Classe alerta (DMD) | janelas inteiramente dentro de um trecho safe_drive da **sessão s2** (carro parado, direção encenada) | mesma condição de carro parado da s5: a diferença entre as classes é a sonolência encenada; a s1 (carro em movimento) é sensibilidade |
| Classe sonolento (DMD) | janelas da **s5** com fechamento anotado ≥ 0,5 s ou bocejo anotado (≥ 0,5 s dentro da janela) | único rótulo por janela que a anotação OpenLABEL da s5 permite; janelas da s5 **sem** sinal ficam fora (ambíguas; como alerta, só na sensibilidade) |
| Rótulo da UTA | do vídeo (KSS autorrelatado 0/5/10): B3 = 0 × 10 (principal), B1 = 0 × 5+10, B2 = 0+5 × 10 | mesmas harmonizações do Exp. 1; o DMD é binário, então B3 é o par comparável |
| Referência de normalização | DMD: janelas safe_drive da **s3** do sujeito; UTA: **último terço do vídeo 0**; nunca entram em treino/teste, nem as janelas de fronteira (que dividem quadros com a referência) | regra final (item 10 das notas): na rodada preliminar as janelas alerta do DMD sobrepunham a referência e ficavam perto de 0 por construção. Nesta rodada, janelas avaliadas que dividem tempo com a referência: DMD 0, UTA 0 |
| Normalização por sujeito (principal) | **centragem** `<f>_c` = x − média das janelas de referência do sujeito; a escala é aprendida no treino (padronização da LR ajustada só no treino) | o z-score continua sem estimativa: com desvio > 0 e ≥ 5 janelas de referência, ele existe em 20% das células sujeito × característica do DMD (3 de 13 sujeitos têm ≥ 5 janelas de referência) e em 70% na UTA; nas 11 características ao mesmo tempo, em 0% dos sujeitos do DMD e 0% dos da UTA (na UTA, só `ear_media`, `ear_dp` têm z em todos os sujeitos). Centrar é a parte do z-score que existe nas duas bases |
| Características | as 11 da janela: `perclos`, `taxa_piscadas_min`, `dur_media_piscada_s`, `dur_max_fechamento_s`, `n_fechamentos_05s`, `ear_media`, `ear_dp`, `amp_media`, `vel_media`, `frac_bocejo`, `taxa_rosto` | em segundos, Hz e razões: não dependem do fps (que muda entre vídeos do mesmo sujeito na UTA) |
| Modelos | RandomForest (300 árvores, `class_weight="balanced"`, `random_state=42`); LogisticRegression (`class_weight="balanced"`, `max_iter=5000`) após `StandardScaler`; os dois após `SimpleImputer(median)` ajustado no treino | hiperparâmetros fixos (os da rodada preliminar); nenhum ajuste olhando o teste. A imputação é necessária porque janelas de 10 s sem piscada não têm duração/amplitude/velocidade (DMD 9,6% das janelas do principal; UTA 34,9% das avaliáveis) |
| Limiar | (i) **0,5 fixo a priori**; (ii) **calibrado no treino**: máxima acurácia balanceada nos escores fora da amostra do próprio treino (C: LOSO do DMD, sem olhar a UTA; A: LOSO aninhado; B: validação interna nos 4 folds de treino); empate → o mais próximo de 0,5 | a mudança de domínio desloca o limiar ótimo; escolher o limiar no DMD é a única calibração honesta de C. O limiar que maximiza a acurácia na própria UTA aparece só como diagnóstico (seção 6), nunca como resultado |
| Validação | A: LOSO nos 13 sujeitos do DMD (as 2 sessões s5 de gB_10, gF_23 e gZ_33 ficam juntas; gE_28, sem s5, fica fora); B: os **5 folds oficiais** (teste = fold k, treino = os outros 4; mesma partição do Exp. 1); C: um modelo treinado em todo o DMD e aplicado a todas as janelas avaliáveis da UTA | por sujeito, sem vazamento de identidade (Exp. 2) |
| Por vídeo | escore do **vídeo lógico** = média dos escores das janelas (as partes `_p1`/`_p2` de F3_32_10 e F5_49_10 contam como um vídeo); limiar 0,5 (regra da Entrega 01) e o calibrado; AUC dos escores médios; a acurácia por voto majoritário serve só para comparar com o Exp. 1 | na UTA cada vídeo tem um rótulo |
| IC 95% | bootstrap percentil sobre **sujeitos**, 1000 reamostragens, seed 42 (UTA sorteada antes do DMD); os modelos não são retreinados na reamostragem | janelas do mesmo sujeito são correlacionadas (sobreposição de 5 s) |
| Quedas | PP1 = **B − C**, pareada (as mesmas reamostragens de sujeitos da UTA em B e C); choque de realidade = **A − C** e laboratório → real com treino real = A − B, com reamostragens independentes das duas bases | especificação |
| Controle do vídeo de referência | modelos de B3 (cada sujeito com o modelo do fold em que foi teste) e de C aplicados às janelas KSS 5: AUC KSS 10 × 5 e KSS 5 × 0 | as janelas alerta da UTA saem do mesmo vídeo da referência; as sonolentas, de outro (seção 9) |
| Importância | queda da AUC agrupada ao embaralhar a característica entre as janelas do MESMO sujeito de teste (20 repetições; A e B com o modelo de cada partição, C com o modelo único); AUC univariada por base; coeficientes padronizados da LR | a mesma permutação dentro do sujeito em A, B e C |
| Sensibilidades | sem normalização; alerta da s1; s5 sem sinal como alerta; janelas de 5 s e 15 s; referência do DMD na s2 (espelha a UTA); sem gE_29 (falha do detector) | pedidas na especificação, mais duas motivadas pelos dados (seções 8 e 11) |

## 3. Tamanho da amostra

| Conjunto | Unidades | Sonolento | Alerta / não sonolento | Sujeitos | Sujeitos com as duas classes |
|---|---:|---:|---:|---:|---:|
| DMD, variante `principal` (janelas) | 510 | 423 | 87 | 13 | 13 |
| DMD, variante `sem_normalizacao` (janelas) | 510 | 423 | 87 | 13 | 13 |
| DMD, variante `alerta_s1` (janelas) | 525 | 423 | 102 | 13 | 13 |
| DMD, variante `s5_sem_sinal_alerta` (janelas) | 656 | 423 | 233 | 13 | 13 |
| DMD, variante `janela_5s` (janelas) | 545 | 308 | 237 | 13 | 13 |
| DMD, variante `janela_15s` (janelas) | 474 | 466 | 8 | 13 | 5 |
| DMD, variante `ref_s2_espelho` (janelas) | 472 | 423 | 49 | 13 | 13 |
| DMD, variante `sem_gE_29` (janelas) | 474 | 393 | 81 | 12 | 12 |
| UTA B3 (janelas avaliáveis, 10 s) | 12487 | 7581 | 4906 | 60 | 60 |
| UTA B1 (janelas avaliáveis, 10 s) | 19955 | 15049 | 4906 | 60 | 60 |
| UTA B2 (janelas avaliáveis, 10 s) | 19955 | 7581 | 12374 | 60 | 60 |
| UTA B3 (vídeos lógicos) | 120 | 60 | 60 | 60 |  |
| UTA B1 (vídeos lógicos) | 180 | 120 | 60 | 60 |  |
| UTA B2 (vídeos lógicos) | 180 | 60 | 120 | 60 |  |
| UTA B3 (janelas de 5 s) | 12609 | 7643 | 4966 | 60 | 60 |
| UTA B3 (janelas de 15 s) | 12365 | 7519 | 4846 | 60 | 60 |
| UTA B3 com referência e fronteira (sensibilidade) | 14987 | 7581 | 7406 | 60 | 60 |

**Vídeos sem nenhuma piscada detectada** (2 de 180): `F2_15_10` (122 janelas; KSS 10); `F3_27_5` (124 janelas; KSS 5). O EAR nunca cruza o limiar calibrado no último terço do vídeo 0 do sujeito (o nível de olho aberto subiu; `results/exp5_caracteristicas.md`, seção 5.5): nessas janelas PERCLOS, taxa de piscadas e fechamentos valem 0 e duração/amplitude/velocidade são imputadas. Eles ficam na avaliação (é o que o app veria); os escores deles estão em `results/exp5b_predicoes_video.csv`.

Com janelas de 10 s o DMD passa a ter 87 janelas alerta (safe_drive da s2) contra 423 sonolentas, com os 13 sujeitos nas duas classes (na rodada preliminar: 18 alerta de 6 sujeitos, sobrepostas à referência). Com 15 s sobram 8 janelas alerta de 5 sujeitos: essa sensibilidade é pouco informativa no DMD.

## 4. Resultados por janela (limiar 0,5)

### 4.1 Variante `principal` — janela de 10 s; DMD: alerta = safe_drive da s2, sonolento = s5 com sinal anotado; características centradas na referência do sujeito (`_c`; DMD: safe_drive da s3; UTA: último terço do vídeo 0)

| Experimento | Modelo | Esquema | Acur. bal. | F1 macro | AUC | AUC média intra-sujeito | Sens. | Espec. | TN / FP / FN / TP |
|---|---|---|---|---|---|---|---|---|---|
| DMD → DMD (A) | RF | DMD | 76,4 [68,3; 83,7] | 72,0 [62,5; 79,6] | 82,4 [73,6; 89,7] | 85,5 [80,0; 90,4] | 83,9 | 69,0 | 60 / 27 / 68 / 355 |
| DMD → DMD (A) | LR | DMD | 78,6 [70,1; 85,3] | 69,4 [59,1; 77,5] | 83,5 [76,1; 88,9] | 86,7 [80,1; 92,6] | 75,7 | 81,6 | 71 / 16 / 103 / 320 |
| UTA → UTA (B) | RF | B3 | 74,2 [70,9; 77,6] | 73,3 [69,7; 76,8] | 80,1 [76,3; 83,8] | 81,2 [77,6; 84,8] | 72,5 | 75,9 | 3723 / 1183 / 2085 / 5496 |
| UTA → UTA (B) | LR | B3 | 72,0 [68,9; 74,9] | 69,2 [65,3; 72,6] | 74,4 [69,7; 78,4] | 75,1 [69,9; 79,5] | 59,0 | 85,0 | 4168 / 738 / 3112 / 4469 |
| DMD → UTA (C) | RF | B3 | 65,2 [61,7; 68,6] | 61,5 [57,4; 65,6] | 69,2 [63,7; 74,3] | 70,0 [64,4; 75,0] | 48,4 | 82,0 | 4021 / 885 / 3913 / 3668 |
| DMD → UTA (C) | LR | B3 | 64,0 [60,7; 67,4] | 56,3 [51,1; 61,3] | 64,5 [58,1; 70,9] | 68,5 [61,9; 74,5] | 33,7 | 94,4 | 4629 / 277 / 5025 / 2556 |
| UTA → UTA (B) | RF | B1 | 67,5 [64,5; 70,4] | 65,7 [62,8; 68,6] | 74,9 [71,2; 78,3] | 76,1 [72,5; 79,6] | 77,0 | 58,0 | 2845 / 2061 / 3466 / 11583 |
| UTA → UTA (B) | LR | B1 | 68,1 [65,2; 70,7] | 60,1 [56,5; 63,5] | 69,7 [65,6; 73,5] | 70,7 [66,2; 74,7] | 56,5 | 79,6 | 3907 / 999 / 6547 / 8502 |
| DMD → UTA (C) | RF | B1 | 61,1 [58,1; 64,2] | 50,0 [46,2; 53,6] | 64,6 [59,9; 69,1] | 64,8 [60,2; 69,2] | 40,2 | 82,0 | 4021 / 885 / 8995 / 6054 |
| DMD → UTA (C) | LR | B1 | 60,8 [58,4; 63,4] | 43,6 [39,4; 48,0] | 62,4 [57,3; 67,5] | 65,1 [59,8; 70,1] | 27,2 | 94,4 | 4629 / 277 / 10962 / 4087 |
| UTA → UTA (B) | RF | B2 | 62,0 [59,1; 64,8] | 62,2 [59,2; 64,9] | 66,7 [63,6; 70,0] | 67,8 [64,2; 71,4] | 49,8 | 74,3 | 9194 / 3180 / 3809 / 3772 |
| UTA → UTA (B) | LR | B2 | 62,6 [59,6; 65,6] | 62,8 [59,7; 65,9] | 65,1 [60,4; 69,1] | 64,8 [59,1; 69,5] | 47,5 | 77,7 | 9611 / 2763 / 3980 / 3601 |
| DMD → UTA (C) | RF | B2 | 61,0 [58,0; 64,1] | 61,1 [58,0; 64,2] | 63,8 [59,3; 68,0] | 64,9 [60,2; 69,2] | 48,4 | 73,6 | 9103 / 3271 / 3913 / 3668 |
| DMD → UTA (C) | LR | B2 | 59,6 [56,6; 62,5] | 59,2 [55,2; 62,7] | 59,5 [54,0; 64,8] | 63,6 [57,7; 68,9] | 33,7 | 85,4 | 10566 / 1808 / 5025 / 2556 |

### 4.2 Variante `sem_normalizacao` — como a principal, com as características brutas (sem normalização por sujeito)

| Experimento | Modelo | Esquema | Acur. bal. | F1 macro | AUC | AUC média intra-sujeito | Sens. | Espec. | TN / FP / FN / TP |
|---|---|---|---|---|---|---|---|---|---|
| DMD → DMD (A) | RF | DMD | 71,5 [60,7; 80,6] | 67,7 [56,3; 77,3] | 81,6 [70,3; 90,1] | 82,8 [73,1; 90,8] | 82,0 | 60,9 | 53 / 34 / 76 / 347 |
| DMD → DMD (A) | LR | DMD | 77,1 [69,8; 82,9] | 67,1 [58,4; 74,0] | 81,1 [71,6; 88,8] | 83,0 [75,5; 89,0] | 72,6 | 81,6 | 71 / 16 / 116 / 307 |
| UTA → UTA (B) | RF | B3 | 62,9 [59,3; 66,2] | 62,8 [59,1; 66,2] | 69,4 [64,8; 73,7] | 70,5 [65,6; 75,0] | 69,4 | 56,4 | 2765 / 2141 / 2319 / 5262 |
| UTA → UTA (B) | LR | B3 | 68,1 [64,0; 72,4] | 65,6 [61,1; 70,2] | 72,5 [67,7; 77,3] | 74,9 [69,6; 79,7] | 56,6 | 79,6 | 3907 / 999 / 3292 / 4289 |
| DMD → UTA (C) | RF | B3 | 63,1 [59,8; 66,3] | 60,8 [57,2; 64,3] | 67,2 [62,5; 71,6] | 68,6 [64,2; 72,9] | 52,2 | 74,0 | 3630 / 1276 / 3620 / 3961 |
| DMD → UTA (C) | LR | B3 | 58,1 [55,5; 61,2] | 45,6 [40,9; 50,5] | 65,2 [60,6; 69,0] | 66,4 [61,0; 71,2] | 18,5 | 97,7 | 4795 / 111 / 6176 / 1405 |
| UTA → UTA (B) | RF | B1 | 57,4 [55,1; 59,8] | 57,5 [55,0; 59,9] | 64,3 [60,7; 68,2] | 64,9 [61,1; 68,8] | 79,4 | 35,4 | 1738 / 3168 / 3099 / 11950 |
| UTA → UTA (B) | LR | B1 | 64,0 [60,5; 67,6] | 56,4 [52,7; 60,1] | 68,3 [64,3; 72,4] | 70,5 [66,1; 74,5] | 53,0 | 74,9 | 3677 / 1229 / 7070 / 7979 |
| DMD → UTA (C) | RF | B1 | 60,2 [57,3; 63,1] | 51,8 [48,7; 55,0] | 63,4 [59,2; 67,2] | 64,7 [60,4; 68,6] | 46,4 | 74,0 | 3630 / 1276 / 8064 / 6985 |
| DMD → UTA (C) | LR | B1 | 55,2 [53,5; 57,4] | 32,2 [28,9; 35,9] | 61,1 [57,4; 64,5] | 61,7 [57,7; 65,5] | 12,7 | 97,7 | 4795 / 111 / 13133 / 1916 |
| UTA → UTA (B) | RF | B2 | 58,4 [55,7; 61,0] | 58,5 [55,7; 61,0] | 60,8 [57,1; 64,2] | 61,5 [57,8; 64,9] | 47,5 | 69,3 | 8579 / 3795 / 3980 / 3601 |
| UTA → UTA (B) | LR | B2 | 61,8 [58,3; 65,3] | 62,0 [58,4; 65,4] | 65,0 [60,2; 69,5] | 66,7 [61,7; 71,3] | 48,9 | 74,7 | 9242 / 3132 / 3875 / 3706 |
| DMD → UTA (C) | RF | B2 | 58,7 [56,1; 61,4] | 58,5 [55,9; 61,0] | 61,8 [57,8; 65,4] | 63,3 [59,2; 67,0] | 52,2 | 65,2 | 8074 / 4300 / 3620 / 3961 |
| DMD → UTA (C) | LR | B2 | 56,8 [54,4; 59,6] | 53,4 [49,3; 57,7] | 61,4 [57,7; 65,2] | 62,6 [57,6; 67,2] | 18,5 | 95,0 | 11752 / 622 / 6176 / 1405 |

"AUC média intra-sujeito" = média, entre os sujeitos com as duas classes, da AUC calculada só com as janelas de cada sujeito: mede se o modelo ordena os estados DENTRO da mesma pessoa (o que importa num app calibrado por usuário), sem exigir que a escala dos escores seja comparável entre pessoas. Sens. = acerto nas janelas sonolentas; Espec. = acerto nas alertas.

## 5. Resultados por vídeo (UTA; escore do vídeo lógico = média dos escores das janelas)

| Variante | Experimento | Modelo | Esquema | Acurácia | Acur. bal. | F1 macro | AUC | Acurácia por voto | TN / FP / FN / TP |
|---|---|---|---|---|---|---|---|---|---|
| `principal` | UTA → UTA (B) | RF | B3 | 85,8 [80,0; 91,7] | 85,8 [80,0; 91,7] | 85,8 [79,5; 91,7] | 92,4 [87,9; 96,2] | 85,8 [80,0; 91,7] | 54 / 6 / 11 / 49 |
| `principal` | UTA → UTA (B) | LR | B3 | 85,8 [80,0; 90,8] | 85,8 [80,0; 90,8] | 85,6 [79,3; 90,8] | 89,0 [82,2; 94,9] | 84,2 [78,3; 89,2] | 59 / 1 / 16 / 44 |
| `principal` | DMD → UTA (C) | RF | B3 | 73,3 [67,5; 79,2] | 73,3 [67,5; 79,2] | 72,4 [65,6; 78,7] | 80,5 [73,0; 87,2] | 69,2 [63,3; 75,8] | 55 / 5 / 27 / 33 |
| `principal` | DMD → UTA (C) | LR | B3 | 65,8 [60,0; 71,7] | 65,8 [60,0; 71,7] | 61,3 [52,4; 69,2] | 76,0 [67,8; 84,7] | 65,0 [59,2; 70,8] | 60 / 0 / 41 / 19 |
| `principal` | UTA → UTA (B) | RF | B1 | 80,6 [76,1; 85,6] | 76,2 [70,4; 82,5] | 77,2 [71,3; 83,0] | 87,5 [82,1; 92,1] | 78,9 [73,9; 83,9] | 38 / 22 / 13 / 107 |
| `principal` | UTA → UTA (B) | LR | B1 | 78,3 [71,7; 84,4] | 82,5 [77,1; 87,5] | 77,8 [71,6; 83,6] | 82,3 [76,0; 88,8] | 73,3 [66,7; 79,4] | 57 / 3 / 36 / 84 |
| `principal` | DMD → UTA (C) | RF | B1 | 57,2 [51,7; 63,3] | 65,8 [60,8; 70,8] | 57,2 [51,2; 63,3] | 74,1 [66,9; 80,8] | 53,9 [48,3; 59,4] | 55 / 5 / 72 / 48 |
| `principal` | DMD → UTA (C) | LR | B1 | 48,3 [42,8; 53,9] | 61,3 [57,1; 65,4] | 46,5 [39,3; 53,1] | 72,7 [65,7; 79,9] | 47,2 [42,2; 52,8] | 60 / 0 / 93 / 27 |
| `principal` | UTA → UTA (B) | RF | B2 | 75,6 [70,6; 80,6] | 68,3 [62,1; 74,6] | 69,5 [62,6; 76,2] | 79,0 [74,6; 83,9] | 74,4 [70,0; 78,9] | 108 / 12 / 32 / 28 |
| `principal` | UTA → UTA (B) | LR | B2 | 77,2 [72,8; 82,2] | 72,5 [66,7; 78,3] | 73,3 [67,5; 79,1] | 75,7 [68,8; 82,0] | 75,0 [70,6; 80,0] | 104 / 16 / 25 / 35 |
| `principal` | DMD → UTA (C) | RF | B2 | 73,9 [68,9; 78,9] | 69,2 [62,9; 75,0] | 69,7 [63,5; 75,4] | 74,2 [67,8; 80,2] | 71,7 [66,7; 77,2] | 100 / 20 / 27 / 33 |
| `principal` | DMD → UTA (C) | LR | B2 | 72,8 [68,9; 76,7] | 62,5 [56,7; 67,9] | 62,9 [55,0; 69,4] | 68,8 [61,9; 75,8] | 72,8 [68,9; 76,7] | 112 / 8 / 41 / 19 |
| `sem_normalizacao` | UTA → UTA (B) | RF | B3 | 75,8 [68,3; 82,5] | 75,8 [68,3; 82,5] | 75,8 [68,3; 82,5] | 80,8 [72,9; 88,1] | 78,3 [71,6; 84,2] | 42 / 18 / 11 / 49 |
| `sem_normalizacao` | UTA → UTA (B) | LR | B3 | 77,5 [70,0; 85,0] | 77,5 [70,0; 85,0] | 77,3 [69,7; 84,8] | 82,8 [75,7; 89,7] | 72,5 [65,8; 80,0] | 52 / 8 / 19 / 41 |
| `sem_normalizacao` | DMD → UTA (C) | RF | B3 | 70,0 [62,5; 76,7] | 70,0 [62,5; 76,7] | 68,5 [60,2; 76,0] | 80,6 [73,1; 87,5] | 68,3 [61,7; 75,0] | 55 / 5 / 31 / 29 |
| `sem_normalizacao` | DMD → UTA (C) | LR | B3 | 56,7 [52,5; 61,7] | 56,7 [52,5; 61,7] | 46,6 [38,7; 55,1] | 81,6 [75,1; 87,6] | 55,0 [51,7; 59,2] | 60 / 0 / 52 / 8 |
| `sem_normalizacao` | UTA → UTA (B) | RF | B1 | 68,9 [65,0; 72,8] | 56,2 [51,3; 61,7] | 54,2 [46,5; 61,4] | 74,8 [67,7; 82,2] | 67,2 [63,9; 70,6] | 11 / 49 / 7 / 113 |
| `sem_normalizacao` | UTA → UTA (B) | LR | B1 | 69,4 [62,8; 76,7] | 72,5 [66,2; 78,8] | 68,7 [62,0; 75,5] | 78,3 [72,0; 84,6] | 62,8 [56,1; 69,4] | 49 / 11 / 44 / 76 |
| `sem_normalizacao` | DMD → UTA (C) | RF | B1 | 55,6 [48,9; 61,7] | 64,6 [58,8; 70,0] | 55,4 [48,4; 61,7] | 74,6 [67,8; 81,2] | 56,1 [49,4; 62,8] | 55 / 5 / 75 / 45 |
| `sem_normalizacao` | DMD → UTA (C) | LR | B1 | 38,3 [35,6; 41,7] | 53,8 [51,7; 56,2] | 33,0 [28,6; 37,8] | 74,4 [68,4; 80,5] | 37,2 [35,0; 40,0] | 60 / 0 / 111 / 9 |
| `sem_normalizacao` | UTA → UTA (B) | RF | B2 | 72,8 [67,2; 78,3] | 64,2 [57,5; 70,8] | 65,0 [57,2; 72,3] | 71,6 [64,6; 78,4] | 74,4 [70,0; 78,9] | 108 / 12 / 37 / 23 |
| `sem_normalizacao` | UTA → UTA (B) | LR | B2 | 72,8 [67,2; 78,3] | 68,3 [62,1; 75,0] | 68,7 [62,2; 75,1] | 74,5 [67,5; 81,2] | 71,7 [65,6; 77,8] | 98 / 22 / 27 / 33 |
| `sem_normalizacao` | DMD → UTA (C) | RF | B2 | 71,1 [65,6; 76,1] | 65,4 [59,6; 71,2] | 66,0 [59,6; 72,0] | 72,8 [66,1; 78,8] | 68,3 [62,2; 73,3] | 99 / 21 / 31 / 29 |
| `sem_normalizacao` | DMD → UTA (C) | LR | B2 | 70,6 [67,8; 73,9] | 56,2 [52,1; 60,8] | 52,5 [44,9; 59,6] | 74,9 [68,8; 80,8] | 69,4 [66,7; 72,2] | 119 / 1 / 52 / 8 |

## 6. Limiar: 0,5 fixo × calibrado só com o treino (B3)

A mudança de domínio desloca a escala dos escores. A única calibração honesta de C é a que usa só o DMD: o limiar que maximiza a acurácia balanceada nos escores fora da amostra do LOSO do DMD (os de A). Para comparar de igual para igual, A e B também recebem um limiar escolhido só com o próprio treino (A: LOSO aninhado, um limiar por sujeito de teste; B: validação interna nos 4 folds de treino, um limiar por fold). Limiares em `results/exp5b_limiares.csv`.

| Variante | Modelo | Limiar de C (LOSO do DMD) | A: acur. bal. 0,5 → calibrado | B: acur. bal. 0,5 → calibrado (limiares por fold) | C: acur. bal. 0,5 → calibrado | C: sens. / espec. no 0,5 | C: sens. / espec. calibrado | C: acurácia por vídeo 0,5 → calibrado |
|---|---|---|---|---|---|---|---|---|
| `principal` | RF | 0,662 | 76,4 [68,3; 83,7] → 78,6 [72,6; 84,2] | 74,2 [70,9; 77,6] → 74,0 [70,7; 77,4] (0,492–0,595) | 65,2 [61,7; 68,6] → 62,9 [60,0; 66,0] | 48,4 / 82,0 | 37,0 / 88,7 | 73,3 [67,5; 79,2] → 60,8 [55,8; 65,8] |
| `principal` | LR | 0,535 | 78,6 [70,1; 85,3] → 78,7 [72,1; 83,8] | 72,0 [68,9; 74,9] → 72,2 [69,1; 75,2] (0,477–0,503) | 64,0 [60,7; 67,4] → 63,3 [60,2; 66,7] | 33,7 / 94,4 | 30,8 / 95,8 | 65,8 [60,0; 71,7] → 64,2 [58,3; 70,0] |
| `sem_normalizacao` | RF | 0,842 | 71,5 [60,7; 80,6] → 76,1 [67,2; 83,4] | 62,9 [59,3; 66,2] → 63,7 [59,8; 67,4] (0,605–0,765) | 63,1 [59,8; 66,3] → 54,7 [53,1; 56,7] | 52,2 / 74,0 | 11,0 / 98,3 | 70,0 [62,5; 76,7] → 50,8 [50,0; 52,5] |
| `sem_normalizacao` | LR | 0,400 | 77,1 [69,8; 82,9] → 76,3 [68,7; 82,3] | 68,1 [64,0; 72,4] → 67,9 [63,9; 72,1] (0,474–0,545) | 58,1 [55,5; 61,2] → 59,4 [56,4; 62,6] | 18,5 / 97,7 | 22,4 / 96,4 | 56,7 [52,5; 61,7] → 57,5 [52,5; 62,5] |

**Diagnóstico (não é resultado):** o limiar que maximiza a acurácia balanceada de C na própria UTA olha o rótulo do teste e mede só o teto de uma recalibração perfeita. Ele mostra quanto da queda no limiar é calibração e quanto é ordenação (a AUC não depende do limiar).

| Variante | Modelo | Limiar oráculo na UTA | C: acur. bal. no oráculo | C: acur. bal. no limiar do DMD | Limiar do DMD |
|---|---|---|---|---|---|
| `principal` | RF | 0,388 | 65,9 | 62,9 | 0,662 |
| `principal` | LR | 0,357 | 65,9 | 63,3 | 0,535 |
| `sem_normalizacao` | RF | 0,502 | 63,1 | 54,7 | 0,842 |
| `sem_normalizacao` | LR | 0,185 | 63,1 | 59,4 | 0,400 |

## 7. Quedas: PP1 (B − C), choque de realidade (A − C) e laboratório → real com treino real (A − B)

B − C é pareada (as mesmas reamostragens de sujeitos da UTA entram em B e em C); A − C e A − B combinam reamostragens independentes do DMD e da UTA. Em pp, com IC 95%; positivo = o segundo termo é pior. Com limiar calibrado, cada experimento usa o limiar escolhido no próprio treino (seção 6).

| Comparação | Variante | Modelo | Esquema | Nível | Limiar | Δ acur. bal. | Δ F1 macro | Δ AUC | Δ AUC intra-sujeito | Δ acurácia | Δ acurácia por voto |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B − C | `principal` | RF | B3 | janela | 0,5 | 9,0 [5,8; 12,2] | 11,7 [8,3; 15,6] | 10,9 [6,7; 15,7] | 11,2 [6,8; 15,7] | — | — |
| B − C | `principal` | LR | B3 | janela | 0,5 | 7,9 [4,8; 10,9] | 12,8 [8,6; 17,2] | 9,8 [3,7; 16,0] | 6,7 [0,1; 13,0] | — | — |
| B − C | `principal` | RF | B3 | janela | calibrado | 11,1 [8,0; 14,2] | 15,8 [11,9; 19,5] | — | — | — | — |
| B − C | `principal` | LR | B3 | janela | calibrado | 8,9 [5,6; 11,9] | 15,0 [10,6; 19,5] | — | — | — | — |
| B − C | `principal` | RF | B3 | video | 0,5 | 12,5 [5,0; 19,2] | 13,4 [5,5; 20,8] | 11,9 [5,6; 18,8] | — | 12,5 [5,0; 19,2] | 16,7 [9,2; 23,3] |
| B − C | `principal` | LR | B3 | video | 0,5 | 20,0 [12,5; 26,7] | 24,3 [15,2; 34,0] | 13,0 [5,8; 20,8] | — | 20,0 [12,5; 26,7] | 19,2 [11,7; 26,7] |
| B − C | `principal` | RF | B3 | video | calibrado | 21,7 [14,2; 29,2] | 28,6 [19,4; 38,3] | — | — | 21,7 [14,2; 29,2] | 21,7 [14,2; 30,0] |
| B − C | `principal` | LR | B3 | video | calibrado | 20,8 [12,5; 28,3] | 25,9 [16,2; 35,9] | — | — | 20,8 [12,5; 28,3] | 24,2 [16,7; 31,7] |
| B − C | `principal` | RF | B1 | janela | 0,5 | 6,4 [2,8; 9,8] | 15,8 [11,9; 19,8] | 10,4 [6,2; 14,6] | 11,3 [6,9; 15,5] | — | — |
| B − C | `principal` | LR | B1 | janela | 0,5 | 7,3 [4,6; 9,8] | 16,4 [12,4; 20,4] | 7,3 [2,9; 11,8] | 5,6 [0,5; 10,8] | — | — |
| B − C | `principal` | RF | B1 | video | 0,5 | 10,4 [3,3; 17,5] | 20,0 [11,7; 27,7] | 13,4 [7,0; 20,0] | — | 23,3 [15,6; 30,0] | 25,0 [17,8; 32,2] |
| B − C | `principal` | LR | B1 | video | 0,5 | 21,2 [15,4; 27,1] | 31,3 [23,8; 39,1] | 9,7 [3,7; 16,1] | — | 30,0 [22,8; 37,2] | 26,1 [18,9; 32,8] |
| B − C | `principal` | RF | B2 | janela | 0,5 | 1,1 [−1,1; 3,4] | 1,1 [−1,1; 3,4] | 2,9 [−0,2; 6,6] | 2,9 [−0,8; 7,4] | — | — |
| B − C | `principal` | LR | B2 | janela | 0,5 | 3,0 [0,2; 5,9] | 3,7 [0,5; 7,1] | 5,6 [0,0; 11,5] | 1,1 [−5,2; 7,2] | — | — |
| B − C | `principal` | RF | B2 | video | 0,5 | −0,8 [−7,1; 5,4] | −0,2 [−6,8; 6,3] | 4,8 [−0,3; 10,3] | — | 1,7 [−3,3; 7,2] | 2,8 [−2,8; 7,8] |
| B − C | `principal` | LR | B2 | video | 0,5 | 10,0 [3,3; 16,7] | 10,4 [3,3; 18,9] | 6,9 [−0,7; 14,5] | — | 4,4 [−1,1; 10,0] | 2,2 [−3,3; 7,8] |
| B − C | `sem_normalizacao` | RF | B3 | janela | 0,5 | −0,2 [−3,4; 2,9] | 2,0 [−1,2; 5,3] | 2,1 [−1,2; 5,8] | 1,9 [−1,7; 5,6] | — | — |
| B − C | `sem_normalizacao` | LR | B3 | janela | 0,5 | 10,0 [5,6; 14,0] | 20,0 [15,1; 24,8] | 7,3 [3,0; 11,7] | 8,5 [3,0; 14,2] | — | — |
| B − C | `sem_normalizacao` | RF | B3 | janela | calibrado | 9,0 [5,5; 12,4] | 21,1 [17,0; 24,9] | — | — | — | — |
| B − C | `sem_normalizacao` | LR | B3 | janela | calibrado | 8,5 [4,2; 12,6] | 16,7 [12,0; 21,3] | — | — | — | — |
| B − C | `sem_normalizacao` | RF | B3 | video | 0,5 | 5,8 [−3,3; 15,8] | 7,2 [−2,8; 17,9] | 0,2 [−5,6; 6,3] | — | 5,8 [−3,3; 15,8] | 10,0 [1,7; 18,3] |
| B − C | `sem_normalizacao` | LR | B3 | video | 0,5 | 20,8 [12,5; 29,2] | 30,7 [20,4; 40,6] | 1,2 [−5,4; 7,9] | — | 20,8 [12,5; 29,2] | 17,5 [9,2; 25,8] |
| B − C | `sem_normalizacao` | RF | B3 | video | calibrado | 15,8 [9,2; 22,5] | 29,3 [20,7; 37,5] | — | — | 15,8 [9,2; 22,5] | 16,7 [10,0; 24,2] |
| B − C | `sem_normalizacao` | LR | B3 | video | calibrado | 20,0 [12,5; 28,3] | 27,5 [18,1; 37,4] | — | — | 20,0 [12,5; 28,3] | 13,3 [5,8; 21,7] |
| B − C | `sem_normalizacao` | RF | B1 | janela | 0,5 | −2,8 [−5,9; 0,4] | 5,6 [2,2; 8,7] | 0,9 [−3,2; 5,0] | 0,3 [−4,0; 4,5] | — | — |
| B − C | `sem_normalizacao` | LR | B1 | janela | 0,5 | 8,7 [4,8; 12,2] | 24,2 [19,8; 28,4] | 7,2 [3,3; 11,1] | 8,7 [4,3; 12,9] | — | — |
| B − C | `sem_normalizacao` | RF | B1 | video | 0,5 | −8,3 [−15,8; −0,4] | −1,2 [−11,9; 10,0] | 0,2 [−6,9; 7,0] | — | 13,3 [5,6; 21,1] | 11,1 [3,9; 17,8] |
| B − C | `sem_normalizacao` | LR | B1 | video | 0,5 | 18,8 [11,7; 25,4] | 35,8 [27,8; 43,5] | 3,9 [−2,5; 10,2] | — | 31,1 [23,9; 38,3] | 25,6 [18,3; 32,8] |
| B − C | `sem_normalizacao` | RF | B2 | janela | 0,5 | −0,3 [−2,7; 2,1] | −0,1 [−2,5; 2,3] | −1,1 [−4,0; 2,1] | −1,8 [−4,9; 1,4] | — | — |
| B − C | `sem_normalizacao` | LR | B2 | janela | 0,5 | 5,0 [2,2; 7,7] | 8,6 [5,0; 11,9] | 3,6 [−0,9; 7,9] | 4,1 [−1,9; 9,6] | — | — |
| B − C | `sem_normalizacao` | RF | B2 | video | 0,5 | −1,2 [−7,5; 6,3] | −1,0 [−8,2; 7,3] | −1,2 [−7,3; 5,5] | — | 1,7 [−3,3; 8,3] | 6,1 [0,0; 12,8] |
| B − C | `sem_normalizacao` | LR | B2 | video | 0,5 | 12,1 [4,6; 19,2] | 16,2 [6,7; 25,3] | −0,4 [−7,1; 6,5] | — | 2,2 [−4,4; 8,3] | 2,2 [−3,9; 7,8] |
| A − C | `principal` | RF | B3 | janela | 0,5 | 11,3 [2,2; 19,8] | 10,5 [0,3; 19,7] | 13,1 [3,5; 22,6] | — | — | — |
| A − C | `principal` | LR | B3 | janela | 0,5 | 14,6 [5,6; 22,0] | 13,0 [2,3; 22,7] | 18,9 [9,7; 27,8] | — | — | — |
| A − C | `principal` | RF | B3 | janela | calibrado | 15,8 [8,8; 22,1] | 13,3 [4,7; 21,0] | — | — | — | — |
| A − C | `principal` | LR | B3 | janela | calibrado | 15,4 [7,8; 21,4] | 12,1 [2,5; 20,7] | — | — | — | — |
| A − C | `principal` | RF | B1 | janela | 0,5 | 15,4 [6,9; 23,3] | 22,0 [12,0; 30,9] | 17,8 [8,7; 27,2] | — | — | — |
| A − C | `principal` | LR | B1 | janela | 0,5 | 17,9 [9,1; 24,9] | 25,7 [15,0; 35,0] | 21,1 [12,5; 28,6] | — | — | — |
| A − C | `principal` | RF | B1 | janela | calibrado | 19,7 [13,1; 25,8] | 26,1 [18,0; 33,6] | — | — | — | — |
| A − C | `principal` | LR | B1 | janela | calibrado | 18,6 [11,3; 24,2] | 25,0 [15,3; 32,8] | — | — | — | — |
| A − C | `principal` | RF | B2 | janela | 0,5 | 15,5 [6,9; 23,7] | 10,9 [1,0; 19,3] | 18,6 [9,0; 27,4] | — | — | — |
| A − C | `principal` | LR | B2 | janela | 0,5 | 19,1 [10,2; 26,2] | 10,2 [−0,3; 19,2] | 24,0 [15,3; 32,1] | — | — | — |
| A − C | `principal` | RF | B2 | janela | calibrado | 18,8 [12,2; 25,2] | 10,1 [1,9; 17,4] | — | — | — | — |
| A − C | `principal` | LR | B2 | janela | calibrado | 19,5 [12,0; 25,3] | 8,3 [−0,9; 15,9] | — | — | — | — |
| A − C | `sem_normalizacao` | RF | B3 | janela | 0,5 | 8,4 [−2,4; 18,3] | 6,9 [−4,6; 17,2] | 14,4 [2,6; 24,0] | — | — | — |
| A − C | `sem_normalizacao` | LR | B3 | janela | 0,5 | 19,0 [11,4; 25,5] | 21,4 [11,9; 30,1] | 15,9 [6,4; 24,9] | — | — | — |
| A − C | `sem_normalizacao` | RF | B3 | janela | calibrado | 21,4 [12,5; 28,9] | 24,5 [13,3; 33,9] | — | — | — | — |
| A − C | `sem_normalizacao` | LR | B3 | janela | calibrado | 16,9 [9,0; 23,8] | 17,9 [7,5; 26,8] | — | — | — | — |
| A − C | `sem_normalizacao` | RF | B1 | janela | 0,5 | 11,3 [0,2; 20,8] | 15,9 [4,1; 26,2] | 18,2 [6,3; 27,4] | — | — | — |
| A − C | `sem_normalizacao` | LR | B1 | janela | 0,5 | 21,9 [14,5; 28,1] | 34,8 [25,4; 42,5] | 19,9 [10,6; 28,5] | — | — | — |
| A − C | `sem_normalizacao` | RF | B1 | janela | calibrado | 22,9 [14,1; 30,6] | 35,7 [24,8; 45,0] | — | — | — | — |
| A − C | `sem_normalizacao` | LR | B1 | janela | calibrado | 20,1 [12,3; 26,4] | 31,5 [21,2; 39,7] | — | — | — | — |
| A − C | `sem_normalizacao` | RF | B2 | janela | 0,5 | 12,7 [2,3; 22,1] | 9,1 [−2,0; 19,0] | 19,8 [8,2; 29,3] | — | — | — |
| A − C | `sem_normalizacao` | LR | B2 | janela | 0,5 | 20,3 [12,6; 26,7] | 13,7 [4,2; 21,9] | 19,7 [9,9; 28,3] | — | — | — |
| A − C | `sem_normalizacao` | RF | B2 | janela | calibrado | 22,4 [13,5; 29,8] | 15,7 [4,5; 25,1] | — | — | — | — |
| A − C | `sem_normalizacao` | LR | B2 | janela | calibrado | 18,7 [10,8; 25,5] | 11,1 [1,0; 19,4] | — | — | — | — |
| A − B | `principal` | RF | B3 | janela | 0,5 | 2,3 [−6,5; 10,4] | −1,3 [−11,4; 7,9] | 2,2 [−7,1; 10,9] | — | — | — |
| A − B | `principal` | LR | B3 | janela | 0,5 | 6,7 [−2,2; 14,0] | 0,2 [−10,0; 9,1] | 9,1 [1,3; 16,6] | — | — | — |
| A − B | `principal` | RF | B3 | janela | calibrado | 4,6 [−2,6; 11,2] | −2,5 [−11,1; 5,4] | — | — | — | — |
| A − B | `principal` | LR | B3 | janela | calibrado | 6,5 [−0,7; 12,4] | −3,0 [−11,8; 4,7] | — | — | — | — |
| A − B | `principal` | RF | B1 | janela | 0,5 | 9,0 [0,3; 17,0] | 6,3 [−3,3; 14,6] | 7,4 [−1,7; 15,9] | — | — | — |
| A − B | `principal` | LR | B1 | janela | 0,5 | 10,6 [1,5; 17,6] | 9,3 [−0,7; 18,2] | 13,8 [6,1; 20,9] | — | — | — |
| A − B | `principal` | RF | B2 | janela | 0,5 | 14,4 [5,6; 22,3] | 9,8 [0,3; 18,3] | 15,7 [6,2; 23,7] | — | — | — |
| A − B | `principal` | LR | B2 | janela | 0,5 | 16,0 [7,4; 23,5] | 6,5 [−3,5; 15,3] | 18,4 [10,6; 25,7] | — | — | — |
| A − B | `sem_normalizacao` | RF | B3 | janela | 0,5 | 8,6 [−2,4; 18,5] | 4,9 [−6,9; 14,9] | 12,2 [0,4; 21,8] | — | — | — |
| A − B | `sem_normalizacao` | LR | B3 | janela | 0,5 | 9,0 [0,5; 16,0] | 1,5 [−8,2; 9,6] | 8,5 [−2,0; 17,6] | — | — | — |
| A − B | `sem_normalizacao` | RF | B3 | janela | calibrado | 12,4 [2,9; 20,6] | 3,3 [−7,6; 12,9] | — | — | — | — |
| A − B | `sem_normalizacao` | LR | B3 | janela | calibrado | 8,4 [−0,3; 15,7] | 1,3 [−9,8; 9,9] | — | — | — | — |
| A − B | `sem_normalizacao` | RF | B1 | janela | 0,5 | 14,1 [3,7; 23,6] | 10,2 [−1,1; 20,3] | 17,4 [5,5; 26,5] | — | — | — |
| A − B | `sem_normalizacao` | LR | B1 | janela | 0,5 | 13,1 [4,6; 19,8] | 10,7 [0,6; 18,3] | 12,7 [2,6; 21,1] | — | — | — |
| A − B | `sem_normalizacao` | RF | B2 | janela | 0,5 | 13,1 [2,3; 22,6] | 9,2 [−2,3; 19,0] | 20,9 [8,7; 30,1] | — | — | — |
| A − B | `sem_normalizacao` | LR | B2 | janela | 0,5 | 15,3 [7,1; 22,3] | 5,1 [−4,2; 13,0] | 16,1 [6,2; 25,6] | — | — | — |

## 8. Sensibilidades (B3; por janela salvo indicação)

Cada linha muda UMA escolha em relação à principal. B é o da variante indicada em `b` (a UTA não muda nas variantes que só alteram o DMD). O limiar calibrado de C vem sempre do LOSO do DMD da própria variante.

| Variante | DMD: sonolento / alerta (sujeitos) | A: AUC | B: AUC | C: AUC | Queda B − C (AUC), pp | C: acur. bal. 0,5 | C: acur. bal. no limiar do DMD (limiar) | C: AUC por vídeo | Queda B − C (AUC por vídeo), pp |
|---|---|---|---|---|---|---|---|---|---|
| **Random Forest** | | | | | | | | | |
| `principal` | 423 / 87 (13) | 82,4 [73,6; 89,7] | 80,1 [76,3; 83,8] | 69,2 [63,7; 74,3] | 10,9 [6,7; 15,7] | 65,2 [61,7; 68,6] | 62,9 [60,0; 66,0] (0,662) | 80,5 [73,0; 87,2] | 11,9 [5,6; 18,8] |
| `sem_normalizacao` | 423 / 87 (13) | 81,6 [70,3; 90,1] | 69,4 [64,8; 73,7] | 67,2 [62,5; 71,6] | 2,1 [−1,2; 5,8] | 63,1 [59,8; 66,3] | 54,7 [53,1; 56,7] (0,842) | 80,6 [73,1; 87,5] | 0,2 [−5,6; 6,3] |
| `alerta_s1` | 423 / 102 (13) | 77,2 [62,8; 90,4] | 80,1 [76,3; 83,8] | 58,6 [53,5; 63,3] | 21,5 [16,6; 26,6] | 58,3 [54,6; 61,8] | 58,1 [54,7; 61,3] (0,592) | 65,6 [57,9; 73,6] | 26,8 [19,2; 34,5] |
| `s5_sem_sinal_alerta` | 423 / 233 (13) | 80,0 [69,6; 87,2] | 80,1 [76,3; 83,8] | 67,5 [62,4; 72,3] | 12,6 [8,5; 16,8] | 60,2 [57,3; 63,3] | 59,7 [57,0; 62,9] (0,512) | 78,4 [71,4; 85,4] | 14,0 [7,5; 20,3] |
| `janela_5s` | 308 / 237 (13) | 80,1 [69,6; 87,6] | 77,5 [74,0; 81,3] | 65,7 [61,5; 70,0] | 11,8 [8,4; 15,7] | 59,8 [57,1; 62,6] | 58,5 [56,0; 61,2] (0,577) | 75,4 [68,6; 82,5] | 15,3 [9,2; 21,8] |
| `janela_15s` | 466 / 8 (13) | 93,5 [82,2; 99,2] | 82,1 [78,4; 86,0] | 69,2 [63,0; 74,8] | 12,9 [7,9; 18,4] | 57,5 [53,4; 61,3] | 67,8 [63,6; 72,0] (0,958) | 75,6 [67,3; 83,2] | 18,1 [10,9; 26,4] |
| `ref_s2_espelho` | 423 / 49 (13) | 86,2 [78,4; 93,5] | 80,1 [76,3; 83,8] | 71,1 [66,0; 75,6] | 9,0 [5,5; 13,2] | 64,0 [60,4; 67,5] | 68,1 [64,8; 71,4] (0,765) | 82,0 [74,4; 89,1] | 10,4 [4,4; 17,6] |
| `sem_gE_29` | 393 / 81 (12) | 87,2 [81,5; 92,2] | 80,1 [76,3; 83,8] | 70,0 [64,2; 75,4] | 10,2 [5,4; 15,4] | 67,7 [64,4; 71,1] | 66,5 [63,4; 69,8] (0,568) | 83,5 [76,2; 90,3] | 8,9 [2,9; 16,2] |
| **Regressão logística** | | | | | | | | | |
| `principal` | 423 / 87 (13) | 83,5 [76,1; 88,9] | 74,4 [69,7; 78,4] | 64,5 [58,1; 70,9] | 9,8 [3,7; 16,0] | 64,0 [60,7; 67,4] | 63,3 [60,2; 66,7] (0,535) | 76,0 [67,8; 84,7] | 13,0 [5,8; 20,8] |
| `sem_normalizacao` | 423 / 87 (13) | 81,1 [71,6; 88,8] | 72,5 [67,7; 77,3] | 65,2 [60,6; 69,0] | 7,3 [3,0; 11,7] | 58,1 [55,5; 61,2] | 59,4 [56,4; 62,6] (0,400) | 81,6 [75,1; 87,6] | 1,2 [−5,4; 7,9] |
| `alerta_s1` | 423 / 102 (13) | 82,5 [69,7; 91,7] | 74,4 [69,7; 78,4] | 58,5 [53,4; 64,0] | 15,9 [9,6; 22,1] | 63,0 [59,8; 66,5] | 55,9 [52,0; 59,8] (0,382) | 68,2 [60,1; 75,9] | 20,8 [12,6; 30,0] |
| `s5_sem_sinal_alerta` | 423 / 233 (13) | 82,0 [75,5; 86,9] | 74,4 [69,7; 78,4] | 65,8 [59,4; 72,0] | 8,5 [3,1; 14,5] | 60,6 [57,9; 63,7] | 62,3 [59,4; 65,6] (0,435) | 79,8 [71,4; 88,8] | 9,2 [2,5; 16,7] |
| `janela_5s` | 308 / 237 (13) | 82,5 [75,9; 87,9] | 70,8 [66,2; 74,6] | 65,4 [59,1; 71,3] | 5,4 [−0,1; 10,8] | 62,8 [59,9; 65,9] | 62,4 [59,6; 65,4] (0,512) | 77,1 [68,3; 85,5] | 11,6 [4,7; 19,2] |
| `janela_15s` | 466 / 8 (13) | 94,9 [88,3; 100,0] | 76,2 [71,5; 80,4] | 66,5 [58,9; 73,6] | 9,7 [3,1; 16,5] | 56,1 [51,1; 60,6] | 64,1 [59,2; 68,7] (0,831) | 69,3 [59,5; 78,5] | 19,9 [11,4; 29,3] |
| `ref_s2_espelho` | 423 / 49 (13) | 85,3 [78,4; 90,9] | 74,4 [69,7; 78,4] | 69,0 [62,3; 75,1] | 5,4 [0,3; 10,9] | 67,5 [63,5; 71,4] | 64,7 [60,6; 68,8] (0,380) | 77,0 [68,5; 85,5] | 12,0 [5,6; 19,0] |
| `sem_gE_29` | 393 / 81 (12) | 85,2 [78,9; 89,6] | 74,4 [69,7; 78,4] | 64,4 [57,7; 70,8] | 10,0 [3,7; 16,3] | 63,6 [60,4; 67,1] | 61,6 [58,7; 65,0] (0,581) | 76,0 [67,9; 84,6] | 13,0 [5,7; 20,6] |

Descrição das variantes: `sem_normalizacao` = como a principal, com as características brutas (sem normalização por sujeito); `alerta_s1` = alerta do DMD = safe_drive da s1 (carro em MOVIMENTO) no lugar da s2; `s5_sem_sinal_alerta` = janelas da s5 sem sinal anotado entram como alerta, junto com as da s2; `janela_5s` = janelas de 5 s (passo 5 s) nas duas bases; `janela_15s` = janelas de 15 s (passo 5 s) nas duas bases (no DMD quase não sobra alerta); `ref_s2_espelho` = referência do DMD = último terço das janelas safe_drive da s2 e alerta = as anteriores (a mesma estrutura da UTA: referência na sessão alerta, sonolento em outra sessão); `sem_gE_29` = sem o motorista gE_29 (usa óculos; o detector de piscadas falhou nele) no DMD.

Na variante `ref_s2_espelho` o alerta do DMD cai para 49 janelas e a referência fica com 1–6 janelas por sujeito (mais ruidosa); o limiar de EAR continua o calibrado na s3 (só a centragem muda).

## 9. Controle: B e C separam sonolência ou só "vídeo diferente da referência"?

As janelas alerta (KSS 0) vêm do mesmo vídeo que a referência da normalização (os 2/3 iniciais do vídeo 0; a referência é o último terço); as sonolentas (KSS 10), de outro vídeo, gravado em outro momento. Um modelo pode separar B3 só por "este trecho não se parece com a referência" (luz, enquadramento, hora do dia). O KSS 5 também é outro vídeo: se o modelo separa KSS 10 de KSS 5, há sinal além do vídeo de referência. Os modelos são os mesmos de B3 (cada sujeito com o modelo do fold em que foi teste) e de C; as janelas KSS 5 nunca entraram no treino.

| Variante | Experimento | Modelo | AUC KSS 0 × 10 (B3) | AUC KSS 5 × 10 | AUC média intra-sujeito 5 × 10 | AUC KSS 0 × 5 | Escore médio KSS 0 / 5 / 10 |
|---|---|---|---|---|---|---|---|
| `principal` | UTA → UTA (B) | RF | 80,1 [76,3; 83,8] | 61,5 [57,1; 65,8] | 62,9 [58,3; 67,5] | 71,3 [66,4; 75,5] | 0,33 / 0,55 / 0,67 |
| `principal` | UTA → UTA (B) | LR | 74,4 [69,7; 78,4] | 60,8 [56,7; 64,8] | 61,1 [55,8; 65,9] | 64,9 [59,8; 69,7] | 0,39 / 0,50 / 0,59 |
| `principal` | DMD → UTA (C) | RF | 69,2 [63,7; 74,3] | 60,2 [55,6; 64,8] | 61,6 [56,5; 66,9] | 59,8 [54,7; 65,0] | 0,29 / 0,38 / 0,50 |
| `principal` | DMD → UTA (C) | LR | 64,5 [58,1; 70,9] | 56,2 [50,9; 61,3] | 60,5 [54,0; 66,5] | 60,2 [54,6; 65,6] | 0,24 / 0,33 / 0,41 |
| `sem_normalizacao` | UTA → UTA (B) | RF | 69,4 [64,8; 73,7] | 60,0 [55,9; 64,2] | 62,1 [57,5; 66,4] | 60,4 [56,5; 64,6] | 0,47 / 0,56 / 0,65 |
| `sem_normalizacao` | UTA → UTA (B) | LR | 72,5 [67,7; 77,3] | 60,5 [55,7; 65,4] | 62,9 [57,7; 67,9] | 64,4 [59,4; 68,9] | 0,41 / 0,50 / 0,59 |
| `sem_normalizacao` | DMD → UTA (C) | RF | 67,2 [62,5; 71,6] | 58,2 [54,1; 62,3] | 59,7 [55,0; 63,9] | 59,5 [54,5; 64,1] | 0,35 / 0,43 / 0,50 |
| `sem_normalizacao` | DMD → UTA (C) | LR | 65,2 [60,6; 69,0] | 58,9 [55,0; 62,9] | 60,2 [54,9; 65,2] | 57,0 [52,8; 60,5] | 0,17 / 0,21 / 0,31 |

Queda B − C nesses contrastes (pareada, pp de AUC × 100):

| Variante | Modelo | Δ AUC 5 × 10 | Δ AUC intra-sujeito 5 × 10 | Δ AUC 0 × 5 |
|---|---|---|---|---|
| `principal` | RF | 1,2 [−3,0; 5,8] | 1,3 [−4,0; 7,1] | 11,5 [6,7; 16,2] |
| `principal` | LR | 4,6 [−0,2; 9,7] | 0,5 [−5,6; 6,5] | 4,7 [−0,2; 10,3] |
| `sem_normalizacao` | RF | 1,8 [−1,7; 5,1] | 2,5 [−1,3; 6,2] | 0,9 [−3,9; 6,0] |
| `sem_normalizacao` | LR | 1,6 [−3,0; 5,9] | 2,8 [−3,3; 8,5] | 7,4 [2,4; 12,0] |

## 10. Quais sinais transferem?

Importância por permutação = queda da AUC agrupada (× 100) ao embaralhar a característica entre as janelas do mesmo sujeito de teste (média ± dp de 20 repetições; variante principal; B3 na UTA). AUC univariada = AUC da característica sozinha (`_c`) como escore de sonolento: > 50 = maior nas janelas sonolentas. Uma característica "discrimina" numa base quando o IC 95% da AUC univariada exclui 50 e a AUC se afasta ≥ 5 pontos de 50 (regra descritiva, não é teste de hipótese). Tabelas ordenadas pela importância em C.

**Random Forest**

| Característica | Import. A (DMD → DMD) | Import. B (UTA → UTA) | Import. C (DMD → UTA) | AUC univ. DMD | AUC univ. UTA B3 | Sentido nas duas bases |
|---|---:|---:|---:|---|---|---|
| `dur_max_fechamento_s` | 10,4 ± 1,2 | 1,7 ± 0,1 | 7,7 ± 0,3 | 85,4 [77,5; 92,5] | 69,4 [63,6; 75,0] | mesmo sentido (transfere) |
| `perclos` | 1,8 ± 0,4 | 1,3 ± 0,1 | 3,1 ± 0,2 | 81,8 [73,8; 90,3] | 70,1 [64,8; 75,3] | mesmo sentido (transfere) |
| `ear_media` | 0,8 ± 0,3 | 11,6 ± 0,2 | 2,8 ± 0,2 | 23,5 [13,1; 33,8] | 36,2 [28,4; 44,1] | mesmo sentido (transfere) |
| `dur_media_piscada_s` | 0,6 ± 0,4 | 0,6 ± 0,1 | 2,8 ± 0,2 | 76,9 [68,6; 84,9] | 70,4 [65,0; 75,9] | mesmo sentido (transfere) |
| `ear_dp` | 1,6 ± 0,6 | 5,9 ± 0,2 | 2,7 ± 0,2 | 72,5 [62,2; 81,4] | 64,7 [59,3; 69,9] | mesmo sentido (transfere) |
| `n_fechamentos_05s` | 1,2 ± 0,3 | 0,2 ± 0,0 | 1,9 ± 0,1 | 77,9 [70,0; 85,3] | 61,7 [58,0; 65,6] | mesmo sentido (transfere) |
| `amp_media` | 1,9 ± 0,4 | 2,2 ± 0,1 | 0,4 ± 0,1 | 48,4 [33,6; 65,5] | 34,3 [27,1; 42,2] | discrimina só na UTA |
| `taxa_rosto` | −0,0 ± 0,1 | 0,2 ± 0,0 | 0,2 ± 0,0 | 56,7 [47,7; 62,7] | 48,7 [47,0; 49,8] | não discrimina em nenhuma |
| `taxa_piscadas_min` | −0,2 ± 0,4 | 0,7 ± 0,1 | 0,1 ± 0,1 | 64,5 [56,9; 72,8] | 63,2 [57,4; 69,0] | mesmo sentido (transfere) |
| `frac_bocejo` | 1,8 ± 0,5 | 0,0 ± 0,0 | 0,1 ± 0,0 | 57,0 [54,7; 59,3] | 50,2 [50,0; 50,5] | discrimina só no DMD |
| `vel_media` | −1,0 ± 0,2 | 0,9 ± 0,1 | −0,7 ± 0,1 | 40,5 [28,1; 56,9] | 32,4 [26,9; 38,0] | discrimina só na UTA |

**Regressão logística**

| Característica | Import. A (DMD → DMD) | Import. B (UTA → UTA) | Import. C (DMD → UTA) | AUC univ. DMD | AUC univ. UTA B3 | Sentido nas duas bases |
|---|---:|---:|---:|---|---|---|
| `dur_max_fechamento_s` | 13,1 ± 2,0 | 4,9 ± 0,2 | 10,9 ± 0,3 | 85,4 [77,5; 92,5] | 69,4 [63,6; 75,0] | mesmo sentido (transfere) |
| `ear_media` | 7,1 ± 0,7 | −0,7 ± 0,1 | 8,8 ± 0,3 | 23,5 [13,1; 33,8] | 36,2 [28,4; 44,1] | mesmo sentido (transfere) |
| `ear_dp` | 2,9 ± 0,8 | 5,3 ± 0,2 | 4,6 ± 0,2 | 72,5 [62,2; 81,4] | 64,7 [59,3; 69,9] | mesmo sentido (transfere) |
| `n_fechamentos_05s` | 2,4 ± 0,7 | 0,1 ± 0,2 | 2,3 ± 0,1 | 77,9 [70,0; 85,3] | 61,7 [58,0; 65,6] | mesmo sentido (transfere) |
| `taxa_piscadas_min` | 0,9 ± 0,2 | −0,1 ± 0,1 | 2,2 ± 0,1 | 64,5 [56,9; 72,8] | 63,2 [57,4; 69,0] | mesmo sentido (transfere) |
| `vel_media` | 0,5 ± 0,4 | 2,4 ± 0,1 | 1,4 ± 0,1 | 40,5 [28,1; 56,9] | 32,4 [26,9; 38,0] | discrimina só na UTA |
| `dur_media_piscada_s` | 0,4 ± 0,2 | 1,2 ± 0,1 | 0,3 ± 0,0 | 76,9 [68,6; 84,9] | 70,4 [65,0; 75,9] | mesmo sentido (transfere) |
| `frac_bocejo` | 8,1 ± 1,9 | 0,1 ± 0,0 | 0,1 ± 0,0 | 57,0 [54,7; 59,3] | 50,2 [50,0; 50,5] | discrimina só no DMD |
| `taxa_rosto` | −0,1 ± 0,1 | 0,2 ± 0,0 | −0,6 ± 0,1 | 56,7 [47,7; 62,7] | 48,7 [47,0; 49,8] | não discrimina em nenhuma |
| `perclos` | −1,4 ± 0,8 | 13,7 ± 0,3 | −2,3 ± 0,2 | 81,8 [73,8; 90,3] | 70,1 [64,8; 75,3] | mesmo sentido (transfere) |
| `amp_media` | 2,0 ± 0,7 | 1,9 ± 0,1 | −3,0 ± 0,2 | 48,4 [33,6; 65,5] | 34,3 [27,1; 42,2] | discrimina só na UTA |

As 3 características mais importantes (queda de AUC × 100) em cada modelo e variante:

| Experimento | Variante | RF | LR |
|---|---|---|---|
| A (DMD → DMD) | `principal` | `dur_max_fechamento_s` 10,4, `amp_media` 1,9, `frac_bocejo` 1,8 | `dur_max_fechamento_s` 13,1, `frac_bocejo` 8,1, `ear_media` 7,1 |
| A (DMD → DMD) | `sem_normalizacao` | `dur_max_fechamento_s` 7,9, `perclos` 3,5, `frac_bocejo` 2,0 | `dur_max_fechamento_s` 15,0, `frac_bocejo` 8,7, `n_fechamentos_05s` 2,1 |
| B (UTA → UTA, B3) | `principal` | `ear_media` 11,6, `ear_dp` 5,9, `amp_media` 2,2 | `perclos` 13,7, `ear_dp` 5,3, `dur_max_fechamento_s` 4,9 |
| B (UTA → UTA, B3) | `sem_normalizacao` | `perclos` 3,6, `ear_dp` 3,5, `dur_max_fechamento_s` 3,4 | `perclos` 7,6, `ear_dp` 5,4, `dur_max_fechamento_s` 4,9 |
| C (DMD → UTA, B3) | `principal` | `dur_max_fechamento_s` 7,7, `perclos` 3,1, `ear_media` 2,8 | `dur_max_fechamento_s` 10,9, `ear_media` 8,8, `ear_dp` 4,6 |
| C (DMD → UTA, B3) | `sem_normalizacao` | `dur_max_fechamento_s` 6,2, `perclos` 3,5, `n_fechamentos_05s` 1,8 | `dur_max_fechamento_s` 13,1, `ear_dp` 4,6, `ear_media` 2,7 |

Coeficientes da regressão logística (escala padronizada; positivo = empurra para "sonolento"; A e B: média ± dp entre os modelos de partição; C: modelo único treinado em todo o DMD; variante principal):

| Característica | A (DMD) | B3 (UTA) | C (todo o DMD) |
|---|---:|---:|---:|
| `perclos` | −1,12 ± 0,18 | 2,24 ± 0,58 | −1,20 |
| `taxa_piscadas_min` | 0,23 ± 0,09 | −0,06 ± 0,12 | 0,24 |
| `dur_media_piscada_s` | 0,23 ± 0,13 | 0,22 ± 0,08 | 0,22 |
| `dur_max_fechamento_s` | 1,42 ± 0,17 | 1,32 ± 0,21 | 1,46 |
| `n_fechamentos_05s` | 0,42 ± 0,15 | −0,81 ± 0,04 | 0,43 |
| `ear_media` | −1,02 ± 0,14 | 0,27 ± 0,07 | −1,03 |
| `ear_dp` | 0,67 ± 0,12 | 0,51 ± 0,13 | 0,67 |
| `amp_media` | 0,81 ± 0,14 | −0,35 ± 0,13 | 0,82 |
| `vel_media` | −0,45 ± 0,17 | −0,30 ± 0,06 | −0,46 |
| `frac_bocejo` | 1,69 ± 0,08 | 0,22 ± 0,19 | 1,72 |
| `taxa_rosto` | 0,14 ± 0,10 | −0,50 ± 0,43 | 0,14 |

Nível das características brutas (mediana nas janelas alerta → sonolentas; variante sem normalização):

| Característica | DMD alerta → sonolento | UTA KSS 0 → KSS 10 |
|---|---|---|
| `perclos` | 0,037 → 0,208 | 0,010 → 0,040 |
| `taxa_piscadas_min` | 17,98 → 29,96 | 5,99 → 12,00 |
| `dur_media_piscada_s` | 0,118 → 0,235 | 0,122 → 0,184 |
| `dur_max_fechamento_s` | 0,13 → 0,97 | 0,07 → 0,17 |
| `n_fechamentos_05s` | 0,00 → 1,00 | 0,00 → 0,00 |
| `ear_media` | 0,259 → 0,237 | 0,264 → 0,254 |
| `ear_dp` | 0,030 → 0,043 | 0,015 → 0,020 |
| `amp_media` | 0,104 → 0,109 | 0,095 → 0,078 |
| `vel_media` | 0,34 → 0,33 | 0,29 → 0,22 |
| `frac_bocejo` | 0,000 → 0,000 | 0,000 → 0,000 |
| `taxa_rosto` | 1,000 → 1,000 | 1,000 → 1,000 |

## 11. PP3 no DMD: óculos (metadado `glasses` do OpenLABEL)

Na UTA-RLDD não há rótulo de óculos por vídeo e, por decisão da equipe, os rostos da UTA não foram anotados: a PP3 sobre óculos só pode ser olhada no DMD, onde 3 dos 13 motoristas usam óculos na sessão de sonolência (s5). O desempenho de A por motorista e a validação do detector de piscadas contra a anotação da s5 (`results/exp5_dmd_validacao_detector.csv`, limiar final) ficam lado a lado:

| Motorista | Óculos s5 / s2 | s2 no mesmo dia da s5 | Janelas sonol. / alerta | A: AUC RF | A: AUC LR | A-RF: sens. / espec. (0,5) | Detector: F1 piscadas | Revocação piscadas | Acur. bal. por quadro | EAR mediano aberto / fechado | PERCLOS médio alerta → sonol. |
|---|---|---|---|---:|---:|---|---:|---:|---:|---|---|
| gA_1 | sim / sim | não | 26 / 8 | 89,9 | 90,4 | 92,3 / 25,0 | 0,897 | 0,892 | 0,960 | 0,226 / 0,122 | 0,168 → 0,328 |
| gE_29 | sim / sim | sim | 30 / 6 | 65,3 | 80,6 | 16,7 / 100,0 | 0,108 | 0,057 | 0,500 | 0,288 / 0,247 | 0,000 → 0,004 |
| gZ_36 | sim / sim | sim | 25 / 6 | 94,0 | 98,7 | 92,0 / 66,7 | 0,797 | 0,906 | 0,969 | 0,245 / 0,161 | 0,038 → 0,274 |
| gA_5 | não / não | não | 25 / 8 | 79,5 | 75,0 | 100,0 / 25,0 | 0,898 | 1,000 | 0,903 | 0,237 / 0,153 | 0,195 → 0,245 |
| gB_10 | não / não | não | 59 / 5 | 96,6 | 98,0 | 91,5 / 100,0 | 0,809 | 0,961 | 0,988 | 0,265 / 0,123 | 0,046 → 0,293 |
| gB_6 | não / não | não | 20 / 5 | 77,5 | 58,0 | 80,0 / 80,0 | 0,488 | 0,328 | 0,815 | 0,303 / 0,211 | 0,001 → 0,076 |
| gB_7 | não / não | não | 28 / 7 | 84,4 | 86,7 | 100,0 / 28,6 | 0,759 | 0,719 | 0,780 | 0,237 / 0,138 | 0,187 → 0,462 |
| gB_9 | não / não | sim | 25 / 8 | 99,0 | 98,5 | 100,0 / 75,0 | 0,729 | 0,737 | 0,946 | 0,252 / 0,153 | 0,034 → 0,215 |
| gC_13 | não / não | não | 22 / 7 | 95,5 | 91,6 | 86,4 / 100,0 | 0,942 | 0,989 | 0,995 | 0,288 / 0,147 | 0,053 → 0,145 |
| gC_14 | não / não | não | 21 / 2 | 69,0 | 69,0 | 66,7 / 50,0 | 0,762 | 1,000 | 0,981 | 0,266 / 0,163 | 0,022 → 0,297 |
| gF_23 | não / não;não | não | 57 / 15 | 89,6 | 92,9 | 86,0 / 80,0 | 0,862 | 0,843 | 0,964 | 0,267 / 0,160 | 0,054 → 0,196 |
| gZ_33 | não / sim;não | não | 61 / 5 | 83,4 | 88,9 | 98,4 / 80,0 | 0,813 | 0,804 | 0,949 | 0,275 / 0,168 | 0,040 → 0,234 |
| gZ_37 | não / não | não | 24 / 5 | 87,5 | 99,2 | 54,2 / 100,0 | 0,644 | 0,564 | 0,741 | 0,245 / 0,167 | 0,010 → 0,122 |

O EAR sem limiar: AUC de −EAR para separar os quadros anotados "fechado" dos "aberto" (transições fora), medianas do EAR nas duas classes e fração dos quadros fechados abaixo do limiar do sujeito (0,75 × EAR mediano da referência; `results/exp5b_oculos_ear_dmd.csv`). Uma AUC alta com queda relativa pequena indica que o sinal existe, mas o limiar relativo não o alcança:

| Sessão s5 | Óculos | AUC −EAR (média dos olhos) | AUC −EAR esq. / dir. | EAR mediano aberto → fechado | Queda relativa | Limiar | Quadros fechados abaixo do limiar | F1 piscadas |
|---|---|---:|---|---|---:|---:|---:|---:|
| gA_1_2019-03-14T14:26:17 | sim | 98,0 | 98,1 / 97,9 | 0,226 → 0,122 | 46,2% | 0,192 | 98,2% | 0,897 |
| gE_29_2019-03-15T13:51:09 | sim | 92,8 | 90,6 / 94,7 | 0,288 → 0,247 | 14,2% | 0,206 | 0,0% | 0,108 |
| gZ_36_2019-04-09T10:49:32 | sim | 99,5 | 99,3 / 99,6 | 0,245 → 0,161 | 34,3% | 0,188 | 96,2% | 0,797 |
| gA_5_2019-03-13T09:06:49 | não | 98,3 | 97,9 / 98,2 | 0,237 → 0,153 | 35,5% | 0,174 | 81,1% | 0,898 |
| gB_10_2019-03-12T10:35:20 | não | 100,0 | 99,9 / 100,0 | 0,266 → 0,118 | 55,6% | 0,203 | 99,6% | 0,798 |
| gB_10_2019-03-13T14:17:28 | não | 99,8 | 99,8 / 99,8 | 0,265 → 0,129 | 51,5% | 0,203 | 99,8% | 0,820 |
| gB_6_2019-03-13T13:37:11 | não | 96,9 | 93,6 / 98,7 | 0,303 → 0,211 | 30,4% | 0,217 | 63,1% | 0,488 |
| gB_7_2019-03-13T13:55:52 | não | 86,1 | 88,3 / 84,2 | 0,237 → 0,138 | 41,9% | 0,189 | 74,4% | 0,759 |
| gB_9_2019-03-07T16:31:48 | não | 99,8 | 99,8 / 99,7 | 0,252 → 0,153 | 39,3% | 0,176 | 89,6% | 0,729 |
| gC_13_2019-03-12T10:03:00 | não | 99,9 | 99,9 / 99,9 | 0,288 → 0,147 | 48,9% | 0,223 | 100,0% | 0,942 |
| gC_14_2019-03-12T09:18:58 | não | 100,0 | 100,0 / 100,0 | 0,266 → 0,163 | 38,8% | 0,227 | 100,0% | 0,762 |
| gF_23_2019-03-11T10:19:19 | não | 99,3 | 99,0 / 99,4 | 0,268 → 0,161 | 39,9% | 0,203 | 91,8% | 0,875 |
| gF_23_2019-03-14T13:49:53 | não | 99,6 | 99,4 / 99,5 | 0,265 → 0,160 | 39,7% | 0,203 | 94,8% | 0,848 |
| gZ_33_2019-04-04T09:29:07 | não | 99,6 | 99,9 / 99,0 | 0,278 → 0,170 | 38,9% | 0,209 | 97,3% | 0,816 |
| gZ_33_2019-04-04T15:02:29 | não | 98,8 | 99,1 / 98,2 | 0,273 → 0,167 | 38,9% | 0,209 | 93,0% | 0,809 |
| gZ_37_2019-04-29T12:06:35 | não | 96,5 | 95,9 / 96,5 | 0,245 → 0,167 | 31,8% | 0,166 | 48,3% | 0,644 |

A agregada por estrato de sujeitos (IC por bootstrap dos sujeitos do estrato; com 3 sujeitos o IC é quase degenerado — ver a coluna de reamostragens válidas):

| Variante | Modelo | Estrato | Sujeitos | Janelas sonol. / alerta | Acur. bal. (0,5) | AUC | AUC média intra-sujeito | Reamostragens válidas (AUC) |
|---|---|---|---:|---|---|---|---|---:|
| `principal` | RF | com óculos (s5) | 3 | 81 / 20 | 62,1 [54,8; 79,3] | 68,4 [59,0; 94,0] | 83,1 [65,3; 94,0] | 1000 |
| `principal` | RF | sem óculos (s5) | 10 | 342 / 67 | 80,1 [71,4; 87,1] | 86,1 [79,0; 91,7] | 86,2 [80,6; 91,8] | 1000 |
| `principal` | RF | s2 no mesmo dia da s5 | 3 | 80 / 20 | 73,1 [58,3; 87,5] | 73,8 [63,3; 99,0] | 86,1 [65,3; 99,0] | 1000 |
| `principal` | RF | s2 em outro dia | 10 | 343 / 67 | 76,9 [67,0; 85,7] | 84,1 [77,1; 89,5] | 85,3 [80,2; 89,8] | 1000 |
| `principal` | LR | com óculos (s5) | 3 | 81 / 20 | 78,4 [58,3; 94,0] | 78,3 [72,0; 98,7] | 89,9 [80,6; 98,7] | 1000 |
| `principal` | LR | sem óculos (s5) | 10 | 342 / 67 | 78,3 [68,1; 85,2] | 84,6 [77,4; 89,8] | 85,8 [77,4; 93,8] | 1000 |
| `principal` | LR | s2 no mesmo dia da s5 | 3 | 80 / 20 | 75,0 [58,3; 94,0] | 79,1 [71,9; 98,7] | 92,6 [80,6; 98,7] | 1000 |
| `principal` | LR | s2 em outro dia | 10 | 343 / 67 | 79,4 [68,9; 86,0] | 84,4 [76,5; 89,5] | 85,0 [76,5; 92,2] | 1000 |
| `sem_normalizacao` | RF | com óculos (s5) | 3 | 81 / 20 | 59,0 [44,1; 94,0] | 65,3 [49,9; 96,0] | 79,0 [50,3; 96,0] | 1000 |
| `sem_normalizacao` | RF | sem óculos (s5) | 10 | 342 / 67 | 74,5 [63,5; 83,2] | 86,0 [76,6; 92,1] | 83,9 [74,4; 92,0] | 1000 |
| `sem_normalizacao` | RF | s2 no mesmo dia da s5 | 3 | 80 / 20 | 71,9 [50,0; 94,0] | 69,8 [50,3; 96,0] | 79,8 [50,3; 96,0] | 1000 |
| `sem_normalizacao` | RF | s2 em outro dia | 10 | 343 / 67 | 70,4 [58,3; 81,5] | 84,5 [75,6; 90,8] | 83,7 [73,6; 91,5] | 1000 |
| `sem_normalizacao` | LR | com óculos (s5) | 3 | 81 / 20 | 70,3 [58,3; 80,0] | 72,2 [71,5; 90,7] | 87,5 [84,4; 90,7] | 1000 |
| `sem_normalizacao` | LR | sem óculos (s5) | 10 | 342 / 67 | 78,4 [69,7; 84,6] | 84,3 [74,4; 90,6] | 81,6 [71,9; 89,6] | 1000 |
| `sem_normalizacao` | LR | s2 no mesmo dia da s5 | 3 | 80 / 20 | 73,8 [58,3; 86,0] | 74,7 [69,9; 98,5] | 91,2 [84,4; 98,5] | 1000 |
| `sem_normalizacao` | LR | s2 em outro dia | 10 | 343 / 67 | 77,3 [68,8; 83,5] | 83,4 [73,2; 89,3] | 80,5 [71,2; 87,9] | 1000 |

Sensibilidade sem o motorista em que o detector falhou (`sem_gE_29`; seção 8):

| Modelo | A: AUC (principal → sem gE_29) | C: AUC (principal → sem gE_29) | C: acur. bal. no limiar do DMD (principal → sem gE_29) |
|---|---|---|---|
| RF | 82,4 [73,6; 89,7] → 87,2 [81,5; 92,2] | 69,2 [63,7; 74,3] → 70,0 [64,2; 75,4] | 62,9 [60,0; 66,0] → 66,5 [63,4; 69,8] |
| LR | 83,5 [76,1; 88,9] → 85,2 [78,9; 89,6] | 64,5 [58,1; 70,9] → 64,4 [57,7; 70,8] | 63,3 [60,2; 66,7] → 61,6 [58,7; 65,0] |

## 12. Por sujeito da UTA (B3, variante principal)

- RF: AUC intra-sujeito > 50 em 57 de 60 sujeitos em B e em 50 em C; mediana 85,4 (B) e 73,0 (C); C melhor que B em 16 sujeitos; correlação de Spearman entre as AUC de B e C por sujeito 0,71.
- LR: AUC intra-sujeito > 50 em 55 de 60 sujeitos em B e em 47 em C; mediana 77,8 (B) e 77,3 (C); C melhor que B em 24 sujeitos; correlação de Spearman entre as AUC de B e C por sujeito 0,42.
- C-RF: fração de janelas previstas "sonolento" por KSS 0 / 5 / 10 = 18,0% / 31,9% / 48,4% no limiar 0,5 e 11,3% / 21,2% / 37,0% no limiar do DMD.
- C-LR: fração de janelas previstas "sonolento" por KSS 0 / 5 / 10 = 5,6% / 20,5% / 33,7% no limiar 0,5 e 4,2% / 17,8% / 30,8% no limiar do DMD.
- C-RF: AUC intra-sujeito < 50 (ordem invertida) em 10 sujeitos: F2_15 (10,1; B 66,5), F4_41 (28,2; B 81,1), F3_27 (29,1; B 67,7), F1_09 (30,8; B 52,8), F5_54 (31,3; B 53,1), F1_06 (36,1; B 46,8), F4_45 (37,2; B 80,4), F5_52 (37,6; B 82,4), F3_28 (37,8; B 69,2), F4_38 (44,5; B 94,4).
- C-LR: AUC intra-sujeito < 50 (ordem invertida) em 13 sujeitos: F2_15 (1,0; B 4,9), F5_54 (8,2; B 45,2), F3_27 (18,5; B 67,7), F4_41 (18,9; B 76,8), F1_09 (21,9; B 42,5), F4_45 (25,2; B 77,1), F3_28 (30,3; B 78,5), F5_49 (30,6; B 82,6), F5_52 (32,9; B 78,1), F1_04 (35,9; B 66,3), F2_21 (39,0; B 88,0), F5_60 (48,3; B 72,9), F1_08 (48,7; B 67,7).
- F2_15 tem vídeo sem nenhuma piscada detectada (F2_15_10): AUC intra-sujeito B-RF 66,5, C-RF 10,1, B-LR 4,9, C-LR 1,0 (o vídeo entra no B3: o limiar de EAR calibrado no vídeo 0 não é cruzado nele e todas as janelas parecem alerta; seção 3).
- F3_27 tem vídeo sem nenhuma piscada detectada (F3_27_5): AUC intra-sujeito B-RF 67,7, C-RF 29,1, B-LR 67,7, C-LR 18,5 (o vídeo é KSS 5 e não entra no B3; a AUC acima não depende dele; seção 3).

Tabela completa no Apêndice A e em `results/exp5b_por_sujeito.csv`.

## 13. Relação com o Exp. 1 (mesmos 5 folds oficiais, B3)

O Exp. 1 (não recalculado; `results/exp1_intra.csv`) treinou nas **características oficiais** (sequências de 30 piscadas do detector dlib dos autores) e testou com a referência DENTRO do teste (como nos `.npy`). O B deste experimento usa as MESMAS partições (teste = fold k), mas as nossas janelas de 10 s do MediaPipe, sem a referência no teste. Métricas por fold, depois a média dos 5 folds (a agregação do Exp. 1). `oficiais` = sem as pastas que não estão no arquivo oficial (F1_12, F5_49), para ter os mesmos sujeitos; `com ref.` = a sensibilidade `uta_ref_no_teste` (referência e fronteira no treino e no teste).

| Modelo | Fonte | AUC por fold (1 / 2 / 3 / 4 / 5) | AUC média | Acur. bal. média | VA por voto média | Acurácia por vídeo (média dos escores) média |
|---|---|---|---:|---:|---:|---:|
| RF | Exp. 1 (oficiais, com ref.) | 91,2 / 96,7 / 91,1 / 88,4 / 96,4 | 92,8 | 84,8 | 85,5 | — |
| RF | Exp. 5b B, oficiais | 76,2 / 79,2 / 80,0 / 88,4 / 78,7 | 80,5 | 74,1 | 85,2 | 85,2 |
| RF | Exp. 5b B, todos os 60 | 76,7 / 79,2 / 80,0 / 88,4 / 79,2 | 80,7 | 74,3 | 85,8 | 85,8 |
| RF | Exp. 5b B com ref., oficiais | 78,0 / 82,2 / 85,0 / 89,4 / 81,5 | 83,2 | 76,6 | 88,6 | 88,6 |
| LR | Exp. 1 (oficiais, com ref.) | 79,0 / 82,0 / 73,8 / 95,8 / 78,6 | 81,8 | 72,2 | 77,5 | — |
| LR | Exp. 5b B, oficiais | 67,2 / 68,9 / 77,1 / 83,0 / 75,6 | 74,3 | 71,6 | 83,3 | 85,2 |
| LR | Exp. 5b B, todos os 60 | 68,1 / 68,9 / 77,1 / 83,0 / 76,4 | 74,7 | 71,9 | 84,2 | 85,8 |
| LR | Exp. 5b B com ref., oficiais | 66,9 / 69,0 / 78,0 / 82,6 / 75,6 | 74,4 | 71,9 | 81,7 | 85,9 |

## 14. Leitura

**Síntese.** Treinar em sonolência ATUADA (DMD) e testar em sonolência REAL (UTA-RLDD) custa, em relação a treinar na própria UTA com as mesmas características, 10,9 [6,7; 15,7] pp de AUC por janela no RF e 9,8 [3,7; 16,0] pp na LR (IC acima de 0 nos dois); por vídeo, 11,9 [5,6; 18,8] e 13,0 [5,8; 20,8] pp de AUC e 12,5 [5,0; 19,2] e 20,0 [12,5; 26,7] pp de acurácia no limiar 0,5. A transferência fica acima do acaso, mas abaixo do teto real. **Ressalva importante:** entre dois vídeos que não são a referência da normalização (KSS 5 × 10), a queda B − C não é detectável (seção 9; RF: 1,2 [−3,0; 5,8] pp em 5 × 10 contra 11,5 [6,7; 16,2] pp em 0 × 5 — a vantagem de B se concentra no contraste com o vídeo alerta, que é também o vídeo da referência; LR: 4,6 [−0,2; 9,7] pp em 5 × 10 e 4,7 [−0,2; 10,3] pp em 0 × 5, nenhum com IC acima de 0). Parte da queda da PP1 no B3 pode, portanto, vir de B explorar a estrutura da UTA-RLDD (as janelas alerta saem do mesmo vídeo da referência) — um atalho indisponível para o modelo treinado no DMD —, e não só de o treino atuado ensinar menos sobre sonolência; a base não permite separar as duas coisas. Calibrar o limiar no DMD — a única calibração honesta — não recupera a perda (RF, LR: a acurácia balanceada cai em vez de subir): o limiar que o DMD sugere se afasta de 0,5 para um lado e o que a UTA precisaria (diagnóstico), para o outro. 

**PP1 — ordenação por janela (AUC, B3).** Treinado só no DMD atuado, o modelo ordena as janelas reais da UTA com AUC 69,2 [63,7; 74,3] (RF) e 64,5 [58,1; 70,9] (LR); treinado na própria UTA com as mesmas características (B, 5 folds), 80,1 [76,3; 83,8] e 74,4 [69,7; 78,4]. A queda da PP1 (B − C) é 10,9 [6,7; 15,7] pp de AUC no RF (o IC fica todo acima de 0) e 9,8 [3,7; 16,0] pp na LR (o IC fica todo acima de 0). **Há queda real na capacidade de ordenar** ao trocar o treino real pelo atuado, nos dois modelos; ainda assim, C fica acima do acaso nos dois. Em proporção, C preserva 64% (RF) e 60% (LR) da distância de B ao acaso (AUC − 50). Dentro de cada pessoa (AUC média intra-sujeito), a queda é 11,2 [6,8; 15,7] (RF) e 6,7 [0,1; 13,0] pp (LR).

**PP1 — por vídeo (a decisão do app: média dos escores das janelas).** AUC por vídeo: B 92,4 [87,9; 96,2] × C 80,5 [73,0; 87,2] no RF (queda 11,9 [5,6; 18,8] pp) e B 89,0 [82,2; 94,9] × C 76,0 [67,8; 84,7] na LR (queda 13,0 [5,8; 20,8] pp). Acurácia por vídeo no limiar 0,5: B 85,8 [80,0; 91,7] × C 73,3 [67,5; 79,2] (RF; queda 12,5 [5,0; 19,2] pp) e B 85,8 [80,0; 90,8] × C 65,8 [60,0; 71,7] (LR; queda 20,0 [12,5; 26,7] pp); com os limiares calibrados no treino, a queda fica em 21,7 [14,2; 29,2] (RF) e 20,8 [12,5; 28,3] pp (LR).

**Limiar.** RF: no limiar 0,5 o modelo do DMD acerta 48,4% das janelas sonolentas e 82,0% das alertas (acur. bal. 65,2 [61,7; 68,6]); o limiar calibrado no DMD (0,662) leva a 37,0% / 88,7% (acur. bal. 62,9 [60,0; 66,0]); o limiar que seria ótimo na própria UTA (diagnóstico) é 0,388, com acur. bal. 65,9; LR: no limiar 0,5 o modelo do DMD acerta 33,7% das janelas sonolentas e 94,4% das alertas (acur. bal. 64,0 [60,7; 67,4]); o limiar calibrado no DMD (0,535) leva a 30,8% / 95,8% (acur. bal. 63,3 [60,2; 66,7]); o limiar que seria ótimo na própria UTA (diagnóstico) é 0,357, com acur. bal. 65,9. O modelo do DMD erra sobretudo por chamar de alerta janelas sonolentas reais (RF, LR: sensibilidade abaixo da especificidade): a sonolência encenada é mais extrema que a real — nas janelas sonolentas, o maior fechamento mediano é 0,97 s no DMD e 0,17 s na UTA, e o PERCLOS mediano 0,208 × 0,040 (características brutas; seção 10). O limiar ótimo na UTA fica abaixo do escolhido no DMD nos dois modelos: a escala dos escores muda com o domínio, e a calibração no DMD não a corrige por completo. A parte da queda que é de ordenação aparece na AUC (item acima); a que é de calibração, na diferença entre a acurácia balanceada no limiar do DMD e no oráculo.

**Choque de realidade (A − C).** O número de laboratório (A: 82,4 [73,6; 89,7] RF, 83,5 [76,1; 88,9] LR, LOSO no DMD) encolhe 13,1 [3,5; 22,6] (RF) e 18,9 [9,7; 27,8] pp (LR) de AUC no dado real. A − B = 2,2 [−7,1; 10,9] (RF) e 9,1 [1,3; 16,6] pp (LR): RF: A e B indistinguíveis (a tarefa encenada não é, para as mesmas características, claramente mais fácil que a real); LR: A acima de B (a tarefa do DMD é mais fácil que a da UTA mesmo com treino real). Ou seja, o choque de realidade (A − C) mistura duas coisas: a diferença de tarefa (A − B) e a queda da transferência propriamente dita (B − C).

**Normalização por sujeito.** Sem centrar as características, C vai a AUC 67,2 [62,5; 71,6] (RF) e 65,2 [60,6; 69,0] (LR), e B a 69,4 [64,8; 73,7] e 72,5 [67,7; 77,3], contra 69,2 [63,7; 74,3] / 64,5 [58,1; 70,9] (C) e 80,1 [76,3; 83,8] / 74,4 [69,7; 78,4] (B) centradas. RF: centrar muda a AUC de B em 10,8 pp e a de C em 2,0 pp; LR: centrar muda a AUC de B em 1,8 pp e a de C em −0,7 pp. Em escala bruta as bases ocupam regiões diferentes (PERCLOS mediano alerta → sonolento: DMD 0,037 → 0,208; UTA 0,010 → 0,040).

**Sensibilidades (seção 8).** RF: AUC de C entre 58,6 (`alerta_s1`) e 71,1 (`ref_s2_espelho`); queda B − C com IC acima de 0 em 7 de 8 variantes (IC contém 0: `sem_normalizacao`); LR: AUC de C entre 58,5 (`alerta_s1`) e 69,0 (`ref_s2_espelho`); queda B − C com IC acima de 0 em 7 de 8 variantes (IC contém 0: `janela_5s`). RF: a maior AUC de C entre as variantes (71,1) fica abaixo da de B principal (80,1). LR: a maior AUC de C entre as variantes (69,0) fica abaixo da de B principal (74,4). Sem normalização, a queda do RF deixa de ser detectável (2,1 [−1,2; 5,8] pp) porque a centragem ajuda muito mais B (69,4 → 80,1) do que C (67,2 → 69,2), e não porque C melhore. Trocar o alerta da s2 (carro parado) pelo da s1 (carro em movimento) PIORA a transferência (AUC de C 69,2 → 58,6 (RF); 64,5 → 58,5 (LR)). Uma explicação plausível, não testada, é que com o carro em movimento o alerta difere da s5 também em movimento, postura e luz, e o modelo aprende essas diferenças; o resultado é coerente com a escolha (feita antes) da s2, de carro parado como a s5, para o principal. Espelhar no DMD a estrutura de referência da UTA (`ref_s2_espelho`) muda a AUC de C em 1,9 pp (RF) e 4,4 pp (LR); o sentido é compatível com parte da queda vir da assimetria de referência entre as bases, mas com 49 janelas alerta e referência de 1 a 6 janelas por sujeito não é uma evidência forte. Com 15 s o DMD tem só 8 janelas alerta (de 5 sujeitos) e o limiar calibrado no DMD vai a 0,958 (RF) e 0,831 (LR): essa variante é pouco informativa.

**Quais sinais transferem (seção 10).** mesmo sentido (transfere): `perclos`, `taxa_piscadas_min`, `dur_media_piscada_s`, `dur_max_fechamento_s`, `n_fechamentos_05s`, `ear_media`, `ear_dp`; discrimina só na UTA: `amp_media`, `vel_media`; discrimina só no DMD: `frac_bocejo`; não discrimina em nenhuma: `taxa_rosto`. Coeficientes da LR com sinal oposto nas duas bases (treinada no DMD × treinada na UTA): `perclos` −1,20 × 2,24 (importância por permutação em C-LR −2,3); `n_fechamentos_05s` 0,43 × −0,81 (importância por permutação em C-LR 2,3); `ear_media` −1,03 × 0,27 (importância por permutação em C-LR 8,8); `amp_media` 0,82 × −0,35 (importância por permutação em C-LR −3,0). Como PERCLOS, fechamentos longos e EAR médio são correlacionados, um coeficiente isolado não se interpreta; o que importa é a importância por permutação em C: quando ela é negativa, embaralhar a característica MELHORA a AUC na UTA, ou seja, o modelo do DMD a usa num sentido que não vale no dado real. O bocejo não transfere: discrimina no DMD (AUC univariada 57,0 [54,7; 59,3]) e quase não aparece na UTA (50,2 [50,0; 50,5]). Em C, as duas características mais importantes são RF: `dur_max_fechamento_s` (7,7), `perclos` (3,1); LR: `dur_max_fechamento_s` (10,9), `ear_media` (8,8).

**Vídeo de referência (seção 9).** Entre KSS 10 e KSS 5 — dois vídeos que não são a referência — a AUC é 61,5 [57,1; 65,8] (B-RF), 60,8 [56,7; 64,8] (B-LR), 60,2 [55,6; 64,8] (C-RF) e 56,2 [50,9; 61,3] (C-LR). Em B-RF, B-LR, C-RF, C-LR o IC fica acima de 50: há sinal de sonolência além de "vídeo diferente da referência". Em B (RF, LR), KSS 0 × 5 separa mais que KSS 5 × 10: parte do que o modelo aprende é "diferente do vídeo 0" (luz, enquadramento, hora da gravação) — confundimento que vale para B, para C e para os `.npy` oficiais, e que a UTA-RLDD não permite eliminar. Em C (RF), KSS 0 × 5 e KSS 5 × 10 separam no mesmo patamar (RF: 59,8 e 60,2): o modelo treinado no DMD não tem acesso a esse atalho da UTA.

**PP3 — óculos (seção 11).** 3 dos 13 motoristas do DMD usam óculos na s5. O detector de piscadas falha em gE_29 (com óculos): F1 de piscadas 0,108 e 0,0% dos quadros anotados como olho fechado abaixo do limiar. O EAR, porém, ainda ordena fechado × aberto (AUC por quadro 92,8); o problema é a amplitude: o EAR cai só 14,2% ao fechar (0,288 → 0,247), contra um limiar de 0,206 (0,75 × EAR de referência). Em A, gE_29 tem AUC 65,3 (RF) e 80,6 (LR), e o RF acerta 16,7% das janelas sonolentas dele. O estrato com óculos em A (RF: 68,4 × 86,1 sem óculos) inclui esse motorista; os outros motoristas de óculos têm AUC individual de 89,9 (gA_1) e 94,0 (gZ_36) no RF. Os outros motoristas de óculos (gA_1: queda 46,2%, F1 0,897; gZ_36: queda 34,3%, F1 0,797) têm detector normal. Sem óculos, a queda relativa do EAR vai de 30,4% a 55,6% e o F1 de piscadas de 0,488 a 0,942. Com 3 motoristas de óculos, dos quais só um falha, **não dá para atribuir a falha aos óculos**: ela pode vir da armação, de reflexo na lente, do formato do olho ou de um fechamento encenado incompleto — nada disso foi medido. O que se pode afirmar é mais estreito: um limiar de EAR relativo ao olho aberto falha quando a queda do EAR ao fechar é pequena, e isso aconteceu num motorista de óculos. Tirar gE_29 do treino muda a AUC de C de 69,2 para 70,0 (RF); 64,5 para 64,4 (LR). Na UTA não há rótulo de óculos: a PP3 sobre óculos em sonolência real fica sem resposta neste trabalho.

**Relação com o Exp. 1 (seção 13).** RF: AUC média por fold 92,8 no Exp. 1 × 80,5 aqui (83,2 com a referência no teste); VA por voto 85,5 × 85,2 (88,6); LR: AUC média por fold 81,8 no Exp. 1 × 74,3 aqui (74,4 com a referência no teste); VA por voto 77,5 × 83,3 (81,7). A AUC do Exp. 1 é por sequência de 30 piscadas (que cobre minutos de vídeo) e a daqui é por janela de 10 s — unidades diferentes, então a AUC não se compara diretamente; a VA por voto é a mesma métrica nos dois (decisão por vídeo). RF: VA de B no mesmo patamar do Exp. 1 (−0,3 pp; médias de 5 folds, sem IC). LR: VA de B acima do Exp. 1 (5,8 pp; médias de 5 folds, sem IC). A queda da PP1 deve ser lida contra B (mesmo extrator, mesmas janelas), não contra o Exp. 1 (recomendação do Exp. 5a).

**Amostra.** UTA com 60 sujeitos (5 folds); DMD com 87 janelas alerta de 13 sujeitos. O IC da queda B − C em AUC por janela tem até 12,4 pp de largura. Na rodada preliminar A era 95,2 (RF) e 96,0 (LR) com janelas alerta sobrepostas à referência; agora, com a referência separada e alerta de outra sessão, 82,4 e 83,5: o número de laboratório da rodada preliminar estava inflado, e os números preliminares de A, B e C não devem ser citados.

## 15. Limitações (valem para qualquer número acima)

1. **Rótulos de natureza diferente.** No DMD o rótulo sonolento vem de EVENTOS anotados (fechamento ≥ 0,5 s, bocejo) que o ator encenou, e o alerta, de trechos de direção segura de OUTRA sessão (s2); na UTA o rótulo é o ESTADO autorrelatado (KSS) do vídeo inteiro (duração mediana de 10,2 min). O modelo do DMD aprende "há um evento encenado nesta janela (e ela não é da s2)"; a UTA pergunta "esta pessoa está sonolenta". Parte do que se mede em C é essa diferença de pergunta, não só de domínio.
2. **Condições de gravação opostas.** DMD: carro PARADO (s2 e s5), câmera fixa, encenação, sessão de sonolência com duração mediana de 3,0 min; UTA: gravação caseira com celular/webcam (fps e resoluções variados), sonolência real, vídeos com duração mediana de 10,2 min. C mistura troca de câmera, de pessoa, de cenário e de fenômeno; nenhuma análise aqui separa esses fatores.
3. **Poucos sujeitos atuados.** 13 motoristas no DMD (3 com óculos), 87 janelas alerta; o modelo de C aprende "alerta" e "sonolento" com pouca variedade de pessoas, e o IC de A é largo.
4. **Alerta do DMD de outro dia.** A s2 (alerta) foi gravada em outro dia que a s5 em 10 dos 13 sujeitos, e a referência (s3) no mesmo dia da s5 em todos: diferenças de montagem, luz e óculos entre dias entram na classe alerta. É o espelho da UTA, onde a referência sai do mesmo vídeo das janelas alerta; a sensibilidade `ref_s2_espelho` e o estrato por dia (seções 8 e 11) medem o efeito, mas não o eliminam.
5. **Classe confundida com o vídeo na UTA**: cada estado é um vídeo gravado em outro momento, e as janelas alerta dividem o vídeo com a referência da normalização (controle na seção 9).
6. **Detector de piscadas imperfeito.** F1 médio de piscadas contra a anotação da s5: 0,731 por motorista, com falha quase total em gE_29 (seção 11); na UTA, 2 vídeos ficaram sem nenhuma piscada detectada (seção 3). As características herdam esses erros.
7. **Janela curta.** Em 10 s a taxa de piscadas é ruidosa e 34,9% das janelas avaliáveis da UTA não têm piscada (duração, amplitude e velocidade imputadas pela mediana do treino).
8. **Bocejo quase ausente na UTA** com o limiar de MAR de 0,5 sustentado por 1 s: um sinal que define parte dos eventos do DMD praticamente não existe no dado real.
9. **IC sem a variância do treino.** O bootstrap reamostra as predições de teste (não retreina); com 13 sujeitos no DMD, a incerteza de C devida ao treino não aparece nos IC.
10. **Muitas comparações.** A variante principal, os modelos, os hiperparâmetros e o limiar 0,5 foram fixados antes; as sensibilidades são descritivas e não há correção para multiplicidade: um IC isolado que exclui 0 numa sensibilidade não é evidência forte.

## 16. Como reproduzir

```bash
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate                              # na raiz do repositório
bash src/features/run_features.sh                      # só se sinais/janelas mudarem (etapa de características)
bash src/experiments/run_exp5b.sh                      # análises + CSV + relatório + figura (~20 min, log em results/logs/)
python src/experiments/exp5_transfer.py --so-relatorio  # refaz .md e figura a partir dos CSV
python src/experiments/exp5_transfer.py --rapido /tmp/x --sem-relatorio  # teste de fumaça (saídas fora de results/)
```

## Apêndice A. Por sujeito da UTA (B3, variante principal): AUC intra-sujeito em B e em C

| Sujeito | Fold | Mapeamento oficial | Janelas alerta / sonolento | AUC B (RF) | AUC C (RF) | AUC B (LR) | AUC C (LR) | Escore médio de C-RF: alerta → sonolento | Vídeo sem piscada |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| F1_01 | 1 | confirmado | 79 / 119 | 61,9 | 67,9 | 50,2 | 86,7 | 0,22 → 0,40 |  |
| F1_02 | 1 | confirmado | 80 / 119 | 69,3 | 69,5 | 76,1 | 65,9 | 0,10 → 0,22 |  |
| F1_03 | 1 | confirmado | 91 / 122 | 100,0 | 96,7 | 99,3 | 95,2 | 0,18 → 0,75 |  |
| F1_04 | 1 | confirmado | 80 / 122 | 90,4 | 61,5 | 66,3 | 35,9 | 0,19 → 0,31 |  |
| F1_05 | 1 | confirmado | 80 / 119 | 74,6 | 56,7 | 83,6 | 68,6 | 0,59 → 0,63 |  |
| F1_06 | 1 | confirmado | 80 / 84 | 46,8 | 36,1 | 60,6 | 55,3 | 0,26 → 0,21 |  |
| F1_07 | 1 | confirmado | 79 / 119 | 88,6 | 93,0 | 56,8 | 85,7 | 0,31 → 0,68 |  |
| F1_08 | 1 | confirmado | 79 / 118 | 77,9 | 52,2 | 67,7 | 48,7 | 0,32 → 0,36 |  |
| F1_09 | 1 | confirmado | 80 / 132 | 52,8 | 30,8 | 42,5 | 21,9 | 0,30 → 0,17 |  |
| F1_10 | 1 | confirmado | 71 / 124 | 67,7 | 81,5 | 64,1 | 84,4 | 0,14 → 0,44 |  |
| F1_11 | 1 | confirmado | 82 / 123 | 90,6 | 69,5 | 88,9 | 88,4 | 0,44 → 0,63 |  |
| F1_12 | 1 | ausente_do_oficial | 73 / 99 | 81,9 | 71,6 | 82,7 | 53,7 | 0,19 → 0,30 |  |
| F2_13 | 2 | confirmado | 80 / 123 | 99,6 | 88,7 | 93,8 | 99,6 | 0,47 → 0,84 |  |
| F2_14 | 2 | confirmado | 80 / 123 | 68,7 | 66,7 | 83,0 | 68,4 | 0,19 → 0,45 |  |
| F2_15 | 2 | provavel | 84 / 122 | 66,5 | 10,1 | 4,9 | 1,0 | 0,25 → 0,03 | F2_15_10 |
| F2_16 | 2 | confirmado | 62 / 119 | 67,8 | 73,9 | 61,6 | 80,8 | 0,36 → 0,57 |  |
| F2_17 | 2 | confirmado | 79 / 120 | 85,1 | 82,6 | 78,1 | 93,2 | 0,36 → 0,74 |  |
| F2_18 | 2 | confirmado | 79 / 126 | 91,7 | 84,9 | 62,8 | 77,8 | 0,20 → 0,57 |  |
| F2_19 | 2 | confirmado | 79 / 119 | 82,8 | 85,3 | 13,8 | 79,2 | 0,12 → 0,25 |  |
| F2_20 | 2 | confirmado | 86 / 142 | 87,5 | 67,6 | 83,0 | 76,8 | 0,53 → 0,72 |  |
| F2_21 | 2 | provavel | 79 / 119 | 85,6 | 57,5 | 88,0 | 39,0 | 0,28 → 0,30 |  |
| F2_22 | 2 | confirmado | 80 / 121 | 94,3 | 92,8 | 97,1 | 81,1 | 0,22 → 0,70 |  |
| F2_23 | 2 | provavel | 77 / 120 | 69,4 | 73,4 | 68,7 | 65,5 | 0,25 → 0,52 |  |
| F2_24 | 2 | confirmado | 97 / 153 | 89,3 | 86,1 | 84,1 | 69,1 | 0,20 → 0,46 |  |
| F3_25 | 3 | indeterminado | 79 / 119 | 94,8 | 95,2 | 74,1 | 92,3 | 0,24 → 0,66 |  |
| F3_26 | 3 | provavel | 81 / 138 | 91,1 | 78,0 | 77,6 | 55,9 | 0,24 → 0,50 |  |
| F3_27 | 3 | indeterminado | 86 / 123 | 67,7 | 29,1 | 67,7 | 18,5 | 0,45 → 0,27 | F3_27_5 |
| F3_28 | 3 | confirmado | 79 / 123 | 69,2 | 37,8 | 78,5 | 30,3 | 0,46 → 0,36 |  |
| F3_29 | 3 | indeterminado | 83 / 127 | 52,1 | 53,6 | 34,2 | 82,9 | 0,55 → 0,56 |  |
| F3_30 | 3 | confirmado | 89 / 120 | 80,7 | 64,6 | 73,7 | 84,7 | 0,37 → 0,52 |  |
| F3_31 | 3 | confirmado | 89 / 128 | 99,2 | 95,9 | 98,5 | 95,0 | 0,36 → 0,86 |  |
| F3_32 | 3 | confirmado | 118 / 118 | 86,7 | 74,2 | 74,6 | 78,4 | 0,31 → 0,56 |  |
| F3_33 | 3 | confirmado | 77 / 111 | 97,6 | 95,9 | 97,2 | 94,6 | 0,24 → 0,87 |  |
| F3_34 | 3 | indeterminado | 99 / 121 | 67,9 | 76,4 | 67,2 | 88,6 | 0,47 → 0,71 |  |
| F3_35 | 3 | confirmado | 81 / 131 | 70,2 | 86,7 | 74,4 | 75,1 | 0,19 → 0,45 |  |
| F3_36 | 3 | confirmado | 81 / 124 | 88,9 | 89,1 | 86,1 | 86,1 | 0,14 → 0,43 |  |
| F4_37 | 4 | confirmado | 90 / 135 | 86,1 | 72,2 | 86,6 | 79,2 | 0,31 → 0,55 |  |
| F4_38 | 4 | confirmado | 79 / 119 | 94,4 | 44,5 | 93,6 | 54,5 | 0,30 → 0,29 |  |
| F4_39 | 4 | confirmado | 85 / 120 | 93,7 | 87,7 | 96,7 | 79,4 | 0,17 → 0,47 |  |
| F4_40 | 4 | confirmado | 79 / 119 | 99,8 | 100,0 | 100,0 | 100,0 | 0,13 → 0,65 |  |
| F4_41 | 4 | confirmado | 82 / 188 | 81,1 | 28,2 | 76,8 | 18,9 | 0,16 → 0,09 |  |
| F4_42 | 4 | confirmado | 79 / 119 | 89,5 | 77,0 | 60,8 | 85,3 | 0,51 → 0,74 |  |
| F4_43 | 4 | provavel | 79 / 119 | 89,9 | 90,4 | 87,6 | 93,4 | 0,20 → 0,62 |  |
| F4_44 | 4 | provavel | 80 / 120 | 75,2 | 64,3 | 84,5 | 61,4 | 0,36 → 0,49 |  |
| F4_45 | 4 | confirmado | 81 / 128 | 80,4 | 37,2 | 77,1 | 25,2 | 0,27 → 0,22 |  |
| F4_46 | 4 | confirmado | 79 / 121 | 96,6 | 77,9 | 96,3 | 98,6 | 0,47 → 0,75 |  |
| F4_47 | 4 | confirmado | 79 / 121 | 83,9 | 68,6 | 84,4 | 85,7 | 0,39 → 0,58 |  |
| F4_48 | 4 | confirmado | 83 / 121 | 99,4 | 98,9 | 94,9 | 93,0 | 0,17 → 0,81 |  |
| F5_49 | 5 | ausente_do_oficial | 80 / 295 | 80,0 | 79,8 | 82,6 | 30,6 | 0,23 → 0,52 |  |
| F5_50 | 5 | confirmado | 79 / 132 | 82,4 | 68,3 | 69,2 | 74,8 | 0,26 → 0,40 |  |
| F5_51 | 5 | confirmado | 79 / 122 | 86,7 | 59,5 | 82,0 | 65,9 | 0,41 → 0,53 |  |
| F5_52 | 5 | provavel | 79 / 119 | 82,4 | 37,6 | 78,1 | 32,9 | 0,21 → 0,19 |  |
| F5_53 | 5 | confirmado | 79 / 124 | 87,7 | 72,6 | 62,7 | 83,5 | 0,18 → 0,43 |  |
| F5_54 | 5 | confirmado | 84 / 125 | 53,1 | 31,3 | 45,2 | 8,2 | 0,25 → 0,15 |  |
| F5_55 | 5 | confirmado | 93 / 136 | 91,1 | 87,4 | 88,7 | 72,9 | 0,13 → 0,57 |  |
| F5_56 | 5 | confirmado | 80 / 124 | 95,8 | 84,4 | 93,1 | 93,8 | 0,17 → 0,53 |  |
| F5_57 | 5 | confirmado | 83 / 121 | 91,8 | 81,6 | 75,8 | 50,8 | 0,19 → 0,52 |  |
| F5_58 | 5 | confirmado | 79 / 120 | 98,9 | 96,8 | 93,9 | 94,3 | 0,19 → 0,76 |  |
| F5_59 | 5 | confirmado | 79 / 119 | 48,0 | 64,7 | 63,9 | 73,5 | 0,17 → 0,38 |  |
| F5_60 | 5 | confirmado | 79 / 124 | 46,1 | 53,7 | 72,9 | 48,3 | 0,63 → 0,63 |  |

