# Exp. 5b: robustez da queda B − C

Execução: 2026-10-26 20:51. Gerado por `src/experiments/exp5b_robustez.py` (mesmos dados, receita de modelos, modelos finais de C e reamostragem de sujeitos do Exp. 5b). Esquema B3 (alerta × sonolento), janelas de 10 s. AUC × 100; quedas em pontos de AUC; IC 95% por bootstrap percentil de sujeitos (1.000 reamostragens).

## 1. Referência alternativa na UTA-RLDD (vídeo de pouca vigilância)

Na rodada principal, a referência da centragem na UTA-RLDD é o último terço do vídeo alerta, e as janelas alertas avaliadas saem do mesmo vídeo. Aqui a referência é a média das janelas do vídeo de pouca vigilância do sujeito (código 5, KSS 6 e 7), que não entra no B3: nem as janelas alertas nem as sonolentas vêm do vídeo da referência. O DMD não muda.

| Modelo | Referência | AUC B | AUC C | Queda B − C (janela) | Queda B − C (vídeo) |
|---|---|---|---|---|---|
| RF | último terço do vídeo alerta (principal) | 80,1 | 69,2 | 10,9 [6,7; 15,7] | 11,9 [5,6; 18,8] |
| RF | vídeo de pouca vigilância | 68,4 | 66,1 | 2,3 [−2,5; 6,8] | −2,3 [−9,1; 4,3] |
| RL | último terço do vídeo alerta (principal) | 74,4 | 64,5 | 9,8 [3,7; 16,0] | 13,0 [5,8; 20,8] |
| RL | vídeo de pouca vigilância | 73,7 | 66,2 | 7,6 [3,0; 12,5] | 8,4 [3,4; 13,8] |

Leitura: com o RF, a queda depende da referência; com a RL, ela se mantém. O RF treinado na UTA-RLDD aproveita a semelhança entre as janelas alertas e a referência tirada do mesmo vídeo; o modelo do DMD não tem como aprender essa estrutura. Ressalva: o limiar de EAR do detector de piscadas continua o do vídeo alerta.

## 2. Bootstrap em dois níveis

Em cada reamostragem, os 13 sujeitos do DMD são sorteados com reposição e C é treinado de novo; os sujeitos da UTA-RLDD seguem a reamostragem pareada do Exp. 5b.

| Modelo | Queda B − C (janela) | IC só UTA-RLDD (Exp. 5b) | IC em dois níveis |
|---|---|---|---|
| RF | 10,9 | [6,7; 15,7] | [5,7; 19,5] |
| RL | 9,8 | [3,7; 16,0] | [1,9; 18,1] |

## 3. Volume do treino de B

B treinado com só 13 sujeitos da UTA-RLDD por fold (sorteados), com todas as janelas deles ou com 510 janelas (o tamanho do treino de C), em 20 sorteios. Valor = mediana dos sorteios; entre colchetes, os percentis 2,5 e 97,5 dos sorteios (variação do sorteio, não IC).

| Modelo | B completo | B com 13 sujeitos | B com 13 sujeitos e 510 janelas | C |
|---|---|---|---|---|
| RF | 80,1 | 78,6 [77,0; 80,3] | 78,8 [76,6; 80,5] | 69,2 |
| RL | 74,4 | 71,5 [68,1; 73,2] | 71,2 [67,8; 73,6] | 64,5 |

## 4. Vazamento no mesmo cenário

Folds oficiais por sujeito contra partição aleatória das janelas em 5 partes (média das sementes 42, 43 e 44). Janelas vizinhas do mesmo vídeo (passo de 5 s, sobreposição de 5 s) caem nos dois lados.

| Modelo | Características | Acur. bal. por sujeito | Acur. bal. aleatória | Inflação (pp) | AUC por sujeito | AUC aleatória | Inflação da AUC |
|---|---|---|---|---|---|---|---|
| RF | centrada | 74,2 | 86,8 | 12,6 | 80,1 | 94,0 | 13,9 |
| RF | bruta | 62,9 | 77,5 | 14,7 | 69,4 | 85,3 | 15,9 |
| RL | centrada | 72,0 | 73,1 | 1,2 | 74,4 | 76,5 | 2,1 |
| RL | bruta | 68,1 | 69,0 | 0,9 | 72,5 | 74,2 | 1,7 |

Para comparar: a queda B − C da acurácia balanceada do RF (limiar 0,5) é 9,0 [5,8; 12,2] pp.

## 5. Ponto de operação da regra (diagnóstico)

Regra do rótulo do DMD aplicada aos nossos sinais (fechamento de 0,5 s ou mais ou bocejo). Para cada modelo, o limiar é o que dá o mesmo falso alarme da regra nas janelas alertas (KSS 0) do B3. Esse limiar usa o rótulo do teste: a tabela só diz onde a regra fica em relação às curvas ROC dos modelos.

Regra: falso alarme 2,6%, sensibilidade 24,1%.

| Modelo | Sensibilidade no falso alarme da regra (%) |
|---|---|
| RF, C (DMD→UTA), características brutas | 15,3 |
| RL, C (DMD→UTA), características brutas | 19,2 |
| RF, C (DMD→UTA), centradas | 16,4 |
| RL, C (DMD→UTA), centradas | 24,7 |
| RF, B (UTA→UTA), centradas | 24,2 |
| RL, B (UTA→UTA), centradas | 27,3 |
