# Exp. 6 — Robustez noturna com o NITYMED (PP3) — rodada final

Gerado por `src/experiments/exp6_noite.py` (versão 2.2) em 27/09/2026; nenhum número digitado à mão: tudo sai de `results/exp6_noite.csv` (longo), `results/exp6_noite_videos.csv` (por vídeo lógico), `results/exp6_noite_janelas.csv` (por janela, com as probabilidades) e dos demais CSV citados. Log: `results/logs/exp6_final_noite.log` (luminância: `results/logs/exp6_final_luminancia.log`). Seed 42. A rodada preliminar (janelas de 30 s, UTA-RLDD só com o Fold 1, DMD só com a s5) foi descartada; seus arquivos foram removidos em 06/10/2026.

**Dados**: janelas de 10 s com passo de 5 s (`data/features/final/<base>/windows_10s.csv`). NITYMED: 126 vídeos, 979 janelas, 529 avaliáveis (fora do primeiro terço, que é a referência). UTA-RLDD completa: 180 vídeos lógicos (182 arquivos) de 60 sujeitos, 22.455 janelas, 19.955 avaliáveis. DMD: 65 sessões de 14 sujeitos (s5 16, s1 17, s2 16, s3 16), gravadas entre 09h06 e 16h36 (hora local do `recordTime`, de 04/03/2019 a 29/04/2019) — por isso "dia"; janelas: 423 sonolentas (s5), 90 alerta da s2 e 109 da s1 (94 com o carro parado e 105 em movimento pelo campo `setup`). Luminância: medida (`results/exp6_luminancia_videos.csv`).

**O NITYMED não tem classe alerta nem anotação quadro a quadro: nenhuma acurácia, AUC ou F1 foi calculada nele.** As medidas são de captura (rosto), de sinal (EAR) e de comportamento dos detectores (fração de janelas marcadas). A comparação noite × dia NÃO isola a noite: as bases também diferem em câmera, distância e tamanho do rosto, carro em movimento × parado × casa, protocolo (encenado × autorrelato × autenticidade não documentada) e pessoas. A UTA-RLDD é autogravada (celular/webcam) e não documenta a hora nem a iluminação; o que se compara é a luminância medida. O DMD em movimento (sessões de distração, direção real) é a única referência diurna com o carro andando: ele separa parte do efeito do movimento do efeito da noite.

## 1. Números-chave

- **(a) Detecção de rosto por quadro** (média por vídeo [IC 95%]): NITYMED 0,9915 [0,9875; 0,9948] (126 vídeos; mín 0,8737 em `Yawning_Male_P1042773_720`; 8 vídeos < 0,95; 3,1% das janelas < 0,90); UTA-RLDD 0,9977 [0,9940; 0,9996] (180 vídeos; mín 0,7380 em `F5_49_10`; 0,7% das janelas < 0,90); DMD s5 0,9978 [0,9958; 0,9995]; DMD em movimento (direção segura) 0,9965 [0,9935; 0,9987]. Diferenças NITYMED − UTA-RLDD: -0,0061 [-0,0107; -0,0014]; − DMD s5 (parado): -0,0063 [-0,0108; -0,0025]; − DMD em movimento: -0,0050 [-0,0094; -0,0005]; − DMD parado: -0,0073 [-0,0113; -0,0038].
- **(b) Qualidade do EAR** (medianas por vídeo): SNR rápida (amplitude das piscadas ÷ ruído quadro a quadro) NITYMED 7,2, UTA-RLDD 21,5, DMD s5 23,1, DMD parado 13,6, **DMD em movimento 8,7**; ruído rápido 0,0084 / 0,0037 / 0,0046 / 0,0059 / 0,0097. No DMD, com os mesmos motoristas, a mesma câmera e o mesmo dia, o carro em movimento muda o ruído rápido em 0,0043 [0,0030; 0,0054] e a SNR rápida em -5,4 [-6,6; -3,5] (mediana das diferenças por sujeito, 11 sujeitos).
- **(c) Taxa de disparo** (janelas de 10 s avaliáveis marcadas "sonolento"; características brutas; probabilidade ≥ 0,5): modelo do DMD (Exp. 5b final): microssono 44,7%, bocejo 73,0% × falso alarme no alerta de dia — UTA KSS 0 26,0%, DMD parado 45,7%, DMD em movimento 42,9% — e disparo na UTA KSS 10 52,2%; modelo da UTA-RLDD (B3): microssono 83,5%, bocejo 89,5% × falso alarme no alerta de dia — UTA KSS 0 43,6%, DMD parado 69,1%, DMD em movimento 77,1% — e disparo na UTA KSS 10 69,4%; regra do rótulo do DMD: microssono 15,0%, bocejo 61,7% × falso alarme no alerta de dia — UTA KSS 0 2,6%, DMD parado 14,9%, DMD em movimento 23,8% — e disparo na UTA KSS 10 24,1%.
- **(d) Microssono**: PERCLOS mediano por vídeo 2,9% (Bocejo 6,5%; UTA KSS 0 1,4%); 58% dos 19 vídeos têm ao menos um fechamento ≥ 0,5 s e 16% ao menos um > 1,5 s (5 no total). Sem referência alerta (nível aberto = p90 do EAR do próprio vídeo): PERCLOS mediano 19,5% e 63% dos vídeos com fechamento > 1,5 s.
- **(e) Ficha**: `data/nitymed/index.csv` e `results/exp6_nitymed_ficha.md` — 126 vídeos em disco (19 Microssono, 107 Bocejo) × 130 declarados; licença da cópia usada (Kaggle) CC BY-ND 4.0; autenticidade dos comportamentos não documentada; 3 vídeos da pasta Bocejo com > 90 s e nenhum bocejo detectado (suspeitos de pasta errada; sensibilidade na seção 5.6).

## 2. Desenho e decisões

| Item | Decisão | Por quê |
|---|---|---|
| Janela | 10 s, passo 5 s, nas três bases (grade por arquivo; nas sessões de distração do DMD, janelas ancoradas nos trechos safe_drive) | regra final (notas, item 9): os trechos safe_drive do DMD são curtos |
| Modelo do DMD | models/exp5/exp5b_final_C_sem_normalizacao_RF.joblib (Exp. 5b final, execução 20260927-211326; metadados models/exp5/exp5b_final_modelos.json) | especificação: aplicar o modelo final de transferência do Exp. 5b |
| Treino do DMD | 510 janelas de 13 sujeitos: 423 sonolentas (s5 com fechamento anotado ≥ 0,5 s ou bocejo) × 87 alerta (safe_drive da s2); gE_28 (sem s5) fora | desenho do Exp. 5b final (notas, itens 7, 9 e 10) |
| Modelo intra-UTA (B3) | treinado aqui (o Exp. 5b não salva modelos intra-UTA), mesma receita: 12487 janelas de 60 sujeitos, KSS 10 = 7581 × KSS 0 = 4906 (KSS 5, referência e fronteira fora); `models/exp6/rf_uta_<variante>.joblib` | especificação (B3) |
| Receita | Pipeline(SimpleImputer(median, keep_empty_features=True), RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=42, n_jobs=6)) — a do Exp. 5b; quando o modelo do 5b é carregado, os modelos fora da amostra são clones dele (sklearn.base.clone: mesmos hiperparâmetros); scikit-learn 1.9.1 | a do Exp. 5b |
| Características | as 11 da janela (`perclos`, `taxa_piscadas_min`, `dur_media_piscada_s`, `dur_max_fechamento_s`, `n_fechamentos_05s`, `ear_media`, `ear_dp`, `amp_media`, `vel_media`, `frac_bocejo`, `taxa_rosto`); sensibilidade sem `taxa_rosto` (10) | `taxa_rosto` mede captura, não sonolência |
| Variante principal no Exp. 6 | **brutas (`sem_normalizacao`)**; a principal do Exp. 5b (centradas `_c`) entra como sensibilidade | o NITYMED não tem trecho alerta: a referência de `_c` (e do limiar de EAR) nele é o primeiro terço do próprio vídeo sonolento, e centrar nela apaga parte do sinal |
| Janelas avaliadas | UTA e NITYMED: `avaliavel` (sem referência nem fronteira); DMD: alerta = janelas de direção segura das sessões s1/s2, agrupadas pelo campo `setup` (carro parado × em movimento), e sonolento = s5 com rótulo | referência nunca avaliada; no NITYMED as janelas que tocam o primeiro terço dividem quadros com a calibração (sensibilidade com todas as janelas na seção 5.6) |
| Fora da amostra | UTA pelo modelo da UTA: 5 folds oficiais; DMD pelo modelo do DMD: deixa-um-sujeito-de-fora (clone do modelo usado); o modelo da outra base e o NITYMED: modelo inteiro | nenhum número do próprio domínio vem de janelas vistas no treino |
| Disparo | probabilidade da classe sonolenta ≥ 0,5 (a regra `s >= limiar` do Exp. 5b); secundários: limiar calibrado só com o treino (a regra do 5b: máxima acurácia balanceada nos escores fora da amostra do treino) e limiar com 10% de falso alarme nas janelas alerta da UTA | nenhum ajuste no NITYMED |
| Regra sem treino | janela marcada se tem fechamento ≥ 0,5 s (EAR < limiar) ou bocejo (MAR > 0,5 por ≥ 1 s) — a regra que define o rótulo do DMD, aplicada aos NOSSOS sinais | separa o detector de sinais do classificador |
| Vídeo lógico | as partes `_p1`/`_p2` (F3_32_10, F5_49_10) são concatenadas em ordem, com tempo contínuo, antes das medidas por vídeo; luminância = mediana das amostras das partes juntas | notas, item 8 |
| DMD nas medidas de sinal | sessões de distração: só os quadros dos trechos `safe_drive` (olhando a estrada); a s5 inteira; a sessão inteira da distração só na tabela de captura | as distrações encenadas (celular, rádio, olhar para trás) derrubam rosto e EAR por outro motivo |
| Carro em movimento × parado | pelo campo `setup` do OpenLABEL (não pela sessão): `gB_10` s1 (2019-03-11 15:24) = Car Stopped, `gB_6` s2 (2019-03-11 13:46) = Car Moving, `gZ_33` s1 (2019-04-08 10:08) = Car Stopped, `gZ_33` s1 (2019-04-08 16:18) = Car Stopped | ruído rápido do EAR nos trechos de direção segura (`results/exp6_noite_videos.csv`): `gB_10` s1 (Car Stopped) 0,0058; `gB_6` s2 (Car Moving) 0,0072; `gZ_33` s1 (Car Stopped) 0,0075; `gZ_33` s1 (Car Stopped) 0,0061 — sessões conforme o protocolo: s1 em movimento mediana 0,0099 (faixa 0,0075–0,0130), s2 parada mediana 0,0055 (faixa 0,0037–0,0124). As faixas se sobrepõem: o ruído dessas sessões não confirma nem desmente o `setup` registrado; o agrupamento por `setup` segue o registro oficial, e a comparação por sessão (s1 × s2) fica como sensibilidade (as duas levam à mesma conclusão na seção 4) |
| Olho aberto (b) | quadro com rosto, EAR ≥ limiar e a mais de 0,1 s de qualquer piscada/fechamento (bordas estendidas) | exclui transições |
| Ruído (b) | (i) DP do EAR nos quadros de olho aberto; (ii) ruído rápido = 1,4826 × MAD do resíduo em torno da mediana móvel de 1 s | (i) inclui deriva lenta (pose); (ii) isola o tremor quadro a quadro |
| Amplitude (b) | média da amplitude **estendida** das piscadas detectadas (`amplitude_ext`) | só existe para piscadas detectadas (viés de seleção) |
| PERCLOS (d) | fração dos quadros com rosto com EAR < limiar do pipeline (0,75 × mediana da referência: último terço do vídeo 0 na UTA, safe_drive da s3 no DMD, primeiro terço do próprio vídeo no NITYMED); sensibilidade: 0,75 × percentil 90 do EAR do próprio vídeo | no NITYMED não há trecho alerta para calibrar |
| IC 95% | bootstrap de agrupamentos, 1000 reamostragens, semente 42: vídeo (NITYMED), sujeito (UTA, DMD); diferenças pareadas: mediana das diferenças por sujeito, bootstrap de sujeitos | vídeos do mesmo motorista do NITYMED não são identificáveis: IC otimista |

## 3. (a) Detecção de rosto por quadro — noite × dia

Fonte: arrays `face` dos `.npz` de `data/signals/<base>/` (Face Mesh em modo vídeo, confianças 0,5, igual ao app), por vídeo lógico; janelas de `data/features/final/<base>/windows_10s.csv`.

| Base / grupo | Vídeos | Agrup. | Quadros | Média por vídeo [IC 95%] | Mediana | p10 | Mínimo | Vídeos < 0,99 | Vídeos < 0,95 | Janelas < 0,90 | Maior lacuna (s), mediana | Maior lacuna (s), máx. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **NITYMED (noite, carro em movimento)** | 126 | 126 | 146.484 | 0,9915 [0,9875; 0,9948] | 1,0000 | 0,9732 | 0,8737 | 26 | 8 | 3,1% | 0,00 | 3,04 |
| Microssono | 19 | 19 | 58.328 | 0,9923 [0,9849; 0,9973] | 0,9981 | 0,9815 | 0,9398 | 5 | 1 | 3,2% | 0,16 | 2,04 |
| Bocejo | 107 | 107 | 88.156 | 0,9914 [0,9869; 0,9949] | 1,0000 | 0,9714 | 0,8737 | 21 | 7 | 3,0% | 0,00 | 3,04 |
| **UTA-RLDD (autogravado)** | 180 | 60 | 3.120.071 | 0,9977 [0,9940; 0,9996] | 1,0000 | 0,9979 | 0,7380 | 3 | 1 | 0,7% | 0,00 | 86,02 |
| KSS 0 (alerta) | 60 | 60 | 1.043.367 | 0,9995 [0,9990; 0,9999] | 1,0000 | 0,9989 | 0,9881 | 1 | 0 | 0,1% | 0,00 | 6,97 |
| KSS 5 | 60 | 60 | 1.015.381 | 0,9993 [0,9988; 0,9997] | 1,0000 | 0,9984 | 0,9901 | 0 | 0 | 0,2% | 0,00 | 5,07 |
| KSS 10 (sonolento) | 60 | 60 | 1.061.323 | 0,9943 [0,9841; 0,9995] | 1,0000 | 0,9974 | 0,7380 | 2 | 1 | 1,7% | 0,00 | 86,02 |
| **DMD s5 (dia, carro parado, sonolência encenada)** | 16 | 13 | 88.222 | 0,9978 [0,9958; 0,9995] | 0,9998 | 0,9945 | 0,9832 | 1 | 0 | 0,4% | 0,03 | 2,02 |
| sem óculos | 13 | 10 | 71.966 | 0,9974 [0,9951; 0,9994] | 0,9997 | 0,9932 | 0,9832 | 1 | 0 | 0,4% | 0,07 | 2,02 |
| com óculos | 3 | 3 | 16.256 | 0,9995 [0,9985; 1,0000] | 1,0000 | 0,9988 | 0,9985 | 0 | 0 | 0,0% | 0,00 | 0,27 |
| **DMD, direção segura com o carro em movimento (dia)** | 15 | 12 | 60.703 | 0,9965 [0,9935; 0,9987] | 0,9989 | 0,9879 | 0,9865 | 3 | 0 | 1,0% | 0,10 | 0,81 |
| **DMD, direção segura com o carro parado (dia)** | 18 | 13 | 63.000 | 0,9989 [0,9977; 0,9997] | 1,0000 | 0,9952 | 0,9919 | 0 | 0 | 0,0% | 0,00 | 0,27 |
| DMD, carro em movimento, sessão inteira (com as distrações encenadas) | 15 | 12 | 111.389 | 0,9878 [0,9778; 0,9946] | 0,9934 | 0,9724 | 0,9406 | 5 | 1 | — | 0,57 | 1,44 |

Diferenças da média por vídeo (bootstrap independente por agrupamento): NITYMED − UTA-RLDD = -0,0061 [-0,0107; -0,0014]; NITYMED − DMD s5 (parado) = -0,0063 [-0,0108; -0,0025]; NITYMED − DMD em movimento = -0,0050 [-0,0094; -0,0005]; NITYMED − DMD parado = -0,0073 [-0,0113; -0,0038]. Pareadas por sujeito no DMD (direção segura): setup (Car Moving − Car Stopped), sessões s1/s2 = -0,0011 [-0,0028; 0,0000] (11 sujeitos); sessão do protocolo (s1 − s2) = -0,0008 [-0,0014; 0,0007] (14 sujeitos).

Os 8 vídeos noturnos com menor detecção: `Yawning_Male_P1042773_720` 0,874 (maior lacuna 2,88 s); `Yawning_Male_P1042776_720` 0,895 (maior lacuna 0,88 s); `Yawning_Male_P1042775_720` 0,930 (maior lacuna 1,60 s); `Yawning_Male_P1042770_720` 0,930 (maior lacuna 1,32 s); `Yawning_Male_P1042784_720` 0,934 (maior lacuna 3,04 s); `Yawning_Female_P1043113_720` 0,935 (maior lacuna 1,80 s); `Microsleep_Male_P1042772_720` 0,940 (maior lacuna 1,76 s); `Yawning_Female_P1043095_720` 0,942 (maior lacuna 0,84 s).
Os 3 vídeos da UTA-RLDD com menor detecção: `F5_49_10` 0,738 (270×480 + 162×288 px; maior lacuna 86,0 s); `F2_23_10` 0,960 (1280×720 px; maior lacuna 17,6 s); `F2_23_0` 0,988 (1280×720 px; maior lacuna 7,0 s). A pior captura entre todas as bases é de um vídeo da UTA-RLDD com resolução muito baixa, não de um vídeo noturno.

Conferência: diferença máxima entre a taxa recalculada dos `.npz` (por vídeo lógico) e a do `index.csv` da extração (ponderada pelos quadros das partes) = 2.22e-16 (371 vídeos lógicos).

## 4. (b) Qualidade do sinal de EAR

Medianas por vídeo (IC 95% por bootstrap de agrupamentos só nas linhas principais). Fonte: `results/exp6_noite_videos.csv`. No DMD em movimento/parado, só os quadros dos trechos de direção segura; a luminância é a da sessão inteira.

| Base / grupo | Vídeos | EAR aberto (média) | DP aberto | Ruído rápido | Amplitude | Amplitude ÷ nível | SNR | SNR rápida | Piscadas/min | Olho (px) | Lum. rosto | Contraste | Bits/pixel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **NITYMED (noite, carro em movimento)** | 126 | 0,247 | 0,0271 | 0,0084 | 0,0653 | 0,254 | 2,06 [1,83; 2,52] | 7,2 [6,4; 7,9] | 12,8 | 41,2 | 82,9 | 30,3 | 0,045 |
| Microssono | 19 | 0,255 | 0,0287 | 0,0084 | 0,0639 | 0,266 | 2,28 | 7,1 | 9,6 | 40,5 | 75,8 | 29,6 | 0,031 |
| Bocejo | 107 | 0,246 | 0,0266 | 0,0084 | 0,0657 | 0,249 | 2,01 | 7,2 | 13,4 | 41,5 | 83,5 | 30,3 | 0,046 |
| **UTA-RLDD (autogravado)** | 180 | 0,266 | 0,0182 | 0,0037 | 0,0796 | 0,295 | 4,28 [3,91; 4,96] | 21,5 [18,3; 23,5] | 10,2 | 50,9 | 99,9 | 40,6 | 0,276 |
| KSS 0 (alerta) | 60 | 0,268 | 0,0166 | 0,0036 | 0,0866 | 0,328 | 5,39 | 24,4 | 5,7 | 52,5 | 107,1 | 39,8 | 0,251 |
| KSS 5 | 60 | 0,265 | 0,0177 | 0,0038 | 0,0831 | 0,306 | 4,55 | 20,9 | 11,1 | 50,1 | 97,7 | 40,6 | 0,276 |
| KSS 10 (sonolento) | 60 | 0,261 | 0,0198 | 0,0040 | 0,0713 | 0,272 | 3,50 | 17,7 | 15,3 | 51,0 | 95,1 | 41,4 | 0,276 |
| **DMD s5 (dia, carro parado, sonolência encenada)** | 16 | 0,268 | 0,0254 | 0,0046 | 0,1025 | 0,397 | 3,87 [3,34; 4,40] | 23,1 [16,1; 27,9] | 29,0 | 42,0 | 103,7 | 42,2 | 0,565 |
| sem óculos | 13 | 0,268 | 0,0253 | 0,0045 | 0,1080 | 0,422 | 3,94 | 25,0 | 29,9 | 42,2 | 98,3 | 42,2 | 0,565 |
| com óculos | 3 | 0,247 | 0,0255 | 0,0054 | 0,0819 | 0,322 | 3,21 | 14,7 | 23,5 | 41,6 | 109,7 | 42,2 | 0,567 |
| **DMD, direção segura com o carro em movimento (dia)** | 15 | 0,270 | 0,0419 | 0,0097 | 0,0932 | 0,340 | 2,37 [1,38; 3,04] | 8,7 [6,2; 10,8] | 18,4 | 42,6 | 101,4 | 41,0 | 0,561 |
| **DMD, direção segura com o carro parado (dia)** | 18 | 0,276 | 0,0268 | 0,0059 | 0,0860 | 0,340 | 2,90 [2,46; 4,27] | 13,6 [12,0; 16,9] | 21,5 | 41,9 | 91,9 | 41,0 | 0,559 |

Diferenças de mediana NITYMED − referência diurna (bootstrap independente por agrupamento):

| Medida | − UTA-RLDD [IC 95%] | − DMD s5 (parado) [IC 95%] | − DMD em movimento [IC 95%] | − DMD parado [IC 95%] |
|---|---:|---:|---:|---:|
| EAR médio com olho aberto | -0,019 [-0,027; -0,010] | -0,021 [-0,029; 0,001] | -0,023 [-0,037; 0,001] | -0,029 [-0,039; 0,005] |
| DP do EAR com olho aberto | 0,0088 [0,0069; 0,0113] | 0,0017 [-0,0021; 0,0042] | -0,0149 [-0,0221; -0,0055] | 0,0003 [-0,0053; 0,0036] |
| Ruído rápido do EAR (DP robusto, resíduo de 1 s) | 0,0046 [0,0041; 0,0053] | 0,0038 [0,0030; 0,0045] | -0,0013 [-0,0024; -0,0004] | 0,0025 [0,0013; 0,0036] |
| Amplitude média das piscadas (estendida) | -0,0143 [-0,0267; -0,0043] | -0,0372 [-0,0550; -0,0185] | -0,0279 [-0,0426; 0,0050] | -0,0207 [-0,0424; -0,0086] |
| Amplitude ÷ nível aberto | -0,041 [-0,085; -0,016] | -0,143 [-0,206; -0,072] | -0,086 [-0,154; 0,011] | -0,086 [-0,142; -0,032] |
| SNR = amplitude ÷ DP aberto | -2,21 [-2,85; -1,54] | -1,81 [-2,47; -1,05] | -0,31 [-1,10; 0,97] | -0,84 [-2,17; -0,11] |
| SNR rápida = amplitude ÷ ruído rápido | -14,28 [-16,38; -11,17] | -15,94 [-20,72; -9,00] | -1,55 [-3,68; 1,14] | -6,39 [-9,83; -4,47] |
| Piscadas por minuto | 2,6 [-0,1; 5,3] | -16,2 [-21,1; -9,9] | -5,6 [-25,7; 8,2] | -8,7 [-16,3; -2,8] |
| Profundidade relativa p90 dos vales | -0,068 [-0,101; -0,040] | -0,188 [-0,261; -0,144] | -0,031 [-0,109; 0,045] | -0,181 [-0,235; -0,126] |
| Quedas ≥ 12% por minuto | 21,5 [16,8; 26,1] | -2,1 [-17,1; 8,0] | -8,0 [-14,6; 5,9] | 1,5 [-4,7; 6,5] |
| Largura do olho (px processados) | -9,7 [-13,2; -7,0] | -0,8 [-2,6; 0,8] | -1,4 [-4,9; 0,4] | -0,7 [-4,0; 0,2] |
| Luminância do rosto (0–255) | -17,0 [-25,2; -10,5] | -20,8 [-28,9; -10,8] | -18,5 [-26,0; -11,2] | -9,1 [-26,3; -2,5] |
| Luminância dos olhos (0–255) | -13,2 [-22,0; -7,4] | -18,3 [-29,0; -4,7] | -10,8 [-14,0; -5,2] | -4,3 [-19,4; 3,8] |
| Contraste RMS do rosto | -10,4 [-12,4; -7,5] | -12,0 [-20,5; -7,7] | -10,7 [-16,5; -8,5] | -10,8 [-15,5; -5,7] |
| Fração do rosto com cinza < 40 | -0,036 [-0,060; -0,009] | -0,029 [-0,072; -0,006] | -0,057 [-0,084; -0,011] | -0,054 [-0,084; -0,020] |
| Taxa de bits do arquivo (Mbit/s) | -6,81 [-7,10; -6,58] | -14,47 [-14,54; -14,42] | -14,34 [-14,40; -14,30] | -14,28 [-14,31; -14,24] |
| Bits por pixel (compressão) | -0,231 [-0,243; -0,192] | -0,520 [-0,523; -0,518] | -0,516 [-0,518; -0,514] | -0,513 [-0,515; -0,512] |

**Efeito do carro em movimento, de dia, no mesmo motorista** (DMD, trechos de direção segura das sessões s1 e s2: mesma câmera, mesmo carro e, em geral, o mesmo dia, com minutos de diferença). Mediana das diferenças por sujeito (média das sessões de cada condição) [IC 95%, bootstrap de sujeitos]:

| Medida | setup (Car Moving − Car Stopped), sessões s1/s2 | sessão do protocolo (s1 − s2) |
|---|---:|---:|
| Ruído rápido do EAR (DP robusto, resíduo de 1 s) | 0,0043 [0,0030; 0,0054] (n = 11; 10 ↑, 1 ↓) | 0,0038 [0,0013; 0,0048] (n = 14; 12 ↑, 2 ↓) |
| SNR rápida = amplitude ÷ ruído rápido | -5,39 [-6,62; -3,48] (n = 11; 0 ↑, 11 ↓) | -5,33 [-6,62; -3,70] (n = 14; 1 ↑, 13 ↓) |
| DP do EAR com olho aberto | 0,0097 [0,0031; 0,0181] (n = 11; 10 ↑, 1 ↓) | 0,0061 [0,0010; 0,0133] (n = 14; 11 ↑, 3 ↓) |
| Amplitude média das piscadas (estendida) | -0,0069 [-0,0166; 0,0184] (n = 11; 4 ↑, 7 ↓) | -0,0090 [-0,0166; 0,0044] (n = 14; 5 ↑, 9 ↓) |
| Taxa de detecção de rosto | -0,0011 [-0,0028; 0,0000] (n = 11; 2 ↑, 8 ↓) | -0,0008 [-0,0014; 0,0007] (n = 14; 4 ↑, 9 ↓) |
| Luminância do rosto (0–255) | 2,9 [-14,6; 22,9] (n = 11; 7 ↑, 4 ↓) | 0,6 [-14,6; 8,0] (n = 14; 7 ↑, 7 ↓) |

Correlações de Spearman entre vídeos da mesma base (a iluminação explica a captura e o sinal?):

| Base | x | y | ρ | p | vídeos |
|---|---|---|---:|---:|---:|
| NITYMED | `lum_rosto` | `taxa_rosto` | 0,17 | 0.058 | 126 |
| NITYMED | `lum_rosto` | `snr_rapido` | 0,14 | 0.13 | 121 |
| NITYMED | `olho_largura_px` | `snr_rapido` | 0,05 | 0.55 | 121 |
| NITYMED | `contraste_rosto` | `snr_rapido` | 0,16 | 0.076 | 121 |
| NITYMED | `lum_olhos` | `ruido_rapido` | -0,23 | 0.01 | 126 |
| NITYMED | `olho_largura_px` | `taxa_rosto` | 0,22 | 0.014 | 126 |
| NITYMED | `bits_por_pixel` | `ruido_rapido` | 0,19 | 0.029 | 126 |
| UTA-RLDD | `lum_rosto` | `taxa_rosto` | 0,05 | 0.53 | 180 |
| UTA-RLDD | `lum_rosto` | `snr_rapido` | 0,20 | 0.0074 | 178 |
| UTA-RLDD | `olho_largura_px` | `snr_rapido` | 0,08 | 0.32 | 178 |
| UTA-RLDD | `contraste_rosto` | `snr_rapido` | 0,20 | 0.0086 | 178 |
| UTA-RLDD | `lum_olhos` | `ruido_rapido` | -0,26 | 0.0005 | 180 |
| UTA-RLDD | `olho_largura_px` | `taxa_rosto` | 0,03 | 0.71 | 180 |
| UTA-RLDD | `bits_por_pixel` | `ruido_rapido` | 0,27 | 0.0003 | 180 |

Os p-valores tratam vídeos como independentes (no NITYMED há 21 motoristas em 126 vídeos): leia como descrição.

## 5. (c) Taxa de disparo dos detectores nos vídeos noturnos

Fração das janelas de 10 s (passo 5 s) marcadas "sonolento" [IC 95% por agrupamento], só janelas avaliáveis. ᴸ = fora da amostra no próprio domínio (UTA: 5 folds oficiais; DMD: deixa-um-sujeito-de-fora). Nos vídeos noturnos, uma janela de microssono não contém necessariamente um microssono (o vídeo também tem fala e olhares ao redor): a fração de janelas é uma cota inferior da sensibilidade, e a fração de vídeos com alguma janela marcada é o análogo por vídeo. **Nenhum número desta seção é acurácia no NITYMED.**

### 5.1 Brutas, 11 características (principal no Exp. 6)

| Grupo | Janelas (agrup.) | DMD: janelas | DMD: vídeos (média ≥ 0,5) | DMD: vídeos (alguma) | UTA B3: janelas | UTA B3: vídeos (média ≥ 0,5) | UTA B3: vídeos (alguma) |
|---|---:|---:|---:|---:|---:|---:|---:|
| NITYMED Microssono (noite, carro em movimento) | 273 (18) | 44,7% [32,7%; 57,4%] | 33% [11%; 56%] | 89% [72%; 100%] | 83,5% [73,8%; 91,1%] | 89% [72%; 100%] | 100% [100%; 100%] |
| NITYMED Bocejo (noite, carro em movimento) | 256 (96) | 73,0% [65,4%; 80,6%] | 80% [72%; 89%] | 85% [78%; 93%] | 89,5% [85,2%; 93,8%] | 93% [88%; 98%] | 95% [90%; 99%] |
| UTA-RLDD alerta, KSS 0 (autogravado) | 4.906 (60) | 26,0% [21,1%; 31,1%] | 8% [2%; 15%] | 95% [88%; 100%] | 43,6% [38,7%; 48,8%] ᴸ | 30% [18%; 42%] | 100% [100%; 100%] |
| UTA-RLDD pouca vigilância, KSS 5 (autogravado) | 7.468 (60) | 40,5% [34,7%; 46,3%] | 27% [15%; 38%] | 98% [95%; 100%] | 58,3% [53,0%; 64,0%] ᴸ | 60% [48%; 72%] | 100% [100%; 100%] |
| UTA-RLDD sonolento, KSS 10 (autogravado) | 7.581 (60) | 52,2% [46,2%; 58,1%] | 48% [37%; 60%] | 100% [100%; 100%] | 69,4% [63,8%; 74,6%] ᴸ | 82% [72%; 90%] | 100% [100%; 100%] |
| DMD alerta, direção segura com o carro em movimento (dia) | 105 (12) | 42,9% [25,0%; 69,2%] ᴸ | 33% [11%; 67%] | 80% [64%; 100%] | 77,1% [63,2%; 88,2%] | 87% [69%; 100%] | 100% [100%; 100%] |
| DMD alerta, direção segura com o carro parado (dia) | 94 (13) | 45,7% [28,6%; 66,1%] ᴸ | 56% [31%; 75%] | 78% [59%; 95%] | 69,1% [53,7%; 82,4%] | 72% [47%; 91%] | 100% [100%; 100%] |
| DMD sonolento atuado s5 (dia, carro parado) | 423 (13) | 82,0% [64,5%; 93,2%] ᴸ | 88% [67%; 100%] | 94% [79%; 100%] | 92,9% [89,4%; 96,3%] | 100% [100%; 100%] | 100% [100%; 100%] |

### 5.2 Centradas `_c`, 11 características (principal do Exp. 5b)

| Grupo | Janelas (agrup.) | DMD: janelas | DMD: vídeos (média ≥ 0,5) | DMD: vídeos (alguma) | UTA B3: janelas | UTA B3: vídeos (média ≥ 0,5) | UTA B3: vídeos (alguma) |
|---|---:|---:|---:|---:|---:|---:|---:|
| NITYMED Microssono (noite, carro em movimento) | 273 (18) | 27,8% [17,0%; 40,3%] | 22% [6%; 44%] | 78% [61%; 94%] | 45,1% [34,6%; 56,0%] | 50% [28%; 72%] | 89% [72%; 100%] |
| NITYMED Bocejo (noite, carro em movimento) | 256 (96) | 69,1% [59,7%; 78,1%] | 78% [69%; 86%] | 85% [78%; 92%] | 63,3% [56,1%; 70,2%] | 70% [60%; 79%] | 77% [69%; 85%] |
| UTA-RLDD alerta, KSS 0 (autogravado) | 4.906 (60) | 18,0% [13,4%; 23,0%] | 8% [2%; 15%] | 85% [75%; 93%] | 24,1% [19,3%; 29,2%] ᴸ | 10% [3%; 18%] | 98% [95%; 100%] |
| UTA-RLDD pouca vigilância, KSS 5 (autogravado) | 7.468 (60) | 31,9% [25,9%; 38,2%] | 25% [13%; 37%] | 93% [87%; 98%] | 56,2% [49,4%; 62,9%] ᴸ | 57% [45%; 68%] | 100% [100%; 100%] |
| UTA-RLDD sonolento, KSS 10 (autogravado) | 7.581 (60) | 48,4% [41,0%; 55,2%] | 55% [43%; 67%] | 98% [95%; 100%] | 72,5% [66,5%; 78,7%] ᴸ | 82% [72%; 90%] | 100% [100%; 100%] |
| DMD alerta, direção segura com o carro em movimento (dia) | 105 (12) | 36,2% [19,5%; 58,9%] ᴸ | 20% [0%; 46%] | 80% [54%; 95%] | 77,1% [65,2%; 92,1%] | 80% [58%; 100%] | 100% [100%; 100%] |
| DMD alerta, direção segura com o carro parado (dia) | 94 (13) | 34,0% [19,8%; 50,0%] ᴸ | 39% [18%; 59%] | 67% [50%; 83%] | 66,0% [47,3%; 83,2%] | 72% [42%; 94%] | 89% [72%; 100%] |
| DMD sonolento atuado s5 (dia, carro parado) | 423 (13) | 83,9% [70,0%; 93,2%] ᴸ | 94% [79%; 100%] | 100% [100%; 100%] | 87,7% [75,4%; 97,0%] | 94% [79%; 100%] | 100% [100%; 100%] |

### 5.3 Brutas sem `taxa_rosto`, 10 características (sensibilidade)

| Grupo | Janelas (agrup.) | DMD: janelas | DMD: vídeos (média ≥ 0,5) | DMD: vídeos (alguma) | UTA B3: janelas | UTA B3: vídeos (média ≥ 0,5) | UTA B3: vídeos (alguma) |
|---|---:|---:|---:|---:|---:|---:|---:|
| NITYMED Microssono (noite, carro em movimento) | 273 (18) | 45,1% [33,5%; 56,6%] | 33% [11%; 56%] | 89% [72%; 100%] | 83,2% [73,8%; 90,8%] | 89% [72%; 100%] | 100% [100%; 100%] |
| NITYMED Bocejo (noite, carro em movimento) | 256 (96) | 72,7% [65,5%; 80,2%] | 81% [73%; 89%] | 85% [78%; 93%] | 89,5% [85,0%; 93,5%] | 94% [89%; 98%] | 95% [90%; 99%] |
| UTA-RLDD alerta, KSS 0 (autogravado) | 4.906 (60) | 28,8% [23,8%; 34,0%] | 12% [5%; 20%] | 95% [88%; 100%] | 44,5% [39,4%; 49,8%] ᴸ | 28% [17%; 42%] | 100% [100%; 100%] |
| UTA-RLDD pouca vigilância, KSS 5 (autogravado) | 7.468 (60) | 43,1% [37,6%; 48,9%] | 28% [17%; 40%] | 98% [95%; 100%] | 58,5% [53,1%; 64,2%] ᴸ | 58% [47%; 72%] | 100% [100%; 100%] |
| UTA-RLDD sonolento, KSS 10 (autogravado) | 7.581 (60) | 53,9% [47,8%; 59,8%] | 50% [37%; 62%] | 100% [100%; 100%] | 69,5% [63,8%; 74,7%] ᴸ | 80% [70%; 90%] | 100% [100%; 100%] |
| DMD alerta, direção segura com o carro em movimento (dia) | 105 (12) | 42,9% [25,0%; 69,2%] ᴸ | 33% [11%; 67%] | 80% [64%; 100%] | 75,2% [62,3%; 86,7%] | 93% [79%; 100%] | 100% [100%; 100%] |
| DMD alerta, direção segura com o carro parado (dia) | 94 (13) | 43,6% [26,5%; 64,0%] ᴸ | 50% [24%; 71%] | 78% [59%; 95%] | 68,1% [56,3%; 79,8%] | 78% [54%; 95%] | 100% [100%; 100%] |
| DMD sonolento atuado s5 (dia, carro parado) | 423 (13) | 83,2% [68,5%; 93,1%] ᴸ | 88% [67%; 100%] | 100% [100%; 100%] | 93,4% [90,0%; 96,7%] | 100% [100%; 100%] | 100% [100%; 100%] |

"vídeos (média ≥ 0,5)" = regra por vídeo da Entrega 01 (média dos escores das janelas); "vídeos (alguma)" = vídeo com pelo menos uma janela marcada. No DMD, "vídeo" = sessão.

### 5.4 Outros limiares (secundários)

Limiar calibrado só com o treino (a regra do Exp. 5b: máxima acurácia balanceada nos escores fora da amostra das janelas de treino — DMD: LOSO; UTA: fora do fold) e limiar com 10% de falso alarme nas janelas alerta da UTA (KSS 0), os dois aplicados sem mudança à noite.

| Modelo | Variante | Limiar calibrado (origem) | UTA KSS 0 | DMD parado | DMD em mov. | UTA KSS 10 | NITYMED bocejo | NITYMED microssono |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Modelo do DMD | sem_normalizacao | 0,8417 (Exp. 5b; aqui 0,8417) | 1,7% | 7,4% | 21,9% | 11,0% | 24,6% | 7,0% |
| Modelo do DMD | principal | 0,6617 (Exp. 5b; aqui 0,6617) | 11,3% | 20,2% | 23,8% | 37,0% | 61,3% | 15,0% |
| Modelo do DMD | sem_taxa_rosto | 0,8450 (calculado aqui) | 1,8% | 9,6% | 22,9% | 11,2% | 27,7% | 9,2% |
| Modelo da UTA-RLDD (B3) | sem_normalizacao | 0,7150 (calculado aqui) | 17,4% | 36,2% | 38,1% | 46,8% | 59,4% | 48,0% |
| Modelo da UTA-RLDD (B3) | principal | 0,5683 (calculado aqui) | 18,5% | 57,4% | 65,7% | 67,8% | 52,7% | 34,8% |
| Modelo da UTA-RLDD (B3) | sem_taxa_rosto | 0,6783 (calculado aqui) | 21,6% | 41,5% | 47,6% | 50,7% | 62,9% | 53,1% |

| Modelo | Variante | Limiar (FAR 10% na UTA KSS 0) | FAR alcançado | UTA KSS 10 | NITYMED bocejo | NITYMED microssono | Obs. |
|---|---|---:|---:|---:|---:|---:|---|
| Modelo do DMD | sem_normalizacao | 0,613 | 9,7% | 34,3% | 66,0% | 27,5% |  |
| Modelo do DMD | principal | 0,690 | 9,9% | 34,6% | 58,6% | 12,1% |  |
| Modelo do DMD | sem_taxa_rosto | 0,633 | 9,9% | 33,6% | 62,9% | 26,0% |  |
| Modelo da UTA-RLDD (B3) | sem_normalizacao | 0,793 | 10,0% | 38,0% | 45,3% | 34,4% |  |
| Modelo da UTA-RLDD (B3) | principal | 0,727 | 10,0% | 53,6% | 31,6% | 13,9% |  |
| Modelo da UTA-RLDD (B3) | sem_taxa_rosto | 0,800 | 9,8% | 37,2% | 44,1% | 33,3% |  |

No RF da UTA as probabilidades de dia são fora do fold e as da noite são do modelo inteiro: a transferência de qualquer limiar é aproximada.

### 5.5 Regra do rótulo do DMD aplicada aos nossos sinais (sem treino) e disparo sem evento

| Grupo | Fechamento ≥ 0,5 s | Bocejo (MAR > 0,5 por ≥ 1 s) | Qualquer um (regra DMD) |
|---|---:|---:|---:|
| NITYMED Microssono (noite, carro em movimento) | 15,0% [5,9%; 26,7%] | 0,0% [0,0%; 0,0%] | 15,0% [5,9%; 26,7%] |
| NITYMED Bocejo (noite, carro em movimento) | 34,4% [26,1%; 43,4%] | 48,0% [36,8%; 59,7%] | 61,7% [51,3%; 72,2%] |
| UTA-RLDD alerta, KSS 0 (autogravado) | 2,6% [1,4%; 4,3%] | 0,0% [0,0%; 0,0%] | 2,6% [1,4%; 4,3%] |
| UTA-RLDD pouca vigilância, KSS 5 (autogravado) | 9,1% [5,9%; 13,3%] | 0,6% [0,1%; 1,6%] | 9,7% [6,3%; 13,9%] |
| UTA-RLDD sonolento, KSS 10 (autogravado) | 23,8% [17,8%; 30,1%] | 0,5% [0,2%; 1,0%] | 24,1% [18,0%; 30,3%] |
| DMD alerta, direção segura com o carro em movimento (dia) | 23,8% [6,8%; 50,0%] | 0,0% [0,0%; 0,0%] | 23,8% [6,8%; 50,0%] |
| DMD alerta, direção segura com o carro parado (dia) | 14,9% [5,7%; 25,5%] | 0,0% [0,0%; 0,0%] | 14,9% [5,7%; 25,5%] |
| DMD sonolento atuado s5 (dia, carro parado) | 70,2% [52,1%; 82,8%] | 13,9% [9,5%; 18,3%] | 75,7% [61,9%; 85,7%] |

Disparo dos modelos (variante `sem_normalizacao`) só nas janelas **sem evento detectado** (nenhum fechamento ≥ 0,5 s e nenhum bocejo nos nossos sinais) — se o modelo marca essas janelas, reage a outra coisa que não fechamento/bocejo (ruído do EAR, piscadas):

| Grupo | Janelas sem evento (fração do grupo) | Modelo do DMD | Modelo da UTA B3 |
|---|---:|---:|---:|
| NITYMED Microssono (noite, carro em movimento) | 232 (85,0%) | 35,8% [25,7%; 46,3%] | 80,6% [70,6%; 89,2%] |
| NITYMED Bocejo (noite, carro em movimento) | 98 (38,3%) | 35,7% [22,0%; 48,5%] | 82,7% [72,5%; 90,9%] |
| UTA-RLDD alerta, KSS 0 (autogravado) | 4.778 (97,4%) | 24,1% [19,2%; 29,3%] | 42,2% [37,2%; 47,6%] |
| UTA-RLDD pouca vigilância, KSS 5 (autogravado) | 6.745 (90,3%) | 35,6% [29,6%; 42,0%] | 54,0% [48,8%; 59,6%] |
| UTA-RLDD sonolento, KSS 10 (autogravado) | 5.754 (75,9%) | 41,0% [35,0%; 46,9%] | 59,9% [53,7%; 65,7%] |
| DMD alerta, direção segura com o carro em movimento (dia) | 80 (76,2%) | 25,0% [16,7%; 40,5%] | 70,0% [56,0%; 81,4%] |
| DMD alerta, direção segura com o carro parado (dia) | 80 (85,1%) | 37,5% [22,1%; 57,8%] | 63,7% [48,4%; 77,8%] |
| DMD sonolento atuado s5 (dia, carro parado) | 103 (24,3%) | 37,9% [18,2%; 65,1%] | 82,5% [77,5%; 89,9%] |

### 5.6 Sensibilidades no NITYMED (variante `sem_normalizacao`)

(i) Todas as janelas (inclui as do primeiro terço, que é a referência, e as de fronteira); (ii) vídeos da pasta Bocejo suspeitos de pasta errada (> 90 s, nenhum bocejo detectado no vídeo inteiro; ver `results/exp6_nitymed_ficha.md`): `Yawning_Female_P1042777_720`, `Yawning_Female_P1043088_720`, `Yawning_Female_P1043107_720`. Os números principais mantêm as janelas avaliáveis e os rótulos de pasta.

| Grupo | Conjunto | Vídeos | Janelas | Modelo do DMD | Modelo da UTA B3 | Regra DMD |
|---|---|---:|---:|---:|---:|---:|
| Microssono | avaliáveis (principal) | 18 | 273 | 44,7% [32,7%; 57,4%] | 83,5% [73,8%; 91,1%] | 15,0% [5,9%; 26,7%] |
| Microssono | todas as janelas | 19 | 438 | 43,6% [31,9%; 56,0%] | 84,5% [78,0%; 90,5%] | 15,8% [6,6%; 25,6%] |
| Bocejo | avaliáveis (principal) | 96 | 256 | 73,0% [65,4%; 80,6%] | 89,5% [85,2%; 93,8%] | 61,7% [51,3%; 72,2%] |
| Bocejo | todas as janelas | 107 | 541 | 71,9% [66,1%; 77,0%] | 87,4% [83,9%; 90,9%] | 57,7% [49,6%; 65,2%] |
| Bocejo | avaliáveis, sem os suspeitos | 93 | 211 | 74,4% [67,3%; 81,7%] | 88,6% [83,5%; 93,5%] | 68,2% [60,4%; 76,6%] |
| Bocejo | avaliáveis, só os suspeitos | 3 | 45 | 66,7% [43,8%; 93,3%] | 93,3% [86,7%; 100,0%] | 31,1% [0,0%; 60,0%] |
| Microssono | avaliáveis, com os suspeitos | 21 | 318 | 47,8% [36,3%; 59,4%] | 84,9% [76,4%; 91,7%] | 17,3% [8,1%; 27,8%] |

### 5.7 DMD por sessão do protocolo (sensibilidade do agrupamento por `setup`)

| Grupo | Janelas (sujeitos) | Modelo do DMD | Modelo da UTA B3 | Regra DMD |
|---|---:|---:|---:|---:|
| DMD alerta da sessão s2 (protocolo: carro parado; classe alerta de treino) | 87 (13) | 39,1% [20,2%; 59,6%] ᴸ | 65,5% [52,4%; 77,1%] | 13,8% [4,0%; 25,9%] |
| DMD alerta da sessão s1 (protocolo: carro em movimento) | 109 (14) | 46,8% [29,8%; 69,0%] ᴸ | 78,9% [65,8%; 88,1%] | 23,9% [6,9%; 46,2%] |

### 5.8 Mudança de distribuição: medianas das características por grupo (brutas, janelas avaliadas)

| Grupo | Janelas | `perclos` | `taxa_piscadas_min` | `dur_media_piscada_s` | `dur_max_fechamento_s` | `n_fechamentos_05s` | `ear_media` | `ear_dp` | `amp_media` | `vel_media` | `frac_bocejo` | `taxa_rosto` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NITYMED Microssono (noite, carro em movimento) | 273 | 0,024 | 6,000 | 0,147 | 0,120 | 0,000 | 0,250 | 0,031 | 0,072 | 0,222 | 0,000 | 1,000 |
| NITYMED Bocejo (noite, carro em movimento) | 256 | 0,068 | 12,960 | 0,181 | 0,320 | 0,000 | 0,238 | 0,033 | 0,069 | 0,222 | 0,000 | 1,000 |
| UTA-RLDD alerta, KSS 0 (autogravado) | 4.906 | 0,010 | 5,994 | 0,122 | 0,067 | 0,000 | 0,264 | 0,015 | 0,095 | 0,285 | 0,000 | 1,000 |
| UTA-RLDD pouca vigilância, KSS 5 (autogravado) | 7.468 | 0,020 | 6,014 | 0,145 | 0,133 | 0,000 | 0,262 | 0,018 | 0,087 | 0,244 | 0,000 | 1,000 |
| UTA-RLDD sonolento, KSS 10 (autogravado) | 7.581 | 0,040 | 11,996 | 0,184 | 0,168 | 0,000 | 0,254 | 0,020 | 0,078 | 0,217 | 0,000 | 1,000 |
| DMD alerta, direção segura com o carro em movimento (dia) | 105 | 0,027 | 12,024 | 0,115 | 0,101 | 0,000 | 0,259 | 0,038 | 0,091 | 0,383 | 0,000 | 1,000 |
| DMD alerta, direção segura com o carro parado (dia) | 94 | 0,044 | 17,976 | 0,126 | 0,134 | 0,000 | 0,259 | 0,030 | 0,104 | 0,350 | 0,000 | 1,000 |
| DMD sonolento atuado s5 (dia, carro parado) | 423 | 0,208 | 29,960 | 0,235 | 0,974 | 1,000 | 0,237 | 0,043 | 0,109 | 0,325 | 0,000 | 1,000 |

Importância por impureza (redução média de Gini) dos modelos inteiros, variante `sem_normalizacao` — o que cada modelo usa:

| Característica | Modelo do DMD | Modelo da UTA B3 |
|---|---:|---:|
| `perclos` | 0,137 | 0,094 |
| `taxa_piscadas_min` | 0,054 | 0,055 |
| `dur_media_piscada_s` | 0,075 | 0,084 |
| `dur_max_fechamento_s` | 0,190 | 0,131 |
| `n_fechamentos_05s` | 0,090 | 0,024 |
| `ear_media` | 0,115 | 0,227 |
| `ear_dp` | 0,119 | 0,214 |
| `amp_media` | 0,088 | 0,076 |
| `vel_media` | 0,081 | 0,083 |
| `frac_bocejo` | 0,040 | 0,000 |
| `taxa_rosto` | 0,010 | 0,012 |

### 5.9 Contrafactual: o que move o disparo noturno

Disparo nas janelas noturnas avaliáveis (modelos inteiros, variante principal do Exp. 6 = brutas, `sem_normalizacao`) quando UMA característica recebe o valor típico do dia; última linha: as três de qualidade do sinal de uma vez. Diagnóstico do modelo, não medida de desempenho. Duas referências de "dia": mediana das janelas alerta KSS 0 da UTA-RLDD e mediana das janelas alerta (s2) do DMD — a classe alerta de treino do modelo do DMD.

**Referência: mediana das janelas alerta KSS 0 da UTA-RLDD**

| Característica trocada | Valor do dia | Modelo do DMD — Microssono (original 44,7%) | Modelo do DMD — Bocejo (original 73,0%) | Modelo da UTA-RLDD (B3) — Microssono (original 83,5%) | Modelo da UTA-RLDD (B3) — Bocejo (original 89,5%) |
|---|---:|---:|---:|---:|---:|
| `perclos` | 0,010 | 41,8% (−2,9 p.p.) | 73,0% (+0,0 p.p.) | 83,5% (+0,0 p.p.) | 91,0% (+1,6 p.p.) |
| `taxa_piscadas_min` | 5,994 | 44,7% (+0,0 p.p.) | 75,0% (+2,0 p.p.) | 82,4% (−1,1 p.p.) | 90,6% (+1,2 p.p.) |
| `dur_media_piscada_s` | 0,122 | 42,1% (−2,6 p.p.) | 77,7% (+4,7 p.p.) | 87,2% (+3,7 p.p.) | 92,6% (+3,1 p.p.) |
| `dur_max_fechamento_s` | 0,067 | 40,3% (−4,4 p.p.) | 73,4% (+0,4 p.p.) | 80,6% (−2,9 p.p.) | 87,9% (−1,6 p.p.) |
| `n_fechamentos_05s` | 0,000 | 44,3% (−0,4 p.p.) | 74,6% (+1,6 p.p.) | 83,5% (+0,0 p.p.) | 89,5% (+0,0 p.p.) |
| `ear_media` | 0,264 | 72,9% (+28,2 p.p.) | 88,7% (+15,6 p.p.) | 93,0% (+9,5 p.p.) | 91,4% (+2,0 p.p.) |
| `ear_dp` | 0,015 | 37,0% (−7,7 p.p.) | 71,9% (−1,2 p.p.) | 64,5% (−19,0 p.p.) | 81,6% (−7,8 p.p.) |
| `amp_media` | 0,095 | 50,9% (+6,2 p.p.) | 77,3% (+4,3 p.p.) | 81,7% (−1,8 p.p.) | 85,5% (−3,9 p.p.) |
| `vel_media` | 0,285 | 52,0% (+7,3 p.p.) | 78,9% (+5,9 p.p.) | 82,8% (−0,7 p.p.) | 87,5% (−2,0 p.p.) |
| `frac_bocejo` | 0,000 | 44,7% (+0,0 p.p.) | 59,4% (−13,7 p.p.) | 83,5% (+0,0 p.p.) | 89,1% (−0,4 p.p.) |
| `taxa_rosto` | 1,000 | 44,0% (−0,7 p.p.) | 72,3% (−0,8 p.p.) | 82,4% (−1,1 p.p.) | 89,1% (−0,4 p.p.) |
| `ear_dp+amp_media+vel_media` | — | 42,9% (−1,8 p.p.) | 77,7% (+4,7 p.p.) | 55,3% (−28,2 p.p.) | 73,8% (−15,6 p.p.) |

**Referência: mediana das janelas alerta da s2 do DMD**

| Característica trocada | Valor do dia | Modelo do DMD — Microssono (original 44,7%) | Modelo do DMD — Bocejo (original 73,0%) | Modelo da UTA-RLDD (B3) — Microssono (original 83,5%) | Modelo da UTA-RLDD (B3) — Bocejo (original 89,5%) |
|---|---:|---:|---:|---:|---:|
| `perclos` | 0,037 | 43,6% (−1,1 p.p.) | 73,8% (+0,8 p.p.) | 85,0% (+1,5 p.p.) | 90,6% (+1,2 p.p.) |
| `taxa_piscadas_min` | 17,976 | 44,0% (−0,7 p.p.) | 74,2% (+1,2 p.p.) | 81,0% (−2,6 p.p.) | 89,1% (−0,4 p.p.) |
| `dur_media_piscada_s` | 0,118 | 43,2% (−1,5 p.p.) | 77,7% (+4,7 p.p.) | 86,4% (+2,9 p.p.) | 93,0% (+3,5 p.p.) |
| `dur_max_fechamento_s` | 0,134 | 45,8% (+1,1 p.p.) | 76,6% (+3,5 p.p.) | 85,0% (+1,5 p.p.) | 86,3% (−3,1 p.p.) |
| `n_fechamentos_05s` | 0,000 | 44,3% (−0,4 p.p.) | 74,6% (+1,6 p.p.) | 83,5% (+0,0 p.p.) | 89,5% (+0,0 p.p.) |
| `ear_media` | 0,259 | 66,3% (+21,6 p.p.) | 84,4% (+11,3 p.p.) | 87,5% (+4,0 p.p.) | 89,5% (+0,0 p.p.) |
| `ear_dp` | 0,030 | 41,8% (−2,9 p.p.) | 71,9% (−1,2 p.p.) | 90,8% (+7,3 p.p.) | 96,1% (+6,6 p.p.) |
| `amp_media` | 0,104 | 47,6% (+2,9 p.p.) | 76,6% (+3,5 p.p.) | 82,8% (−0,7 p.p.) | 85,5% (−3,9 p.p.) |
| `vel_media` | 0,344 | 34,8% (−9,9 p.p.) | 74,2% (+1,2 p.p.) | 78,0% (−5,5 p.p.) | 84,4% (−5,1 p.p.) |
| `frac_bocejo` | 0,000 | 44,7% (+0,0 p.p.) | 59,4% (−13,7 p.p.) | 83,5% (+0,0 p.p.) | 89,1% (−0,4 p.p.) |
| `taxa_rosto` | 1,000 | 44,0% (−0,7 p.p.) | 72,3% (−0,8 p.p.) | 82,4% (−1,1 p.p.) | 89,1% (−0,4 p.p.) |
| `ear_dp+amp_media+vel_media` | — | 30,0% (−14,7 p.p.) | 71,5% (−1,6 p.p.) | 81,0% (−2,6 p.p.) | 83,6% (−5,9 p.p.) |

### 5.10 Conferência com as predições do Exp. 5b final

Mesma receita e mesmos dados ⇒ mesmas probabilidades. Comparação janela a janela com `results/exp5b_predicoes.csv` (RF; o CSV do 5b grava o escore com 6 casas decimais, então diferenças da ordem de 1e-07 são arredondamento):

| Comparação | Variante | Janelas | Sem par | Diferença máxima | Diferença média |
|---|---|---:|---:|---:|---:|
| modelo do DMD inteiro aplicado à UTA | sem_normalizacao | 19.955 | 0 | 3.33e-07 | 2.24e-07 |
| modelo do DMD inteiro aplicado à UTA | principal | 19.955 | 0 | 3.33e-07 | 2.24e-07 |
| modelo do DMD, fora da amostra (LOSO) | sem_normalizacao | 510 | 0 | 3.33e-07 | 2.33e-07 |
| modelo do DMD, fora da amostra (LOSO) | principal | 510 | 0 | 3.33e-07 | 2.12e-07 |
| modelo da UTA B3, fora da amostra | sem_normalizacao | 12.487 | 0 | 3.33e-07 | 2.18e-07 |
| modelo da UTA B3, fora da amostra | principal | 12.487 | 0 | 3.33e-07 | 2.18e-07 |

### 5.11 Todos os modelos finais salvos pelo Exp. 5b (treinados no DMD) aplicados à noite e ao dia

Modelos inteiros em todas as colunas. Na coluna do DMD em movimento (sessão s1, que não entra no treino) os sujeitos estão no treino com outras sessões, então ela NÃO é fora da amostra; o número fora da amostra (deixa-um-sujeito-de-fora) está na seção 5.1.

| Arquivo | Modelo | Variante | UTA KSS 0 (falso alarme) | DMD alerta em movimento (sujeitos do treino) | UTA KSS 10 | NITYMED bocejo | NITYMED microssono |
|---|---|---|---:|---:|---:|---:|---:|
| `exp5b_final_C_principal_LR.joblib` | LR | principal | 5,6% [3,8%; 7,8%] | 21,0% [8,1%; 41,9%] | 33,7% [26,5%; 41,5%] | 49,2% [39,8%; 59,4%] | 13,6% [6,3%; 22,4%] |
| `exp5b_final_C_principal_RF.joblib` | RF | principal | 18,0% [13,4%; 23,0%] | 40,0% [23,4%; 65,7%] | 48,4% [41,0%; 55,2%] | 69,1% [59,7%; 78,1%] | 27,8% [17,0%; 40,3%] |
| `exp5b_final_C_sem_normalizacao_LR.joblib` | LR | sem_normalizacao | 2,3% [1,1%; 3,9%] | 25,7% [9,3%; 52,2%] | 18,5% [12,9%; 24,9%] | 59,4% [50,0%; 69,6%] | 14,7% [5,7%; 25,6%] |
| `exp5b_final_C_sem_normalizacao_RF.joblib` | RF | sem_normalizacao | 26,0% [21,1%; 31,1%] | 39,0% [22,2%; 64,0%] | 52,2% [46,2%; 58,1%] | 73,0% [65,4%; 80,6%] | 44,7% [32,7%; 57,4%] |

## 6. (d) PERCLOS e fechamentos longos nos vídeos de microssono

PERCLOS = fração dos quadros com rosto com EAR < limiar. "Pipeline" = limiar de `blinks.py` (0,75 × mediana da referência: primeiro terço do próprio vídeo no NITYMED, último terço do vídeo 0 na UTA, safe_drive da s3 no DMD). "p90" = 0,75 × percentil 90 do EAR do próprio vídeo (não depende de um trecho alerta). Medianas por vídeo [IC 95%]; DMD em movimento/parado: só os trechos de direção segura.

| Grupo | Vídeos | Minutos | PERCLOS (pipeline) | p90 entre vídeos | p90 entre janelas | Fech. ≥ 0,5 s/min | Vídeos c/ fech. ≥ 0,5 s | Vídeos c/ fech. > 1,5 s | Fech. > 1,5 s (total) | PERCLOS (p90) | Vídeos c/ fech. > 1,5 s (p90) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Microssono | 19 | 38,9 | 2,9% [2,1%; 6,3%] | 11,7% | 14,2% | 0,48 | 58% | 16% | 5 | 19,5% [12,5%; 22,9%] | 63% |
| Bocejo | 107 | 58,8 | 6,5% [4,9%; 8,6%] | 21,4% | 29,6% | 1,39 | 52% | 23% | 40 | 18,3% [14,5%; 21,5%] | 51% |
| KSS 0 (alerta) | 60 | 624,1 | 1,4% [0,9%; 1,9%] | 5,1% | 6,3% | 0,00 | 45% | 8% | 12 | 2,4% [2,1%; 3,4%] | 28% |
| KSS 5 | 60 | 629,7 | 3,4% [2,7%; 4,5%] | 10,2% | 12,3% | 0,24 | 68% | 32% | 128 | 5,5% [4,5%; 7,5%] | 50% |
| KSS 10 (sonolento) | 60 | 639,2 | 7,7% [5,0%; 8,9%] | 28,1% | 39,9% | 1,05 | 88% | 60% | 440 | 8,9% [7,4%; 13,1%] | 68% |
| DMD s5 (dia, carro parado, sonolência encenada) | 16 | 49,4 | 20,5% [13,3%; 22,7%] | 29,7% | 42,3% | 5,11 | 94% | 81% | 85 | 23,6% [16,5%; 27,0%] | 81% |
| DMD, direção segura com o carro em movimento (dia) | 15 | 34,0 | 5,4% [1,1%; 21,3%] | 28,1% | 36,3% | 0,49 | 60% | 13% | 12 | 18,4% [13,0%; 52,1%] | 60% |
| DMD, direção segura com o carro parado (dia) | 18 | 35,3 | 6,2% [4,5%; 11,2%] | 16,2% | 22,7% | 0,61 | 67% | 22% | 5 | 13,3% [11,2%; 21,0%] | 56% |

Os vídeos de microssono, um a um (ordenados pelo nº de fechamentos ≥ 0,5 s):

| Vídeo | Duração (s) | Rosto | Limiar | PERCLOS | Fech. ≥ 0,5 s | ≥ 1 s | > 1,5 s | Maior (s) | Limiar p90 | PERCLOS p90 | Fech. ≥ 0,5 s (p90) | > 1,5 s (p90) | Piscadas/min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Microsleep_Female_P1043106_720` | 124,0 | 1,000 | 0,207 | 19,2% | 15 | 5 | 3 | 1,64 | 0,222 | 27,6% | 22 | 3 | 28,1 |
| `Microsleep_Male_P1042797_720` | 125,8 | 1,000 | 0,204 | 10,7% | 12 | 5 | 0 | 1,40 | 0,243 | 27,2% | 16 | 8 | 12,4 |
| `Microsleep_Female_P1043089_720` | 124,8 | 0,988 | 0,180 | 15,4% | 9 | 4 | 1 | 3,72 | 0,212 | 39,9% | 23 | 10 | 23,4 |
| `Microsleep_Male_P1042762_720` | 123,8 | 0,999 | 0,162 | 8,7% | 5 | 2 | 1 | 1,68 | 0,190 | 20,5% | 11 | 7 | 13,1 |
| `Microsleep_Male_P1042786_720` | 131,0 | 0,981 | 0,191 | 7,7% | 4 | 0 | 0 | 0,88 | 0,208 | 19,5% | 12 | 1 | 16,8 |
| `Microsleep_Female_P1043067_720` | 131,4 | 1,000 | 0,178 | 3,7% | 2 | 0 | 0 | 0,80 | 0,206 | 20,2% | 15 | 6 | 7,8 |
| `Microsleep_Female_P1043068_720` | 146,6 | 0,995 | 0,225 | 2,2% | 2 | 1 | 0 | 1,20 | 0,246 | 7,3% | 6 | 0 | 3,3 |
| `Microsleep_Female_P1043115_720` | 122,2 | 0,998 | 0,219 | 6,3% | 1 | 0 | 0 | 0,56 | 0,238 | 14,9% | 10 | 1 | 23,1 |
| `Microsleep_Male_P1042793_720` | 125,8 | 0,997 | 0,206 | 2,9% | 1 | 0 | 0 | 0,88 | 0,249 | 28,0% | 21 | 4 | 9,6 |
| `Microsleep_Female_P1043129_720` | 123,8 | 0,998 | 0,207 | 2,2% | 1 | 0 | 0 | 0,52 | 0,225 | 5,1% | 1 | 0 | 6,8 |
| `Microsleep_Female_P1042787_720` | 131,5 | 0,994 | 0,239 | 1,6% | 1 | 0 | 0 | 0,52 | 0,258 | 6,9% | 1 | 0 | 6,0 |
| `Microsleep_Female_P1043114_720` | 14,0 | 1,000 | 0,224 | 5,1% | 0 | 0 | 0 | 0,16 | 0,274 | 22,9% | 1 | 0 | 25,7 |
| `Microsleep_Male_P1042751_720` | 128,6 | 0,999 | 0,157 | 3,9% | 0 | 0 | 0 | 0,48 | 0,196 | 21,8% | 16 | 1 | 12,1 |
| `Microsleep_Male_P1042757_720` | 146,0 | 0,999 | 0,255 | 2,6% | 0 | 0 | 0 | 0,32 | 0,280 | 7,6% | 6 | 0 | 10,3 |
| `Microsleep_Male_P1042772_720` | 124,2 | 0,940 | 0,175 | 2,1% | 0 | 0 | 0 | 0,32 | 0,208 | 17,5% | 14 | 1 | 5,1 |
| `Microsleep_Male_P1043075_720` | 131,4 | 0,987 | 0,168 | 1,9% | 0 | 0 | 0 | 0,16 | 0,188 | 12,5% | 3 | 0 | 7,9 |
| `Microsleep_Male_P1042756_720` | 124,0 | 0,982 | 0,164 | 1,7% | 0 | 0 | 0 | 0,40 | 0,213 | 29,0% | 18 | 3 | 3,5 |
| `Microsleep_Male_P1042767_720` | 129,2 | 0,998 | 0,186 | 1,6% | 0 | 0 | 0 | 0,48 | 0,220 | 18,5% | 14 | 3 | 5,6 |
| `Microsleep_Male_P1043122_720` | 125,0 | 1,000 | 0,165 | 1,0% | 0 | 0 | 0 | 0,36 | 0,185 | 9,1% | 3 | 0 | 1,9 |

Exemplos (figura `results/figures/fig_exp6_microssono.pdf`, só séries temporais de EAR — sem imagem): Mais fechamentos ≥ 0,5 s = `Microsleep_Female_P1043106_720`; Menos fechamentos e menor PERCLOS (≥ 60 s) = `Microsleep_Male_P1043122_720`. Regras de escolha: o vídeo de microssono com mais fechamentos ≥ 0,5 s; e, entre os de ≥ 60 s, o com menos fechamentos e menor PERCLOS.

Números dos exemplos — `Microsleep_Female_P1043106_720`: EAR p1 = 0,145, mediana com olho aberto = 0,266, limiar do pipeline = 0,207, limiar p90 = 0,222; `Microsleep_Male_P1043122_720`: EAR p1 = 0,165, mediana com olho aberto = 0,216, limiar do pipeline = 0,165, limiar p90 = 0,185. No DMD (s5), a mediana do EAR dentro dos intervalos anotados como olho fechado vai de 0,118 a 0,247 por sessão (`results/sinais_alinhamento_dmd.csv`): o EAR do Face Mesh com olho fechado não vai a 0.

Inspeção visual INTERNA feita na rodada preliminar nestes mesmos dois vídeos (recortes dos olhos vistos pelo analista em 27/09/2026, não publicados e não gravados no repositório; o limiar de EAR do NITYMED não mudou entre as rodadas): nos quadros de menor EAR de `Microsleep_Female_P1043106_720` os olhos estão fechados; no exemplo de menor PERCLOS, parte dos quadros sorteados ao acaso (EAR perto da mediana) já mostra pálpebras semicerradas — o trecho de referência (primeiro terço do vídeo) não é alerta, e o limiar relativo não enxerga o estreitamento sustentado.

**Calibração sem referência alerta.** No NITYMED o limiar de EAR (e a centragem `_c`) vem do primeiro terço do PRÓPRIO vídeo, que já pode estar sonolento: o limiar encolhe junto com o olho e os fechamentos sustentados deixam de cruzá-lo. A sensibilidade "p90" (nível de olho aberto tirado dos momentos mais abertos do próprio vídeo) mostra o tamanho do efeito (PERCLOS mediano dos microssonos 2,9% → 19,5%; nos vídeos alerta da UTA, a mesma troca vai de 1,4% a 2,4%, então o salto não é artefato do p90). Para o app, isso reforça a calibração em estado alerta no início do uso (a UTA-RLDD tem vídeo alerta; o NITYMED não).

## 7. (e) Ficha do dataset

Ver `results/exp6_nitymed_ficha.md` e `data/nitymed/index.csv` (126 linhas; SHA-256, codec, fps, resolução, duração, taxa de bits, taxa de rosto, bocejos detectados e janelas de 10 s por vídeo). Chaves no `paper/refs.bib`: `petrellis2021dsd` (presente), `petrellis2021vlsisoc` (presente), `petrellis2022kaggle` (presente), `petrellis2022` (presente). Citar o conjunto de dados e ao menos um dos artigos exigidos pelos autores (notas, item 1).

## 8. Leitura

1. **Captura**: a taxa de detecção de rosto à noite é menor que nas referências diurnas, e a diferença é pequena (NITYMED − UTA-RLDD -0,0061 [-0,0107; -0,0014]; − DMD s5 -0,0063 [-0,0108; -0,0025]; − DMD em movimento -0,0050 [-0,0094; -0,0005]). O que muda à noite é a cauda: 8 dos 126 vídeos ficam abaixo de 0,95 e 3,1% das janelas abaixo de 0,90 (UTA-RLDD: 1 vídeo(s), 0,7%; DMD s5: 0, 0,4%; DMD em movimento: 0, 1,0%). O pior vídeo de todas as bases, porém, não é noturno: é da UTA-RLDD (`F5_49_10`, 0,738; 270×480 + 162×288 px) — enquadramento e resolução pesam mais que a noite nos casos extremos. Dentro do NITYMED, a correlação entre a luminância do rosto e a detecção é fraca (ρ = 0,17, 126 vídeos).
2. **Sinal**: a SNR rápida do EAR cai de 21,5 (UTA-RLDD), 23,1 (DMD s5) e 13,6 (DMD parado) para 7,2 à noite: o ruído quadro a quadro noturno é 2,2× o da UTA-RLDD e 1,4× o do DMD parado, e a amplitude das piscadas é menor (0,065 × 0,080 e 0,086). **Mas o DMD com o carro em movimento, de dia, já tem ruído rápido 0,0097 e SNR rápida 8,7**: no mesmo motorista, com a mesma câmera, o movimento aumenta o ruído rápido em 0,0043 [0,0030; 0,0054] e muda a SNR rápida em -5,4 [-6,6; -3,5] (11 sujeitos, ruído maior em 10 deles; pela sessão do protocolo, s1 − s2: 0,0038 [0,0013; 0,0048] e -5,3 [-6,6; -3,7], 14 sujeitos). A diferença de SNR rápida entre a noite e o DMD em movimento não se distingue de zero (-1,6 [-3,7; 1,1]; ruído rápido -0,0013 [-0,0024; -0,0004]): a perda de qualidade do EAR à noite é do mesmo tamanho da que o carro em movimento produz de dia (vibração, movimentos de cabeça da direção real e luz que varia) e, com estes dados, não pode ser atribuída à escuridão por si. A noite ainda tem menos luz no rosto (luminância mediana 82,9 × 101,4 no DMD em movimento e 99,9 na UTA-RLDD) e muito mais compressão (0,045 × 0,561 e 0,276 bits por pixel); esses fatores não são separáveis aqui. Dentro do NITYMED, ρ(luminância dos olhos, ruído rápido) = -0,23.
3. **Detectores** (nenhum número é acurácia: o NITYMED não tem negativos). O modelo do DMD (Exp. 5b final) marca 26,0% das janelas alerta da UTA-RLDD, 45,7% das janelas alerta do DMD com o carro parado (fora da amostra) e 42,9% com o carro em movimento; nas sonolentas, 52,2% (UTA KSS 10) e 82,0% (DMD s5, fora da amostra). À noite marca 44,7% das janelas de microssono e 73,0% das de bocejo — nos microssonos, no nível do falso alarme diurno (o IC 95% do microssono contém o falso alarme do DMD parado e do DMD em movimento), então esse disparo não informa sensibilidade; nos bocejos, acima dele. O modelo da UTA-RLDD (B3) tem 43,6% de falso alarme no alerta da UTA (fora do fold), 69,1% no DMD parado e 77,1% no DMD em movimento, e 69,4% de disparo na UTA KSS 10; à noite marca 83,5% (microssono) e 89,5% (bocejo), inclusive 80,6% das janelas de microssono sem fechamento nem bocejo detectado (alerta da UTA sem evento: 42,2%; DMD em movimento sem evento: 70,0%). As características mais usadas por esse modelo são `ear_media`, `ear_dp`, `dur_max_fechamento_s`, e o `ear_dp` mediano das janelas de microssono é 0,031 contra 0,015 no alerta da UTA e 0,038 no DMD em movimento — o ruído do EAR (noturno ou do carro andando) é lido como "sonolência"; contrafactual: com o `ear_dp` típico do alerta da UTA, o disparo desse modelo nos microssonos iria de 83,5% para 64,5%, e trocando `ear_dp`, `amp_media`, `vel_media`, para 55,3%. A regra sem treino tem falso alarme de 2,6% no alerta da UTA, 14,9% no DMD parado e 23,8% no DMD em movimento; à noite marca 15,0% das janelas de microssono e 61,7% das de bocejo (UTA KSS 10: 24,1%; DMD s5: 75,7%) — nos microssonos, menos do que no alerta diurno com o carro em movimento.
4. **Microssono**: com o limiar do pipeline, o PERCLOS mediano dos vídeos de microssono é 2,9% (menor que o dos vídeos de bocejo, 6,5%) e só 16% deles têm fechamento > 1,5 s. Sem referência alerta (p90 do próprio vídeo), o PERCLOS mediano vai a 19,5% e 63% dos vídeos passam a ter fechamento > 1,5 s: o limiar relativo calibrado num trecho que já não é alerta esconde os fechamentos (seção 6).
5. **Resposta à PP3 (noite)**: a perda noturna do pipeline aparece na qualidade do sinal de EAR, não na detecção de rosto; e uma perda do mesmo tamanho aparece de dia no DMD com o carro em movimento, no mesmo motorista — com os dados disponíveis, a perda de sinal não se concentra na noite: acompanha a condução real (carro andando). Os modelos de janela disparam muito à noite, mas também disparam no alerta de dia com o carro em movimento: sem classe alerta noturna, esse disparo não mede sensibilidade à sonolência. A calibração do limiar exige um trecho alerta do próprio motorista.

## 9. Limitações e pendências

1. **Noite confundida**: câmera, enquadramento, resolução, compressão, protocolo e pessoas mudam junto com a iluminação; o DMD em movimento separa o fator "carro andando" só dentro do DMD (mesmos motoristas e câmera), e a comparação dele com o NITYMED ainda troca câmera, compressão, taxa de quadros (NITYMED 25 quadros/s × DMD 29,8; o ruído rápido é medido quadro a quadro) e pessoas.
2. **Motoristas não identificados no NITYMED** (21 declarados, 126 vídeos): IC e p-valores por vídeo são otimistas.
3. **Autenticidade não documentada** do microssono/bocejo do NITYMED (provavelmente sob instrução): taxa de disparo ≠ sensibilidade a sonolência real; o NITYMED não entra na PP1.
4. **Modelo do DMD**: o final do Exp. 5b (models/exp5/exp5b_final_C_sem_normalizacao_RF.joblib (Exp. 5b final, execução 20260927-211326; metadados models/exp5/exp5b_final_modelos.json)).
5. **Janelas avaliáveis no NITYMED**: só as que não tocam o primeiro terço (referência). Nos vídeos de bocejo (15–25 s) isso deixa 1–2 janelas por vídeo; 12 vídeos não têm nenhuma (sensibilidade com todas as janelas na seção 5.6).
6. **`setup` do DMD fora do protocolo** em gB_10 s1 de 2019-03-11 15:24 (Car Stopped), gB_6 s2 de 2019-03-11 13:46 (Car Moving), gZ_33 s1 de 2019-04-08 10:08 (Car Stopped), gZ_33 s1 de 2019-04-08 16:18 (Car Stopped): o agrupamento principal usa o `setup` (registro oficial); ruído rápido do EAR nos trechos de direção segura (`results/exp6_noite_videos.csv`): `gB_10` s1 (Car Stopped) 0,0058; `gB_6` s2 (Car Moving) 0,0072; `gZ_33` s1 (Car Stopped) 0,0075; `gZ_33` s1 (Car Stopped) 0,0061 — sessões conforme o protocolo: s1 em movimento mediana 0,0099 (faixa 0,0075–0,0130), s2 parada mediana 0,0055 (faixa 0,0037–0,0124). As faixas se sobrepõem: o ruído dessas sessões não confirma nem desmente o `setup` registrado. A comparação por sessão do protocolo fica como sensibilidade.
7. **Luminância**: 24 quadros com rosto por arquivo; no DMD, da sessão inteira (não só dos trechos de direção segura). A UTA-RLDD não documenta hora nem iluminação: "dia" vale só para o DMD (gravado entre 09h e 17h, pelo `recordTime`).
8. **Sem classe alerta noturna**: nenhum limiar pode ser calibrado para a noite com estes dados; a PP3 noturna fica respondida no nível de captura e sinal, não de acurácia.

## 10. Arquivos

- Código: `src/experiments/exp6_noite.py` (.venv), `src/experiments/exp6_luminancia.py` (.venv-mp), `src/experiments/run_exp6.sh`.
- Resultados: `results/exp6_noite.csv` (longo), `results/exp6_noite_videos.csv` (por vídeo lógico e recorte), `results/exp6_noite_janelas.csv` (por janela, com as probabilidades), `results/exp6_noite_importancias.csv`, `results/exp6_noite_contrafactual.csv`, `results/exp6_luminancia_videos.csv`, `results/exp6_nitymed_ficha.md`, `data/nitymed/index.csv`.
- Figuras: `results/figures/fig_exp6_noite.{pdf,png}` (A: quadros sem rosto por vídeo; B: SNR do EAR por vídeo; C: taxa de disparo), `results/figures/fig_exp6_microssono.{pdf,png}` (PERCLOS por vídeo e dois exemplos de EAR). Nenhuma contém rosto.
- Modelos: dmd_sem_normalizacao: models/exp5/exp5b_final_C_sem_normalizacao_RF.joblib (Exp. 5b final, execução 20260927-211326; metadados models/exp5/exp5b_final_modelos.json); dmd_principal: models/exp5/exp5b_final_C_principal_RF.joblib (Exp. 5b final, execução 20260927-211326; metadados models/exp5/exp5b_final_modelos.json); dmd_sem_taxa_rosto: models/exp6/rf_dmd_sem_taxa_rosto.joblib (treinado aqui, receita do Exp. 5b); uta_sem_normalizacao: models/exp6/rf_uta_sem_normalizacao.joblib (treinado aqui, receita do Exp. 5b); uta_principal: models/exp6/rf_uta_principal.joblib (treinado aqui, receita do Exp. 5b); uta_sem_taxa_rosto: models/exp6/rf_uta_sem_taxa_rosto.joblib (treinado aqui, receita do Exp. 5b); metadados em `models/exp6/modelos.json`.
- Logs: `results/logs/exp6_final_noite.log`, `results/logs/exp6_final_luminancia.log`, `results/logs/exp6_final_noite_disparo_resumo.csv`.
