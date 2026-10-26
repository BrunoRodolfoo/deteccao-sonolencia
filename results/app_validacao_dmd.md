# Validação offline do motor do app (detector v2) no DMD

Gerado em 2026-09-28T01:52:35.032Z por `tools/app_validacao/validar_motor_dmd.mjs` (log: `results/logs/app_validacao_dmd_20260927-225227.log`). Sinais por quadro extraídos com o MESMO `face_landmarker.task` da PWA (`tools/app_validacao/extrair_sinais_app.py`, MediaPipe 0.10.21 Tasks, CPU); o código avaliado é o `app/pwa/engine.js` importado sem cópia, com a calibração do app (10 s iniciais de cada vídeo). O LSTM não entra (no app vale no máximo +10 pontos).

## Olhos por quadro (DMD s5, sonolência encenada; quadros anotados 'open' x 'close', transições excluídas)

| Regra | Sensibilidade | Especificidade | Acurácia bal. | Kappa |
|---|---:|---:|---:|---:|
| ear_v1 | 0.994 | 0.949 | 0.971 | 0.831 |
| blend050 | 0.921 | 0.999 | 0.960 | 0.942 |
| app | 0.927 | 0.997 | 0.962 | 0.938 |

Médias entre as 16 sessões. `ear_v1` = EAR < 0,75 x mediana dos 10 s iniciais (regra da v1); `blend050` = os dois eyeBlink >= 0,5; `app` = regra final do app (CFG.eyes).

## Variantes da regra dos olhos no monitor completo (escolha da regra final)

Cada variante roda o monitor inteiro (nível, alarmes) em s5 (sonolência encenada, com anotação de olho) e em s1 (direção real, motoristas acordados). Revocação = fechamentos anotados >= X casados um-para-um com episódios detectados >= X; precisão = episódios >= X que se sobrepõem a olho não aberto anotado (fechado, fechando ou abrindo). Kappa por quadro = média das 16 sessões s5.

| Variante | Kappa s5 | Sens. s5 | Espec. s5 | Revoc. >=0,5 s | Precisão >=0,5 s | Revoc. >=1 s | Precisão >=1 s | % tempo vermelho s5 | % vermelho s1 | Alarmes/h s1 | Microssonos/h s1 | Fech. 0,5-1 s/h s1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ear_v1 | 0.831 | 0.994 | 0.950 | 0.985 | 0.961 | 0.982 | 0.971 | 77.0 | 26.1 | 32.3 | 66.4 | 176.4 |
| blend050 | 0.941 | 0.910 | 0.999 | 0.929 | 1.000 | 0.912 | 1.000 | 67.5 | 3.3 | 3.8 | 7.6 | 13.3 |
| blend045 | 0.954 | 0.939 | 0.998 | 0.955 | 0.996 | 0.912 | 0.993 | 68.4 | 3.3 | 3.8 | 7.6 | 23.7 |
| blend055 | 0.902 | 0.845 | 0.999 | 0.884 | 1.000 | 0.761 | 1.000 | 65.5 | 3.3 | 3.8 | 7.6 | 9.5 |
| blend050_rel030 | 0.935 | 0.900 | 0.999 | 0.919 | 1.000 | 0.876 | 1.000 | 67.4 | 0.1 | 1.9 | 2.8 | 12.3 |
| fusao_media035_2travas | 0.902 | 0.948 | 0.978 | 0.949 | 0.973 | 0.947 | 0.988 | 72.6 | 5.0 | 9.5 | 12.3 | 37.0 |
| fusao_min040_4travas | 0.938 | 0.918 | 0.997 | 0.929 | 1.000 | 0.920 | 1.000 | 68.6 | 3.3 | 3.8 | 7.6 | 14.2 |
| fusao_min045_4travas | 0.942 | 0.915 | 0.998 | 0.929 | 1.000 | 0.920 | 1.000 | 67.5 | 3.3 | 3.8 | 7.6 | 14.2 |
| blend050_rel025 | 0.941 | 0.910 | 0.999 | 0.929 | 1.000 | 0.912 | 1.000 | 67.5 | 3.3 | 3.8 | 7.6 | 13.3 |
| fusao_min040_4travas_rel025 | 0.938 | 0.918 | 0.997 | 0.929 | 1.000 | 0.920 | 1.000 | 68.6 | 3.3 | 3.8 | 7.6 | 14.2 |
| fusao_min040_4travas_rel030 | 0.933 | 0.909 | 0.997 | 0.924 | 1.000 | 0.876 | 1.000 | 68.5 | 0.1 | 1.9 | 2.8 | 13.3 |
| app | 0.938 | 0.918 | 0.997 | 0.929 | 1.000 | 0.920 | 1.000 | 68.6 | 3.3 | 3.8 | 7.6 | 14.2 |
| v1_app_anterior (EAR + PERCLOS>15% + olho fechado>=1 s; sem LSTM) | | | | | | | | 79.5 | 55.6 | 49.6 | | |

`app` = configuração final (CFG.eyes em app/pwa/engine.js).

## Ajuste final (27/09/2026): alternativas avaliadas no monitor completo

Cada linha é a configuração FINAL do app com UMA mudança. s1 = direção real, motoristas acordados (falso alarme); s5 = sonolência encenada (detecção). Revocação/precisão de fechamentos: casamento um-para-um estrito com a anotação 'close'. "Microssonos sobre fechamento anotado >= 1 s" = fechamentos anotados >= 1 s que dispararam o vermelho imediato. A 'final' é a do app; as outras foram avaliadas e NÃO aplicadas (motivo no app/pwa/README.md). Alarmes = entradas no vermelho; com a v2.1 o estado volta ao amarelo entre microssonos, então alarmes/h não é comparável com a v2.0 — compare também o % do tempo em vermelho (sirene).

| Alternativa | s1 % vermelho | s1 % amarelo | s1 alarmes/h | s1 avisos/h | s1 fech. 0,5-1 s | s1 microssonos | s1 bocejos | s1 PERCLOS p95 | s5 kappa | s5 revoc. >=1 s | s5 microssonos sobre fech. >=1 s | s5 bocejos | s5 % vermelho | s5 alarmes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| final | 3.3 | 4.1 | 3.79 | 1.90 | 15 | 8 | 0 | 1.2 % | 0.938 | 0.920 | 104/113 | 31 | 68.6 | 34 |
| perclos_com_piscadas | 5.1 | 2.5 | 3.79 | 1.90 | 15 | 8 | 0 | 10.0 % | 0.938 | 0.920 | 104/113 | 31 | 72.0 | 25 |
| histerese_010 | 3.7 | 15.2 | 3.79 | 17.07 | 46 | 11 | 0 | 2.9 % | 0.946 | 0.938 | 106/113 | 31 | 70.3 | 32 |
| histerese_005 | 3.5 | 7.9 | 3.79 | 7.59 | 29 | 9 | 0 | 2.2 % | 0.944 | 0.920 | 104/113 | 31 | 69.2 | 32 |
| microssono_12s | 3.3 | 4.1 | 4.74 | 1.90 | 15 | 8 | 0 | 1.2 % | 0.938 | 0.920 | 95/113 | 31 | 68.0 | 28 |
| bocejo_10s | 5.0 | 2.9 | 2.85 | 1.90 | 15 | 8 | 2 | 1.2 % | 0.938 | 0.920 | 104/113 | 36 | 68.8 | 34 |
| microssono_40pt_5min | 3.3 | 4.6 | 3.79 | 1.90 | 15 | 8 | 0 | 1.2 % | 0.938 | 0.920 | 104/113 | 31 | 70.7 | 26 |

Tabela completa: `results/app_validacao_dmd_ajuste_final.csv`.

## Duração dos fechamentos: app - anotação (DMD s5)

Pares um-para-um (fechamentos >= 0,3 s do app x 'close' anotado >= 0,3 s): n = 231; diferença p25 / mediana / p75 = 67 / 168 / 235 ms. O app mede do meio do fechamento (os dois eyeBlink >= 0,5) ao meio da reabertura; a anotação 'close' exclui as fases 'closing' e 'opening'. Por sessão (mediana, ms): gA_1_s5 336 (n=25); gA_5_s5 168 (n=25); gB_10_s5 168 (n=23); gB_10_s5 302 (n=13); gB_6_s5 50 (n=8); gB_7_s5 17 (n=12); gB_9_s5 134 (n=23); gC_13_s5 168 (n=8); gC_14_s5 134 (n=7); gE_29_s5 -571 (n=12); gF_23_s5 470 (n=7); gF_23_s5 67 (n=10); gZ_33_s5 235 (n=18); gZ_33_s5 235 (n=20); gZ_36_s5 168 (n=9); gZ_37_s5 -134 (n=11).

## Fechamentos por evento (app, DMD s5)

| Duração mínima | Anotados | Detectados | Revocação | Precisão |
|---|---:|---:|---:|---:|
| >= 500 ms | 198 | 234 | 0.929 | 0.786 |
| >= 1000 ms | 113 | 139 | 0.920 | 0.748 |

## Bocejos por evento (DMD s5: bocejos anotados com e sem a mão na frente da boca)

| Detector | Anotados (sem/com mão) | Detectados | Revocação sem mão | Revocação com mão | Precisão |
|---|---:|---:|---:|---:|---:|
| app (jawOpen>=0,4 e >= repouso+0,3, ou MAR>=repouso+0,4; por 1,5 s) | 85 (47/38) | 31 | 0.638 | 0.026 | 1.000 |
| v1 (MAR>0,6 por 2 s) | 85 (47/38) | 15 | 0.319 | 0.000 | 1.000 |

Grade (jawOpen x duração mínima) com os falsos bocejos por hora na direção real (s1): `results/app_validacao_dmd_bocejo_grade.csv`.

| jawOpen >= | Duração mín. | Revocação sem mão | Revocação com mão | Precisão s5 | Falsos/h em s1 |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 1000 ms | 0.787 | 0.026 | 1.000 | 1.897 |
| 0.3 | 1500 ms | 0.681 | 0.026 | 1.000 | 0.000 |
| 0.3 | 2000 ms | 0.596 | 0.026 | 1.000 | 0.000 |
| 0.35 | 1000 ms | 0.745 | 0.026 | 1.000 | 1.897 |
| 0.35 | 1500 ms | 0.660 | 0.026 | 1.000 | 0.000 |
| 0.35 | 2000 ms | 0.574 | 0.026 | 1.000 | 0.000 |
| 0.4 | 1000 ms | 0.723 | 0.026 | 0.972 | 1.897 |
| 0.4 | 1500 ms | 0.638 | 0.026 | 1.000 | 0.000 |
| 0.4 | 2000 ms | 0.553 | 0.026 | 1.000 | 0.000 |
| 0.45 | 1000 ms | 0.702 | 0.026 | 0.971 | 1.897 |
| 0.45 | 1500 ms | 0.617 | 0.026 | 1.000 | 0.000 |
| 0.45 | 2000 ms | 0.553 | 0.026 | 1.000 | 0.000 |
| 0.5 | 1000 ms | 0.681 | 0.026 | 0.971 | 1.897 |
| 0.5 | 1500 ms | 0.617 | 0.026 | 1.000 | 0.000 |
| 0.5 | 2000 ms | 0.553 | 0.026 | 1.000 | 0.000 |

## Monitor completo por tipo de sessão

| Sessões | Minutos ativos | Bocejos | Quedas | Cabeceios | Fech. 0,5-1 s | Microssonos | Alarmes (vermelho) | Alarmes/h | Avisos/h | % tempo SONOLENTO | % ATENÇÃO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s5 (16) | 46.6 | 31 | 5 | 1 | 95 | 134 | 34 | 43.74 | 14.15 | 68.6 | 9.8 |
| s1 (17) | 63.3 | 0 | 1 | 0 | 15 | 8 | 4 | 3.79 | 1.90 | 3.3 | 4.1 |

## Direção real (s1) por ação anotada

| Ação | Minutos | Eventos de sono/h | Alarmes/h | % SONOLENTO | Bocejos | Quedas | Cabeceios | Fech. | Microssonos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TOTAL | 63.3 | 22.8 | 3.8 | 3.3 | 0 | 1 | 0 | 15 | 8 |
| safe_drive | 35.7 | 11.8 | 1.7 | 4.0 | 0 | 0 | 0 | 4 | 3 |
| radio | 6.6 | 72.4 | 27.1 | 2.8 | 0 | 0 | 0 | 3 | 5 |
| drinking | 6.6 | 27.2 | 0.0 | 3.1 | 0 | 0 | 0 | 3 | 0 |
| reach_side | 5.2 | 57.6 | 0.0 | 3.3 | 0 | 1 | 0 | 4 | 0 |
| talking_to_passenger | 5.0 | 0.0 | 0.0 | 1.1 | 0 | 0 | 0 | 0 | 0 |
| unclassified | 3.8 | 15.8 | 0.0 | 1.6 | 0 | 0 | 0 | 1 | 0 |
| change_gear | 0.2 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0 |
| hair_and_makeup | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0 |
| sem_anotacao | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 0 | 0 | 0 | 0 |

## Diagnóstico da regra dos olhos

Quadros de s5 anotados como olho fechado x aberto (após a calibração): distribuição da média de eyeLookDown e do menor eyeBlink. Por sessão (`results/app_validacao_dmd_sessoes.csv`): fração dos quadros ativos com rosto marcados como fechados, fração pelo caminho A (blendshape) e eyeLookDown mediano.

| Grupo | Quadros | eyeLookDown p10 / p50 / p90 | menor eyeBlink p10 / p50 / p90 |
|---|---:|---|---|
| fechado_anotado_s5 | 8744 | 0.618 / 0.767 / 0.859 | 0.509 / 0.669 / 0.771 |
| aberto_anotado_s5 | 52564 | 0.083 / 0.260 / 0.469 | 0.061 / 0.136 / 0.241 |

Direção real (s1), sessões com mais tempo em SONOLENTO:

| Sessão | % SONOLENTO | Alarmes | Microssonos | % quadros fechados | % fechados pelo caminho A | eyeLookDown mediano |
|---|---:|---:|---:|---:|---:|---:|
| gA_5_s1 | 55.8 | 4 | 8 | 16.6 | 15.1 | 0.580 |
| gA_1_s1 | 0.0 | 0 | 0 | 4.4 | 4.4 | 0.602 |
| gB_10_s1 | 0.0 | 0 | 0 | 3.2 | 2.2 | 0.405 |

Demais 16 sessões s1: % quadros fechados mediano 3.2, eyeLookDown mediano 0.338.

## Sinal do pitch da matriz de pose (convenção do app: pitch > 0 = cabeça para baixo)

Quadro a quadro, correlação entre o pitch (`poseFromMatrix`) e a posição vertical relativa da ponta do nariz na imagem, (nariz_y - olhos_y) / (queixo_y - olhos_y) (marcos 1, 33/263 e 152), que aumenta quando a cabeça baixa (o nariz se projeta para baixo). Por vídeo em `results/app_validacao_dmd_pose_sinal.csv`.

| Vídeos | Correlação > 0 | Correlação mín. | Mediana | Máx. | Pitch mediano por vídeo (mín. / mediana / máx.) |
|---:|---:|---:|---:|---:|---|
| 33 | 33 | 0.526 | 0.857 | 0.949 | -26.0° / -10.9° / 2.1° |

Pitch mediano negativo é o esperado no DMD: a câmera fica abaixo do rosto (no painel), o que equivale a ver a cabeça levemente para cima. O app usa sempre o pitch RELATIVO à calibração (Δpitch), então essa inclinação fixa da câmera não conta como queda.
