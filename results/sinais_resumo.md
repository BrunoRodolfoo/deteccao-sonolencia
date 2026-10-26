# Extração de sinais por quadro (EAR/MAR/pose) — resumo

Gerado por `src/features/summarize_signals.py` a partir de `data/signals/<base>/index.csv` e dos `.npz` (extração: `src/features/extract_signals.py`, MediaPipe Face Mesh 0.10.21 legado, `refine_landmarks=False`, confiança de detecção/rastreamento 0,5, modo vídeo). CSVs: `results/sinais_resumo.csv` (por base), `results/sinais_resumo_videos.csv` (por vídeo), `results/sinais_resumo_estratos.csv` (estratos). Logs: `results/logs/extract_signals_<base>.log`.

| Base | Vídeos | Sujeitos | Quadros | Horas | fps (mín / mediana / máx) | Resoluções originais (nº vídeos) | Reduzidos p/ lado ≤1280 | Com rotação no metadado |
|---|---:|---:|---:|---:|---|---|---:|---:|
| DMD (sessões s1, s2, s3 e s5; dia) | 65 | 14 | 454.085 | 4,24 | 29,76 / 29,76 / 29,76 | 1280x720 (65) | 0 | 0 |
| NITYMED (noite, carro em movimento) | 126 | 126 | 146.484 | 1,63 | 24,00 / 25,00 / 25,00 | 1280x720 (126) | 0 | 0 |
| UTA-RLDD (real, celular/webcam) | 182 | 60 | 3.120.071 | 31,55 | 11,99 / 29,97 / 30,17 | 1080x1920 (59); 720x1280 (38); 1280x720 (31); 1080x720 (19); 1920x1080 (15); 1440x2560 (4); 427x240 (2); 848x480 (1); 480x848 (1); 480x720 (1); 320x568 (1); 2560x1440 (1); 480x640 (1); 240x426 (1); 352x640 (1); 720x480 (1); 480x853 (1); 480x270 (1); 270x480 (1); 162x288 (1); 480x320 (1) | 79 | 82 |

NITYMED não tem identificador de motorista: cada vídeo conta como um "sujeito". DMD: 16 sessões s5, 17 sessões s1, 16 sessões s2, 16 sessões s3 (s5 = sonolência atuada com o carro parado; s1 = direção com o carro em movimento; s2 e s3 = carro parado). UTA-RLDD: 5 folds (F1, F2, F3, F4, F5), 182 arquivos de vídeo e 180 vídeos lógicos (vídeos divididos em partes contam uma vez); a rotação do metadado foi aplicada (`CAP_PROP_ORIENTATION_AUTO=1`). Sujeitos da UTA com fps diferente entre os próprios vídeos (máx/mín > 1,05): `F1_04` (0: 25,00, 5: 30,00, 10: 25,00 fps); `F1_06` (0: 24,96, 5: 29,91, 10: 23,82 fps); `F1_11` (0: 29,94, 5: 15,03, 10: 17,16 fps); `F2_18` (0: 19,37, 5: 14,86, 10: 12,00 fps); `F3_28` (0: 29,99, 5: 24,01, 10: 24,01 fps); `F3_29` (0: 30,17, 5: 16,67, 10: 16,67 fps); `F3_32` (0: 11,99, 5: 15,32, 10: 20,69, 10: 20,73 fps); `F3_33` (0: 30,01, 5: 15,00, 10: 29,97 fps); `F3_34` (0: 15,36, 5: 29,59, 10: 30,00 fps); `F4_41` (0: 29,97, 5: 14,95, 10: 29,97 fps); `F4_45` (0: 12,50, 5: 14,25, 10: 14,25 fps); `F5_54` (0: 29,95, 5: 16,04, 10: 29,95 fps); `F5_55` (0: 30,01, 5: 23,40, 10: 23,92 fps) — Duration/Frequency em quadros dependem do fps; o z-score por sujeito só cancela o fator se o fps for igual nos vídeos do sujeito, o que NÃO vale para esses: usar duração em segundos).


## Taxa de detecção de rosto (fração de quadros em que o Face Mesh devolveu landmarks)

| Base | Média por vídeo | Ponderada por quadros | Mínimo por vídeo (vídeo) | Vídeos < 0,95 | Maior lacuna contínua sem rosto (s) | Mediana do EAR (mediana dos vídeos) |
|---|---:|---:|---|---:|---:|---:|
| DMD (sessões s1, s2, s3 e s5; dia) | 0,9843 | 0,9868 | 0,9151 (`gZ_33_s3_2019-04-04T09;33;20+02;00`) | 7 | 4,6 | 0,263 |
| NITYMED (noite, carro em movimento) | 0,9915 | 0,9902 | 0,8737 (`Yawning_Male_P1042773_720`) | 8 | 3,0 | 0,239 |
| UTA-RLDD (real, celular/webcam) | 0,9973 | 0,9954 | 0,6669 (`F5_49_10_p2`) | 1 | 86,0 | 0,265 |

### Por estrato

| Base | Estrato | Vídeos | Detecção média | Detecção mínima | Mediana do EAR |
|---|---|---:|---:|---:|---:|
| dmd | sem óculos | 51 | 0,9840 | 0,9151 | 0,263 |
| dmd | óculos | 14 | 0,9857 | 0,9235 | 0,256 |
| nitymed | Microsleep/Female | 8 | 0,9966 | 0,9875 | 0,289 |
| nitymed | Microsleep/Male | 11 | 0,9892 | 0,9398 | 0,238 |
| nitymed | Yawning/Female | 59 | 0,9948 | 0,9354 | 0,231 |
| nitymed | Yawning/Male | 48 | 0,9873 | 0,8737 | 0,235 |
| uta | rótulo 0 | 60 | 0,9995 | 0,9881 | 0,267 |
| uta | rótulo 10 | 62 | 0,9933 | 0,6669 | 0,259 |
| uta | rótulo 5 | 60 | 0,9993 | 0,9901 | 0,266 |

### Leitura para a PP3 (com cautela)

- A NITYMED (noite) tem taxa média de detecção de 0,9915, acima da do DMD (0,9843); o pior vídeo dela tem 0,8737, e 8 de 126 vídeos ficam abaixo de 0,95, contra 8 de 247 no DMD e na UTA-RLDD juntos. A diferença absoluta é pequena: a taxa ponderada é dmd 0,9868 / nitymed 0,9902 / uta 0,9954.
- **Não dá para atribuir a diferença só à noite**: as bases diferem também em câmera, distância (na NITYMED a câmera fica atrás do volante e o rosto é pequeno), carro em movimento e protocolo (vídeos de bocejo). Na inspeção dos 4 vídeos NITYMED de menor taxa, os quadros sem rosto eram cabeça de perfil, mão na frente do rosto e boca muito aberta durante o bocejo — a perda coincide com os eventos de interesse, o que enviesa MAR/EAR justamente neles.
- "Detectou rosto" ≠ "landmarks corretos": em modo vídeo o Face Mesh rastreia entre quadros e pode manter landmarks imprecisos. A qualidade do EAR precisa ser avaliada pelos eventos (no DMD, pela concordância com a anotação: `results/sinais_alinhamento_dmd.csv`).
- No DMD, a detecção média foi 0,9857 nas 14 sessões com óculos e 0,9840 nas 51 sem óculos (poucos motoristas de óculos: leitura descritiva).


## Velocidade de processamento (quadros/s por processo, em CPU, com vários processos em paralelo no Mac)

| Base | Mediana | Mín | Máx |
|---|---:|---:|---:|
| DMD (sessões s1, s2, s3 e s5; dia) | 93 | 67 | 136 |
| NITYMED (noite, carro em movimento) | 198 | 178 | 274 |
| UTA-RLDD (real, celular/webcam) | 183 | 26 | 321 |

A velocidade inclui decodificação, rotação e redução do quadro; vídeos 1080p/1440p HEVC de iPhone são os mais lentos. Não é medida de latência no celular (isso é o Exp. 4/PP2).

