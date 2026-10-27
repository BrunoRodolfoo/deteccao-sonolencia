# Teste independente do app em sono real (UTA-RLDD)

Gerado em 26/10/2026, 23h02, por `tools/app_validacao/validar_motor_uta.mjs` (log: `results/logs/app_validacao_uta_20261026-230247.log`). Uma linha por vídeo em `results/app_validacao_uta.csv`.

## Como foi feito

- **O que foi testado:** o motor do app 2.1, `app/pwa/engine.js`, importado sem cópia, com a configuração final (`CFG`), pelo mesmo caminho da validação offline do DMD (`results/app_validacao_dmd.md`). O arquivo de hoje (md5 ef2c9d3e…) difere do usado na validação no navegador (md5 c4cc34e5…) só num comentário e no texto do PERCLOS em "Por quê?"; a configuração e a lógica são as mesmas. O LSTM não entra: no app ele soma no máximo 10 pontos e só quando os sinais visíveis já deram amarelo, então não muda a entrada no amarelo; o tempo em vermelho do app completo pode ser um pouco maior.
- **Sinais:** o mesmo Face Landmarker da PWA, com blendshapes e pose (`tools/app_validacao/extrair_sinais_app.py`), em todos os quadros de cada vídeo. A rotação do celular foi aplicada e o quadro foi reduzido para no máximo 1280 px no lado maior, como na cadeia offline e no DMD. O tempo de cada quadro é o carimbo do arquivo (há vídeos com fps variável).
- **Calibração:** a do app, nos 10 s iniciais com rosto de cada vídeo, como no início de cada viagem. Nos vídeos sonolentos, isso quer dizer calibrar já com sono.
- **Variante:** o monitor calibra no começo do vídeo acordado da mesma pessoa e depois recebe o vídeo sonolento, como quem calibra acordado e fica com sono depois. Só a base da calibração (limiar do olho, boca e pose de repouso) passa de um vídeo para o outro; a duração de base das piscadas sai, nas duas formas, das 20 primeiras piscadas do próprio vídeo. Os quadros avaliados são os mesmos nas duas formas.
- **Métricas:** depois da calibração, até o fim do vídeo (cerca de 10 min). Verde, amarelo e vermelho são os estados do app (nível abaixo de 40, de 40 a 69, 70 ou mais, com histerese, ou vermelho forçado por microssono e cabeça caída). Alarme = entrada no vermelho. "Chega ao amarelo" = algum tempo em amarelo ou vermelho. AUC = chance de um vídeo sonolento ter valor maior que um acordado (empate vale meio). IC de 95 % por bootstrap de sujeitos (1.000 reamostras, semente 42; cada pessoa sorteada entra com os dois vídeos).

## Amostra e regra de escolha

- **Regra, fixada antes de rodar:** todos os sujeitos dos folds oficiais 2 e 4 da UTA-RLDD (`data/features/final/uta/windows_10s.csv`). São os dois folds sem nenhum dos 3 sujeitos do teste no navegador (Fold1/09, Fold3/31 e Fold5/51 estão nos folds 1, 3 e 5). Nenhum sujeito ou vídeo foi tirado depois.
- **Amostra:** 24 sujeitos (Fold2/13, Fold2/14, Fold2/15, Fold2/16, Fold2/17, Fold2/18, Fold2/19, Fold2/20, Fold2/21, Fold2/22, Fold2/23, Fold2/24, Fold4/37, Fold4/38, Fold4/39, Fold4/40, Fold4/41, Fold4/42, Fold4/43, Fold4/44, Fold4/45, Fold4/46, Fold4/47, Fold4/48), com o vídeo 0 (acordado, KSS 1 a 3) e o vídeo 10 (sonolento, KSS 8 e 9) de cada um: 48 vídeos e 8,2 h avaliadas.
- **Tempo:** antes de escolher, um vídeo (Fold4/43, vídeo 10: 720x1280, 10,0 min) foi extraído sozinho em 117,7 s. Com esse tempo e os da cadeia offline nos mesmos vídeos, a estimativa para os 48 vídeos foi de cerca de 40 a 45 min com 6 processos. A extração dos 47 restantes levou 87,3 min de relógio com 6 processos, mas o computador ficou suspenso (tampa fechada) das 21h40 às 22h30, por 50,1 min; descontada a suspensão, foram cerca de 37,2 min de processamento. Os vídeos 1080p e 1440p são os mais lentos. O motor do app roda em segundos.

## Resultado principal

| Medida | Calibração no próprio vídeo | Calibração do vídeo acordado |
|---|---|---|
| Acordados com algum alarme vermelho | 2/24 (8,3 %) | igual (mesma calibração) |
| Tempo em vermelho nos acordados | 0,46 %, com 3 alarmes em 4,0 h (0,74 por hora) | igual |
| Acordados que chegam ao amarelo | 5/24 (20,8 %), com 1,68 % do tempo em amarelo | igual |
| Sonolentos que chegam ao amarelo | 17/24 (70,8 %) | 19/24 (79,2 %) |
| Sonolentos que chegam ao vermelho | 15/24 (62,5 %) | 17/24 (70,8 %) |
| Tempo fora do verde: acordados / sonolentos (soma dos vídeos) | 2,14 % / 33,76 % | 2,14 % / 37,98 % |
| Tempo em vermelho nos sonolentos | 15,03 % | 18,58 % |
| Minutos até o 1º amarelo / 1º vermelho nos sonolentos (mediana entre os que chegam) | 3,0 / 3,9 | 2,2 / 3,9 |
| Pessoas com mais tempo fora do verde no vídeo sonolento (mais / igual / menos) | 16 / 7 / 1 | 19 / 5 / 0 |
| AUC do % do tempo fora do verde [IC 95 %] | 0,812 [0,705; 0,918] | 0,855 [0,760; 0,948] |
| AUC do nível médio [IC 95 %] | 0,839 [0,736; 0,935] | 0,891 [0,811; 0,965] |

Teste do sinal (bilateral, sem os empates) na comparação pareada: p < 0,001 com a calibração no próprio vídeo e p < 0,001 com a do acordado. Nível médio maior no vídeo sonolento: 20 de 24 pessoas (22 com a calibração do acordado).

| Eventos somados | Fechamentos de 0,5 a 1 s | Microssonos | Bocejos | Quedas da cabeça | Cabeceios | Vídeos com algum desses |
|---|---:|---:|---:|---:|---:|---:|
| 24 acordados | 17 | 5 | 1 | 0 | 0 | 6 |
| 24 sonolentos | 160 | 113 | 4 | 7 | 0 | 18 |
| 24 sonolentos, calibração do acordado | 220 | 174 | 6 | 10 | 1 | 20 |

## Leitura honesta

1. **Acordados: 2 de 24 vídeos com alarme.** Foram 3 alarmes em 4,0 h (0,74 por hora) e 0,46 % do tempo em vermelho; 5 vídeos chegaram ao amarelo. Os alarmes foram de Fold2/15 e Fold4/44, e todos começaram por microssono (olho fechado por 1 s ou mais, segundo o app); sem anotação de olho na UTA-RLDD, não dá para saber se foram olhos fechados de verdade ou olhar para baixo.
2. **Sonolentos: 17 de 24 chegaram ao amarelo e 15 ao vermelho**, o primeiro vermelho em mediana 3,9 min depois de calibrar. 7 ficaram verdes o vídeo inteiro com KSS 8 ou 9; em 6 deles o app não viu nenhum fechamento de 0,5 s ou mais, bocejo ou queda da cabeça.
3. **Por pessoa:** em 16 de 24 o vídeo sonolento ficou mais tempo fora do verde que o acordado, 7 empataram (verdes nos dois) e 1 teve o contrário (Fold4/44, que calibrou o vídeo sonolento com os olhos fechados em 21,3 % do tempo). AUC de 0,812 [0,705; 0,918] pelo % do tempo fora do verde e de 0,839 [0,736; 0,935] pelo nível médio: acima do acaso (0,5) com o IC inteiro, mas longe de 1.
4. **Calibrar já com sono atrapalha.** Em 7 dos 24 vídeos sonolentos, a calibração viu os olhos fechados em mais de 10 % do tempo, e em 8 houve algum aviso de calibração (o app avisa e pede para recalibrar); nos acordados, os olhos fechados passaram de 10 % em 2. Com a calibração do vídeo acordado, 19 sonolentos chegaram ao amarelo e 17 ao vermelho, o vídeo sonolento ficou mais tempo fora do verde em 19 pessoas e menos em nenhuma, e a AUC passou a 0,855 [0,760; 0,948] e 0,891 [0,811; 0,965].
5. **Para a banca:** com sono real, declarado pela própria pessoa, o app saiu do verde em 17 de 24 sonolentos e em 5 de 24 acordados, e alarmou 15 sonolentos contra 2 acordados. Ele reage a sinais visíveis (fechamentos longos e microssonos), não ao sono declarado: 7 sonolentos ficaram verdes. O teste foi com pessoas sentadas, não dirigindo.

## Limitações

- **Sentado, não dirigindo.** Os vídeos da UTA-RLDD foram gravados pelas próprias pessoas, com o celular ou a webcam, sentadas diante da câmera e sem dirigir. Na direção há movimento, luz que muda e olhares para os espelhos; no DMD, só o movimento do carro já derrubou a qualidade do sinal dos olhos (Exp. 6). O falso alarme em movimento continua medido pela direção real do DMD (`results/app_validacao_dmd.md`).
- **O rótulo é o que a pessoa declarou (KSS) para o vídeo inteiro**, não sinais anotados quadro a quadro. O app só vê sinais visíveis (olhos, boca, cabeça): quem disse estar com sono mas não fechou os olhos nem bocejou fica verde, e aqui não dá para medir revocação por evento.
- **Calibração.** No uso normal, cada vídeo sonolento é calibrado já com sono (o app calibra no início de cada viagem). A variante usa a calibração do vídeo acordado, mas cruza duas gravações feitas em momentos diferentes: a posição da câmera e a luz podem mudar, e a pose de base da cabeça pode ficar deslocada.
- **Offline, no Mac.** É o mesmo código do motor do app, mas não o app no celular: o LSTM ficou de fora, o quadro foi reduzido para no máximo 1280 px (no celular o app recebe 640x480) e 8 dos 48 vídeos, de 4 pessoas (Fold2/18, Fold2/24, Fold4/44, Fold4/45), têm menos de 20 quadros por segundo, o que mede as piscadas com menos quadros. O rosto foi achado em pelo menos 95,0 % dos quadros de cada vídeo.
- **Amostra.** 24 pessoas de 2 dos 5 folds, escolhidas pela regra acima e não pelos resultados. Com esse tamanho, os intervalos de confiança são largos; as outras pessoas da base não foram testadas aqui.

## Por pessoa

A = vídeo acordado (0); S = vídeo sonolento (10), calibrado no próprio começo; S\* = vídeo sonolento com a calibração do vídeo acordado. Rosto = % dos quadros com rosto. Fora do verde e vermelho em % do tempo avaliado. Eventos = fechamentos de 0,5 a 1 s / microssonos / bocejos / quedas e cabeceios da cabeça. Piscadas/min = fechamentos de 50 a 500 ms que o app marcou como olho fechado: o app não foi feito para contar piscadas, e valores perto de 0 querem dizer que, nas piscadas rápidas daquela pessoa, a pálpebra não chegou ao limiar de "fechado". Olho fechado = % dos quadros com rosto marcados como olho fechado.

| Pessoa | Rosto A / S | Fora do verde A / S / S\* | Vermelho A / S / S\* | Alarmes A / S / S\* | Nível médio A / S / S\* | Nível máx. A / S / S\* | Eventos A | Eventos S | Eventos S\* | Piscadas/min A / S | Olho fechado A / S |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Fold2/13 | 99,7 / 100,0 | 0,0 / 51,4 / 51,4 | 0,0 / 33,3 / 33,3 | 0 / 7 / 7 | 0,0 / 48,9 / 48,9 | 0 / 100 / 100 | 0 / 0 / 0 / 0 | 11 / 24 / 0 / 0 | 11 / 24 / 0 / 0 | 0,2 / 18,4 | 0,1 / 16,2 |
| Fold2/14 | 100,0 / 100,0 | 0,0 / 32,3 / 32,3 | 0,0 / 0,0 / 0,0 | 0 / 0 / 0 | 0,0 / 24,3 / 24,3 | 0 / 60 / 60 | 0 / 0 / 0 / 0 | 13 / 0 / 0 / 0 | 13 / 0 / 0 / 0 | 8,9 / 11,4 | 2,0 / 4,5 |
| Fold2/15 | 100,0 / 100,0 | 15,0 / 17,6 / 17,6 | 9,9 / 8,2 / 8,2 | 1 / 1 / 1 | 14,8 / 16,3 / 16,3 | 91 / 89,9 / 89,9 | 4 / 3 / 0 / 0 | 3 / 0 / 0 / 1 | 3 / 0 / 0 / 1 | 33,2 / 19,0 | 10,1 / 5,0 |
| Fold2/16 | 100,0 / 100,0 | 0,0 / 0,0 / 0,0 | 0,0 / 0,0 / 0,0 | 0 / 0 / 0 | 0,0 / 0,0 / 0,0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0,6 / 5,3 | 0,2 / 1,4 |
| Fold2/17 | 100,0 / 99,8 | 0,0 / 55,5 / 55,5 | 0,0 / 11,5 / 11,5 | 0 / 3 / 3 | 0,0 / 35,2 / 35,2 | 0 / 83,8 / 83,8 | 0 / 0 / 0 / 0 | 13 / 3 / 0 / 0 | 13 / 3 / 0 / 0 | 1,2 / 12,6 | 0,2 / 6,8 |
| Fold2/18 | 100,0 / 100,0 | 5,9 / 86,7 / 99,0 | 0,0 / 22,3 / 97,7 | 0 / 12 / 3 | 5,1 / 54,3 / 97,5 | 49,7 / 80,9 / 100 | 2 / 0 / 0 / 0 | 21 / 10 / 1 / 0 | 50 / 62 / 1 / 0 | 26,2 / 6,1 | 9,0 / 7,2 |
| Fold2/19 | 100,0 / 100,0 | 0,0 / 0,0 / 0,0 | 0,0 / 0,0 / 0,0 | 0 / 0 / 0 | 0,0 / 0,0 / 0,0 | 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 26,5 / 18,8 | 5,7 / 4,2 |
| Fold2/20 | 99,3 / 99,4 | 3,2 / 74,7 / 89,3 | 0,0 / 74,1 / 74,4 | 0 / 2 / 3 | 4,7 / 76,7 / 83,5 | 43,6 / 100 / 100 | 2 / 0 / 0 / 0 | 11 / 24 / 0 / 4 | 23 / 31 / 0 / 7 | 12,0 / 14,1 | 3,2 / 28,1 |
| Fold2/21 | 100,0 / 100,0 | 0,0 / 18,2 / 18,2 | 0,0 / 0,5 / 0,5 | 0 / 1 / 1 | 0,0 / 13,2 / 13,2 | 0 / 75 / 75 | 0 / 0 / 0 / 0 | 1 / 1 / 2 / 0 | 1 / 1 / 2 / 0 | 8,4 / 30,4 | 2,3 / 8,4 |
| Fold2/22 | 100,0 / 100,0 | 0,0 / 83,8 / 83,8 | 0,0 / 13,9 / 13,9 | 0 / 2 / 2 | 0,0 / 52,0 / 52,0 | 0 / 79,3 / 79,3 | 0 / 0 / 0 / 0 | 36 / 2 / 0 / 0 | 36 / 2 / 0 / 0 | 6,3 / 10,1 | 1,8 / 9,9 |
| Fold2/23 | 95,0 / 99,1 | 0,0 / 24,3 / 24,3 | 0,0 / 0,3 / 0,3 | 0 / 1 / 1 | 0,0 / 23,5 / 23,5 | 0 / 75 / 75 | 0 / 0 / 0 / 0 | 5 / 1 / 0 / 0 | 5 / 1 / 0 / 0 | 1,6 / 7,1 | 0,3 / 2,9 |
| Fold2/24 | 100,0 / 100,0 | 0,0 / 0,0 / 2,9 | 0,0 / 0,0 / 0,2 | 0 / 0 / 1 | 6,1 / 0,0 / 9,6 | 25 / 0 / 75 | 1 / 0 / 1 / 0 | 0 / 0 / 0 / 0 | 3 / 1 / 0 / 0 | 46,1 / 0,0 | 10,4 / 0,0 |
| Fold4/37 | 100,0 / 99,8 | 4,6 / 52,2 / 52,2 | 0,0 / 11,2 / 11,2 | 0 / 9 / 9 | 8,9 / 34,1 / 34,1 | 47,8 / 82,9 / 82,9 | 4 / 0 / 0 / 0 | 10 / 8 / 0 / 0 | 10 / 8 / 0 / 0 | 3,1 / 17,6 | 1,3 / 8,7 |
| Fold4/38 | 100,0 / 100,0 | 0,0 / 83,0 / 83,0 | 0,0 / 53,7 / 53,7 | 0 / 5 / 5 | 1,9 / 67,2 / 67,2 | 11,1 / 100 / 100 | 0 / 0 / 0 / 0 | 2 / 22 / 0 / 0 | 2 / 22 / 0 / 0 | 5,9 / 4,0 | 1,3 / 19,1 |
| Fold4/39 | 100,0 / 100,0 | 0,0 / 10,4 / 21,0 | 0,0 / 0,3 / 0,3 | 0 / 1 / 1 | 0,0 / 11,7 / 15,8 | 0 / 75 / 75 | 0 / 0 / 0 / 0 | 3 / 1 / 0 / 0 | 5 / 1 / 0 / 0 | 4,2 / 32,3 | 0,9 / 11,9 |
| Fold4/40 | 100,0 / 100,0 | 0,0 / 0,0 / 0,0 | 0,0 / 0,0 / 0,0 | 0 / 0 / 0 | 0,0 / 2,7 / 2,7 | 0 / 25 / 25 | 0 / 0 / 0 / 0 | 2 / 0 / 0 / 0 | 2 / 0 / 0 / 0 | 13,3 / 22,6 | 3,1 / 9,2 |
| Fold4/41 | 100,0 / 100,0 | 0,0 / 77,1 / 77,6 | 0,0 / 67,9 / 68,2 | 0 / 4 / 4 | 0,0 / 58,7 / 62,9 | 0 / 100 / 100 | 0 / 0 / 0 / 0 | 1 / 4 / 0 / 1 | 1 / 4 / 0 / 2 | 2,6 / 0,9 | 0,8 / 63,2 |
| Fold4/42 | 99,9 / 99,9 | 0,0 / 0,0 / 9,0 | 0,0 / 0,0 / 0,0 | 0 / 0 / 0 | 0,0 / 0,0 / 12,3 | 0 / 0 / 47,8 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 5 / 0 / 0 / 0 | 1,2 / 17,0 | 0,4 / 4,0 |
| Fold4/43 | 99,9 / 99,9 | 0,0 / 0,0 / 0,0 | 0,0 / 0,0 / 0,0 | 0 / 0 / 0 | 0,0 / 0,0 / 0,0 | 0 / 3,7 / 3,7 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 10,2 / 12,3 | 2,8 / 4,1 |
| Fold4/44 | 100,0 / 99,9 | 21,5 / 11,0 / 68,1 | 0,7 / 0,0 / 9,1 | 2 / 0 / 2 | 14,7 / 14,8 / 44,5 | 75 / 60 / 83,9 | 4 / 2 / 0 / 0 | 6 / 0 / 0 / 0 | 15 / 1 / 2 / 0 | 14,1 / 15,9 | 4,0 / 4,3 |
| Fold4/45 | 100,0 / 99,8 | 0,0 / 29,5 / 29,5 | 0,0 / 1,1 / 1,1 | 0 / 3 / 3 | 0,0 / 20,8 / 20,8 | 0 / 75 / 75 | 0 / 0 / 0 / 0 | 7 / 3 / 0 / 0 | 7 / 3 / 0 / 0 | 14,1 / 23,9 | 3,6 / 9,3 |
| Fold4/46 | 100,0 / 100,0 | 0,0 / 21,4 / 16,8 | 0,0 / 0,8 / 0,8 | 0 / 2 / 2 | 0,0 / 14,5 / 14,5 | 0 / 75 / 75 | 0 / 0 / 0 / 0 | 2 / 2 / 1 / 0 | 2 / 2 / 1 / 0 | 0,3 / 17,4 | 0,1 / 5,0 |
| Fold4/47 | 100,0 / 100,0 | 0,0 / 0,0 / 0,0 | 0,0 / 0,0 / 0,0 | 0 / 0 / 0 | 0,0 / 0,2 / 0,2 | 0 / 8,6 / 8,6 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 | 0,2 / 4,9 | 0,0 / 1,2 |
| Fold4/48 | 100,0 / 100,0 | 0,0 / 51,8 / 51,8 | 0,0 / 25,6 / 25,6 | 0 / 5 / 5 | 3,2 / 40,4 / 40,4 | 11,1 / 100 / 100 | 0 / 0 / 0 / 0 | 13 / 8 / 0 / 1 | 13 / 8 / 0 / 1 | 33,3 / 10,2 | 8,0 / 18,4 |

