# Validação do app (PWA v2.1, ajuste final) com vídeos reais: câmera falsa do Chrome headless

Gerado em 27/09/2026, depois do ajuste final do detector. Os números vêm de `results/app_validacao.csv` (uma linha
por versão, vídeo e métrica; as linhas desta validação têm `versao_app = 2.1 (ajuste final)`). O log com a linha do
tempo de cada vídeo está em `results/logs/app_validacao_20260927-232519.log`. A validação anterior (v2.0), que motivou o
ajuste, está no fim deste arquivo, sem alterações, na seção "Antes do ajuste (v2.0)".

Os scripts ficam em `tools/app_validacao/`:
- `preparar_clipe.py` monta os clipes;
- `rodar_app_video.mjs` roda o app no Chrome (sempre mudo) e coleta os dados;
- `analisar_app_videos.mjs` gera o CSV (com `--versao` e `--anterior`), o log e as tabelas abaixo.

## Resumo

- **Acordado, o app ficou verde o tempo todo.** Nos 3 vídeos ALERTA do UTA-RLDD (3 pessoas de folds diferentes, 3 min cada): 100 % do tempo em verde, 0 alarmes e **0 avisos** nos três. Na v2.0, o Fold3/31 (quem pisca 75-76 vezes por minuto) ficava 27,2 % do tempo em amarelo, com 3 "ding".
- **A detecção não mudou.** DMD s5 (sonolência encenada, 4 sessões): os mesmos 42 microssonos, cada um com vermelho imediato; olho fechado por quadro com kappa 0,902 (v2.0: 0,903); fechamentos >= 1 s com revocação 31/34 e precisão 31/42 (iguais); bocejos 6/24, sem nenhum falso (igual).
- **Custo medido:** o tempo em vermelho caiu numa sessão encenada (gC_14: 72,5 % -> 36,7 %). Agora, depois de cada microssono, o estado volta ao amarelo se os outros sinais não sustentam o vermelho (é o que faz o roteiro da banca "reabrir -> volta" funcionar); ali, nos intervalos entre os microssonos, o PERCLOS ficou em 5-9 % e o nível ficou entre 40 e 69. Os 7 microssonos dispararam o vermelho na hora, e as entradas no vermelho foram de 1 para 6.
- **Cabeça:** o artefato do rosto sumindo com a cabeça baixa acabou (v2.0: "queda (saiu do quadro)" em 2 clipes do NITYMED, um deles com vermelho forçado; v2.1: nenhum). O registro agora mostra a inclinação lateral ("cabeça tombando com os olhos fechados (39° para o lado)").
- **Calibração:** o aviso novo ("olhos fechados em 23 % da calibração") apareceu só onde devia: no vídeo SONOLENTO do Fold3/31, calibrado já com sono.
- Tudo mudo (`--mute-audio`, `?volume=0`, volume e ganho 0 conferidos a cada 1 s), 60 quadros por segundo, 0 exceções de JavaScript nas 14 execuções.

## Como foi feito (v2.1)

- **O que foi testado:** o próprio app na versão final (`APP_VERSION 2.1`; md5 do `engine.js` c4cc34e5…, do `app.js` c104d475…, do `index.html` b13893cd…), idêntica à instalada em `app/pwa` ao fim do ajuste. Uma cópia congelada foi servida na porta 8769 (só em 127.0.0.1); a 8443 do celular não foi tocada.
- **Mesmos clipes da validação anterior:** os mesmos 14 trechos, com os mesmos parâmetros (`tools/app_validacao/preparar_clipe.py`: 30 fps, 640x480 ou 480x640 com faixas pretas, 5 s de preto antes e 3 s depois; o teste sintético da cabeça com as mesmas transformações). Os clipes foram gerados de novo a partir dos vídeos das bases e apagados depois de cada execução.
- **Silêncio:** todo Chrome foi aberto com `--mute-audio`, o app com `?autostart=1&volume=0`, e agora **sem** `--autoplay-policy=no-user-gesture-required` (política padrão do Chrome: sem toque do usuário o app nem cria o AudioContext). A cada 1 s o script conferiu volume 0 % e ganho 0 e abortaria se não estivessem assim; nenhuma execução abortou. Nenhum som tocou no Mac.
- **Coleta e análise:** iguais às da v2.0 (`rodar_app_video.mjs` envolve `window.__app.monitor.update`; `analisar_app_videos.mjs` sincroniza com a anotação do DMD pela correlação do eyeBlink e casa os eventos com tolerância de 0,5 s). Novas colunas por quadro: PERCLOS com as piscadas (informativo), se o LSTM somou e o piso após o vermelho forçado.
- **CSV:** `results/app_validacao.csv` tem agora a coluna `versao_app`: as linhas `2.0 (antes do ajuste)` são as da validação anterior, sem recálculo; as `2.1 (ajuste final)` são destas execuções.
- **Privacidade:** nenhum quadro ou rosto foi salvo. Os dados por quadro (só números) estão em `data/signals/app_headless/` (v2.1) e `data/signals/app_headless/v2.0_antes_do_ajuste/` (v2.0), que o Git ignora.

## Resultados (v2.1)

### DMD s5 (sonolência encenada, anotação quadro a quadro)

| Sessão | Óculos | FPS | Olho fechado por quadro: sens. / espec. / kappa | Fechamentos >= 1 s: revocação | Precisão | Fechamentos >= 0,5 s: revocação | Precisão | Bocejos: revocação (sem mão / com mão) | Precisão | 1º vermelho após calibrar | Tempo em vermelho | Alarmes |
|---|---|---:|---|---|---|---|---|---|---|---:|---:|---:|
| gE_29 (óculos) | sim | 60 | 0,795 / 1,000 / 0,848 | 12/13 (92 %) | 12/12 (100 %) | 14/18 (78 %) | 14/14 (100 %) | 2/4 (50 %) (2/2 · 0/2) | 2/2 (100 %) | 44,0 s | 72,2 % | 2 |
| gZ_33 | não | 60 | 0,991 / 1,000 / 0,995 | 6/6 (100 %) | 6/15 (40 %) | 18/18 (100 %) | 18/20 (90 %) | 1/9 (11 %) (1/5 · 0/4) | 1/1 (100 %) | 36,4 s | 78,8 % | 1 |
| gC_14 | não | 60 | 0,999 / 1,000 / 0,999 | 7/7 (100 %) | 7/7 (100 %) | 7/7 (100 %) | 7/8 (88 %) | 0/5 (0 %) (0/3 · 0/2) | — | 48,4 s | 36,7 % | 6 |
| gB_7 (extra: quedas de cabeça reais no motor offline) | não | 60 | 0,716 / 0,989 / 0,777 | 6/8 (75 %) | 6/8 (75 %) | 9/11 (82 %) | 9/11 (82 %) | 3/6 (50 %) (3/4 · 0/2) | 3/3 (100 %) | 36,2 s | 78,2 % | 2 |
| **Total** | | | 0,858 / 0,998 / 0,902 | 31/34 (91 %) | 31/42 (74 %) | 48/54 (89 %) | 48/53 (91 %) | 6/24 (25 %) (6/14 · 0/10) | 6/6 (100 %) | | | |

### Tempo em cada nível e eventos (após a calibração)

| Vídeo | Base | Janela | Rosto | Verde | Amarelo | Vermelho | Alarmes | Avisos | Bocejos | Fech. 0,5-1 s | Microssonos | Quedas | Cabeceios | Cabeça baixa | Nível máx. | Piscadas/min | Olho fechado |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gE_29 (óculos) | DMD s5 (encenada) | 170 s | 100 % | 25,9 % | 1,9 % | 72,2 % | 2 | 0 | 2 | 2 | 12 | 0 | 0 | 0 | 100 | 29 | 18,9 % |
| gZ_33 | DMD s5 (encenada) | 172 s | 100 % | 17,0 % | 4,2 % | 78,8 % | 1 | 1 | 1 | 5 | 15 | 0 | 0 | 0 | 100 | 16 | 16,9 % |
| gC_14 | DMD s5 (encenada) | 176 s | 100 % | 27,5 % | 35,8 % | 36,7 % | 6 | 0 | 0 | 1 | 7 | 0 | 0 | 0 | 100 | 17 | 12,8 % |
| gB_7 (extra: quedas de cabeça reais no motor offline) | DMD s5 (encenada) | 172 s | 100 % | 20,9 % | 0,9 % | 78,2 % | 2 | 1 | 3 | 3 | 8 | 1 | 1 | 0 | 100 | 70 | 21,5 % |
| Fold1/09, vídeo 0 | UTA-RLDD 0 (alerta) | 180 s | 100 % | 100,0 % | 0,0 % | 0,0 % | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 3,1 % |
| Fold3/31, vídeo 0 | UTA-RLDD 0 (alerta) | 180 s | 100 % | 100,0 % | 0,0 % | 0,0 % | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 76 | 12,0 % |
| Fold5/51, vídeo 0 | UTA-RLDD 0 (alerta) | 180 s | 100 % | 100,0 % | 0,0 % | 0,0 % | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9 | 2,0 % |
| Fold1/09, vídeo 10 | UTA-RLDD 10 (sonolento) | 180 s | 100 % | 100,0 % | 0,0 % | 0,0 % | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 25 | 13 | 2,8 % |
| Fold3/31, vídeo 10 | UTA-RLDD 10 (sonolento) | 180 s | 100 % | 6,4 % | 0,0 % | 93,5 % | 1 | 1 | 1 | 9 | 10 | 0 | 0 | 0 | 100 | 29 | 43,1 % |
| Fold5/51, vídeo 10 | UTA-RLDD 10 (sonolento) | 180 s | 100 % | 100,0 % | 0,0 % | 0,0 % | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 2,9 % |
| Yawning/Female P1042778 | NITYMED bocejo (noite) | 49 s | 99 % | 66,1 % | 33,9 % | 0,0 % | 0 | 1 | 3 | 0 | 0 | 0 | 0 | 0 | 60 | 60 | 10,7 % |
| Yawning/Male P1042760 | NITYMED bocejo (noite) | 41 s | 100 % | 32,8 % | 23,9 % | 43,3 % | 2 | 0 | 3 | 0 | 1 | 0 | 0 | 0 | 75 | 52 | 11,1 % |
| Microsleep/Male P1042751 | NITYMED microssono (noite) | 119 s | 100 % | 100,0 % | 0,0 % | 0,0 % | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 25 | 24 | 5,2 % |

### Cabeça: teste sintético (transformação do vídeo com ffmpeg)

| Transformação (trecho) | Olhos (anotação) | Δpitch medido | \|Δroll\| máx. | \|Δyaw\| máx. | Evento de cabeça | Vermelho forçado |
|---|---|---:|---:|---:|---|---|
| pitch (topo perto) 45° (20,0-23,3 s) | abertos | -10,8 a 7,2° | 12,9° | 45,1° | nenhum | não |
| pitch (topo longe) 40° (26,0-29,3 s) | abertos | -4,1 a 0,5° | 6,6° | 6,9° | nenhum | não |
| achatamento 40° (32,0-35,3 s) | abertos | -1,9 a 1,1° | 10,9° | 10,5° | nenhum | não |
| roll 45° (40,0-43,3 s) | abertos | -4,8 a 0,4° | 43,4° | 5,0° | cabeca_baixa | não |
| roll 45° (62,2-63,6 s) | fechados | -1,6 a 0,1° | 40,0° | 4,0° | queda | sim |
| roll 45° (71,1-73,2 s) | fechados | -0,6 a 1,0° | 46,1° | 4,0° | queda | sim |

Eventos de cabeça fora das janelas transformadas (cabeça parada): 0. Controle (mesmo trecho sem transformação, execução dmd_gC14_s5): 0 evento(s) de cabeça.

### Antes x depois do ajuste (mesmos clipes)

| Vídeo | Base | Verde (antes → depois) | Amarelo | Vermelho | Alarmes | Avisos | Nível máx. | Microssonos | Bocejos |
|---|---|---|---|---|---|---|---|---|---|
| gE_29 (óculos) | DMD s5 (encenada) | 25,9 → 25,9 % | 0,0 → 1,9 % | 74,1 → 72,2 % | 1 → 2 | 0 → 0 | 100 → 100 | 12 → 12 | 2 → 2 |
| gZ_33 | DMD s5 (encenada) | 17,0 → 17,0 % | 4,2 → 4,2 % | 78,8 → 78,8 % | 1 → 1 | 1 → 1 | 100 → 100 | 15 → 15 | 1 → 1 |
| gC_14 | DMD s5 (encenada) | 27,5 → 27,5 % | 0,0 → 35,8 % | 72,5 → 36,7 % | 1 → 6 | 0 → 0 | 100 → 100 | 7 → 7 | 0 → 0 |
| gB_7 (extra: quedas de cabeça reais no motor offline) | DMD s5 (encenada) | 20,8 → 20,9 % | 0,3 → 0,9 % | 78,9 → 78,2 % | 1 → 2 | 1 → 1 | 100 → 100 | 8 → 8 | 3 → 3 |
| Fold1/09, vídeo 0 | UTA-RLDD 0 (alerta) | 100,0 → 100,0 % | 0,0 → 0,0 % | 0,0 → 0,0 % | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| Fold3/31, vídeo 0 | UTA-RLDD 0 (alerta) | 72,8 → 100,0 % | 27,2 → 0,0 % | 0,0 → 0,0 % | 0 → 0 | 3 → 0 | 46 → 1 | 0 → 0 | 0 → 0 |
| Fold5/51, vídeo 0 | UTA-RLDD 0 (alerta) | 100,0 → 100,0 % | 0,0 → 0,0 % | 0,0 → 0,0 % | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| Fold1/09, vídeo 10 | UTA-RLDD 10 (sonolento) | 100,0 → 100,0 % | 0,0 → 0,0 % | 0,0 → 0,0 % | 0 → 0 | 0 → 0 | 25 → 25 | 0 → 0 | 0 → 0 |
| Fold3/31, vídeo 10 | UTA-RLDD 10 (sonolento) | 6,5 → 6,4 % | 0,0 → 0,0 % | 93,5 → 93,5 % | 1 → 1 | 1 → 1 | 100 → 100 | 10 → 10 | 1 → 1 |
| Fold5/51, vídeo 10 | UTA-RLDD 10 (sonolento) | 100,0 → 100,0 % | 0,0 → 0,0 % | 0,0 → 0,0 % | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 | 0 → 0 |
| Yawning/Female P1042778 | NITYMED bocejo (noite) | 66,1 → 66,1 % | 33,9 → 33,9 % | 0,0 → 0,0 % | 0 → 0 | 1 → 1 | 70 → 60 | 0 → 0 | 3 → 3 |
| Yawning/Male P1042760 | NITYMED bocejo (noite) | 32,7 → 32,8 % | 0,0 → 23,9 % | 67,3 → 43,3 % | 1 → 2 | 0 → 0 | 88 → 75 | 1 → 1 | 3 → 3 |
| Microsleep/Male P1042751 | NITYMED microssono (noite) | 65,0 → 100,0 % | 35,0 → 0,0 % | 0,0 → 0,0 % | 0 → 0 | 1 → 0 | 48 → 25 | 0 → 0 | 0 → 0 |

Legenda da tabela do DMD: igual à da v2.0 (seção "Antes do ajuste"). "Antes x depois" compara as linhas `2.0` e `2.1` do CSV para o mesmo clipe.

## Leitura honesta (v2.1)

**1. Acordado (UTA-RLDD vídeo 0): 0 alarmes e 0 avisos em 9 min.**
- O PERCLOS da v2.1 só conta fechamentos >= 0,5 s: ficou em 0,0 % nos três vídeos alerta. Com as piscadas, como na v2.0, teria chegado a 5,3 %, 14,4 % e 5,0 % (coluna informativa `perclos_com_piscadas`).
- O +10 do LSTM não entrou em nenhum quadro desses vídeos (na v2.0, em 50 % dos quadros do Fold3/31): os sinais visíveis nunca chegaram a 40.
- Nível máximo: 0, 1 e 0 (v2.0: 0, 46 e 0).
- Cuidado com a amostra: 0 alarmes em 0,15 h não prova taxa zero (limite superior de 95 % de ~20 alarmes por hora, regra dos três). A referência maior é a direção real do DMD s1 (63 min, `results/app_validacao_dmd.md`): 3,3 % do tempo em vermelho, todo de um motorista (gA_5), e 16 de 17 sessões sem alarme.

**2. Sonolência encenada (DMD s5): mesma detecção; menos tempo em vermelho numa sessão.**
- Olhos por quadro (4 sessões): sensibilidade 0,858, especificidade 0,998, kappa 0,902. A regra dos olhos não mudou; a diferença para a v2.0 (0,856 / 0,998 / 0,903) é a variação entre execuções do navegador.
- Fechamentos >= 1 s: revocação 31/34 e precisão 31/42; >= 0,5 s: 48/54 e 48/53. Primeiro vermelho de 36 a 48 s depois da calibração, como na v2.0.
- Tempo em vermelho: 72,2 % (gE_29), 78,8 % (gZ_33), 36,7 % (gC_14) e 78,2 % (gB_7), contra 74,1 / 78,8 / 72,5 / 78,9 % na v2.0. Na gC_14, depois dos 3 primeiros microssonos (em 12 s), o nível parou em 69 (fechamentos no teto de 60 pontos + 9 do PERCLOS de 9,1 %), sem chegar a 70: amarelo até o microssono seguinte. Cada um dos 7 microssonos disparou o vermelho imediato.
- Entradas no vermelho (alarmes): 2, 1, 6 e 2 (v2.0: 1 em cada). Não é comparável com a v2.0, que ficava no vermelho contínuo.

**3. Bocejo: igual à v2.0.** Revocação de 6/24 no DMD (6/14 sem a mão, 0/10 com a mão) e precisão de 6/6. Os perdidos são bocejos encenados curtos (0,5-1,4 s de boca bem aberta) ou com a mão na frente. No NITYMED, 3 e 3 bocejos, como antes. Baixar a duração mínima para 1,0 s foi avaliado e rejeitado (2 bocejos falsos em 63 min de direção real).

**4. UTA-RLDD sonolento (vídeo 10): igual à v2.0.** Fold3/31 ficou 93,5 % em vermelho (10 microssonos, PERCLOS de até 76 %); Fold1/09 e Fold5/51 ficaram verdes (sem sinais visíveis de sono nos 3 primeiros minutos). O Fold3/31 recebeu o aviso novo: olhos fechados em 23 % da calibração.

**5. NITYMED à noite (infravermelho).**
- Vídeo de microssono: 100 % verde (v2.0: 35 % em amarelo). Nenhuma das versões achou fechamento >= 1 s. A v2.0 somava o PERCLOS das piscadas (até 8,8 %) e contou 2 fechamentos de 0,5-1 s; na v2.1 o primeiro deles mediu 0,498 s (na v2.0, 0,500 s), logo abaixo do limite de 0,5 s (variação entre execuções), e só 1 contou.
- Vídeo de bocejo (homem): 43,3 % em vermelho (v2.0: 67,3 %), 23,9 % em amarelo; o microssono dele disparou o vermelho imediato nas duas versões.
- No fim do clipe de microssono, quando o vídeo vira tela preta com a cabeça 19° abaixo, a v2.0 registrava "queda (saiu do quadro)" e forçava o vermelho; a v2.1 registra só "cabeça baixa com os olhos abertos (19° para baixo): distração, 0 ponto".

**6. Cabeça.** O teste sintético deu o mesmo resultado que na v2.0: inclinação lateral de 45° com os olhos abertos = "cabeça baixa" (distração); com os olhos fechados = "queda" e vermelho forçado nas 2 vezes; 0 eventos fora das janelas transformadas e 0 no controle. O texto agora diz "cabeça tombando com os olhos fechados (39° para o lado)" em vez de "(0° abaixo)". No gB_7 real: "cabeça caindo com os olhos fechados (28° para baixo)" e "cabeceio (37°, 0,6 s)", como antes. A revocação da cabeça continua sem medida (não há anotação de cabeceio).

**7. Desempenho.** 60 quadros por segundo no Chrome do Mac, com a GPU, e 0 exceções. O iPhone não foi medido com a v2.1.

## O que mudou no ajuste final (detalhes, princípios e efeitos em `app/pwa/README.md`, seção 9)

| Mudança | Efeito nestes vídeos | Efeito na direção real do DMD (s1, 63 min, `results/app_validacao_dmd.md`) |
|---|---|---|
| PERCLOS só com fechamentos >= 0,5 s | acordados: PERCLOS máx. 0 % (com piscadas: até 14,4 %) | PERCLOS p95 10,0 % -> 1,2 %; % do tempo em vermelho 5,1 -> 3,3 (só esta mudança) |
| LSTM só soma com os sinais visíveis >= 40 | Fold3/31 alerta: 3 avisos -> 0 | não medido offline (o validador não roda o LSTM) |
| Rosto que some só é queda com olhos fechados >= 0,4 s | NITYMED: 2 "quedas (saiu do quadro)" -> 0 | 1 queda em 63 min (igual: ela veio de olhos fechados, não do rosto perdido) |
| Microssono 25 pt/2 min + 1 min de ATENÇÃO + saída do forçado em 70 | gC_14: vermelho 72,5 % -> 36,7 %; roteiro da banca: amarelo 1,7 s depois de reabrir (teste [23]) | v2.0 -> v2.1 com todas as mudanças: vermelho 5,8 % -> 3,3 % e entradas no vermelho 1 -> 4 (todas do gA_5, que ia de 97 % para 56 % do tempo em vermelho). Só os pontos antigos do microssono (40 pt/5 min, sem piso), com o resto da v2.1: 3,3 % em vermelho e 4,6 % em amarelo |
| Aviso de olhos fechados na calibração (> 10 %) | aparece só no Fold3/31 sonolento (23 %) | — |
| Histerese nos olhos, microssono em 1,2 s, bocejo em 1,0 s | não aplicados | pioram o falso alarme ou perdem detecção (tabela "Ajuste final") |

## Limitações

- A amostra é pequena: 4 sessões do DMD, 3 pessoas do UTA e 3 vídeos do NITYMED, cerca de 40 min de vídeo. O DMD s5 é sonolência encenada; o UTA e o NITYMED não têm anotação de eventos.
- A calibração acontece no começo de cada vídeo, inclusive nos de sonolência (agora com aviso quando os olhos ficam fechados demais).
- Execuções repetidas do mesmo clipe variam um pouco (o relógio da câmera falsa e o Face Landmarker não são determinísticos): fechamentos perto de 0,5 s podem entrar ou não (NITYMED microssono).
- Tudo rodou no Chrome do Mac, não no Safari/WebKit do iPhone.
- A cabeça caindo para a frente (pitch) não tem revocação medida.

## Antes do ajuste (v2.0) — validação original, mantida sem alterações

Texto e tabelas da validação da versão 2.0 (`APP_VERSION 2.0`, md5 do `engine.js` a7dfafa1…, do `app.js` afe858bf…), gerados em 27/09/2026 a partir do log `results/logs/app_validacao_20260927-220950.log`. As linhas correspondentes do CSV estão em `results/app_validacao.csv` com `versao_app = 2.0 (antes do ajuste)`. Os títulos desceram um nível; o resto é o texto original.

Gerado em 27/09/2026. Os números vêm de `results/app_validacao.csv` (uma linha por vídeo e métrica). O log com a linha do tempo de cada vídeo está em `results/logs/app_validacao_20260927-220950.log`.

Os scripts ficam em `tools/app_validacao/`:
- `preparar_clipe.py` monta os clipes;
- `rodar_app_video.mjs` roda o app no Chrome e coleta os dados;
- `analisar_app_videos.mjs` gera o CSV, o log e as tabelas abaixo.

### Como foi feito

- **O que foi testado:** o próprio app, sem nenhuma mudança no código, rodando no Chrome 154 headless do Mac. Uma cópia congelada de `app/pwa` foi servida na porta 8766; a 8443 do celular não foi tocada. A versão é a `APP_VERSION 2.0` (md5 do `engine.js` a7dfafa1…, do `app.js` afe858bf…), idêntica à de `app/pwa` no fim das execuções.
- **Câmera:** cada vídeo virou um `.y4m` de 30 fps usado como câmera falsa (`--use-file-for-fake-video-capture`).
  - Vídeo deitado: 640x480. Vídeo em pé (retrato, como o celular no suporte): 480x640. Em ambos, a imagem foi reduzida sem distorcer, com faixas pretas nas sobras.
  - Antes do trecho há 5 s de tela preta, e depois dele 3 s.
  - Nos clipes em retrato, o Chrome entregou 480x480 ao app, cortando o centro para atender ao pedido de 640x480. O rosto foi encontrado em 100 % dos quadros mesmo assim.
- **Silêncio:** todo Chrome foi aberto com `--mute-audio`, e o app com `?autostart=1&volume=0`. A cada 1 s o script conferiu volume 0 % e ganho 0, e abortaria o teste se não estivessem assim; nenhuma execução abortou. Nenhum som tocou no Mac.
- **Coleta:** um script injetado envolve `window.__app.monitor.update` e guarda, a cada quadro, a decisão do motor (olhos, bocejo, cabeça, nível, estado, sirene, eventos) com o relógio do vídeo (`video.currentTime`).
  - O Chrome processou 60 quadros por segundo em todas as execuções (p95 do intervalo entre quadros: 18 ms). Foram 0 exceções de JavaScript.
- **Calibração:** acontece no próprio vídeo, que é o que o usuário faz no app: 10 s de rosto e depois 20 piscadas.
  - Nos vídeos de sonolência (DMD s5, UTA 10, NITYMED), a calibração **não** é feita em estado alerta. É uma limitação conhecida: a linha de base já sai "sonolenta".
- **Sincronia com a anotação do DMD:** o eyeBlink do app foi correlacionado com o eyeBlink extraído offline do mesmo vídeo.
  - Deslocamento encontrado: 0 a −0,033 s, ou seja, no máximo 1 quadro, sem deriva entre a 1ª e a 2ª metade. Correlação de 0,966 a 0,991.
  - Os eventos foram casados um a um, pela maior sobreposição, com tolerância de 0,5 s.
- **Privacidade:** nenhum quadro ou rosto foi salvo. Os `.y4m` ficaram na pasta temporária e foram apagados depois de cada execução. Os dados por quadro (só números) estão em `data/signals/app_headless/`, que o Git ignora.

### Resultados

#### DMD s5 (sonolência encenada, anotação quadro a quadro)

| Sessão | Óculos | FPS | Olho fechado por quadro: sens. / espec. / kappa | Fechamentos >= 1 s: revocação | Precisão | Fechamentos >= 0,5 s: revocação | Precisão | Bocejos: revocação (sem mão / com mão) | Precisão | 1º vermelho após calibrar | Tempo em vermelho | Alarmes |
|---|---|---:|---|---|---|---|---|---|---|---:|---:|---:|
| gE_29 (óculos) | sim | 60 | 0,794 / 1,000 / 0,847 | 12/13 (92 %) | 12/12 (100 %) | 14/18 (78 %) | 14/14 (100 %) | 2/4 (50 %) (2/2 · 0/2) | 2/2 (100 %) | 43,9 s | 74,1 % | 1 |
| gZ_33 | não | 60 | 0,989 / 1,000 / 0,994 | 6/6 (100 %) | 6/15 (40 %) | 18/18 (100 %) | 18/21 (86 %) | 1/9 (11 %) (1/5 · 0/4) | 1/1 (100 %) | 36,4 s | 78,8 % | 1 |
| gC_14 | não | 60 | 0,998 / 1,000 / 0,999 | 7/7 (100 %) | 7/7 (100 %) | 7/7 (100 %) | 7/8 (88 %) | 0/5 (0 %) (0/3 · 0/2) | — | 48,4 s | 72,5 % | 1 |
| gB_7 (extra: quedas de cabeça reais no motor offline) | não | 60 | 0,713 / 0,991 / 0,780 | 6/8 (75 %) | 6/8 (75 %) | 9/11 (82 %) | 9/11 (82 %) | 3/6 (50 %) (3/4 · 0/2) | 3/3 (100 %) | 36,2 s | 78,9 % | 1 |
| **Total** | | | 0,856 / 0,998 / 0,903 | 31/34 (91 %) | 31/42 (74 %) | 48/54 (89 %) | 48/54 (89 %) | 6/24 (25 %) (6/14 · 0/10) | 6/6 (100 %) | | | |

#### Tempo em cada nível e eventos (após a calibração)

| Vídeo | Base | Janela | Rosto | Verde | Amarelo | Vermelho | Alarmes | Avisos | Bocejos | Fech. 0,5-1 s | Microssonos | Quedas | Cabeceios | Cabeça baixa | Nível máx. | Piscadas/min | Olho fechado |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gE_29 (óculos) | DMD s5 (encenada) | 170 s | 100 % | 25,9 % | 0,0 % | 74,1 % | 1 | 0 | 2 | 2 | 12 | 0 | 0 | 0 | 100 | 31 | 19,0 % |
| gZ_33 | DMD s5 (encenada) | 172 s | 100 % | 17,0 % | 4,2 % | 78,8 % | 1 | 1 | 1 | 6 | 15 | 0 | 0 | 0 | 100 | 16 | 16,9 % |
| gC_14 | DMD s5 (encenada) | 176 s | 100 % | 27,5 % | 0,0 % | 72,5 % | 1 | 0 | 0 | 1 | 7 | 0 | 0 | 0 | 100 | 17 | 12,9 % |
| gB_7 (extra: quedas de cabeça reais no motor offline) | DMD s5 (encenada) | 171 s | 100 % | 20,8 % | 0,3 % | 78,9 % | 1 | 1 | 3 | 3 | 8 | 1 | 1 | 0 | 100 | 69 | 21,3 % |
| Fold1/09, vídeo 0 | UTA-RLDD 0 (alerta) | 180 s | 100 % | 100,0 % | 0,0 % | 0,0 % | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 14 | 3,1 % |
| Fold3/31, vídeo 0 | UTA-RLDD 0 (alerta) | 180 s | 100 % | 72,8 % | 27,2 % | 0,0 % | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 46 | 75 | 11,9 % |
| Fold5/51, vídeo 0 | UTA-RLDD 0 (alerta) | 180 s | 100 % | 100,0 % | 0,0 % | 0,0 % | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 2,0 % |
| Fold1/09, vídeo 10 | UTA-RLDD 10 (sonolento) | 180 s | 100 % | 100,0 % | 0,0 % | 0,0 % | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 25 | 13 | 2,8 % |
| Fold3/31, vídeo 10 | UTA-RLDD 10 (sonolento) | 180 s | 100 % | 6,5 % | 0,0 % | 93,5 % | 1 | 1 | 1 | 9 | 10 | 0 | 0 | 0 | 100 | 29 | 43,2 % |
| Fold5/51, vídeo 10 | UTA-RLDD 10 (sonolento) | 180 s | 100 % | 100,0 % | 0,0 % | 0,0 % | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 10 | 2,9 % |
| Yawning/Female P1042778 | NITYMED bocejo (noite) | 49 s | 99 % | 66,1 % | 33,9 % | 0,0 % | 0 | 1 | 3 | 0 | 0 | 0 | 0 | 0 | 70 | 56 | 11,0 % |
| Yawning/Male P1042760 | NITYMED bocejo (noite) | 41 s | 100 % | 32,7 % | 0,0 % | 67,3 % | 1 | 0 | 3 | 0 | 1 | 0 | 0 | 0 | 88 | 52 | 11,4 % |
| Microsleep/Male P1042751 | NITYMED microssono (noite) | 119 s | 100 % | 65,0 % | 35,0 % | 0,0 % | 0 | 1 | 0 | 2 | 0 | 0 | 0 | 0 | 48 | 25 | 5,4 % |

#### Cabeça: teste sintético (transformação do vídeo com ffmpeg)

| Transformação (trecho) | Olhos (anotação) | Δpitch medido | \|Δroll\| máx. | \|Δyaw\| máx. | Evento de cabeça | Vermelho forçado |
|---|---|---:|---:|---:|---|---|
| pitch (topo perto) 45° (20,0-23,3 s) | abertos | -10,8 a 7,4° | 12,9° | 45,7° | nenhum | não |
| pitch (topo longe) 40° (26,0-29,3 s) | abertos | -4,1 a 0,5° | 6,6° | 6,9° | nenhum | não |
| achatamento 40° (32,0-35,3 s) | abertos | -1,9 a 1,1° | 10,8° | 10,5° | nenhum | não |
| roll 45° (40,0-43,3 s) | abertos | -4,8 a 0,4° | 43,4° | 5,1° | cabeca_baixa | não |
| roll 45° (62,2-63,6 s) | fechados | -1,6 a 0,1° | 40,1° | 4,0° | queda | sim |
| roll 45° (71,1-73,2 s) | fechados | -0,6 a 1,0° | 46,1° | 4,1° | queda | sim |

Eventos de cabeça fora das janelas transformadas (cabeça parada): 0. Controle (mesmo trecho sem transformação, execução dmd_gC14_s5): 0 evento(s) de cabeça.


Legenda da tabela do DMD:
- A revocação de fechamentos é o número de fechamentos anotados que o app achou. A precisão é o número de detecções do app que correspondem a um fechamento anotado com a mesma duração mínima.
- "1º vermelho" é o tempo entre o fim da calibração e o primeiro estado SONOLENTO.
- gB_7 é uma sessão extra, escolhida porque o motor offline já registrava quedas de cabeça nela.

### Leitura honesta

**1. Acordado, o app não disparou o alarme.** É o número mais importante para a banca.
- Nos 3 vídeos ALERTA do UTA-RLDD (3 pessoas de folds diferentes, 3 min cada, 9 min no total): 0 alarmes e 0 s em vermelho.
- Dois deles ficaram 100 % do tempo em verde.
- O terceiro (Fold3/31) ficou 27 % do tempo em amarelo, com 3 "ding".
  - Essa pessoa pisca muito: 75 piscadas por minuto, de cerca de 100 ms, contra 10 a 14 nos outros dois.
  - Por isso o PERCLOS dela chegou a 12-14 % só com piscadas (36 pontos, nenhum fechamento ≥ 0,5 s).
  - Os +10 pontos do LSTM (ativo em 50 % dos quadros) levaram o nível a 40.
  - **Sem o LSTM, esse vídeo teria ficado sempre verde** (nível máximo de 36).
- Cuidado com a amostra: 0 alarmes em 0,15 h não prova que a taxa é zero. O limite superior de 95 % (regra dos três) é de cerca de 20 alarmes por hora.
- A referência maior continua sendo a validação offline na direção real do DMD s1 (1 h): 0,95 alarme por hora (`results/app_validacao_dmd.md`).

**2. Sonolência encenada (DMD s5): o app ficou vermelho nas 4 sessões.**
- O vermelho chegou de 36 a 48 s depois da calibração e ocupou de 72 % a 79 % do tempo.
- Houve 1 alarme por sessão: a sirene começa uma vez e continua enquanto o nível fica alto.
- **Olhos, quadro a quadro, no navegador** (4 sessões): sensibilidade 0,856, especificidade 0,998, kappa 0,903.
  - É coerente com a validação offline nas 16 sessões (kappa 0,938).
  - A pior sessão foi a gB_7 (kappa 0,780). A do motorista de óculos (gE_29) teve kappa 0,847: o app mede fechamentos um pouco mais curtos que a anotação (mediana −0,32 s).
- **Fechamentos ≥ 1 s:** revocação 31/34 (91 %) e precisão 31/42 (74 %).
  - Quase todos os "microssonos" que não batem com a anotação são fechamentos reais de 0,5 a 1 s que o app mediu mais longos.
  - Na gZ_33 foram 9 de 15, e a duração do app ficou em média 0,26 s acima da anotada. O app conta como fechado parte do "fechando" e do "abrindo", que a anotação separa.
  - Os números batem com a validação offline (0,920 / 0,748).

**3. Bocejo continua sendo o ponto fraco.**
- Revocação de 6/24 (25 %): 6/14 bocejos sem a mão na frente da boca e 0/10 com a mão. A precisão foi de 6/6 (nenhum bocejo falso).
- O jawOpen no navegador é praticamente igual ao do Python (mesmo modelo). O problema não é o navegador, e sim a regra do bocejo:
  - Parte dos bocejos encenados é curta: o jawOpen fica ≥ 0,4 por 0,5 a 1,4 s, menos que o mínimo de 1,5 s. Exemplos: gZ_33 em 38,3 s e 120,9 s; gC_14 em 41,8 s.
  - Outros abrem pouco a boca: jawOpen máximo de 0,36.
  - Com a mão na boca, o jawOpen fica entre 0,1 e 0,5 e o bocejo some.
- No NITYMED, o app contou 3 bocejos em cada um dos 2 vídeos de bocejo. O detector offline por MAR (`data/nitymed/index.csv`) também contou 3 e 3. Os dois concordam, mas isso **não** é anotação.

**4. UTA-RLDD SONOLENTO (vídeo 10): 1 de 3 pessoas ficou vermelha.**
- Fold3/31 ficou 93,5 % em vermelho, com microssonos de até 11,9 s.
- Fold1/09 e Fold5/51 ficaram verdes (nível máximo de 25 e 0). Nos 3 primeiros minutos, esses dois piscam no mesmo ritmo e ficam com o olho fechado pelo mesmo tempo que nos seus vídeos alerta (2,8 % contra 3,1 %; 2,9 % contra 2,0 %).
- O rótulo do UTA é a sonolência que a própria pessoa declarou (KSS), e não sinais visíveis. O app mede sinais visíveis, então não acusa quem não os mostra.
- Dizer isso na banca é mais honesto que prometer detectar "sono" em geral.

**5. NITYMED à noite (infravermelho): o rosto foi encontrado em 99-100 % dos quadros.**
- No vídeo de microssono, o app não achou nenhum fechamento ≥ 1 s. Achou 2 fechamentos de 0,5-0,7 s, que deixaram a tela amarela em 35 % do tempo.
- A base não tem anotação de eventos, então não dá para calcular revocação.

**6. Cabeça.**
- **Roll (rotação no plano, com ffmpeg) funciona de ponta a ponta:**
  - Com a cabeça tombada 45° e os olhos abertos, o app registrou só "cabeça baixa" (distração, 0 ponto).
  - Tombada durante os 2 fechamentos anotados, deu "queda" e vermelho forçado nas duas vezes.
  - Com a cabeça parada houve 0 eventos de cabeça, tanto fora das janelas transformadas quanto no mesmo trecho sem transformação (controle).
- **Pitch (cabeça caindo para a frente) não se simula com transformação 2D do quadro.**
  - Medido: perspectiva com o topo afastando = −4 a −8° (o landmarker leu como cabeça subindo). Topo aproximando = virar para o lado (|Δyaw| até 46°). Achatamento vertical = cerca de 0°.
  - Para o pitch valem os testes do motor com matrizes sintéticas (`app/pwa/test_engine.mjs`) e um vídeo real: na gB_7 (DMD s5), o app registrou 1 "queda" (28°) e 1 "cabeceio" (37°), perto das 2 quedas do motor offline.
  - **Não há anotação de cabeceio**, então a revocação da cabeça continua sem medida.
- **Artefato observado:** quando o rosto some com a cabeça ≥ 10° abaixo, o app registra "queda (saiu do quadro)" e força o vermelho depois de 2 s. Aconteceu no fim de 2 clipes do NITYMED, quando o vídeo virou tela preta (fora da janela avaliada).
  - Numa demonstração, algo parecido pode ocorrer se a pessoa se abaixar e sair do quadro.
- **Texto na interface:** a queda detectada pelo roll aparece como "cabeça caindo (0° abaixo)", porque o texto mostra só o pitch.

**7. Desempenho.** A 60 quadros por segundo no Chrome do Mac, com a GPU, e sem nenhuma exceção. O iPhone não foi medido aqui; falta rodar o "Benchmark 60 s" da versão nova no aparelho.

### Ajustes recomendados (não aplicados)

1. **LSTM não pode mudar o estado sozinho.** Só somar os +10 quando os outros sinais já passam de 40, ou tirar o LSTM do nível.
   - Evidência: todo o amarelo da pessoa acordada (Fold3/31, vídeo 0) veio desses +10. Sem eles, o nível máximo foi 36.
2. **PERCLOS sem as piscadas normais.** Há três caminhos:
   - contar só fechamentos ≥ 0,3-0,5 s;
   - ou descontar a fração de olho fechado medida na fase 2 da calibração;
   - ou começar a rampa em 10 % em vez de 7,5 %, mantendo os 40 pontos em 15 %.

   Evidência: quem pisca 75 vezes por minuto tem PERCLOS de 12-14 % acordado; os outros alertas ficaram entre 2 e 3 %.
3. **Tirar a oscilação do "fechado".** Usar histerese no eyeBlink (fecha em 0,5 e reabre abaixo de 0,4) ou exigir 2 quadros seguidos.
   - Evidência: no motorista de óculos (gE_29), a piscada mediana ficou em 50 ms, com 31 "piscadas" por minuto.
   - O efeito precisa ser medido com `tools/app_validacao/validar_motor_dmd.mjs` antes de mudar.
4. **Microssono em 1,2 s, ou medir só a parte totalmente fechada.** Hoje, 9 de 15 microssonos da gZ_33 são fechamentos anotados de 0,5-1 s, com +0,26 s de viés.
   - Conferir offline se a revocação de fechamentos ≥ 1 s (hoje 91 %) se mantém.
5. **Bocejo: duração mínima de 1,5 s para 1,0 s** (jawOpen ≥ 0,4).
   - Na grade offline (`results/app_validacao_dmd_bocejo_grade.csv`), a revocação sem mão sobe de 0,638 para 0,723, ao custo de 1,9 bocejo falso por hora na direção real.
   - Um bocejo vale 25 pontos, que sozinho não chega ao amarelo.
   - O bocejo com a mão na boca não se resolve com o jawOpen: fica documentado como limitação.
6. **Cabeça que sai do quadro:** exigir olhos fechados ≥ 0,4 s antes de o rosto sumir, ou subir `lostDownDeg` de 10° para 15°, para quem se abaixa para pegar algo não virar alarme de sono. Mostrar o roll no texto do evento (ex.: "cabeça tombando 40° para o lado").
7. **Peso da duração das piscadas:** não passou de 3 pontos em nenhum vídeo. Hoje é um sinal inerte; decidir entre reduzir o limiar de 1,25x para 1,15x ou retirá-lo do texto "Por quê?".
8. **Calibração:** avisar e pedir para recalibrar se mais de 10 % dos quadros da fase 1 tiverem os olhos fechados (eyeBlink ≥ 0,5). Na sessão sonolenta do Fold3/31, o limiar de EAR caiu de 0,262, no vídeo alerta, para 0,214.

### Limitações

- A amostra é pequena: 3 + 1 sessões do DMD, 3 pessoas do UTA e 3 vídeos do NITYMED, cerca de 40 min de vídeo.
- O DMD s5 é sonolência encenada. O UTA e o NITYMED não têm anotação de eventos.
- A calibração acontece no começo de cada vídeo, inclusive nos de sonolência.
- Tudo rodou no Chrome do Mac, que não é o Safari/WebKit do iPhone.
- A cabeça caindo para a frente (pitch) não tem revocação medida.
