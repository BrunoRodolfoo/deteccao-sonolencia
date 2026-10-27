# PWA: detecção de sonolência no navegador (detector v2.1)

Aplicação web sem etapa de build (JavaScript puro + ES modules) que roda **no próprio celular**, no
navegador: iPhone (Safari ou Chrome, ambos com o motor WebKit), Android (Chrome) e computador. Todo o
processamento é local; nenhum quadro de vídeo é armazenado ou enviado.

**Em uma frase:** três sinais que a banca consegue ver e conferir (**olhos, bocejo e cabeça**) somam
pontos num **nível de sonolência de 0 a 100** que sempre diz o **porquê**; microssono e cabeça caída
com os olhos fechados disparam o **vermelho na hora**; os modelos de aprendizado de máquina (LSTM e
CNN) aparecem como **estimativa experimental** e nunca mudam a cor sozinhos.

```
câmera frontal -> MediaPipe Face Landmarker (WebAssembly + WebGL, delegate GPU/CPU)
               -> 478 marcos + 52 blendshapes + matriz de pose 4x4
               -> engine.js (DrowsinessMonitor, código puro testado em Node):
                    OLHOS  = EAR calibrado + eyeBlink (blendshape), travas de "olhar para baixo"
                             -> piscadas, fechamentos >= 0,5 s, microssono >= 1 s,
                                PERCLOS 60 s só com fechamentos >= 0,5 s (piscadas normais não contam)
                    BOCEJO = jawOpen (blendshape) ou MAR relativo, sustentado >= 1,5 s
                    CABEÇA = pitch/yaw/roll da matriz -> queda (>= 18° por 0,8 s c/ olhos fechados), cabeceio
               -> nível 0-100 (soma de pontos com decaimento) + histerese
               -> ALERTA (verde) / ATENÇÃO (amarelo + "ding") / SONOLENTO (vermelho + sirene + tela de alarme)
  em paralelo, só informativo: recorte dos olhos 48x48 -> eye_state.tflite (CNN, Exp. 3)
                               4 características por piscada -> janela 30x4 -> blink_temporal.tflite (LSTM, Exp. 1)
```

**Versão 2.1 (ajuste final, 27/09/2026):** o que mudou em relação à 2.0, por quê e o efeito medido está
na seção 9. Resumo: o PERCLOS deixou de contar piscadas normais, o LSTM só soma com os sinais visíveis
já em ATENÇÃO, o rosto que some só vira "cabeça caindo" com os olhos fechados antes, e, ao reabrir os
olhos depois de um microssono, o app volta ao amarelo em ~2 s (antes ficava ~1,5 min no vermelho).

**Interface nova (27/10/2026), mesmo detector 2.1.** Telas no estilo do iPhone (fonte do sistema, listas
agrupadas, abas Dirigir, Viagens e Ajustes), com **tema claro e escuro**: Ajustes > Aparência tem
Automático (segue o aparelho), Claro e Escuro. O caminho de uma viagem:

- **Início** ("Nova viagem"): lembretes antes de sair, "Testar som" e **Iniciar viagem** (o toque que
  libera o som no iPhone). A câmera só liga quando a viagem começa.
- **Calibração**: a câmera em tela cheia com o círculo de enquadramento e o anel de progresso dos 10 s; com
  aviso de calibração, o app para e oferece **Recalibrar**; sem aviso, segue sozinho para a viagem.
- **Em viagem**: o mostrador do nível (0 a 100, faixas 40 e 70 marcadas no arco), o estado em palavras, o
  cartão "Hora de uma pausa" no amarelo, os três sinais (olhos, bocejos, cabeça), a lista **"Por que N"**
  (os pontos que somam o nível, em linguagem simples), o gráfico dos últimos 10 min e, no fim da página,
  **Detalhes técnicos** (EAR, latência, **Benchmark 60 s**, exportar sessão, modelos, registro de eventos).
  Barra de baixo: Som (volume e "Testar som"), Recalibrar e Encerrar.
- **Alarme**: no vermelho, a tela inteira fica vermelha (igual nos dois temas) com o motivo; o
  **"deslize: estou acordado"** precisa ser arrastado até o fim (um toque não silencia; pelo teclado,
  Enter ou espaço) e silencia só o vermelho por nível, por 30 s: a tela volta para a viagem (em vermelho, com
  "Som silenciado por 30 s") e, se o nível seguir alto quando os 30 s acabarem, o alarme volta com som. Num
  microssono ele nem aparece.
- **Encerrar** desliga a câmera e abre o **resumo da viagem** (aba Viagens): gráfico do nível, alarmes,
  avisos, bocejos, olhos fechados e cabeça, tempo em cada estado e os momentos. O último resumo fica salvo
  no aparelho (localStorage) e aparece no Início.
- **Como funciona**: faixas, pontos, tetos e o vermelho na hora, com os números lidos do próprio `E.CFG`.

O detector, os parâmetros e o benchmark são os mesmos da 2.1 (o número da versão não mudou): a tela só lê
o resultado do monitor, e o registro da viagem e a atualização da tela ficam fora do tempo medido por quadro.
Ícone novo (olho fechando dentro do arco do nível; ponto âmbar = aviso): `icon.svg`, `apple-touch-icon.png`
(180 px, tela de início do iPhone), `icon-192.png` e `icon-512.png`.

## 1. O que o app detecta

Todos os limiares e pesos estão num único lugar: `CFG` em `engine.js`. A tabela diz o valor e o
porquê de cada um. A origem de cada número (literatura, comparação no DMD ou regra de projeto) e como ele foi
testado estão na seção 13. "Validação offline" = `tools/app_validacao/validar_motor_dmd.mjs` sobre o DMD
(seção 8; log `results/logs/app_validacao_dmd_20260927-225227.log`).

### 1.1 Olhos

| Regra | Valor | Por quê |
|---|---|---|
| Fechado, caminho A (blendshape) | `min(eyeBlinkLeft, eyeBlinkRight) >= max(0,5; eyeBlink de repouso + 0,25)` | O blendshape do MediaPipe separa piscar de olhar para baixo melhor que o EAR. Os **dois** olhos (piscadela não conta). O "+0,25 sobre o repouso" evita que pálpebra pesada ou sol vire "fechado". |
| Fechado, caminho B (EAR) | EAR < 0,75 x mediana do EAR na calibração **e** menor eyeBlink >= 0,4 **e** nenhuma trava | 0,75 teve o maior kappa por quadro entre 0,60 e 0,90 no DMD (`results/exp5_dmd_sensibilidade_limiar.csv`: kappa médio 0,773). Mantém o detector funcionando se o blendshape oscilar. |
| Travas do caminho B | cabeça >= 10° abaixo da base; olhar para baixo (média eyeLookDown >= 0,6); olhos apertados (eyeSquint >= 0,45, risada/sol); olhando para o lado (\|Δyaw\| >= 20°) | Olhar para o painel/celular abaixa a pálpebra e derruba o EAR **sem** o olho fechar (era o maior falso positivo da v1). Nessas situações só o blendshape decide. |
| Escolha da regra | a "fusão com 4 travas" entre 11 variantes | Kappa por quadro 0,938 (EAR sozinho: 0,831) e, na direção real, 3,3 % do tempo em vermelho (EAR sozinho: 26,1 %), com a v2.1. Tabela completa em `results/app_validacao_dmd_variantes_olhos.csv`. |
| Piscada | fechamento de 50 ms a 0,5 s | abaixo de 50 ms é ruído de 1 quadro; entra na duração típica da pessoa. |
| Fechamento longo | 0,5 s a 1 s | conta como sinal (pontos que decaem em 2 min) e aparece por minuto no cartão "Olhos". |
| **Microssono** | olhos fechados >= 1 s | **VERMELHO imediato** (escolha conservadora: já não é piscada). A duração do app vai de eyeBlink >= 0,5 na descida a < 0,5 na subida (meio fechamento), por isso sai 0,17 s mais longa que o "fechado" da anotação do DMD, que exclui o "fechando" e o "abrindo" (mediana de 231 pares em s5; p25-p75: 0,07-0,24 s). Subir o limiar para 1,2 s foi avaliado e **rejeitado** (seção 9). |
| **PERCLOS** | tempo de olho fechado **em fechamentos >= 0,5 s** / max(tempo observado; 60 s), janela de 60 s | O PERCLOS clássico (P80: olhos pelo menos 80 % fechados; Wierwille et al., 1994) mede fechamentos **lentos** da pálpebra, não piscadas (Dinges e Grace, 1998). Não há um limite único: estudos e monitores usam de 8 % a 12 % sobre o PERCLOS de todos os fechamentos (12 % no relatório da AAA de 2018, citando Wierwille et al., 1994). O app usa 15 % sobre um PERCLOS mais restrito, só com fechamentos de 0,5 s ou mais, o que é mais conservador: uma escolha de projeto para evitar falso alarme. Um fechamento só entra quando completa 0,5 s (o mesmo limite de "fechamento longo") e aí entra inteiro. Na v2.0 as piscadas contavam: quem pisca 75 vezes por minuto acordado passava de 12 % (seção 9). O denominador mínimo de 60 s impede que UM fechamento logo após a calibração infle o PERCLOS. O PERCLOS "com as piscadas" continua visível em "Detalhes técnicos", só como informação. |
| Rosto perdido | > 0,3 s encerra o fechamento em andamento | não inventa olho fechado quando o rosto some. |
| Histerese na reabertura (`reopenHyst`) | **desligada** (0) | Avaliada (reabrir só abaixo de 0,4) e rejeitada: na direção real os fechamentos de 0,5-1 s foram de 15 para 46 e os avisos de 1,9 para 17,1 por hora (seção 9). |

### 1.2 Bocejo

| Regra | Valor | Por quê |
|---|---|---|
| Boca aberta | `jawOpen >= max(0,4; jawOpen de repouso + 0,3)` **ou** MAR >= MAR de repouso + 0,4 | O `jawOpen` do blendshape funciona melhor que o MAR sozinho (o detector v1, MAR > 0,6 por 2 s, achou 32 % dos bocejos sem a mão no DMD; este acha 64 %). O MAR relativo é apoio. |
| Duração mínima | 1,5 s | Na direção real (DMD s1), 1,5 s deu **0 bocejo falso por hora** (fala, rádio, bebida); 1,0 s deu 1,9 por hora. Bocejos espontâneos duram em média cerca de 6 s (Provine, 1986): 1,5 s de boca bem aberta cabe num bocejo de verdade. Os bocejos que o app perde no DMD são **encenados** e curtos (0,5-1,4 s de boca aberta). Baixar para 1,0 s foi avaliado e **rejeitado** (seção 9). |
| Tolerâncias | boca fecha < 0,15 s no meio: continua o mesmo bocejo; rosto perdido (mão na frente) até 1 s: continua | bocejos reais "respiram"; a fala abre e fecha por mais tempo. |
| Contagem | bocejos nos últimos 5 min | é o que o cartão "Bocejo" mostra. |

Limite honesto: com a **mão na frente da boca** o bocejo quase nunca é detectado (1 de 38 no DMD).

### 1.3 Cabeça caindo

A matriz 4x4 do Face Landmarker (`outputFacialTransformationMatrixes`) leva o modelo canônico do rosto
para o espaço métrico da câmera (cm; câmera na origem olhando para -Z, Y para cima). No tasks-vision
para web, `data` é o `packed_data` do proto **coluna a coluna** (translação em `data[12..14]`);
`poseFromMatrix` confere a ordem pelos próprios valores e extrai, com R = Rz(roll)·Ry(yaw)·Rx(pitch):
`pitch = atan2(R21, R22)`, `yaw = asin(-R20)`, `roll = atan2(R10, R00)`. Convenção: **pitch > 0 =
cabeça para baixo**, conferida nos 33 vídeos do DMD (correlação positiva em 33 de 33 entre o pitch e a
posição relativa da ponta do nariz na imagem, mediana 0,857; `results/app_validacao_dmd_pose_sinal.csv`)
e por matrizes sintéticas nos testes. O app usa sempre o ângulo **relativo à calibração** (Δpitch), então
a inclinação fixa do celular (câmera abaixo do rosto) não conta como queda.

| Evento | Regra | Por quê |
|---|---|---|
| **Queda** ("cabeça caindo") | Δpitch >= 18° (ou \|Δroll\| >= 30°, tombando para o lado) por >= 0,8 s **com os olhos fechados >= 0,4 s contínuos** no episódio | 15-20° e 0,8 s foram a especificação; exigir olhos fechados separa **sono** de **olhar o painel/celular**. Conta o maior fechamento contínuo, não a soma: olhando o celular, piscadas comuns (0,1-0,3 s) somariam 0,8 s em poucos segundos. O registro diz a direção: "cabeça caindo … (28° para baixo)" ou "cabeça tombando … (40° para o lado)". |
| **Vermelho imediato** | caída com olhos fechados >= 0,8 s contínuos, ou rosto perdido >= 2 s depois de olhos fechados >= 0,4 s | cabeça caída com os olhos fechados é perigo imediato. |
| **Cabeceio** | desce do neutro a >= 15° em <= 0,5 s e volta a < 10° em <= 1,5 s, com olhos fechados >= 0,4 s | o "pescar" de sono; uma piscada comum (0,1-0,3 s) no meio não basta. |
| Cabeça baixa com olhos abertos | >= 2 s abaixo, olhos abertos | é **distração**, não sono: aparece no registro, **0 ponto**. |
| Olhando para o lado | \|Δyaw\| > 25° anula o episódio | retrovisor/ombro não é queda. |
| Rosto some logo após baixar >= 10° | queda fora do quadro (até 4 s) **só se os olhos estavam fechados >= 0,4 s contínuos quando o rosto sumiu** | quem se abaixa para pegar algo sai do quadro de olhos **abertos**. Na v2.0, o fim de 2 clipes do NITYMED (vídeo virando tela preta com a cabeça baixa) virava "queda" e vermelho em 2 s. |
| Deriva da postura | perto do neutro (< 8°), a base acompanha devagar (constante de 2 min) | a postura muda ao longo da viagem. |

Limite honesto: o DMD não tem anotação de cabeceio, então **não há revocação medida** para a cabeça;
há a taxa de falso positivo (1 queda em 63,3 min de direção real, durante "pegar algo ao lado") e os
testes com sequências sintéticas.

## 2. Como o nível de sonolência (0-100) é calculado

Nível = soma dos pontos abaixo (máximo 100). Cada evento vale pontos no instante em que acontece e
**perde valor linearmente** até zerar no tempo de decaimento. A lista "Por que N" (tela "Em viagem") mostra exatamente
essas parcelas.

| Parcela | Pontos | Decaimento / teto | Por quê |
|---|---|---|---|
| PERCLOS (60 s, só fechamentos >= 0,5 s) | 0 até 7,5 %; **40 em 15 %**; mesma inclinação depois | teto 60 | 15 % (referência de projeto, mais conservadora que os 8 % a 12 % usuais) sozinho leva a ATENÇÃO, nunca a vermelho. |
| Fechamento de 0,5-1 s | 25 cada | some em 2 min; teto 60 (junto com microssonos) | a especificação pede "fechamentos longos por minuto"; 2 em pouco tempo = ATENÇÃO. |
| Microssono (depois de reabrir) | 25 cada | some em 2 min; teto 60 (junto com os de 0,5-1 s) | o efeito principal do microssono é o **vermelho imediato** enquanto os olhos estão fechados; depois ele conta como mais um fechamento longo, e o piso abaixo garante 1 min de ATENÇÃO. |
| Bocejo | 25 cada | some em 5 min; teto 50 | 1 bocejo não alarma ninguém; 2 bocejos em até ~2 min = ATENÇÃO. |
| Queda ou cabeceio | 30 cada | some em 5 min; teto 60 | 2 quedas = 60; com qualquer outro sinal, vermelho. |
| Piscadas mais lentas | 0 em 1,25x a duração típica da calibração, 20 em 2x | teto 20 (mín. 5 piscadas em 2 min) | piscadas alongam com o sono; peso pequeno por ser um sinal ruidoso. Nos vídeos reais nunca passou de 3 pontos (seção 9). |
| Modelo LSTM (experimental) | +10 | só com "sonolento" (p >= 0,6) em 5 predições seguidas **e** com os outros sinais já >= 40 | nunca muda a cor sozinho: reforça um quadro que já é de cansaço (seção 4). |
| Piso após microssono/cabeça caída | o que faltar para 40 | durante 1 min depois de terminar o vermelho forçado | quem acabou de cochilar não volta ao verde segundos depois de abrir os olhos. Aparece na lista "Por que N" com o tempo que falta. |

| Estado | Entra | Sai | O que acontece |
|---|---|---|---|
| ALERTA (verde) | nível < 40 | não se aplica | nada toca |
| ATENÇÃO (amarelo) | nível >= 40 | < 35 | um "ding" ao entrar e a cada novo sinal, **no máximo 1 a cada 20 s** |
| SONOLENTO (vermelho) | nível >= 70, ou **microssono**, ou **cabeça caída com olhos fechados** | < 60 se entrou pelo nível; < 70 se foi forçado (microssono/cabeça) e a condição acabou | **sirene** de dois tons enquanto durar (mínimo 2 s), tela inteira piscando, vibração (Android) |

- **Nenhum sinal sozinho chega a 70** (tetos de 60 ou menos): o vermelho por nível exige sinais
  combinados. Só microssono e cabeça caída com olhos fechados forçam o vermelho (nível exibido >= 75).
- O nível exibido **sobe na hora** e **desce no máximo 3 pontos por segundo**; a histerese (40/35 e
  70/60) evita o pisca-pisca entre estados. A margem de saída do vermelho (60) vale para o vermelho que
  **entrou pelo nível**, que é onde o pisca-pisca acontece; quando o vermelho foi forçado, o nível não
  estava perto do limiar, e o estado volta a seguir o nível com o limiar de entrada (70) assim que a
  condição acaba. O número exibido é arredondado para baixo (69,6 aparece como 69, nunca "70" num cartão
  amarelo).
- Exemplos (dos testes em `test_engine.mjs`): 1 microssono -> vermelho na hora; 1,6 s depois de reabrir
  os olhos passa a amarelo, fica em ATENÇÃO pelo piso de 1 min e 68 s depois está verde (nível 16).
  2 bocejos -> ATENÇÃO (nível 48). 2 bocejos + 1 cabeceio -> SONOLENTO pelo nível. **O roteiro da banca**
  (2 bocejos, cabeça baixa, olhos fechados 2 s): vermelho com sirene no microssono e, ao reabrir, amarelo
  em 1,7 s (teste [23]); na v2.0 o mesmo roteiro ficava ~1,5 min no vermelho com a sirene tocando.
- **"Estou acordado"** (deslize na tela do alarme) corta a sirene na hora e silencia o vermelho **por nível** por
  30 s (se o nível continuar alto, ela volta); o vermelho forçado (microssono, cabeça caída) nunca é
  silenciado.

## 3. Som e volume do alarme

- **Vermelho:** sirene hi-lo 880/1250 Hz (troca a cada 0,35 s, onda quadrada), repetida enquanto a
  condição durar, no mínimo 2 s. **Amarelo:** "ding-dong" curto. Tela inteira piscando no vermelho
  (<= 1,25 Hz, abaixo do limite de 3 flashes/s; com "reduzir movimento" vira um véu fixo).
- **iPhone:** o som só é liberado por um **toque** (regra do WebKit). O app libera no toque em
  "Iniciar viagem", "Testar som" ou "Recalibrar" (cria/retoma o AudioContext dentro do toque e toca um
  buffer silencioso) e define `navigator.audioSession.type = "playback"` antes de criar o AudioContext,
  o que faz o alarme tocar **mesmo com a chave de silencioso ligada** (Audio Session API do WebKit,
  iOS 16.4+, exposta também ao WKWebView que o Chrome do iOS usa; falta confirmar no aparelho, e "Detalhes
  técnicos > Modelos > Áudio" mostra se a API existe e o estado do som). Sem essa API (iOS antigo), um
  `<audio>` silencioso em loop faz o mesmo papel. Se o sistema pausar o áudio (ligação, app em segundo
  plano), qualquer toque na tela reativa.
- **Controle "Volume do alarme" (0-100 %), na folha "Som do alarme"** (botão "Som" da viagem ou Ajustes >
  Volume do alarme), **junto do "Testar som".** **Padrão: 20 %**, baixo de
  propósito, para testar perto de outras pessoas sem susto. O valor escolhido fica salvo no aparelho
  (`localStorage`, chave `sonolencia.volumeAlarme`, com try/catch: em modo privado vale o padrão). Ganho =
  (volume/100)² (curva quadrática, porque a percepção de volume é aproximadamente logarítmica):
  20 % = -28 dB em relação a 100 %; 50 % = -12 dB. Em 0 % o alarme **não toca** (a tela ainda pisca) e o
  app avisa.
- **Na demonstração para a banca: suba o controle para 100 % e o volume de mídia do aparelho para o
  máximo** e toque em "Testar som" antes de começar (seção 6). Depois da banca, se quiser, volte a 20 %.
- `?volume=0..100` na URL sobrepõe o valor salvo **sem gravar**. Os testes automáticos usam
  `?volume=0` e o Chrome com `--mute-audio`, e abortam se o volume ou o ganho saírem de 0: nenhum teste
  toca som no computador.
- Limitações: o app **não consegue aumentar o volume de mídia** do aparelho (só o próprio ganho, até
  100 %); no iPhone **não há vibração** para páginas web (`navigator.vibrate` não existe no iOS); ao
  liberar o som no modo "playback", o iOS **pausa música de outro app** que esteja tocando.

## 4. Modelos de aprendizado de máquina (experimental)

- **LSTM de piscadas** (`blink_temporal.tflite`, Exp. 1): acurácia por vídeo de 57,2 % em 3 classes
  (`results/exp1_intra.md`; acaso 33 %), com características que, no celular, vêm de uma calibração
  curta (20 piscadas) e ficam fora da distribuição do treino. Na v1 ele podia disparar o vermelho
  sozinho ("sonolento" por 10 s). Na v2.1 aparece pequeno, como "estimativa do modelo (experimental)", e
  soma **no máximo +10 pontos**, só com "sonolento" (p >= 0,6) em 5 predições seguidas **e com os sinais
  visíveis já em ATENÇÃO (>= 40)**. Motivo da trava nova: numa pessoa acordada do UTA-RLDD (Fold3/31,
  vídeo alerta) o LSTM dizia "sonolento" em 50 % dos quadros e os +10 dele produziram 3 avisos sozinhos
  (seção 9). O cartão do modelo mostra "sonolento, mas não soma" quando ele é elegível e a trava segura.
- **CNN dos olhos** (`eye_state_*.tflite`, Exp. 3): mostra p(aberto) de cada olho; **não entra no
  nível**.
- Motivo: a generalização de modelos treinados para uma pessoa/condição nova é limitada (Exp. 5b,
  `results/exp5b_transferencia.md`). Os sinais interpretáveis (olhos, bocejo, cabeça) é que decidem.

## 5. Calibração

1. **Fase 1: 10 s olhando para a frente** (barra de progresso no vídeo): "Apoie o celular como no
   painel do carro, olhe para a frente com os olhos abertos e a boca fechada. Pode piscar
   normalmente." Mede, por mediana (que as piscadas não deslocam): limiar do EAR, eyeBlink de repouso, jawOpen
   e MAR de repouso e a pose neutra (pitch/yaw/roll). O tempo só conta com o rosto visível.
2. Avisos ao fim da fase 1 (na tela, com o pedido de **Recalibrar**): EAR de olho aberto muito baixo;
   rosto muito de lado; sem pose; e, na v2.1, **olhos fechados em mais de 10 % dos quadros** (os dois
   eyeBlink >= 0,5). Acordado, 15-20 piscadas por minuto de 0,1-0,3 s dão 2-10 % do tempo com o olho
   fechado; acima disso a linha de base sairia com a pessoa já sonolenta (no UTA-RLDD Fold3/31, o limiar
   do EAR calibrado no vídeo sonolento foi 0,214, contra 0,262 no alerta).
3. **Fase 2: 20 piscadas espontâneas** (em segundo plano, o detector já funciona): duração típica da
   piscada (base da parcela "piscadas mais lentas") e o z-score das 4 características do LSTM.
4. **Recalibrar** reinicia tudo (útil ao trocar de pessoa ou de posição do celular).

## 6. Roteiro da demonstração para a banca (2 min)

**Antes de entrar na sala (fora do cronômetro, ~1 min):**
- Celular no suporte, **na altura do painel** (câmera um pouco abaixo do rosto), luz de frente, sem
  contraluz. Mesma rede do computador (ou "Acesso Pessoal" do iPhone).
- Abra o app e toque em **Iniciar viagem** (esse toque libera o som no iPhone).
- **Volume:** o app começa com o "Volume do alarme" em **20 %** (padrão, para testes sem susto). Para a
  banca, **suba o controle para 100 %**, deixe o volume de mídia do celular no máximo e toque em
  **Testar som** (ding + 1,6 s de sirene). Ative "Manter tela ligada" (se o aparelho permitir) ou
  desligue o bloqueio automático da tela.
- Toque em **Recalibrar** logo antes de começar (10 s olhando para a frente): nível 0 e registro limpo.

**Na frente da banca (a ordem importa: os sinais acumulam no nível):**

| Tempo | Faça | O app mostra |
|---|---|---|
| 0:00-0:15 | **Pisque normalmente**, olhando para a frente. | **Verde** (Alerta), nível 0. Os três sinais (Olhos, Bocejos, Cabeça) ao vivo; sem lista "Por que": piscadas normais não sobem o nível (o PERCLOS só conta fechamentos de 0,5 s ou mais). |
| 0:15-0:45 | **Boceje duas vezes**, boca bem aberta por ~2 s, **de olhos abertos**, com ~15 s entre os bocejos. | Linha Bocejos: "1 em 5 min" (continua verde, nível ~23); no 2º, **"2 em 5 min" e Atenção (amarelo) com "ding"** e o cartão "Hora de uma pausa" (nível ~48). |
| 0:45-1:00 | **Baixe a cabeça ~30°** por 3-4 s, **mantendo os olhos abertos** (olhe para a banca), e volte. | Registro: **"cabeça baixa com os olhos abertos (… para baixo): distração, 0 ponto"**. Sem alarme. Explique: com os olhos fechados seria "cabeça caindo" e vermelho na hora. |
| 1:00-1:05 | **Feche os olhos por 2 s.** | Com 1 s: **tela de alarme vermelha + sirene**, "Acorde!" e "Olhos fechados há 1,x s (microssono)". |
| 1:05-1:15 | **Reabra os olhos.** | A sirene para e a tela **volta ao amarelo em ~2 s** (1,7 s no teste [23]); o registro atualiza "microssono 2,x s". O app fica pelo menos 1 min em ATENÇÃO: quem acabou de cochilar não volta direto ao verde. |
| 1:15-2:00 | Mostre a lista **"Por que N"** e o **registro de eventos** (em "Detalhes técnicos"). | "2 bocejos nos últimos 5 min" (cerca de +40) e "1 microssono em 2 min" (cerca de +20; teste [23]): cada ponto do nível tem origem. Registro com horário: bocejo, bocejo, aviso, cabeça baixa, microssono, alarme. Se sobrar tempo: "Detalhes técnicos" (EAR, eyeBlink, jawOpen, pitch/yaw/roll, latência por estágio) e o LSTM como estimativa experimental. |

Cuidados:
- **Olhos abertos ao bocejar e ao baixar a cabeça.** Bocejar de olhos fechados por 1 s ou mais é
  microssono para o app (olho fechado é olho fechado) e antecipa o vermelho.
- Deixe o passo da cabeça **entre** o 2º bocejo e o fechamento dos olhos: com o 2º bocejo só 3 s antes
  de fechar os olhos (e os bocejos a 5 s um do outro), o nível passa de 70 e o vermelho dura 8,1 s depois
  de reabrir (pior caso do teste [25]); com 15 s ou mais entre eles, ~2 s.
- Para repetir do zero (nível 0, sem eventos), toque em **Recalibrar** (10 s).
- Planos B: sem som -> toque em "Testar som" de novo (o iOS pode ter pausado o áudio) e confira o volume
  do alarme e o de mídia; rosto não detectado -> afaste-se da janela/luz de fundo; sem rede ->
  "Acesso Pessoal" do iPhone. O celular não precisa de internet se `app/pwa/vendor/` estiver completa
  (seção 7); confira em "Detalhes técnicos > Modelos > Bibliotecas": "local (./vendor/)".

Perguntas prováveis da banca (respostas curtas e honestas, seção 8 e 10): bocejo com a mão na frente
quase não é detectado; óculos e noite prejudicam os olhos; 1 motorista acordado em 17 do DMD teve alarme
falso (olhar para baixo que o blendshape lê como olho fechado); a cabeça não tem revocação medida; o
LSTM não decide nada sozinho porque não generaliza bem para uma pessoa nova (Exp. 5b).

## 7. Como rodar

1. Os modelos já estão em `app/pwa/models/` (`eye_state_int8.tflite`, `eye_state_fp32.tflite`,
   `blink_temporal.tflite` + JSON). Sem eles o app roda em **modo degradado**: o detector (olhos,
   bocejo, cabeça) funciona; só os modelos experimentais ficam desligados, com aviso na tela.
2. Uma vez, com internet: `bash tools/baixar_dependencias_app.sh`. Ele baixa para `app/pwa/vendor/`
   (~36 MB, fora do Git) o MediaPipe Tasks Vision com os WASM, o `face_landmarker.task` e o
   TensorFlow.js/tfjs-tflite, nas mesmas versões do `app.js`, e confere cada arquivo pelo SHA-256
   (`--conferir` só confere). O app usa essa cópia primeiro e a CDN como reserva: com a pasta completa,
   o celular não precisa de internet (na PUC, sem dados móveis, basta estar na rede do computador).
3. No computador, na raiz do repositório, com o celular **na mesma rede**:
   `python3 app/pwa/serve_https.py` (porta 8443; gera o certificado autoassinado em `app/.pwa-cert/`,
   fora da pasta servida). Ao subir, ele confere `vendor/`; se faltar algo, avisa e oferece baixar.
4. No celular, abra `https://<IP-do-computador>:8443/`, aceite o aviso do certificado e permita a
   câmera. Toque em **Iniciar viagem**.
5. Alternativa com HTTPS válido: GitHub Pages (copie `app/pwa/` com `models/` para `docs/` e ative
   Pages em `/docs`; sem `vendor/`, as dependências vêm da CDN).

Parâmetros de URL: `?autostart=1` liga a câmera ao abrir; `&bench=1` inicia o benchmark de 60 s ao
ligar; `&volume=0..100` define o volume do alarme só nesta visita (não grava).

**Benchmark 60 s** (em "Detalhes técnicos"): grava por quadro as latências, os sinais (EAR, eyeBlink,
jawOpen, pose), o PERCLOS (e o PERCLOS com as piscadas, informativo), o nível, o estado, o estado de
cada sinal, se o LSTM somou, o piso após microssono e os eventos; o resumo traz a contagem de cada tipo
de evento, o tempo em cada nível, o número de alarmes/avisos e o volume usado. O resultado vai
**sozinho** para o computador (`POST /api/benchmark` do `serve_https.py` -> `results/phone/`); os botões
de download continuam como reserva. Importação: `python tools/import_phone_benchmark.py results/phone`.
**Exportar sessão** salva a calibração, a configuração (`CFG`), o registro de eventos, as piscadas e as
predições do LSTM.

### 7.1 Envio automático do benchmark e nome do aparelho (acrescentado em 27/09/2026)

- **Envio:** ao fim do "Benchmark 60 s" o app faz `POST /api/benchmark` para o próprio `serve_https.py`, que grava
  `results/phone/benchmark_pwa_<data>.json` (resumo) e `results/phone/benchmark_pwa_<data>_frames.csv` (por quadro, só
  números). A tela mostra "enviado para o computador: ..." ou, se falhar (ex.: app servido pelo GitHub Pages), pede para
  usar os botões de download e copiar os dois arquivos para `results/phone/`.
- **Nome do aparelho:** no Chrome do Android o app lê os *User-Agent Client Hints*
  (`navigator.userAgentData.getHighEntropyValues(["model", "platformVersion"])`); o importador traduz o código Samsung
  (`SM-S918*` = Galaxy S23 Ultra, `SM-S916*` = S23+, `SM-S911*` = S23). O iOS não informa o modelo ao navegador: o nome
  vem de `results/phone/aparelhos.json` (`[{"contem": "<trecho do user-agent ou do nome do arquivo>", "aparelho": "<nome>"}]`;
  hoje: `iPhone OS 26_6_1` -> "iPhone 17 Pro (512 GB)"). Para outro iPhone, acrescente uma linha antes de importar.
- **Depois do teste:** `bash tools/atualizar_celular.sh` importa `results/phone/`, refaz a figura do Exp. 4, os números
  do artigo e do pôster (conferidos) e recompila os dois. Na PUC a rede isola os aparelhos: ponha o Mac e o celular no
  Acesso Pessoal (hotspot) do iPhone. Passo a passo: `docs/GUIA_EQUIPE.md`, seção 1.
- **O que já está em `results/phone/`:** duas execuções no iPhone 17 Pro com a **v1.0** (21:58Z e 22:41Z de 27/09;
  `results/exp4_phone_latency.md`), válidas só para o custo computacional. Na de 22:41Z a v1 deu falso alarme por
  PERCLOS com o usuário acordado, um dos motivos da reescrita. A v2.1 ainda não foi medida no iPhone nem no Galaxy.

## 8. Validação (números de execuções reais, com a origem)

**Testes do motor:** `node app/pwa/test_engine.mjs` -> **217/217** (`results/logs/pwa_test_engine_20260927-231003.log`): as 64 asserções
da v1, inalteradas, e as do detector v2/v2.1: pose a partir de matrizes sintéticas nas duas ordens,
fusão dos olhos e travas, fechamentos, PERCLOS por tempo e só com fechamentos >= 0,5 s, bocejo, cabeça
caindo/cabeceio/lado/rosto perdido (com e sem olhos fechados), pontos com decaimento, nível com
histerese e saída do vermelho forçado, piso de 1 min em ATENÇÃO, alarme sem áudio, conselheiro do LSTM
com a trava de 40, monitor completo com quadros sintéticos (inclusive quem pisca 75 vezes por minuto), a
regra do volume, o roteiro de 2 min da banca (inclusive o pior caso de tempo) e casos difíceis. Asserções trocadas de propósito
no ajuste final (regra mudou; explicação no cabeçalho do arquivo): [16] rosto perdido com olhos abertos
não é mais queda; [21] S2 (1 min de ATENÇÃO após o microssono) e S7 (LSTM só soma com os sinais
visíveis >= 40); [23] roteiro da banca.

**Motor contra a anotação do DMD** (offline, mesmo `face_landmarker.task` da PWA, o `engine.js` do app
importado sem cópia; calibração = 10 s iniciais de cada vídeo; `results/app_validacao_dmd.md`, log
`results/logs/app_validacao_dmd_20260927-225227.log`). s5 = sonolência **encenada** (16 sessões, 46,6 min); s1 = **direção real** com motoristas
acordados (17 sessões, 63,3 min). "v2.0" = log `results/logs/app_validacao_dmd_20260927-212228.log`.

| Medida | v2.1 (final) | v2.0 | Referência |
|---|---|---|---|
| Olho fechado por quadro (s5): sensibilidade / especificidade / kappa | 0,918 / 0,997 / 0,938 | igual (regra dos olhos não mudou) | EAR sozinho (v1): 0,994 / 0,949 / 0,831 |
| Fechamentos >= 0,5 s (s5): revocação / precisão (um-para-um) | 0,929 / 0,786 (198 anotados) | igual | sem referência |
| Fechamentos >= 1 s (s5): revocação / precisão (um-para-um) | 0,920 / 0,748 (113 anotados) | igual | sem referência |
| Microssonos (s5) / fechamentos anotados >= 1 s que dispararam o vermelho | 134 / 104 de 113 | igual | com 1,2 s: 95 de 113 |
| Bocejos (s5): revocação sem mão / com mão / precisão | 0,638 / 0,026 / 1,000 (47 sem mão, 38 com mão) | igual | v1 (MAR > 0,6 por 2 s): 0,319 / 0,000 / 1,000 |
| Bocejos falsos na direção real (s1) | 0 por hora (1,5 s) | igual | 1,9 por hora com 1,0 s |
| Direção real (s1): % do tempo em vermelho (sirene) | **3,3 %** | 5,8 % | v1 (sem o LSTM): 55,6 % |
| Direção real (s1): % do tempo em amarelo / avisos por hora | 4,1 % / 1,90 | 2,2 % / 1,90 | sem referência |
| Direção real (s1): entradas no vermelho (alarmes) | 4 em 63 min (3,79/h), **todas do mesmo motorista** (gA_5); 16 de 17 sessões sem alarme | 1 (0,95/h), o mesmo motorista, que ficou 97 % do tempo no vermelho | v1: 49,6/h |
| Direção real (s1): PERCLOS p95 (após 60 s) | **1,2 %** | 10,0 % | referência do app: 15 % |
| Sonolência encenada (s5): % do tempo em vermelho / sessões com alarme | 68,6 % / 16 de 16 | 74,4 % / 16 de 16 | sem referência |
| Sinal do pitch (33 vídeos) | correlação > 0 em 33 de 33 (mediana 0,857) | igual | sem referência |

Leitura honesta: o vermelho da direção real vem **todo de uma sessão** (sujeito gA_5): o app marca os
olhos dele como fechados em 16,6 % dos quadros (15,1 % pelo caminho A, o blendshape), contra 3,2 % na
mediana das outras 16 sessões, e ele olha bastante para baixo (eyeLookDown mediano 0,58 x 0,34). Na v2.0 ele ficava 97 % do
tempo no vermelho (1 alarme contínuo); na v2.1, 56 % no vermelho e 33 % no amarelo, com 4 entradas no
vermelho (o estado volta ao amarelo entre os "microssonos"). Continua um alarme falso: não há correção
barata, porque nos quadros anotados do DMD o olho **fechado** também tem eyeLookDown alto (mediana
0,77; aberto: 0,26). O DMD s5 é sonolência **encenada**; não mede sonolência real.

**PWA no Chrome headless com vídeos reais como câmera** (Mac, Chrome 154, SEMPRE mudo: `--mute-audio`
+ `?volume=0` + política de autoplay padrão; cópia congelada da v2.1 servida na porta 8769;
`results/app_validacao.md`, `results/app_validacao.csv`, log `results/logs/app_validacao_20260927-232519.log`):

| Vídeo (3 min após calibrar; NITYMED: clipe inteiro) | Base | v2.1: verde / amarelo / vermelho | Alarmes · avisos | v2.0: verde / amarelo / vermelho | Alarmes · avisos |
|---|---|---|---|---|---|
| Fold1/09, vídeo 0 | UTA-RLDD alerta | 100 / 0 / 0 % | 0 · 0 | 100 / 0 / 0 % | 0 · 0 |
| Fold3/31, vídeo 0 (pisca 76 vezes/min) | UTA-RLDD alerta | **100 / 0 / 0 %** | **0 · 0** | 72,8 / 27,2 / 0 % | 0 · 3 |
| Fold5/51, vídeo 0 | UTA-RLDD alerta | 100 / 0 / 0 % | 0 · 0 | 100 / 0 / 0 % | 0 · 0 |
| gE_29 (óculos) | DMD s5 encenada | 25,9 / 1,9 / 72,2 % | 2 · 0 | 25,9 / 0 / 74,1 % | 1 · 0 |
| gZ_33 | DMD s5 encenada | 17,0 / 4,2 / 78,8 % | 1 · 1 | 17,0 / 4,2 / 78,8 % | 1 · 1 |
| gC_14 | DMD s5 encenada | 27,5 / 35,8 / **36,7 %** | 6 · 0 | 27,5 / 0 / 72,5 % | 1 · 0 |
| gB_7 | DMD s5 encenada | 20,9 / 0,9 / 78,2 % | 2 · 1 | 20,8 / 0,3 / 78,9 % | 1 · 1 |
| Fold1/09, vídeo 10 | UTA-RLDD sonolento | 100 / 0 / 0 % | 0 · 0 | 100 / 0 / 0 % | 0 · 0 |
| Fold3/31, vídeo 10 | UTA-RLDD sonolento | 6,4 / 0 / 93,5 % | 1 · 1 | 6,5 / 0 / 93,5 % | 1 · 1 |
| Fold5/51, vídeo 10 | UTA-RLDD sonolento | 100 / 0 / 0 % | 0 · 0 | 100 / 0 / 0 % | 0 · 0 |
| Yawning/Female P1042778 | NITYMED bocejo (noite) | 66,1 / 33,9 / 0 % | 0 · 1 | 66,1 / 33,9 / 0 % | 0 · 1 |
| Yawning/Male P1042760 | NITYMED bocejo (noite) | 32,8 / 23,9 / 43,3 % | 2 · 0 | 32,7 / 0 / 67,3 % | 1 · 0 |
| Microsleep/Male P1042751 | NITYMED microssono (noite) | 100 / 0 / 0 % | 0 · 0 | 65,0 / 35,0 / 0 % | 0 · 1 |

- **Acordado (caso 1):** 0 alarmes e 0 avisos nos 3 vídeos alerta do UTA-RLDD (9 min), com PERCLOS de 0 %
  (com as piscadas, como na v2.0, seria 5,3 %, 14,4 % e 5,0 %) e o +10 do LSTM em 0 % dos quadros (v2.0: 50 %
  no Fold3/31). 0 alarmes em 0,15 h não prova taxa zero (limite superior de 95 % de ~20 por hora); a referência
  maior continua sendo a direção real do DMD acima.
- **Sonolência encenada (caso 2), 4 sessões do DMD s5 no navegador:** olho fechado por quadro com sensibilidade
  0,858, especificidade 0,998 e kappa 0,902 (v2.0: 0,856 / 0,998 / 0,903); fechamentos >= 1 s com revocação
  31/34 e precisão 31/42 (iguais); bocejos 6/24 (6/14 sem a mão, 0/10 com a mão) e nenhum falso (igual); os
  mesmos 42 microssonos, cada um com vermelho imediato; primeiro vermelho de 36 a 48 s após calibrar (igual).
  **Custo:** na gC_14 o tempo em vermelho caiu de 72,5 % para 36,7 %: depois de 3 microssonos em 12 s o nível
  parou em 69 (fechamentos no teto de 60 + PERCLOS de 9 %) e o estado ficou amarelo até o microssono seguinte.
- **Cabeça e rosto perdido:** as 2 "quedas (saiu do quadro)" do fim dos clipes do NITYMED sumiram (a do
  microssono virou "cabeça baixa com os olhos abertos (19° para baixo): distração, 0 ponto"); o teste
  sintético deu o mesmo resultado da v2.0 e o texto mostra "(39° para o lado)".
- **Calibração:** o aviso de olhos fechados apareceu só no Fold3/31 sonolento (23 % da calibração).
- 60 quadros por segundo, 0 exceções de JavaScript e silêncio conferido a cada 1 s nas 14 execuções.
- **Interface** (`results/logs/pwa_headless_ui_v21_20260927-232706.log`): sem rolagem horizontal em 320, 390 e
  1280 px; um clique **real** (gesto do DevTools, sem a flag de autoplay) em "Testar som" deixa o AudioContext
  em `running`, com ganho final 0 (`?volume=0`); 0 exceções; modelos carregados.

**iPhone 17 Pro (iOS 26.6.1, Chrome 154/WebKit), app 1.0, 27/09/2026** (`results/phone/`, `results/exp4_phone_latency.md`):
2 execuções de 60 s, com tempo total por quadro (p50) de 11 e 23 ms, mediana de 17 ms, quase todo no Face Landmarker.
O app 2.1 ainda **não** foi medido no iPhone nem no Galaxy S23 Ultra (pendência até 09/11: rodar "Benchmark 60 s" com a
versão nova nos dois aparelhos).

### 8.1 Teste independente em sono real (UTA-RLDD, 26/10/2026)

Tudo o que está acima foi medido nos mesmos dados em que as regras foram escolhidas. Este teste não teve ajuste
(`results/app_validacao_uta.md`; `tools/app_validacao/validar_motor_uta.mjs`; log
`results/logs/app_validacao_uta_20261026-230247.log`).

- Amostra, com a regra fixada antes de rodar: todas as 24 pessoas dos folds oficiais 2 e 4 da UTA-RLDD, que não
  entraram em nenhuma etapa do ajuste nem no teste no navegador. De cada uma, o vídeo acordado (KSS de 1 a 3) e o
  sonolento (KSS 8 e 9), com cerca de 10 min cada: 48 vídeos e 8,2 h avaliadas.
- O que roda: o `engine.js` do app, importado sem cópia, com a configuração final e sem o LSTM, sobre os sinais do
  mesmo Face Landmarker da PWA (`tools/app_validacao/extrair_sinais_app.py --uta-folds 2 4 --uta-rotulos 0 10`, em
  `data/signals/app_tasks/uta/`). A calibração é a do app, nos 10 s iniciais de cada vídeo; numa variante, o vídeo
  sonolento usa a calibração do vídeo acordado da mesma pessoa.

| Medida | Calibração no próprio vídeo | Calibração do vídeo acordado |
|---|---|---|
| Acordados com algum alarme | 2 de 24 (0,46% do tempo em vermelho; 3 alarmes em 4,0 h) | igual |
| Acordados que chegam ao amarelo | 5 de 24 | igual |
| Sonolentos que chegam ao amarelo / ao vermelho | 17 / 15 de 24 | 19 / 17 de 24 |
| Pessoas com mais tempo fora do verde no vídeo sonolento (mais / igual / menos) | 16 / 7 / 1 | 19 / 5 / 0 |
| AUC do % do tempo fora do verde, sonolento × acordado [IC 95%] | 0,812 [0,705; 0,918] | 0,855 [0,760; 0,948] |

Leitura: com o sono declarado pela própria pessoa, o app alarmou 15 dos 24 sonolentos e 2 dos 24 acordados. Ele reage a
sinais visíveis, não ao sono declarado: 7 sonolentos ficaram verdes o vídeo inteiro, e em 6 deles o app não viu nenhum
fechamento de 0,5 s ou mais, bocejo ou queda da cabeça. Calibrar já com sono atrapalha: em 7 dos 24 vídeos sonolentos,
os olhos ficaram fechados em mais de 10% da calibração, e em 8 houve algum aviso (o app avisa e pede para recalibrar).

Nas mesmas 24 pessoas, a AUC por vídeo dos modelos do Exp. 5b e da regra do rótulo (`results/app_comparacao_uta.md`,
`src/experiments/comparar_app_uta.py`): RF treinado no DMD, 0,811; regra do rótulo do DMD, 0,853; RF treinado na própria
UTA-RLDD, 0,953 com a centragem pelo vídeo acordado e 0,806 sem ela. O app (0,812), sem treino nessa base, fica no nível
deles; o RF da UTA-RLDD só passa à frente com o atalho da centragem discutido no Exp. 5b.

Ressalva: essas pessoas estavam sentadas diante da câmera, sem dirigir. Na direção há movimento, luz que muda e
olhares para os espelhos, e o falso alarme em movimento continua medido só pela direção real do DMD (acima). O teste
roda no Mac, com o código do motor do app, e não no celular.

## 9. Ajuste final (27/09/2026): o que mudou, por quê e o efeito medido

Ponto de partida: a validação da v2.0 com vídeos reais (`results/app_validacao.md`, seção "Antes do
ajuste") e a verificação do código. Regra seguida: **nenhum limiar escolhido olhando só os vídeos usados
para medir**. Cada mudança tem um princípio (literatura, física do sinal ou consistência com uma regra
que já existia) e o efeito foi medido em dados que não a motivaram: a direção real do DMD (s1, 17
motoristas acordados) para falso alarme e o DMD s5 (16 sessões com anotação) para detecção, na tabela
"Ajuste final" de `results/app_validacao_dmd.md` (`results/app_validacao_dmd_ajuste_final.csv`). Depois,
os mesmos clipes da validação anterior rodaram de novo no navegador (seção 8).

**Aplicado:**

| # | Mudança | Princípio | Efeito medido |
|---|---|---|---|
| 1 | **LSTM só soma +10 com os sinais visíveis >= 40** | um modelo com 57 % de acurácia por vídeo e características fora da distribuição não pode ser o que muda a cor; a especificação já dizia "sem disparar alarme sozinho" | no navegador, UTA-RLDD Fold3/31 alerta: 27,2 % do tempo em amarelo e 3 avisos -> 0 % e 0 (o +10 entrava em 50 % dos quadros; agora em 0 %). Nas sessões encenadas ele ainda soma quando os sinais visíveis já passam de 40 (gB_7: 22 % dos quadros). Testes [21] S7/S7b e [25] |
| 2 | **PERCLOS só com fechamentos >= 0,5 s** (`perclos.minEpisodeMs = longMs`) | o PERCLOS mede fechamentos lentos, não piscadas (Dinges e Grace, 1998); 0,5 s é o limite que o próprio app já usava entre piscada e fechamento longo (uma piscada dura 0,1-0,4 s) | DMD s1 (só esta mudança): PERCLOS p95 10,0 % -> 1,2 % e tempo em vermelho 5,1 % -> 3,3 %; DMD s5: os mesmos microssonos e bocejos. No navegador, vídeos alerta do UTA: PERCLOS máximo de 5,3 / 14,4 / 5,0 % (com piscadas) -> 0 / 0 / 0 %. Teste [25]: 75 piscadas/min por 3 min = sempre ALERTA |
| 3 | **Rosto que some só vira queda com olhos fechados >= 0,4 s contínuos naquele instante** (`head.lostEyesMs`) e o rosto perdido deixou de valer como sono por si só | a mesma evidência de sono (`eyesMs`) que a queda com o rosto visível já exigia; quem se abaixa sai do quadro de olhos abertos | no navegador, as 2 "quedas (saiu do quadro)" do fim dos clipes do NITYMED (uma com vermelho forçado) sumiram; a do microssono virou "cabeça baixa com os olhos abertos" (0 ponto). DMD s1: 1 queda em 63 min, como antes. Testes [16] e [25] |
| 4 | **Texto do evento com a direção**: "cabeça caindo … (28° para baixo)" ou "cabeça tombando … (40° para o lado)" | na v2.0 a queda pela inclinação lateral aparecia como "(0° abaixo)" | teste [25] |
| 5 | **Aviso de calibração com olhos fechados em > 10 % dos quadros** (`calib.maxClosedFrac`) | acordado, piscadas dão 2-10 % do tempo com o olho fechado; acima disso a base sai sonolenta | no navegador, o aviso apareceu só no UTA-RLDD Fold3/31 sonolento (olhos fechados em 23 % da calibração); nenhum nos vídeos alerta nem no DMD. Teste [25] |
| 6 | **Microssono: 25 pontos por 2 min depois de reabrir (antes 40 por 5 min), piso de 1 min em ATENÇÃO e saída do vermelho forçado pelo limiar de entrada (70)** | o vermelho do microssono é "enquanto a condição durar" (especificação A); depois, a especificação E pede "fechamentos longos por minuto"; a margem de saída (60) serve contra o pisca-pisca do nível, e o forçado não estava perto do limiar | roteiro da banca (teste [23]): ao reabrir, amarelo em 1,7 s (v2.0: vermelho com sirene por ~1,5 min). v2.0 -> v2.1 com todas as mudanças juntas: direção real (s1) 5,8 % -> 3,3 % do tempo em vermelho; s5 74,4 % -> 68,6 %, com os mesmos 134 microssonos, cada um com vermelho imediato. Como agora o estado volta ao amarelo entre microssonos, as **entradas** no vermelho aumentaram (s1: 1 -> 4, todas do gA_5; s5: 17 -> 34): alarmes/h não é comparável entre as versões. **Custo** no navegador: gC_14 (DMD s5) 72,5 % -> 36,7 % do tempo em vermelho, com os 7 microssonos disparando o vermelho na hora |

**Avaliado e não aplicado** (tabela "Ajuste final" de `results/app_validacao_dmd.md`):

| Proposta | Resultado | Decisão |
|---|---|---|
| Histerese no eyeBlink (fecha em 0,5, reabre abaixo de 0,4) | s5: kappa 0,938 -> 0,946 e 2 microssonos anotados a mais; **s1: fechamentos de 0,5-1 s 15 -> 46, avisos 1,9 -> 17,1 por hora, 15 % do tempo em amarelo** (com 0,05: 29 e 7,6/h). Olhar para baixo depois de uma piscada deixa o eyeBlink entre 0,4 e 0,5 e o "fechado" se estende | rejeitada: aumenta o falso alarme em quem está acordado. Fica no código como opção desligada (`reopenHyst: 0`). Exigir "2 quadros seguidos" não muda o nível por construção: fechamentos de 1 quadro (< 50 ms) já não contam como piscada, nem no PERCLOS novo (>= 0,5 s), nem na cabeça (>= 0,4 s) |
| Microssono a partir de 1,2 s | s5: fechamentos anotados >= 1 s que disparam o vermelho **104 -> 95 de 113**; s1: os mesmos 8 microssonos | rejeitado: perde detecção e não reduz o falso alarme em quem está acordado. O viés de +0,17 s vem da definição (meio fechamento, como a duração de piscada a meia amplitude), não de erro |
| Bocejo com 1,0 s em vez de 1,5 s | s5: revocação sem mão 0,638 -> 0,723; **s1: 0 -> 2 bocejos falsos (1,9 por hora)** e % em vermelho 3,3 -> 5,0 | rejeitado pelo objetivo do ajuste (menos falso alarme). Os bocejos perdidos são encenados e curtos; na demonstração, boceje de verdade (~2 s) |
| Peso da duração das piscadas (não passou de 3 pontos em nenhum vídeo, na v2.0 e na v2.1) | a 30-60 FPS, 1 quadro (17-33 ms) é 15-30 % de uma piscada de ~110 ms: um limiar menor que 1,25x ficaria abaixo da resolução | mantido como está (peso pequeno, documentado); não é retirado para seguir a especificação E |

## 10. Limites conhecidos

- Óculos escuros, reflexo nas lentes e pouca luz prejudicam os olhos (Exp. 3; NITYMED no Exp. 6). No
  motorista de óculos do DMD (gE_29), o eyeBlink de um olho fica em 0,4-0,5 com o olho fechado e a trava
  de "olhar para baixo" bloqueia o EAR: o app mede os fechamentos dele mais curtos que a anotação (mediana
  de -0,57 s; seção "Duração dos fechamentos" de `results/app_validacao_dmd.md`).
- Bocejo com a mão na frente da boca quase nunca é detectado; bocejos curtos (< 1,5 s de boca bem
  aberta), como os encenados, também não.
- Cabeça: sem revocação medida (o DMD não anota cabeceio); só falso positivo e testes sintéticos. A
  cabeça caindo para a frente não se simula transformando o vídeo (`results/app_validacao.md`, item 6 da
  leitura da v2.0).
- Um motorista acordado em 17 teve alarme falso persistente (seção 8).
- Depois de microssonos espaçados, o estado volta ao amarelo se os outros sinais não chegam a 70: na sessão
  encenada gC_14, 36,7 % do tempo em vermelho, contra 72,5 % na v2.0 (cada microssono continua disparando o
  vermelho na hora). É o preço de o app "voltar" em segundos depois que a pessoa abre os olhos.
- Pessoas que parecem acordadas mas relatam sono (UTA-RLDD, rótulo por autoavaliação) não mostram sinais
  visíveis e não são acusadas: o app mede sinais visíveis, não "sono" em geral. No teste independente, 7 de 24
  sonolentos ficaram verdes (seção 8.1).
- Os pesos do nível são **escolhas de projeto documentadas** (origem de cada um na seção 13), não ajustadas em dados de sonolência
  real; o DMD s5 é encenado. A amostra da validação com vídeos é pequena (~40 min no navegador; 8,2 h no teste
  independente da seção 8.1, com pessoas sentadas).
- A PWA usa WebAssembly/WebGL; o tempo medido é limite superior em relação a um app nativo.
- No iPhone: sem vibração; o volume de mídia do aparelho limita o alarme; a tela pode apagar se o
  navegador não oferecer Wake Lock (desligue o bloqueio automático). A liberação do som depois de voltar
  de segundo plano (`visibilitychange`) sem toque ainda não foi conferida no aparelho: se falhar, um toque
  na tela religa.

## 11. Arquivos

| Arquivo | Função |
|---|---|
| `index.html`, `styles.css` | interface: telas Início, Calibração, Em viagem, alarme, Viagens (resumo), Ajustes, Como funciona e Detalhes técnicos; tema claro e escuro |
| `icon.svg`, `apple-touch-icon.png`, `icon-192.png`, `icon-512.png` | ícone do app (o PNG de 180 px é o da tela de início do iPhone) |
| `app.js` | câmera, Face Landmarker, som (Web Audio, volume), interface, calibração, benchmark, exportação |
| `engine.js` | motor puro testável em Node: EAR/MAR, pose, fusão dos olhos, fechamentos, PERCLOS, bocejo, cabeça, nível, alarme, volume (e as classes da v1) |
| `tflite_io.js` | entrada/saída dos modelos TFLite (dtype, quantização) |
| `test_engine.mjs` | testes do motor: `node app/pwa/test_engine.mjs` |
| `sw.js`, `manifest.webmanifest`, `icon.svg` | PWA (rede primeiro, cache como reserva; cache `sonolencia-pwa-v5`, que guarda também `vendor/` e as respostas da CDN) |
| `serve_https.py` | servidor HTTPS local (8443) que recebe o benchmark e confere `vendor/` ao subir |
| `check_tflite_web.py` | verifica se um `.tflite` carrega no runtime da PWA |
| `models/` | cópias dos `.tflite`/`.json` (ver `models/README.md`) |
| `vendor/` | dependências locais, fora do Git: `bash tools/baixar_dependencias_app.sh` |
| `../../tools/app_validacao/` | `extrair_sinais_app.py` (sinais do DMD e, com `--uta-folds`, da UTA-RLDD, com o mesmo modelo), `validar_motor_dmd.mjs` (validação offline e alternativas do ajuste final), `sensibilidade_pesos.mjs` (sensibilidade dos pesos do nível, seção 13), `validar_motor_uta.mjs` (teste independente na UTA-RLDD, seção 8.1), `teste_headless_pwa.mjs` (teste da interface no Chrome headless, sempre mudo), `preparar_clipe.py` + `rodar_app_video.mjs` + `analisar_app_videos.mjs` (app no navegador com vídeos reais como câmera) |

Dependências (primeiro `vendor/`, depois a CDN): `@mediapipe/tasks-vision@1.0.1` + `face_landmarker.task`
float16 v1; `@tensorflow/tfjs-core@4.22.0`, `@tensorflow/tfjs-backend-cpu@4.22.0`,
`@tensorflow/tfjs-tflite@0.0.1-alpha.10` (módulos WASM em `.../wasm/`). O tfjs-tflite tenta antes um
`dist/tflite_web_api_cc_simd.js` que não existe nem na CDN: o 404 no console é esperado.

## 12. Testes

```bash
node --check app/pwa/app.js && node --check app/pwa/engine.js && node --check app/pwa/tflite_io.js
node app/pwa/test_engine.mjs                                   # 217 asserções
node tools/app_validacao/validar_motor_dmd.mjs                 # precisa de data/signals/app_tasks/dmd (local)
node tools/app_validacao/sensibilidade_pesos.mjs               # idem; 353 configurações, cerca de 1,5 min com 6 threads
node tools/app_validacao/validar_motor_uta.mjs                 # precisa de data/signals/app_tasks/uta (local; extrair_sinais_app.py --uta-folds 2 4)
python3 app/pwa/serve_https.py --port 8767 &                   # outra porta: nunca derrube a 8443 do celular
node tools/app_validacao/teste_headless_pwa.mjs --url "https://localhost:8767/?autostart=1" --seconds 20
```

Não use `&bench=1` no teste do computador com esse servidor: o benchmark seria gravado em
`results/phone/` como se fosse do celular (para testar o envio, sirva a pasta com uma cópia do
`serve_https.py` fora do repositório).

**Todo teste que abre o app no computador é mudo:** `--mute-audio` no Chrome, `?volume=0` no app, a
política de autoplay padrão (sem toque não há áudio) e o script aborta se o volume ou o ganho saírem de 0.
O Chrome headless novo toca som de verdade nos alto-falantes do computador. Os cliques de
`teste_headless_pwa.mjs` (`--click-sound`, `--click-at`) são gestos reais do DevTools, como um toque no
iPhone.

Referências citadas: Wierwille et al. (1994), DOT HS 808 247; Dinges e Grace (1998), FHWA-MCRT-98-006
(ambas em `paper/refs.bib`); Provine, R. R. (1986), "Yawning as a stereotyped action pattern and
releasing stimulus", *Ethology* 72(2):109-122 (não está no `refs.bib`).

## 13. De onde vem cada parâmetro

A pergunta da banca é justa: os números do app foram calculados ou escolhidos no olho? Cada parâmetro de `CFG`
(`engine.js`) tem uma destas três origens:

- (a) literatura ou definição clássica;
- (b) escolhido comparando variantes contra a anotação quadro a quadro do DMD (`tools/app_validacao/validar_motor_dmd.mjs`; s5 = sonolência encenada, para a detecção; s1 = direção real com motoristas acordados, para o falso alarme);
- (c) regra de projeto, sem ajuste em dados, seguindo os critérios da seção 13.2.

### 13.1 Tabela

| Parâmetro (`CFG`) | Valor | Origem | Como foi testado |
|---|---|---|---|
| Janela do PERCLOS (`perclos.windowMs`) | 60 s | (a) Wierwille et al. (1994); Dinges e Grace (1998) | testes do motor (PERCLOS medido por tempo) |
| PERCLOS só com fechamentos lentos (`perclos.minEpisodeMs`) | conta só fechamentos de 0,5 s ou mais | (a) o PERCLOS mede fechamentos lentos da pálpebra, não piscadas (Dinges e Grace, 1998); o corte de 0,5 s é o mesmo do fechamento longo (c) | direção real do DMD: com as piscadas dentro, PERCLOS p95 de 10,0% e 5,1% do tempo em vermelho; só com fechamentos de 0,5 s ou mais, 1,2% e 3,3% (`results/app_validacao_dmd_ajuste_final.csv`). Teste do motor: 75 piscadas por minuto por 3 min ficam sempre no verde |
| Pontos do PERCLOS (`score.perclos`) | 0 até 7,5%; 40 pontos em 15%; teto de 60 | (c) regra de projeto; a literatura não tem um limite único (de 8% a 12% sobre o PERCLOS de todos os fechamentos), e o nosso PERCLOS conta só fechamentos de 0,5 s ou mais | testes do motor |
| Microssono (`eyes.microsleepMs`) | olhos fechados por 1 s ou mais: vermelho na hora | (a) microssonos duram de 1 a 15 s (Skorucak et al., 2020); um fechamento de 1 s já não é piscada | 1,2 s foi avaliado no DMD e rejeitado: os fechamentos anotados de 1 s ou mais que disparam o vermelho caem de 104 para 95 de 113 |
| Piscada e fechamento longo (`eyes.blinkMinMs`, `blinkMaxMs`, `longMs`) | piscada de 50 ms a 0,5 s; fechamento longo de 0,5 a 1 s | 0,5 s: (a) uma piscada dura de 0,1 a 0,4 s; 50 ms: (c) 1 quadro a 30 a 60 quadros por segundo | DMD s5: fechamentos de 0,5 s ou mais com revocação 0,929 e precisão 0,786 |
| Limiar do EAR (`calib.earFactor`) | 0,75 × mediana do EAR na calibração | (b) fixado antes e conferido no DMD: entre 0,60 e 0,90, deu o maior kappa por quadro (0,773); o F1 de piscadas foi um pouco maior com 0,80 (0,775 contra 0,749) | `results/exp5_dmd_sensibilidade_limiar.csv` (16 sessões) |
| Regra do olho fechado (`eyes.blendClosed`, `blendRel`, `blendSupport`, `earPath`) | os dois eyeBlink em 0,5 ou mais e 0,25 acima do repouso; o EAR também marca fechado quando o menor eyeBlink passa de 0,4 e nenhuma trava está ativa | (b) escolhida entre 11 variantes: só o EAR, só o eyeBlink (limiar de 0,45 a 0,55, folga de 0,25 ou 0,30 sobre o repouso) e fusões do eyeBlink com o EAR (apoio de 0,35 a 0,45, com 2 ou 4 travas) | kappa por quadro de 0,938 (EAR sozinho: 0,831); 3,3% do tempo em vermelho na direção real (EAR sozinho: 26,1%). As variantes com folga de 0,30 baixavam esse vermelho para 0,1%, mas perdiam fechamentos de 1 s ou mais (revocação de 0,876 contra 0,920). `results/app_validacao_dmd_variantes_olhos.csv` |
| Valores das travas (`lookDownGate`, `headDownGateDeg`, `squintGate`, `asideGateDeg`) | olhar para baixo 0,6; cabeça 10° abaixo; olhos apertados 0,45; olhar para o lado 20° | (c) regra de projeto; a comparação foi entre ter 2 ou 4 travas, não entre valores de cada uma | dentro da comparação das 11 variantes |
| Histerese na reabertura (`eyes.reopenHyst`) | desligada | (b) avaliada no DMD e rejeitada | direção real: fechamentos de 0,5 a 1 s foram de 15 para 46, e os avisos, de 1,9 para 17,1 por hora |
| Boca aberta (`yawn.jawOpen`, `jawRel`, `marRel`) | jawOpen de 0,4 ou mais e 0,3 acima do repouso, ou MAR 0,4 acima do repouso | jawOpen: (b) grade no DMD; folgas sobre o repouso: (c) | na grade (`results/app_validacao_dmd_bocejo_grade.csv`), com 1,5 s, jawOpen de 0,3 a 0,5 dá revocação sem a mão de 0,617 a 0,681 e nenhum bocejo falso na direção real. O valor 0,4 fica no meio da faixa; não é o melhor ponto da grade |
| Duração mínima do bocejo (`yawn.minMs`) | 1,5 s | (b) grade no DMD; combina com a duração média de um bocejo espontâneo, cerca de 6 s (Provine, 1986) | com 1,0 s, a revocação sem a mão sobe de 0,638 para 0,723, mas aparecem 1,9 bocejos falsos por hora na direção real, e o tempo em vermelho sobe de 3,3% para 5,0% |
| Tolerâncias e contagem do bocejo (`yawn.gapMs`, `lostMaxMs`, `windowMs`) | a boca pode fechar por menos de 0,15 s; mão ou rosto perdido até 1 s; contagem em 5 min | (c) | testes do motor |
| Queda da cabeça (`head.dropDeg`, `dropMs`, `eyesMs`) | 18° por 0,8 s, com os olhos fechados por 0,4 s seguidos | (c) especificação do projeto (de 15 a 20° e 0,8 s). Os olhos fechados separam sono de olhar o painel, e 0,4 s fica acima de uma piscada comum (0,1 a 0,3 s) | sem revocação medida, porque o DMD não anota cabeceio; na direção real, 1 queda falsa em 63,3 min; testes sintéticos |
| Cabeceio e cabeça de lado (`nodDeg`, `nodOnsetMs`, `nodMaxMs`, `rollDeg`, `yawAsideDeg`) | desce 15° em até 0,5 s e volta em até 1,5 s; tombar 30° conta como queda; olhar 25° para o lado anula o episódio | (c) | testes sintéticos |
| Rosto que some depois de baixar (`head.lostEyesMs`) | só vira queda com os olhos fechados por 0,4 s seguidos | (c) a mesma exigência da queda com o rosto visível | no navegador, as 2 quedas falsas do fim de clipes do NITYMED sumiram |
| Cabeça baixa de olhos abertos (`head.downOpenMs`) | 2 s ou mais: distração, 0 ponto | (c) | testes do motor; roteiro da banca |
| Calibração (`calib.neutralMs`, `nBlinks`, `maxClosedFrac`) | 10 s olhando para a frente; 20 piscadas; aviso se os olhos ficarem fechados em mais de 10% do tempo | (c). O aviso: acordado, de 15 a 20 piscadas por minuto, de 0,1 a 0,3 s cada, dão de 2% a 10% do tempo com o olho fechado | no navegador, o aviso só apareceu no vídeo sonolento Fold3/31 da UTA-RLDD (23% da calibração) |
| Pontos por evento (`score`) | fechamento de 0,5 a 1 s: 25; microssono, depois de reabrir: 25; bocejo: 25; queda ou cabeceio: 30; piscadas mais lentas: de 0 (1,25 vez a duração da calibração) a 20 (2 vezes) | (c) critérios da seção 13.2 | testes do motor (2 bocejos levam ao amarelo, com nível 48; 2 bocejos e 1 cabeceio levam ao vermelho); direção real do DMD com 3,3% do tempo em vermelho e 16 de 17 sessões sem alarme; sensibilidade dos pesos: em 300 configurações sorteadas (cada peso, teto e decaimento de 0,7 a 1,3 vez; limiares 40 e 70 a até 10 pontos), o vermelho na direção real ficou entre 1,7% e 4,9% (percentis 5 e 95) e o da sonolência encenada entre 56,2% e 72,9% |
| Tetos | fechamentos e microssonos: 60; PERCLOS: 60; bocejo: 50; cabeça: 60; piscadas lentas: 20 | (c) os pontos de um sinal só não chegam a 70 | os mesmos testes da linha acima |
| Decaimento | fechamentos e microssono: 2 min; bocejo e cabeça: 5 min | (c) evento antigo perde peso | os mesmos testes |
| Estados (`levels`) | amarelo a partir de 40 (sai abaixo de 35); vermelho a partir de 70 (sai abaixo de 60) | (c) histerese contra o pisca-pisca entre estados | testes do motor |
| Vermelho forçado e piso (`forcedFloor`, `postForced`) | microssono ou cabeça caída com os olhos fechados: nível exibido de pelo menos 75; depois, pelo menos 40 por 1 min | (c) | roteiro da banca: ao reabrir os olhos, amarelo em 1,7 s; na versão 2.0 (microssono com 40 pontos por 5 min, sem o piso e com saída em 60), o vermelho continuava por cerca de 1,5 min |
| LSTM (`score.ml`) | no máximo 10 pontos, com "sonolento" (p de 0,6 ou mais) em 5 predições seguidas e só com o resto já em 40 ou mais | (c) um modelo com 57,2% de acurácia por vídeo (Exp. 1) só reforça um quadro de cansaço | num vídeo de pessoa acordada da UTA-RLDD, os avisos causados pelo LSTM caíram de 3 para 0 |
| Descida do nível (`maxFallPerS`) | no máximo 3 pontos por segundo | (c) | testes do motor |
| Alarme (`alarm`) e volume | sirene de pelo menos 2 s; "ding" no máximo a cada 20 s; "Estou acordado" silencia por 30 s; volume padrão de 20% | (c) | testes do motor e da interface, sempre mudos |

Referências: Wierwille et al. (1994) e Dinges e Grace (1998) para o PERCLOS; Provine (1986) para a duração do
bocejo; Skorucak et al. (2020, Frontiers in Neuroscience 14:8) para o microssono de 1 a 15 s. O limite de 15% do PERCLOS
não vem de uma fonte: é regra de projeto, mais conservadora que os 8% a 12% usados em outros trabalhos.

### 13.2 Critérios dos pesos

Os pontos, os tetos e os tempos de decaimento seguem regras escritas antes, e não um ajuste em dados:

- um evento isolado não muda a cor: 1 bocejo, 1 fechamento longo ou 1 queda ficam no verde (25 ou 30 pontos);
- dois eventos do mesmo tipo levam ao amarelo (2 bocejos: nível 48 no teste do motor);
- os pontos de um sinal só não chegam ao vermelho: todo teto é de 60 ou menos, e o vermelho pelo nível começa em 70;
- dois sinais diferentes juntos chegam ao vermelho (2 bocejos e 1 cabeceio, no teste do motor);
- perigo imediato força o vermelho na hora: microssono e cabeça caindo com os olhos fechados;
- evento antigo perde peso: os pontos caem em linha reta até zerar em 2 min (olhos) ou 5 min (bocejo e cabeça);
- o LSTM só reforça um quadro que já é de cansaço e nunca muda a cor sozinho.

### 13.3 Por que os pesos não foram calculados por otimização

Para otimizar os pesos, seria preciso uma base de sono real com esses sinais marcados ao longo do tempo, e ela não
existe. A UTA-RLDD tem sono real, mas só um rótulo por vídeo (o KSS que a própria pessoa relatou), sem dizer em que
momento ela estava sonolenta nem que sinal apareceu. O DMD tem eventos marcados quadro a quadro, mas encenados.
Ajustar os pesos no DMD repetiria o problema da PP1: o que se aprende com sono encenado transfere mal para o real
(no Exp. 5b, o modelo treinado no DMD reconhece só 48,4% das janelas sonolentas da UTA-RLDD). Por isso os pesos
seguem critérios que qualquer pessoa confere na lista "Por que" do app, e o DMD serviu para medir a detecção dos eventos
(s5) e o falso alarme em quem está acordado (s1).

### 13.4 O que dá para afirmar

- Os números do DMD da seção 8 são de desenvolvimento: a regra dos olhos, o bocejo e os ajustes finais foram
  escolhidos comparando variantes nessas mesmas sessões.
- Os pesos não foram ajustados em sono real. No teste independente da seção 8.1 (24 pessoas da UTA-RLDD fora do
  ajuste, sentadas, sem dirigir), o app alarmou 15 dos 24 vídeos sonolentos e 2 dos 24 acordados.
- Sensibilidade dos pesos (cada peso variado em até 30%; `results/app_sensibilidade_pesos.md`): mudando um número por vez, só o teto de 60 pontos dos fechamentos somados, o limiar de 70 e a descida máxima do nível mexem no falso alarme (de 1,8% a 4,7% do tempo em vermelho na direção real); os outros mudam menos de 0,05 ponto. Com tudo variando junto, em 300 configurações sorteadas, o vermelho na direção real ficou entre 1,7% e 4,9% (percentis 5 e 95; a configuração final dá 3,3%) e o da sonolência encenada entre 56,2% e 72,9%; 291 das 300 (97%) mantêm o falso alarme até 5% e a sonolência encenada acima de 50%. As 9 que passaram de 5% têm o limiar de SONOLENTO entre 60 e 65. Todo o vermelho da direção real está numa sessão só (gA_5), em episódios que começaram com um microssono detectado num motorista que olhava para baixo.
