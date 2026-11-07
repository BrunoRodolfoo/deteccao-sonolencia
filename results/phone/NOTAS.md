# Notas das medições no celular (Exp. 4)

Condições de cada sessão de benchmark (60 s por execução), para ler junto de `results/exp4_phone_latency.md`.

## 06/11/2026: iPhone 17 Pro (512 GB), app 2.1

- Medido por André Luís Oliveira Ferreira da Silva, no aparelho dele.
- Horário: das 22:53 às 22:58 (horário de Brasília). Uma calibração só, às 22:53, sem aviso (limiar do EAR 0,203), e
  três execuções na mesma viagem: a 2ª começou cerca de 45 s depois do fim da 1ª, e a 3ª, 8 s depois do fim da 2ª.
- Sistema e navegador: iOS 26.6.1 e Chrome 156 (motor WebKit), lidos do user-agent. O modelo vem de
  `results/phone/aparelhos.json`, porque o iPhone não o informa.
- Face Landmarker na GPU, câmera a 60 quadros por segundo (vídeo 480x640). Dependências locais (`./vendor/`),
  conferidas em "Detalhes técnicos > Modelos > Bibliotecas".
- Rede: o Mac e o iPhone no mesmo Wi-Fi de casa.
- Bateria entre 73 % e 72 % às 22:55, depois da 1ª execução. A temperatura do aparelho não foi anotada.
- Eventos: nenhum na 1ª e na 2ª execução. Na 3ª, o André estava de fato com sono: aos 5 s, os dois olhos ficaram
  fechados por cerca de 1 s (EAR perto de 0,01, com a cabeça parada), e o app marcou microssono e tocou o alarme, no
  volume padrão de 20 %. O nível foi a 75 e desceu devagar (52 s no amarelo). O tempo por quadro dessa execução ficou
  igual ao das outras.
- Resultado: tempo total por quadro (p50) de 11, 11 e 12 ms (p95 de 13, 14 e 14 ms); FPS médio de 60,0 a 60,3; Face
  Landmarker com 9 a 10 ms e CNN dos olhos com 2 ms. O iPhone registra os tempos em milissegundos inteiros, e por isso
  o motor de sinais aparece com 0 ms.
- As duas execuções de 27/09 (app 1.0) continuam nesta pasta, mas ficam fora das médias: o artigo e o pôster usam só a
  versão mais nova do app em cada aparelho.

## 05/11/2026: Samsung Galaxy S23 Ultra (SM-S918B), app 2.1

- Medido por Paulo Henrique Perin, no aparelho dele.
- Horário: das 19:27 às 19:31 (horário de Brasília). Três execuções seguidas, com poucos segundos entre elas; o
  roteiro pedia cerca de 1 min.
- Sistema e navegador: Android 16 e Chrome 156, lidos pelo próprio benchmark (Client Hints e user-agent).
- Face Landmarker na GPU. Dependências locais (`./vendor/`, servidas pelo `app/pwa/serve_https.py`), conferidas em
  "Detalhes técnicos > Modelos > Bibliotecas".
- Rede: o Mac ligado ao roteador Wi-Fi do Galaxy, porque o Wi-Fi da PUC bloqueia a conexão entre aparelhos.
- Bateria entre 69 % e 67 %. A temperatura do aparelho não foi anotada.
- Calibração sem aviso; nenhum evento nem alarme nas três execuções (rosto parado diante da câmera).
- Resultado: tempo total por quadro (p50) de 40,1, 39,1 e 39,5 ms; FPS médio de 22,4 a 22,8.
- O app já estava com a interface nova (27/10/2026); o detector, os parâmetros e o que o benchmark mede são os da
  versão 2.1.

## 27/09/2026: iPhone 17 Pro, app 1.0

- Duas execuções no aparelho de um dos autores (André), com a lógica antiga do app (antes do detector 2.x).
- Tempo total por quadro (p50) de 23 e 11 ms. Detalhes em `results/exp4_phone_latency.md`.
