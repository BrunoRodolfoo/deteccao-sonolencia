# Notas das medições no celular (Exp. 4)

Condições de cada sessão de benchmark (60 s por execução), para ler junto de `results/exp4_phone_latency.md`.

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
