# App × modelos nas mesmas pessoas da UTA-RLDD

Gerado por `src/experiments/comparar_app_uta.py`. 24 sujeitos (os do teste do app em `results/app_validacao_uta.md`), um vídeo acordado e um sonolento de cada. AUC por vídeo (sonolento × acordado); IC 95% por bootstrap de sujeitos (1.000 reamostragens, semente 42). Os IC daqui servem para comparar os detectores entre si; para o app, o IC citado no artigo é o de `results/app_validacao_uta.md`, que usa outra sequência de sorteios (a AUC é a mesma, e os limites mudam só no último dígito). A diferença para o app (detector menos app, % do tempo fora do verde) usa os mesmos sorteios nos dois detectores (bootstrap pareado).

| Detector | Medida | AUC por vídeo [IC 95%] | Diferença para o app [IC 95%] |
|---|---|---|---|
| app 2.1 (sem treino nesta base) | % do tempo fora do verde | 0,812 [0,705; 0,911] | referência |
| app 2.1 (sem treino nesta base) | nível médio | 0,839 [0,742; 0,931] | 0,027 [−0,026; 0,091] |
| RF treinado no DMD (C) | escore médio, com centragem | 0,811 [0,686; 0,915] | −0,001 [−0,171; 0,153] |
| RL treinada no DMD (C) | escore médio, com centragem | 0,755 [0,627; 0,894] | −0,056 [−0,231; 0,118] |
| RF treinado no DMD (C) | escore médio, sem centragem | 0,780 [0,656; 0,899] | −0,032 [−0,166; 0,118] |
| RL treinada no DMD (C) | escore médio, sem centragem | 0,825 [0,727; 0,925] | 0,013 [−0,128; 0,156] |
| RF treinado na UTA-RLDD (B) | escore médio, com centragem | 0,953 [0,898; 0,997] | 0,141 [0,042; 0,248] |
| RL treinada na UTA-RLDD (B) | escore médio, com centragem | 0,887 [0,785; 0,983] | 0,076 [−0,047; 0,208] |
| RF treinado na UTA-RLDD (B) | escore médio, sem centragem | 0,806 [0,686; 0,927] | −0,006 [−0,144; 0,128] |
| RL treinada na UTA-RLDD (B) | escore médio, sem centragem | 0,797 [0,691; 0,915] | −0,015 [−0,142; 0,127] |
| regra do rótulo do DMD (sem treino) | fração de janelas marcadas | 0,853 [0,754; 0,944] | 0,042 [−0,081; 0,168] |

Leitura: no bootstrap pareado, a diferença para o app só é detectável em RF treinado na UTA-RLDD (B), escore médio, com centragem (0,141). O app, sem treino nem ajuste nesta base, não se distingue do RF treinado no DMD nem da regra do rótulo, que é a base dos seus sinais de olho. O RF treinado na própria UTA-RLDD só fica acima com a centragem pelo trecho final do vídeo alerta, o atalho discutido em `results/exp5b_robustez.md`; sem centragem, ele fica perto do app.
