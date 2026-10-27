# App × modelos nas mesmas pessoas da UTA-RLDD

Gerado por `src/experiments/comparar_app_uta.py`. 24 sujeitos (os do teste do app em `results/app_validacao_uta.md`), um vídeo acordado e um sonolento de cada. AUC por vídeo (sonolento × acordado); IC 95% por bootstrap de sujeitos (1.000 reamostragens, semente 42). Os IC daqui servem para comparar os detectores entre si; para o app, o IC citado no artigo é o de `results/app_validacao_uta.md`, que usa outra sequência de sorteios (a AUC é a mesma, e os limites mudam só no último dígito).

| Detector | Medida | AUC por vídeo [IC 95%] |
|---|---|---|
| app 2.1 (sem treino nesta base) | % do tempo fora do verde | 0,812 [0,705; 0,911] |
| app 2.1 (sem treino nesta base) | nível médio | 0,839 [0,742; 0,931] |
| RF treinado no DMD (C) | escore médio, com centragem | 0,811 [0,686; 0,915] |
| RL treinada no DMD (C) | escore médio, com centragem | 0,755 [0,627; 0,894] |
| RF treinado no DMD (C) | escore médio, sem centragem | 0,780 [0,656; 0,899] |
| RL treinada no DMD (C) | escore médio, sem centragem | 0,825 [0,727; 0,925] |
| RF treinado na UTA-RLDD (B) | escore médio, com centragem | 0,953 [0,898; 0,997] |
| RL treinada na UTA-RLDD (B) | escore médio, com centragem | 0,887 [0,785; 0,983] |
| RF treinado na UTA-RLDD (B) | escore médio, sem centragem | 0,806 [0,686; 0,927] |
| RL treinada na UTA-RLDD (B) | escore médio, sem centragem | 0,797 [0,691; 0,915] |
| regra do rótulo do DMD (sem treino) | fração de janelas marcadas | 0,853 [0,754; 0,944] |

Leitura: o app, sem treino nem ajuste nesta base, fica no nível do RF treinado no DMD e perto da regra do rótulo, que é a base dos seus sinais de olho. O RF treinado na própria UTA-RLDD só fica bem acima com a centragem pelo trecho final do vídeo alerta, o atalho discutido em `results/exp5b_robustez.md`; sem centragem, ele fica no mesmo nível do app.
