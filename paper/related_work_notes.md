# Notas de revisão bibliográfica (verificadas): artigo IEEE, Equipe 14

Data da verificação: 20/09/2026. Todas as chaves abaixo correspondem a `paper/refs.bib`.
Cada item traz de 2 a 4 linhas com o que a referência afirma e que interessa ao artigo, com os
números exatos extraídos da fonte primária (PDF/página do editor/arXiv/página do dataset).
Itens marcados **[UNVERIFIED]** têm alguma parte não confirmada em fonte primária; use com cautela.
Acrescentadas na revisão final do artigo (26/10/2026): **perkins2023** (seção 7), **xie2026** e **pradeep2026** (seção 3).
Para essas três, os metadados vêm do Crossref (comentário no `refs.bib`) e as notas dizem só o que o artigo usa de cada uma.

**Revisão crítica (20/09/2026):** todos os DOIs e identificadores arXiv foram reconferidos (Crossref, DataCite, API do arXiv) e todos os números abaixo foram localizados nos textos extraídos ou nos resumos oficiais. Correções feitas nesta revisão: massoz2018 (70,69 → **70,68 %**), rajkumarsingh2021 (o valor 90,4 % é fisiológico + facial, não "só face"), shen2020/joshi2020/fu2024 agora com veículo, páginas e DOI confirmados.

---

## 1. Dataset principal (UTA-RLDD) e baseline temporal

**ghoddoosian2019**: Ghoddoosian, Galib, Athitsos, CVPRW 2019, pp. 178-187, DOI 10.1109/CVPRW.2019.00027.
- Apresenta o RLDD (UTA-RLDD): ~30 h de vídeo RGB, 60 participantes, 3 vídeos/participante (alerta, pouca vigilância, sonolento) = 180 vídeos, autogravados com celular/webcam, taxa de quadros sempre < 30 fps, tamanho total 111,3 GB; 51 homens/9 mulheres, idade de 20 a 59 anos (média 25, DP 6); óculos em 21 dos 180 vídeos; barba considerável em 72 dos 180.
- Rótulos autoatribuídos com base na KSS: alerta = níveis de 1 a 3; pouca vigilância = níveis 6 e 7; sonolento = níveis 8 e 9 (níveis 4 e 5 não usados). Rótulos numéricos 0/5/10. 60 sujeitos divididos aleatoriamente em 5 folds de 12 (validação cruzada por sujeito).
- Baseline: detector de piscada + 4 características por piscada (frequência, amplitude, duração, velocidade de abertura), normalizadas por sujeito (z-score com estatísticas de 1/3 do vídeo de alerta), sequência de 30 piscadas → HM-LSTM (4 camadas, 32 unidades) → regressão 0-10 discretizada (<3,3 / 3,3-6,6 / >6,6) → voto por vídeo. Adam, lr 0,000053, batch 64, 80 épocas.
- Resultados (média 5 folds, Tabela 2): HM-LSTM BSA 54 % / VA 65,2 %; LSTM 52,8 % / 61,4 %; camadas densas 52 % / 57 %; julgamento humano VA 57,8 %. VA por fold (Tabela 3): HM-LSTM 0,64/0,61/0,70/0,64/0,67; humanos 0,62/0,59/0,60/0,53/0,55.
- Custo: de 35 a 80 fps (quadros de 568×320 a 1920×1080) em workstation Linux com Intel Xeon E3-1270 V2 @ 3,5 GHz e 16 GB RAM (sem GPU mencionada). Código: github.com/rezaghoddoosian.

**chung2017**: HM-LSTM (Chung, Ahn, Bengio, ICLR 2017, arXiv:1609.01704). Arquitetura recorrente hierárquica multiescala usada como núcleo do baseline do UTA-RLDD.

## 2. NTHU-DDD (sonolência atuada) e a escalada de acurácias > 90 %

**weng2016**: Weng, Lai, Lai, ACCV 2016 Workshops (LNCS 10118, pp. 117-133, DOI 10.1007/978-3-319-54526-4_9).
- Introduz o NTHU-DDD e o HTDBN (DBNs para face/cabeça + dois HMMs contínuos). É a citação obrigatória do dataset segundo a página oficial da NTHU.
- A acurácia do HTDBN no NTHU-DDD não foi lida no capítulo Springer (acesso bloqueado); fonte secundária: Massoz et al. 2018 (Tabela 4) reproduzem **84,82 % de acurácia / F1 85,39 %**. Se citar o número, indicar "conforme reportado em [massoz2018]".

**park2016**: Park, Pan, Kang, Yoo, ACCV 2016 WS (pp. 154-164, DOI 10.1007/978-3-319-54526-4_12). Rede DDD (3 redes profundas + SVM). Acurácia no conjunto de avaliação do NTHU-DDD reportada nas tabelas comparativas de lyu2018: **73,06 %** (DDD-IAA). Em yu2019 (Tabela III, arXiv), DDD-FFA média 0,708 e DDD-IAA 0,730.

**shih2016**: Shih & Hsu, MSTN (LNCS 10118, pp. 146-153, DOI 10.1007/978-3-319-54526-4_11). CNN espacial + LSTM + suavização temporal. O resumo indexado e a Tabela 4 de massoz2018 reportam **82,61 %** (F1 82,82 %) no conjunto de teste do NTHU-DDD; as tabelas de lyu2018 e shen2020 citam **85,52 %** no conjunto de avaliação.

**yu2019**: Yu, Park, Lee, Jeon, IEEE T-ITS 20(11):4206-4218, DOI 10.1109/TITS.2018.2883823 (versão de autor arXiv:1910.09722).
- 3D-DCNN com aprendizado de representação adaptativo à condição (óculos/iluminação). Usa só treino (360 vídeos, 722.223 quadros, 18 sujeitos) e avaliação (20 vídeos, 173.259 quadros, 4 sujeitos) do NTHU-DDD, pois o teste não é público/anotado.
- Acurácia média no conjunto de avaliação (Tabela III/IV do arXiv): **0,762** (por cenário: dia sem óculos 0,796; dia óculos 0,781; dia óculos escuros 0,738; noite sem óculos 0,765; noite óculos 0,734). Comparados: LeNet 0,601, AlexNet 0,659, VGG-FaceNet 0,678, LRCN 0,629, FlowImageNet 0,615.
- Execução: 38,1 fps (28,6 ms) em Core i7 3,4 GHz + GTX TITAN. Observação: a tabela de lyu2018 lista uma versão anterior de Yu et al. com 72,60 %.

**lyu2018**: Lyu, Yuan, Chen, arXiv:1801.02325. LMDF (CNN multi-granularidade + LSTMs, memória de 120 quadros). **90,05 %** no conjunto de avaliação do NTHU-DDD; ~37 fps em GPU Tesla M40 e ~3 fps em CPU. Descreve o NTHU-DDD: treino com 18 voluntários (10 homens, 8 mulheres), avaliação com 4 (2 homens, 2 mulheres).

**shen2020**: Shen, Zhao, Zhang, Zhang, RICAI 2020 (ACM, Xangai, pp. 271-277, DOI 10.1145/3438872.3439093; preprint arXiv:2010.06235). Rede two-stream multi-feature (CLAHE + fluxo óptico + SE-blocks). Reporta **94,46 %** no NTHU-DDD; sua tabela comparativa lista MSTN 85,52 %, DDD (ConvCGRNN) 84,81 %, MCNN/LMDF 90,05 %, Li et al. (DB-LSTM) 93,6 %. Atenção ao protocolo: o texto diz que "um conjunto de vídeos [dos 18 de treino] foi escolhido como validação e três como teste", ou seja, não usa necessariamente o conjunto de avaliação oficial.

**Leitura para o artigo:** no NTHU-DDD as acurácias publicadas escalaram de ~73 % (2016) para valores entre 90 % e 94 % (de 2018 a 2020) em poucos anos, com protocolos de partição heterogêneos (avaliação oficial de 4 sujeitos vs. subconjuntos do treino). No UTA-RLDD (real, por sujeito), o baseline fica em 54 % (BSA) / 65,2 % (VA) e humanos em 57,8 %.

## 3. Evidências de inflação por vazamento de sujeito / queda em avaliação subject-independent ou cross-dataset

**wijnands2020**: Wijnands et al., Neural Computing and Applications 32(13):9731-9743, DOI 10.1007/s00521-019-04506-0.
- Usa NTHU-DDD (treino 18 pessoas, 8,5 h; avaliação 4 pessoas, 1,5 h; teste de 14 pessoas não disponível). Alerta explicitamente que "alguns estudos reportam acurácia nos dados de teste do DDD treinando em treino+avaliação", e reporta desempenho *out-of-sample* no conjunto de avaliação: InceptionV1 69,6 %; I3D 75,4 %; MobileNetV2_1.4 71,8 %; modelo 3D móvel proposto **73,9 %** (Tabela 4). Com sujeitos disjuntos, esses modelos ficam de 70 % a 75 %. Os 90,05 % de lyu2018 (Seção 2) também são do conjunto de avaliação, com pessoas fora do treino: a diferença entre os dois grupos não vem de vazamento de sujeito.
- Latência em Samsung Galaxy S7 (Tabela 7): 1,0 s (5 quadros), **1,1 s (10 quadros)**, 1,9 s (30 quadros); fusão tardia 5,6 s; fusão lenta 31,7 s; multiplicador de profundidade 0,35 → 0,5 s. App Android com TensorFlow.

**bakker2022**: Bakker et al., IEEE T-ITS 23(5):4791-4800, DOI 10.1109/TITS.2021.3090272.
- Vídeo em direção naturalística (13 motoristas, sessões de 90 min dia/noite, KSS de 1 a 9; alerta = KSS de 1 a 6, sonolento = de 7 a 9). Modelo genérico + módulo **personalizado**: acurácia 92 % (sens. 91,7 %, espec. 92,3 %, F1 90,4 %); **sem personalização (subject-independent) a acurácia cai para 72 %**; baseline PERCLOS 68 %. Diferença de 20 pontos atribuível à dependência de sujeito.

**massoz2018**: Massoz, Verly, Van Droogenbroeck, Sensors 18(9):2801, DOI 10.3390/s18092801.
- Caracterização multiescala a partir de vídeo da face, 29 sujeitos com privação de sono e PVT; avaliação **leave-one-subject-out (29 folds)**. Acurácias globais (Tabelas 4 e 5 do texto integral, PMC6165048): **70,68 %** (janela 5 s), 85,45 % (15 s), 89,82 % (30 s), 94,22 % (60 s) [TNR/TPR: 72,26/58,69; 86,29/71,84; 90,44/75,76; 94,80/74,19]; o resumo arredonda para 70/85/89/94 %. Mostra que a acurácia depende fortemente da janela temporal e estabelece o LOSO como protocolo de referência.
- A Tabela 4 desse artigo também reproduz as acurácias reportadas no NTHU-DDD por Weng et al. (HTDBN) **84,82 %** (F1 85,39 %), Shih & Hsu (MSTN) **82,61 %** (F1 82,82 %) e Huynh et al. **87,46 %** (F1 87,97 %). É uma fonte secundária útil enquanto os capítulos Springer não forem lidos.

**florez2026**: Florez-Zela, arXiv:2606.08123v2 (2026). Diretamente relevante ao estágio de percepção ocular deste TCC.
- Seis modelos compactos de estado do olho treinados no **MRL Eye** (84.898 imagens, 37 participantes, partição por sujeito com seed 42) e avaliados **zero-shot** no RT-BENE (14 participantes, 107.350 quadros). Macro-F1 limpo no MRL: de 0,9566 a 0,9794; **zero-shot no RT-BENE: de 0,2066 a 0,7771**. MobileNetV3-Large: 0,9794 → 0,6034; ShuffleNetV2: 0,9566 → 0,2066; DeiT-Tiny 0,9752 → 0,7771.
- Latência: apenas MobileNetV3-Large e ShuffleNetV2 cumpriram o limite de 33,333 ms (P95, par binocular) em Jetson Nano com TensorRT FP32.

**rahman2020 (EyeNet)**: Rahman et al., ICACT 2020, DOI 10.23919/ICACT48636.2020.9061472.
- CNN pequena em recortes 24×24. Intra-dataset: CEW 99 %, ZJU 98,8 %, MRL Eye 99,16 % (com óculos) / 99,55 % (sem óculos). **Cross-dataset (Tabela IV): MRL→CEW 90,94 %; MRL→ZJU 90,12 %; ZJU→CEW 93,07 %; CEW→ZJU 67,34 %.** Queda de 6 a 30 pontos ao trocar de domínio. Estatísticas usadas: CEW 4.846 recortes (2.462 abertos, 2.384 fechados); MRL 84.898 (42.951 abertos, 41.947 fechados), 37 sujeitos (33 homens, 4 mulheres).

**cui2021**: Cui et al., Cyberworlds 2021, DOI 10.1109/CW52790.2021.00041 (EEG, não vídeo). LOSO em 11 sujeitos: 72,97 %, contra 55,42 % a 69,27 % dos baselines; ilustra que, mesmo em sinais fisiológicos, a acurácia subject-independent fica na faixa de 70 %.

**fu2024**: Fu, Boutros, Lin, Damer, IEEE T-IV 9(11):7279-7300, 2024, DOI 10.1109/TIV.2024.3395889 (preprint arXiv:2408.12990; survey). Lista os trabalhos que reportam LOSO (Cui 2021: 72,97 %; Cui 2022: 78,35 %; Paulo 2021: 75,87 %) e observa que modelos treinados em condições específicas "podem não ser diretamente aplicáveis" fora delas.

**joshi2020**: Joshi et al., IEEE IV 2020 (HSIM Workshop), pp. 207-212, DOI 10.1109/IV47402.2020.9304579 (arXiv:2010.11162). Vídeos in-the-wild de trabalhadores noturnos, 4 níveis; melhor modelo macro ROC-AUC 0,78 vs. 0,72 do baseline. É evidência de que, em dados reais, o desempenho fica longe dos 90 %+ de datasets atuados.

**xu2025 [UNVERIFIED protocolo]**: Xu et al., Information 16(4):294, DOI 10.3390/info16040294. O resumo oficial (Crossref/OpenAlex) reporta VA de 86,52 % no UTA-RLDD com características faciais híbridas (olhos, boca, pose da cabeça, olhar) + ensemble RF/XGBoost/MLP. Não foi possível confirmar no texto integral se a partição respeita os folds oficiais por sujeito; se não respeitar, é exemplo do tipo de inflação que o Exp. 1 deste TCC mede. Citar só com essa ressalva.

**xie2026**: Xie e Xiao, Frontiers in Neurorobotics 20:1857548, 2026, DOI 10.3389/fnbot.2026.1857548.
- Janela de tempo adaptativa, guiada pela confiança do modelo, para detectar fadiga em tempo real em veículos autônomos de nível 2 e 3, validada em várias bases.
- O que o artigo usa (Seção II): cruzam NTHU-DDD, YawDD, UTA-RLDD e DROZY e relatam queda média de 12,8 pp ao trocar de base.
- Diferença para o nosso trabalho (Seção II do artigo, vale também para pradeep2026): nenhum dos dois compara sonolência atuada e real com as mesmas características nas duas bases, IC pareado por sujeito e controles da referência da normalização.

**pradeep2026**: Pradeep e AlSaafin, Proc. 6th Int. Conf. on Electrical, Computer and Energy Technologies (ICECET), IEEE, 2026, pp. 1-5, DOI 10.1109/ICECET65726.2026.11632568.
- Compara modelos de vídeo de base e modelos exploratórios de aprendizado profundo em várias bases de sonolência.
- O que o artigo usa (Seção II): medem a perda ao treinar na UTA-RLDD e testar no DROZY. O artigo não cita número deste trabalho; conferir no texto antes de citar algum.
- O DROZY é a base sugerida nos trabalhos futuros do artigo para repetir a transferência com outra sonolência real.

## 4. Sonolência em smartphone / embarcado com latência reportada

**wijnands2020**: ver Seção 3. Galaxy S7: 1,1 s por inferência de 10 quadros (rede 3D) e 31,7 s com fusão lenta.

**jabbar2018**: Jabbar et al., Procedia CS 130:400-407, DOI 10.1016/j.procs.2018.04.060.
- MLP (3 camadas ocultas, dropout 0,2) sobre 68 marcos faciais do dlib; NTHU-DDD com 18 sujeitos de treino + 4 de avaliação (200 vídeos, 599.894 imagens). Acurácia **81 %**; modelo ≤ **100 KB** (vs. 547 MB do Faster-RCNN VGG16); tempo de execução **43,4 ms** medido em GTX 1080 (não no celular); app Android via JNI + dlib.

**reddy2017**: Reddy et al., CVPRW 2017, DOI 10.1109/CVPRW.2017.59.
- Compressão professor-aluno de uma CNN para Jetson TK1; dataset próprio (33 sujeitos: 25 treino / 4 validação / 4 teste, 3 classes). Modelo final: **10 MB**, 89,5 % de acurácia de teste, **18,9 ms** de inferência e **14,9 fps** ponta a ponta no TK1 (1,4 ms / 90,1 fps em GTX 1080). AlexNet-Faster-RCNN roda a 1,1 fps no TK1.

**dasgupta2019**: Dasgupta, Rahman, Routray, IEEE T-ITS 20(11):4045-4054, DOI 10.1109/TITS.2018.2879609. Sistema em smartphone em 3 estágios: PERCLOS pela câmera frontal (com iluminação NIR à noite), razão voz/não-voz do microfone, e verificação por toque. Latência numérica não extraída: não citar número.

**galarza2018**: Galarza et al., ICITS 2018, AISC 721, DOI 10.1007/978-3-319-73450-7_53. App Android (olhos, cabeça, bocejo). Números não extraídos: citar apenas como exemplo de sistema em smartphone.

**rajkumarsingh2021**: Rajkumarsingh & Totah, R&D Journal 37:26-34 (2021), DOI 10.17159/2309-8988/2021/v37a4 (acesso aberto, CC BY). Samsung Galaxy J5 Prime + Google Mobile Vision Face API (probabilidade de olho aberto/sorriso), análise a cada 1 min durante a condução; **93,3 %** (70/75 testes) combinando análise facial + detecção de excesso de velocidade; antes de dirigir, **88,5 %** só com análise fisiológica (frequência cardíaca pela câmera) e **90,4 %** fisiológica + facial (Tabelas 7 e 10 do PDF). Amostras pequenas (de 75 a 130 testes).

## 5. Ferramentas móveis: MediaPipe, MobileNetV3, quantização

**lugaresi2019**: MediaPipe (arXiv:1906.08172), framework de grafos de percepção multiplataforma, mede desempenho e consumo de recursos; código aberto.

**bazarevsky2019**: BlazeFace (arXiv:1907.05047), detector facial para GPU móvel, de 200 a mais de 1000 fps em aparelhos topo de linha; usado como estágio 1 do Face Mesh.

**kartynnik2019**: Face Mesh (arXiv:1907.06724), malha 3D de **468 vértices** a partir de câmera monocular, de 100 a mais de 1000 fps em GPU móvel; base do Face Landmarker do MediaPipe usado para o EAR.

**howard2019**: MobileNetV3, ICCV 2019, pp. 1314-1324, DOI 10.1109/ICCV.2019.00140.
- Latências medidas com TFLite em um único núcleo grande, batch 1 (Pixel 1/2/3). Float: **V3-Large 1.0: 75,2 % top-1, 219 M MAdds, 5,4 M parâmetros, 51/61/44 ms**; V3-Large 0.75: 73,3 %, 39 ms (P-1); **V3-Small 1.0: 67,4 %, 66 M MAdds, 2,9 M parâmetros, 15,8/19,4/14,4 ms**; V3-Small 0.75: 65,4 %, 12,8 ms. Quantizado int8 (Tabela 4): V3-Large 73,8 %, 44/42,5/31,7 ms; V3-Small 64,9 %, 15,5/14,9/10,7 ms; V2 1.0 70,9 %, 52 ms.
- V3-Large é 3,2 % mais preciso que V2 com 20 % menos latência; V3-Small é 6,6 % mais preciso que V2 com latência comparável.

**jacob2018**: Jacob et al., CVPR 2018, pp. 2704-2713, DOI 10.1109/CVPR.2018.00286. Esquema de quantização inteira (int8) com treino consciente de quantização; ResNet no ImageNet: float 76,4/78,0/78,8 % vs. int8 74,9/76,6/76,7 %, ou seja, "dentro de 2 %" do ponto flutuante.

**krishnamoorthi2018**: whitepaper (arXiv:1806.08342). A quantização pós-treino de pesos (por canal) e ativações (por camada) em 8 bits fica **dentro de 2 %** da acurácia float para várias CNNs; modelo 4× menor; de 2 a 3× mais rápido em CPU e até 10× em DSPs com SIMD de ponto fixo.

## 6. Medidas clássicas: PERCLOS, EAR, KSS

**wierwille1994**: NHTSA DOT HS 808 247 / VPISU ISE 94-04 (dez. 1994, 247 p.; registro TRID 00712943 em trid.trb.org/view/448128). Estudo em simulador que originou o PERCLOS: proporção do tempo (por minuto) em que os olhos estão pelo menos 80 % fechados.

**dinges1998**: FHWA-MCRT-98-006 (out. 1998; ROSAP DOI 10.21949/1502740, autor corporativo FMCSA Technology Division). Entre as medidas oculares avaliadas, PERCLOS foi "a determinação mais confiável e válida do nível de alerta"; reflete fechamentos lentos das pálpebras (droops), não piscadas.

**soukupova2016**: Soukupová & Čech, 21st CVWW, Rimske Toplice, 3 a 5 fev. 2016 (PDF oficial em cmp.felk.cvut.cz conferido). Define o **EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2‖p1−p4‖)** a partir de 6 marcos do olho e detecta piscadas com SVM sobre janela temporal do EAR (tempo real, câmera comum). Observação: o UTA-RLDD usa a variante sem o fator 2 no denominador.

**kazemi2014**: Kazemi & Sullivan, CVPR 2014, DOI 10.1109/CVPR.2014.241. Alinhamento facial por ensemble de árvores de regressão (~1 ms), implementado no dlib (68 marcos) e usado por soukupova2016 e jabbar2018.

**akerstedt1990**: Åkerstedt & Gillberg, Int. J. Neuroscience 52(1-2):29-37, DOI 10.3109/00207459008994241. Escala KSS (1 extremamente alerta … 9 extremamente sonolento); em 8 sujeitos mantidos acordados, correlações intraindividuais altas entre sonolência subjetiva e sinais EEG/EOG; alterações fisiológicas tendem a aparecer só em sonolência extrema.

## 7. Revisões

**perkins2023**: Perkins, Sitaula, Burke e Marzbanrad, IEEE Transactions on Intelligent Vehicles 8(2):1319-1338, 2023, DOI 10.1109/TIV.2022.3224690 (arXiv:2109.08355).
- Revisão do que falta para levar a previsão de sonolência do motorista ao uso real.
- O que o artigo usa (Seções I e II): só 15 dos 126 estudos revisados usam dados de estrada; muitos estudos põem trechos da mesma pessoa no treino e no teste; os estudos comportamentais costumam pedir aos voluntários que encenem a sonolência.
- É a referência que apoia a lacuna de método da Introdução (pouca sonolência real e partições que misturam a mesma pessoa).

**sikander2019**: IEEE T-ITS 20(6):2339-2352. Revisão de sistemas de detecção de fadiga; nota que resultados promissores são obtidos "em ambientes restritos" e que muito progresso ainda é necessário.

**albadawi2022**: Sensors 22(5):2069. Revisão da última década em 4 categorias (comportamentais, fisiológicas, veiculares, híbridas).

**ngxande2017**: PRASA-RobMech 2017, pp. 156-161. Revisão de medidas comportamentais com SVM, CNN e HMM.

**fu2024**: survey IEEE T-IV 2024 (ver Seção 3).

## 8. Outros datasets (ver `dataset_facts.md`)

massoz2016 (DROZY), abtahi2014 (YawDD), ortega2020 (DMD), petrellis2022 (NITYMED), fl3d, nthuddd.

## Itens solicitados que NÃO foram encontrados
- **barrett2019**: nenhuma publicação de 2019 com esse sobrenome relacionada a sonolência/vazamento de sujeito foi localizada. Não incluída no .bib para não inventar referência.
