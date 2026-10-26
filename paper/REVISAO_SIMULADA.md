# Revisão por pares simulada — Entrega 03a (artigo IEEE, versão final)

**Artigo:** "Detecção de Sonolência em Dispositivos Móveis: Avaliação da Generalização entre Sonolência Atuada e Real" (Equipe 14, Turma 8ºA).

**Atualização de 06/10/2026:** depois desta revisão, o texto foi enxugado para 7 páginas e sem travessões, a Fig. 1 foi redesenhada em preto e branco e a tabela do Exp. 3 por condição passou para o texto. Os números não mudaram. As linhas e tabelas citadas abaixo são as da versão de 28/09.

**Versão revisada:** `paper/artigo.pdf` compilado em 28/09/2026 às 00:59, já aprovado pela conferência automática dos números (`src/experiments/conferir_numeros.py`). Tinha 8 páginas, 29 referências e 6 tabelas. As linhas citadas abaixo como "l." são as do `artigo.tex` dessa versão. As correções da seção 6 foram aplicadas depois, e o artigo corrigido também tem 8 páginas.

**Postura:** revisão exigente, no estilo do "Reviewer 2" dos slides da disciplina, que critica falta de novidade técnica, avaliação restrita a variantes de uma mesma base, espaço gasto com conceitos básicos e figuras redundantes.

**Rubrica da disciplina:**
1. novidade e relevância;
2. clareza, referências e escrita;
3. qualidade técnica e avaliação experimental;
4. facilidade de implementação;
5. recomendação.

**Escala:** 1 = insuficiente, 2 = regular, 3 = bom, 4 = excelente.

**Material lido:**
- o PDF inteiro, com texto extraído por PyMuPDF e as 8 páginas renderizadas;
- `artigo.tex`, `numeros.tex` e `refs.bib`;
- os relatórios finais `results/exp1_intra.md`, `exp2_leakage.md`, `exp3_eye_state.md`, `exp5a_validacao.md`, `exp5b_transferencia.md` e `exp6_noite.md`;
- `results/app_validacao.md`, `app_validacao_dmd.md` e `exp4_phone_latency.{md,csv}`;
- os benchmarks brutos em `results/phone/*.json|*_frames.csv`;
- os CSV `exp5b_transferencia.csv` e `exp6_noite.csv`, relidos por código próprio;
- `app/pwa/app.js` (laço de quadros e câmera) e `app/pwa/README.md`;
- `paper/related_work_notes.md` e `paper/fontes_brasil.md`;
- as notas de fechamento.

---

## 1. Parecer resumido

O artigo responde à pergunta do título com um experimento direto e bem controlado, a transferência DMD → UTA-RLDD:
- usa o mesmo extrator nas duas pontas;
- tem uma referência intradomínio nos mesmos *folds* oficiais;
- dá IC por *bootstrap* de sujeitos;
- inclui três controles que questionam o próprio resultado.

Separa também, com os mesmos motoristas, o efeito da noite e o do carro em movimento sobre o sinal ocular. Esse resultado não é trivial e está bem delimitado. A rastreabilidade é exemplar: todo número sai de `results/` por macros, com conferência independente.

As fraquezas estão na leitura dos resultados, não nos dados:
- **Comparação injusta.** O resumo e a Discussão comparavam a taxa de disparo dos classificadores, no limiar arbitrário de 0,5, com a de uma regra calibrada por sujeito. O próprio Exp. 6 mostra que, com o limiar calibrado no treino ou com a regressão logística, os modelos disparam tão pouco quanto a regra.
- **Confusão não declarada.** A queda da PP1 era atribuída ao "tipo de sonolência". Não se dizia que ela também depende do volume do treino: 510 janelas de 13 motoristas contra as de 48 sujeitos por *fold*.
- **Validação dentro da amostra.** A validação do app era apresentada como validação, mas as regras foram escolhidas nas mesmas sessões e nos mesmos clipes.
- **Comparações entre tarefas com acasos diferentes.** A BSA de 86,9% em B3 tem acaso de 68,6%.

Nada disso exige experimento novo para ser dito com precisão. As correções de texto foram aplicadas nesta rodada (seção 6), sem sair de 8 páginas.

**Recomendação:** aceitar com revisões obrigatórias. Todas são textuais e já foram aplicadas. As pendências experimentais estão na seção 5; nenhuma impede a entrega.

## 2. Notas por critério

| Critério | Versão recebida | Após as correções | Justificativa |
|---|---|---|---|
| 1. Novidade e relevância | **3** | **3** | A contribuição é metodológica, não técnica: RF e RL sobre 11 características manuais e um app de regras. O desenho de avaliação é original no escopo: mesmo extrator nas duas pontas, referência intradomínio, controles do vídeo de referência e separação entre noite e movimento com os mesmos motoristas. A relevância está bem argumentada com dados brasileiros, com a ressalva correta de que eles não isolam o sono. Contra: a conclusão "atuada → real" vem de um único par de bases, e a busca que sustenta a novidade é limitada ("nenhum *desses* trabalhos"). |
| 2. Clareza, referências e escrita | **3** | **3–4** | O texto é preciso e as referências foram verificadas em fonte primária, sem nenhuma UNVERIFIED citada. Contra, na versão recebida: <br>• resumo com PP1–PP3 sem definição e com 90% binário contraposto a 54–65% em três classes sem dizer isso; <br>• BSA comparada entre tarefas com acasos diferentes; <br>• equação do EAR exibida, um conceito básico; <br>• Tabela II redundante com a coluna "Papel" da Tabela I; <br>• parágrafo da Seção IV-C repetindo a Fig. 1 e parágrafo da Seção VI-E repetindo quase toda a Tabela VI; <br>• frase-roteiro das seções. |
| 3. Qualidade técnica e avaliação | **3** | **3** | Pontos fortes: partição por sujeito, limiar fixado antes, IC por sujeito, 8 variantes de sensibilidade, controles contra o próprio resultado, validação do detector contra anotação quadro a quadro e do extrator contra as características oficiais. Contra: as fraquezas M1–M8 (seção 4). As que dependem de experimento continuam abertas: B com 13 sujeitos, validação do app fora do ajuste e a versão 2.1 medida no celular. Por isso a nota não sobe. |
| 4. Facilidade de implementação | **3** | **3** | Há código, *logs*, CSV, macros com origem, conferência automática, um comando que refaz figura, números, artigo e pôster, hiperparâmetros no texto e uma PWA que roda. Contra: <br>• o repositório citado no artigo (l.85) não tem o trabalho descrito, pois o último *push* foi em 26/08, segundo o item 5 das notas de fechamento; <br>• o DMD exige formulário; <br>• o app Android não foi compilado; <br>• a versão descrita (2.1) não foi medida no celular. |

**Recomendação:** aceitar com revisões obrigatórias, já aplicadas.

## 3. Pontos fortes

1. **Um experimento direto para a pergunta do título.**
   - O DMD encenado e a UTA-RLDD real passam pelo mesmo extrator (MediaPipe) e pelas mesmas 11 características.
   - A referência intradomínio (B) usa os *folds* oficiais.
   - A queda B − C é pareada: 10,9 [6,7; 15,7] pontos de AUC por janela e 11,9 [5,6; 18,8] por vídeo.
   - O mesmo padrão aparece na RL.
2. **Controles contra o próprio resultado.**
   - (i) Entre KSS 8–9 e 6–7, B e C empatam (queda de 1,2 [−3,0; 5,8]).
   - (ii) Sem centragem, a queda do RF não é detectável.
   - (iii) O alerta com o carro em movimento piora C.
   - Poucos trabalhos da área mostram quando o próprio número encolhe.
3. **Noite contra movimento.** Com os mesmos motoristas e a mesma câmera, o carro em movimento muda a SNR do EAR em −5,4 [−6,6; −3,5], e a diferença entre a noite e o movimento não se distingue de zero. Esse resultado muda a leitura usual de "a noite é o problema".
4. **Honestidade.**
   - A versão do app é dita em cada medida.
   - O iPhone tem n = 2 e o Galaxy S23 Ultra aparece como "medição prevista".
   - O Android não foi compilado.
   - O NITYMED não é tratado como sonolência real.
   - O falso alarme conhecido é descrito com a causa (gA_5: olhar para baixo).
   - O falso alarme da versão 1.0 com um autor acordado é contado como lição.
5. **Rastreabilidade.** São 576 números em macros, gerados de `results/`, com conferência independente (576/576) e compilação em teste.
6. **Detector validado antes do uso.** Contra a anotação quadro a quadro do DMD, o F1 de piscadas é 0,749 e o κ por quadro é 0,773, com o fator 0,75 fixado antes. O Exp. 5a usa permutação pareada e LOSO.
7. **Ética e LGPD.**
   - Nenhum rosto foi publicado.
   - As licenças de cada base são respeitadas.
   - O processamento fica no aparelho.
   - O texto trata do risco de falsa segurança ("um alarme ausente não significa condutor alerta").

## 4. Fraquezas concretas (com localização)

"Maior" indica que a fraqueza muda a conclusão ou o resumo; "moderada", que muda uma interpretação; "menor", que é de forma. O estado depois desta rodada vem em "Correção".

### M1 (maior). Comparação da taxa de disparo injusta e seletiva

**Onde:** resumo (l.63), Seção VI-C (l.273, p. 5–6), Discussão PP3 (l.344, p. 7) e Conclusão (l.354).

**Problema:** o argumento que justificava o desenho do app era "classificadores aprendidos disparam em 26,0–43,6% das janelas de pessoas acordadas, contra 2,6% de uma regra". Isso compara modelos no limiar 0,5, arbitrário para um RF com classes balanceadas treinado com 83% de janelas sonolentas, com uma regra que usa o limiar de EAR do próprio sujeito. O próprio Exp. 6 (`results/exp6_noite.md`, seções 5.4 e 5.11) mostra outra coisa:
- Com o limiar calibrado só no treino, o RF do DMD marca 1,7% das janelas alertas e 11,0% das sonolentas da UTA-RLDD.
- A RL do DMD, no limiar 0,5, marca 2,3% e 18,5%.
- A regra marca 2,6% e 24,1%.
- Com o carro em movimento, o RF calibrado e a regra disparam em 21,9% e 23,8% das janelas de motoristas acordados.

A conclusão honesta é outra. Com falso alarme semelhante, a regra acha mais janelas sonolentas que os modelos do DMD, sem treino e com explicação para cada alarme. Nenhuma das duas abordagens resolve o carro em movimento.

**Correção:** o texto foi reescrito nos quatro lugares, com 5 macros novos. O resumo diz "no limiar 0,5" e traz a comparação com falso alarme semelhante. A legenda da Fig. 3 diz "limiar 0,5".

**Observação não aplicada, por falta de espaço:** o 2,6% da regra na UTA-RLDD também se beneficia do vídeo de referência. O limiar de EAR sai do último terço do vídeo 0, de onde vêm as janelas alertas. No DMD parado, cuja referência está em outra sessão, a mesma regra dispara em 14,9% (Fig. 3C).

### M2 (maior). A PP1 atribuía ao tipo de sonolência uma queda que mistura outros fatores

**Onde:** Discussão PP1 (l.340), Limitações (l.346) e Conclusão (l.354).

**Problema:** "Treinar em sonolência atuada custa 10,9 pontos" generaliza a partir de **um único par de bases**, e C e B diferem em muito mais que o tipo de sonolência:
- C aprende com 510 janelas (87 alertas) de 13 motoristas; B aprende, a cada *fold*, com as janelas de 48 sujeitos, de um total de 12.487;
- a câmera e o cenário são outros: carro parado contra gravação caseira;
- os rótulos têm naturezas diferentes: evento encenado contra KSS do vídeo.

O relatório do Exp. 5b já dizia que "nenhuma análise aqui separa esses fatores"; o artigo não dizia.

**Correção:**
- A PP1 da Discussão passou a dizer "treinar no DMD em vez de na UTA-RLDD" e declara a confusão com os números (510 janelas de 13 motoristas contra as de 48 sujeitos).
- A limitação (2) foi ampliada.
- A Conclusão menciona "treino menor".
- O trabalho futuro "treinar B com tantos sujeitos quanto o DMD" foi acrescentado.

### M3 (maior). Resumo sem a ressalva decisiva e controle relatado só para o RF

**Onde:** resumo (l.63) e controle (ii) da Seção VI-A (l.212).

**Problema:**
- A queda aparece no resumo como resposta limpa à PP1. Mas os controles mostram que ela se concentra no contraste com o vídeo alerta, que é também a referência da centragem.
- Sem centragem, o texto só dizia que a queda do RF some. Na RL ela persiste: 7,3 [3,0; 11,7] pontos (`exp5b_transferencia.md`, seção 7).
- No esquema B2 (sonolento contra os demais), a queda do RF não é detectável: 2,9 [−0,2; 6,6].
- O texto também não usava a robustez que os dados dão: em 7 das 8 variantes do RF, o IC da queda fica acima de zero.

**Correção:**
- O resumo ganhou a frase "A queda se concentra no contraste com o vídeo alerta, também referência da normalização".
- O controle (ii) agora traz o resultado da RL.
- A Seção VI-A traz a frase das 7 de 8 variantes.
- O B2 não entrou, por falta de espaço (recomendação 5.6).

### M4 (maior). A validação do app era de desenvolvimento, não de teste

**Onde:** Seção IV-C (l.166), Seção VI-E (l.311), Tabela VI, contribuição 4 (l.85) e Conclusão (l.354).

**Problema:** `results/app_validacao_dmd.md` mostra que a regra dos olhos foi escolhida comparando variantes nas mesmas sessões s5 e s1 do DMD ("escolha da regra final"). O "ajuste final" também foi decidido nesses dados. Em `results/app_validacao.md`, a passagem da versão 2.0 para a 2.1 foi motivada pelos mesmos 14 clipes do navegador: o Fold3/31 alerta ficava 27,2% do tempo em amarelo na 2.0. Logo, os números da tabela e do resumo são de desenvolvimento e tendem a ser otimistas:
- 3,3% do tempo em vermelho;
- 16 de 17 sessões sem alarme;
- κ de 0,938;
- 100% do tempo em verde nos vídeos alertas.

A frase "os pesos não foram ajustados em dados de sonolência real" era verdadeira, mas incompleta.

**Correção:** a ressalva agora aparece em quatro lugares:
- na Seção IV-C ("foram escolhidos nos dados da validação");
- na Seção VI-E ("os números são de desenvolvimento, e não de um teste independente");
- na limitação (6);
- na Conclusão ("dado de desenvolvimento").

O trabalho futuro "validar o app fora das sessões do ajuste" foi acrescentado.

### M5 (moderada). Acurácias comparadas entre tarefas com acasos diferentes

**Onde:** Seção VI-B (l.215), Discussão PP1 (l.340) e resumo (l.63).

**Problema:**
- "Sem a classe intermediária (B3), chega a BSA de 86,9%": a BSA é acurácia simples, e em B3 a classe majoritária já dá 68,6% (`exp1_intra.md`).
- "BSA de 54,5 contra 86,9%" compara três classes (acaso de 44,4) com o binário (acaso de 68,6).
- No resumo, "acima de 90%" (NTHU-DDD, binário) aparecia contraposto a "entre 54 e 65%" (UTA-RLDD, três classes) sem dizer que as tarefas diferem.

**Correção:**
- As comparações entre tarefas passaram para acurácia balanceada: 57,8 contra 84,8%, com acaso de 33,3 e 50%.
- O resumo diz "em três classes".
- A legenda da tabela do Exp. 1 e 2 define BSA e VA.

### M6 (moderada). O rótulo "vazamento de identidade" não corresponde à evidência

**Onde:** PP1 na Introdução (l.78), contribuição 2 (l.83) e Conclusão (l.354).

**Problema:** o próprio Exp. 2 mostra outra coisa. A partição que isola a identidade (por vídeo, com os outros vídeos do sujeito no treino) fica **abaixo** do protocolo por sujeito (−7,0 pp). A inflação de +45,3 pp vem de janelas sobrepostas do **mesmo vídeo**, e portanto do mesmo rótulo, nos dois lados. O relatório diz "vazamento de vídeo e de identidade".

**Correção:**
- A PP1 passou a falar de "partição com vazamento entre treino e teste".
- A contribuição 2 fala de "vazamento entre treino e teste (partição por janela, por vídeo e por sujeito)".
- A Seção VI-B ganhou a frase "o que infla a acurácia é o mesmo vídeo, e com ele o mesmo rótulo, aparecer nos dois lados […]; a mesma pessoa com outro rótulo não a infla".
- A Conclusão fala de "vazamento por janelas do mesmo vídeo".
- A palavra-chave foi mantida, porque o artigo discute a identidade.

### M7 (moderada). Ausência de evidência lida como evidência de equivalência

**Onde:** Seção VI-B, Exp. 5a (l.242).

**Problema:** "O custo não se distingue de zero. […] Isso autoriza usar o nosso extrator" é uma leitura forte demais:
- O IC do custo vai de −4,8 a +8,5 pp.
- No nível das características, a concordância por vídeo é modesta: Spearman de 0,17 a 0,54 (`exp5a_validacao.md`).
- Com as bordas das piscadas no limiar, o custo sobe para +9,3 pp.

**Correção:** o texto diz "não há perda detectável […], embora o IC largo não prove equivalência; com essa ressalva, usamos o nosso". Os Spearman não entraram, por falta de espaço.

### M8 (moderada). Números do iPhone apresentados de forma enganosa

**Onde:** resumo (l.63), Seção VI-D (l.278), Tabela V (l.282), PP2 (l.342) e Conclusão (l.354).

**Problemas:**
- **"Mediana de 2 execuções" = 17,0 ms.** Com n = 2, a mediana é a média de 11,0 e 23,0 ms, um valor que nenhuma execução observou. A diferença de 2× entre as execuções não tem explicação: mesmo aparelho, mesma resolução (480×640), mesmo *delegate*.
- **"Acima dos 30 da câmera, o que indica que o laço reprocessa o quadro mais recente."** A taxa da câmera não foi registrada: o JSON do benchmark não a guarda, e o `getUserMedia` não fixa `frameRate`. O laço (`app/pwa/app.js`, l.767–777) roda por `requestAnimationFrame` e só processa quando `video.currentTime` muda. Os 40,5–60,0 por segundo são a taxa do laço, não a vazão.
- **Resolução do relógio.** Os tempos no iPhone vêm em milissegundos inteiros, o que indica essa resolução. O LSTM, com 1,0 / 1,0 ms, está no limite da medida.

**Correção:**
- O resumo, a PP2 e a Conclusão dão a faixa, de 11,0 a 23,0 ms.
- A Seção VI-D diz "variou de 11,0 a 23,0 ms entre as execuções, diferença não explicada" e reescreve a frase do laço ("número que não mede a vazão, pois a taxa da câmera não foi registrada").
- A linha da tabela virou "Quadros/s (laço)".
- A resolução do relógio não entrou, por falta de espaço.

### Fraquezas menores

- **m9. Equivalência afirmada (Conclusão, l.354).** "A perda à noite tem o tamanho da produzida pelo carro em movimento" afirma equivalência a partir de uma diferença não significativa (−1,6 [−3,7; 1,1]). *Correção:* "é compatível com".
- **m10. Linguagem causal (l.210, l.340).** "A razão aparece nas características brutas" e "mecanismo visível" atribuem causa a uma observação descritiva. *Correção:* "Uma explicação aparece" e "mecanismo plausível".
- **m11. PP1–PP3 no resumo (l.63).** O resumo usava esses rótulos sem defini-los. *Correção:* retirados.
- **m12. Redundância e conceitos básicos (a crítica clássica do Reviewer 2).** *Correção:* todos condensados; o espaço liberado foi usado em M1–M8.
  - equação exibida do EAR (l.146–149);
  - Tabela II, experimentos × perguntas (l.172–194), redundante com a coluna "Papel" da Tabela I;
  - parágrafo da Seção IV-C (l.166) repetindo a caixa "Nível 0–100" da Fig. 1;
  - Seção VI-E (l.311) repetindo quase toda a Tabela VI;
  - Discussão PP3 (l.344) repetindo os seis valores da Tabela IV;
  - frase-roteiro das seções (l.87).

  Com isso, a Tabela II virou uma frase na Seção V, e as tabelas seguintes foram renumeradas (II a V).
- **m13. Legenda da Fig. 3 (l.269).** Não dizia que o disparo é no limiar 0,5 nem que o RF do DMD é avaliado fora do sujeito no próprio DMD. *Correção:* acrescentado.
- **m14. Legenda da tabela do Exp. 1 e 2 (l.219).** Não definia BSA e VA. *Correção:* acrescentado.
- **m15. Reprodutibilidade (l.85).** O artigo diz "código, *logs* e resultados estão em github.com/BrunoRodolfoo/…", mas o trabalho novo não está lá. *Não corrigível aqui:* o *push* depende do aval do André. É **bloqueante antes de 09/11**.
- **m16. Exp. 3 frouxamente ligado ao sistema final.** No app, a CNN só é exibida. Além disso, CEW → MRL cruza RGB e infravermelho, de modo que o efeito das condições também se confunde com a modalidade. O texto já diz "associações". *Sem mudança*, por falta de espaço.
- **m17. Novidade limitada à amostra citada.** A afirmação de novidade vale para "desses trabalhos". O joshi2020, já verificado em `related_work_notes.md` (AUC 0,78 em dados reais *in-the-wild*), poderia reforçar a lacuna. *Sem mudança*, por falta de espaço.

## 5. O que exige novos experimentos (não corrigido; sugerido para a Entrega 02 e o pôster)

1. **B com 13 sujeitos.** Subamostrar repetidamente a UTA-RLDD para o mesmo número de sujeitos do DMD, e se possível o mesmo número de janelas alertas. Isso separaria o tipo de sonolência do volume do treino (M2). É barato: só roda de novo o `exp5_transfer.py` com subamostragem. **Prioridade 1.**
2. **Regra contra modelos por ponto de operação.** Traçar a curva ROC dos modelos contra o ponto da regra, com falso alarme fixo na UTA KSS 0 e na s1 do DMD (M1). Os dados já estão em `exp6_noite_janelas.csv` e `exp5b_predicoes.csv`.
3. **App fora do ajuste.** Validar o app em sessões ou pessoas que não entraram na escolha das regras. Uma opção é validação cruzada por motorista das escolhas feitas no DMD; outra, campo com voluntários e KSS (M4).
4. **Versão 2.1 medida no celular.** Pelo menos 3 execuções no iPhone, registrando a taxa da câmera (`track.getSettings().frameRate`), e o Galaxy S23 Ultra (M8).
5. **Queda da cabeça.** Medir a revocação com uma anotação manual de cabeceios; hoje só há teste sintético.
6. **Esquema B2 no texto.** Relatar o B2 (sonolento contra os demais, o mais próximo da decisão do app). O número já existe no relatório do Exp. 5b.
7. **Vídeo de referência na UTA-RLDD.** Isolar esse efeito com um trecho alerta gravado no mesmo dia do vídeo sonolento.

## 6. Correções aplicadas em `artigo.tex` nesta revisão

1. **Resumo, reescrito (M1, M3, M5, M6, M8, m11).** Tem 250 palavras pela mesma contagem usada antes (eram 249). Mudou em cinco pontos:
   - "em três classes";
   - a ressalva do vídeo de referência;
   - o vazamento descrito como janelas do mesmo vídeo;
   - o disparo "no limiar 0,5", com a regra comparada com falso alarme semelhante;
   - o iPhone por faixa, com n = 2 e sem os rótulos PP.
2. **Introdução (M6, m12).**
   - A PP1 fala de "partição com vazamento entre treino e teste".
   - A contribuição 2 fala de "vazamento entre treino e teste (partição por janela, por vídeo e por sujeito)".
   - A frase-roteiro das seções foi retirada.
3. **Seções II e III (m12).**
   - O parágrafo "Bases" não repete a Tabela I.
   - A "Troca do NTHU-DDD" ficou mais curta, com o mesmo conteúdo.
4. **Seção IV-A (m12).** A equação exibida do EAR virou definição na linha, com a citação de Soukupová e Čech.
5. **Seção IV-C (M4, m12).**
   - O parágrafo do nível remete à Fig. 1 em vez de repeti-la.
   - Ganhou a ressalva de que a regra dos olhos e os ajustes finais foram escolhidos nos dados da validação.
6. **Seção V (m12).** A Tabela II (mapa dos experimentos) virou uma frase, e a frase da reprodutibilidade ficou mais curta.
7. **Seção VI-A (M3, m10).**
   - "Uma explicação aparece".
   - Controle (ii): "na RL, a queda persiste (7,3 [3,0; 11,7])".
   - Frase nova: o IC da queda do RF fica acima de zero em 7 de 8 variantes; a exceção é a sem centragem.
8. **Seção VI-B (M5, M6, M7).**
   - B3 em acurácia balanceada: 84,8%, com acaso de 50%.
   - Frase nova: o que infla é o mesmo vídeo com o mesmo rótulo, não a mesma pessoa.
   - Exp. 5a: "não há perda detectável […], embora o IC largo não prove equivalência".
9. **Seção VI-C (M1, m12, m13).**
   - Parágrafo do disparo com o limiar calibrado no treino (1,7 / 11,0%), a RL (2,3 / 18,5%), a regra (2,6 / 24,1%) e o carro em movimento (21,9 e 23,8%).
   - Parágrafo dos óculos mais curto.
   - O Exp. 3 cita só o modelo final nos recortes oficiais.
   - Legenda da Fig. 3 com o limiar 0,5 e o modelo do DMD avaliado fora do sujeito.
   - A Fig. 3 é declarada no início da subseção, e por isso volta a ficar na página 6.
10. **Seção VI-D (M8).**
    - As KB da CNN saíram do texto; estão na tabela.
    - O desktop aparece sem repetir números.
    - O iPhone aparece por faixa entre execuções, com "diferença não explicada".
    - O laço é "disparado a cada atualização da tela […] não mede a vazão".
    - A legenda da tabela diz "faixa entre execuções no texto", e a linha virou "Quadros/s (laço)".
11. **Seção VI-E (M4, m12).**
    - O texto não repete a tabela; mantém a interpretação: κ contra o EAR sozinho, bocejo com a mão, gA_5, os 3 vídeos sonolentos no navegador e a cabeça.
    - Ganhou a ressalva "números de desenvolvimento".
12. **Discussão PP1 (M2, M5, m10).**
    - "Treinar no DMD (atuado) em vez de na UTA-RLDD (real)".
    - Confusão com o volume do treino (510 janelas de 13 motoristas contra as de 48 sujeitos por *fold*), a câmera, o cenário e o rótulo.
    - "Mecanismo plausível".
    - Acurácia balanceada com os acasos.
    - "O vazamento é, nestas medidas, o de maior magnitude".
13. **Discussão PP2 (M8).** Faixa de 11,0 a 23,0 ms, sem a "mediana de 2".
14. **Discussão PP3 (M1, m12).**
    - Remete à Tabela III em vez de repetir os valores.
    - Traz o argumento do ponto de operação.
    - "Nenhum dos dois resolve o carro em movimento (21,9 e 23,8%)".
15. **Limitações (M2, M4).**
    - (1) ficou mais curta.
    - (2) ganhou as 87 janelas alertas e diz que nenhuma análise separa o tipo de sonolência do volume do treino e da câmera.
    - (6) diz que foram "os mesmos dados usados para escolher as regras".
16. **Conclusão e trabalhos futuros (M2, M4, M6, M8, m9).**
    - "Compatível com".
    - "Treino menor".
    - "Vazamento por janelas do mesmo vídeo".
    - A regra com falso alarme semelhante.
    - "Dado de desenvolvimento".
    - Faixa do iPhone.
    - Trabalhos futuros novos: B com tantos sujeitos quanto o DMD, e o app validado fora do ajuste.
17. **Numeração.** As tabelas passaram a ser I a V, com a antiga II removida. Todas as chamadas usam `\ref`.

### Números novos (todos de `results/`, pelos macros)

Os macros novos estão na função `revisao_pares()` de `src/experiments/make_paper_numbers.py`, que agora gera 576 números e 1.164 macros.

| Macro | Valor | Origem |
|---|---|---|
| `\ExpSeisDisparoModeloDMDCalUTAAlerta`, `…CalUTASonolento`, `…CalDMDMovimento` | 1,7 [0,8; 2,8]; 11,0 [7,9; 15,1]; 21,9 [9,3; 40,7] | `results/exp6_noite.csv`, seção c, "Modelo do DMD", `sem_normalizacao`, `taxa_disparo_janela_limiar_calibrado` |
| `\ExpSeisDisparoModeloDMDLRUTAAlerta`, `…LRUTASonolento` | 2,3 [1,1; 3,9]; 18,5 [12,9; 24,9] | `results/exp6_noite.csv`, seção c, "Exp. 5b: exp5b_final_C_sem_normalizacao_LR.joblib" |
| `\ExpCincoBLRSemNormQuedaAUCJanela` | 7,3 [3,0; 11,7] | `results/exp5b_transferencia.csv`, `queda_B_menos_C`, `sem_normalizacao`, LR, B3, janela, AUC |
| `\ExpCincoBSensVariantesICPositivo` / `…Total` | 7 / 8 | `results/exp5b_transferencia.md`, seção 14 (conferido também no CSV: a exceção é `sem_normalizacao`) |
| `\ExpCincoBSujeitosTreinoB` | 48 | derivado: `\BaseUTASujeitos` × 4/5 (5 *folds* oficiais de 12) |

Os únicos números novos escritos à mão são os acasos da acurácia balanceada (33,3 e 50%). São constantes de método.

### Verificações feitas

- **Compilação:** `tectonic` gera 8 páginas, com 0 erros, 0 *Overfull* e 0 citações, referências ou macros indefinidos.
- **Resumo:** 250 palavras pela mesma contagem que deu 249 na versão anterior.
- **Ramo do Galaxy S23 Ultra:** simulado fora do repositório, com `\SamsungMedidotrue` e os macros do Samsung como travessão. Compila com 8 páginas e 0 *Overfull*.
- **Números:** `src/experiments/conferir_numeros.py` confere 576/576, com 0 divergências e LaTeX ok. Os novos números foram também relidos à mão nos CSV.
- **Consistência com o estado do projeto:**
  - o iPhone nunca aparece como S23 Ultra;
  - o S23 Ultra continua como "medição prevista";
  - o NITYMED nunca aparece como sonolência real;
  - nenhum rosto foi publicado;
  - o Android continua "não compilado";
  - a versão 2.1 continua "não medida no aparelho".

## 7. Pendências para a equipe

1. **Repositório (bloqueante antes de 09/11).** Fazer o *push* do trabalho novo, com o aval do André. Até lá, a frase "código, *logs* e resultados estão em …" da contribuição 4 não é verdadeira.
2. **PRF.** Conferir no navegador a URL da PRF e as datas de acesso (pendência herdada).
3. **Novos benchmarks do iPhone.** Depois de medir, rodar `bash tools/atualizar_celular.sh` e reler:
   - a Seção VI-D;
   - a Tabela IV (antiga V);
   - a PP2;
   - o resumo.

   O texto já usa a faixa entre execuções e o número de execuções por macro, e continua certo com n > 2.
4. **Entrega 02 (31/10).** Se houver tempo, rodar o experimento 5.1 (B com 13 sujeitos). Se a queda sobreviver, o argumento da PP1 fica muito mais forte para a revisão por pares de 12/11.
5. **Pôster.** O pôster usa os mesmos números, por `poster/numeros.json`. Vale revisar, no mesmo espírito, a frase do disparo e a do vazamento, e rodar de novo `poster/build_poster.py`, que não foi alterado nesta rodada.
