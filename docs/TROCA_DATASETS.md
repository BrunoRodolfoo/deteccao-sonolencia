# Troca de bases: NTHU-DDD → DMD + NITYMED

**Para:** Prof. Júlio César Nievola (orientador)
**De:** Equipe 14 (8ºA). André Luís O. F. da Silva, Bruno Rodolfo da Silveira, Bruno Sandoval Ribeiro, Paulo Henrique Perin
**Data:** 27/09/2026
**Assunto:** substituição da base de sonolência atuada prevista na Entrega 01
**Situação:** **aprovada.** O orientador autorizou a troca verbalmente, sem pedir documento formal (registrado em 04/10/2026).
**Atualizado em 26/10/2026:** a seção 5 traz a leitura dos resultados depois dos controles do Exp. 5b (`results/exp5b_robustez.md`).

## 1. O que aconteceu

O planejamento da Entrega 01 usava duas bases:
- o **NTHU-DDD**, com sonolência atuada, para o treino;
- a **UTA-RLDD**, com sonolência real, para o teste.

O acesso ao NTHU-DDD depende de um *Dataset License Agreement* assinado por chefia institucional e enviado por e-mail aos autores.
O pedido não foi atendido até **20/09**, a data-limite fixada no próprio planejamento. Nenhum arquivo do NTHU-DDD foi baixado ou usado.

Em 20/09 adotamos a contingência prevista no plano:
- o vazamento entre treino e teste na UTA-RLDD (Exp. 2);
- a generalização do estágio ocular entre MRL Eye e CEW (Exp. 3).

Faltava, porém, um treino **atuado** para responder diretamente à PP1, e uma condição **noturna** para a PP3.
Em 27/09 obtivemos duas bases públicas que não exigem assinatura institucional e rodamos com elas os experimentos finais.

## 2. As bases substitutas

| | **DMD** (Vicomtech; Ortega et al., ECCV Workshops 2020) | **NITYMED** (Petrellis et al., IEEE DataPort 2022) |
|---|---|---|
| Papel | substitui o NTHU-DDD como **treino atuado** (PP1) | cobre a condição **noite** (PP3) |
| Sonolência | **encenada** ("These states were acted"): microssonos, bocejos, pálpebra pesada; carro parado | condução noturna **real**, com carros em movimento. Bocejos e microssonos de autenticidade **não documentada**; os 3 bocejos por vídeo sugerem que foram feitos sob instrução |
| Volume usado | 13 motoristas com sessão de sonolência (16 sessões, ~3 min cada, 29,76 fps), mais as sessões de distração dos mesmos motoristas: "direção segura", classe alerta | 126 vídeos (19 de microssono, 107 de bocejo), 25 fps, 720p. A ficha oficial diz 130 |
| Anotação | quadro a quadro: estado do olho, piscada, bocejo; trechos "Safe Driving" nas sessões de distração | só a categoria do vídeo (pasta) |
| Acesso | formulário no site (maiores de 18, uso acadêmico), sem assinatura | Kaggle `nikospetrellis/nitymed`, conta gratuita |
| Licença | CC BY-NC-ND 4.0: nenhum quadro publicado | CC BY-ND 4.0 (cópia do Kaggle). É obrigatório citar o dataset e um artigo dos autores |

A **UTA-RLDD** (sonolência real, 60 sujeitos) continua sendo a base de teste. **MRL Eye** e **CEW** continuam no Exp. 3.

## 3. O que continua igual

- Título, objetivo e as três perguntas de pesquisa (PP1 atuado → real; PP2 custo no celular; PP3 condições adversas).
- Partição **por sujeito** em todas as avaliações. O Exp. 2 é a exceção deliberada: é o controle que mede o vazamento.
- Os dois pipelines (marcos faciais + características; CNN do olho), as métricas por janela e por vídeo, e a regra de predição por vídeo pela média dos escores das janelas.
- Os experimentos já concluídos em 20/09 (Exp. 1 a 4), que não dependiam do NTHU-DDD.

## 4. O que muda

| Item | Planejado (Entrega 01) | Final | Motivo |
|---|---|---|---|
| Treino atuado | NTHU-DDD | DMD | acesso (seção 1) |
| Noite (PP3) | cenários noturnos do NTHU-DDD | NITYMED, com o DMD em movimento como controle | o NTHU-DDD não veio. Como o NITYMED é gravado com o carro andando, sem o controle a noite ficaria confundida com o movimento |
| Janela | 30 s, passo 5 s | **10 s**, passo 5 s, nas três bases | os trechos de direção segura do DMD duram de 10 a 26 s. Nenhum chega a 30 s |
| Classe alerta no treino | trechos alertas do NTHU-DDD | janelas de "direção segura" da sessão com o carro parado (s2), a mesma condição da sessão de sonolência | a sessão de sonolência do DMD quase não tem trechos alertas |
| Novos experimentos | nenhum | Exp. 5a (validação do nosso extrator contra as características oficiais da UTA-RLDD), Exp. 5b (DMD → UTA-RLDD), Exp. 6 (noite) | respondem à PP1 e à PP3 com dados, não só por contraste com a literatura |
| Condições da PP3 | óculos, óculos escuros, noite | óculos (MRL Eye; no DMD, 3 dos 13 motoristas usam óculos na sessão de sonolência), iluminação (MRL Eye), noite (NITYMED) | **óculos escuros ficam sem cobertura em dado real**. Isso vira limitação declarada e caso de falha na demonstração |
| Sujeitos no treino atuado | 18 (NTHU-DDD) | 13 (DMD) | menos sujeitos: os intervalos de confiança por bootstrap sobre sujeitos refletem isso |
| Aparelho do Exp. 4 | Galaxy S23 Ultra | iPhone 17 Pro, além do Galaxy S23 Ultra; execuções e versão do app em `results/exp4_phone_latency.md` | o Galaxy é de um colega e só pode ser medido na PUC; o iPhone é de um dos autores |

## 5. Impacto nos resultados

Os intervalos entre colchetes são IC 95% por bootstrap sobre sujeitos.

- A PP1 passa a ter resposta direta (Exp. 5b). Com o mesmo extrator e as mesmas características, os modelos treinados no sono encenado do DMD reconhecem o sono real da UTA-RLDD pior que os treinados na própria UTA-RLDD.
  - Na regressão logística, a perda fica de 5,4 a 9,8 pontos de AUC por janela nas quatro formas testadas de normalizar os dados de cada pessoa, sempre com IC acima de zero.
  - No Random Forest, a perda é de 10,9 [6,7; 15,7] na normalização principal. Com a referência no vídeo de pouca vigilância ou sem centragem, ela fica entre 2,1 e 2,3 e não é detectável: na UTA-RLDD as janelas alertas saem do mesmo vídeo usado como referência, e o RF treinado nela aproveita essa semelhança.
  - O achado mais firme é a intensidade. Na UTA-RLDD, o modelo do DMD reconhece 82,0% das janelas alertas, mas só 48,4% das sonolentas: o sono encenado é mais extremo (nas janelas sonolentas, o maior fechamento mediano é de 0,97 s no DMD e de 0,17 s na UTA-RLDD).
  - O volume do treino explica pouco: com só 13 sujeitos da UTA-RLDD por fold, como no DMD, a queda continua (9,3 pontos no RF e 7,0 na RL).
  - Fontes: `results/exp5b_transferencia.md` e `results/exp5b_robustez.md`.
- O extrator não é a explicação. Trocar o extrator oficial (dlib) pelo nosso (MediaPipe) custa +2,0 pp [−4,8; +8,5], diferença compatível com zero. Fonte: `results/exp5a_validacao.md`.
- Noite (PP3): parte do efeito é movimento. À noite, a razão sinal/ruído do EAR cai para 7,2, contra 21,5 na UTA-RLDD. Mas, com os mesmos motoristas do DMD, o carro em movimento já leva essa razão a 8,7, e a diferença noite − movimento (−1,6 [−3,7; +1,1]) não exclui zero. Fonte: `results/exp6_noite.md`.
- Regra × modelos. No limiar 0,5, os dois Random Forest com características brutas disparam em 26,0% (treinado no DMD) e 43,6% (treinado na UTA-RLDD) das janelas de pessoas acordadas, contra 2,6% de uma regra por sinais visíveis. Esse limiar é arbitrário: com o mesmo falso alarme da regra, a RL do DMD centrada acha tantas janelas sonolentas quanto ela (24,7% contra 24,1%). Fontes: `results/exp6_noite.md` e `results/exp5b_robustez.md`, seção 5.
- O NITYMED não sustenta afirmações sobre sono real. Ele entra só como condição de captura noturna: detecção de rosto, qualidade do sinal e taxa de disparo. Nenhuma acurácia é calculada nele.
- O app de demonstração muda. Ele passou a alarmar por sinais visíveis (olhos, bocejo, cabeça), e o aprendizado de máquina ficou só como estimativa. A regra não é mais precisa que os modelos; ela foi escolhida porque cada alarme tem explicação e porque não precisa de treino. Na direção real do DMD, com motoristas acordados, o app ficou 3,3% do tempo em vermelho. Fonte: `results/app_validacao_dmd.md`.

## 6. Decisão

O orientador autorizou a troca de bases (item 1) verbalmente, sem documento formal (registrado em 04/10/2026). Os itens 2 e 3 decorrem da troca e não foram discutidos à parte; vale mencioná-los na próxima conversa com ele.

O que foi apresentado:
1. substituir o NTHU-DDD por DMD (treino atuado) e NITYMED (noite) no artigo, no pôster e na Entrega 02;
2. adotar a janela de 10 s no lugar da de 30 s do planejamento;
3. registrar o iPhone 17 Pro como aparelho medido, mantendo o Galaxy S23 Ultra como medição prevista.

Caso o NTHU-DDD chegue antes de 09/11, ele entrará só como trabalho futuro ou análise complementar, sem alterar os resultados principais.

Detalhes: `docs/ENTREGA02_relatorio.md`, seção 9. Relatórios de cada experimento: `results/*.md`.
