# Guia da equipe: do celular à sessão de pôsteres

Passo a passo para os quatro integrantes. Versão de 26/10/2026.

**Datas:**
- Entrega 02 (código, metodologia implementada e resultados): 31/10 (sábado).
- Artigo IEEE (de 6 a 8 páginas) e pôster: 09/11 (segunda).
- Revisão por pares de 2 artigos de outros grupos: 12/11 (quinta).
- Sessão de pôsteres: 16/11 (segunda).

**Onde está cada coisa:**
- Estado e resultados: `docs/ENTREGA02_relatorio.md` (o PDF para o Canvas é `docs/ENTREGA02_relatorio.pdf`).
- Troca de bases (aprovada pelo orientador em conversa; registro em 04/10/2026): `docs/TROCA_DATASETS.md`.
- Números do artigo e do pôster: `results/numeros_resumo.md`. Todos são gerados, nunca digitados.
- App: `app/pwa/README.md` (lógica, validação, limites e o roteiro de 2 min da seção 6).

**Regras:**
- Execute os comandos na raiz do repositório (a pasta `deteccao-sonolencia`). O caminho tem espaços e parênteses, então use aspas.
- Há dois ambientes Python (`README.md`, seção 5):
  - `.venv` para quase tudo;
  - `.venv-mp` para o que usa MediaPipe no computador: a demo de webcam e a extração de sinais.
- Chame o Python do ambiente certo pelo caminho completo. Assim não há o que "ativar".
- Som: teste automático sempre mudo (Chrome com `--mute-audio`). O alarme do app fica em 20% e só sobe para 100% na hora da demonstração.

```bash
cd "<pasta onde está o repositório>/deteccao-sonolencia"   # ajuste ao seu computador
.venv/bin/python --version        # 3.11.x; instalação: README.md, seção 5
```

## 1. Medir o app no celular

Situação: até agora, só o app 1.0 foi medido, no iPhone 17 Pro (2 execuções de 27/09). Falta medir o app 2.1 no iPhone
e no Galaxy S23 Ultra até 09/11, na PUC ou com o colega que tem o Galaxy. Esta seção serve para essa medição e para
repeti-la se o app mudar.

**Objetivo:** 3 benchmarks de 60 s da PWA 2.1 em cada aparelho, gravados sozinhos em `results/phone/`, e depois
um comando que atualiza a tabela e a figura do Exp. 4, os números, o artigo e o pôster.

**Leve:** o Mac com o repositório (`.venv` pronto), o carregador, o iPhone do André (para o Acesso Pessoal) e um suporte para o celular.

1. **Dependências do app (uma vez, em casa, com internet).**
   ```bash
   bash tools/baixar_dependencias_app.sh             # 13 arquivos, cerca de 36 MB, em app/pwa/vendor/ (fora do Git)
   bash tools/baixar_dependencias_app.sh --conferir  # na véspera: confere sem rede; deve dizer "vendor/ completa"
   ```
   Com `vendor/` completa, o celular não precisa de internet: basta estar na mesma rede do Mac.
2. **Rede.** A rede da PUC isola os aparelhos, e um não enxerga o outro. Por isso:
   - no iPhone, ligue "Ajustes → Acesso Pessoal → Permitir Outras Conexões" (o nome exato varia com a versão do iOS);
   - conecte o Mac e o outro celular nesse Wi-Fi;
   - desligue VPN nos aparelhos.
3. **Servidor no Mac:**
   ```bash
   python3 app/pwa/serve_https.py          # imprime https://<IP-do-Mac>:8443/ ; deixe a janela aberta
   ```
   Ao subir, ele confere `app/pwa/vendor/`; se faltar algo, avisa e oferece baixar. O certificado autoassinado fica
   em `app/.pwa-cert/`. Ele contém a chave privada: fica fora do Git e não deve ser compartilhado.
4. **No celular** (Chrome no Samsung; Safari ou Chrome no iPhone):
   - abra `https://<IP-do-Mac>:8443/`;
   - em "Sua conexão não é particular", toque em "Avançado → Ir para <IP> (não seguro)";
   - permita a câmera;
   - em "Detalhes técnicos > Modelos > Bibliotecas", confira "local (./vendor/)".

   Se a câmera não abrir no Samsung: `chrome://flags` → *Insecure origins treated as secure* → `https://<IP-do-Mac>:8443` → *Relaunch*.
5. **Ambiente:**
   - rosto bem iluminado, de frente, sem contraluz;
   - aparelho no suporte, a uns 40 cm;
   - bateria acima de 50%;
   - sem óculos;
   - feche os outros apps.
6. **Iniciar câmera → calibração.** São 10 s olhando para a frente, com os olhos abertos e a boca fechada. Piscar normalmente é permitido. Se aparecer um aviso, toque em "Recalibrar".
7. **"Detalhes técnicos" → Benchmark 60 s.** Fique parado, olhando para a tela e piscando normalmente. Não boceje e não feche os olhos: é uma medida de custo.

   Ao terminar, a tela deve dizer "enviado para o computador: results/phone/benchmark_pwa_<data>.json, ..._frames.csv". Se o envio falhar, use os botões de download e copie os dois arquivos para `results/phone/` (cabo ou Drive).
8. **Repita até ter 3 execuções.** Espere cerca de 1 min entre elas, com a tela ligada. Anote em `results/phone/NOTAS.md`:
   - data e hora;
   - versão do sistema e do navegador (no Chrome do Android, `chrome://version`);
   - o delegate mostrado (GPU ou CPU);
   - iluminação;
   - se o aparelho esquentou.

   Não edite os números dos arquivos.
9. **Nome do aparelho.** No Android, o app lê o modelo pelos Client Hints do Chrome, e `SM-S918*` vira "Samsung Galaxy S23 Ultra" sozinho. No iPhone, o nome vem de `results/phone/aparelhos.json`. Se aparecer "Android (modelo não informado)" ou um iOS novo, acrescente em `results/phone/aparelhos.json` uma linha `{"contem": "<trecho do nome do arquivo ou do user-agent>", "aparelho": "<nome>"}`.
10. **Um comando atualiza tudo** (cerca de 1 min; nada sai do computador, nenhum commit):
    ```bash
    bash tools/atualizar_celular.sh            # importa -> figura do Exp. 4 -> números conferidos -> artigo -> pôster
    #   --sem-artigo / --sem-poster para pular um dos dois
    ```
    O comando também refaz o pôster com título e autores nos metadados (chama `poster/build_poster.py --pdf`). O pôster
    precisa do template do professor em `material_disciplina/`, ao lado do repositório, ou no caminho de `POSTER_TEMPLATE`;
    sem ele, use `--sem-poster`.
    Confira:
    - `results/exp4_phone_latency.md`: uma linha por execução, com a coluna "Versão do app";
    - a figura `results/figures/fig_exp4_latencia.png`: cada aparelho com a sua barra;
    - o log impresso: `artigo.pdf` de 6 a 8 páginas e 0 macros indefinidas; pôster com 1 página.

    Com o Samsung medido, `\ifSamsungMedido` passa a verdadeiro e as macros `\CelularSamsung...` passam a existir (`paper/README.md`).

Nunca apresente números do iPhone como se fossem do Samsung, nem números do Mac ou do Chrome headless (`results/pwa_headless_*`) como se fossem do celular.

**App Android nativo.** Ele mantém a lógica 1.0 e nunca foi compilado (`app/android/README.md`). O artigo reporta só a
PWA e cita o app nativo como limitação. Só tente se sobrar tempo e houver Android Studio (JDK 17, SDK 36):
- `bash app/android/tools/fetch_assets.sh`;
- abra `app/android/`, rode `./gradlew :app:testDebugUnitTest` e depois Run → Benchmark 60 s → Compartilhar → `results/phone/`.

## 2. Demonstração do app (sem internet)

### 2.1 Véspera

- `bash tools/baixar_dependencias_app.sh --conferir` deve dizer "vendor/ completa". Se faltar algo, rode sem `--conferir`, em casa, com internet.
- Teste o caminho todo uma vez: `python3 app/pwa/serve_https.py`, celular na mesma rede, "Iniciar câmera", "local (./vendor/)" em "Detalhes técnicos > Modelos > Bibliotecas".
- Ensaie o roteiro da seção 2.2 duas vezes antes de 16/11, com o volume do alarme em 20%.
- Grave um vídeo da tela do app fazendo o roteiro inteiro. É o plano B se a câmera ou a rede falharem.
- Leve: o Mac com o repositório, o carregador, o suporte do celular, óculos escuros e uma forma de reduzir a luz (para o caso de falha deliberado).

### 2.2 Roteiro (cerca de 2 min; detalhes em `app/pwa/README.md`, seção 6)

**Antes de chamar alguém (fora do cronômetro):**
1. Celular no suporte, na altura do painel, luz de frente, sem contraluz.
2. Mac e celular na mesma rede (na PUC, o Acesso Pessoal do iPhone). A rede não precisa ter internet.
3. `python3 app/pwa/serve_https.py` no Mac; no celular, abra o app e toque em "Iniciar câmera" (esse toque também libera o som no iPhone).
4. Confira "local (./vendor/)" em "Detalhes técnicos > Modelos > Bibliotecas".
5. Toque em "Recalibrar" (10 s olhando para a frente): nível 0 e registro limpo.

**Volume: só na hora.** O "Volume do alarme" começa em 20%, de propósito. Suba para 100%, ponha o volume de mídia do
celular no máximo e toque em "Testar som" só quando a pessoa já estiver na frente do pôster, logo antes do passo 4 abaixo.
Terminou a demonstração, volte o controle para 20%.

**Na frente da pessoa (a ordem importa: os sinais acumulam no nível):**

| Tempo | Faça | O app mostra |
|---|---|---|
| 0:00 a 0:15 | Pisque normalmente, olhando para a frente | Verde, nível 0. Piscadas normais não sobem o nível |
| 0:15 a 0:45 | Boceje duas vezes, boca bem aberta por uns 2 s, de olhos abertos, com uns 15 s entre os bocejos | "1 bocejo", ainda verde; no segundo, "2 bocejos" e amarelo com um "ding" |
| 0:45 a 1:00 | Baixe a cabeça uns 30° por 3 ou 4 s, mantendo os olhos abertos, e volte | "cabeça baixa com os olhos abertos: distração, 0 ponto". Sem alarme |
| 1:00 a 1:05 | Feche os olhos por 2 s | Com 1 s: vermelho, sirene e tela piscando (microssono) |
| 1:05 a 1:15 | Reabra os olhos | A sirene para e o app volta ao amarelo em uns 2 s; fica pelo menos 1 min em atenção |
| 1:15 a 2:00 | Mostre "Por quê?" e o registro de eventos | Cada ponto do nível tem origem: bocejos, microssono, horário de cada evento |

Cuidados:
- Olhos abertos ao bocejar e ao baixar a cabeça. Olho fechado por 1 s ou mais é microssono para o app, e isso antecipa o vermelho.
- Para recomeçar do zero, toque em "Recalibrar".
- Sem som: toque em "Testar som" de novo e confira o volume do alarme e o de mídia. Rosto não detectado: afaste-se da janela ou da luz de fundo.

**Demo de webcam (opcional, no notebook):** `.venv-mp/bin/python src/realtime/demo_webcam.py` (`q` sai; `--camera 1` se houver
duas câmeras). Ela calibra o EAR nos primeiros segundos, com os olhos abertos, e o alerta é o bipe do terminal: deixe o volume do Mac moderado.

### 2.3 Caso de falha deliberado: o que mostrar e o que dizer

**Mostrar:**
1. Ponha os óculos escuros, ou apague a luz de frente e fique de costas para a janela.
2. No celular, abra "Detalhes técnicos". O EAR e o eyeBlink ficam instáveis, e a CNN do olho fica incerta.
3. O app pode deixar de ver um fechamento, ou ler um olhar para baixo como olho fechado.
4. Tire os óculos e mostre que o app se recupera.

**Dizer:**
1. "O app mede sinais visíveis. Quando a câmera não vê o olho, ele não tem o que medir."
2. "É o que medimos no Exp. 3: com óculos, a transferência do estado do olho cai para AUC 65,1, contra 93,4 sem óculos. No Exp. 6, à noite a qualidade do sinal cai, mas o carro em movimento, de dia, já derruba o sinal quase tanto quanto a noite."
3. "A gente ficou com a regra por sinais visíveis, mas não porque ela acerta mais. Com o mesmo falso alarme, a regressão logística treinada no DMD acha tantas janelas sonolentas quanto a regra (24,7% contra 24,1%). A regra ganha por explicar cada alarme e por não precisar de treino. Na direção real do DMD, o app ficou 3,3% do tempo em vermelho, e o falso alarme que sobrou é de um motorista que olha para baixo."

Se perguntarem sobre sono real, a resposta honesta é do Exp. 5b: o modelo treinado em sono encenado reconhece 82,0% das janelas alertas reais, mas só 48,4% das sonolentas. O app foi testado à parte em 24 pessoas da UTA-RLDD fora do ajuste, sentadas e sem dirigir: alarme em 2 dos 24 vídeos acordados e vermelho em 15 dos 24 sonolentos. Os outros sonolentos não mostraram sinais visíveis, e o app não os detecta.

Se perguntarem sobre privacidade: o vídeo não sai do celular. Com internet, a biblioteca do MediaPipe manda telemetria de uso para um servidor do Google depois de cerca de 60 s, sem imagem; sem internet, nada sai do aparelho.

## 3. Pitch de 60 a 90 s para quem passar pelo pôster

Para falar, não para ler. Cada um treina com as próprias palavras; o importante são os números.

> Oi! A gente trabalhou com detecção de sonolência no celular.
>
> Muitos artigos da área passam de 90% de acerto, mas testando em vídeos em que as pessoas fingem estar com sono. A nossa pergunta foi: o que o modelo aprende com sono fingido funciona com sono de verdade?
>
> Para responder, a gente treinou numa base de sono encenado, o DMD, e testou na UTA-RLDD, em que as pessoas gravaram vídeos com sono real. O extrator de sinais é o mesmo nas duas, e ninguém do teste aparece no treino.
>
> Resultado: piora. O modelo treinado com sono fingido reconhece só 48,4% dos trechos de sono real. Faz sentido: no sono fingido, o fechamento de olho mais longo de cada trecho dura quase um segundo (0,97 s, em mediana); no real, 0,17 s.
>
> A gente também mediu um erro de protocolo fácil de cometer: se trechos do mesmo vídeo caem no treino e no teste, a acurácia pula de 54,2% para 99,5%.
>
> E fizemos um app que roda no próprio celular, sem mandar vídeo para lugar nenhum, e funciona até sem internet. Ele usa regras simples de olhos, bocejo e cabeça e sempre mostra por que alarmou. A gente testou o app em 24 pessoas com sono de verdade, que não entraram em nenhum ajuste: ele alarmou em 15 dos 24 vídeos com sono e em 2 dos 24 sem sono. Eram pessoas sentadas, não dirigindo. Quer ver?

Se a pessoa quiser mais detalhe:
- "Então o modelo não serve?" Ele fica acima do acaso (AUC 69,2 na UTA-RLDD), mas deixa passar metade do sono real. Na regressão logística, a perda fica de 5,4 a 9,8 pontos de AUC em todas as formas de normalizar. No Random Forest, a perda depende de como normalizamos os dados de cada pessoa.
- "Por que não usar o modelo no app?" No mesmo nível de falso alarme, a regra e a regressão logística acham quase as mesmas janelas sonolentas (24,1% e 24,7%). A regra ganha por ser explicável e não precisar de treino.
- "E esses 90% da literatura, são vazamento?" Não dá para dizer isso. Os 90% do NTHU-DDD foram medidos com outras pessoas no teste. O que a gente mostrou é que, se o mesmo vídeo cai nos dois lados, o número infla muito; com outra pessoa no teste, isso não acontece.
- "E à noite?" O sinal do olho piora (SNR 7,2 contra 21,5 na UTA-RLDD), mas o carro andando, de dia, já piora quase igual (8,7).
- "E com óculos?" O estado do olho transfere pior entre bases: AUC 65,1 com óculos, contra 93,4 sem. Óculos escuros a gente não conseguiu testar com dado real.
- "E com sono de verdade, funciona?" Testamos em 24 pessoas da UTA-RLDD que não entraram no ajuste, sentadas e sem dirigir. Houve alarme em 2 dos 24 vídeos acordados, e 15 dos 24 sonolentos chegaram ao vermelho; a AUC entre sonolento e acordado foi 0,812. Mas 7 sonolentos ficaram verdes: o app reage a olho fechado e bocejo, e quem diz que está com sono sem mostrar esses sinais não é pego. Nas mesmas pessoas, o modelo treinado no DMD deu AUC 0,811 e a regra do rótulo, 0,853; o modelo treinado na própria UTA-RLDD só passa disso (0,953) com o atalho da centragem pelo vídeo acordado, e sem ele fica em 0,806. O app, sem treino nenhum, ficou no nível deles.
- "Funciona no carro de verdade?" Não testamos em campo. Em vídeos de direção real com motoristas acordados, o app ficou 3,3% do tempo em vermelho.
- "Quanto tempo leva no celular?" Com o app 1.0, no iPhone 17 Pro, 17 ms por quadro (mediana de 2 execuções, de 11 a 23 ms), quase todo no detector de rosto. O app 2.1 ainda vai ser medido no iPhone e no Galaxy S23 Ultra.

### 3.1 "Como vocês definiram as regras e os pesos?"

O professor já perguntou ("cálculo ou achismo?"). Resposta curta, para falar com as próprias palavras. A tabela
completa, com a origem de cada número e como ele foi testado, está em `app/pwa/README.md`, seção 13.

> Nem tudo veio do mesmo lugar, e a gente separa em três tipos.
>
> Uma parte vem da literatura: o PERCLOS em 60 segundos contando só os fechamentos lentos, de meio segundo ou mais, e o microssono de 1 segundo, que é onde começa a faixa de 1 a 15 segundos usada para microssono. O limite de 15% do PERCLOS é nosso: é mais alto que os 8% a 12% de outros trabalhos, para o app não alarmar à toa.
>
> Outra parte a gente conferiu ou escolheu com a anotação quadro a quadro do DMD. O fator 0,75 do limiar do EAR foi fixado antes e só conferido lá: deu a maior concordância por quadro. A regra do olho fechado saiu de 11 variantes, e o bocejo, de uma grade de valores. Por exemplo, com 1 segundo de boca aberta o app achava mais bocejos, mas na direção real, com motoristas acordados, o tempo em vermelho subia de 3,3% para 5,0%. Por isso ficou 1,5 segundo.
>
> E os pontos do nível são regra de projeto, com critério escrito: os pontos de um sinal só não chegam ao vermelho, dois sinais diferentes juntos chegam, microssono e cabeça caindo de olhos fechados levam direto ao vermelho, e evento antigo perde peso em 2 a 5 minutos.
>
> A gente não otimizou esses pesos porque não existe base de sono real com esses sinais marcados ao longo do tempo. Ajustar no DMD, que é encenado, seria repetir o problema que o próprio trabalho mediu. Para ver se o resultado depende muito dos pesos, variamos todos em até 30% e os limiares 40 e 70 em até 10 pontos, em 300 combinações sorteadas. O falso alarme na direção real ficou entre 1,7% e 4,9% do tempo, perto dos 3,3% da configuração final, e 97% das combinações continuaram boas. O que mais mexe é o limiar de 70 e o teto dos fechamentos, e é isso que a gente calibraria primeiro com dados de sono real.

Se insistirem ("então é achismo?"): não é otimização, mas também não é chute. Cada número tem origem escrita, os
que dava para comparar foram comparados no DMD, e o "Por quê?" do app mostra de onde vem cada ponto. Num teste
independente, em 24 pessoas da UTA-RLDD que não entraram no ajuste, o app alarmou 15 dos 24 vídeos sonolentos e 2 dos
24 acordados. O que ainda falta é testar dirigindo: essas pessoas estavam sentadas.

## 4. Overleaf

1. **O que subir.** Crie o projeto e envie a pasta `paper/` inteira: `artigo.tex`, `refs.bib`, `numeros.tex` (gerado; sem ele as macros de número ficam indefinidas) e `figures/` (PDFs). Os `.md` podem ficar de fora.
2. **Configuração.** Compilador pdfLaTeX; documento principal `artigo.tex`. O Overleaf roda o BibTeX sozinho.
3. **Números.** Nunca digite número no Overleaf. Se um resultado mudar, por exemplo depois de uma medição no celular:
   - rode `bash tools/atualizar_celular.sh` no Mac;
   - suba de novo `numeros.tex`.
4. **Texto.** Edições de texto feitas no Overleaf voltam para `paper/artigo.tex` no repositório antes de cada commit (baixe o `.tex`). Uma pessoa edita por vez, ou use o histórico do Overleaf.
5. **Conferência local** (a mesma do comando único):
   ```bash
   cd paper && ~/.local/bin/tectonic --keep-logs artigo.tex && cd ..
   .venv/bin/python -c "import pymupdf; print(len(pymupdf.open('paper/artigo.pdf')), 'páginas')"   # de 6 a 8 (hoje: 8)
   grep -c "Overfull" paper/artigo.log                                                            # 0
   grep -cE "Undefined control sequence|(Citation|Reference) .* undefined" paper/artigo.log        # 0 (avisos de fonte "TU/ptm" não contam)
   ```
6. **Citações obrigatórias das bases:**
   - UTA-RLDD: `ghoddoosian2019`;
   - DMD: `ortega2020`;
   - NITYMED: `petrellis2022` e `petrellis2021dsd` ou `petrellis2021vlsisoc`;
   - MRL Eye: `fusek2018`;
   - CEW: `song2014`.

   Não cite chaves com `UNVERIFIED` no `refs.bib`.

## 5. Revisão por pares (entrega em 12/11)

Cada grupo revisa pelo menos dois artigos de outros grupos, pelo sistema de gestão da conferência indicado pelo
professor (slides da disciplina). O cronograma abaixo supõe que os artigos chegam logo depois da entrega de 09/11; se chegarem em outro dia, desloque as etapas.

### 5.1 Duplas (sugestão; a equipe pode trocar)

| Dupla | Integrantes | Artigo |
|---|---|---|
| 1 | André e Bruno Sandoval | o primeiro artigo designado |
| 2 | Bruno Rodolfo e Paulo | o segundo artigo designado |

Cada dupla escreve um parecer. A outra dupla lê o rascunho antes do envio, e os quatro dão o ok final.

### 5.2 Cronograma

| Dia | O que fazer |
|---|---|
| 09/11 (segunda) | Entregar artigo e pôster. Quando os artigos chegarem pelo sistema, cada dupla baixa o seu e faz uma leitura inteira, sem anotar |
| 10/11 (terça) | Segunda leitura com a lista da seção 5.4. Preencher o modelo da seção 5.3 em rascunho |
| 11/11 (quarta) | Trocar os rascunhos entre as duplas. Cada uma marca o que ficou vago, sem exemplo ou duro demais. Ajustar |
| 12/11 (quinta) | Leitura final dos quatro e envio no sistema, dentro do prazo do professor. Guardar o comprovante |

Os artigos dos outros grupos e os nossos pareceres não vão para o GitHub nem para a pasta do projeto: ficam no
sistema da disciplina e numa pasta privada da equipe.

### 5.3 Modelo de parecer

Copie o bloco abaixo, preencha e cole no formulário do sistema. Use a escala e as opções que o formulário pedir.

```
PARECER: <título do artigo> (<identificação no sistema>)
Revisores: <dupla>, Equipe 14

Resumo do artigo (3 a 5 linhas, com as nossas palavras)
<problema, método, bases ou dados, resultado principal>

Pontos fortes
1. <o quê> (<seção, tabela ou figura>)
2. <o quê> (<onde>)

Pontos fracos (cada um com uma sugestão de correção)
1. <o quê> (<onde>). Sugestão: <como corrigir>
2. <o quê> (<onde>). Sugestão: <como corrigir>

Novidade e relevância: <nota>
<o que é novo em relação aos trabalhos que o próprio artigo cita; por que o problema importa>

Clareza, referências e escrita: <nota>
<estrutura, figuras legíveis, referências completas e atuais, formato IEEE, de 6 a 8 páginas>

Qualidade técnica e avaliação experimental: <nota>
<partição dos dados (por sujeito?), métricas, comparação com a literatura, desvio ou IC,
conclusões apoiadas nos números>

Facilidade de implementação: <nota>
<dá para reproduzir? código disponível, bases públicas, hiperparâmetros, recursos computacionais>

Recomendação: <opção do formulário>
Confiança do revisor: <opção do formulário> (<por quê: conhecemos bem o tema ou só parte dele>)

Comentários aos autores
- <perguntas e erros pontuais, com página ou seção>
```

### 5.4 O que conferir nos artigos dos outros

É o mesmo que cobram da gente:
- partição por sujeito: trechos do mesmo vídeo, ou da mesma pessoa com o mesmo rótulo, nos dois lados inflam o resultado;
- unidade da métrica (quadro, janela, vídeo, pessoa) e se ela é a mesma nas comparações;
- origem dos números das tabelas e se o resumo bate com os resultados;
- intervalos de confiança ou desvio entre folds;
- licenças de imagem das bases (rostos publicados sem permissão);
- referências que existem e estão completas;
- figuras legíveis impressas em preto e branco.

Tom: escreva como a gente gostaria de receber. Cada crítica com o lugar exato e uma sugestão. Nada de comentário genérico,
como "melhorar a escrita". Elogio também precisa dizer onde.

### 5.5 O que esperar sobre o nosso artigo

Quase tudo já tem resposta no texto:
- Rótulos de natureza diferente (evento encenado no DMD, KSS autorrelatado na UTA-RLDD): limitação (1) do artigo.
- A queda do RF depende da normalização: Tabela II. A RL perde de 5,4 a 9,8 pontos por janela em todas as formas de normalizar, e o achado mais firme é a intensidade (48,4% das janelas sonolentas reais).
- Volume do treino (o DMD tem 510 janelas de 13 motoristas): com B treinado em 13 sujeitos, a queda continua (9,3 no RF e 7,0 na RL).
- Vazamento e literatura: a inflação vem de janelas do mesmo vídeo nos dois lados; os 90% do NTHU-DDD foram medidos com outras pessoas no teste.
- Regra × modelos: no mesmo falso alarme, a RL do DMD acha tantas janelas sonolentas quanto a regra.
- NITYMED sem classe alerta: ele só entra como condição noite, sem acurácia.
- Celular: só o app 1.0 foi medido, no iPhone, em 2 execuções; o app 2.1 e o Galaxy ainda vão ser medidos; o app Android nativo não foi avaliado.
- Pesos do nível do app escolhidos, não ajustados, e validação feita nos mesmos dados em que as regras foram escolhidas (números de desenvolvimento). Resposta pronta na seção 3.1.
- Validação do app fora do ajuste: o teste independente na UTA-RLDD (24 pessoas, Tabela V) responde, mas com pessoas sentadas e o sono declarado como rótulo.

Se recebermos pareceres sobre o nosso artigo, as correções entram até 14/11. Qualquer número novo passa por `make_paper_numbers.py`, nunca por digitação.

## 6. Subir para o GitHub (push só com o "ok" do André)

**Situação (26/10, noite):**
- o repositório `github.com/BrunoRodolfoo/deteccao-sonolencia` é público;
- todo o trabalho até a noite de 26/10 foi enviado nesse dia, em commits separados por assunto. Esses commits são do André, que fez o trabalho no computador com as bases;
- daqui em diante, cada um faz o commit do que fizer (por exemplo, o Paulo, a medição do Galaxy);
- falta só a medição do app 2.1 no iPhone e no Galaxy S23 Ultra, que entra num commit próprio, de quem medir.

O `.gitignore` já exclui:
- `data/` e os vídeos;
- `.venv`, `.venv-mp`;
- `app/.pwa-cert/`, com a chave privada;
- `app/pwa/vendor/` (dependências do app, baixadas de novo pelo script);
- os modelos regeneráveis grandes (`models/exp1/`, `models/exp6/`);
- a figura interna com rostos do DMD (`results/figures/exp5_dmd_verificacao_visual.jpg`), se for recriada.

```bash
cd "<pasta onde está o repositório>/deteccao-sonolencia"
git status --short | head -50                      # nada de data/, .venv*, .pwa-cert, vendor/, *.npz de cache, *.mp4
git status --short -uall | awk '{print $2}' | while read f; do [ -f "$f" ] && echo "$(du -k "$f" | cut -f1) KB $f"; done | sort -rn | head -15
#   maior esperado: results/exp5b_predicoes.csv (~32 MB; fonte dos IC do Exp. 5b); nenhum arquivo > 50 MB
git status --short -uall | grep -iE "\.(jpg|jpeg|mp4|mov|y4m)$"   # deve sair vazio (a figura com rostos é ignorada)
git check-ignore -v results/figures/exp5_dmd_verificacao_visual.jpg app/.pwa-cert app/pwa/vendor   # as três devem aparecer
```

O que falta, depois da medição do app 2.1 nos celulares e do `bash tools/atualizar_celular.sh`:

```bash
git add results/phone results/exp4_phone_latency.* results/figures/fig_exp4_latencia.* paper poster results/numeros_resumo.md
git commit -m "Exp. 4: app 2.1 no iPhone 17 Pro e no Galaxy S23 Ultra"
git tag -a entrega02 -m "Entrega 02 - 31/10/2026"
git push origin main && git push origin entrega02
```

Confirme que o repositório continua público (o artigo e o QR code do pôster apontam para ele).

## 7. Checklist final de cada entrega

### 7.1 Entrega 02 (31/10): código, metodologia implementada e resultados

- [x] Aval do orientador sobre a troca de bases: autorização verbal, registrada em 04/10/2026 (`docs/TROCA_DATASETS.md`)
- [ ] A seção "Como verificar em 5 minutos" do relatório roda sem erro (inclui `bash run_all.sh --dry-run`, que lista as 11 etapas)
- [ ] `results/*.md` finais presentes (Exp. 1, 2, 3, 5a, 5b, controles do 5b em `exp5b_robustez.md`, 6, app)
- [ ] `results/exp4_phone_latency.md` com as execuções do app 2.1 nos celulares
- [ ] `.venv/bin/python src/experiments/conferir_numeros.py` sem divergências
- [ ] `docs/ENTREGA02_relatorio.md` revisado pelos quatro; `docs/ENTREGA02_relatorio.pdf` enviado no Canvas
- [x] Envio ao GitHub em 26/10, em commits por assunto, com o repositório público; o link vai no documento enviado ao professor
- [ ] Commit da medição do app 2.1 nos celulares, tag `entrega02` e push (seção 6)

### 7.2 Entrega 03a (09/11): artigo IEEE (de 6 a 8 páginas) e pôster

- [ ] `artigo.tex` com `\input{numeros}` e só macros nos números; de 6 a 8 páginas; 0 overfull; 0 indefinidos
- [ ] `paper/artigo.pdf` recompilado depois da medição no celular (`bash tools/atualizar_celular.sh`)
- [ ] Texto final com a leitura nova da PP1 (RL de 5,4 a 9,8 pontos; RF dependente da normalização; 48,4% das janelas sonolentas reais), do vazamento (mesmo vídeo nos dois lados; os 90% do NTHU-DDD são com outras pessoas no teste) e da regra × modelos
- [ ] Ressalvas escritas:
  - [ ] rótulos de natureza diferente no Exp. 5b
  - [ ] NITYMED sem classe alerta e de autenticidade não documentada
  - [ ] celular: aparelhos, execuções e versão do app conforme `paper/numeros.tex`
  - [ ] Android nativo não avaliado
  - [ ] validação do app nos mesmos dados em que as regras foram escolhidas
- [ ] Nenhum quadro do DMD, UTA-RLDD ou NITYMED; só recortes de olho do MRL Eye
- [ ] Pôster: `build_poster.py` lendo `poster/numeros.json`; mostra os sinais detectados (olhos, bocejo, cabeça, nível de 0 a 100); título e autores iguais aos do artigo; orientador; QR code para o repositório; teste de legibilidade a 25% de zoom; PDF gerado por `bash tools/atualizar_celular.sh`

### 7.3 Revisão por pares (12/11)

- [ ] Dois pareceres enviados pelo sistema (seção 5), com cópia numa pasta privada, fora do GitHub
- [ ] Críticas recebidas incorporadas até 14/11; números novos só via `make_paper_numbers.py`

### 7.4 Sessão de pôsteres (16/11)

- [ ] Pôster impresso ou em tela, conforme a disciplina; PDF de reserva no celular
- [ ] Dois ensaios da demonstração sem internet (seção 2); volume em 100% só na hora; plano B com o vídeo gravado
- [ ] Pitch de 60 a 90 s ensaiado pelos quatro (seção 3)
- [ ] Respostas curtas prontas (seção 3): PP1, vazamento, noite × movimento, regra × modelos, falso alarme do app, limites (mão na boca, óculos escuros, cabeça sem revocação medida); origem das regras e dos pesos do app (seção 3.1)

## 8. Som no iPhone

O alarme precisa tocar no dia. Faça estes testes uma vez, em casa, com o volume do alarme em 20% e o ambiente calmo.

1. **Som com o modo silencioso ligado.** O iPhone 17 Pro não tem chave lateral: o silencioso liga pelo botão de Ação ou pela Central de Controle. Com o silencioso ligado:
   1. toque em "Iniciar câmera", depois em "Testar som";
   2. o ding e a sirene devem tocar, porque o app usa `navigator.audioSession.type = "playback"`;
   3. confira o estado em "Detalhes técnicos → Modelos → Áudio";
   4. repita com o silencioso desligado;
   5. anote o resultado em `results/phone/NOTAS.md`.
2. **Volta do som depois de sair do app:**
   1. com o som funcionando, vá para a tela inicial, abra outro app por uns 30 s e volte;
   2. sem tocar na tela, feche os olhos por 2 s: a sirene deve tocar;
   3. se não tocar, um toque na tela deve religar o som (limite conhecido, `app/pwa/README.md`, seção 10).

   Anote se foi preciso tocar. Teste também com a tela bloqueada e desbloqueada.
3. **Falso alarme, acordado.** Use o app de 3 a 5 min, olhando o celular como no painel: deve ficar verde. Se ficar amarelo ou vermelho sem motivo, anote o que aparece em "Por quê?" e no registro de eventos, por exemplo "olhar para baixo".

Não mude limiar nenhum na véspera: qualquer ajuste precisa ser medido de novo no DMD s1 e s5, como na seção 9 do `app/pwa/README.md`.
