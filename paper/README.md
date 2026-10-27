# Artigo IEEE (Entrega 03a): `paper/`

Artigo de resultados do TCC **"Detecção de Sonolência em Dispositivos Móveis: Avaliação da
Generalização entre Sonolência Atuada e Real"** (PUCPR, Experiência Criativa: Projeto Transformador II,
Turma 8ºA, Equipe 14). Formato IEEE *conference* (`IEEEtran`, duas colunas, de 6 a 8 páginas, português).

**Estado (27/10/2026): versão final revisada.** Depois de uma banca simulada, o texto foi reescrito com a
leitura nova da PP1 (controles do Exp. 5b em `results/exp5b_robustez.md`), do vazamento e da comparação entre a
regra e os modelos. São 8 páginas, 31 referências (nenhuma marcada `UNVERIFIED`), as Tabelas I a V e 3 figuras.
O texto não usa travessões. Nenhum número de resultado é digitado no `.tex`: todos vêm de `numeros.tex` (seção
"Como os números são gerados").

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `artigo.tex` | fonte do artigo (UTF-8, `babel` brazil, `IEEEtran` conference); faz `\input{numeros}` |
| `artigo.pdf` | PDF compilado; é refeito por `bash tools/atualizar_celular.sh` (ou pelo `tectonic`, abaixo) |
| `numeros.tex` | gerado por `src/experiments/make_paper_numbers.py`; não edite à mão |
| `refs.bib` | referências verificadas; o artigo cita 31 chaves. As duas marcadas `UNVERIFIED` (`xu2025` e `fl3d`) não são citadas |
| `figures/` | figuras do artigo e o diagrama TikZ da Fig. 1 (ver `figures/README.md`) |
| `related_work_notes.md`, `dataset_facts.md`, `fontes_brasil.md` | notas de verificação bibliográfica usadas na redação (não entram na compilação) |
| `REVISAO_SIMULADA.md` | revisão por pares simulada da versão de 28/09: notas por critério, fraquezas com localização e correções |

Figuras usadas pelo artigo (todas em `figures/`):

| Figura | Arquivo | Origem |
|---|---|---|
| Fig. 1, cadeias de processamento (offline e PWA 2.1) | `fig_pipeline.pdf` | TikZ, `figures/fig_pipeline.tex` |
| Fig. 2, Exp. 5b (PP1): A, B e C por janela e por vídeo | `fig_exp5b_transferencia.pdf` | `src/experiments/make_figures.py` (grava em `results/figures/` e copia o PDF para cá) |
| Fig. 3, Exp. 6 (PP3): SNR do EAR e taxa de disparo | `fig_exp6_noite.pdf` | idem |

Tabelas: I, bases; II, controles da queda B − C do Exp. 5b (normalização, IC com os motoristas do DMD
reamostrados, vídeos fora da referência, B com 13 sujeitos); III, Exp. 1 e 2; IV, Exp. 4 (a coluna do Galaxy
S23 Ultra só aparece com `\ifSamsungMedido` verdadeiro); V, validação do protótipo. Nenhuma figura ou tabela
mostra rosto (DMD, UTA-RLDD e NITYMED têm restrições de imagem; ver `data/README.md`).

`figures/fig_exp4_latencia.{pdf,png}` fica na pasta, mas não entra no artigo (a Tabela IV traz os números); as
figuras dos Exp. 1, 2, 3 e 5a ficam só em `results/figures/`. Os diagramas de reserva (desenho experimental,
sinal de uma piscada, protocolos de partição) e as cópias PNG sem uso foram para a Lixeira em 26/10/2026
(`docs/ESTRUTURA.md`, seção 3).

## Estrutura do artigo e fonte de cada parte

| Seção | Conteúdo | Arquivos de origem |
|---|---|---|
| Resumo, I | números principais de PP1, PP2, PP3 e do vazamento; ABRAMET e PRF, com a ressalva de que a categoria não isola o sono; lacuna de método (Perkins et al.) | `results/*.md` (via macros); `fontes_brasil.md`; `related_work_notes.md` |
| II | trabalhos relacionados: sinais visuais, protocolos e generalização (inclui `perkins2023`, `xie2026` e `pradeep2026`), celulares | `related_work_notes.md`, `dataset_facts.md` |
| III | bases (Tabela I), troca do NTHU-DDD, DMD (alerta = direção segura da s2), NITYMED (autenticidade não documentada), KSS da UTA-RLDD | `results/exp5_caracteristicas.md`, `results/exp6_nitymed_ficha.md` |
| IV | extração (MediaPipe 0.10.21), detector validado contra a anotação do DMD, janelas de 10 s, rótulos, centragem com referência separada, modelos, protótipo PWA 2.1 (Fig. 1) | `results/exp5_caracteristicas.md`, `app/pwa/README.md` |
| V | protocolo, métricas, IC por *bootstrap* de sujeitos, reprodutibilidade | `results/exp5b_transferencia.md`, `results/exp6_noite.md` |
| VI-A | Exp. 5b (PP1, Fig. 2) e controles da queda (Tabela II): referência no vídeo KSS 5, sem centragem, vídeos fora da referência, B com 13 sujeitos, alerta da s1 | `results/exp5b_transferencia.md`, `results/exp5b_robustez.md` |
| VI-B | Exp. 1, 2 e 5a (Tabela III) e o vazamento no cenário do Exp. 5b | `results/exp1_intra.md`, `results/exp2_leakage.md`, `results/exp5a_validacao.md`, `results/exp5b_robustez.md` |
| VI-C | Exp. 4 (PP2, Tabela IV): quantização e tempo por quadro no celular | `results/exp4_*.csv`, `results/exp4_phone_latency.md` |
| VI-D | Exp. 3 por condição; Exp. 6 (noite × movimento, disparo; Fig. 3); óculos no DMD | `results/exp3_eye_state.md`, `results/exp6_noite.md`, `results/exp5b_transferencia.md` (seção 11) |
| VI-E | validação do app 2.1 (Tabela V): offline no DMD, no navegador, teste independente em sono real na UTA-RLDD (com a comparação com os modelos nas mesmas pessoas) e sensibilidade dos pesos | `results/app_validacao_dmd.md`, `results/app_validacao.md`, `results/app_validacao_uta.md`, `results/app_comparacao_uta.md`, `results/app_sensibilidade_pesos.md` |
| VII | discussão de PP1 a PP3 (inclui a regra × modelos no mesmo falso alarme), limitações, ética e LGPD | `results/exp5b_robustez.md` (seção 5) e os demais |
| VIII | conclusão e trabalhos futuros | n/a |

## Como compilar (local)

```bash
export PATH="$HOME/.local/bin:$PATH"
cd paper                              # a partir da raiz do repositório
tectonic artigo.tex                   # roda o BibTeX e reexecuta o LaTeX sozinho -> artigo.pdf
tectonic --keep-logs artigo.tex       # mantém artigo.log e artigo.blg para inspeção
```

Conferências depois de compilar (as mesmas que `tools/atualizar_celular.sh` faz):

```bash
../.venv/bin/python -c "import pymupdf; print(len(pymupdf.open('artigo.pdf')))"   # de 6 a 8 (hoje: 8)
grep -c "Overfull" artigo.log                      # 0
grep -c "Undefined control sequence" artigo.log    # 0
grep -E "(Reference|Citation) .* undefined" artigo.log   # nada
```

Diagrama TikZ: `cd figures && tectonic fig_pipeline.tex` (detalhes em `figures/README.md`).

## Overleaf

1. Envie a pasta `paper/` com `artigo.tex`, **`numeros.tex`**, `refs.bib` e `figures/`
   (bastam os PDFs `fig_pipeline.pdf`, `fig_exp5b_transferencia.pdf` e `fig_exp6_noite.pdf`).
   Sem `numeros.tex` o artigo não compila (ele traz todos os números).
2. Compilador: **pdfLaTeX** (ou XeLaTeX); documento principal `artigo.tex`. O Overleaf roda o BibTeX
   automaticamente.
3. Não envie `artigo.pdf`, `.aux`, `.bbl`, `.log`. Se um número mudar, **não o edite no Overleaf**:
   regenere `numeros.tex` localmente (abaixo) e reenvie o arquivo.

## Como os números são gerados

- `numeros.tex` é gerado por `src/experiments/make_paper_numbers.py` a partir dos CSV, JSON e MD finais
  de `results/` (e `models/*.json`). O mesmo conteúdo vai para `poster/numeros.json` (pôster), e a
  tabela macro → valor → arquivo → coluna e filtro vai para `results/numeros_resumo.md`.
- `src/experiments/conferir_numeros.py` recalcula cada número com uma implementação independente,
  confere `numeros.tex` × `numeros.json`, procura o valor no `.md` de resultados e compila um documento
  com todos os macros. Estado em 27/10/2026: 658 números (1.299 macros), 0 divergências.
- Convenções: percentuais **sem** o símbolo `%` (escreva `\ExpCincoBCSensJanela\%`); diferenças em pontos
  percentuais; `\NomeIC` = "[inf; sup]", `\NomeComIC` = "valor [inf; sup]", `\NomeMDP` = "média ± dp";
  negativos com o sinal de menos tipográfico. Depois de um macro seguido de palavra, use `{}`
  (`\ExpSeisSNRNoite{} no NITYMED`), senão o TeX come o espaço.
- Em 28/09 entrou no gerador a função `artigo_extra()`: validação do detector de piscadas contra a anotação do
  DMD (`\SinaisDetector...`), duração dos trechos de direção segura, horas e quadros por segundo das bases,
  estratos de iluminação e reflexo do Exp. 3, sensibilidades e controles do Exp. 5b, o motorista de óculos do
  DMD, o PERCLOS com limiar pelo p90 (Exp. 6), a latência por estágio da PWA no desktop e o EAR sozinho no app.
  A revisão por pares simulada (28/09) acrescentou `revisao_pares()`: disparo do RF do DMD com o limiar
  calibrado no treino e da RL do DMD, queda da RL sem centragem, a queda nas 8 variantes do Exp. 5b e sujeitos
  de treino de B por *fold*.
- Em 26/10 entraram os números de `results/exp5b_robustez.csv` (gerado por `src/experiments/exp5b_robustez.py`):
  referência no vídeo KSS 5 (`\ExpCincoBRefAlt...`), IC em dois níveis (`\ExpCincoBBootDuplo...`), B com 13
  sujeitos (`\ExpCincoBTreze...`), vazamento no cenário do Exp. 5b (`\ExpCincoBVazamento...`) e ponto de operação
  da regra (`\ExpCincoBOpRegra...`), além das faixas derivadas `\ExpCincoBLRQuedaNormMin/Max` e
  `\ExpCincoBQuedaOutrasNormMin/Max`. Também em 26/10 entraram os números dos dois testes novos do app: o teste independente na
  UTA-RLDD (`\AppUTA...`, de `results/app_validacao_uta.csv`) e a sensibilidade dos pesos (`\AppPesos...`, de
  `results/app_sensibilidade_pesos.csv`). Todos passam pela mesma conferência.
- Regerar tudo: `.venv/bin/python src/experiments/make_paper_numbers.py && .venv/bin/python src/experiments/conferir_numeros.py`
  e recompilar.

## Depois de uma medição no celular

Um único comando, na raiz do repositório: `bash tools/atualizar_celular.sh` (ou `--sem-poster`). Ele importa
`results/phone/`, refaz a figura do Exp. 4, os números, este PDF e o pôster. O texto já está preparado: enquanto
o Galaxy S23 Ultra não tiver dados, `\ifSamsungMedido` é falso e as frases e a coluna dele somem do resumo, das
contribuições (Seção I), da Seção VI-C, da Tabela IV, da Discussão (PP2) e das limitações; com dados, a Tabela IV passa a mostrar
`\CelularSamsung...` e as frases mudam sozinhas. Se algum estágio faltar no benchmark, o `\providecommand` do
preâmbulo imprime "n/a" em vez de quebrar a compilação. Depois de cada medição, releia o resumo, a Seção VI-C e a
conclusão: a versão do app, o número de execuções e os tempos vêm de `\CelularIPhone...` e `\CelularSamsung...`.

## Decisões de redação

Desde 28/09/2026:
- PP1 é respondida pelo Exp. 5b (DMD → UTA-RLDD, mesmo extrator nas duas pontas); o vazamento (Exp. 2), a
  reprodução (Exp. 1) e a validação do extrator (Exp. 5a) aparecem de forma compacta.
- O NITYMED entra só na PP3 (condição noite), nunca como sonolência real; o texto diz que a autenticidade
  dos comportamentos não é documentada e registra 126 dos 130 vídeos declarados.
- Troca do NTHU-DDD descrita com a razão verificável (o termo exige assinatura do chefe de departamento
  ou diretor de laboratório), a vantagem (RGB × RGB) e as perdas (13 motoristas atuados, sem óculos
  escuros).
- App: descreve a versão 2.1 (olhos, bocejo, cabeça, nível de 0 a 100, amarelo em 40 e vermelho em 70, sirene,
  LSTM limitado a 10 pontos); diz que os pesos são escolhas de projeto, que a cabeça não tem revocação medida
  e que a validação usa os mesmos dados em que as regras foram escolhidas.
- Referências: notas de processo (datas de conferência, "cópia usada") ficam em comentários `%` no
  `refs.bib`, para não aparecerem impressas; meses em português (`{set.}`, `{out.}`); nomes compostos
  protegidos (`{Ca\~{n}as}`, `{Van Droogenbroeck}`).

Em 26/10/2026, depois da banca simulada:
- PP1: a RL perde de 5,4 a 9,8 pontos de AUC por janela nas quatro formas de normalizar; o RF perde 10,9 só na normalização
  principal e, com outra referência ou sem centragem, de 2,1 a 2,3 (não detectável), porque na UTA-RLDD as janelas
  alertas saem do mesmo vídeo da referência. O achado mais firme é a intensidade: o modelo do DMD reconhece só
  48,4% das janelas sonolentas reais. O volume do treino explica pouco (B com 13 sujeitos).
- Vazamento: a inflação vem de janelas do mesmo vídeo nos dois lados, não de reconhecer a pessoa. O texto diz
  que os 90% do NTHU-DDD foram medidos com outras pessoas no teste e que esse vazamento não explica a distância
  até a UTA-RLDD. No cenário do Exp. 5b, a partição aleatória infla o RF em 12,6 pp e a RL em só 1,2 pp.
- Regra × modelos: a regra não é "mais precisa"; no mesmo falso alarme, a RL do DMD centrada acha tantas
  janelas sonolentas quanto ela. A vantagem da regra é ser explicável e não precisar de treino.
- Três referências novas (`perkins2023`, `xie2026`, `pradeep2026`; notas em `related_work_notes.md`) e, depois, a
  fonte da definição de microssono (`skorucak2020`): 31 no total.
- App (Seção VI-E e Tabela V): teste independente em sono real na UTA-RLDD, sensibilidade dos pesos e, nas mesmas
  24 pessoas, a comparação com os modelos do Exp. 5b e com a regra do rótulo (`results/app_comparacao_uta.md`).
- Celular: o tempo por quadro aparece em milissegundos inteiros, como o iPhone registra (app 1.0: 17 ms de mediana,
  de 11 a 23 ms entre as 2 execuções); o app 2.1 ainda vai ser medido no iPhone e no Galaxy S23 Ultra.

## Pendências para a equipe

1. Se ainda não foi feito: abrir no navegador a URL da PRF (`prf2026`), que responde 401 a clientes
   automatizados, e atualizar a data de acesso das duas fontes brasileiras no `refs.bib`.
2. Conferir que o repositório citado no artigo está público e com a versão da entrega.
3. Depois da medição no celular, rodar `bash tools/atualizar_celular.sh` e conferir as 8 páginas.
4. Leitura final pelo orientador.

## Histórico

- 20/09/2026: primeira versão de 8 páginas (UTA-RLDD, MRL e CEW; NTHU-DDD "substituído por experimentos
  indiretos"), com revisão crítica e revisão por pares simulada (`REVISAO_SIMULADA.md`).
- 27/09/2026: `numeros.tex` e o comando único `tools/atualizar_celular.sh`.
- 28/09/2026: reescrita com os Exp. 5a, 5b e 6, o protótipo 2.1 e o diagrama `fig_pipeline` refeito.
- 28/09/2026 (conferência crítica): números conferidos contra os CSV e ressalvas acrescentadas ao texto. A
  sonolência do DMD é "mais extrema" em parte por construção do rótulo (janelas com evento); o Exp. 5b também
  troca a pergunta (evento encenado × estado relatado); no Exp. 3, o reflexo reduz a AUC, mas não a acurácia
  balanceada; o p90 também infla o PERCLOS de motoristas acordados em movimento; a vantagem do nosso extrator em
  B3 não se sustenta no LOSO; os disparos do Exp. 6 comparam características sem centragem; a latência do celular
  e do desktop é da versão 1.0.
- 28/09/2026 (madrugada): revisão por pares simulada da versão final (`REVISAO_SIMULADA.md`) e correções de
  texto: disparo com limiar calibrado e RL, confusão com o volume do treino na PP1, validação do app declarada
  como de desenvolvimento, acurácia balanceada nas comparações entre tarefas, vazamento descrito como "janelas do
  mesmo vídeo", tempos do celular por faixa entre execuções, Tabela II (mapa) substituída por uma frase.
- 06/10/2026: texto enxugado para 7 páginas e sem travessões (intervalos escritos como "de X a Y"); Fig. 1
  redesenhada como diagrama de blocos em preto e branco, numa coluna; a tabela do Exp. 3 por condição virou texto.
  Nenhum número mudou.
- 26/10/2026: banca simulada e reescrita. Entraram os controles do Exp. 5b (Tabela II nova, `results/exp5b_robustez.md`),
  a leitura nova da PP1, do vazamento e da regra × modelos e três referências; o artigo passou a ter 8 páginas,
  31 referências, Tabelas I a V e 3 figuras. Os diagramas de reserva saíram de `figures/`.
