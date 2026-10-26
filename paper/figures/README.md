# Figuras do artigo

O artigo (`../artigo.tex`) inclui três PDFs desta pasta. Todos já têm a largura com que entram no artigo,
portanto não são reduzidos na inclusão e o menor texto tem 7 pt ou mais no PDF final. Estilo comum: fundo
branco, preto e tons de cinza (legíveis impressos em preto e branco), vírgula decimal, sem travessões.

| Arquivo | No artigo | Origem | Largura |
|---|---|---|---|
| `fig_pipeline.pdf` | Fig. 1, cadeias de processamento | `fig_pipeline.tex` (TikZ) | 8,33 cm, tamanho natural (uma coluna) |
| `fig_exp5b_transferencia.pdf` | Fig. 2, Exp. 5b (PP1) | `src/experiments/make_figures.py` | `\columnwidth` (252 pt) |
| `fig_exp6_noite.pdf` | Fig. 3, Exp. 6 (PP3) | `src/experiments/make_figures.py` | `\textwidth` (516 pt) |
| `fig_exp4_latencia.pdf/.png` | fora do artigo e do pôster (a Tabela IV traz os números) | cópia de `results/figures/`, feita por `tools/atualizar_celular.sh` | uma coluna |

Nenhuma figura contém rosto das bases de vídeo (DMD: licença sem derivados e rostos; UTA-RLDD:
autorização de imagem parcial; NITYMED: CC BY-ND). A figura de amostras, `results/figures/amostras_mrl.pdf`, tem só
recortes de olho do MRL Eye; o CEW não aparece em figura nenhuma, porque a licença dele proíbe redistribuição.

## Fig. 1 (`fig_pipeline.tex`)

Diagrama de blocos retangulares em preto e branco, na fonte do artigo (Times, 8 pt), com rótulos de uma ou
duas linhas: (a) vídeos, EAR, MAR e pose, janelas de 10 s, centragem por sujeito e RF e RL; (b) câmera, Face
Landmarker, olhos, bocejo e cabeça, nível de 0 a 100 e alarme. A centragem por sujeito só existe no Exp. 5b
(o Exp. 6 usa as características brutas), por isso a caixa diz "Centragem por sujeito (Exp. 5b)" (versão de
26/10/2026). Os detalhes de cada etapa ficam no texto (Seção IV).

## Fig. 2 e Fig. 3 (gráficos)

Geradas por `src/experiments/make_figures.py` a partir dos mesmos CSV que dão os números do texto (nenhum
valor digitado à mão). O script grava PDF e PNG de 300 dpi em `results/figures/`, copia o PDF para cá e
interrompe se algum texto ficar abaixo de 7 pt.

- **Fig. 2**: `results/exp5b_transferencia.csv` (variante principal; B3 na UTA-RLDD; AUC, acurácia balanceada e
  acurácia, por janela e por vídeo; IC 95% por bootstrap de sujeitos). RF = círculo preto, RL = quadrado
  cinza; cheio = limiar 0,5, vazado = limiar calibrado no treino; linha tracejada = acaso (50). O DMD (A) não
  tem nível de vídeo, por isso os painéis por vídeo só têm B e C.
- **Fig. 3**: (A) SNR do EAR por vídeo (`results/exp6_noite_videos.csv`, coluna `snr_rapido`; o traço é a
  mediana de `results/exp6_noite.csv`, seção b) no NITYMED, no DMD em movimento e parado e na UTA-RLDD;
  (B) taxa de disparo no limiar 0,5 com IC 95% (`results/exp6_noite.csv`, seção c, `taxa_disparo_janela`,
  características brutas): RF do DMD (cinza escuro), RF da UTA-RLDD (cinza claro) e regra de fechamento
  de 0,5 s ou mais ou bocejo (branco). Grupos e filtros vêm de `src/experiments/exp6_noite.py`.

`exp5_transfer.py` e `exp6_noite.py` também gravam em `results/figures/` uma versão de trabalho com o mesmo
nome, colorida, para os relatórios. Depois de rodar um deles, refaça as figuras do artigo (o `run_all.sh`
já faz isso na etapa `figuras`, que vem depois dos experimentos):

```bash
# na raiz do repositório
.venv/bin/python src/experiments/make_figures.py --only fig_exp5b_transferencia fig_exp6_noite
```

## Compilação e conferência

Compilador: `tectonic` (em `~/.local/bin/tectonic`; baixa os pacotes sob demanda na primeira execução).

```bash
export PATH="$HOME/.local/bin:$PATH"
cd paper/figures                            # a partir da raiz do repositório
tectonic fig_pipeline.tex                   # gera fig_pipeline.pdf ao lado do .tex
../../.venv/bin/python render.py            # largura x altura (cm) e menor fonte (pt) de cada fig_*.pdf
../../.venv/bin/python render.py --png PASTA   # também grava prévias PNG de 300 dpi em PASTA
```

Conferido em 26/10/2026: `fig_pipeline` mede 8,33 × 3,78 cm (Times 8 pt); `fig_exp5b_transferencia`,
8,86 × 8,64 cm; `fig_exp6_noite`, 18,14 × 5,97 cm; nas duas, 7 pt nas marcas, legendas e valores e 8 pt nos
títulos e rótulos dos eixos.

## Uso no artigo (IEEEtran)

```latex
\graphicspath{{figures/}}
\includegraphics{fig_pipeline.pdf}                                  % Fig. 1, tamanho natural
\includegraphics[width=\columnwidth]{fig_exp5b_transferencia.pdf}   % Fig. 2
\includegraphics[width=\textwidth]{fig_exp6_noite.pdf}              % Fig. 3 (figure*)
```

## Observações

- Os parâmetros escritos na Fig. 1 (janela de 10 s, nível de 0 a 100) são escolhas de desenho documentadas
  em `results/exp5_caracteristicas.md` e `app/pwa/README.md`, não resultados.
- Os diagramas de reserva que ficavam aqui (desenho experimental dos slides, sinal de uma piscada e
  protocolos de partição), o `preamble.tex` que só eles usavam e as cópias PNG que nada usava saíram em
  26/10/2026.
