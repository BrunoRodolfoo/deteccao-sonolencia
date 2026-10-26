# Figuras de resultados (Entrega 02, artigo IEEE e pôster)

Cada figura é gerada por um script a partir dos CSV e JSON de `results/` e `models/`. Nenhum número de
resultado é digitado nos scripts; as únicas constantes são os valores publicados por Ghoddoosian et al.
(2019), Tabelas 2 e 3, também listados em `results/exp1_intra.md`. Nenhuma figura mostra rosto das bases
de vídeo (DMD, UTA-RLDD e NITYMED), e nenhuma mostra recortes do CEW, cuja licença proíbe redistribuição.

```bash
.venv/bin/python src/experiments/make_figures.py                 # Exp. 1 a 4 e as Fig. 2 e 3 do artigo
.venv/bin/python src/experiments/make_figures.py --only fig_exp3_cross --cew-variant crop
.venv/bin/python src/experiments/make_figures.py --only fig_exp5b_transferencia fig_exp6_noite
.venv/bin/python src/pipeline_b/make_figures.py                  # amostras_mrl
```

O `run_all.sh` faz isso na etapa `figuras`, depois dos experimentos.

Formato: `<nome>.pdf` (vetorial, com a fonte embutida) e `<nome>.png` (300 dpi, para o pôster e os
relatórios). As figuras dos Exp. 1 a 4 têm largura fixa de **8,5 cm** (uma coluna IEEE; usar
`\includegraphics[width=\columnwidth]{...}` sem redimensionar, para manter 8 pt no corpo). Fontes: 8 pt em
eixos e rótulos, de 7 a 7,5 pt em legendas e anotações, 6,5 pt só em notas de rodapé. Cores: azul `#2a78d6`
e laranja `#eb6834` (par conferido para daltonismo), cinzas para as referências publicadas (HM-LSTM = cinza
escuro tracejado; humanos = cinza claro pontilhado) e para acaso e eixos. Vírgula decimal em todos os
rótulos. As Fig. 2 e 3 do artigo são em tons de cinza e têm as larguras descritas em `paper/figures/README.md`.

| Figura | Conteúdo | Gerada por | Fonte dos dados (arquivo, colunas e linhas) |
|---|---|---|---|
| `fig_exp1_va_por_fold` | VA por fold (3 classes) do melhor clássico (RF) e do melhor temporal (LSTM), escolhidos pela VA média; HM-LSTM e humanos por fold; linha de acaso por vídeo | `src/experiments/make_figures.py` | `results/exp1_intra.csv` (`scheme=3c`, `metric=va`, folds 1 a 5; acaso = `model=Chance(majoritaria)`); HM-LSTM 64/61/70/64/67 e humanos 62/59/60/53/55 = Tabela 3 de Ghoddoosian et al. (2019), constantes no script |
| `fig_exp1_bsa_vs_va` | BSA e VA (média ± dp populacional, ddof=0, nos 5 folds) para LogReg, SVM-RBF, RF, LSTM e CNN1D; barras hachuradas = HM-LSTM (54,0 / 65,2) e humanos (VA 57,8) publicados; linhas tracejadas = acaso em BSA e VA | `src/experiments/make_figures.py` | `results/exp1_intra.csv` (`scheme=3c`, `metric∈{bsa,va}`), os mesmos valores de `results/exp1_intra.md` |
| `fig_exp2_vazamento` | Partições (i) janela, (ii) vídeo e (iii) sujeito para RF e LSTM; painel BSA e painel VA; rótulos = diferença em pp para (iii) | `src/experiments/make_figures.py` | `results/exp2_leakage_plot.csv` (`metric∈{bsa,va}`, colunas `mean` e `std`); VA não existe em (i) por construção (`n/a`); ver `results/exp2_leakage.md` |
| `fig_exp3_cross` | (a) matriz treino × teste {MRL, CEW} com acurácia balanceada média ± dp (ddof=1, como em `exp3_eye_state.md`) nos 5 folds, com a célula MRL→CEW também para o modelo final (37 sujeitos); (b) barras por condição do MRL (óculos, iluminação, reflexo): MRL→MRL (fora da amostra) contra CEW→MRL (média ± dp dos 5 modelos de fold) | `src/experiments/make_figures.py` | `results/exp3_eye_state.csv` (`stratum=todos`, `metric=bal_acc`; protocolos `A_intra_mrl`, `B_folds_mrl_to_cew`, `B_final_mrl_to_cew`, `C_cew_<var>_to_mrl`, `D_intra_cew_<var>`) e `results/exp3_strat_conditions.csv` (`test_on=MRL`, `stratum∈{glasses,lighting,reflections}`, `metric=bal_acc`; A com `fold=oof`, C com folds 1 a 5). `--cew-variant patch` (padrão, recorte 24×24 oficial) ou `crop` |
| `fig_exp4_latencia` | (a) latência p50 (barra) e p95 (bigode) em lote 1 no desktop: CNN do olho fp32 e int8 com 1 e 4 threads e LSTM de piscadas; (b) tamanho dos `.tflite` em KB; (c) celular, por aparelho: Face Landmarker, CNN, motor ou LSTM e total por quadro (mediana entre as execuções de cada aparelho) | `src/experiments/make_figures.py`; refeita por `tools/atualizar_celular.sh` depois de cada medição no celular | `results/exp4_desktop_latency.csv` (`p50_ms`, `p95_ms`, `tamanho_kb`, `maquina`, `tf_version`); `models/blink_temporal.json` (`latencia_mac_cpu_ms`, `tamanho_kb`); `results/exp4_phone_latency.csv` (`*_ms_p50`, `*_ms_p95`, `aparelho`, versão do app), gerado por `tools/import_phone_benchmark.py` |
| `fig_exp5a_validacao`, `fig_exp5a_distribuicoes` | Exp. 5a: custo de trocar o extrator (dlib → MediaPipe) com IC; distribuição de cada uma das 4 características por classe, oficial × nosso | `src/experiments/exp5a_validacao.py` | `results/exp5a_validacao*.csv` |
| `fig_exp5b_transferencia` | Fig. 2 do artigo. Exp. 5b: AUC e acurácia por janela e por vídeo nos cenários A (DMD→DMD), B (UTA→UTA) e C (DMD→UTA) | `src/experiments/make_figures.py` (versão do artigo, em tons de cinza) | `results/exp5b_transferencia.csv` (seção abaixo) |
| `fig_exp6_noite` | Fig. 3 do artigo. Exp. 6: (A) SNR do EAR por vídeo; (B) taxa de disparo no limiar 0,5 | `src/experiments/make_figures.py` (versão do artigo, em tons de cinza) | `results/exp6_noite.csv` e `results/exp6_noite_videos.csv` (detalhes em `paper/figures/README.md`) |
| `fig_exp6_microssono` | PERCLOS por vídeo e exemplos de séries de EAR em vídeos do NITYMED (só sinais, sem imagem) | `src/experiments/exp6_noite.py` | `results/exp6_noite_videos.csv` e sinais em `data/signals/nitymed/` |
| `diag_sinal_dmd.png` | Verificação do alinhamento: EAR do Face Mesh contra a anotação de olho fechado, 20 s do motorista gA_1 do DMD (só sinais, sem imagem) | `src/features/check_dmd_alignment.py` | `results/sinais_alinhamento_dmd.csv` |
| `amostras_mrl` | Recortes de olho do MRL Eye (48×48, infravermelho): aberto e fechado, com e sem óculos, iluminação boa e ruim, 3 exemplos por condição | `src/pipeline_b/make_figures.py` | `data/cache/mrl_48.npz` (via `src/pipeline_b/datasets.py`), sorteio com semente 42 |

Observações:
- `fig_exp4_latencia` só tem os números do celular que estão em `results/exp4_phone_latency.csv`; nenhum
  valor de celular é inserido à mão. Depois de uma medição nova, `bash tools/atualizar_celular.sh` refaz a
  figura e a copia para `paper/figures/`.
- `fig_exp2_vazamento`: o que infla (i) são janelas do mesmo vídeo nos dois lados (sequências vizinhas dividem
  28 das 30 piscadas). Em (ii) o resultado fica abaixo de (iii); uma explicação provável, que não isolamos, é que
  o modelo associa a aparência da pessoa a rótulos que não são os do vídeo de teste. Ver `results/exp2_leakage.md`.
- `amostras_mrl` substituiu em 26/10/2026 a antiga `amostras_mrl_cew`, que tinha recortes do CEW e foi para a
  Lixeira (o repositório é público e a licença do CEW proíbe redistribuição).

## Exp. 5b: `fig_exp5b_transferencia`

**Rodada final** (27/09/2026; a rodada preliminar, com janelas de 30 s e só o Fold 1, foi descartada). Ponto +
IC 95% (bootstrap percentil sobre sujeitos, 1000 reamostragens, seed 42); pontos em vez de barras porque a linha
de base natural da AUC e da acurácia balanceada é o acaso (50), não 0. Grupos: A = DMD → DMD (sonolência atuada,
deixando um sujeito de fora entre os 13), B = UTA → UTA (sonolência real, 5 folds oficiais) e C = DMD → UTA
(transferência: treino em todo o DMD, teste na UTA inteira). Painéis: AUC por janela de 10 s; AUC por vídeo
(escore do vídeo lógico = média dos escores das janelas; A não tem nível de vídeo); acurácia balanceada por
janela; acurácia por vídeo. RF = círculo preto, RL = quadrado cinza; marcador cheio = limiar 0,5, vazado =
limiar calibrado só com o treino (C: o limiar que maximiza a acurácia balanceada no LOSO do DMD, sem olhar a UTA;
A: LOSO aninhado; B: validação interna nos 4 folds de treino); linha tracejada = acaso (50). Variante principal
(DMD: alerta = safe_drive da s2, sonolento = s5 com fechamento anotado de 0,5 s ou mais ou bocejo; características
centradas na referência do sujeito), esquema B3 (KSS 0 × 10) na UTA. Dados: `results/exp5b_transferencia.csv`
(`experimento∈{A_dmd_intra,B_uta_intra,C_dmd_para_uta}`, `variante=principal`, `estrato=todos`,
`limiar_tipo∈{fixo_0.5,calibrado_treino}`, `metrica∈{auc,acuracia_balanceada,acuracia}`).

`src/experiments/exp5_transfer.py` (e `exp6_noite.py`, para a Fig. 3) também grava aqui uma versão de trabalho
colorida, com o mesmo nome. Depois de rodar um deles, refaça a versão do artigo:
`.venv/bin/python src/experiments/make_figures.py --only fig_exp5b_transferencia fig_exp6_noite`.

Os controles da queda B − C (referência no vídeo KSS 5, sem centragem, B com 13 sujeitos, bootstrap em dois
níveis) não têm figura: estão na Tabela II do artigo e em `results/exp5b_robustez.md`.
