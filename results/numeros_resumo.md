# Números do artigo e do pôster (gerados de results/)

Gerado em 27/10/2026 11:31 por `src/experiments/make_paper_numbers.py`. Cada linha é um macro de `paper/numeros.tex` (o mesmo valor está em `poster/numeros.json`). Nenhum número foi digitado: todos saem do arquivo e da coluna/filtro indicados. Macros derivados (`IC`, `ComIC`, `DP`, `MDP`) saem da mesma linha. Percentuais sem o símbolo %; diferenças em pontos percentuais (pp).

Galaxy S23 Ultra: sem dados; macro `\SamsungPendente` = "medição prevista" e `\ifSamsungMedido` falso.

Total: 658 números (1298 macros).

| Macro | Valor | Unidade | O que é | Arquivo de origem | Coluna / filtro |
|---|---|---|---|---|---|
| `\BaseUTASujeitos` | 60 |  | UTA-RLDD: sujeitos (todos os vídeos em disco) | `results/exp5b_transferencia.csv` | `n_sujeitos` onde experimento=B_uta_intra, variante=principal, modelo=RF, esquema=B1, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia |
| `\BaseUTAVideos` | 180 |  | UTA-RLDD: vídeos lógicos | `results/exp5b_transferencia.csv` | `n_unidades` onde experimento=B_uta_intra, variante=principal, modelo=RF, esquema=B1, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia |
| `\BaseUTAArquivos` | 182 |  | UTA-RLDD: arquivos de vídeo (dois vídeos sonolentos vêm em 2 partes) | `results/exp5b_transferencia.md` | regex `180 vídeos lógicos \((\d+) arquivos\)` (grupo 1) |
| `\BaseUTASequencias` | 8.661 |  | UTA-RLDD: sequências de 30 piscadas (características oficiais, 5 folds) | `results/exp2_leakage.md` | regex `Pool: (\d+) sequências de teste dos 5 folds oficiais, (\d+) vídeos, (\d+) sujeitos` (grupo 1) |
| `\BaseUTASujeitosOficiais` | 58 |  | UTA-RLDD: sujeitos nos .npy oficiais | `results/exp2_leakage.md` | regex `Pool: (\d+) sequências de teste dos 5 folds oficiais, (\d+) vídeos, (\d+) sujeitos` (grupo 3) |
| `\BaseUTAVideosOficiais` | 174 |  | UTA-RLDD: vídeos nos .npy oficiais | `results/exp2_leakage.md` | regex `Pool: (\d+) sequências de teste dos 5 folds oficiais, (\d+) vídeos, (\d+) sujeitos` (grupo 2) |
| `\BaseUTAJanelasBTres` | 12.487 |  | UTA-RLDD: janelas de 10 s avaliáveis, B3 | `results/exp5b_transferencia.csv` | `n_unidades` onde experimento=B_uta_intra, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia |
| `\BaseUTAJanelasSonolentas` | 7.581 |  | UTA-RLDD: janelas KSS 10 | `results/exp5b_transferencia.csv` | `n_pos` onde experimento=B_uta_intra, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia |
| `\BaseUTAJanelasAlerta` | 4.906 |  | UTA-RLDD: janelas KSS 0 avaliáveis | `results/exp5b_transferencia.csv` | `n_neg` onde experimento=B_uta_intra, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia |
| `\BaseMRLImagens` | 84.898 |  | MRL Eye: imagens | `results/exp3_dataset_stats.md` | regex `Imagens: \*\*(\d+)\*\*` (grupo 1) |
| `\BaseMRLSujeitos` | 37 |  | MRL Eye: sujeitos | `results/exp3_dataset_stats.md` | regex `sujeitos: \*\*(\d+)\*\*` (grupo 1) |
| `\BaseCEWImagens` | 4.846 |  | CEW: recortes de olho | `results/exp3_eye_state.csv` | `value` onde protocol=B_final_mrl_to_cew, test_on=CEW_patch, fold=final, stratum=todos, metric=n |
| `\BaseDMDSujeitos` | 13 |  | DMD: sujeitos com sessão de sonolência (s5) | `results/exp5b_transferencia.csv` | `n_sujeitos` onde experimento=A_dmd_intra, variante=principal, modelo=RF, esquema=DMD, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia |
| `\BaseDMDJanelas` | 510 |  | DMD: janelas de 10 s do Exp. 5b (principal) | `results/exp5b_transferencia.csv` | `n_unidades` onde experimento=A_dmd_intra, variante=principal, modelo=RF, esquema=DMD, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia |
| `\BaseDMDJanelasSonolentas` | 423 |  | DMD: janelas sonolentas (s5 com fechamento >= 0,5 s ou bocejo) | `results/exp5b_transferencia.csv` | `n_pos` onde experimento=A_dmd_intra, variante=principal, modelo=RF, esquema=DMD, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia |
| `\BaseDMDJanelasAlerta` | 87 |  | DMD: janelas alerta (safe_drive da s2) | `results/exp5b_transferencia.csv` | `n_neg` onde experimento=A_dmd_intra, variante=principal, modelo=RF, esquema=DMD, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia |
| `\BaseDMDSessoesSonolencia` | 16 |  | DMD: sessões de sonolência s5 | `results/exp6_noite.md` | regex `DMD: (\d+) sessões de (\d+) sujeitos \(s5 (\d+), s1 (\d+), s2 (\d+), s3 (\d+)\)` (grupo 3) |
| `\BaseDMDSessoesDirecaoReal` | 17 |  | DMD: sessões s1 (carro em movimento, direção real) | `results/exp6_noite.md` | regex `DMD: (\d+) sessões de (\d+) sujeitos \(s5 (\d+), s1 (\d+), s2 (\d+), s3 (\d+)\)` (grupo 4) |
| `\BaseDMDSessoes` | 65 |  | DMD: sessões usadas (s1, s2, s3, s5) | `results/exp6_noite.md` | regex `DMD: (\d+) sessões de (\d+) sujeitos \(s5 (\d+), s1 (\d+), s2 (\d+), s3 (\d+)\)` (grupo 1) |
| `\BaseDMDSujeitosTodos` | 14 |  | DMD: sujeitos baixados (13 com s5 + gE_28) | `results/exp6_noite.md` | regex `DMD: (\d+) sessões de (\d+) sujeitos \(s5 (\d+), s1 (\d+), s2 (\d+), s3 (\d+)\)` (grupo 2) |
| `\BaseNITYMEDVideos` | 126 |  | NITYMED: vídeos em disco | `results/exp6_noite.csv` | `n_videos` onde secao=a, grupo=nitymed&#124;todos, metrica=taxa_rosto_media_por_video |
| `\BaseNITYMEDMicrossono` | 19 |  | NITYMED: vídeos de microssono | `results/exp6_noite.csv` | `n_videos` onde secao=a, grupo=nitymed&#124;Microsleep, metrica=taxa_rosto_media_por_video |
| `\BaseNITYMEDBocejo` | 107 |  | NITYMED: vídeos de bocejo | `results/exp6_noite.csv` | `n_videos` onde secao=a, grupo=nitymed&#124;Yawning, metrica=taxa_rosto_media_por_video |
| `\BaseNITYMEDDeclarados` | 130 |  | NITYMED: vídeos declarados na página oficial | `results/exp6_nitymed_ficha.md` | regex `Conteúdo declarado \&#124; (\d+) vídeos` (grupo 1) |
| `\BaseNITYMEDJanelas` | 979 |  | NITYMED: janelas de 10 s | `results/exp6_noite.md` | regex `NITYMED: (\d+) vídeos, (\d+) janelas, (\d+) avaliáveis` (grupo 2) |
| `\BaseNITYMEDJanelasAvaliaveis` | 529 |  | NITYMED: janelas avaliáveis (fora do 1º terço) | `results/exp6_noite.md` | regex `NITYMED: (\d+) vídeos, (\d+) janelas, (\d+) avaliáveis` (grupo 3) |
| `\ExpUmRFBSA` | 54,5 ± 9,3 | % (×100) | Exp. 1 RandomForest, esquema 3c, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=3c, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFAcBal` | 57,8 ± 7,9 | % (×100) | Exp. 1 RandomForest, esquema 3c, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=3c, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFFMacro` | 56,0 ± 9,1 | % (×100) | Exp. 1 RandomForest, esquema 3c, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=3c, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFAUC` | 73,5 ± 6,8 | % (×100) | Exp. 1 RandomForest, esquema 3c, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=3c, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFVA` | 57,3 ± 7,2 | % (×100) | Exp. 1 RandomForest, esquema 3c, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=3c, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMBSA` | 50,8 ± 8,9 | % (×100) | Exp. 1 LSTM, esquema 3c, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=3c, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMAcBal` | 53,1 ± 6,7 | % (×100) | Exp. 1 LSTM, esquema 3c, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=3c, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMFMacro` | 51,8 ± 7,8 | % (×100) | Exp. 1 LSTM, esquema 3c, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=3c, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMAUC` | 68,6 ± 8,2 | % (×100) | Exp. 1 LSTM, esquema 3c, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=3c, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMVA` | 57,2 ± 7,5 | % (×100) | Exp. 1 LSTM, esquema 3c, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=3c, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNBSA` | 48,0 ± 6,8 | % (×100) | Exp. 1 CNN1D, esquema 3c, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=3c, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNAcBal` | 51,4 ± 5,2 | % (×100) | Exp. 1 CNN1D, esquema 3c, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=3c, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNFMacro` | 50,0 ± 6,4 | % (×100) | Exp. 1 CNN1D, esquema 3c, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=3c, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNAUC` | 68,4 ± 6,1 | % (×100) | Exp. 1 CNN1D, esquema 3c, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=3c, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNVA` | 55,0 ± 8,2 | % (×100) | Exp. 1 CNN1D, esquema 3c, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=3c, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMBSA` | 53,0 ± 3,8 | % (×100) | Exp. 1 SVM-RBF, esquema 3c, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=3c, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMAcBal` | 56,2 ± 4,5 | % (×100) | Exp. 1 SVM-RBF, esquema 3c, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=3c, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMFMacro` | 54,2 ± 3,9 | % (×100) | Exp. 1 SVM-RBF, esquema 3c, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=3c, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMAUC` | 69,9 ± 5,1 | % (×100) | Exp. 1 SVM-RBF, esquema 3c, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=3c, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMVA` | 54,0 ± 4,0 | % (×100) | Exp. 1 SVM-RBF, esquema 3c, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=3c, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRBSA` | 49,4 ± 4,3 | % (×100) | Exp. 1 LogReg, esquema 3c, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=3c, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRAcBal` | 49,7 ± 3,0 | % (×100) | Exp. 1 LogReg, esquema 3c, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=3c, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRFMacro` | 48,5 ± 3,6 | % (×100) | Exp. 1 LogReg, esquema 3c, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=3c, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRAUC` | 68,9 ± 5,5 | % (×100) | Exp. 1 LogReg, esquema 3c, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=3c, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRVA` | 50,7 ± 7,6 | % (×100) | Exp. 1 LogReg, esquema 3c, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=3c, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmChanceBSA` | 44,4 ± 2,4 | % (×100) | Exp. 1 chance (classe majoritária), esquema 3c, bsa | `results/exp1_intra.csv` | `value` [mean] onde model=Chance(majoritaria), scheme=3c, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmChanceVA` | 33,3 ± 0,0 | % (×100) | Exp. 1 chance (classe majoritária), esquema 3c, va | `results/exp1_intra.csv` | `value` [mean] onde model=Chance(majoritaria), scheme=3c, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFBTresBSA` | 86,9 ± 3,9 | % (×100) | Exp. 1 RandomForest, esquema B3, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=B3, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFBTresAcBal` | 84,8 ± 3,7 | % (×100) | Exp. 1 RandomForest, esquema B3, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=B3, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFBTresFMacro` | 84,6 ± 4,1 | % (×100) | Exp. 1 RandomForest, esquema B3, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=B3, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFBTresAUC` | 92,8 ± 3,3 | % (×100) | Exp. 1 RandomForest, esquema B3, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=B3, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFBTresVA` | 85,5 ± 6,6 | % (×100) | Exp. 1 RandomForest, esquema B3, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=B3, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMBTresBSA` | 83,0 ± 6,2 | % (×100) | Exp. 1 LSTM, esquema B3, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=B3, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMBTresAcBal` | 80,8 ± 2,3 | % (×100) | Exp. 1 LSTM, esquema B3, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=B3, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMBTresFMacro` | 80,3 ± 5,1 | % (×100) | Exp. 1 LSTM, esquema B3, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=B3, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMBTresAUC` | 91,1 ± 4,1 | % (×100) | Exp. 1 LSTM, esquema B3, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=B3, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMBTresVA` | 81,2 ± 7,7 | % (×100) | Exp. 1 LSTM, esquema B3, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=B3, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNBTresBSA` | 85,2 ± 5,1 | % (×100) | Exp. 1 CNN1D, esquema B3, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=B3, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNBTresAcBal` | 83,8 ± 1,3 | % (×100) | Exp. 1 CNN1D, esquema B3, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=B3, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNBTresFMacro` | 83,1 ± 4,4 | % (×100) | Exp. 1 CNN1D, esquema B3, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=B3, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNBTresAUC` | 93,8 ± 2,5 | % (×100) | Exp. 1 CNN1D, esquema B3, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=B3, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNBTresVA` | 84,5 ± 5,9 | % (×100) | Exp. 1 CNN1D, esquema B3, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=B3, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMBTresBSA` | 85,5 ± 4,9 | % (×100) | Exp. 1 SVM-RBF, esquema B3, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=B3, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMBTresAcBal` | 85,0 ± 3,0 | % (×100) | Exp. 1 SVM-RBF, esquema B3, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=B3, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMBTresFMacro` | 83,6 ± 5,1 | % (×100) | Exp. 1 SVM-RBF, esquema B3, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=B3, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMBTresAUC` | 92,5 ± 3,5 | % (×100) | Exp. 1 SVM-RBF, esquema B3, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=B3, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMBTresVA` | 82,0 ± 5,3 | % (×100) | Exp. 1 SVM-RBF, esquema B3, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=B3, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRBTresBSA` | 74,0 ± 9,1 | % (×100) | Exp. 1 LogReg, esquema B3, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=B3, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRBTresAcBal` | 72,2 ± 7,9 | % (×100) | Exp. 1 LogReg, esquema B3, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=B3, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRBTresFMacro` | 71,0 ± 9,0 | % (×100) | Exp. 1 LogReg, esquema B3, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=B3, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRBTresAUC` | 81,8 ± 7,4 | % (×100) | Exp. 1 LogReg, esquema B3, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=B3, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRBTresVA` | 77,5 ± 9,8 | % (×100) | Exp. 1 LogReg, esquema B3, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=B3, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmChanceBTresBSA` | 68,6 ± 4,8 | % (×100) | Exp. 1 chance (classe majoritária), esquema B3, bsa | `results/exp1_intra.csv` | `value` [mean] onde model=Chance(majoritaria), scheme=B3, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmChanceBTresVA` | 50,0 ± 0,0 | % (×100) | Exp. 1 chance (classe majoritária), esquema B3, va | `results/exp1_intra.csv` | `value` [mean] onde model=Chance(majoritaria), scheme=B3, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFBUmBSA` | 87,4 ± 3,2 | % (×100) | Exp. 1 RandomForest, esquema B1, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=B1, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFBUmAcBal` | 79,8 ± 4,3 | % (×100) | Exp. 1 RandomForest, esquema B1, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=B1, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFBUmFMacro` | 79,4 ± 2,9 | % (×100) | Exp. 1 RandomForest, esquema B1, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=B1, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFBUmAUC` | 88,7 ± 5,1 | % (×100) | Exp. 1 RandomForest, esquema B1, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=B1, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFBUmVA` | 82,2 ± 3,3 | % (×100) | Exp. 1 RandomForest, esquema B1, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=B1, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMBUmBSA` | 81,1 ± 9,4 | % (×100) | Exp. 1 LSTM, esquema B1, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=B1, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMBUmAcBal` | 73,8 ± 4,9 | % (×100) | Exp. 1 LSTM, esquema B1, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=B1, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMBUmFMacro` | 72,2 ± 7,7 | % (×100) | Exp. 1 LSTM, esquema B1, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=B1, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMBUmAUC` | 85,2 ± 7,9 | % (×100) | Exp. 1 LSTM, esquema B1, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=B1, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMBUmVA` | 77,8 ± 8,3 | % (×100) | Exp. 1 LSTM, esquema B1, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=B1, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNBUmBSA` | 83,2 ± 8,5 | % (×100) | Exp. 1 CNN1D, esquema B1, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=B1, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNBUmAcBal` | 75,6 ± 6,7 | % (×100) | Exp. 1 CNN1D, esquema B1, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=B1, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNBUmFMacro` | 74,2 ± 7,4 | % (×100) | Exp. 1 CNN1D, esquema B1, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=B1, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNBUmAUC` | 87,5 ± 6,5 | % (×100) | Exp. 1 CNN1D, esquema B1, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=B1, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNBUmVA` | 81,3 ± 6,8 | % (×100) | Exp. 1 CNN1D, esquema B1, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=B1, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMBUmBSA` | 85,1 ± 2,9 | % (×100) | Exp. 1 SVM-RBF, esquema B1, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=B1, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMBUmAcBal` | 78,0 ± 1,7 | % (×100) | Exp. 1 SVM-RBF, esquema B1, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=B1, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMBUmFMacro` | 76,6 ± 2,4 | % (×100) | Exp. 1 SVM-RBF, esquema B1, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=B1, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMBUmAUC` | 89,9 ± 2,2 | % (×100) | Exp. 1 SVM-RBF, esquema B1, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=B1, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMBUmVA` | 81,8 ± 6,4 | % (×100) | Exp. 1 SVM-RBF, esquema B1, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=B1, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRBUmBSA` | 78,5 ± 6,7 | % (×100) | Exp. 1 LogReg, esquema B1, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=B1, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRBUmAcBal` | 60,1 ± 5,5 | % (×100) | Exp. 1 LogReg, esquema B1, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=B1, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRBUmFMacro` | 60,3 ± 5,8 | % (×100) | Exp. 1 LogReg, esquema B1, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=B1, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRBUmAUC` | 80,3 ± 6,5 | % (×100) | Exp. 1 LogReg, esquema B1, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=B1, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRBUmVA` | 70,9 ± 4,5 | % (×100) | Exp. 1 LogReg, esquema B1, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=B1, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmChanceBUmBSA` | 81,4 ± 2,8 | % (×100) | Exp. 1 chance (classe majoritária), esquema B1, bsa | `results/exp1_intra.csv` | `value` [mean] onde model=Chance(majoritaria), scheme=B1, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmChanceBUmVA` | 66,7 ± 0,0 | % (×100) | Exp. 1 chance (classe majoritária), esquema B1, va | `results/exp1_intra.csv` | `value` [mean] onde model=Chance(majoritaria), scheme=B1, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFBDoisBSA` | 66,5 ± 9,2 | % (×100) | Exp. 1 RandomForest, esquema B2, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=B2, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFBDoisAcBal` | 63,4 ± 10,2 | % (×100) | Exp. 1 RandomForest, esquema B2, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=B2, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFBDoisFMacro` | 62,9 ± 10,5 | % (×100) | Exp. 1 RandomForest, esquema B2, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=B2, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFBDoisAUC` | 70,8 ± 11,6 | % (×100) | Exp. 1 RandomForest, esquema B2, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=B2, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmRFBDoisVA` | 70,1 ± 4,1 | % (×100) | Exp. 1 RandomForest, esquema B2, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=RandomForest, scheme=B2, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMBDoisBSA` | 60,9 ± 8,6 | % (×100) | Exp. 1 LSTM, esquema B2, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=B2, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMBDoisAcBal` | 59,3 ± 9,9 | % (×100) | Exp. 1 LSTM, esquema B2, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=B2, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMBDoisFMacro` | 58,5 ± 9,5 | % (×100) | Exp. 1 LSTM, esquema B2, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=B2, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMBDoisAUC` | 63,7 ± 11,8 | % (×100) | Exp. 1 LSTM, esquema B2, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=B2, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLSTMBDoisVA` | 67,0 ± 7,7 | % (×100) | Exp. 1 LSTM, esquema B2, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LSTM, scheme=B2, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNBDoisBSA` | 61,2 ± 9,9 | % (×100) | Exp. 1 CNN1D, esquema B2, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=B2, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNBDoisAcBal` | 60,8 ± 11,1 | % (×100) | Exp. 1 CNN1D, esquema B2, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=B2, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNBDoisFMacro` | 59,7 ± 10,4 | % (×100) | Exp. 1 CNN1D, esquema B2, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=B2, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNBDoisAUC` | 65,2 ± 12,9 | % (×100) | Exp. 1 CNN1D, esquema B2, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=B2, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmCNNBDoisVA` | 66,5 ± 6,7 | % (×100) | Exp. 1 CNN1D, esquema B2, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=CNN1D, scheme=B2, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMBDoisBSA` | 66,2 ± 5,9 | % (×100) | Exp. 1 SVM-RBF, esquema B2, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=B2, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMBDoisAcBal` | 63,9 ± 7,1 | % (×100) | Exp. 1 SVM-RBF, esquema B2, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=B2, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMBDoisFMacro` | 63,7 ± 7,0 | % (×100) | Exp. 1 SVM-RBF, esquema B2, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=B2, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMBDoisAUC` | 66,2 ± 12,1 | % (×100) | Exp. 1 SVM-RBF, esquema B2, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=B2, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmSVMBDoisVA` | 69,4 ± 3,1 | % (×100) | Exp. 1 SVM-RBF, esquema B2, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=SVM-RBF, scheme=B2, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRBDoisBSA` | 66,8 ± 6,5 | % (×100) | Exp. 1 LogReg, esquema B2, bsa (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=B2, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRBDoisAcBal` | 63,4 ± 7,2 | % (×100) | Exp. 1 LogReg, esquema B2, bal_acc (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=B2, metric=bal_acc, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRBDoisFMacro` | 63,0 ± 8,1 | % (×100) | Exp. 1 LogReg, esquema B2, f1_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=B2, metric=f1_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRBDoisAUC` | 66,9 ± 10,3 | % (×100) | Exp. 1 LogReg, esquema B2, auc_macro (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=B2, metric=auc_macro, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmLRBDoisVA` | 73,6 ± 2,3 | % (×100) | Exp. 1 LogReg, esquema B2, va (média ± dp, ddof=0, 5 folds) | `results/exp1_intra.csv` | `value` [mean] onde model=LogReg, scheme=B2, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmChanceBDoisBSA` | 59,2 ± 4,5 | % (×100) | Exp. 1 chance (classe majoritária), esquema B2, bsa | `results/exp1_intra.csv` | `value` [mean] onde model=Chance(majoritaria), scheme=B2, metric=bsa, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmChanceBDoisVA` | 66,7 ± 0,0 | % (×100) | Exp. 1 chance (classe majoritária), esquema B2, va | `results/exp1_intra.csv` | `value` [mean] onde model=Chance(majoritaria), scheme=B2, metric=va, fold={1,2,3,4,5}; dp std0 |
| `\ExpUmHMLSTMBSA` | 54,0 | % | HM-LSTM (Ghoddoosian et al., 2019), BSA | `results/exp1_intra.md` | regex `HM-LSTM: BSA ([\d.]+)% / VA ([\d.]+)%; LSTM: BSA ([\d.]+)% / VA ([\d.]+)%; Camadas densas: BSA ([\d.]+)% / VA ([\d.]+)%; Humanos: BSA — / VA ([\d.]+)%` (grupo 1) |
| `\ExpUmHMLSTMVA` | 65,2 | % | HM-LSTM (Ghoddoosian et al., 2019), VA | `results/exp1_intra.md` | regex `HM-LSTM: BSA ([\d.]+)% / VA ([\d.]+)%; LSTM: BSA ([\d.]+)% / VA ([\d.]+)%; Camadas densas: BSA ([\d.]+)% / VA ([\d.]+)%; Humanos: BSA — / VA ([\d.]+)%` (grupo 2) |
| `\ExpUmRefLSTMBSA` | 52,8 | % | LSTM do artigo original, BSA | `results/exp1_intra.md` | regex `HM-LSTM: BSA ([\d.]+)% / VA ([\d.]+)%; LSTM: BSA ([\d.]+)% / VA ([\d.]+)%; Camadas densas: BSA ([\d.]+)% / VA ([\d.]+)%; Humanos: BSA — / VA ([\d.]+)%` (grupo 3) |
| `\ExpUmRefLSTMVA` | 61,4 | % | LSTM do artigo original, VA | `results/exp1_intra.md` | regex `HM-LSTM: BSA ([\d.]+)% / VA ([\d.]+)%; LSTM: BSA ([\d.]+)% / VA ([\d.]+)%; Camadas densas: BSA ([\d.]+)% / VA ([\d.]+)%; Humanos: BSA — / VA ([\d.]+)%` (grupo 4) |
| `\ExpUmRefDensasBSA` | 52,0 | % | camadas densas do artigo original, BSA | `results/exp1_intra.md` | regex `HM-LSTM: BSA ([\d.]+)% / VA ([\d.]+)%; LSTM: BSA ([\d.]+)% / VA ([\d.]+)%; Camadas densas: BSA ([\d.]+)% / VA ([\d.]+)%; Humanos: BSA — / VA ([\d.]+)%` (grupo 5) |
| `\ExpUmRefDensasVA` | 57,0 | % | camadas densas do artigo original, VA | `results/exp1_intra.md` | regex `HM-LSTM: BSA ([\d.]+)% / VA ([\d.]+)%; LSTM: BSA ([\d.]+)% / VA ([\d.]+)%; Camadas densas: BSA ([\d.]+)% / VA ([\d.]+)%; Humanos: BSA — / VA ([\d.]+)%` (grupo 6) |
| `\ExpUmHumanosVA` | 57,8 | % | julgamento humano (artigo original), VA | `results/exp1_intra.md` | regex `HM-LSTM: BSA ([\d.]+)% / VA ([\d.]+)%; LSTM: BSA ([\d.]+)% / VA ([\d.]+)%; Camadas densas: BSA ([\d.]+)% / VA ([\d.]+)%; Humanos: BSA — / VA ([\d.]+)%` (grupo 7) |
| `\ExpUmDifHMLSTMRFVA` | 7,9 | pp | HM-LSTM − nosso RF, VA 3 classes (pp) | `(derivado)` | `ExpUmHMLSTMVA - ExpUmRFVA*100` com ExpUmHMLSTMVA, ExpUmRFVA |
| `\ExpUmDifRFHumanosVA` | −0,5 | pp | nosso RF − humanos, VA 3 classes (pp) | `(derivado)` | `ExpUmRFVA*100 - ExpUmHumanosVA` com ExpUmHumanosVA, ExpUmRFVA |
| `\ExpUmLSTMParametros` | 5.324 |  | LSTM exportado: parâmetros | `models/blink_temporal.json` | chave `treino.parametros` |
| `\ExpUmLSTMKB` | 101,8 | KB | LSTM .tflite: tamanho | `models/blink_temporal.json` | chave `tamanho_kb` |
| `\ExpUmLSTMConcordancia` | 100,0 | % (×100) | LSTM TFLite x Keras: concordância do argmax | `models/blink_temporal.json` | chave `validacao_tflite_vs_keras.concordancia_argmax` |
| `\ExpUmLSTMPcinquentaMac` | 0,033 | ms | LSTM TFLite no Mac (CPU, lote 1): p50 | `models/blink_temporal.json` | chave `latencia_mac_cpu_ms.p50` |
| `\ExpUmLSTMPnoventaecincoMac` | 0,045 | ms | LSTM TFLite no Mac (CPU, lote 1): p95 | `models/blink_temporal.json` | chave `latencia_mac_cpu_ms.p95` |
| `\ExpDoisRFJanelaBSA` | 99,5 | % (×100) | Exp. 2 RandomForest, partição por janela, bsa (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=janela, model=RandomForest, metric=bsa |
| `\ExpDoisRFJanelaAcBal` | 99,6 | % (×100) | Exp. 2 RandomForest, partição por janela, bal_acc (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=janela, model=RandomForest, metric=bal_acc |
| `\ExpDoisRFJanelaAUC` | 100,0 | % (×100) | Exp. 2 RandomForest, partição por janela, auc_macro (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=janela, model=RandomForest, metric=auc_macro |
| `\ExpDoisRFVideoBSA` | 47,2 | % (×100) | Exp. 2 RandomForest, partição por video, bsa (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=video, model=RandomForest, metric=bsa |
| `\ExpDoisRFVideoAcBal` | 52,5 | % (×100) | Exp. 2 RandomForest, partição por video, bal_acc (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=video, model=RandomForest, metric=bal_acc |
| `\ExpDoisRFVideoAUC` | 67,9 | % (×100) | Exp. 2 RandomForest, partição por video, auc_macro (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=video, model=RandomForest, metric=auc_macro |
| `\ExpDoisRFVideoVA` | 52,8 | % (×100) | Exp. 2 RandomForest, partição por video, va (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=video, model=RandomForest, metric=va |
| `\ExpDoisRFSujeitoBSA` | 54,2 | % (×100) | Exp. 2 RandomForest, partição por sujeito, bsa (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=sujeito, model=RandomForest, metric=bsa |
| `\ExpDoisRFSujeitoAcBal` | 57,4 | % (×100) | Exp. 2 RandomForest, partição por sujeito, bal_acc (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=sujeito, model=RandomForest, metric=bal_acc |
| `\ExpDoisRFSujeitoAUC` | 73,4 | % (×100) | Exp. 2 RandomForest, partição por sujeito, auc_macro (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=sujeito, model=RandomForest, metric=auc_macro |
| `\ExpDoisRFSujeitoVA` | 55,7 | % (×100) | Exp. 2 RandomForest, partição por sujeito, va (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=sujeito, model=RandomForest, metric=va |
| `\ExpDoisRFInflacaoJanelaBSA` | 45,3 | pp | Exp. 2 RandomForest: janela − sujeito, bsa (pp; positivo = inflado) | `(derivado)` | `(ExpDoisRFJanelaBSA - ExpDoisRFSujeitoBSA)*100` com ExpDoisRFJanelaBSA, ExpDoisRFSujeitoBSA |
| `\ExpDoisRFInflacaoJanelaAcBal` | 42,1 | pp | Exp. 2 RandomForest: janela − sujeito, bal_acc (pp; positivo = inflado) | `(derivado)` | `(ExpDoisRFJanelaAcBal - ExpDoisRFSujeitoAcBal)*100` com ExpDoisRFJanelaAcBal, ExpDoisRFSujeitoAcBal |
| `\ExpDoisRFInflacaoJanelaAUC` | 26,6 | pp | Exp. 2 RandomForest: janela − sujeito, auc_macro (pp; positivo = inflado) | `(derivado)` | `(ExpDoisRFJanelaAUC - ExpDoisRFSujeitoAUC)*100` com ExpDoisRFJanelaAUC, ExpDoisRFSujeitoAUC |
| `\ExpDoisRFInflacaoVideoBSA` | −7,0 | pp | Exp. 2 RandomForest: video − sujeito, bsa (pp; positivo = inflado) | `(derivado)` | `(ExpDoisRFVideoBSA - ExpDoisRFSujeitoBSA)*100` com ExpDoisRFVideoBSA, ExpDoisRFSujeitoBSA |
| `\ExpDoisRFInflacaoVideoAcBal` | −5,0 | pp | Exp. 2 RandomForest: video − sujeito, bal_acc (pp; positivo = inflado) | `(derivado)` | `(ExpDoisRFVideoAcBal - ExpDoisRFSujeitoAcBal)*100` com ExpDoisRFVideoAcBal, ExpDoisRFSujeitoAcBal |
| `\ExpDoisRFInflacaoVideoAUC` | −5,5 | pp | Exp. 2 RandomForest: video − sujeito, auc_macro (pp; positivo = inflado) | `(derivado)` | `(ExpDoisRFVideoAUC - ExpDoisRFSujeitoAUC)*100` com ExpDoisRFVideoAUC, ExpDoisRFSujeitoAUC |
| `\ExpDoisRFInflacaoVideoVA` | −2,8 | pp | Exp. 2 RandomForest: video − sujeito, va (pp; positivo = inflado) | `(derivado)` | `(ExpDoisRFVideoVA - ExpDoisRFSujeitoVA)*100` com ExpDoisRFVideoVA, ExpDoisRFSujeitoVA |
| `\ExpDoisLSTMJanelaBSA` | 87,3 | % (×100) | Exp. 2 LSTM, partição por janela, bsa (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=janela, model=LSTM, metric=bsa |
| `\ExpDoisLSTMJanelaAcBal` | 87,9 | % (×100) | Exp. 2 LSTM, partição por janela, bal_acc (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=janela, model=LSTM, metric=bal_acc |
| `\ExpDoisLSTMJanelaAUC` | 97,1 | % (×100) | Exp. 2 LSTM, partição por janela, auc_macro (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=janela, model=LSTM, metric=auc_macro |
| `\ExpDoisLSTMVideoBSA` | 45,1 | % (×100) | Exp. 2 LSTM, partição por video, bsa (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=video, model=LSTM, metric=bsa |
| `\ExpDoisLSTMVideoAcBal` | 48,9 | % (×100) | Exp. 2 LSTM, partição por video, bal_acc (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=video, model=LSTM, metric=bal_acc |
| `\ExpDoisLSTMVideoAUC` | 62,3 | % (×100) | Exp. 2 LSTM, partição por video, auc_macro (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=video, model=LSTM, metric=auc_macro |
| `\ExpDoisLSTMVideoVA` | 46,2 | % (×100) | Exp. 2 LSTM, partição por video, va (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=video, model=LSTM, metric=va |
| `\ExpDoisLSTMSujeitoBSA` | 49,2 | % (×100) | Exp. 2 LSTM, partição por sujeito, bsa (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=sujeito, model=LSTM, metric=bsa |
| `\ExpDoisLSTMSujeitoAcBal` | 51,0 | % (×100) | Exp. 2 LSTM, partição por sujeito, bal_acc (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=sujeito, model=LSTM, metric=bal_acc |
| `\ExpDoisLSTMSujeitoAUC` | 67,6 | % (×100) | Exp. 2 LSTM, partição por sujeito, auc_macro (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=sujeito, model=LSTM, metric=auc_macro |
| `\ExpDoisLSTMSujeitoVA` | 50,8 | % (×100) | Exp. 2 LSTM, partição por sujeito, va (média) | `results/exp2_leakage_plot.csv` | `mean` onde protocol=sujeito, model=LSTM, metric=va |
| `\ExpDoisLSTMInflacaoJanelaBSA` | 38,1 | pp | Exp. 2 LSTM: janela − sujeito, bsa (pp; positivo = inflado) | `(derivado)` | `(ExpDoisLSTMJanelaBSA - ExpDoisLSTMSujeitoBSA)*100` com ExpDoisLSTMJanelaBSA, ExpDoisLSTMSujeitoBSA |
| `\ExpDoisLSTMInflacaoJanelaAcBal` | 36,9 | pp | Exp. 2 LSTM: janela − sujeito, bal_acc (pp; positivo = inflado) | `(derivado)` | `(ExpDoisLSTMJanelaAcBal - ExpDoisLSTMSujeitoAcBal)*100` com ExpDoisLSTMJanelaAcBal, ExpDoisLSTMSujeitoAcBal |
| `\ExpDoisLSTMInflacaoJanelaAUC` | 29,6 | pp | Exp. 2 LSTM: janela − sujeito, auc_macro (pp; positivo = inflado) | `(derivado)` | `(ExpDoisLSTMJanelaAUC - ExpDoisLSTMSujeitoAUC)*100` com ExpDoisLSTMJanelaAUC, ExpDoisLSTMSujeitoAUC |
| `\ExpDoisLSTMInflacaoVideoBSA` | −4,1 | pp | Exp. 2 LSTM: video − sujeito, bsa (pp; positivo = inflado) | `(derivado)` | `(ExpDoisLSTMVideoBSA - ExpDoisLSTMSujeitoBSA)*100` com ExpDoisLSTMVideoBSA, ExpDoisLSTMSujeitoBSA |
| `\ExpDoisLSTMInflacaoVideoAcBal` | −2,1 | pp | Exp. 2 LSTM: video − sujeito, bal_acc (pp; positivo = inflado) | `(derivado)` | `(ExpDoisLSTMVideoAcBal - ExpDoisLSTMSujeitoAcBal)*100` com ExpDoisLSTMVideoAcBal, ExpDoisLSTMSujeitoAcBal |
| `\ExpDoisLSTMInflacaoVideoAUC` | −5,3 | pp | Exp. 2 LSTM: video − sujeito, auc_macro (pp; positivo = inflado) | `(derivado)` | `(ExpDoisLSTMVideoAUC - ExpDoisLSTMSujeitoAUC)*100` com ExpDoisLSTMVideoAUC, ExpDoisLSTMSujeitoAUC |
| `\ExpDoisLSTMInflacaoVideoVA` | −4,6 | pp | Exp. 2 LSTM: video − sujeito, va (pp; positivo = inflado) | `(derivado)` | `(ExpDoisLSTMVideoVA - ExpDoisLSTMSujeitoVA)*100` com ExpDoisLSTMVideoVA, ExpDoisLSTMSujeitoVA |
| `\ExpDoisVideosVistosJanela` | 97,1 | % (×100) | Exp. 2 partição por janela: vídeos do teste vistos no treino | `results/exp2_leakage_plot.csv` | `mean` onde protocol=janela, model=RandomForest, metric=leak_videos |
| `\ExpTresIntraMRLAc` | 94,1 ± 3,4 | % (×100) | Exp. 3 A intra-MRL, acc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=A_intra_mrl, test_on=MRL, metric=acc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresMRLCEWPatchAc` | 79,0 ± 8,5 | % (×100) | Exp. 3 B MRL→CEW patch (modelos dos folds), acc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=B_folds_mrl_to_cew, test_on=CEW_patch, metric=acc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresMRLCEWCropAc` | 80,6 ± 10,3 | % (×100) | Exp. 3 B MRL→CEW crop (modelos dos folds), acc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=B_folds_mrl_to_cew, test_on=CEW_crop, metric=acc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresCEWPatchMRLAc` | 68,1 ± 1,9 | % (×100) | Exp. 3 C CEW patch→MRL, acc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, test_on=MRL, metric=acc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresCEWCropMRLAc` | 67,1 ± 1,8 | % (×100) | Exp. 3 C CEW crop→MRL, acc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=C_cew_crop_to_mrl, test_on=MRL, metric=acc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresIntraCEWPatchAc` | 94,7 ± 1,1 | % (×100) | Exp. 3 D intra-CEW patch, acc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=D_intra_cew_patch, test_on=CEW_patch, metric=acc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresIntraCEWCropAc` | 94,5 ± 1,3 | % (×100) | Exp. 3 D intra-CEW crop, acc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=D_intra_cew_crop, test_on=CEW_crop, metric=acc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresFinalMRLCEWPatchAc` | 85,7 | % (×100) | Exp. 3 B modelo final (37 sujeitos) → CEW_patch, acc | `results/exp3_eye_state.csv` | `value` onde protocol=B_final_mrl_to_cew, test_on=CEW_patch, fold=final, stratum=todos, metric=acc |
| `\ExpTresFinalMRLCEWCropAc` | 90,0 | % (×100) | Exp. 3 B modelo final (37 sujeitos) → CEW_crop, acc | `results/exp3_eye_state.csv` | `value` onde protocol=B_final_mrl_to_cew, test_on=CEW_crop, fold=final, stratum=todos, metric=acc |
| `\ExpTresIntraMRLAcBal` | 94,2 ± 3,5 | % (×100) | Exp. 3 A intra-MRL, bal_acc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=A_intra_mrl, test_on=MRL, metric=bal_acc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresMRLCEWPatchAcBal` | 78,7 ± 8,6 | % (×100) | Exp. 3 B MRL→CEW patch (modelos dos folds), bal_acc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=B_folds_mrl_to_cew, test_on=CEW_patch, metric=bal_acc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresMRLCEWCropAcBal` | 80,3 ± 10,5 | % (×100) | Exp. 3 B MRL→CEW crop (modelos dos folds), bal_acc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=B_folds_mrl_to_cew, test_on=CEW_crop, metric=bal_acc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresCEWPatchMRLAcBal` | 68,4 ± 1,9 | % (×100) | Exp. 3 C CEW patch→MRL, bal_acc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, test_on=MRL, metric=bal_acc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresCEWCropMRLAcBal` | 67,4 ± 1,8 | % (×100) | Exp. 3 C CEW crop→MRL, bal_acc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=C_cew_crop_to_mrl, test_on=MRL, metric=bal_acc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresIntraCEWPatchAcBal` | 94,7 ± 1,1 | % (×100) | Exp. 3 D intra-CEW patch, bal_acc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=D_intra_cew_patch, test_on=CEW_patch, metric=bal_acc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresIntraCEWCropAcBal` | 94,4 ± 1,4 | % (×100) | Exp. 3 D intra-CEW crop, bal_acc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=D_intra_cew_crop, test_on=CEW_crop, metric=bal_acc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresFinalMRLCEWPatchAcBal` | 85,5 | % (×100) | Exp. 3 B modelo final (37 sujeitos) → CEW_patch, bal_acc | `results/exp3_eye_state.csv` | `value` onde protocol=B_final_mrl_to_cew, test_on=CEW_patch, fold=final, stratum=todos, metric=bal_acc |
| `\ExpTresFinalMRLCEWCropAcBal` | 89,9 | % (×100) | Exp. 3 B modelo final (37 sujeitos) → CEW_crop, bal_acc | `results/exp3_eye_state.csv` | `value` onde protocol=B_final_mrl_to_cew, test_on=CEW_crop, fold=final, stratum=todos, metric=bal_acc |
| `\ExpTresIntraMRLAUC` | 98,6 ± 1,3 | % (×100) | Exp. 3 A intra-MRL, auc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=A_intra_mrl, test_on=MRL, metric=auc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresMRLCEWPatchAUC` | 93,2 ± 4,7 | % (×100) | Exp. 3 B MRL→CEW patch (modelos dos folds), auc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=B_folds_mrl_to_cew, test_on=CEW_patch, metric=auc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresMRLCEWCropAUC` | 93,5 ± 4,9 | % (×100) | Exp. 3 B MRL→CEW crop (modelos dos folds), auc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=B_folds_mrl_to_cew, test_on=CEW_crop, metric=auc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresCEWPatchMRLAUC` | 84,2 ± 2,8 | % (×100) | Exp. 3 C CEW patch→MRL, auc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, test_on=MRL, metric=auc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresCEWCropMRLAUC` | 83,6 ± 3,1 | % (×100) | Exp. 3 C CEW crop→MRL, auc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=C_cew_crop_to_mrl, test_on=MRL, metric=auc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresIntraCEWPatchAUC` | 98,9 ± 0,5 | % (×100) | Exp. 3 D intra-CEW patch, auc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=D_intra_cew_patch, test_on=CEW_patch, metric=auc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresIntraCEWCropAUC` | 98,8 ± 0,3 | % (×100) | Exp. 3 D intra-CEW crop, auc (média ± dp, ddof=1, 5 folds) | `results/exp3_eye_state.csv` | `value` [mean] onde protocol=D_intra_cew_crop, test_on=CEW_crop, metric=auc, fold={1,2,3,4,5}, stratum=todos; dp std1 |
| `\ExpTresFinalMRLCEWPatchAUC` | 96,8 | % (×100) | Exp. 3 B modelo final (37 sujeitos) → CEW_patch, auc | `results/exp3_eye_state.csv` | `value` onde protocol=B_final_mrl_to_cew, test_on=CEW_patch, fold=final, stratum=todos, metric=auc |
| `\ExpTresFinalMRLCEWCropAUC` | 97,6 | % (×100) | Exp. 3 B modelo final (37 sujeitos) → CEW_crop, auc | `results/exp3_eye_state.csv` | `value` onde protocol=B_final_mrl_to_cew, test_on=CEW_crop, fold=final, stratum=todos, metric=auc |
| `\ExpTresQuedaMRLCEWPatch` | 15,1 | pp | Exp. 3 queda de acurácia intra-MRL − MRL→CEW patch (pp) | `(derivado)` | `(ExpTresIntraMRLAc - ExpTresMRLCEWPatchAc)*100` com ExpTresIntraMRLAc, ExpTresMRLCEWPatchAc |
| `\ExpTresQuedaCEWMRLPatch` | 26,5 | pp | Exp. 3 queda de acurácia intra-CEW patch − CEW patch→MRL (pp) | `(derivado)` | `(ExpTresIntraCEWPatchAc - ExpTresCEWPatchMRLAc)*100` com ExpTresIntraCEWPatchAc, ExpTresCEWPatchMRLAc |
| `\ExpTresIntraMRLComOculosAc` | 88,8 | % (×100) | Exp. 3 A intra-MRL com óculos (out-of-fold agregado), acc | `results/exp3_strat_conditions.csv` | `value` onde protocol=A_intra_mrl, fold=oof, stratum=glasses, level=com óculos, metric=acc |
| `\ExpTresCEWPatchMRLComOculosAc` | 65,2 ± 5,1 | % (×100) | Exp. 3 C CEW patch→MRL com óculos, acc (média ± dp, ddof=1) | `results/exp3_strat_conditions.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, fold={1,2,3,4,5}, stratum=glasses, level=com óculos, metric=acc; dp std1 |
| `\ExpTresIntraMRLComOculosAcBal` | 88,5 | % (×100) | Exp. 3 A intra-MRL com óculos (out-of-fold agregado), bal_acc | `results/exp3_strat_conditions.csv` | `value` onde protocol=A_intra_mrl, fold=oof, stratum=glasses, level=com óculos, metric=bal_acc |
| `\ExpTresCEWPatchMRLComOculosAcBal` | 61,1 ± 2,8 | % (×100) | Exp. 3 C CEW patch→MRL com óculos, bal_acc (média ± dp, ddof=1) | `results/exp3_strat_conditions.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, fold={1,2,3,4,5}, stratum=glasses, level=com óculos, metric=bal_acc; dp std1 |
| `\ExpTresIntraMRLComOculosAUC` | 95,7 | % (×100) | Exp. 3 A intra-MRL com óculos (out-of-fold agregado), auc | `results/exp3_strat_conditions.csv` | `value` onde protocol=A_intra_mrl, fold=oof, stratum=glasses, level=com óculos, metric=auc |
| `\ExpTresCEWPatchMRLComOculosAUC` | 65,1 ± 4,3 | % (×100) | Exp. 3 C CEW patch→MRL com óculos, auc (média ± dp, ddof=1) | `results/exp3_strat_conditions.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, fold={1,2,3,4,5}, stratum=glasses, level=com óculos, metric=auc; dp std1 |
| `\ExpTresIntraMRLSemOculosAc` | 96,1 | % (×100) | Exp. 3 A intra-MRL sem óculos (out-of-fold agregado), acc | `results/exp3_strat_conditions.csv` | `value` onde protocol=A_intra_mrl, fold=oof, stratum=glasses, level=sem óculos, metric=acc |
| `\ExpTresCEWPatchMRLSemOculosAc` | 69,3 ± 4,4 | % (×100) | Exp. 3 C CEW patch→MRL sem óculos, acc (média ± dp, ddof=1) | `results/exp3_strat_conditions.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, fold={1,2,3,4,5}, stratum=glasses, level=sem óculos, metric=acc; dp std1 |
| `\ExpTresIntraMRLSemOculosAcBal` | 96,0 | % (×100) | Exp. 3 A intra-MRL sem óculos (out-of-fold agregado), bal_acc | `results/exp3_strat_conditions.csv` | `value` onde protocol=A_intra_mrl, fold=oof, stratum=glasses, level=sem óculos, metric=bal_acc |
| `\ExpTresCEWPatchMRLSemOculosAcBal` | 71,8 ± 4,0 | % (×100) | Exp. 3 C CEW patch→MRL sem óculos, bal_acc (média ± dp, ddof=1) | `results/exp3_strat_conditions.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, fold={1,2,3,4,5}, stratum=glasses, level=sem óculos, metric=bal_acc; dp std1 |
| `\ExpTresIntraMRLSemOculosAUC` | 99,4 | % (×100) | Exp. 3 A intra-MRL sem óculos (out-of-fold agregado), auc | `results/exp3_strat_conditions.csv` | `value` onde protocol=A_intra_mrl, fold=oof, stratum=glasses, level=sem óculos, metric=auc |
| `\ExpTresCEWPatchMRLSemOculosAUC` | 93,4 ± 2,3 | % (×100) | Exp. 3 C CEW patch→MRL sem óculos, auc (média ± dp, ddof=1) | `results/exp3_strat_conditions.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, fold={1,2,3,4,5}, stratum=glasses, level=sem óculos, metric=auc; dp std1 |
| `\ExpTresParametros` | 62.865 |  | EyeNet: parâmetros | `results/exp3_meta.json` | chave `eyenet_params` |
| `\ExpQuatroCNNIntPcinquenta` | 0,060 | ms | Exp. 4 CNN do olho int8, 1 thread(s), desktop, p50_ms | `results/exp4_desktop_latency.csv` | `p50_ms` onde modelo=int8, threads=1 |
| `\ExpQuatroCNNIntPnoventaecinco` | 0,064 | ms | Exp. 4 CNN do olho int8, 1 thread(s), desktop, p95_ms | `results/exp4_desktop_latency.csv` | `p95_ms` onde modelo=int8, threads=1 |
| `\ExpQuatroCNNIntQuatroThreadsPcinquenta` | 0,070 | ms | Exp. 4 CNN do olho int8, 4 thread(s), desktop, p50_ms | `results/exp4_desktop_latency.csv` | `p50_ms` onde modelo=int8, threads=4 |
| `\ExpQuatroCNNIntQuatroThreadsPnoventaecinco` | 0,087 | ms | Exp. 4 CNN do olho int8, 4 thread(s), desktop, p95_ms | `results/exp4_desktop_latency.csv` | `p95_ms` onde modelo=int8, threads=4 |
| `\ExpQuatroCNNIntKB` | 72,1 | KB | Exp. 4 CNN do olho int8: tamanho do .tflite | `results/exp4_desktop_latency.csv` | `tamanho_kb` onde modelo=int8, threads=1 |
| `\ExpQuatroCNNFloatPcinquenta` | 0,067 | ms | Exp. 4 CNN do olho fp32, 1 thread(s), desktop, p50_ms | `results/exp4_desktop_latency.csv` | `p50_ms` onde modelo=fp32, threads=1 |
| `\ExpQuatroCNNFloatPnoventaecinco` | 0,070 | ms | Exp. 4 CNN do olho fp32, 1 thread(s), desktop, p95_ms | `results/exp4_desktop_latency.csv` | `p95_ms` onde modelo=fp32, threads=1 |
| `\ExpQuatroCNNFloatQuatroThreadsPcinquenta` | 0,082 | ms | Exp. 4 CNN do olho fp32, 4 thread(s), desktop, p50_ms | `results/exp4_desktop_latency.csv` | `p50_ms` onde modelo=fp32, threads=4 |
| `\ExpQuatroCNNFloatQuatroThreadsPnoventaecinco` | 0,099 | ms | Exp. 4 CNN do olho fp32, 4 thread(s), desktop, p95_ms | `results/exp4_desktop_latency.csv` | `p95_ms` onde modelo=fp32, threads=4 |
| `\ExpQuatroCNNFloatKB` | 248,2 | KB | Exp. 4 CNN do olho fp32: tamanho do .tflite | `results/exp4_desktop_latency.csv` | `tamanho_kb` onde modelo=fp32, threads=1 |
| `\ExpQuatroModelosKB` | 173,9 | KB | Exp. 4 CNN int8 + LSTM: tamanho total dos modelos aprendidos | `(derivado)` | `ExpQuatroCNNIntKB + ExpUmLSTMKB` com ExpQuatroCNNIntKB, ExpUmLSTMKB |
| `\ExpQuatroAcCEWPatchKeras` | 85,7 | % (×100) | Exp. 4 acurácia keras em CEW_patch | `results/exp4_tflite_eval.csv` | `value` onde dataset=CEW_patch, modelo=keras, metric=acc |
| `\ExpQuatroAcCEWPatchInt` | 85,6 | % (×100) | Exp. 4 acurácia int8 em CEW_patch | `results/exp4_tflite_eval.csv` | `value` onde dataset=CEW_patch, modelo=int8, metric=acc |
| `\ExpQuatroAcCEWCropKeras` | 90,0 | % (×100) | Exp. 4 acurácia keras em CEW_crop | `results/exp4_tflite_eval.csv` | `value` onde dataset=CEW_crop, modelo=keras, metric=acc |
| `\ExpQuatroAcCEWCropInt` | 89,7 | % (×100) | Exp. 4 acurácia int8 em CEW_crop | `results/exp4_tflite_eval.csv` | `value` onde dataset=CEW_crop, modelo=int8, metric=acc |
| `\ExpQuatroAcMRLCompletoKeras` | 98,5 | % (×100) | Exp. 4 acurácia keras em MRL_completo | `results/exp4_tflite_eval.csv` | `value` onde dataset=MRL_completo, modelo=keras, metric=acc |
| `\ExpQuatroAcMRLCompletoInt` | 98,5 | % (×100) | Exp. 4 acurácia int8 em MRL_completo | `results/exp4_tflite_eval.csv` | `value` onde dataset=MRL_completo, modelo=int8, metric=acc |
| `\ExpQuatroQuedaQuantizacaoMax` | 0,31 | pp | Exp. 4 maior queda de acurácia int8 − Keras (pp) | `(derivado)` | `max(abs(ExpQuatroAcCEWPatchKeras-ExpQuatroAcCEWPatchInt), abs(ExpQuatroAcCEWCropKeras-ExpQuatroAcCEWCropInt), abs(ExpQuatroAcMRLCompletoKeras-ExpQuatroAcMRLCompletoInt))*100` com ExpQuatroAcCEWPatchKeras, ExpQuatroAcCEWPatchInt, ExpQuatroAcCEWCropKeras, ExpQuatroAcCEWCropInt, ExpQuatroAcMRLCompletoKeras, ExpQuatroAcMRLCompletoInt |
| `\MacPWATotalPcinquenta` | 12,7 | ms | PWA v1.0 no Chrome headless do Mac (20/09, vídeo 320x240): total por quadro p50 | `results/pwa_headless_mac_bench_20260920-190012.json` | chave `latencia_ms.total_por_quadro.p50` |
| `\MacPWATotalPnoventaecinco` | 14,8 | ms | PWA v1.0 no Chrome headless do Mac (20/09, vídeo 320x240): total por quadro p95 | `results/pwa_headless_mac_bench_20260920-190012.json` | chave `latencia_ms.total_por_quadro.p95` |
| `\MacPWAFPS` | 29,9 |  | PWA v1.0 no Chrome headless do Mac (20/09, vídeo 320x240): FPS médio | `results/pwa_headless_mac_bench_20260920-190012.json` | chave `fps_medio` |
| `\CelularIPhoneNome` | iPhone 17 Pro (512 GB) |  | Exp. 4 nome do aparelho (iPhone 17 Pro (512 GB)) | `results/exp4_phone_latency.csv` | `aparelho` (valores distintos) onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneExecucoes` | 2 |  | Exp. 4 iPhone 17 Pro (512 GB): número de execuções de 60 s | `results/exp4_phone_latency.csv` | `(linhas)` [count] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneVersaoApp` | 1.0 |  | Exp. 4 iPhone 17 Pro (512 GB): versao_app (valores distintos entre as execuções, em ordem) | `results/exp4_phone_latency.csv` | `versao_app` (valores distintos) onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneSistema` | iOS 26.6.1 |  | Exp. 4 iPhone 17 Pro (512 GB): sistema (valores distintos entre as execuções, em ordem) | `results/exp4_phone_latency.csv` | `sistema` (valores distintos) onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneNavegador` | Chrome 154 (iOS, motor WebKit) |  | Exp. 4 iPhone 17 Pro (512 GB): navegador (valores distintos entre as execuções, em ordem) | `results/exp4_phone_latency.csv` | `navegador` (valores distintos) onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneDelegate` | GPU |  | Exp. 4 iPhone 17 Pro (512 GB): delegate_landmarker (valores distintos entre as execuções, em ordem) | `results/exp4_phone_latency.csv` | `delegate_landmarker` (valores distintos) onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneTotalPcinquenta` | 17,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): total_ms p50 por quadro — mediana entre as execuções | `results/exp4_phone_latency.csv` | `total_ms_p50` [median] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneTotalPcinquentaMin` | 11,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): total_ms p50 por quadro — menor execução | `results/exp4_phone_latency.csv` | `total_ms_p50` [min] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneTotalPcinquentaMax` | 23,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): total_ms p50 por quadro — maior execução | `results/exp4_phone_latency.csv` | `total_ms_p50` [max] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneTotalPnoventaecinco` | 22,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): total_ms p95 por quadro — mediana entre as execuções | `results/exp4_phone_latency.csv` | `total_ms_p95` [median] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneTotalPnoventaecincoMin` | 14,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): total_ms p95 por quadro — menor execução | `results/exp4_phone_latency.csv` | `total_ms_p95` [min] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneTotalPnoventaecincoMax` | 30,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): total_ms p95 por quadro — maior execução | `results/exp4_phone_latency.csv` | `total_ms_p95` [max] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneLandmarkerPcinquenta` | 14,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): landmarker_ms p50 por quadro — mediana entre as execuções | `results/exp4_phone_latency.csv` | `landmarker_ms_p50` [median] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneLandmarkerPcinquentaMin` | 9,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): landmarker_ms p50 por quadro — menor execução | `results/exp4_phone_latency.csv` | `landmarker_ms_p50` [min] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneLandmarkerPcinquentaMax` | 19,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): landmarker_ms p50 por quadro — maior execução | `results/exp4_phone_latency.csv` | `landmarker_ms_p50` [max] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneLandmarkerPnoventaecinco` | 18,5 | ms | Exp. 4 iPhone 17 Pro (512 GB): landmarker_ms p95 por quadro — mediana entre as execuções | `results/exp4_phone_latency.csv` | `landmarker_ms_p95` [median] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneLandmarkerPnoventaecincoMin` | 11,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): landmarker_ms p95 por quadro — menor execução | `results/exp4_phone_latency.csv` | `landmarker_ms_p95` [min] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneLandmarkerPnoventaecincoMax` | 26,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): landmarker_ms p95 por quadro — maior execução | `results/exp4_phone_latency.csv` | `landmarker_ms_p95` [max] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneCNNPcinquenta` | 2,5 | ms | Exp. 4 iPhone 17 Pro (512 GB): cnn_ms p50 por quadro — mediana entre as execuções | `results/exp4_phone_latency.csv` | `cnn_ms_p50` [median] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneCNNPcinquentaMin` | 2,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): cnn_ms p50 por quadro — menor execução | `results/exp4_phone_latency.csv` | `cnn_ms_p50` [min] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneCNNPcinquentaMax` | 3,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): cnn_ms p50 por quadro — maior execução | `results/exp4_phone_latency.csv` | `cnn_ms_p50` [max] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneCNNPnoventaecinco` | 4,5 | ms | Exp. 4 iPhone 17 Pro (512 GB): cnn_ms p95 por quadro — mediana entre as execuções | `results/exp4_phone_latency.csv` | `cnn_ms_p95` [median] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneCNNPnoventaecincoMin` | 3,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): cnn_ms p95 por quadro — menor execução | `results/exp4_phone_latency.csv` | `cnn_ms_p95` [min] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneCNNPnoventaecincoMax` | 6,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): cnn_ms p95 por quadro — maior execução | `results/exp4_phone_latency.csv` | `cnn_ms_p95` [max] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneLSTMPcinquenta` | 1,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): engine_ms p50 por quadro — mediana entre as execuções | `results/exp4_phone_latency.csv` | `engine_ms_p50` [median] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneLSTMPcinquentaMin` | 1,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): engine_ms p50 por quadro — menor execução | `results/exp4_phone_latency.csv` | `engine_ms_p50` [min] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneLSTMPcinquentaMax` | 1,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): engine_ms p50 por quadro — maior execução | `results/exp4_phone_latency.csv` | `engine_ms_p50` [max] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneLSTMPnoventaecinco` | 1,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): engine_ms p95 por quadro — mediana entre as execuções | `results/exp4_phone_latency.csv` | `engine_ms_p95` [median] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneLSTMPnoventaecincoMin` | 1,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): engine_ms p95 por quadro — menor execução | `results/exp4_phone_latency.csv` | `engine_ms_p95` [min] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneLSTMPnoventaecincoMax` | 1,0 | ms | Exp. 4 iPhone 17 Pro (512 GB): engine_ms p95 por quadro — maior execução | `results/exp4_phone_latency.csv` | `engine_ms_p95` [max] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneFPS` | 50,2 |  | Exp. 4 iPhone 17 Pro (512 GB): quadros por segundo (median entre execuções) | `results/exp4_phone_latency.csv` | `fps_medio` [median] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneFPSMin` | 40,5 |  | Exp. 4 iPhone 17 Pro (512 GB): quadros por segundo (min entre execuções) | `results/exp4_phone_latency.csv` | `fps_medio` [min] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneFPSMax` | 60,0 |  | Exp. 4 iPhone 17 Pro (512 GB): quadros por segundo (max entre execuções) | `results/exp4_phone_latency.csv` | `fps_medio` [max] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneTotalPcinquentaInteiro` | 17 | ms | Exp. 4 iPhone 17 Pro (512 GB): total p50 por quadro, em ms inteiros | `(derivado)` | `CelularIPhoneTotalPcinquenta` com CelularIPhoneTotalPcinquenta |
| `\CelularIPhoneTotalPcinquentaMinInteiro` | 11 | ms | Exp. 4 iPhone 17 Pro (512 GB): total p50 por quadro (min), em ms inteiros | `(derivado)` | `CelularIPhoneTotalPcinquentaMin` com CelularIPhoneTotalPcinquentaMin |
| `\CelularIPhoneTotalPcinquentaMaxInteiro` | 23 | ms | Exp. 4 iPhone 17 Pro (512 GB): total p50 por quadro (max), em ms inteiros | `(derivado)` | `CelularIPhoneTotalPcinquentaMax` com CelularIPhoneTotalPcinquentaMax |
| `\CelularIPhoneQuadros` | 6.031 |  | Exp. 4 iPhone 17 Pro (512 GB): quadros somados das execuções | `results/exp4_phone_latency.csv` | `quadros` [sum] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneQuadrosComRosto` | 6.031 |  | Exp. 4 iPhone 17 Pro (512 GB): quadros com rosto | `results/exp4_phone_latency.csv` | `quadros_com_rosto` [sum] onde aparelho=iPhone 17 Pro (512 GB), versao_app=1.0 |
| `\CelularIPhoneTaxaRosto` | 100,0 | % | Exp. 4 iPhone 17 Pro (512 GB): fração de quadros com rosto detectado | `(derivado)` | `CelularIPhoneQuadrosComRosto / CelularIPhoneQuadros * 100` com CelularIPhoneQuadrosComRosto, CelularIPhoneQuadros |
| `\CelularVUmFalsoAlarmePERCLOS` | 25,8 | % (×100) | app v1.0 no iPhone (22:41Z, usuário acordado): PERCLOS no fim do benchmark | `results/phone/benchmark_pwa_2026-09-27T22-41-35-365Z.json` | chave `perclos_final` |
| `\CelularVUmFalsoAlarmeQuadros` | 2.281 |  | app v1.0 no iPhone (22:41Z): quadros com alarme ligado | `results/phone/benchmark_pwa_2026-09-27T22-41-35-365Z.json` | chave `quadros_em_alerta` |
| `\CelularVUmFalsoAlarmeQuadrosTotal` | 3.600 |  | app v1.0 no iPhone (22:41Z): quadros do benchmark | `results/phone/benchmark_pwa_2026-09-27T22-41-35-365Z.json` | chave `quadros` |
| `\CelularVUmFalsoAlarmeFracao` | 63,4 | % | app v1.0 no iPhone (22:41Z): % dos quadros com alarme ligado (falso alarme) | `(derivado)` | `CelularVUmFalsoAlarmeQuadros / CelularVUmFalsoAlarmeQuadrosTotal * 100` com CelularVUmFalsoAlarmeQuadros, CelularVUmFalsoAlarmeQuadrosTotal |
| `\ExpCincoACustoAcBalDif` | 2,0 [−4,8; 8,5] | pp (×100) | Exp. 5a treino_oficial_outros_folds 3c bal_acc: oficial − nosso (pp, IC bootstrap por sujeito) | `results/exp5a_validacao_custo.csv` | `diferenca` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=bal_acc; IC `boot_dif_ic_inf`/`boot_dif_ic_sup` |
| `\ExpCincoACustoAcBalOficial` | 58,9 | % (×100) | Exp. 5a treino_oficial_outros_folds 3c bal_acc: características oficiais | `results/exp5a_validacao_custo.csv` | `oficial` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=bal_acc |
| `\ExpCincoACustoAcBalNosso` | 57,0 | % (×100) | Exp. 5a treino_oficial_outros_folds 3c bal_acc: nosso extrator (MediaPipe) | `results/exp5a_validacao_custo.csv` | `nosso` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=bal_acc |
| `\ExpCincoACustoAcBalP` | 0,555 |  | Exp. 5a treino_oficial_outros_folds 3c bal_acc: p da permutação pareada | `results/exp5a_validacao_custo.csv` | `perm_p` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=bal_acc |
| `\ExpCincoACustoVADif` | 2,1 [−6,4; 9,9] | pp (×100) | Exp. 5a treino_oficial_outros_folds 3c va: oficial − nosso (pp, IC bootstrap por sujeito) | `results/exp5a_validacao_custo.csv` | `diferenca` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=va; IC `boot_dif_ic_inf`/`boot_dif_ic_sup` |
| `\ExpCincoACustoVAOficial` | 58,9 | % (×100) | Exp. 5a treino_oficial_outros_folds 3c va: características oficiais | `results/exp5a_validacao_custo.csv` | `oficial` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=va |
| `\ExpCincoACustoVANosso` | 56,7 | % (×100) | Exp. 5a treino_oficial_outros_folds 3c va: nosso extrator (MediaPipe) | `results/exp5a_validacao_custo.csv` | `nosso` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=va |
| `\ExpCincoACustoVAP` | 0,734 |  | Exp. 5a treino_oficial_outros_folds 3c va: p da permutação pareada | `results/exp5a_validacao_custo.csv` | `perm_p` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=va |
| `\ExpCincoACustoBTresAcBalDif` | 0,5 [−6,4; 7,2] | pp (×100) | Exp. 5a treino_oficial_outros_folds B3 bal_acc: oficial − nosso (pp, IC bootstrap por sujeito) | `results/exp5a_validacao_custo.csv` | `diferenca` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=bal_acc; IC `boot_dif_ic_inf`/`boot_dif_ic_sup` |
| `\ExpCincoACustoBTresAcBalOficial` | 85,5 | % (×100) | Exp. 5a treino_oficial_outros_folds B3 bal_acc: características oficiais | `results/exp5a_validacao_custo.csv` | `oficial` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=bal_acc |
| `\ExpCincoACustoBTresAcBalNosso` | 85,0 | % (×100) | Exp. 5a treino_oficial_outros_folds B3 bal_acc: nosso extrator (MediaPipe) | `results/exp5a_validacao_custo.csv` | `nosso` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=bal_acc |
| `\ExpCincoACustoBTresAcBalP` | 0,904 |  | Exp. 5a treino_oficial_outros_folds B3 bal_acc: p da permutação pareada | `results/exp5a_validacao_custo.csv` | `perm_p` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=bal_acc |
| `\ExpCincoACustoBTresVADif` | 2,1 [−6,4; 10,6] | pp (×100) | Exp. 5a treino_oficial_outros_folds B3 va: oficial − nosso (pp, IC bootstrap por sujeito) | `results/exp5a_validacao_custo.csv` | `diferenca` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=va; IC `boot_dif_ic_inf`/`boot_dif_ic_sup` |
| `\ExpCincoACustoBTresVAOficial` | 88,3 | % (×100) | Exp. 5a treino_oficial_outros_folds B3 va: características oficiais | `results/exp5a_validacao_custo.csv` | `oficial` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=va |
| `\ExpCincoACustoBTresVANosso` | 86,2 | % (×100) | Exp. 5a treino_oficial_outros_folds B3 va: nosso extrator (MediaPipe) | `results/exp5a_validacao_custo.csv` | `nosso` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=va |
| `\ExpCincoACustoBTresVAP` | 0,806 |  | Exp. 5a treino_oficial_outros_folds B3 va: p da permutação pareada | `results/exp5a_validacao_custo.csv` | `perm_p` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=va |
| `\ExpCincoAMesmoAcBalDif` | −1,7 [−10,4; 5,7] | pp (×100) | Exp. 5a mesmo_extrator_5_folds 3c bal_acc: oficial − nosso (pp, IC bootstrap por sujeito) | `results/exp5a_validacao_custo.csv` | `diferenca` onde secao=mesmo_extrator_5_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=bal_acc; IC `boot_dif_ic_inf`/`boot_dif_ic_sup` |
| `\ExpCincoAMesmoAcBalOficial` | 54,6 | % (×100) | Exp. 5a mesmo_extrator_5_folds 3c bal_acc: características oficiais | `results/exp5a_validacao_custo.csv` | `oficial` onde secao=mesmo_extrator_5_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=bal_acc |
| `\ExpCincoAMesmoAcBalNosso` | 56,3 | % (×100) | Exp. 5a mesmo_extrator_5_folds 3c bal_acc: nosso extrator (MediaPipe) | `results/exp5a_validacao_custo.csv` | `nosso` onde secao=mesmo_extrator_5_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=bal_acc |
| `\ExpCincoAMesmoAcBalP` | 0,712 |  | Exp. 5a mesmo_extrator_5_folds 3c bal_acc: p da permutação pareada | `results/exp5a_validacao_custo.csv` | `perm_p` onde secao=mesmo_extrator_5_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=bal_acc |
| `\ExpCincoAMesmoVADif` | −5,0 [−15,6; 5,7] | pp (×100) | Exp. 5a mesmo_extrator_5_folds 3c va: oficial − nosso (pp, IC bootstrap por sujeito) | `results/exp5a_validacao_custo.csv` | `diferenca` onde secao=mesmo_extrator_5_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=va; IC `boot_dif_ic_inf`/`boot_dif_ic_sup` |
| `\ExpCincoAMesmoVAOficial` | 55,3 | % (×100) | Exp. 5a mesmo_extrator_5_folds 3c va: características oficiais | `results/exp5a_validacao_custo.csv` | `oficial` onde secao=mesmo_extrator_5_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=va |
| `\ExpCincoAMesmoVANosso` | 60,3 | % (×100) | Exp. 5a mesmo_extrator_5_folds 3c va: nosso extrator (MediaPipe) | `results/exp5a_validacao_custo.csv` | `nosso` onde secao=mesmo_extrator_5_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=va |
| `\ExpCincoAMesmoVAP` | 0,460 |  | Exp. 5a mesmo_extrator_5_folds 3c va: p da permutação pareada | `results/exp5a_validacao_custo.csv` | `perm_p` onde secao=mesmo_extrator_5_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=va |
| `\ExpCincoAMesmoBTresAcBalDif` | −6,7 [−12,9; −1,8] | pp (×100) | Exp. 5a mesmo_extrator_5_folds B3 bal_acc: oficial − nosso (pp, IC bootstrap por sujeito) | `results/exp5a_validacao_custo.csv` | `diferenca` onde secao=mesmo_extrator_5_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=bal_acc; IC `boot_dif_ic_inf`/`boot_dif_ic_sup` |
| `\ExpCincoAMesmoBTresAcBalOficial` | 83,1 | % (×100) | Exp. 5a mesmo_extrator_5_folds B3 bal_acc: características oficiais | `results/exp5a_validacao_custo.csv` | `oficial` onde secao=mesmo_extrator_5_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=bal_acc |
| `\ExpCincoAMesmoBTresAcBalNosso` | 89,8 | % (×100) | Exp. 5a mesmo_extrator_5_folds B3 bal_acc: nosso extrator (MediaPipe) | `results/exp5a_validacao_custo.csv` | `nosso` onde secao=mesmo_extrator_5_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=bal_acc |
| `\ExpCincoAMesmoBTresAcBalP` | 0,029 |  | Exp. 5a mesmo_extrator_5_folds B3 bal_acc: p da permutação pareada | `results/exp5a_validacao_custo.csv` | `perm_p` onde secao=mesmo_extrator_5_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=bal_acc |
| `\ExpCincoAMesmoBTresVADif` | −7,4 [−13,8; −2,1] | pp (×100) | Exp. 5a mesmo_extrator_5_folds B3 va: oficial − nosso (pp, IC bootstrap por sujeito) | `results/exp5a_validacao_custo.csv` | `diferenca` onde secao=mesmo_extrator_5_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=va; IC `boot_dif_ic_inf`/`boot_dif_ic_sup` |
| `\ExpCincoAMesmoBTresVAOficial` | 87,2 | % (×100) | Exp. 5a mesmo_extrator_5_folds B3 va: características oficiais | `results/exp5a_validacao_custo.csv` | `oficial` onde secao=mesmo_extrator_5_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=va |
| `\ExpCincoAMesmoBTresVANosso` | 94,7 | % (×100) | Exp. 5a mesmo_extrator_5_folds B3 va: nosso extrator (MediaPipe) | `results/exp5a_validacao_custo.csv` | `nosso` onde secao=mesmo_extrator_5_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=va |
| `\ExpCincoAMesmoBTresVAP` | 0,039 |  | Exp. 5a mesmo_extrator_5_folds B3 va: p da permutação pareada | `results/exp5a_validacao_custo.csv` | `perm_p` onde secao=mesmo_extrator_5_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=va |
| `\ExpCincoAMesmoLOSOAcBalDif` | 0,4 [−6,7; 7,7] | pp (×100) | Exp. 5a mesmo_extrator_loso 3c bal_acc: oficial − nosso (pp, IC bootstrap por sujeito) | `results/exp5a_validacao_custo.csv` | `diferenca` onde secao=mesmo_extrator_loso, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=bal_acc; IC `boot_dif_ic_inf`/`boot_dif_ic_sup` |
| `\ExpCincoAMesmoLOSOAcBalOficial` | 55,6 | % (×100) | Exp. 5a mesmo_extrator_loso 3c bal_acc: características oficiais | `results/exp5a_validacao_custo.csv` | `oficial` onde secao=mesmo_extrator_loso, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=bal_acc |
| `\ExpCincoAMesmoLOSOAcBalNosso` | 55,2 | % (×100) | Exp. 5a mesmo_extrator_loso 3c bal_acc: nosso extrator (MediaPipe) | `results/exp5a_validacao_custo.csv` | `nosso` onde secao=mesmo_extrator_loso, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=bal_acc |
| `\ExpCincoAMesmoLOSOAcBalP` | 0,910 |  | Exp. 5a mesmo_extrator_loso 3c bal_acc: p da permutação pareada | `results/exp5a_validacao_custo.csv` | `perm_p` onde secao=mesmo_extrator_loso, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=bal_acc |
| `\ExpCincoAMesmoLOSOVADif` | −0,7 [−9,2; 7,1] | pp (×100) | Exp. 5a mesmo_extrator_loso 3c va: oficial − nosso (pp, IC bootstrap por sujeito) | `results/exp5a_validacao_custo.csv` | `diferenca` onde secao=mesmo_extrator_loso, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=va; IC `boot_dif_ic_inf`/`boot_dif_ic_sup` |
| `\ExpCincoAMesmoLOSOVAOficial` | 55,3 | % (×100) | Exp. 5a mesmo_extrator_loso 3c va: características oficiais | `results/exp5a_validacao_custo.csv` | `oficial` onde secao=mesmo_extrator_loso, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=va |
| `\ExpCincoAMesmoLOSOVANosso` | 56,0 | % (×100) | Exp. 5a mesmo_extrator_loso 3c va: nosso extrator (MediaPipe) | `results/exp5a_validacao_custo.csv` | `nosso` onde secao=mesmo_extrator_loso, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=va |
| `\ExpCincoAMesmoLOSOVAP` | 1,000 |  | Exp. 5a mesmo_extrator_loso 3c va: p da permutação pareada | `results/exp5a_validacao_custo.csv` | `perm_p` onde secao=mesmo_extrator_loso, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=va |
| `\ExpCincoAMesmoLOSOBTresAcBalDif` | −4,5 [−11,0; 0,6] | pp (×100) | Exp. 5a mesmo_extrator_loso B3 bal_acc: oficial − nosso (pp, IC bootstrap por sujeito) | `results/exp5a_validacao_custo.csv` | `diferenca` onde secao=mesmo_extrator_loso, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=bal_acc; IC `boot_dif_ic_inf`/`boot_dif_ic_sup` |
| `\ExpCincoAMesmoLOSOBTresAcBalOficial` | 85,0 | % (×100) | Exp. 5a mesmo_extrator_loso B3 bal_acc: características oficiais | `results/exp5a_validacao_custo.csv` | `oficial` onde secao=mesmo_extrator_loso, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=bal_acc |
| `\ExpCincoAMesmoLOSOBTresAcBalNosso` | 89,5 | % (×100) | Exp. 5a mesmo_extrator_loso B3 bal_acc: nosso extrator (MediaPipe) | `results/exp5a_validacao_custo.csv` | `nosso` onde secao=mesmo_extrator_loso, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=bal_acc |
| `\ExpCincoAMesmoLOSOBTresAcBalP` | 0,163 |  | Exp. 5a mesmo_extrator_loso B3 bal_acc: p da permutação pareada | `results/exp5a_validacao_custo.csv` | `perm_p` onde secao=mesmo_extrator_loso, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=bal_acc |
| `\ExpCincoAMesmoLOSOBTresVADif` | −3,2 [−9,6; 4,3] | pp (×100) | Exp. 5a mesmo_extrator_loso B3 va: oficial − nosso (pp, IC bootstrap por sujeito) | `results/exp5a_validacao_custo.csv` | `diferenca` onde secao=mesmo_extrator_loso, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=va; IC `boot_dif_ic_inf`/`boot_dif_ic_sup` |
| `\ExpCincoAMesmoLOSOBTresVAOficial` | 88,3 | % (×100) | Exp. 5a mesmo_extrator_loso B3 va: características oficiais | `results/exp5a_validacao_custo.csv` | `oficial` onde secao=mesmo_extrator_loso, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=va |
| `\ExpCincoAMesmoLOSOBTresVANosso` | 91,5 | % (×100) | Exp. 5a mesmo_extrator_loso B3 va: nosso extrator (MediaPipe) | `results/exp5a_validacao_custo.csv` | `nosso` onde secao=mesmo_extrator_loso, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=va |
| `\ExpCincoAMesmoLOSOBTresVAP` | 0,546 |  | Exp. 5a mesmo_extrator_loso B3 va: p da permutação pareada | `results/exp5a_validacao_custo.csv` | `perm_p` onde secao=mesmo_extrator_loso, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=va |
| `\ExpCincoASujeitos` | 47 |  | Exp. 5a sujeitos com par confirmado | `results/exp5a_validacao_custo.csv` | `n_sujeitos` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=bal_acc |
| `\ExpCincoAVideos` | 141 |  | Exp. 5a vídeos (3 classes) | `results/exp5a_validacao_custo.csv` | `n_videos` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=3c, variante=estendida, metrica=bal_acc |
| `\ExpCincoAVideosBTres` | 94 |  | Exp. 5a vídeos (B3) | `results/exp5a_validacao_custo.csv` | `n_videos` onde secao=treino_oficial_outros_folds, conjunto_sujeitos=confirmado, fold=todos, esquema=B3, variante=estendida, metrica=bal_acc |
| `\ExpCincoBAcBalJanelaA` | 76,4 [68,3; 83,7] | % (×100) | Exp. 5b RF: (A) DMD→DMD, acur. bal. por janela, limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=A_dmd_intra, variante=principal, modelo=RF, esquema=DMD, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBAcBalJanelaB` | 74,2 [70,9; 77,6] | % (×100) | Exp. 5b RF: (B) UTA→UTA, acur. bal. por janela, limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=B_uta_intra, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBAcBalJanelaC` | 65,2 [61,7; 68,6] | % (×100) | Exp. 5b RF: (C) DMD→UTA, acur. bal. por janela, limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBQuedaAcBalJanela` | 9,0 [5,8; 12,2] | pp | Exp. 5b RF: queda B − C, acur. bal. por janela, limiar fixo_0.5 (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBChoqueAcBalJanela` | 11,3 [2,2; 19,8] | pp | Exp. 5b RF: choque de realidade A − C, acur. bal. por janela, limiar fixo_0.5 (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=choque_A_menos_C, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBAcVideoB` | 85,8 [80,0; 91,7] | % (×100) | Exp. 5b RF: (B) UTA→UTA, acurácia por vídeo, limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=B_uta_intra, variante=principal, modelo=RF, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBAcVideoC` | 73,3 [67,5; 79,2] | % (×100) | Exp. 5b RF: (C) DMD→UTA, acurácia por vídeo, limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=RF, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBQuedaAcVideo` | 12,5 [5,0; 19,2] | pp | Exp. 5b RF: queda B − C, acurácia por vídeo, limiar fixo_0.5 (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=principal, modelo=RF, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBCSensJanela` | 48,4 [41,0; 55,2] | % (×100) | Exp. 5b RF: (C) DMD→UTA, sensibilidade por janela (sonolentas/alertas reconhecidas), limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=sensibilidade; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBCSensVideo` | 55,0 [43,3; 66,7] | % (×100) | Exp. 5b RF: (C) DMD→UTA, sensibilidade por vídeo, limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=RF, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=sensibilidade; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBCEspecJanela` | 82,0 [77,0; 86,6] | % (×100) | Exp. 5b RF: (C) DMD→UTA, especificidade por janela (sonolentas/alertas reconhecidas), limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=especificidade; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBCEspecVideo` | 91,7 [85,0; 98,3] | % (×100) | Exp. 5b RF: (C) DMD→UTA, especificidade por vídeo, limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=RF, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=especificidade; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBAcBalJanelaACal` | 78,6 [72,6; 84,2] | % (×100) | Exp. 5b RF: (A) DMD→DMD, acur. bal. por janela, limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=A_dmd_intra, variante=principal, modelo=RF, esquema=DMD, nivel=janela, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBAcBalJanelaBCal` | 74,0 [70,7; 77,4] | % (×100) | Exp. 5b RF: (B) UTA→UTA, acur. bal. por janela, limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=B_uta_intra, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBAcBalJanelaCCal` | 62,9 [60,0; 66,0] | % (×100) | Exp. 5b RF: (C) DMD→UTA, acur. bal. por janela, limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBQuedaAcBalJanelaCal` | 11,1 [8,0; 14,2] | pp | Exp. 5b RF: queda B − C, acur. bal. por janela, limiar calibrado_treino (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBChoqueAcBalJanelaCal` | 15,8 [8,8; 22,1] | pp | Exp. 5b RF: choque de realidade A − C, acur. bal. por janela, limiar calibrado_treino (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=choque_A_menos_C, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBAcVideoBCal` | 82,5 [76,7; 88,3] | % (×100) | Exp. 5b RF: (B) UTA→UTA, acurácia por vídeo, limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=B_uta_intra, variante=principal, modelo=RF, esquema=B3, nivel=video, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBAcVideoCCal` | 60,8 [55,8; 65,8] | % (×100) | Exp. 5b RF: (C) DMD→UTA, acurácia por vídeo, limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=RF, esquema=B3, nivel=video, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBQuedaAcVideoCal` | 21,7 [14,2; 29,2] | pp | Exp. 5b RF: queda B − C, acurácia por vídeo, limiar calibrado_treino (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=principal, modelo=RF, esquema=B3, nivel=video, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBCSensJanelaCal` | 37,0 [30,6; 43,6] | % (×100) | Exp. 5b RF: (C) DMD→UTA, sensibilidade por janela (sonolentas/alertas reconhecidas), limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=calibrado_treino, metrica=sensibilidade; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBCSensVideoCal` | 21,7 [11,7; 31,7] | % (×100) | Exp. 5b RF: (C) DMD→UTA, sensibilidade por vídeo, limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=RF, esquema=B3, nivel=video, estrato=todos, limiar_tipo=calibrado_treino, metrica=sensibilidade; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBCEspecJanelaCal` | 88,7 [85,1; 92,1] | % (×100) | Exp. 5b RF: (C) DMD→UTA, especificidade por janela (sonolentas/alertas reconhecidas), limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=calibrado_treino, metrica=especificidade; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBCEspecVideoCal` | 100,0 [100,0; 100,0] | % (×100) | Exp. 5b RF: (C) DMD→UTA, especificidade por vídeo, limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=RF, esquema=B3, nivel=video, estrato=todos, limiar_tipo=calibrado_treino, metrica=especificidade; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBAUCJanelaA` | 82,4 [73,6; 89,7] | % (×100) | Exp. 5b RF: (A) DMD→DMD, AUC por janela | `results/exp5b_transferencia.csv` | `valor` onde experimento=A_dmd_intra, variante=principal, modelo=RF, esquema=DMD, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBAUCJanelaB` | 80,1 [76,3; 83,8] | % (×100) | Exp. 5b RF: (B) UTA→UTA, AUC por janela | `results/exp5b_transferencia.csv` | `valor` onde experimento=B_uta_intra, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBAUCJanelaC` | 69,2 [63,7; 74,3] | % (×100) | Exp. 5b RF: (C) DMD→UTA, AUC por janela | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBQuedaAUCJanela` | 10,9 [6,7; 15,7] | pp | Exp. 5b RF: queda B − C, AUC por janela (pontos de AUC×100) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBChoqueAUCJanela` | 13,1 [3,5; 22,6] | pp | Exp. 5b RF: choque de realidade A − C, AUC por janela | `results/exp5b_transferencia.csv` | `valor` onde experimento=choque_A_menos_C, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLabAUCJanela` | 2,2 [−7,1; 10,9] | pp | Exp. 5b RF: laboratório − real com treino real, A − B, AUC por janela | `results/exp5b_transferencia.csv` | `valor` onde experimento=lab_A_menos_B, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBAUCVideoB` | 92,4 [87,9; 96,2] | % (×100) | Exp. 5b RF: (B) UTA→UTA, AUC por vídeo | `results/exp5b_transferencia.csv` | `valor` onde experimento=B_uta_intra, variante=principal, modelo=RF, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBAUCVideoC` | 80,5 [73,0; 87,2] | % (×100) | Exp. 5b RF: (C) DMD→UTA, AUC por vídeo | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=RF, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBQuedaAUCVideo` | 11,9 [5,6; 18,8] | pp | Exp. 5b RF: queda B − C, AUC por vídeo | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=principal, modelo=RF, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLimiarCal` | 0,662 |  | Exp. 5b RF: limiar calibrado no LOSO do DMD (aplicado à UTA) | `results/exp5b_transferencia.csv` | `limiar_valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia_balanceada |
| `\ExpCincoBLRAcBalJanelaA` | 78,6 [70,1; 85,3] | % (×100) | Exp. 5b LR: (A) DMD→DMD, acur. bal. por janela, limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=A_dmd_intra, variante=principal, modelo=LR, esquema=DMD, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRAcBalJanelaB` | 72,0 [68,9; 74,9] | % (×100) | Exp. 5b LR: (B) UTA→UTA, acur. bal. por janela, limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=B_uta_intra, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRAcBalJanelaC` | 64,0 [60,7; 67,4] | % (×100) | Exp. 5b LR: (C) DMD→UTA, acur. bal. por janela, limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRQuedaAcBalJanela` | 7,9 [4,8; 10,9] | pp | Exp. 5b LR: queda B − C, acur. bal. por janela, limiar fixo_0.5 (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRChoqueAcBalJanela` | 14,6 [5,6; 22,0] | pp | Exp. 5b LR: choque de realidade A − C, acur. bal. por janela, limiar fixo_0.5 (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=choque_A_menos_C, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRAcVideoB` | 85,8 [80,0; 90,8] | % (×100) | Exp. 5b LR: (B) UTA→UTA, acurácia por vídeo, limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=B_uta_intra, variante=principal, modelo=LR, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRAcVideoC` | 65,8 [60,0; 71,7] | % (×100) | Exp. 5b LR: (C) DMD→UTA, acurácia por vídeo, limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=LR, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRQuedaAcVideo` | 20,0 [12,5; 26,7] | pp | Exp. 5b LR: queda B − C, acurácia por vídeo, limiar fixo_0.5 (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=principal, modelo=LR, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=acuracia; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRCSensJanela` | 33,7 [26,5; 41,5] | % (×100) | Exp. 5b LR: (C) DMD→UTA, sensibilidade por janela (sonolentas/alertas reconhecidas), limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=sensibilidade; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRCSensVideo` | 31,7 [20,0; 43,3] | % (×100) | Exp. 5b LR: (C) DMD→UTA, sensibilidade por vídeo, limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=LR, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=sensibilidade; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRCEspecJanela` | 94,4 [92,2; 96,2] | % (×100) | Exp. 5b LR: (C) DMD→UTA, especificidade por janela (sonolentas/alertas reconhecidas), limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=especificidade; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRCEspecVideo` | 100,0 [100,0; 100,0] | % (×100) | Exp. 5b LR: (C) DMD→UTA, especificidade por vídeo, limiar fixo_0.5 | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=LR, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=especificidade; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRAcBalJanelaACal` | 78,7 [72,1; 83,8] | % (×100) | Exp. 5b LR: (A) DMD→DMD, acur. bal. por janela, limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=A_dmd_intra, variante=principal, modelo=LR, esquema=DMD, nivel=janela, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRAcBalJanelaBCal` | 72,2 [69,1; 75,2] | % (×100) | Exp. 5b LR: (B) UTA→UTA, acur. bal. por janela, limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=B_uta_intra, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRAcBalJanelaCCal` | 63,3 [60,2; 66,7] | % (×100) | Exp. 5b LR: (C) DMD→UTA, acur. bal. por janela, limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRQuedaAcBalJanelaCal` | 8,9 [5,6; 11,9] | pp | Exp. 5b LR: queda B − C, acur. bal. por janela, limiar calibrado_treino (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRChoqueAcBalJanelaCal` | 15,4 [7,8; 21,4] | pp | Exp. 5b LR: choque de realidade A − C, acur. bal. por janela, limiar calibrado_treino (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=choque_A_menos_C, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia_balanceada; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRAcVideoBCal` | 85,0 [79,2; 90,0] | % (×100) | Exp. 5b LR: (B) UTA→UTA, acurácia por vídeo, limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=B_uta_intra, variante=principal, modelo=LR, esquema=B3, nivel=video, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRAcVideoCCal` | 64,2 [58,3; 70,0] | % (×100) | Exp. 5b LR: (C) DMD→UTA, acurácia por vídeo, limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=LR, esquema=B3, nivel=video, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRQuedaAcVideoCal` | 20,8 [12,5; 28,3] | pp | Exp. 5b LR: queda B − C, acurácia por vídeo, limiar calibrado_treino (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=principal, modelo=LR, esquema=B3, nivel=video, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRCSensJanelaCal` | 30,8 [24,1; 38,2] | % (×100) | Exp. 5b LR: (C) DMD→UTA, sensibilidade por janela (sonolentas/alertas reconhecidas), limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=calibrado_treino, metrica=sensibilidade; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRCSensVideoCal` | 28,3 [16,7; 40,0] | % (×100) | Exp. 5b LR: (C) DMD→UTA, sensibilidade por vídeo, limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=LR, esquema=B3, nivel=video, estrato=todos, limiar_tipo=calibrado_treino, metrica=sensibilidade; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRCEspecJanelaCal` | 95,8 [94,0; 97,2] | % (×100) | Exp. 5b LR: (C) DMD→UTA, especificidade por janela (sonolentas/alertas reconhecidas), limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=calibrado_treino, metrica=especificidade; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRCEspecVideoCal` | 100,0 [100,0; 100,0] | % (×100) | Exp. 5b LR: (C) DMD→UTA, especificidade por vídeo, limiar calibrado_treino | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=LR, esquema=B3, nivel=video, estrato=todos, limiar_tipo=calibrado_treino, metrica=especificidade; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRAUCJanelaA` | 83,5 [76,1; 88,9] | % (×100) | Exp. 5b LR: (A) DMD→DMD, AUC por janela | `results/exp5b_transferencia.csv` | `valor` onde experimento=A_dmd_intra, variante=principal, modelo=LR, esquema=DMD, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRAUCJanelaB` | 74,4 [69,7; 78,4] | % (×100) | Exp. 5b LR: (B) UTA→UTA, AUC por janela | `results/exp5b_transferencia.csv` | `valor` onde experimento=B_uta_intra, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRAUCJanelaC` | 64,5 [58,1; 70,9] | % (×100) | Exp. 5b LR: (C) DMD→UTA, AUC por janela | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRQuedaAUCJanela` | 9,8 [3,7; 16,0] | pp | Exp. 5b LR: queda B − C, AUC por janela (pontos de AUC×100) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRChoqueAUCJanela` | 18,9 [9,7; 27,8] | pp | Exp. 5b LR: choque de realidade A − C, AUC por janela | `results/exp5b_transferencia.csv` | `valor` onde experimento=choque_A_menos_C, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRLabAUCJanela` | 9,1 [1,3; 16,6] | pp | Exp. 5b LR: laboratório − real com treino real, A − B, AUC por janela | `results/exp5b_transferencia.csv` | `valor` onde experimento=lab_A_menos_B, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRAUCVideoB` | 89,0 [82,2; 94,9] | % (×100) | Exp. 5b LR: (B) UTA→UTA, AUC por vídeo | `results/exp5b_transferencia.csv` | `valor` onde experimento=B_uta_intra, variante=principal, modelo=LR, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRAUCVideoC` | 76,0 [67,8; 84,7] | % (×100) | Exp. 5b LR: (C) DMD→UTA, AUC por vídeo | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=LR, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRQuedaAUCVideo` | 13,0 [5,8; 20,8] | pp | Exp. 5b LR: queda B − C, AUC por vídeo | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=principal, modelo=LR, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRLimiarCal` | 0,535 |  | Exp. 5b LR: limiar calibrado no LOSO do DMD (aplicado à UTA) | `results/exp5b_transferencia.csv` | `limiar_valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=calibrado_treino, metrica=acuracia_balanceada |
| `\ExpSeisRostoNoite` | 99,15 [98,75; 99,48] | % (×100) | Exp. 6 detecção de rosto por quadro (média por vídeo), nitymed|todos | `results/exp6_noite.csv` | `valor` onde secao=a, grupo=nitymed&#124;todos, metrica=taxa_rosto_media_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisSNRNoite` | 7,2 [6,4; 7,9] |  | Exp. 6 SNR rápida do EAR (mediana por vídeo), nitymed|todos | `results/exp6_noite.csv` | `valor` onde secao=b, grupo=nitymed&#124;todos, metrica=snr_rapido_mediana_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisRuidoNoite` | 0,0084 [0,0080; 0,0090] |  | Exp. 6 ruído rápido do EAR (mediana por vídeo), nitymed|todos | `results/exp6_noite.csv` | `valor` onde secao=b, grupo=nitymed&#124;todos, metrica=ruido_rapido_mediana_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisRostoNoiteMicrossono` | 99,23 [98,49; 99,73] | % (×100) | Exp. 6 detecção de rosto por quadro (média por vídeo), nitymed|Microsleep | `results/exp6_noite.csv` | `valor` onde secao=a, grupo=nitymed&#124;Microsleep, metrica=taxa_rosto_media_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisSNRNoiteMicrossono` | 7,1 [6,4; 11,0] |  | Exp. 6 SNR rápida do EAR (mediana por vídeo), nitymed|Microsleep | `results/exp6_noite.csv` | `valor` onde secao=b, grupo=nitymed&#124;Microsleep, metrica=snr_rapido_mediana_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisRuidoNoiteMicrossono` | 0,0084 [0,0061; 0,0095] |  | Exp. 6 ruído rápido do EAR (mediana por vídeo), nitymed|Microsleep | `results/exp6_noite.csv` | `valor` onde secao=b, grupo=nitymed&#124;Microsleep, metrica=ruido_rapido_mediana_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisRostoNoiteBocejo` | 99,14 [98,69; 99,49] | % (×100) | Exp. 6 detecção de rosto por quadro (média por vídeo), nitymed|Yawning | `results/exp6_noite.csv` | `valor` onde secao=a, grupo=nitymed&#124;Yawning, metrica=taxa_rosto_media_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisSNRNoiteBocejo` | 7,2 [5,9; 7,9] |  | Exp. 6 SNR rápida do EAR (mediana por vídeo), nitymed|Yawning | `results/exp6_noite.csv` | `valor` onde secao=b, grupo=nitymed&#124;Yawning, metrica=snr_rapido_mediana_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisRuidoNoiteBocejo` | 0,0084 [0,0079; 0,0089] |  | Exp. 6 ruído rápido do EAR (mediana por vídeo), nitymed|Yawning | `results/exp6_noite.csv` | `valor` onde secao=b, grupo=nitymed&#124;Yawning, metrica=ruido_rapido_mediana_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisRostoUTA` | 99,77 [99,40; 99,96] | % (×100) | Exp. 6 detecção de rosto por quadro (média por vídeo), uta|todos | `results/exp6_noite.csv` | `valor` onde secao=a, grupo=uta&#124;todos, metrica=taxa_rosto_media_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisSNRUTA` | 21,5 [18,3; 23,5] |  | Exp. 6 SNR rápida do EAR (mediana por vídeo), uta|todos | `results/exp6_noite.csv` | `valor` onde secao=b, grupo=uta&#124;todos, metrica=snr_rapido_mediana_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisRuidoUTA` | 0,0037 [0,0035; 0,0040] |  | Exp. 6 ruído rápido do EAR (mediana por vídeo), uta|todos | `results/exp6_noite.csv` | `valor` onde secao=b, grupo=uta&#124;todos, metrica=ruido_rapido_mediana_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisRostoUTAAlerta` | 99,95 [99,90; 99,99] | % (×100) | Exp. 6 detecção de rosto por quadro (média por vídeo), uta|KSS 0 | `results/exp6_noite.csv` | `valor` onde secao=a, grupo=uta&#124;KSS 0, metrica=taxa_rosto_media_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisSNRUTAAlerta` | 24,4 [21,6; 26,1] |  | Exp. 6 SNR rápida do EAR (mediana por vídeo), uta|KSS 0 | `results/exp6_noite.csv` | `valor` onde secao=b, grupo=uta&#124;KSS 0, metrica=snr_rapido_mediana_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisRuidoUTAAlerta` | 0,0036 [0,0033; 0,0038] |  | Exp. 6 ruído rápido do EAR (mediana por vídeo), uta|KSS 0 | `results/exp6_noite.csv` | `valor` onde secao=b, grupo=uta&#124;KSS 0, metrica=ruido_rapido_mediana_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisRostoDMDSonolencia` | 99,78 [99,58; 99,95] | % (×100) | Exp. 6 detecção de rosto por quadro (média por vídeo), dmd_s5|todos | `results/exp6_noite.csv` | `valor` onde secao=a, grupo=dmd_s5&#124;todos, metrica=taxa_rosto_media_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisSNRDMDSonolencia` | 23,1 [16,1; 27,9] |  | Exp. 6 SNR rápida do EAR (mediana por vídeo), dmd_s5|todos | `results/exp6_noite.csv` | `valor` onde secao=b, grupo=dmd_s5&#124;todos, metrica=snr_rapido_mediana_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisRuidoDMDSonolencia` | 0,0046 [0,0043; 0,0053] |  | Exp. 6 ruído rápido do EAR (mediana por vídeo), dmd_s5|todos | `results/exp6_noite.csv` | `valor` onde secao=b, grupo=dmd_s5&#124;todos, metrica=ruido_rapido_mediana_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisRostoDMDParado` | 99,89 [99,77; 99,97] | % (×100) | Exp. 6 detecção de rosto por quadro (média por vídeo), dmd_par|safe | `results/exp6_noite.csv` | `valor` onde secao=a, grupo=dmd_par&#124;safe, metrica=taxa_rosto_media_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisSNRDMDParado` | 13,6 [12,0; 16,9] |  | Exp. 6 SNR rápida do EAR (mediana por vídeo), dmd_par|safe | `results/exp6_noite.csv` | `valor` onde secao=b, grupo=dmd_par&#124;safe, metrica=snr_rapido_mediana_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisRuidoDMDParado` | 0,0059 [0,0051; 0,0068] |  | Exp. 6 ruído rápido do EAR (mediana por vídeo), dmd_par|safe | `results/exp6_noite.csv` | `valor` onde secao=b, grupo=dmd_par&#124;safe, metrica=ruido_rapido_mediana_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisRostoDMDMovimento` | 99,65 [99,35; 99,87] | % (×100) | Exp. 6 detecção de rosto por quadro (média por vídeo), dmd_mov|safe | `results/exp6_noite.csv` | `valor` onde secao=a, grupo=dmd_mov&#124;safe, metrica=taxa_rosto_media_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisSNRDMDMovimento` | 8,7 [6,2; 10,8] |  | Exp. 6 SNR rápida do EAR (mediana por vídeo), dmd_mov|safe | `results/exp6_noite.csv` | `valor` onde secao=b, grupo=dmd_mov&#124;safe, metrica=snr_rapido_mediana_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisRuidoDMDMovimento` | 0,0097 [0,0090; 0,0106] |  | Exp. 6 ruído rápido do EAR (mediana por vídeo), dmd_mov|safe | `results/exp6_noite.csv` | `valor` onde secao=b, grupo=dmd_mov&#124;safe, metrica=ruido_rapido_mediana_por_video; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDifRostoNoiteUTA` | −0,61 [−1,07; −0,14] | pp (×100) | Exp. 6 detecção de rosto NITYMED − uta|todos (pp) | `results/exp6_noite.csv` | `valor` onde secao=a, metrica=dif_taxa_rosto_media_nitymed&#124;todos_menos_uta&#124;todos; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDifSNRNoiteUTA` | −14,3 [−16,4; −11,2] |  | Exp. 6 SNR rápida NITYMED − uta|todos (diferença de medianas) | `results/exp6_noite.csv` | `valor` onde secao=b, metrica=dif_mediana_snr_rapido_nitymed&#124;todos_menos_uta&#124;todos; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDifRostoNoiteDMDSonolencia` | −0,63 [−1,08; −0,25] | pp (×100) | Exp. 6 detecção de rosto NITYMED − dmd_s5|todos (pp) | `results/exp6_noite.csv` | `valor` onde secao=a, metrica=dif_taxa_rosto_media_nitymed&#124;todos_menos_dmd_s5&#124;todos; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDifSNRNoiteDMDSonolencia` | −15,9 [−20,7; −9,0] |  | Exp. 6 SNR rápida NITYMED − dmd_s5|todos (diferença de medianas) | `results/exp6_noite.csv` | `valor` onde secao=b, metrica=dif_mediana_snr_rapido_nitymed&#124;todos_menos_dmd_s5&#124;todos; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDifRostoNoiteDMDParado` | −0,73 [−1,13; −0,38] | pp (×100) | Exp. 6 detecção de rosto NITYMED − dmd_par|safe (pp) | `results/exp6_noite.csv` | `valor` onde secao=a, metrica=dif_taxa_rosto_media_nitymed&#124;todos_menos_dmd_par&#124;safe; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDifSNRNoiteDMDParado` | −6,4 [−9,8; −4,5] |  | Exp. 6 SNR rápida NITYMED − dmd_par|safe (diferença de medianas) | `results/exp6_noite.csv` | `valor` onde secao=b, metrica=dif_mediana_snr_rapido_nitymed&#124;todos_menos_dmd_par&#124;safe; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDifRostoNoiteDMDMovimento` | −0,50 [−0,94; −0,05] | pp (×100) | Exp. 6 detecção de rosto NITYMED − dmd_mov|safe (pp) | `results/exp6_noite.csv` | `valor` onde secao=a, metrica=dif_taxa_rosto_media_nitymed&#124;todos_menos_dmd_mov&#124;safe; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDifSNRNoiteDMDMovimento` | −1,6 [−3,7; 1,1] |  | Exp. 6 SNR rápida NITYMED − dmd_mov|safe (diferença de medianas) | `results/exp6_noite.csv` | `valor` onde secao=b, metrica=dif_mediana_snr_rapido_nitymed&#124;todos_menos_dmd_mov&#124;safe; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDifSNRMovimentoParado` | −5,4 [−6,6; −3,5] |  | Exp. 6 DMD mesmos motoristas: SNR rápida em movimento − parado (mediana pareada) | `results/exp6_noite.csv` | `valor` onde secao=b, metrica=dif_pareada_snr_rapido_dmd_mov&#124;safe_menos_dmd_par&#124;safe; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDifRuidoMovimentoParado` | 0,0043 [0,0030; 0,0054] |  | Exp. 6 DMD mesmos motoristas: ruído rápido em movimento − parado | `results/exp6_noite.csv` | `valor` onde secao=b, metrica=dif_pareada_ruido_rapido_dmd_mov&#124;safe_menos_dmd_par&#124;safe; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisSujeitosPareadosMovimento` | 11 |  | Exp. 6 sujeitos na comparação pareada movimento × parado | `results/exp6_noite.csv` | `n_agrupamentos` onde secao=b, metrica=dif_pareada_snr_rapido_dmd_mov&#124;safe_menos_dmd_par&#124;safe |
| `\ExpSeisDisparoModeloDMDMicrossono` | 44,7 [32,7; 57,4] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), Modelo do DMD, Microsleep | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=Microsleep, modelo=Modelo do DMD, variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloDMDBocejo` | 73,0 [65,4; 80,6] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), Modelo do DMD, Yawning | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=Yawning, modelo=Modelo do DMD, variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloDMDUTAAlerta` | 26,0 [21,1; 31,1] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), Modelo do DMD, KSS 0 | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=KSS 0, modelo=Modelo do DMD, variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloDMDUTAIntermediario` | 40,5 [34,7; 46,3] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), Modelo do DMD, KSS 5 | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=KSS 5, modelo=Modelo do DMD, variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloDMDUTASonolento` | 52,2 [46,2; 58,1] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), Modelo do DMD, KSS 10 | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=KSS 10, modelo=Modelo do DMD, variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloDMDDMDParado` | 45,7 [28,6; 66,1] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), Modelo do DMD, alerta_parado | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=alerta_parado, modelo=Modelo do DMD, variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloDMDDMDMovimento` | 42,9 [25,0; 69,2] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), Modelo do DMD, alerta_movimento | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=alerta_movimento, modelo=Modelo do DMD, variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloDMDDMDSonolento` | 82,0 [64,5; 93,2] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), Modelo do DMD, sonolento | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=sonolento, modelo=Modelo do DMD, variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloUTAMicrossono` | 83,5 [73,8; 91,1] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), Modelo da UTA-RLDD (B3), Microsleep | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=Microsleep, modelo=Modelo da UTA-RLDD (B3), variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloUTABocejo` | 89,5 [85,2; 93,8] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), Modelo da UTA-RLDD (B3), Yawning | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=Yawning, modelo=Modelo da UTA-RLDD (B3), variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloUTAUTAAlerta` | 43,6 [38,7; 48,8] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), Modelo da UTA-RLDD (B3), KSS 0 | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=KSS 0, modelo=Modelo da UTA-RLDD (B3), variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloUTAUTAIntermediario` | 58,3 [53,0; 64,0] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), Modelo da UTA-RLDD (B3), KSS 5 | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=KSS 5, modelo=Modelo da UTA-RLDD (B3), variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloUTAUTASonolento` | 69,4 [63,8; 74,6] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), Modelo da UTA-RLDD (B3), KSS 10 | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=KSS 10, modelo=Modelo da UTA-RLDD (B3), variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloUTADMDParado` | 69,1 [53,7; 82,4] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), Modelo da UTA-RLDD (B3), alerta_parado | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=alerta_parado, modelo=Modelo da UTA-RLDD (B3), variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloUTADMDMovimento` | 77,1 [63,2; 88,2] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), Modelo da UTA-RLDD (B3), alerta_movimento | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=alerta_movimento, modelo=Modelo da UTA-RLDD (B3), variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloUTADMDSonolento` | 92,9 [89,4; 96,3] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), Modelo da UTA-RLDD (B3), sonolento | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=sonolento, modelo=Modelo da UTA-RLDD (B3), variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoRegraMicrossono` | 15,0 [5,9; 26,7] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), regra_dmd, Microsleep | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=Microsleep, modelo=regra_dmd, variante=, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoRegraBocejo` | 61,7 [51,3; 72,2] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), regra_dmd, Yawning | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=Yawning, modelo=regra_dmd, variante=, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoRegraUTAAlerta` | 2,6 [1,4; 4,3] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), regra_dmd, KSS 0 | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=KSS 0, modelo=regra_dmd, variante=, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoRegraUTAIntermediario` | 9,7 [6,3; 13,9] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), regra_dmd, KSS 5 | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=KSS 5, modelo=regra_dmd, variante=, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoRegraUTASonolento` | 24,1 [18,0; 30,3] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), regra_dmd, KSS 10 | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=KSS 10, modelo=regra_dmd, variante=, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoRegraDMDParado` | 14,9 [5,7; 25,5] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), regra_dmd, alerta_parado | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=alerta_parado, modelo=regra_dmd, variante=, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoRegraDMDMovimento` | 23,8 [6,8; 50,0] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), regra_dmd, alerta_movimento | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=alerta_movimento, modelo=regra_dmd, variante=, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoRegraDMDSonolento` | 75,7 [61,9; 85,7] | % (×100) | Exp. 6 taxa de disparo (janelas de 10 s marcadas sonolento), regra_dmd, sonolento | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=sonolento, modelo=regra_dmd, variante=, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\AppVersao` | 2.1 |  | versão do app (PWA) descrita e validada | `app/pwa/app.js` | regex `const APP_VERSION = "([\d.]+)"` (grupo 1) |
| `\AppLimiarAmarelo` | 40 |  | app: nível de entrada no amarelo (escolha de projeto) | `app/pwa/README.md` | regex `\&#124; ATENÇÃO \(amarelo\) \&#124; nível >= (\d+) \&#124;` (grupo 1) |
| `\AppLimiarVermelho` | 70 |  | app: nível de entrada no vermelho (escolha de projeto) | `app/pwa/README.md` | regex `\&#124; SONOLENTO \(vermelho\) \&#124; nível >= (\d+),` (grupo 1) |
| `\AppVolumePadrao` | 20 | % | app: volume padrão do alarme | `app/pwa/README.md` | regex `\*\*Padrão: (\d+) %\*\*` (grupo 1) |
| `\AppUTAAlertaVideos` | 3 |  | app no navegador: vídeos UTA-RLDD alerta (vídeo 0) testados | `results/app_validacao.csv` | `(linhas)` [count] onde versao_app=2.1 (ajuste final), metrica=pct_tempo_alerta, base=UTA-RLDD 0 (alerta) |
| `\AppUTAAlertaVerdeMin` | 100,0 | % | app no navegador: menor % do tempo em verde nos vídeos alerta da UTA | `results/app_validacao.csv` | `valor` [min] onde versao_app=2.1 (ajuste final), video={uta_F1_09_v0,uta_F3_31_v0,uta_F5_51_v0}, metrica=pct_tempo_alerta |
| `\AppUTAAlertaAlarmes` | 0 |  | app no navegador: alarmes nos vídeos alerta da UTA (soma) | `results/app_validacao.csv` | `valor` [sum] onde versao_app=2.1 (ajuste final), video={uta_F1_09_v0,uta_F3_31_v0,uta_F5_51_v0}, metrica=n_alarme |
| `\AppUTAAlertaAvisos` | 0 |  | app no navegador: avisos (amarelo) nos vídeos alerta da UTA (soma) | `results/app_validacao.csv` | `valor` [sum] onde versao_app=2.1 (ajuste final), video={uta_F1_09_v0,uta_F3_31_v0,uta_F5_51_v0}, metrica=n_aviso |
| `\AppUTASonolentoVideos` | 3 |  | app no navegador: vídeos UTA-RLDD sonolentos (vídeo 10) testados | `results/app_validacao.csv` | `(linhas)` [count] onde versao_app=2.1 (ajuste final), video={uta_F1_09_v10,uta_F3_31_v10,uta_F5_51_v10}, metrica=n_alarme |
| `\AppUTASonolentoComVermelho` | 1 |  | app no navegador: vídeos UTA sonolentos que chegaram ao vermelho (n_alarme = 1) | `results/app_validacao.csv` | `(linhas)` [count] onde versao_app=2.1 (ajuste final), video={uta_F1_09_v10,uta_F3_31_v10,uta_F5_51_v10}, metrica=n_alarme, valor=1 |
| `\AppKappaNavegador` | 0,902 |  | app no navegador (4 sessões DMD s5): kappa do olho fechado por quadro (linha Total da v2.1) | `results/app_validacao.md` | regex `\&#124; \*\*Total\*\* \&#124; \&#124; \&#124; ([\d,]+) / ([\d,]+) / ([\d,]+) \&#124;` (grupo 3) |
| `\AppBocejosNavegadorAcertos` | 6 |  | app no navegador (4 sessões DMD s5): bocejos_vp_anotados (soma) | `results/app_validacao.csv` | `valor` [sum] onde versao_app=2.1 (ajuste final), base=DMD s5 (encenada), metrica=bocejos_vp_anotados |
| `\AppBocejosNavegadorAnotados` | 24 |  | app no navegador (4 sessões DMD s5): bocejos_anotados (soma) | `results/app_validacao.csv` | `valor` [sum] onde versao_app=2.1 (ajuste final), base=DMD s5 (encenada), metrica=bocejos_anotados |
| `\AppBocejosNavegadorAnotadosSemMao` | 14 |  | app no navegador (4 sessões DMD s5): bocejos_anotados_sem_mao (soma) | `results/app_validacao.csv` | `valor` [sum] onde versao_app=2.1 (ajuste final), base=DMD s5 (encenada), metrica=bocejos_anotados_sem_mao |
| `\AppDMDMinutosDirecaoReal` | 63,3 |  | motor do app offline no DMD: minutos ativos de direção real (s1) | `results/app_validacao_dmd_ajuste_final.csv` | `s1_min` onde alternativa=final |
| `\AppDMDVermelhoDirecaoReal` | 3,3 | % | motor do app offline no DMD: % do tempo em vermelho na direção real (s1) | `results/app_validacao_dmd_ajuste_final.csv` | `s1_pct_vermelho` onde alternativa=final |
| `\AppDMDAmareloDirecaoReal` | 4,1 | % | motor do app offline no DMD: % do tempo em amarelo na direção real (s1) | `results/app_validacao_dmd_ajuste_final.csv` | `s1_pct_amarelo` onde alternativa=final |
| `\AppDMDAlarmesDirecaoReal` | 4 |  | motor do app offline no DMD: alarmes (entradas no vermelho) na direção real | `results/app_validacao_dmd_ajuste_final.csv` | `s1_alarmes` onde alternativa=final |
| `\AppDMDAlarmesHoraDirecaoReal` | 3,8 |  | motor do app offline no DMD: alarmes por hora na direção real | `results/app_validacao_dmd_ajuste_final.csv` | `s1_alarmes_h` onde alternativa=final |
| `\AppDMDBocejosFalsosDirecaoReal` | 0 |  | motor do app offline no DMD: bocejos detectados na direção real (falsos) | `results/app_validacao_dmd_ajuste_final.csv` | `s1_bocejos` onde alternativa=final |
| `\AppDMDMinutosSonolencia` | 46,6 |  | motor do app offline no DMD: minutos ativos na sonolência encenada (s5) | `results/app_validacao_dmd_ajuste_final.csv` | `s5_min` onde alternativa=final |
| `\AppDMDVermelhoSonolencia` | 68,6 | % | motor do app offline no DMD: % do tempo em vermelho na sonolência encenada (s5) | `results/app_validacao_dmd_ajuste_final.csv` | `s5_pct_vermelho` onde alternativa=final |
| `\AppKappaOffline` | 0,938 |  | motor do app offline no DMD: kappa do olho fechado por quadro (média das 16 sessões s5) | `results/app_validacao_dmd_ajuste_final.csv` | `s5_kappa` onde alternativa=final |
| `\AppFechUmSRevocacao` | 0,920 |  | motor do app offline no DMD: revocação de fechamentos >= 1 s (s5) | `results/app_validacao_dmd_ajuste_final.csv` | `s5_revoc_fech1` onde alternativa=final |
| `\AppFechUmSPrecisao` | 0,748 |  | motor do app offline no DMD: precisão de fechamentos >= 1 s (s5) | `results/app_validacao_dmd_ajuste_final.csv` | `s5_prec_fech1` onde alternativa=final |
| `\AppFechMeioSRevocacao` | 0,929 |  | motor do app offline no DMD: revocação de fechamentos >= 0,5 s (s5) | `results/app_validacao_dmd_ajuste_final.csv` | `s5_revoc_fech05` onde alternativa=final |
| `\AppFechMeioSPrecisao` | 0,786 |  | motor do app offline no DMD: precisão de fechamentos >= 0,5 s (s5) | `results/app_validacao_dmd_ajuste_final.csv` | `s5_prec_fech05` onde alternativa=final |
| `\AppDMDVermelhoVUmDirecaoReal` | 55,6 | % | lógica da v1 (EAR + PERCLOS > 15% + olho fechado >= 1 s) no mesmo DMD s1: % do tempo em vermelho | `results/app_validacao_dmd_variantes_olhos.csv` | `pct_vermelho_s1` onde variante=v1_app_anterior (EAR + PERCLOS>15% + olho fechado>=1 s; sem LSTM) |
| `\AppDMDSessoesDirecaoReal` | 17 |  | motor offline: sessões de direção real (s1) | `results/app_validacao_dmd_sessoes.csv` | `(linhas)` [count] onde sessao=s1 |
| `\AppDMDSessoesSemAlarme` | 16 |  | motor offline: sessões s1 sem nenhum alarme | `results/app_validacao_dmd_sessoes.csv` | `(linhas)` [count] onde sessao=s1, alarmes=0 |
| `\AppGACincoVermelho` | 55,8 | % (×100) | motor offline: % do tempo em vermelho do motorista gA_5 na direção real (falso alarme conhecido) | `results/app_validacao_dmd_sessoes.csv` | `frac_sonolento` onde sessao=s1, sujeito=gA_5 |
| `\AppBocejoAcertosSemMao` | 30 |  | motor offline (16 sessões s5): bocejos sem a mão detectados | `results/app_validacao_dmd_bocejo.csv` | `tp_sem_mao` [sum] onde detector=app (jawOpen>=0,4 e >= repouso+0,3, ou MAR>=repouso+0,4; por 1,5 s) |
| `\AppBocejoAnotadosSemMao` | 47 |  | motor offline (16 sessões s5): bocejos anotados sem a mão | `results/app_validacao_dmd_bocejo.csv` | `n_sem_mao` [sum] onde detector=app (jawOpen>=0,4 e >= repouso+0,3, ou MAR>=repouso+0,4; por 1,5 s) |
| `\AppBocejoAcertosComMao` | 1 |  | motor offline (16 sessões s5): bocejos com a mão detectados | `results/app_validacao_dmd_bocejo.csv` | `tp_com_mao` [sum] onde detector=app (jawOpen>=0,4 e >= repouso+0,3, ou MAR>=repouso+0,4; por 1,5 s) |
| `\AppBocejoAnotadosComMao` | 38 |  | motor offline (16 sessões s5): bocejos anotados com a mão | `results/app_validacao_dmd_bocejo.csv` | `n_com_mao` [sum] onde detector=app (jawOpen>=0,4 e >= repouso+0,3, ou MAR>=repouso+0,4; por 1,5 s) |
| `\AppBocejoFalsos` | 0 |  | motor offline (16 sessões s5): bocejos detectados sem anotação (s5) | `results/app_validacao_dmd_bocejo.csv` | `fp` [sum] onde detector=app (jawOpen>=0,4 e >= repouso+0,3, ou MAR>=repouso+0,4; por 1,5 s) |
| `\AppBocejoRevocacaoSemMao` | 0,638 |  | motor offline: revocação de bocejos sem a mão na boca | `(derivado)` | `AppBocejoAcertosSemMao / AppBocejoAnotadosSemMao` com AppBocejoAcertosSemMao, AppBocejoAnotadosSemMao |
| `\AppBocejoRevocacaoComMao` | 0,026 |  | motor offline: revocação de bocejos com a mão na boca | `(derivado)` | `AppBocejoAcertosComMao / AppBocejoAnotadosComMao` com AppBocejoAcertosComMao, AppBocejoAnotadosComMao |
| `\SinaisDetectorFUmPiscadas` | 0,749 |  | detector de piscadas (limiar final) x anotação do DMD s5: F1 de piscadas (média das 16 sessões) | `results/exp5_dmd_validacao_detector.csv` | `f1_piscadas` [mean] onde sessao=s5 |
| `\SinaisDetectorPrecisaoPiscadas` | 0,811 |  | detector de piscadas (limiar final) x anotação do DMD s5: precisão de piscadas (média das 16 sessões) | `results/exp5_dmd_validacao_detector.csv` | `precisao_piscadas` [mean] onde sessao=s5 |
| `\SinaisDetectorRevocacaoPiscadas` | 0,775 |  | detector de piscadas (limiar final) x anotação do DMD s5: revocação de piscadas (média das 16 sessões) | `results/exp5_dmd_validacao_detector.csv` | `revocacao_piscadas` [mean] onde sessao=s5 |
| `\SinaisDetectorAcBalQuadro` | 0,900 |  | detector de piscadas (limiar final) x anotação do DMD s5: acurácia balanceada olho fechado por quadro (média das 16 sessões) | `results/exp5_dmd_validacao_detector.csv` | `q_acuracia_bal` [mean] onde sessao=s5 |
| `\SinaisDetectorKappaQuadro` | 0,773 |  | detector de piscadas (limiar final) x anotação do DMD s5: kappa olho fechado por quadro (média das 16 sessões) | `results/exp5_dmd_validacao_detector.csv` | `q_kappa` [mean] onde sessao=s5 |
| `\SinaisDetectorSessoes` | 16 |  | detector de piscadas: sessões s5 do DMD avaliadas | `results/exp5_dmd_validacao_detector.csv` | `(linhas)` [count] onde sessao=s5 |
| `\SinaisSafeDriveMin` | 10 | s | DMD: duração mínima dos trechos safe_drive (s) | `results/exp5_caracteristicas.md` | regex `os trechos safe_drive do DMD têm (\d+)–(\d+) s` (grupo 1) |
| `\SinaisSafeDriveMax` | 26 | s | DMD: duração máxima dos trechos safe_drive (s; nenhum cabe numa janela de 30 s) | `results/exp5_caracteristicas.md` | regex `os trechos safe_drive do DMD têm (\d+)–(\d+) s` (grupo 2) |
| `\BaseDMDHoras` | 4,24 | h | DMD: horas de vídeo com sinais extraídos (todas as sessões usadas) | `results/exp5_caracteristicas.md` | regex `\&#124; DMD \&#124; \*\*total\*\* \&#124; (\d+) \&#124; (\d+) \&#124; (\d+) \&#124; ([\d.]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,–]+) \&#124;` (grupo 5) |
| `\BaseDMDFPSFaixa` | 29,76 |  | DMD: quadros por segundo do contêiner (valor ou faixa) | `results/exp5_caracteristicas.md` | regex `\&#124; DMD \&#124; \*\*total\*\* \&#124; (\d+) \&#124; (\d+) \&#124; (\d+) \&#124; ([\d.]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,–]+) \&#124;` (grupo 8) |
| `\BaseNITYMEDHoras` | 1,63 | h | NITYMED: horas de vídeo com sinais extraídos (todas as sessões usadas) | `results/exp5_caracteristicas.md` | regex `\&#124; NITYMED \&#124; \*\*total\*\* \&#124; (\d+) \&#124; (\d+) \&#124; (\d+) \&#124; ([\d.]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,–]+) \&#124;` (grupo 5) |
| `\BaseNITYMEDFPSFaixa` | 24,00–25,00 |  | NITYMED: quadros por segundo do contêiner (valor ou faixa) | `results/exp5_caracteristicas.md` | regex `\&#124; NITYMED \&#124; \*\*total\*\* \&#124; (\d+) \&#124; (\d+) \&#124; (\d+) \&#124; ([\d.]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,–]+) \&#124;` (grupo 8) |
| `\BaseUTAHoras` | 31,55 | h | UTA-RLDD: horas de vídeo com sinais extraídos (todas as sessões usadas) | `results/exp5_caracteristicas.md` | regex `\&#124; UTA\-RLDD \&#124; \*\*total\*\* \&#124; (\d+) \&#124; (\d+) \&#124; (\d+) \&#124; ([\d.]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,–]+) \&#124;` (grupo 5) |
| `\BaseUTAFPSFaixa` | 11,99–30,17 |  | UTA-RLDD: quadros por segundo do contêiner (valor ou faixa) | `results/exp5_caracteristicas.md` | regex `\&#124; UTA\-RLDD \&#124; \*\*total\*\* \&#124; (\d+) \&#124; (\d+) \&#124; (\d+) \&#124; ([\d.]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,–]+) \&#124;` (grupo 8) |
| `\ExpTresIntraMRLIlumRuimAcBal` | 94,3 | % (×100) | Exp. 3 A intra-MRL iluminação ruim (out-of-fold agregado), acurácia balanceada | `results/exp3_strat_conditions.csv` | `value` onde protocol=A_intra_mrl, fold=oof, stratum=lighting, level=iluminação ruim, metric=bal_acc |
| `\ExpTresCEWPatchMRLIlumRuimAcBal` | 63,8 ± 3,1 | % (×100) | Exp. 3 C CEW patch→MRL iluminação ruim, bal_acc (média ± dp, ddof=1) | `results/exp3_strat_conditions.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, fold={1,2,3,4,5}, stratum=lighting, level=iluminação ruim, metric=bal_acc; dp std1 |
| `\ExpTresCEWPatchMRLIlumRuimAUC` | 81,1 ± 3,2 | % (×100) | Exp. 3 C CEW patch→MRL iluminação ruim, auc (média ± dp, ddof=1) | `results/exp3_strat_conditions.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, fold={1,2,3,4,5}, stratum=lighting, level=iluminação ruim, metric=auc; dp std1 |
| `\ExpTresIntraMRLIlumBoaAcBal` | 92,6 | % (×100) | Exp. 3 A intra-MRL iluminação boa (out-of-fold agregado), acurácia balanceada | `results/exp3_strat_conditions.csv` | `value` onde protocol=A_intra_mrl, fold=oof, stratum=lighting, level=iluminação boa, metric=bal_acc |
| `\ExpTresCEWPatchMRLIlumBoaAcBal` | 75,1 ± 2,5 | % (×100) | Exp. 3 C CEW patch→MRL iluminação boa, bal_acc (média ± dp, ddof=1) | `results/exp3_strat_conditions.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, fold={1,2,3,4,5}, stratum=lighting, level=iluminação boa, metric=bal_acc; dp std1 |
| `\ExpTresCEWPatchMRLIlumBoaAUC` | 89,6 ± 1,6 | % (×100) | Exp. 3 C CEW patch→MRL iluminação boa, auc (média ± dp, ddof=1) | `results/exp3_strat_conditions.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, fold={1,2,3,4,5}, stratum=lighting, level=iluminação boa, metric=auc; dp std1 |
| `\ExpTresIntraMRLReflexoGrandeAcBal` | 84,2 | % (×100) | Exp. 3 A intra-MRL reflexo grande (out-of-fold agregado), acurácia balanceada | `results/exp3_strat_conditions.csv` | `value` onde protocol=A_intra_mrl, fold=oof, stratum=reflections, level=reflexo grande, metric=bal_acc |
| `\ExpTresCEWPatchMRLReflexoGrandeAcBal` | 65,7 ± 3,6 | % (×100) | Exp. 3 C CEW patch→MRL reflexo grande, bal_acc (média ± dp, ddof=1) | `results/exp3_strat_conditions.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, fold={1,2,3,4,5}, stratum=reflections, level=reflexo grande, metric=bal_acc; dp std1 |
| `\ExpTresCEWPatchMRLReflexoGrandeAUC` | 69,2 ± 3,5 | % (×100) | Exp. 3 C CEW patch→MRL reflexo grande, auc (média ± dp, ddof=1) | `results/exp3_strat_conditions.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, fold={1,2,3,4,5}, stratum=reflections, level=reflexo grande, metric=auc; dp std1 |
| `\ExpTresIntraMRLSemReflexoAcBal` | 95,5 | % (×100) | Exp. 3 A intra-MRL sem reflexo (out-of-fold agregado), acurácia balanceada | `results/exp3_strat_conditions.csv` | `value` onde protocol=A_intra_mrl, fold=oof, stratum=reflections, level=sem reflexo, metric=bal_acc |
| `\ExpTresCEWPatchMRLSemReflexoAcBal` | 64,8 ± 3,4 | % (×100) | Exp. 3 C CEW patch→MRL sem reflexo, bal_acc (média ± dp, ddof=1) | `results/exp3_strat_conditions.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, fold={1,2,3,4,5}, stratum=reflections, level=sem reflexo, metric=bal_acc; dp std1 |
| `\ExpTresCEWPatchMRLSemReflexoAUC` | 86,9 ± 3,5 | % (×100) | Exp. 3 C CEW patch→MRL sem reflexo, auc (média ± dp, ddof=1) | `results/exp3_strat_conditions.csv` | `value` [mean] onde protocol=C_cew_patch_to_mrl, fold={1,2,3,4,5}, stratum=reflections, level=sem reflexo, metric=auc; dp std1 |
| `\ExpCincoBSemNormAUCJanelaB` | 69,4 [64,8; 73,7] | % (×100) | Exp. 5b RF: (B) UTA→UTA sem normalização, AUC por janela | `results/exp5b_transferencia.csv` | `valor` onde experimento=B_uta_intra, variante=sem_normalizacao, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBSemNormAUCJanelaC` | 67,2 [62,5; 71,6] | % (×100) | Exp. 5b RF: (C) DMD→UTA sem normalização, AUC por janela | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=sem_normalizacao, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBSemNormQuedaAUCJanela` | 2,1 [−1,2; 5,8] | pp | Exp. 5b RF: queda B − C sem normalização, AUC por janela (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=sem_normalizacao, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBAlertaSUmAUCJanelaC` | 58,6 [53,5; 63,3] | % (×100) | Exp. 5b RF: (C) DMD→UTA com o alerta do DMD tirado da s1 (carro em movimento), AUC por janela | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=alerta_s1, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBControleAUCCincoDezB` | 61,5 [57,1; 65,8] | % (×100) | Exp. 5b RF: (B) AUC KSS 10 × KSS 5 (controle) | `results/exp5b_transferencia.csv` | `valor` onde experimento=B_uta_intra, variante=principal, modelo=RF, esquema=K10v5, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBControleAUCCincoDezC` | 60,2 [55,6; 64,8] | % (×100) | Exp. 5b RF: (C) AUC KSS 10 × KSS 5 (controle) | `results/exp5b_transferencia.csv` | `valor` onde experimento=C_dmd_para_uta, variante=principal, modelo=RF, esquema=K10v5, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBControleQuedaCincoDez` | 1,2 [−3,0; 5,8] | pp | Exp. 5b RF: queda B − C no contraste KSS 10 × 5 (pp de AUC) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=principal, modelo=RF, esquema=K10v5, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBControleQuedaZeroCinco` | 11,5 [6,7; 16,2] | pp | Exp. 5b RF: queda B − C no contraste KSS 5 × 0 (pp de AUC) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=principal, modelo=RF, esquema=K5v0, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBMedianaFechMaxSonolentoDMD` | 0,97 |  | Exp. 5b: mediana de dur_max_fechamento_s (bruto) nas janelas sonolentas, DMD | `results/exp5b_importancia.csv` | `mediana_sonolento` onde tipo=auc_univariada, experimento=dmd, variante=sem_normalizacao, esquema=DMD, caracteristica=dur_max_fechamento_s |
| `\ExpCincoBMedianaPERCLOSSonolentoDMD` | 0,208 |  | Exp. 5b: mediana de perclos (bruto) nas janelas sonolentas, DMD | `results/exp5b_importancia.csv` | `mediana_sonolento` onde tipo=auc_univariada, experimento=dmd, variante=sem_normalizacao, esquema=DMD, caracteristica=perclos |
| `\ExpCincoBAUCUnivBocejoDMD` | 57,0 | % (×100) | Exp. 5b: AUC univariada da fração de bocejo, DMD | `results/exp5b_importancia.csv` | `valor` onde tipo=auc_univariada, experimento=dmd, variante=principal, esquema=DMD, caracteristica=frac_bocejo |
| `\ExpCincoBMedianaFechMaxSonolentoUTA` | 0,17 |  | Exp. 5b: mediana de dur_max_fechamento_s (bruto) nas janelas sonolentas, UTA | `results/exp5b_importancia.csv` | `mediana_sonolento` onde tipo=auc_univariada, experimento=uta, variante=sem_normalizacao, esquema=B3, caracteristica=dur_max_fechamento_s |
| `\ExpCincoBMedianaPERCLOSSonolentoUTA` | 0,040 |  | Exp. 5b: mediana de perclos (bruto) nas janelas sonolentas, UTA | `results/exp5b_importancia.csv` | `mediana_sonolento` onde tipo=auc_univariada, experimento=uta, variante=sem_normalizacao, esquema=B3, caracteristica=perclos |
| `\ExpCincoBAUCUnivBocejoUTA` | 50,2 | % (×100) | Exp. 5b: AUC univariada da fração de bocejo, UTA | `results/exp5b_importancia.csv` | `valor` onde tipo=auc_univariada, experimento=uta, variante=principal, esquema=B3, caracteristica=frac_bocejo |
| `\ExpCincoBMotoristasOculos` | 3 |  | DMD s5: sessões (= motoristas) com óculos | `results/exp5b_oculos_ear_dmd.csv` | `(linhas)` [count] onde oculos=True |
| `\ExpCincoBOculosFalhaFUm` | 0,108 |  | DMD s5, motorista de óculos gE_29: F1 de piscadas do detector | `results/exp5b_oculos_ear_dmd.csv` | `det_f1_piscadas` onde sujeito=gE_29 |
| `\ExpCincoBOculosFalhaQuedaEAR` | 14,2 | % (×100) | DMD s5, gE_29: queda relativa do EAR mediano de aberto para fechado | `results/exp5b_oculos_ear_dmd.csv` | `queda_relativa` onde sujeito=gE_29 |
| `\ExpCincoBOculosFalhaAUCEAR` | 92,8 | % (×100) | DMD s5, gE_29: AUC de −EAR para separar quadros fechados de abertos | `results/exp5b_oculos_ear_dmd.csv` | `auc_menos_ear` onde sujeito=gE_29 |
| `\ExpCincoBSemOculosQuedaEARMin` | 30,4 | % (×100) | DMD s5, motoristas sem óculos: queda relativa do EAR (min entre sessões) | `results/exp5b_oculos_ear_dmd.csv` | `queda_relativa` [min] onde oculos=False |
| `\ExpCincoBSemOculosQuedaEARMax` | 55,6 | % (×100) | DMD s5, motoristas sem óculos: queda relativa do EAR (max entre sessões) | `results/exp5b_oculos_ear_dmd.csv` | `queda_relativa` [max] onde oculos=False |
| `\ExpCincoBQuedaAUCIntraSujeito` | 11,2 [6,8; 15,7] | pp | Exp. 5b RF: queda B − C na AUC média intra-sujeito (pp de AUC) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=principal, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc_media_intra_sujeito; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisPERCLOSMicrossono` | 2,9 | % (×100) | Exp. 6 PERCLOS mediano por vídeo, nitymed|Microsleep, limiar do pipeline (referência) | `results/exp6_noite.csv` | `valor` onde secao=d, grupo=nitymed&#124;Microsleep, metrica=perclos_mediana_por_video |
| `\ExpSeisPERCLOSMicrossonoPNoventa` | 19,5 | % (×100) | Exp. 6 PERCLOS mediano por vídeo, nitymed|Microsleep, limiar pelo p90 do EAR do próprio vídeo | `results/exp6_noite.csv` | `valor` onde secao=d, grupo=nitymed&#124;Microsleep, metrica=perclos_p90_mediana_por_video |
| `\ExpSeisPERCLOSUTAAlerta` | 1,4 | % (×100) | Exp. 6 PERCLOS mediano por vídeo, uta|KSS 0, limiar do pipeline (referência) | `results/exp6_noite.csv` | `valor` onde secao=d, grupo=uta&#124;KSS 0, metrica=perclos_mediana_por_video |
| `\ExpSeisPERCLOSUTAAlertaPNoventa` | 2,4 | % (×100) | Exp. 6 PERCLOS mediano por vídeo, uta|KSS 0, limiar pelo p90 do EAR do próprio vídeo | `results/exp6_noite.csv` | `valor` onde secao=d, grupo=uta&#124;KSS 0, metrica=perclos_p90_mediana_por_video |
| `\ExpSeisPERCLOSDMDMovimento` | 5,4 | % (×100) | Exp. 6 PERCLOS mediano por vídeo, dmd_mov|safe, limiar do pipeline (referência) | `results/exp6_noite.csv` | `valor` onde secao=d, grupo=dmd_mov&#124;safe, metrica=perclos_mediana_por_video |
| `\ExpSeisPERCLOSDMDMovimentoPNoventa` | 18,4 | % (×100) | Exp. 6 PERCLOS mediano por vídeo, dmd_mov|safe, limiar pelo p90 do EAR do próprio vídeo | `results/exp6_noite.csv` | `valor` onde secao=d, grupo=dmd_mov&#124;safe, metrica=perclos_p90_mediana_por_video |
| `\MacPWALandmarkerPcinquenta` | 9,6 | ms | PWA v1.0 no Chrome headless do Mac (20/09, vídeo 320x240): landmarker p50 | `results/pwa_headless_mac_bench_20260920-190012.json` | chave `latencia_ms.landmarker.p50` |
| `\MacPWALandmarkerPnoventaecinco` | 11,2 | ms | PWA v1.0 no Chrome headless do Mac (20/09, vídeo 320x240): landmarker p95 | `results/pwa_headless_mac_bench_20260920-190012.json` | chave `latencia_ms.landmarker.p95` |
| `\MacPWACNNPcinquenta` | 3,0 | ms | PWA v1.0 no Chrome headless do Mac (20/09, vídeo 320x240): cnn_olhos_2olhos p50 | `results/pwa_headless_mac_bench_20260920-190012.json` | chave `latencia_ms.cnn_olhos_2olhos.p50` |
| `\MacPWACNNPnoventaecinco` | 3,8 | ms | PWA v1.0 no Chrome headless do Mac (20/09, vídeo 320x240): cnn_olhos_2olhos p95 | `results/pwa_headless_mac_bench_20260920-190012.json` | chave `latencia_ms.cnn_olhos_2olhos.p95` |
| `\MacPWALSTMPcinquenta` | 0,2 | ms | PWA v1.0 no Chrome headless do Mac (20/09, vídeo 320x240): temporal p50 | `results/pwa_headless_mac_bench_20260920-190012.json` | chave `latencia_ms.temporal.p50` |
| `\MacPWALSTMPnoventaecinco` | 0,3 | ms | PWA v1.0 no Chrome headless do Mac (20/09, vídeo 320x240): temporal p95 | `results/pwa_headless_mac_bench_20260920-190012.json` | chave `latencia_ms.temporal.p95` |
| `\AppKappaEARSozinho` | 0,831 |  | motor offline: kappa por quadro com o EAR sozinho (regra da v1) | `results/app_validacao_dmd_variantes_olhos.csv` | `kappa_s5` onde variante=ear_v1 |
| `\AppDMDVermelhoEARSozinho` | 26,1 | % | motor offline: % do tempo em vermelho na direção real com o EAR sozinho nos olhos | `results/app_validacao_dmd_variantes_olhos.csv` | `pct_vermelho_s1` onde variante=ear_v1 |
| `\AppNavegadorSessoesDMD` | 4 |  | app no navegador: sessões do DMD s5 testadas | `results/app_validacao.csv` | `(linhas)` [count] onde versao_app=2.1 (ajuste final), base=DMD s5 (encenada), metrica=n_alarme |
| `\AppNavegadorVideosNITYMED` | 3 |  | app no navegador: vídeos do NITYMED testados | `results/app_validacao.csv` | `(linhas)` [count] onde versao_app=2.1 (ajuste final), base={NITYMED bocejo (noite),NITYMED microssono (noite)}, metrica=n_alarme |
| `\ExpSeisDisparoModeloDMDCalUTAAlerta` | 1,7 [0,8; 2,8] | % (×100) | Exp. 6 taxa de disparo do RF do DMD (brutas) com o limiar calibrado só no treino (LOSO do DMD), KSS 0 | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=KSS 0, modelo=Modelo do DMD, variante=sem_normalizacao, metrica=taxa_disparo_janela_limiar_calibrado; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloDMDCalUTASonolento` | 11,0 [7,9; 15,1] | % (×100) | Exp. 6 taxa de disparo do RF do DMD (brutas) com o limiar calibrado só no treino (LOSO do DMD), KSS 10 | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=KSS 10, modelo=Modelo do DMD, variante=sem_normalizacao, metrica=taxa_disparo_janela_limiar_calibrado; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloDMDCalDMDMovimento` | 21,9 [9,3; 40,7] | % (×100) | Exp. 6 taxa de disparo do RF do DMD (brutas) com o limiar calibrado só no treino (LOSO do DMD), alerta_movimento | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=alerta_movimento, modelo=Modelo do DMD, variante=sem_normalizacao, metrica=taxa_disparo_janela_limiar_calibrado; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloDMDLRUTAAlerta` | 2,3 [1,1; 3,9] | % (×100) | Exp. 6 taxa de disparo da RL do DMD (brutas, limiar 0,5; modelo final do Exp. 5b), KSS 0 | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=KSS 0, modelo=Exp. 5b: exp5b_final_C_sem_normalizacao_LR.joblib, variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpSeisDisparoModeloDMDLRUTASonolento` | 18,5 [12,9; 24,9] | % (×100) | Exp. 6 taxa de disparo da RL do DMD (brutas, limiar 0,5; modelo final do Exp. 5b), KSS 10 | `results/exp6_noite.csv` | `valor` onde secao=c, grupo=KSS 10, modelo=Exp. 5b: exp5b_final_C_sem_normalizacao_LR.joblib, variante=sem_normalizacao, metrica=taxa_disparo_janela; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRSemNormQuedaAUCJanela` | 7,3 [3,0; 11,7] | pp | Exp. 5b RL: queda B − C sem normalização por sujeito, AUC por janela (pp de AUC) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=sem_normalizacao, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBSensVariantesICPositivo` | 7 |  | Exp. 5b RF: variantes (incluindo a principal) em que o IC da queda B − C de AUC fica acima de 0 | `results/exp5b_transferencia.md` | regex `RF: AUC de C entre [\d,]+ \(`alerta_s1`\) e [\d,]+ \(`ref_s2_espelho`\); queda B − C com IC acima de 0 em (\d+) de (\d+) variantes` (grupo 1) |
| `\ExpCincoBSensVariantesTotal` | 8 |  | Exp. 5b RF: total de variantes analisadas (principal + sensibilidades) | `results/exp5b_transferencia.md` | regex `RF: AUC de C entre [\d,]+ \(`alerta_s1`\) e [\d,]+ \(`ref_s2_espelho`\); queda B − C com IC acima de 0 em (\d+) de (\d+) variantes` (grupo 2) |
| `\ExpCincoBSujeitosTreinoB` | 48 |  | Exp. 5b: sujeitos da UTA-RLDD no treino de B a cada fold (4 dos 5 folds oficiais de 12) | `(derivado)` | `BaseUTASujeitos*4/5` com BaseUTASujeitos |
| `\ExpCincoBRefAltAUCJanelaB` | 68,4 [63,5; 72,8] |  | Exp. 5b robustez: RF, (B) UTA→UTA com a centragem pelo vídeo KSS 5, AUC por janela | `results/exp5b_robustez.csv` | `valor` onde analise=referencia_kss5, variante=ref_video_kss5, modelo=RF, nivel=janela, metrica=auc_B; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBRefAltAUCJanelaC` | 66,1 [61,4; 70,9] |  | Exp. 5b robustez: RF, (C) DMD→UTA com a centragem pelo vídeo KSS 5, AUC por janela | `results/exp5b_robustez.csv` | `valor` onde analise=referencia_kss5, variante=ref_video_kss5, modelo=RF, nivel=janela, metrica=auc_C; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBRefAltQuedaAUCJanela` | 2,3 [−2,5; 6,8] | pp | Exp. 5b robustez: RF, queda B − C com a centragem pelo vídeo KSS 5, AUC por janela (pp) | `results/exp5b_robustez.csv` | `valor` onde analise=referencia_kss5, variante=ref_video_kss5, modelo=RF, nivel=janela, metrica=queda_auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBRefAltQuedaAUCVideo` | −2,3 [−9,1; 4,3] | pp | Exp. 5b robustez: RF, queda B − C com a centragem pelo vídeo KSS 5, AUC por vídeo (pp) | `results/exp5b_robustez.csv` | `valor` onde analise=referencia_kss5, variante=ref_video_kss5, modelo=RF, nivel=video, metrica=queda_auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBBootDuploQuedaAUCJanela` | 10,9 [5,7; 19,5] | pp | Exp. 5b robustez: RF, queda B − C por janela com IC em dois níveis (retreino de C) | `results/exp5b_robustez.csv` | `valor` onde analise=bootstrap_dois_niveis, variante=ref_video_alerta, modelo=RF, nivel=janela, metrica=queda_auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBTrezeAUCJanelaB` | 78,6 [77,0; 80,3] |  | Exp. 5b robustez: RF, B treinado com 13 sujeitos por fold (mediana dos sorteios; faixa = percentis 2,5 e 97,5) | `results/exp5b_robustez.csv` | `valor` onde analise=volume_treino, variante=13_sujeitos, modelo=RF, nivel=janela, metrica=auc_B; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBTrezeQuinhentasAUCJanelaB` | 78,8 [76,6; 80,5] |  | Exp. 5b robustez: RF, B treinado com 13 sujeitos e 510 janelas por fold (mediana dos sorteios) | `results/exp5b_robustez.csv` | `valor` onde analise=volume_treino, variante=13_sujeitos_510_janelas, modelo=RF, nivel=janela, metrica=auc_B; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBVazamentoAcBal` | 12,6 | pp | Exp. 5b robustez: RF, inflação pela partição aleatória de janelas no cenário do Exp. 5b (acuracia_balanceada, pp) | `results/exp5b_robustez.csv` | `valor` onde analise=vazamento_mesmo_cenario, variante=centrada, modelo=RF, nivel=janela, metrica=acuracia_balanceada_inflacao |
| `\ExpCincoBVazamentoAcBalSujeito` | 74,2 | % | Exp. 5b robustez: RF, acuracia_balanceada com os folds oficiais por sujeito (cenário do Exp. 5b) | `results/exp5b_robustez.csv` | `valor` onde analise=vazamento_mesmo_cenario, variante=centrada, modelo=RF, nivel=janela, metrica=acuracia_balanceada_por_sujeito |
| `\ExpCincoBVazamentoAcBalAleatoria` | 86,8 | % | Exp. 5b robustez: RF, acuracia_balanceada com a partição aleatória de janelas (média de 3 sementes) | `results/exp5b_robustez.csv` | `valor` onde analise=vazamento_mesmo_cenario, variante=centrada, modelo=RF, nivel=janela, metrica=acuracia_balanceada_aleatoria |
| `\ExpCincoBVazamentoAUC` | 13,9 | pp | Exp. 5b robustez: RF, inflação pela partição aleatória de janelas no cenário do Exp. 5b (auc, pp) | `results/exp5b_robustez.csv` | `valor` onde analise=vazamento_mesmo_cenario, variante=centrada, modelo=RF, nivel=janela, metrica=auc_inflacao |
| `\ExpCincoBVazamentoAUCSujeito` | 80,1 | % | Exp. 5b robustez: RF, auc com os folds oficiais por sujeito (cenário do Exp. 5b) | `results/exp5b_robustez.csv` | `valor` onde analise=vazamento_mesmo_cenario, variante=centrada, modelo=RF, nivel=janela, metrica=auc_por_sujeito |
| `\ExpCincoBVazamentoAUCAleatoria` | 94,0 | % | Exp. 5b robustez: RF, auc com a partição aleatória de janelas (média de 3 sementes) | `results/exp5b_robustez.csv` | `valor` onde analise=vazamento_mesmo_cenario, variante=centrada, modelo=RF, nivel=janela, metrica=auc_aleatoria |
| `\ExpCincoBOpRegraSensCBruta` | 15,3 | % | Exp. 5b robustez: RF, C_bruta: sensibilidade no falso alarme da regra nas janelas KSS 0 (diagnóstico) | `results/exp5b_robustez.csv` | `valor` onde analise=ponto_operacao_regra, variante=C_bruta, modelo=RF, nivel=janela, metrica=sensibilidade_no_fa_da_regra |
| `\ExpCincoBOpRegraSensCCentrada` | 16,4 | % | Exp. 5b robustez: RF, C_centrada: sensibilidade no falso alarme da regra nas janelas KSS 0 (diagnóstico) | `results/exp5b_robustez.csv` | `valor` onde analise=ponto_operacao_regra, variante=C_centrada, modelo=RF, nivel=janela, metrica=sensibilidade_no_fa_da_regra |
| `\ExpCincoBOpRegraSensBCentrada` | 24,2 | % | Exp. 5b robustez: RF, B_centrada: sensibilidade no falso alarme da regra nas janelas KSS 0 (diagnóstico) | `results/exp5b_robustez.csv` | `valor` onde analise=ponto_operacao_regra, variante=B_centrada, modelo=RF, nivel=janela, metrica=sensibilidade_no_fa_da_regra |
| `\ExpCincoBTrezeQuedaAUCJanela` | 9,3 | pp | Exp. 5b RF: queda B − C com B treinado em 13 sujeitos (mediana dos sorteios − C), pp de AUC | `(derivado)` | `ExpCincoBTrezeAUCJanelaB - 100*ExpCincoBAUCJanelaC` com ExpCincoBTrezeAUCJanelaB, ExpCincoBAUCJanelaC |
| `\ExpCincoBLRRefAltAUCJanelaB` | 73,7 [69,3; 78,0] |  | Exp. 5b robustez: RL, (B) UTA→UTA com a centragem pelo vídeo KSS 5, AUC por janela | `results/exp5b_robustez.csv` | `valor` onde analise=referencia_kss5, variante=ref_video_kss5, modelo=LR, nivel=janela, metrica=auc_B; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRRefAltAUCJanelaC` | 66,2 [60,3; 71,7] |  | Exp. 5b robustez: RL, (C) DMD→UTA com a centragem pelo vídeo KSS 5, AUC por janela | `results/exp5b_robustez.csv` | `valor` onde analise=referencia_kss5, variante=ref_video_kss5, modelo=LR, nivel=janela, metrica=auc_C; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRRefAltQuedaAUCJanela` | 7,6 [3,0; 12,5] | pp | Exp. 5b robustez: RL, queda B − C com a centragem pelo vídeo KSS 5, AUC por janela (pp) | `results/exp5b_robustez.csv` | `valor` onde analise=referencia_kss5, variante=ref_video_kss5, modelo=LR, nivel=janela, metrica=queda_auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRRefAltQuedaAUCVideo` | 8,4 [3,4; 13,8] | pp | Exp. 5b robustez: RL, queda B − C com a centragem pelo vídeo KSS 5, AUC por vídeo (pp) | `results/exp5b_robustez.csv` | `valor` onde analise=referencia_kss5, variante=ref_video_kss5, modelo=LR, nivel=video, metrica=queda_auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRBootDuploQuedaAUCJanela` | 9,8 [1,9; 18,1] | pp | Exp. 5b robustez: RL, queda B − C por janela com IC em dois níveis (retreino de C) | `results/exp5b_robustez.csv` | `valor` onde analise=bootstrap_dois_niveis, variante=ref_video_alerta, modelo=LR, nivel=janela, metrica=queda_auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRTrezeAUCJanelaB` | 71,5 [68,1; 73,2] |  | Exp. 5b robustez: RL, B treinado com 13 sujeitos por fold (mediana dos sorteios; faixa = percentis 2,5 e 97,5) | `results/exp5b_robustez.csv` | `valor` onde analise=volume_treino, variante=13_sujeitos, modelo=LR, nivel=janela, metrica=auc_B; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRTrezeQuinhentasAUCJanelaB` | 71,2 [67,8; 73,6] |  | Exp. 5b robustez: RL, B treinado com 13 sujeitos e 510 janelas por fold (mediana dos sorteios) | `results/exp5b_robustez.csv` | `valor` onde analise=volume_treino, variante=13_sujeitos_510_janelas, modelo=LR, nivel=janela, metrica=auc_B; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRVazamentoAcBal` | 1,2 | pp | Exp. 5b robustez: RL, inflação pela partição aleatória de janelas no cenário do Exp. 5b (acuracia_balanceada, pp) | `results/exp5b_robustez.csv` | `valor` onde analise=vazamento_mesmo_cenario, variante=centrada, modelo=LR, nivel=janela, metrica=acuracia_balanceada_inflacao |
| `\ExpCincoBLRVazamentoAcBalSujeito` | 72,0 | % | Exp. 5b robustez: RL, acuracia_balanceada com os folds oficiais por sujeito (cenário do Exp. 5b) | `results/exp5b_robustez.csv` | `valor` onde analise=vazamento_mesmo_cenario, variante=centrada, modelo=LR, nivel=janela, metrica=acuracia_balanceada_por_sujeito |
| `\ExpCincoBLRVazamentoAcBalAleatoria` | 73,1 | % | Exp. 5b robustez: RL, acuracia_balanceada com a partição aleatória de janelas (média de 3 sementes) | `results/exp5b_robustez.csv` | `valor` onde analise=vazamento_mesmo_cenario, variante=centrada, modelo=LR, nivel=janela, metrica=acuracia_balanceada_aleatoria |
| `\ExpCincoBLRVazamentoAUC` | 2,1 | pp | Exp. 5b robustez: RL, inflação pela partição aleatória de janelas no cenário do Exp. 5b (auc, pp) | `results/exp5b_robustez.csv` | `valor` onde analise=vazamento_mesmo_cenario, variante=centrada, modelo=LR, nivel=janela, metrica=auc_inflacao |
| `\ExpCincoBLRVazamentoAUCSujeito` | 74,4 | % | Exp. 5b robustez: RL, auc com os folds oficiais por sujeito (cenário do Exp. 5b) | `results/exp5b_robustez.csv` | `valor` onde analise=vazamento_mesmo_cenario, variante=centrada, modelo=LR, nivel=janela, metrica=auc_por_sujeito |
| `\ExpCincoBLRVazamentoAUCAleatoria` | 76,5 | % | Exp. 5b robustez: RL, auc com a partição aleatória de janelas (média de 3 sementes) | `results/exp5b_robustez.csv` | `valor` onde analise=vazamento_mesmo_cenario, variante=centrada, modelo=LR, nivel=janela, metrica=auc_aleatoria |
| `\ExpCincoBLROpRegraSensCBruta` | 19,2 | % | Exp. 5b robustez: RL, C_bruta: sensibilidade no falso alarme da regra nas janelas KSS 0 (diagnóstico) | `results/exp5b_robustez.csv` | `valor` onde analise=ponto_operacao_regra, variante=C_bruta, modelo=LR, nivel=janela, metrica=sensibilidade_no_fa_da_regra |
| `\ExpCincoBLROpRegraSensCCentrada` | 24,7 | % | Exp. 5b robustez: RL, C_centrada: sensibilidade no falso alarme da regra nas janelas KSS 0 (diagnóstico) | `results/exp5b_robustez.csv` | `valor` onde analise=ponto_operacao_regra, variante=C_centrada, modelo=LR, nivel=janela, metrica=sensibilidade_no_fa_da_regra |
| `\ExpCincoBLROpRegraSensBCentrada` | 27,3 | % | Exp. 5b robustez: RL, B_centrada: sensibilidade no falso alarme da regra nas janelas KSS 0 (diagnóstico) | `results/exp5b_robustez.csv` | `valor` onde analise=ponto_operacao_regra, variante=B_centrada, modelo=LR, nivel=janela, metrica=sensibilidade_no_fa_da_regra |
| `\ExpCincoBLRTrezeQuedaAUCJanela` | 7,0 | pp | Exp. 5b RL: queda B − C com B treinado em 13 sujeitos (mediana dos sorteios − C), pp de AUC | `(derivado)` | `ExpCincoBLRTrezeAUCJanelaB - 100*ExpCincoBLRAUCJanelaC` com ExpCincoBLRTrezeAUCJanelaB, ExpCincoBLRAUCJanelaC |
| `\ExpCincoBJanelaCincoQuedaAUCJanela` | 11,8 [8,4; 15,7] | pp | Exp. 5b RF: queda B − C de AUC por janela com janelas de 5 s (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=janela_5s, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRJanelaCincoQuedaAUCJanela` | 5,4 [−0,1; 10,8] | pp | Exp. 5b LR: queda B − C de AUC por janela com janelas de 5 s (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=janela_5s, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRControleQuedaCincoDez` | 4,6 [−0,2; 9,7] | pp | Exp. 5b RL: queda B − C no contraste KSS 10 × 5 (pp de AUC) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=principal, modelo=LR, esquema=K10v5, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBRefSDoisQuedaAUCJanela` | 9,0 [5,5; 13,2] | pp | Exp. 5b RF: queda B − C de AUC por janela, variante ref_s2_espelho (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=ref_s2_espelho, modelo=RF, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBSemNormQuedaAUCVideo` | 0,2 [−5,6; 6,3] | pp | Exp. 5b RF: queda B − C de AUC por video, variante sem_normalizacao (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=sem_normalizacao, modelo=RF, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRRefSDoisQuedaAUCJanela` | 5,4 [0,3; 10,9] | pp | Exp. 5b LR: queda B − C de AUC por janela, variante ref_s2_espelho (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=ref_s2_espelho, modelo=LR, esquema=B3, nivel=janela, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBLRSemNormQuedaAUCVideo` | 1,2 [−5,4; 7,9] | pp | Exp. 5b LR: queda B − C de AUC por video, variante sem_normalizacao (pp) | `results/exp5b_transferencia.csv` | `valor` onde experimento=queda_B_menos_C, variante=sem_normalizacao, modelo=LR, esquema=B3, nivel=video, estrato=todos, limiar_tipo=fixo_0.5, metrica=auc; IC `ic95_inf`/`ic95_sup` |
| `\ExpCincoBMedianaFechMaxAlertaDMD` | 0,13 |  | Exp. 5b: mediana de dur_max_fechamento_s (bruto) nas janelas alertas, DMD | `results/exp5b_importancia.csv` | `mediana_alerta` onde tipo=auc_univariada, experimento=dmd, variante=sem_normalizacao, esquema=DMD, caracteristica=dur_max_fechamento_s |
| `\ExpCincoBMedianaPERCLOSAlertaDMD` | 0,037 |  | Exp. 5b: mediana de perclos (bruto) nas janelas alertas, DMD | `results/exp5b_importancia.csv` | `mediana_alerta` onde tipo=auc_univariada, experimento=dmd, variante=sem_normalizacao, esquema=DMD, caracteristica=perclos |
| `\ExpCincoBMedianaFechMaxAlertaUTA` | 0,07 |  | Exp. 5b: mediana de dur_max_fechamento_s (bruto) nas janelas alertas, UTA | `results/exp5b_importancia.csv` | `mediana_alerta` onde tipo=auc_univariada, experimento=uta, variante=sem_normalizacao, esquema=B3, caracteristica=dur_max_fechamento_s |
| `\ExpCincoBMedianaPERCLOSAlertaUTA` | 0,010 |  | Exp. 5b: mediana de perclos (bruto) nas janelas alertas, UTA | `results/exp5b_importancia.csv` | `mediana_alerta` onde tipo=auc_univariada, experimento=uta, variante=sem_normalizacao, esquema=B3, caracteristica=perclos |
| `\ExpCincoBLRQuedaNormMin` | 5,4 | pp | Exp. 5b: menor queda da RL entre as quatro normalizações (pp de AUC por janela) | `(derivado)` | `min(ExpCincoBLRQuedaAUCJanela, ExpCincoBLRSemNormQuedaAUCJanela, ExpCincoBLRRefAltQuedaAUCJanela, ExpCincoBLRRefSDoisQuedaAUCJanela)` com ExpCincoBLRQuedaAUCJanela, ExpCincoBLRSemNormQuedaAUCJanela, ExpCincoBLRRefAltQuedaAUCJanela, ExpCincoBLRRefSDoisQuedaAUCJanela |
| `\ExpCincoBLRQuedaNormMax` | 9,8 | pp | Exp. 5b: maior queda da RL entre as quatro normalizações (pp de AUC por janela) | `(derivado)` | `max(ExpCincoBLRQuedaAUCJanela, ExpCincoBLRSemNormQuedaAUCJanela, ExpCincoBLRRefAltQuedaAUCJanela, ExpCincoBLRRefSDoisQuedaAUCJanela)` com ExpCincoBLRQuedaAUCJanela, ExpCincoBLRSemNormQuedaAUCJanela, ExpCincoBLRRefAltQuedaAUCJanela, ExpCincoBLRRefSDoisQuedaAUCJanela |
| `\ExpCincoBQuedaOutrasNormMin` | 2,1 | pp | Exp. 5b: menor queda do RF sem a referência principal (pp de AUC por janela) | `(derivado)` | `min(ExpCincoBSemNormQuedaAUCJanela, ExpCincoBRefAltQuedaAUCJanela)` com ExpCincoBSemNormQuedaAUCJanela, ExpCincoBRefAltQuedaAUCJanela |
| `\ExpCincoBQuedaOutrasNormMax` | 2,3 | pp | Exp. 5b: maior queda do RF sem a referência principal (pp de AUC por janela) | `(derivado)` | `max(ExpCincoBSemNormQuedaAUCJanela, ExpCincoBRefAltQuedaAUCJanela)` com ExpCincoBSemNormQuedaAUCJanela, ExpCincoBRefAltQuedaAUCJanela |
| `\AppPesosSUmVermelhoMediana` | 3,3 | % | app, pesos sorteados (300): s1: % do tempo em vermelho, mediana | `results/app_sensibilidade_pesos.md` | regex `\&#124; s1:\ %\ do\ tempo\ em\ vermelho \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124;` (grupo 2) |
| `\AppPesosSUmVermelhoPcinco` | 1,7 | % | app, pesos sorteados (300): s1: % do tempo em vermelho, pcinco | `results/app_sensibilidade_pesos.md` | regex `\&#124; s1:\ %\ do\ tempo\ em\ vermelho \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124;` (grupo 3) |
| `\AppPesosSUmVermelhoPnoventaecinco` | 4,9 | % | app, pesos sorteados (300): s1: % do tempo em vermelho, pnoventaecinco | `results/app_sensibilidade_pesos.md` | regex `\&#124; s1:\ %\ do\ tempo\ em\ vermelho \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124;` (grupo 4) |
| `\AppPesosSCincoVermelhoMediana` | 67,7 | % | app, pesos sorteados (300): s5: % do tempo em vermelho, mediana | `results/app_sensibilidade_pesos.md` | regex `\&#124; s5:\ %\ do\ tempo\ em\ vermelho \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124;` (grupo 2) |
| `\AppPesosSCincoVermelhoPcinco` | 56,2 | % | app, pesos sorteados (300): s5: % do tempo em vermelho, pcinco | `results/app_sensibilidade_pesos.md` | regex `\&#124; s5:\ %\ do\ tempo\ em\ vermelho \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124;` (grupo 3) |
| `\AppPesosSCincoVermelhoPnoventaecinco` | 72,9 | % | app, pesos sorteados (300): s5: % do tempo em vermelho, pnoventaecinco | `results/app_sensibilidade_pesos.md` | regex `\&#124; s5:\ %\ do\ tempo\ em\ vermelho \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124; ([\d,]+) \&#124;` (grupo 4) |
| `\AppPesosBoas` | 291 |  | app, pesos sorteados: configurações com s1 em vermelho até 5% e s5 acima de 50% (quantas) | `results/app_sensibilidade_pesos.md` | regex `\*\*as duas coisas juntas: (\d+) de (\d+) \((\d+) %\)\*\*` (grupo 1) |
| `\AppPesosTotal` | 300 |  | app, pesos sorteados: configurações com s1 em vermelho até 5% e s5 acima de 50% (total sorteado) | `results/app_sensibilidade_pesos.md` | regex `\*\*as duas coisas juntas: (\d+) de (\d+) \((\d+) %\)\*\*` (grupo 2) |
| `\AppPesosBoasPct` | 97 |  | app, pesos sorteados: configurações com s1 em vermelho até 5% e s5 acima de 50% (%) | `results/app_sensibilidade_pesos.md` | regex `\*\*as duas coisas juntas: (\d+) de (\d+) \((\d+) %\)\*\*` (grupo 3) |
| `\AppUTASujeitos` | 24 |  | app em sono real (UTA-RLDD): pessoas testadas, fora do ajuste | `results/app_validacao_uta.md` | regex `\*\*Amostra:\*\* (\d+) sujeitos` (grupo 1) |
| `\AppUTAAcordadosAlarme` | 2 |  | app em sono real (UTA-RLDD), calibração no próprio vídeo: vídeos acordados com alarme | `results/app_validacao_uta.md` | regex `\&#124; Acordados\ com\ algum\ alarme\ vermelho \&#124; (\d+)/\d+` (grupo 1) |
| `\AppUTAAcordadosAmarelo` | 5 |  | app em sono real (UTA-RLDD), calibração no próprio vídeo: vídeos acordados que chegam ao amarelo | `results/app_validacao_uta.md` | regex `\&#124; Acordados\ que\ chegam\ ao\ amarelo \&#124; (\d+)/\d+` (grupo 1) |
| `\AppUTASonolentosAmarelo` | 17 |  | app em sono real (UTA-RLDD), calibração no próprio vídeo: vídeos sonolentos que chegam ao amarelo | `results/app_validacao_uta.md` | regex `\&#124; Sonolentos\ que\ chegam\ ao\ amarelo \&#124; (\d+)/\d+` (grupo 1) |
| `\AppUTASonolentosVermelho` | 15 |  | app em sono real (UTA-RLDD), calibração no próprio vídeo: vídeos sonolentos que chegam ao vermelho | `results/app_validacao_uta.md` | regex `\&#124; Sonolentos\ que\ chegam\ ao\ vermelho \&#124; (\d+)/\d+` (grupo 1) |
| `\AppUTASonolentosVerdes` | 7 |  | app em sono real: vídeos sonolentos que ficaram verdes o tempo todo | `(derivado)` | `AppUTASujeitos - AppUTASonolentosAmarelo` com AppUTASujeitos, AppUTASonolentosAmarelo |
| `\AppUTAAcordadosPctVermelho` | 0,46 | % | app em sono real: % do tempo em vermelho nos vídeos acordados | `results/app_validacao_uta.md` | regex `\&#124; Tempo\ em\ vermelho\ nos\ acordados \&#124; ([\d,]+) %` (grupo 1) |
| `\AppUTAParesMais` | 16 |  | app em sono real: pessoas com o vídeo sonolento mais tempo fora do verde que o acordado (mais) | `results/app_validacao_uta.md` | regex `\&#124; Pessoas\ com\ mais\ tempo\ fora\ do\ verde\ no\ vídeo\ sonolento\ \(mais\ /\ igual\ /\ menos\) \&#124; (\d+) / (\d+) / (\d+) \&#124;` (grupo 1) |
| `\AppUTAParesEmpate` | 7 |  | app em sono real: pessoas com o vídeo sonolento mais tempo fora do verde que o acordado (empate) | `results/app_validacao_uta.md` | regex `\&#124; Pessoas\ com\ mais\ tempo\ fora\ do\ verde\ no\ vídeo\ sonolento\ \(mais\ /\ igual\ /\ menos\) \&#124; (\d+) / (\d+) / (\d+) \&#124;` (grupo 2) |
| `\AppUTAParesMenos` | 1 |  | app em sono real: pessoas com o vídeo sonolento mais tempo fora do verde que o acordado (menos) | `results/app_validacao_uta.md` | regex `\&#124; Pessoas\ com\ mais\ tempo\ fora\ do\ verde\ no\ vídeo\ sonolento\ \(mais\ /\ igual\ /\ menos\) \&#124; (\d+) / (\d+) / (\d+) \&#124;` (grupo 3) |
| `\AppUTAAUCForaVerdeComIC` | 0,812 [0,705; 0,918] |  | app em sono real, calibração no próprio vídeo: AUC do % do tempo fora do verde [IC 95 %] | `results/app_validacao_uta.md` | regex `\&#124; AUC\ do\ %\ do\ tempo\ fora\ do\ verde\ \[IC\ 95\ %\] \&#124; (([\d,]+) \[[\d,]+; [\d,]+\]) \&#124; (([\d,]+) \[[\d,]+; [\d,]+\]) \&#124;` (grupo 1) |
| `\AppUTAAUCForaVerde` | 0,812 |  | app em sono real, calibração no próprio vídeo: AUC do % do tempo fora do verde [IC 95 %] (valor) | `results/app_validacao_uta.md` | regex `\&#124; AUC\ do\ %\ do\ tempo\ fora\ do\ verde\ \[IC\ 95\ %\] \&#124; (([\d,]+) \[[\d,]+; [\d,]+\]) \&#124; (([\d,]+) \[[\d,]+; [\d,]+\]) \&#124;` (grupo 2) |
| `\AppUTAAUCForaVerdeCalAcordado` | 0,855 |  | app em sono real, calibração do vídeo acordado: AUC do % do tempo fora do verde [IC 95 %] (valor) | `results/app_validacao_uta.md` | regex `\&#124; AUC\ do\ %\ do\ tempo\ fora\ do\ verde\ \[IC\ 95\ %\] \&#124; (([\d,]+) \[[\d,]+; [\d,]+\]) \&#124; (([\d,]+) \[[\d,]+; [\d,]+\]) \&#124;` (grupo 4) |
| `\AppUTAAUCNivelMedioComIC` | 0,839 [0,736; 0,935] |  | app em sono real, calibração no próprio vídeo: AUC do nível médio [IC 95 %] | `results/app_validacao_uta.md` | regex `\&#124; AUC\ do\ nível\ médio\ \[IC\ 95\ %\] \&#124; (([\d,]+) \[[\d,]+; [\d,]+\]) \&#124; (([\d,]+) \[[\d,]+; [\d,]+\]) \&#124;` (grupo 1) |
| `\AppUTAAUCNivelMedio` | 0,839 |  | app em sono real, calibração no próprio vídeo: AUC do nível médio [IC 95 %] (valor) | `results/app_validacao_uta.md` | regex `\&#124; AUC\ do\ nível\ médio\ \[IC\ 95\ %\] \&#124; (([\d,]+) \[[\d,]+; [\d,]+\]) \&#124; (([\d,]+) \[[\d,]+; [\d,]+\]) \&#124;` (grupo 2) |
| `\AppUTAAUCNivelMedioCalAcordado` | 0,891 |  | app em sono real, calibração do vídeo acordado: AUC do nível médio [IC 95 %] (valor) | `results/app_validacao_uta.md` | regex `\&#124; AUC\ do\ nível\ médio\ \[IC\ 95\ %\] \&#124; (([\d,]+) \[[\d,]+; [\d,]+\]) \&#124; (([\d,]+) \[[\d,]+; [\d,]+\]) \&#124;` (grupo 4) |
| `\AppCompCRF` | 0,811 [0,686; 0,915] |  | AUC por vídeo nas 24 pessoas do teste do app: RF treinado no DMD, com centragem | `results/app_comparacao_uta.csv` | `auc` onde detector=C_treino_DMD_RF, medida=principal; IC `ic95_inf`/`ic95_sup` |
| `\AppCompCLR` | 0,755 [0,627; 0,894] |  | AUC por vídeo nas 24 pessoas do teste do app: RL treinada no DMD, com centragem | `results/app_comparacao_uta.csv` | `auc` onde detector=C_treino_DMD_LR, medida=principal; IC `ic95_inf`/`ic95_sup` |
| `\AppCompBRF` | 0,953 [0,898; 0,997] |  | AUC por vídeo nas 24 pessoas do teste do app: RF treinado na UTA-RLDD, com centragem | `results/app_comparacao_uta.csv` | `auc` onde detector=B_treino_UTA_RF, medida=principal; IC `ic95_inf`/`ic95_sup` |
| `\AppCompBRFSemNorm` | 0,806 [0,686; 0,927] |  | AUC por vídeo nas 24 pessoas do teste do app: RF treinado na UTA-RLDD, sem centragem | `results/app_comparacao_uta.csv` | `auc` onde detector=B_treino_UTA_RF, medida=sem_normalizacao; IC `ic95_inf`/`ic95_sup` |
| `\AppCompRegra` | 0,853 [0,754; 0,944] |  | AUC por vídeo nas 24 pessoas do teste do app: regra do rótulo do DMD (sem treino) | `results/app_comparacao_uta.csv` | `auc` onde detector=regra_rotulo_DMD, medida=fracao_janelas; IC `ic95_inf`/`ic95_sup` |

## Conferência automática

Executada em 27/10/2026 11:31 por `src/experiments/conferir_numeros.py` (leitura e formatação independentes das do gerador).

- **Recalculados a partir do arquivo de origem e idênticos no JSON e no TeX: 658 de 658** números (1299 macros em `paper/numeros.tex`, 1299 no JSON; sobrando só no TeX: 0; só no JSON: 0).
- Encontrados no `.md` de resultados correspondente (valor e IC com a mesma precisão): 538; parcialmente (valor ou só parte do IC): 17; não encontrados: 0; sem `.md` de referência (derivados, agregados entre execuções do celular, regex no próprio .md): 103.
- Compilação de teste (IEEEtran + babel brazil, todos os macros impressos): ok — tectonic rc=0; 0 'Undefined control sequence'.
- Divergências: 0.

### Por número

| Macro | Valor | Recalculado | No .md |
|---|---|---|---|
| `\BaseUTASujeitos` | 60 | ok | sim |
| `\BaseUTAVideos` | 180 | ok | sim |
| `\BaseUTAArquivos` | 182 | ok | — |
| `\BaseUTASequencias` | 8.661 | ok | — |
| `\BaseUTASujeitosOficiais` | 58 | ok | — |
| `\BaseUTAVideosOficiais` | 174 | ok | — |
| `\BaseUTAJanelasBTres` | 12.487 | ok | sim |
| `\BaseUTAJanelasSonolentas` | 7.581 | ok | sim |
| `\BaseUTAJanelasAlerta` | 4.906 | ok | sim |
| `\BaseMRLImagens` | 84.898 | ok | — |
| `\BaseMRLSujeitos` | 37 | ok | — |
| `\BaseCEWImagens` | 4.846 | ok | sim |
| `\BaseDMDSujeitos` | 13 | ok | sim |
| `\BaseDMDJanelas` | 510 | ok | sim |
| `\BaseDMDJanelasSonolentas` | 423 | ok | sim |
| `\BaseDMDJanelasAlerta` | 87 | ok | sim |
| `\BaseDMDSessoesSonolencia` | 16 | ok | — |
| `\BaseDMDSessoesDirecaoReal` | 17 | ok | — |
| `\BaseDMDSessoes` | 65 | ok | — |
| `\BaseDMDSujeitosTodos` | 14 | ok | — |
| `\BaseNITYMEDVideos` | 126 | ok | sim |
| `\BaseNITYMEDMicrossono` | 19 | ok | sim |
| `\BaseNITYMEDBocejo` | 107 | ok | sim |
| `\BaseNITYMEDDeclarados` | 130 | ok | — |
| `\BaseNITYMEDJanelas` | 979 | ok | — |
| `\BaseNITYMEDJanelasAvaliaveis` | 529 | ok | — |
| `\ExpUmRFBSA` | 54,5 | ok | sim |
| `\ExpUmRFAcBal` | 57,8 | ok | sim |
| `\ExpUmRFFMacro` | 56,0 | ok | sim |
| `\ExpUmRFAUC` | 73,5 | ok | sim |
| `\ExpUmRFVA` | 57,3 | ok | sim |
| `\ExpUmLSTMBSA` | 50,8 | ok | sim |
| `\ExpUmLSTMAcBal` | 53,1 | ok | sim |
| `\ExpUmLSTMFMacro` | 51,8 | ok | sim |
| `\ExpUmLSTMAUC` | 68,6 | ok | sim |
| `\ExpUmLSTMVA` | 57,2 | ok | sim |
| `\ExpUmCNNBSA` | 48,0 | ok | sim |
| `\ExpUmCNNAcBal` | 51,4 | ok | sim |
| `\ExpUmCNNFMacro` | 50,0 | ok | sim |
| `\ExpUmCNNAUC` | 68,4 | ok | sim |
| `\ExpUmCNNVA` | 55,0 | ok | sim |
| `\ExpUmSVMBSA` | 53,0 | ok | sim |
| `\ExpUmSVMAcBal` | 56,2 | ok | sim |
| `\ExpUmSVMFMacro` | 54,2 | ok | sim |
| `\ExpUmSVMAUC` | 69,9 | ok | sim |
| `\ExpUmSVMVA` | 54,0 | ok | sim |
| `\ExpUmLRBSA` | 49,4 | ok | sim |
| `\ExpUmLRAcBal` | 49,7 | ok | sim |
| `\ExpUmLRFMacro` | 48,5 | ok | sim |
| `\ExpUmLRAUC` | 68,9 | ok | sim |
| `\ExpUmLRVA` | 50,7 | ok | sim |
| `\ExpUmChanceBSA` | 44,4 | ok | sim |
| `\ExpUmChanceVA` | 33,3 | ok | sim |
| `\ExpUmRFBTresBSA` | 86,9 | ok | sim |
| `\ExpUmRFBTresAcBal` | 84,8 | ok | sim |
| `\ExpUmRFBTresFMacro` | 84,6 | ok | sim |
| `\ExpUmRFBTresAUC` | 92,8 | ok | sim |
| `\ExpUmRFBTresVA` | 85,5 | ok | sim |
| `\ExpUmLSTMBTresBSA` | 83,0 | ok | sim |
| `\ExpUmLSTMBTresAcBal` | 80,8 | ok | sim |
| `\ExpUmLSTMBTresFMacro` | 80,3 | ok | sim |
| `\ExpUmLSTMBTresAUC` | 91,1 | ok | sim |
| `\ExpUmLSTMBTresVA` | 81,2 | ok | sim |
| `\ExpUmCNNBTresBSA` | 85,2 | ok | sim |
| `\ExpUmCNNBTresAcBal` | 83,8 | ok | sim |
| `\ExpUmCNNBTresFMacro` | 83,1 | ok | sim |
| `\ExpUmCNNBTresAUC` | 93,8 | ok | sim |
| `\ExpUmCNNBTresVA` | 84,5 | ok | sim |
| `\ExpUmSVMBTresBSA` | 85,5 | ok | sim |
| `\ExpUmSVMBTresAcBal` | 85,0 | ok | sim |
| `\ExpUmSVMBTresFMacro` | 83,6 | ok | sim |
| `\ExpUmSVMBTresAUC` | 92,5 | ok | sim |
| `\ExpUmSVMBTresVA` | 82,0 | ok | sim |
| `\ExpUmLRBTresBSA` | 74,0 | ok | sim |
| `\ExpUmLRBTresAcBal` | 72,2 | ok | sim |
| `\ExpUmLRBTresFMacro` | 71,0 | ok | sim |
| `\ExpUmLRBTresAUC` | 81,8 | ok | sim |
| `\ExpUmLRBTresVA` | 77,5 | ok | sim |
| `\ExpUmChanceBTresBSA` | 68,6 | ok | sim |
| `\ExpUmChanceBTresVA` | 50,0 | ok | sim |
| `\ExpUmRFBUmBSA` | 87,4 | ok | sim |
| `\ExpUmRFBUmAcBal` | 79,8 | ok | sim |
| `\ExpUmRFBUmFMacro` | 79,4 | ok | sim |
| `\ExpUmRFBUmAUC` | 88,7 | ok | sim |
| `\ExpUmRFBUmVA` | 82,2 | ok | sim |
| `\ExpUmLSTMBUmBSA` | 81,1 | ok | sim |
| `\ExpUmLSTMBUmAcBal` | 73,8 | ok | sim |
| `\ExpUmLSTMBUmFMacro` | 72,2 | ok | sim |
| `\ExpUmLSTMBUmAUC` | 85,2 | ok | sim |
| `\ExpUmLSTMBUmVA` | 77,8 | ok | sim |
| `\ExpUmCNNBUmBSA` | 83,2 | ok | sim |
| `\ExpUmCNNBUmAcBal` | 75,6 | ok | sim |
| `\ExpUmCNNBUmFMacro` | 74,2 | ok | sim |
| `\ExpUmCNNBUmAUC` | 87,5 | ok | sim |
| `\ExpUmCNNBUmVA` | 81,3 | ok | sim |
| `\ExpUmSVMBUmBSA` | 85,1 | ok | sim |
| `\ExpUmSVMBUmAcBal` | 78,0 | ok | sim |
| `\ExpUmSVMBUmFMacro` | 76,6 | ok | sim |
| `\ExpUmSVMBUmAUC` | 89,9 | ok | sim |
| `\ExpUmSVMBUmVA` | 81,8 | ok | sim |
| `\ExpUmLRBUmBSA` | 78,5 | ok | sim |
| `\ExpUmLRBUmAcBal` | 60,1 | ok | sim |
| `\ExpUmLRBUmFMacro` | 60,3 | ok | sim |
| `\ExpUmLRBUmAUC` | 80,3 | ok | sim |
| `\ExpUmLRBUmVA` | 70,9 | ok | sim |
| `\ExpUmChanceBUmBSA` | 81,4 | ok | sim |
| `\ExpUmChanceBUmVA` | 66,7 | ok | sim |
| `\ExpUmRFBDoisBSA` | 66,5 | ok | sim |
| `\ExpUmRFBDoisAcBal` | 63,4 | ok | sim |
| `\ExpUmRFBDoisFMacro` | 62,9 | ok | sim |
| `\ExpUmRFBDoisAUC` | 70,8 | ok | sim |
| `\ExpUmRFBDoisVA` | 70,1 | ok | sim |
| `\ExpUmLSTMBDoisBSA` | 60,9 | ok | sim |
| `\ExpUmLSTMBDoisAcBal` | 59,3 | ok | sim |
| `\ExpUmLSTMBDoisFMacro` | 58,5 | ok | sim |
| `\ExpUmLSTMBDoisAUC` | 63,7 | ok | sim |
| `\ExpUmLSTMBDoisVA` | 67,0 | ok | sim |
| `\ExpUmCNNBDoisBSA` | 61,2 | ok | sim |
| `\ExpUmCNNBDoisAcBal` | 60,8 | ok | sim |
| `\ExpUmCNNBDoisFMacro` | 59,7 | ok | sim |
| `\ExpUmCNNBDoisAUC` | 65,2 | ok | sim |
| `\ExpUmCNNBDoisVA` | 66,5 | ok | sim |
| `\ExpUmSVMBDoisBSA` | 66,2 | ok | sim |
| `\ExpUmSVMBDoisAcBal` | 63,9 | ok | sim |
| `\ExpUmSVMBDoisFMacro` | 63,7 | ok | sim |
| `\ExpUmSVMBDoisAUC` | 66,2 | ok | sim |
| `\ExpUmSVMBDoisVA` | 69,4 | ok | sim |
| `\ExpUmLRBDoisBSA` | 66,8 | ok | sim |
| `\ExpUmLRBDoisAcBal` | 63,4 | ok | sim |
| `\ExpUmLRBDoisFMacro` | 63,0 | ok | sim |
| `\ExpUmLRBDoisAUC` | 66,9 | ok | sim |
| `\ExpUmLRBDoisVA` | 73,6 | ok | sim |
| `\ExpUmChanceBDoisBSA` | 59,2 | ok | sim |
| `\ExpUmChanceBDoisVA` | 66,7 | ok | sim |
| `\ExpUmHMLSTMBSA` | 54,0 | ok | — |
| `\ExpUmHMLSTMVA` | 65,2 | ok | — |
| `\ExpUmRefLSTMBSA` | 52,8 | ok | — |
| `\ExpUmRefLSTMVA` | 61,4 | ok | — |
| `\ExpUmRefDensasBSA` | 52,0 | ok | — |
| `\ExpUmRefDensasVA` | 57,0 | ok | — |
| `\ExpUmHumanosVA` | 57,8 | ok | — |
| `\ExpUmDifHMLSTMRFVA` | 7,9 | ok | — |
| `\ExpUmDifRFHumanosVA` | −0,5 | ok | — |
| `\ExpUmLSTMParametros` | 5.324 | ok | sim |
| `\ExpUmLSTMKB` | 101,8 | ok | sim |
| `\ExpUmLSTMConcordancia` | 100,0 | ok | — |
| `\ExpUmLSTMPcinquentaMac` | 0,033 | ok | sim |
| `\ExpUmLSTMPnoventaecincoMac` | 0,045 | ok | sim |
| `\ExpDoisRFJanelaBSA` | 99,5 | ok | sim |
| `\ExpDoisRFJanelaAcBal` | 99,6 | ok | sim |
| `\ExpDoisRFJanelaAUC` | 100,0 | ok | sim |
| `\ExpDoisRFVideoBSA` | 47,2 | ok | sim |
| `\ExpDoisRFVideoAcBal` | 52,5 | ok | sim |
| `\ExpDoisRFVideoAUC` | 67,9 | ok | sim |
| `\ExpDoisRFVideoVA` | 52,8 | ok | sim |
| `\ExpDoisRFSujeitoBSA` | 54,2 | ok | sim |
| `\ExpDoisRFSujeitoAcBal` | 57,4 | ok | sim |
| `\ExpDoisRFSujeitoAUC` | 73,4 | ok | sim |
| `\ExpDoisRFSujeitoVA` | 55,7 | ok | sim |
| `\ExpDoisRFInflacaoJanelaBSA` | 45,3 | ok | sim |
| `\ExpDoisRFInflacaoJanelaAcBal` | 42,1 | ok | sim |
| `\ExpDoisRFInflacaoJanelaAUC` | 26,6 | ok | sim |
| `\ExpDoisRFInflacaoVideoBSA` | −7,0 | ok | sim |
| `\ExpDoisRFInflacaoVideoAcBal` | −5,0 | ok | sim |
| `\ExpDoisRFInflacaoVideoAUC` | −5,5 | ok | sim |
| `\ExpDoisRFInflacaoVideoVA` | −2,8 | ok | sim |
| `\ExpDoisLSTMJanelaBSA` | 87,3 | ok | sim |
| `\ExpDoisLSTMJanelaAcBal` | 87,9 | ok | sim |
| `\ExpDoisLSTMJanelaAUC` | 97,1 | ok | sim |
| `\ExpDoisLSTMVideoBSA` | 45,1 | ok | sim |
| `\ExpDoisLSTMVideoAcBal` | 48,9 | ok | sim |
| `\ExpDoisLSTMVideoAUC` | 62,3 | ok | sim |
| `\ExpDoisLSTMVideoVA` | 46,2 | ok | sim |
| `\ExpDoisLSTMSujeitoBSA` | 49,2 | ok | sim |
| `\ExpDoisLSTMSujeitoAcBal` | 51,0 | ok | sim |
| `\ExpDoisLSTMSujeitoAUC` | 67,6 | ok | sim |
| `\ExpDoisLSTMSujeitoVA` | 50,8 | ok | sim |
| `\ExpDoisLSTMInflacaoJanelaBSA` | 38,1 | ok | sim |
| `\ExpDoisLSTMInflacaoJanelaAcBal` | 36,9 | ok | sim |
| `\ExpDoisLSTMInflacaoJanelaAUC` | 29,6 | ok | sim |
| `\ExpDoisLSTMInflacaoVideoBSA` | −4,1 | ok | sim |
| `\ExpDoisLSTMInflacaoVideoAcBal` | −2,1 | ok | sim |
| `\ExpDoisLSTMInflacaoVideoAUC` | −5,3 | ok | sim |
| `\ExpDoisLSTMInflacaoVideoVA` | −4,6 | ok | sim |
| `\ExpDoisVideosVistosJanela` | 97,1 | ok | sim |
| `\ExpTresIntraMRLAc` | 94,1 | ok | sim |
| `\ExpTresMRLCEWPatchAc` | 79,0 | ok | sim |
| `\ExpTresMRLCEWCropAc` | 80,6 | ok | sim |
| `\ExpTresCEWPatchMRLAc` | 68,1 | ok | sim |
| `\ExpTresCEWCropMRLAc` | 67,1 | ok | sim |
| `\ExpTresIntraCEWPatchAc` | 94,7 | ok | sim |
| `\ExpTresIntraCEWCropAc` | 94,5 | ok | sim |
| `\ExpTresFinalMRLCEWPatchAc` | 85,7 | ok | sim |
| `\ExpTresFinalMRLCEWCropAc` | 90,0 | ok | sim |
| `\ExpTresIntraMRLAcBal` | 94,2 | ok | sim |
| `\ExpTresMRLCEWPatchAcBal` | 78,7 | ok | sim |
| `\ExpTresMRLCEWCropAcBal` | 80,3 | ok | sim |
| `\ExpTresCEWPatchMRLAcBal` | 68,4 | ok | sim |
| `\ExpTresCEWCropMRLAcBal` | 67,4 | ok | sim |
| `\ExpTresIntraCEWPatchAcBal` | 94,7 | ok | sim |
| `\ExpTresIntraCEWCropAcBal` | 94,4 | ok | sim |
| `\ExpTresFinalMRLCEWPatchAcBal` | 85,5 | ok | sim |
| `\ExpTresFinalMRLCEWCropAcBal` | 89,9 | ok | sim |
| `\ExpTresIntraMRLAUC` | 98,6 | ok | sim |
| `\ExpTresMRLCEWPatchAUC` | 93,2 | ok | sim |
| `\ExpTresMRLCEWCropAUC` | 93,5 | ok | sim |
| `\ExpTresCEWPatchMRLAUC` | 84,2 | ok | sim |
| `\ExpTresCEWCropMRLAUC` | 83,6 | ok | sim |
| `\ExpTresIntraCEWPatchAUC` | 98,9 | ok | sim |
| `\ExpTresIntraCEWCropAUC` | 98,8 | ok | sim |
| `\ExpTresFinalMRLCEWPatchAUC` | 96,8 | ok | sim |
| `\ExpTresFinalMRLCEWCropAUC` | 97,6 | ok | sim |
| `\ExpTresQuedaMRLCEWPatch` | 15,1 | ok | — |
| `\ExpTresQuedaCEWMRLPatch` | 26,5 | ok | — |
| `\ExpTresIntraMRLComOculosAc` | 88,8 | ok | sim |
| `\ExpTresCEWPatchMRLComOculosAc` | 65,2 | ok | sim |
| `\ExpTresIntraMRLComOculosAcBal` | 88,5 | ok | sim |
| `\ExpTresCEWPatchMRLComOculosAcBal` | 61,1 | ok | sim |
| `\ExpTresIntraMRLComOculosAUC` | 95,7 | ok | sim |
| `\ExpTresCEWPatchMRLComOculosAUC` | 65,1 | ok | sim |
| `\ExpTresIntraMRLSemOculosAc` | 96,1 | ok | sim |
| `\ExpTresCEWPatchMRLSemOculosAc` | 69,3 | ok | sim |
| `\ExpTresIntraMRLSemOculosAcBal` | 96,0 | ok | sim |
| `\ExpTresCEWPatchMRLSemOculosAcBal` | 71,8 | ok | sim |
| `\ExpTresIntraMRLSemOculosAUC` | 99,4 | ok | sim |
| `\ExpTresCEWPatchMRLSemOculosAUC` | 93,4 | ok | sim |
| `\ExpTresParametros` | 62.865 | ok | sim |
| `\ExpQuatroCNNIntPcinquenta` | 0,060 | ok | sim |
| `\ExpQuatroCNNIntPnoventaecinco` | 0,064 | ok | sim |
| `\ExpQuatroCNNIntQuatroThreadsPcinquenta` | 0,070 | ok | sim |
| `\ExpQuatroCNNIntQuatroThreadsPnoventaecinco` | 0,087 | ok | sim |
| `\ExpQuatroCNNIntKB` | 72,1 | ok | sim |
| `\ExpQuatroCNNFloatPcinquenta` | 0,067 | ok | sim |
| `\ExpQuatroCNNFloatPnoventaecinco` | 0,070 | ok | sim |
| `\ExpQuatroCNNFloatQuatroThreadsPcinquenta` | 0,082 | ok | sim |
| `\ExpQuatroCNNFloatQuatroThreadsPnoventaecinco` | 0,099 | ok | sim |
| `\ExpQuatroCNNFloatKB` | 248,2 | ok | sim |
| `\ExpQuatroModelosKB` | 173,9 | ok | — |
| `\ExpQuatroAcCEWPatchKeras` | 85,7 | ok | sim |
| `\ExpQuatroAcCEWPatchInt` | 85,6 | ok | sim |
| `\ExpQuatroAcCEWCropKeras` | 90,0 | ok | sim |
| `\ExpQuatroAcCEWCropInt` | 89,7 | ok | sim |
| `\ExpQuatroAcMRLCompletoKeras` | 98,5 | ok | sim |
| `\ExpQuatroAcMRLCompletoInt` | 98,5 | ok | sim |
| `\ExpQuatroQuedaQuantizacaoMax` | 0,31 | ok | sim |
| `\MacPWATotalPcinquenta` | 12,7 | ok | — |
| `\MacPWATotalPnoventaecinco` | 14,8 | ok | — |
| `\MacPWAFPS` | 29,9 | ok | — |
| `\CelularIPhoneNome` | iPhone 17 Pro (512 GB) | ok | sim |
| `\CelularIPhoneExecucoes` | 2 | ok | — |
| `\CelularIPhoneVersaoApp` | 1.0 | ok | sim |
| `\CelularIPhoneSistema` | iOS 26.6.1 | ok | sim |
| `\CelularIPhoneNavegador` | Chrome 154 (iOS, motor WebKit) | ok | sim |
| `\CelularIPhoneDelegate` | GPU | ok | sim |
| `\CelularIPhoneTotalPcinquenta` | 17,0 | ok | — |
| `\CelularIPhoneTotalPcinquentaMin` | 11,0 | ok | sim |
| `\CelularIPhoneTotalPcinquentaMax` | 23,0 | ok | sim |
| `\CelularIPhoneTotalPnoventaecinco` | 22,0 | ok | — |
| `\CelularIPhoneTotalPnoventaecincoMin` | 14,0 | ok | sim |
| `\CelularIPhoneTotalPnoventaecincoMax` | 30,0 | ok | sim |
| `\CelularIPhoneLandmarkerPcinquenta` | 14,0 | ok | — |
| `\CelularIPhoneLandmarkerPcinquentaMin` | 9,0 | ok | sim |
| `\CelularIPhoneLandmarkerPcinquentaMax` | 19,0 | ok | sim |
| `\CelularIPhoneLandmarkerPnoventaecinco` | 18,5 | ok | — |
| `\CelularIPhoneLandmarkerPnoventaecincoMin` | 11,0 | ok | sim |
| `\CelularIPhoneLandmarkerPnoventaecincoMax` | 26,0 | ok | sim |
| `\CelularIPhoneCNNPcinquenta` | 2,5 | ok | — |
| `\CelularIPhoneCNNPcinquentaMin` | 2,0 | ok | sim |
| `\CelularIPhoneCNNPcinquentaMax` | 3,0 | ok | sim |
| `\CelularIPhoneCNNPnoventaecinco` | 4,5 | ok | — |
| `\CelularIPhoneCNNPnoventaecincoMin` | 3,0 | ok | sim |
| `\CelularIPhoneCNNPnoventaecincoMax` | 6,0 | ok | sim |
| `\CelularIPhoneLSTMPcinquenta` | 1,0 | ok | — |
| `\CelularIPhoneLSTMPcinquentaMin` | 1,0 | ok | sim |
| `\CelularIPhoneLSTMPcinquentaMax` | 1,0 | ok | sim |
| `\CelularIPhoneLSTMPnoventaecinco` | 1,0 | ok | — |
| `\CelularIPhoneLSTMPnoventaecincoMin` | 1,0 | ok | sim |
| `\CelularIPhoneLSTMPnoventaecincoMax` | 1,0 | ok | sim |
| `\CelularIPhoneFPS` | 50,2 | ok | — |
| `\CelularIPhoneFPSMin` | 40,5 | ok | sim |
| `\CelularIPhoneFPSMax` | 60,0 | ok | sim |
| `\CelularIPhoneTotalPcinquentaInteiro` | 17 | ok | — |
| `\CelularIPhoneTotalPcinquentaMinInteiro` | 11 | ok | — |
| `\CelularIPhoneTotalPcinquentaMaxInteiro` | 23 | ok | — |
| `\CelularIPhoneQuadros` | 6.031 | ok | — |
| `\CelularIPhoneQuadrosComRosto` | 6.031 | ok | — |
| `\CelularIPhoneTaxaRosto` | 100,0 | ok | — |
| `\CelularVUmFalsoAlarmePERCLOS` | 25,8 | ok | — |
| `\CelularVUmFalsoAlarmeQuadros` | 2.281 | ok | — |
| `\CelularVUmFalsoAlarmeQuadrosTotal` | 3.600 | ok | — |
| `\CelularVUmFalsoAlarmeFracao` | 63,4 | ok | — |
| `\ExpCincoACustoAcBalDif` | 2,0 | ok | sim |
| `\ExpCincoACustoAcBalOficial` | 58,9 | ok | sim |
| `\ExpCincoACustoAcBalNosso` | 57,0 | ok | sim |
| `\ExpCincoACustoAcBalP` | 0,555 | ok | sim |
| `\ExpCincoACustoVADif` | 2,1 | ok | sim |
| `\ExpCincoACustoVAOficial` | 58,9 | ok | sim |
| `\ExpCincoACustoVANosso` | 56,7 | ok | sim |
| `\ExpCincoACustoVAP` | 0,734 | ok | sim |
| `\ExpCincoACustoBTresAcBalDif` | 0,5 | ok | sim |
| `\ExpCincoACustoBTresAcBalOficial` | 85,5 | ok | sim |
| `\ExpCincoACustoBTresAcBalNosso` | 85,0 | ok | sim |
| `\ExpCincoACustoBTresAcBalP` | 0,904 | ok | sim |
| `\ExpCincoACustoBTresVADif` | 2,1 | ok | sim |
| `\ExpCincoACustoBTresVAOficial` | 88,3 | ok | sim |
| `\ExpCincoACustoBTresVANosso` | 86,2 | ok | sim |
| `\ExpCincoACustoBTresVAP` | 0,806 | ok | sim |
| `\ExpCincoAMesmoAcBalDif` | −1,7 | ok | sim |
| `\ExpCincoAMesmoAcBalOficial` | 54,6 | ok | sim |
| `\ExpCincoAMesmoAcBalNosso` | 56,3 | ok | sim |
| `\ExpCincoAMesmoAcBalP` | 0,712 | ok | sim |
| `\ExpCincoAMesmoVADif` | −5,0 | ok | sim |
| `\ExpCincoAMesmoVAOficial` | 55,3 | ok | sim |
| `\ExpCincoAMesmoVANosso` | 60,3 | ok | sim |
| `\ExpCincoAMesmoVAP` | 0,460 | ok | sim |
| `\ExpCincoAMesmoBTresAcBalDif` | −6,7 | ok | sim |
| `\ExpCincoAMesmoBTresAcBalOficial` | 83,1 | ok | sim |
| `\ExpCincoAMesmoBTresAcBalNosso` | 89,8 | ok | sim |
| `\ExpCincoAMesmoBTresAcBalP` | 0,029 | ok | sim |
| `\ExpCincoAMesmoBTresVADif` | −7,4 | ok | sim |
| `\ExpCincoAMesmoBTresVAOficial` | 87,2 | ok | sim |
| `\ExpCincoAMesmoBTresVANosso` | 94,7 | ok | sim |
| `\ExpCincoAMesmoBTresVAP` | 0,039 | ok | sim |
| `\ExpCincoAMesmoLOSOAcBalDif` | 0,4 | ok | sim |
| `\ExpCincoAMesmoLOSOAcBalOficial` | 55,6 | ok | sim |
| `\ExpCincoAMesmoLOSOAcBalNosso` | 55,2 | ok | sim |
| `\ExpCincoAMesmoLOSOAcBalP` | 0,910 | ok | sim |
| `\ExpCincoAMesmoLOSOVADif` | −0,7 | ok | sim |
| `\ExpCincoAMesmoLOSOVAOficial` | 55,3 | ok | sim |
| `\ExpCincoAMesmoLOSOVANosso` | 56,0 | ok | sim |
| `\ExpCincoAMesmoLOSOVAP` | 1,000 | ok | sim |
| `\ExpCincoAMesmoLOSOBTresAcBalDif` | −4,5 | ok | sim |
| `\ExpCincoAMesmoLOSOBTresAcBalOficial` | 85,0 | ok | sim |
| `\ExpCincoAMesmoLOSOBTresAcBalNosso` | 89,5 | ok | sim |
| `\ExpCincoAMesmoLOSOBTresAcBalP` | 0,163 | ok | sim |
| `\ExpCincoAMesmoLOSOBTresVADif` | −3,2 | ok | sim |
| `\ExpCincoAMesmoLOSOBTresVAOficial` | 88,3 | ok | sim |
| `\ExpCincoAMesmoLOSOBTresVANosso` | 91,5 | ok | sim |
| `\ExpCincoAMesmoLOSOBTresVAP` | 0,546 | ok | sim |
| `\ExpCincoASujeitos` | 47 | ok | sim |
| `\ExpCincoAVideos` | 141 | ok | sim |
| `\ExpCincoAVideosBTres` | 94 | ok | sim |
| `\ExpCincoBAcBalJanelaA` | 76,4 | ok | sim |
| `\ExpCincoBAcBalJanelaB` | 74,2 | ok | sim |
| `\ExpCincoBAcBalJanelaC` | 65,2 | ok | sim |
| `\ExpCincoBQuedaAcBalJanela` | 9,0 | ok | sim |
| `\ExpCincoBChoqueAcBalJanela` | 11,3 | ok | sim |
| `\ExpCincoBAcVideoB` | 85,8 | ok | sim |
| `\ExpCincoBAcVideoC` | 73,3 | ok | sim |
| `\ExpCincoBQuedaAcVideo` | 12,5 | ok | sim |
| `\ExpCincoBCSensJanela` | 48,4 | ok | sim |
| `\ExpCincoBCSensVideo` | 55,0 | ok | parcial |
| `\ExpCincoBCEspecJanela` | 82,0 | ok | sim |
| `\ExpCincoBCEspecVideo` | 91,7 | ok | sim |
| `\ExpCincoBAcBalJanelaACal` | 78,6 | ok | sim |
| `\ExpCincoBAcBalJanelaBCal` | 74,0 | ok | sim |
| `\ExpCincoBAcBalJanelaCCal` | 62,9 | ok | sim |
| `\ExpCincoBQuedaAcBalJanelaCal` | 11,1 | ok | sim |
| `\ExpCincoBChoqueAcBalJanelaCal` | 15,8 | ok | sim |
| `\ExpCincoBAcVideoBCal` | 82,5 | ok | sim |
| `\ExpCincoBAcVideoCCal` | 60,8 | ok | sim |
| `\ExpCincoBQuedaAcVideoCal` | 21,7 | ok | sim |
| `\ExpCincoBCSensJanelaCal` | 37,0 | ok | sim |
| `\ExpCincoBCSensVideoCal` | 21,7 | ok | sim |
| `\ExpCincoBCEspecJanelaCal` | 88,7 | ok | sim |
| `\ExpCincoBCEspecVideoCal` | 100,0 | ok | sim |
| `\ExpCincoBAUCJanelaA` | 82,4 | ok | sim |
| `\ExpCincoBAUCJanelaB` | 80,1 | ok | sim |
| `\ExpCincoBAUCJanelaC` | 69,2 | ok | sim |
| `\ExpCincoBQuedaAUCJanela` | 10,9 | ok | sim |
| `\ExpCincoBChoqueAUCJanela` | 13,1 | ok | sim |
| `\ExpCincoBLabAUCJanela` | 2,2 | ok | sim |
| `\ExpCincoBAUCVideoB` | 92,4 | ok | sim |
| `\ExpCincoBAUCVideoC` | 80,5 | ok | sim |
| `\ExpCincoBQuedaAUCVideo` | 11,9 | ok | sim |
| `\ExpCincoBLimiarCal` | 0,662 | ok | sim |
| `\ExpCincoBLRAcBalJanelaA` | 78,6 | ok | sim |
| `\ExpCincoBLRAcBalJanelaB` | 72,0 | ok | sim |
| `\ExpCincoBLRAcBalJanelaC` | 64,0 | ok | sim |
| `\ExpCincoBLRQuedaAcBalJanela` | 7,9 | ok | sim |
| `\ExpCincoBLRChoqueAcBalJanela` | 14,6 | ok | sim |
| `\ExpCincoBLRAcVideoB` | 85,8 | ok | sim |
| `\ExpCincoBLRAcVideoC` | 65,8 | ok | sim |
| `\ExpCincoBLRQuedaAcVideo` | 20,0 | ok | sim |
| `\ExpCincoBLRCSensJanela` | 33,7 | ok | parcial |
| `\ExpCincoBLRCSensVideo` | 31,7 | ok | parcial |
| `\ExpCincoBLRCEspecJanela` | 94,4 | ok | sim |
| `\ExpCincoBLRCEspecVideo` | 100,0 | ok | sim |
| `\ExpCincoBLRAcBalJanelaACal` | 78,7 | ok | sim |
| `\ExpCincoBLRAcBalJanelaBCal` | 72,2 | ok | sim |
| `\ExpCincoBLRAcBalJanelaCCal` | 63,3 | ok | sim |
| `\ExpCincoBLRQuedaAcBalJanelaCal` | 8,9 | ok | sim |
| `\ExpCincoBLRChoqueAcBalJanelaCal` | 15,4 | ok | sim |
| `\ExpCincoBLRAcVideoBCal` | 85,0 | ok | sim |
| `\ExpCincoBLRAcVideoCCal` | 64,2 | ok | sim |
| `\ExpCincoBLRQuedaAcVideoCal` | 20,8 | ok | sim |
| `\ExpCincoBLRCSensJanelaCal` | 30,8 | ok | parcial |
| `\ExpCincoBLRCSensVideoCal` | 28,3 | ok | sim |
| `\ExpCincoBLRCEspecJanelaCal` | 95,8 | ok | sim |
| `\ExpCincoBLRCEspecVideoCal` | 100,0 | ok | sim |
| `\ExpCincoBLRAUCJanelaA` | 83,5 | ok | sim |
| `\ExpCincoBLRAUCJanelaB` | 74,4 | ok | sim |
| `\ExpCincoBLRAUCJanelaC` | 64,5 | ok | sim |
| `\ExpCincoBLRQuedaAUCJanela` | 9,8 | ok | sim |
| `\ExpCincoBLRChoqueAUCJanela` | 18,9 | ok | sim |
| `\ExpCincoBLRLabAUCJanela` | 9,1 | ok | sim |
| `\ExpCincoBLRAUCVideoB` | 89,0 | ok | sim |
| `\ExpCincoBLRAUCVideoC` | 76,0 | ok | sim |
| `\ExpCincoBLRQuedaAUCVideo` | 13,0 | ok | sim |
| `\ExpCincoBLRLimiarCal` | 0,535 | ok | sim |
| `\ExpSeisRostoNoite` | 99,15 | ok | sim |
| `\ExpSeisSNRNoite` | 7,2 | ok | sim |
| `\ExpSeisRuidoNoite` | 0,0084 | ok | parcial |
| `\ExpSeisRostoNoiteMicrossono` | 99,23 | ok | sim |
| `\ExpSeisSNRNoiteMicrossono` | 7,1 | ok | sim |
| `\ExpSeisRuidoNoiteMicrossono` | 0,0084 | ok | parcial |
| `\ExpSeisRostoNoiteBocejo` | 99,14 | ok | sim |
| `\ExpSeisSNRNoiteBocejo` | 7,2 | ok | sim |
| `\ExpSeisRuidoNoiteBocejo` | 0,0084 | ok | parcial |
| `\ExpSeisRostoUTA` | 99,77 | ok | sim |
| `\ExpSeisSNRUTA` | 21,5 | ok | sim |
| `\ExpSeisRuidoUTA` | 0,0037 | ok | parcial |
| `\ExpSeisRostoUTAAlerta` | 99,95 | ok | sim |
| `\ExpSeisSNRUTAAlerta` | 24,4 | ok | sim |
| `\ExpSeisRuidoUTAAlerta` | 0,0036 | ok | parcial |
| `\ExpSeisRostoDMDSonolencia` | 99,78 | ok | sim |
| `\ExpSeisSNRDMDSonolencia` | 23,1 | ok | sim |
| `\ExpSeisRuidoDMDSonolencia` | 0,0046 | ok | sim |
| `\ExpSeisRostoDMDParado` | 99,89 | ok | sim |
| `\ExpSeisSNRDMDParado` | 13,6 | ok | sim |
| `\ExpSeisRuidoDMDParado` | 0,0059 | ok | parcial |
| `\ExpSeisRostoDMDMovimento` | 99,65 | ok | sim |
| `\ExpSeisSNRDMDMovimento` | 8,7 | ok | sim |
| `\ExpSeisRuidoDMDMovimento` | 0,0097 | ok | parcial |
| `\ExpSeisDifRostoNoiteUTA` | −0,61 | ok | sim |
| `\ExpSeisDifSNRNoiteUTA` | −14,3 | ok | sim |
| `\ExpSeisDifRostoNoiteDMDSonolencia` | −0,63 | ok | sim |
| `\ExpSeisDifSNRNoiteDMDSonolencia` | −15,9 | ok | sim |
| `\ExpSeisDifRostoNoiteDMDParado` | −0,73 | ok | sim |
| `\ExpSeisDifSNRNoiteDMDParado` | −6,4 | ok | sim |
| `\ExpSeisDifRostoNoiteDMDMovimento` | −0,50 | ok | sim |
| `\ExpSeisDifSNRNoiteDMDMovimento` | −1,6 | ok | sim |
| `\ExpSeisDifSNRMovimentoParado` | −5,4 | ok | sim |
| `\ExpSeisDifRuidoMovimentoParado` | 0,0043 | ok | sim |
| `\ExpSeisSujeitosPareadosMovimento` | 11 | ok | sim |
| `\ExpSeisDisparoModeloDMDMicrossono` | 44,7 | ok | sim |
| `\ExpSeisDisparoModeloDMDBocejo` | 73,0 | ok | sim |
| `\ExpSeisDisparoModeloDMDUTAAlerta` | 26,0 | ok | sim |
| `\ExpSeisDisparoModeloDMDUTAIntermediario` | 40,5 | ok | sim |
| `\ExpSeisDisparoModeloDMDUTASonolento` | 52,2 | ok | sim |
| `\ExpSeisDisparoModeloDMDDMDParado` | 45,7 | ok | sim |
| `\ExpSeisDisparoModeloDMDDMDMovimento` | 42,9 | ok | sim |
| `\ExpSeisDisparoModeloDMDDMDSonolento` | 82,0 | ok | sim |
| `\ExpSeisDisparoModeloUTAMicrossono` | 83,5 | ok | sim |
| `\ExpSeisDisparoModeloUTABocejo` | 89,5 | ok | sim |
| `\ExpSeisDisparoModeloUTAUTAAlerta` | 43,6 | ok | sim |
| `\ExpSeisDisparoModeloUTAUTAIntermediario` | 58,3 | ok | sim |
| `\ExpSeisDisparoModeloUTAUTASonolento` | 69,4 | ok | sim |
| `\ExpSeisDisparoModeloUTADMDParado` | 69,1 | ok | sim |
| `\ExpSeisDisparoModeloUTADMDMovimento` | 77,1 | ok | sim |
| `\ExpSeisDisparoModeloUTADMDSonolento` | 92,9 | ok | sim |
| `\ExpSeisDisparoRegraMicrossono` | 15,0 | ok | sim |
| `\ExpSeisDisparoRegraBocejo` | 61,7 | ok | sim |
| `\ExpSeisDisparoRegraUTAAlerta` | 2,6 | ok | sim |
| `\ExpSeisDisparoRegraUTAIntermediario` | 9,7 | ok | sim |
| `\ExpSeisDisparoRegraUTASonolento` | 24,1 | ok | sim |
| `\ExpSeisDisparoRegraDMDParado` | 14,9 | ok | sim |
| `\ExpSeisDisparoRegraDMDMovimento` | 23,8 | ok | sim |
| `\ExpSeisDisparoRegraDMDSonolento` | 75,7 | ok | sim |
| `\AppVersao` | 2.1 | ok | — |
| `\AppLimiarAmarelo` | 40 | ok | — |
| `\AppLimiarVermelho` | 70 | ok | — |
| `\AppVolumePadrao` | 20 | ok | — |
| `\AppUTAAlertaVideos` | 3 | ok | sim |
| `\AppUTAAlertaVerdeMin` | 100,0 | ok | sim |
| `\AppUTAAlertaAlarmes` | 0 | ok | sim |
| `\AppUTAAlertaAvisos` | 0 | ok | sim |
| `\AppUTASonolentoVideos` | 3 | ok | sim |
| `\AppUTASonolentoComVermelho` | 1 | ok | sim |
| `\AppKappaNavegador` | 0,902 | ok | — |
| `\AppBocejosNavegadorAcertos` | 6 | ok | sim |
| `\AppBocejosNavegadorAnotados` | 24 | ok | sim |
| `\AppBocejosNavegadorAnotadosSemMao` | 14 | ok | sim |
| `\AppDMDMinutosDirecaoReal` | 63,3 | ok | sim |
| `\AppDMDVermelhoDirecaoReal` | 3,3 | ok | sim |
| `\AppDMDAmareloDirecaoReal` | 4,1 | ok | sim |
| `\AppDMDAlarmesDirecaoReal` | 4 | ok | sim |
| `\AppDMDAlarmesHoraDirecaoReal` | 3,8 | ok | sim |
| `\AppDMDBocejosFalsosDirecaoReal` | 0 | ok | sim |
| `\AppDMDMinutosSonolencia` | 46,6 | ok | sim |
| `\AppDMDVermelhoSonolencia` | 68,6 | ok | sim |
| `\AppKappaOffline` | 0,938 | ok | sim |
| `\AppFechUmSRevocacao` | 0,920 | ok | sim |
| `\AppFechUmSPrecisao` | 0,748 | ok | sim |
| `\AppFechMeioSRevocacao` | 0,929 | ok | sim |
| `\AppFechMeioSPrecisao` | 0,786 | ok | sim |
| `\AppDMDVermelhoVUmDirecaoReal` | 55,6 | ok | sim |
| `\AppDMDSessoesDirecaoReal` | 17 | ok | sim |
| `\AppDMDSessoesSemAlarme` | 16 | ok | — |
| `\AppGACincoVermelho` | 55,8 | ok | sim |
| `\AppBocejoAcertosSemMao` | 30 | ok | sim |
| `\AppBocejoAnotadosSemMao` | 47 | ok | sim |
| `\AppBocejoAcertosComMao` | 1 | ok | sim |
| `\AppBocejoAnotadosComMao` | 38 | ok | sim |
| `\AppBocejoFalsos` | 0 | ok | sim |
| `\AppBocejoRevocacaoSemMao` | 0,638 | ok | sim |
| `\AppBocejoRevocacaoComMao` | 0,026 | ok | sim |
| `\SinaisDetectorFUmPiscadas` | 0,749 | ok | sim |
| `\SinaisDetectorPrecisaoPiscadas` | 0,811 | ok | sim |
| `\SinaisDetectorRevocacaoPiscadas` | 0,775 | ok | sim |
| `\SinaisDetectorAcBalQuadro` | 0,900 | ok | sim |
| `\SinaisDetectorKappaQuadro` | 0,773 | ok | sim |
| `\SinaisDetectorSessoes` | 16 | ok | sim |
| `\SinaisSafeDriveMin` | 10 | ok | — |
| `\SinaisSafeDriveMax` | 26 | ok | — |
| `\BaseDMDHoras` | 4,24 | ok | — |
| `\BaseDMDFPSFaixa` | 29,76 | ok | — |
| `\BaseNITYMEDHoras` | 1,63 | ok | — |
| `\BaseNITYMEDFPSFaixa` | 24,00–25,00 | ok | — |
| `\BaseUTAHoras` | 31,55 | ok | — |
| `\BaseUTAFPSFaixa` | 11,99–30,17 | ok | — |
| `\ExpTresIntraMRLIlumRuimAcBal` | 94,3 | ok | sim |
| `\ExpTresCEWPatchMRLIlumRuimAcBal` | 63,8 | ok | sim |
| `\ExpTresCEWPatchMRLIlumRuimAUC` | 81,1 | ok | sim |
| `\ExpTresIntraMRLIlumBoaAcBal` | 92,6 | ok | sim |
| `\ExpTresCEWPatchMRLIlumBoaAcBal` | 75,1 | ok | sim |
| `\ExpTresCEWPatchMRLIlumBoaAUC` | 89,6 | ok | sim |
| `\ExpTresIntraMRLReflexoGrandeAcBal` | 84,2 | ok | sim |
| `\ExpTresCEWPatchMRLReflexoGrandeAcBal` | 65,7 | ok | sim |
| `\ExpTresCEWPatchMRLReflexoGrandeAUC` | 69,2 | ok | sim |
| `\ExpTresIntraMRLSemReflexoAcBal` | 95,5 | ok | sim |
| `\ExpTresCEWPatchMRLSemReflexoAcBal` | 64,8 | ok | sim |
| `\ExpTresCEWPatchMRLSemReflexoAUC` | 86,9 | ok | sim |
| `\ExpCincoBSemNormAUCJanelaB` | 69,4 | ok | sim |
| `\ExpCincoBSemNormAUCJanelaC` | 67,2 | ok | sim |
| `\ExpCincoBSemNormQuedaAUCJanela` | 2,1 | ok | sim |
| `\ExpCincoBAlertaSUmAUCJanelaC` | 58,6 | ok | sim |
| `\ExpCincoBControleAUCCincoDezB` | 61,5 | ok | sim |
| `\ExpCincoBControleAUCCincoDezC` | 60,2 | ok | sim |
| `\ExpCincoBControleQuedaCincoDez` | 1,2 | ok | sim |
| `\ExpCincoBControleQuedaZeroCinco` | 11,5 | ok | sim |
| `\ExpCincoBMedianaFechMaxSonolentoDMD` | 0,97 | ok | sim |
| `\ExpCincoBMedianaPERCLOSSonolentoDMD` | 0,208 | ok | sim |
| `\ExpCincoBAUCUnivBocejoDMD` | 57,0 | ok | sim |
| `\ExpCincoBMedianaFechMaxSonolentoUTA` | 0,17 | ok | sim |
| `\ExpCincoBMedianaPERCLOSSonolentoUTA` | 0,040 | ok | sim |
| `\ExpCincoBAUCUnivBocejoUTA` | 50,2 | ok | sim |
| `\ExpCincoBMotoristasOculos` | 3 | ok | sim |
| `\ExpCincoBOculosFalhaFUm` | 0,108 | ok | sim |
| `\ExpCincoBOculosFalhaQuedaEAR` | 14,2 | ok | sim |
| `\ExpCincoBOculosFalhaAUCEAR` | 92,8 | ok | sim |
| `\ExpCincoBSemOculosQuedaEARMin` | 30,4 | ok | sim |
| `\ExpCincoBSemOculosQuedaEARMax` | 55,6 | ok | sim |
| `\ExpCincoBQuedaAUCIntraSujeito` | 11,2 | ok | sim |
| `\ExpSeisPERCLOSMicrossono` | 2,9 | ok | sim |
| `\ExpSeisPERCLOSMicrossonoPNoventa` | 19,5 | ok | sim |
| `\ExpSeisPERCLOSUTAAlerta` | 1,4 | ok | sim |
| `\ExpSeisPERCLOSUTAAlertaPNoventa` | 2,4 | ok | sim |
| `\ExpSeisPERCLOSDMDMovimento` | 5,4 | ok | sim |
| `\ExpSeisPERCLOSDMDMovimentoPNoventa` | 18,4 | ok | sim |
| `\MacPWALandmarkerPcinquenta` | 9,6 | ok | — |
| `\MacPWALandmarkerPnoventaecinco` | 11,2 | ok | — |
| `\MacPWACNNPcinquenta` | 3,0 | ok | — |
| `\MacPWACNNPnoventaecinco` | 3,8 | ok | — |
| `\MacPWALSTMPcinquenta` | 0,2 | ok | — |
| `\MacPWALSTMPnoventaecinco` | 0,3 | ok | — |
| `\AppKappaEARSozinho` | 0,831 | ok | sim |
| `\AppDMDVermelhoEARSozinho` | 26,1 | ok | sim |
| `\AppNavegadorSessoesDMD` | 4 | ok | sim |
| `\AppNavegadorVideosNITYMED` | 3 | ok | sim |
| `\ExpSeisDisparoModeloDMDCalUTAAlerta` | 1,7 | ok | sim |
| `\ExpSeisDisparoModeloDMDCalUTASonolento` | 11,0 | ok | parcial |
| `\ExpSeisDisparoModeloDMDCalDMDMovimento` | 21,9 | ok | parcial |
| `\ExpSeisDisparoModeloDMDLRUTAAlerta` | 2,3 | ok | sim |
| `\ExpSeisDisparoModeloDMDLRUTASonolento` | 18,5 | ok | sim |
| `\ExpCincoBLRSemNormQuedaAUCJanela` | 7,3 | ok | sim |
| `\ExpCincoBSensVariantesICPositivo` | 7 | ok | — |
| `\ExpCincoBSensVariantesTotal` | 8 | ok | — |
| `\ExpCincoBSujeitosTreinoB` | 48 | ok | — |
| `\ExpCincoBRefAltAUCJanelaB` | 68,4 | ok | parcial |
| `\ExpCincoBRefAltAUCJanelaC` | 66,1 | ok | parcial |
| `\ExpCincoBRefAltQuedaAUCJanela` | 2,3 | ok | sim |
| `\ExpCincoBRefAltQuedaAUCVideo` | −2,3 | ok | sim |
| `\ExpCincoBBootDuploQuedaAUCJanela` | 10,9 | ok | sim |
| `\ExpCincoBTrezeAUCJanelaB` | 78,6 | ok | sim |
| `\ExpCincoBTrezeQuinhentasAUCJanelaB` | 78,8 | ok | sim |
| `\ExpCincoBVazamentoAcBal` | 12,6 | ok | sim |
| `\ExpCincoBVazamentoAcBalSujeito` | 74,2 | ok | sim |
| `\ExpCincoBVazamentoAcBalAleatoria` | 86,8 | ok | sim |
| `\ExpCincoBVazamentoAUC` | 13,9 | ok | sim |
| `\ExpCincoBVazamentoAUCSujeito` | 80,1 | ok | sim |
| `\ExpCincoBVazamentoAUCAleatoria` | 94,0 | ok | sim |
| `\ExpCincoBOpRegraSensCBruta` | 15,3 | ok | sim |
| `\ExpCincoBOpRegraSensCCentrada` | 16,4 | ok | sim |
| `\ExpCincoBOpRegraSensBCentrada` | 24,2 | ok | sim |
| `\ExpCincoBTrezeQuedaAUCJanela` | 9,3 | ok | — |
| `\ExpCincoBLRRefAltAUCJanelaB` | 73,7 | ok | parcial |
| `\ExpCincoBLRRefAltAUCJanelaC` | 66,2 | ok | parcial |
| `\ExpCincoBLRRefAltQuedaAUCJanela` | 7,6 | ok | sim |
| `\ExpCincoBLRRefAltQuedaAUCVideo` | 8,4 | ok | sim |
| `\ExpCincoBLRBootDuploQuedaAUCJanela` | 9,8 | ok | sim |
| `\ExpCincoBLRTrezeAUCJanelaB` | 71,5 | ok | sim |
| `\ExpCincoBLRTrezeQuinhentasAUCJanelaB` | 71,2 | ok | sim |
| `\ExpCincoBLRVazamentoAcBal` | 1,2 | ok | sim |
| `\ExpCincoBLRVazamentoAcBalSujeito` | 72,0 | ok | sim |
| `\ExpCincoBLRVazamentoAcBalAleatoria` | 73,1 | ok | sim |
| `\ExpCincoBLRVazamentoAUC` | 2,1 | ok | sim |
| `\ExpCincoBLRVazamentoAUCSujeito` | 74,4 | ok | sim |
| `\ExpCincoBLRVazamentoAUCAleatoria` | 76,5 | ok | sim |
| `\ExpCincoBLROpRegraSensCBruta` | 19,2 | ok | sim |
| `\ExpCincoBLROpRegraSensCCentrada` | 24,7 | ok | sim |
| `\ExpCincoBLROpRegraSensBCentrada` | 27,3 | ok | sim |
| `\ExpCincoBLRTrezeQuedaAUCJanela` | 7,0 | ok | — |
| `\ExpCincoBJanelaCincoQuedaAUCJanela` | 11,8 | ok | sim |
| `\ExpCincoBLRJanelaCincoQuedaAUCJanela` | 5,4 | ok | sim |
| `\ExpCincoBLRControleQuedaCincoDez` | 4,6 | ok | sim |
| `\ExpCincoBRefSDoisQuedaAUCJanela` | 9,0 | ok | sim |
| `\ExpCincoBSemNormQuedaAUCVideo` | 0,2 | ok | sim |
| `\ExpCincoBLRRefSDoisQuedaAUCJanela` | 5,4 | ok | sim |
| `\ExpCincoBLRSemNormQuedaAUCVideo` | 1,2 | ok | sim |
| `\ExpCincoBMedianaFechMaxAlertaDMD` | 0,13 | ok | sim |
| `\ExpCincoBMedianaPERCLOSAlertaDMD` | 0,037 | ok | sim |
| `\ExpCincoBMedianaFechMaxAlertaUTA` | 0,07 | ok | sim |
| `\ExpCincoBMedianaPERCLOSAlertaUTA` | 0,010 | ok | sim |
| `\ExpCincoBLRQuedaNormMin` | 5,4 | ok | — |
| `\ExpCincoBLRQuedaNormMax` | 9,8 | ok | — |
| `\ExpCincoBQuedaOutrasNormMin` | 2,1 | ok | — |
| `\ExpCincoBQuedaOutrasNormMax` | 2,3 | ok | — |
| `\AppPesosSUmVermelhoMediana` | 3,3 | ok | — |
| `\AppPesosSUmVermelhoPcinco` | 1,7 | ok | — |
| `\AppPesosSUmVermelhoPnoventaecinco` | 4,9 | ok | — |
| `\AppPesosSCincoVermelhoMediana` | 67,7 | ok | — |
| `\AppPesosSCincoVermelhoPcinco` | 56,2 | ok | — |
| `\AppPesosSCincoVermelhoPnoventaecinco` | 72,9 | ok | — |
| `\AppPesosBoas` | 291 | ok | — |
| `\AppPesosTotal` | 300 | ok | — |
| `\AppPesosBoasPct` | 97 | ok | — |
| `\AppUTASujeitos` | 24 | ok | — |
| `\AppUTAAcordadosAlarme` | 2 | ok | — |
| `\AppUTAAcordadosAmarelo` | 5 | ok | — |
| `\AppUTASonolentosAmarelo` | 17 | ok | — |
| `\AppUTASonolentosVermelho` | 15 | ok | — |
| `\AppUTASonolentosVerdes` | 7 | ok | — |
| `\AppUTAAcordadosPctVermelho` | 0,46 | ok | — |
| `\AppUTAParesMais` | 16 | ok | — |
| `\AppUTAParesEmpate` | 7 | ok | — |
| `\AppUTAParesMenos` | 1 | ok | — |
| `\AppUTAAUCForaVerdeComIC` | 0,812 [0,705; 0,918] | ok | — |
| `\AppUTAAUCForaVerde` | 0,812 | ok | — |
| `\AppUTAAUCForaVerdeCalAcordado` | 0,855 | ok | — |
| `\AppUTAAUCNivelMedioComIC` | 0,839 [0,736; 0,935] | ok | — |
| `\AppUTAAUCNivelMedio` | 0,839 | ok | — |
| `\AppUTAAUCNivelMedioCalAcordado` | 0,891 | ok | — |
| `\AppCompCRF` | 0,811 | ok | sim |
| `\AppCompCLR` | 0,755 | ok | sim |
| `\AppCompBRF` | 0,953 | ok | sim |
| `\AppCompBRFSemNorm` | 0,806 | ok | sim |
| `\AppCompRegra` | 0,853 | ok | sim |
