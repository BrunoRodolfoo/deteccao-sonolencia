# Exp. 5 — Características da rodada final: janelas de 10 s, rótulos v2 do DMD e sequências dos 5 folds

Gerado por `src/features/report_exp5.py` em 27/09/2026, a partir dos CSVs citados em cada seção (nenhum número digitado à mão). Reproduzir tudo a partir dos sinais por quadro (`data/signals/`): `bash src/features/run_features.sh` (no `.venv`). Saídas de dados em `data/features/final/<base>/`; logs em `results/logs/exp5_final_*.log`. Testes: `src/features/test_features.py` (19 testes de asserts puros, todos passaram; `results/logs/exp5_final_testes.log`). A rodada preliminar (janelas de 30 s, só o Fold 1 da UTA-RLDD e só a s5 do DMD) foi descartada; seus arquivos foram removidos em 06/10/2026.

## 1. Números-chave

- **Sinais conferidos nos `index.csv`** (`results/exp5_contagens_sinais.csv`): DMD 65 sessões de 14 sujeitos (s5 16, s1 17, s2 16, s3 16); UTA-RLDD 182 arquivos = 180 vídeos lógicos de 60 sujeitos (5 folds); NITYMED 126 vídeos. Vídeos em disco: DMD 65, NITYMED 126, UTA-RLDD 182 — todos com sinal completo (nenhuma extração foi necessária nesta frente).
- **DMD, janelas de 10 s (passo 5 s)** (`data/features/final/dmd/windows_10s.csv`): principal = **87 alerta** (safe_drive da s2) × **423 sonolento atuado** (s5 com fechamento anotado ≥ 0,5 s ou bocejo), 13 sujeitos com as duas classes; fora do principal: 146 janelas da s5 sem sinal anotado (sensibilidade), 109 alerta da s1 (sensibilidade), 62 de referência (s3; nunca avaliadas). Na rodada preliminar (30 s, só s5, rótulo anotado, sem as janelas de referência) eram 18 alerta × 471 sonolento, com as alerta sobrepostas à referência.
- **UTA-RLDD, janelas de 10 s** (`data/features/final/uta/windows_10s.csv`): 22.455 janelas; avaliáveis KSS 0 / 5 / 10 = 4.906 / 7.468 / 7.581; 2.380 de referência (último terço do vídeo 0) e 120 de fronteira (cruzam o corte de 2/3), fora de treino e teste.
- **Referência de normalização por sujeito, separada da avaliação**: UTA 29–58 janelas por sujeito (mediana 39); DMD 3–9 janelas da s3 por sujeito; referência na s2 (sem s3 ou s3 com < 3 janelas): gE_28 (2 janelas). Por sujeito: seções 4.5 e 5.2.
- **z-score por sujeito continua não estimável** em parte das características (desvio 0 ou < 5 janelas na referência): no DMD o z é definido em 18,2% das células (unidade × característica) e na UTA em 70,5%; na UTA só `ear_media`, `ear_dp` têm z em todos os sujeitos. **A centragem (`<f>_c`) segue como normalização principal** (seção 5.3).
- **Alinhamento anotação ↔ vídeo nas 49 sessões de distração**: nº de quadros = `total_frames` em 49/49, `frame_shift` = 0 em 49/49; olhar fora da estrada × desvio da pose da cabeça: melhor defasagem mediana 2 quadros (38 de 40 sessões com correlação ≥ 0,2 entre 0 e 5 quadros; máximo 18); fala × atividade da boca: −11 a 1 quadros. **Deslocamento adotado: 0** (como na s5).
- **Mapeamento sujeito oficial ↔ pasta nos 5 folds** (`results/exp5_uta_mapeamento_oficial.csv`): 47 confirmados pela coincidência das piscadas oficiais com o nosso EAR, 7 prováveis (ordem de listagem ou duração) e 4 indeterminados (todos no Fold 3), de 58 sujeitos oficiais. **Pastas ausentes das características oficiais: F1_12 (Fold 1) e F5_49 (Fold 5)** (seção 5.4).
- **Sequências de 30 piscadas (passo 2)** (`data/features/final/uta/sequences.npz`): 8.843 sequências de 60 sujeitos e 178 dos 180 vídeos lógicos (sem nenhuma piscada detectada, logo sem sequência: F2_15_10, F3_27_5; seção 5.5); alerta / pouca vigilância / sonolento = 1.021 / 3.470 / 4.352.

## 2. Sinais disponíveis (conferência dos `index.csv`)

Fonte: `data/signals/<base>/index.csv` → `results/exp5_contagens_sinais.csv` (só arquivos com `complete` = True e `.npz` presente). Duração e fps do contêiner; taxa de rosto = fração de quadros com landmarks do Face Mesh.

| Base | Grupo | Arquivos | Vídeos lógicos | Sujeitos | Quadros | Horas | Taxa de rosto (média) | Taxa de rosto (mín.) | fps |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DMD | s1 (distração; carro em movimento) | 17 | 17 | 14 | 118.086 | 1,10 | 0,9880 | 0,9406 | 29,76 |
| DMD | s2 (distração; carro parado) | 16 | 16 | 14 | 218.342 | 2,04 | 0,9846 | 0,9262 | 29,76 |
| DMD | s3 (distração; carro parado, uma atividade) | 16 | 16 | 13 | 29.435 | 0,27 | 0,9667 | 0,9151 | 29,76 |
| DMD | s5 (sonolência atuada; carro parado) | 16 | 16 | 13 | 88.222 | 0,82 | 0,9978 | 0,9832 | 29,76 |
| DMD | **total** | 65 | 65 | 14 | 454.085 | 4,24 | 0,9843 | 0,9151 | 29,76 |
| NITYMED | Microsleep | 19 | 19 | 19 | 58.328 | 0,65 | 0,9923 | 0,9398 | 25,00 |
| NITYMED | Yawning | 107 | 107 | 107 | 88.156 | 0,98 | 0,9914 | 0,8737 | 24,00–25,00 |
| NITYMED | **total** | 126 | 126 | 126 | 146.484 | 1,63 | 0,9915 | 0,8737 | 24,00–25,00 |
| UTA-RLDD | Fold 1 | 36 | 36 | 12 | 616.732 | 5,99 | 0,9995 | 0,9955 | 15,03–30,17 |
| UTA-RLDD | Fold 2 | 36 | 36 | 12 | 602.622 | 6,24 | 0,9976 | 0,9595 | 12,00–30,00 |
| UTA-RLDD | Fold 3 | 37 | 36 | 12 | 614.543 | 6,51 | 0,9993 | 0,9916 | 11,99–30,17 |
| UTA-RLDD | Fold 4 | 36 | 36 | 12 | 601.222 | 6,28 | 0,9994 | 0,9923 | 12,01–30,02 |
| UTA-RLDD | Fold 5 | 37 | 36 | 12 | 684.952 | 6,54 | 0,9907 | 0,6669 | 16,04–30,02 |
| UTA-RLDD | **total** | 182 | 180 | 60 | 3.120.071 | 31,55 | 0,9973 | 0,6669 | 11,99–30,17 |

UTA-RLDD: os arquivos a mais em relação aos 180 vídeos são as partes dos dois vídeos sonolentos divididos: `F3_32_10_p1` (421 s, 20,69 fps, taxa de rosto 0,999); `F3_32_10_p2` (183 s, 20,73 fps, taxa de rosto 1,000); `F5_49_10_p1` (318 s, 29,67 fps, taxa de rosto 1,000); `F5_49_10_p2` (1.171 s, 29,67 fps, taxa de rosto 0,667). Resolução original das partes: `F3_32_10_p1` 1080×1920; `F3_32_10_p2` 1080×1920; `F5_49_10_p1` 270×480; `F5_49_10_p2` 162×288 (os demais vídeos do mesmo sujeito têm 1080×1920). DMD: sujeitos sem s5: gE_28; sem s3: gE_28.

## 3. Parâmetros e decisões desta rodada

| Item | Decisão | Por quê |
|---|---|---|
| Janela | **10 s, passo 5 s** nas três bases (`--janela`, `--passo`); 5 s e 15 s para sensibilidade | decisão de 27/09 (os trechos safe_drive do DMD têm 10–26 s; 30 s não cabem) |
| Grade | UTA, NITYMED e DMD s5: grade global a partir de 0 s, por arquivo (partes _p1/_p2 separadas; nenhuma janela atravessa o corte; coluna `video` = vídeo lógico, `arquivo` = parte). DMD s1/s2/s3: janelas **ancoradas no início de cada trecho safe_drive** e inteiramente dentro dele | numa grade global a janela só cabe no trecho quando a grade coincide com ele; ancorar dá 261 janelas safe_drive em 10 s contra 155 na grade global |
| Características | as 11 da rodada preliminar, só com quadros com rosto e em unidades independentes do fps (`perclos`, `taxa_piscadas_min`, `dur_media_piscada_s`, `dur_max_fechamento_s`, `n_fechamentos_05s`, `ear_media`, `ear_dp`, `amp_media`, `vel_media`, `frac_bocejo`, `taxa_rosto`); `n_piscadas` como coluna auxiliar | em 10 s a taxa de piscadas é ruidosa (poucas piscadas por janela) |
| Referência (normalização) | UTA: janelas inteiramente no **último terço do vídeo 0**; DMD: janelas safe_drive das **sessões s3** do sujeito (se < 3 janelas: último terço, por sessão, das janelas safe_drive da s2); NITYMED: janelas no **primeiro terço** do vídeo (pseudo-referência = o terço inteiro quando nenhuma janela cabe nele) | regra final de 27/09; a referência nunca é avaliada |
| Exclusões | `janela_ref` (referência) e `sobrepoe_ref` (fronteira: divide quadros com a referência) nunca entram em treino/teste; coluna `avaliavel` marca o que entra no principal | correção do vazamento da rodada preliminar (alerta do DMD sobreposto à referência) |
| Limiar de EAR | 0,75 × mediana do EAR (quadros com rosto) do **mesmo trecho de referência**, por sujeito, aplicado a todos os arquivos do sujeito: UTA último terço do vídeo 0 (antes: vídeo 0 inteiro); DMD safe_drive das s3 (antes: primeiros 30 s de cada s5); NITYMED primeiro terço | o limiar também é calibração por sujeito: calculá-lo com as janelas alerta avaliadas era um vazamento menor do mesmo tipo |
| Normalização principal | `<f>_c` = x − média das janelas de referência da unidade; `<f>_z` só com ≥ 5 janelas de referência e desvio > 0 | o z segue indefinido em várias características (seção 5.3) |
| Rótulos DMD (v2) | `uso`: `alerta` (s2 safe_drive), `sonolento` (s5 com fechamento anotado ≥ 0,5 s ou bocejo, ≥ 0,5 s dentro da janela), `s5_sem_sinal`, `alerta_s1`, `referencia`, `fronteira`, `s3_nao_usada`; `y` (principal), `y_sens_s1`, `y_sens_s5_sem_sinal`; sujeito sem s5 (gE_28): `y` = NaN | item 9 das notas de fechamento |
| Rótulos UTA | do vídeo: `rotulo_kss`, `classe` 0/1/2, B1, B2, B3 (B3 principal) | iguais à rodada preliminar |
| Sequências | 30 piscadas, passo 2; z-score por sujeito com o último terço das piscadas do vídeo 0 (`--terco primeiro` como sensibilidade); partes concatenadas com contagem contínua | como o Preprocessing.py original |

## 4. DMD

### 4.1 Sessões por sujeito: datas, montagem e óculos

Fonte: `data/features/final/dmd/blinks_index.csv` (metadados do OpenLABEL: `recordTime`, `setup`, `glasses`).

| Sujeito | s1 (movimento) | s2 (parado) | s3 (parado) | s5 (sonolência) | s2 no mesmo dia da s5? |
|---|---|---|---|---|---|
| gA_1 | 08/03 09h31 óculos | 08/03 09h21 óculos | 14/03 14h31 óculos | 14/03 14h26 óculos | não |
| gA_5 | 08/03 10h57 | 08/03 10h46 | 13/03 09h10 | 13/03 09h06 | não |
| gB_10 | 11/03 15h24 (**Car Stopped**) | 11/03 15h15 | 12/03 10h40; 13/03 14h22 | 12/03 10h35; 13/03 14h17 | não |
| gB_6 | 11/03 13h55 | 11/03 13h46 (**Car Moving**) | 13/03 13h41 | 13/03 13h37 | não |
| gB_7 | 11/03 14h22 | 11/03 14h12 | 13/03 13h59 | 13/03 13h55 | não |
| gB_9 | 07/03 16h36 | 07/03 16h21 | 07/03 16h29 | 07/03 16h31 | sim |
| gC_13 | 04/03 10h26 | 04/03 10h11 | 12/03 10h06 | 12/03 10h03 | não |
| gC_14 | 04/03 11h56 | 04/03 11h48 | 12/03 09h23 | 12/03 09h18 | não |
| gE_28 | 15/03 10h23 | 15/03 10h12 | — | — | — (sem s5) |
| gE_29 | 15/03 13h58 óculos | 15/03 13h42 óculos | 15/03 13h54 óculos | 15/03 13h51 óculos | sim |
| gF_23 | 04/03 16h21; 05/03 09h23; 05/03 10h17 | 04/03 16h09; 05/03 10h09 | 11/03 09h31; 14/03 13h54 | 11/03 10h19; 14/03 13h49 | não |
| gZ_33 | 08/04 10h08 (**Car Stopped**) óculos; 08/04 16h18 (**Car Stopped**) | 08/04 09h59 óculos; 08/04 16h09 | 04/04 09h33; 04/04 15h06 | 04/04 09h29; 04/04 15h02 | não |
| gZ_36 | 09/04 10h55 óculos | 09/04 10h39 óculos | 09/04 10h53 óculos | 09/04 10h49 óculos | sim |
| gZ_37 | 08/04 15h56 | 08/04 15h45 | 29/04 12h10 | 29/04 12h06 | não |

**Estrutura temporal (fato dos metadados, relevante para o Exp. 5b):** a s3 (referência) foi gravada no mesmo dia e poucos minutos antes ou depois da s5 (sonolento) em 13 de 13 sujeitos com s5; a s2 (exemplos alerta), em outro dia em 10 de 13. Assim, na normalização pela s3, as janelas sonolentas são centradas numa referência da mesma sessão de gravação e as alerta numa de outro dia (outra montagem da câmera, iluminação, óculos): diferenças de montagem entram na classe alerta. É o espelho do que ocorre na UTA, onde a referência sai do mesmo vídeo das janelas alerta. Metadados `setup` que contrariam a convenção do README (s1 = em movimento; s2/s3/s5 = parado) estão em **negrito** na tabela: gB_10 s1 (11/03 15h24): 'Car Stopped'; gB_6 s2 (11/03 13h46): 'Car Moving'; gZ_33 s1 (08/04 10h08): 'Car Stopped'; gZ_33 s1 (08/04 16h18): 'Car Stopped'. Sujeitos com óculos em só parte das sessões: gZ_33 (óculos em: s1 08/04 10h08, s2 08/04 09h59).

### 4.2 Limiar de EAR por sujeito e nível do EAR em cada sessão

Limiar = 0,75 × mediana do EAR no safe_drive das s3 do sujeito (`ear_ref_mediana`). Para mostrar quanto o nível de olho aberto muda entre sessões, a coluna ao lado dá a mediana do EAR nos primeiros 30 s de cada sessão (a referência da rodada preliminar), dividida pela mediana de referência.

| Sujeito | Referência do limiar | Quadros de referência | Mediana do EAR (ref.) | Limiar | Mediana dos 30 s iniciais ÷ ref., por sessão |
|---|---|---:|---:|---:|---|
| gA_1 | safe_drive das s3 | 1.312 | 0,256 | 0,192 | s1 0,81; s2 0,98; s3 0,94; s5 0,82 |
| gA_5 | safe_drive das s3 | 1.449 | 0,232 | 0,174 | s1 0,70; s2 0,94; s3 1,03; s5 0,95 |
| gB_10 | safe_drive das s3 | 2.636 | 0,270 | 0,203 | s1 0,92; s2 1,00; s3 0,96; s3 0,99; s5 0,95; s5 0,92 |
| gB_6 | safe_drive das s3 | 1.424 | 0,289 | 0,217 | s1 0,92; s2 1,02; s3 1,00; s5 1,01 |
| gB_7 | safe_drive das s3 | 1.262 | 0,251 | 0,189 | s1 0,98; s2 0,92; s3 1,00; s5 0,87 |
| gB_9 | safe_drive das s3 | 1.104 | 0,234 | 0,176 | s1 1,00; s2 1,11; s3 1,02; s5 1,08 |
| gC_13 | safe_drive das s3 | 1.377 | 0,298 | 0,223 | s1 0,89; s2 1,04; s3 0,99; s5 0,99 |
| gC_14 | safe_drive das s3 | 1.249 | 0,302 | 0,227 | s1 0,84; s2 0,98; s3 1,01; s5 0,89 |
| gE_28 | último terço do safe_drive das s2 | 1.184 | 0,295 | 0,221 | s1 0,89; s2 0,96 |
| gE_29 | safe_drive das s3 | 1.195 | 0,275 | 0,206 | s1 0,94; s2 1,02; s3 1,00; s5 1,03 |
| gF_23 | safe_drive das s3 | 2.679 | 0,270 | 0,203 | s1 1,01; s1 1,01; s1 1,00; s2 1,11; s2 1,09; s3 1,05; s3 0,97; s5 1,00; s5 0,99 |
| gZ_33 | safe_drive das s3 | 2.605 | 0,278 | 0,209 | s1 1,00; s1 1,03; s2 1,04; s2 0,99; s3 1,02; s3 1,03; s5 1,00; s5 0,94 |
| gZ_36 | safe_drive das s3 | 1.128 | 0,250 | 0,188 | s1 1,06; s2 1,10; s3 1,01; s5 1,05 |
| gZ_37 | safe_drive das s3 | 1.041 | 0,221 | 0,166 | s1 0,99; s2 1,20; s3 0,98; s5 1,08 |

A razão vai de 0,70 a 1,20 (mediana 1,00; na s1, com o carro em movimento, mediana 0,98): o nível de olho aberto varia entre sessões do mesmo sujeito, o que desloca PERCLOS e contagem de piscadas quando o limiar vem de outra sessão — efeito inerente à calibração por sujeito (o mesmo de F1_03_10 na UTA).

### 4.3 Balanço de classes (janelas de 10 s)

Fonte: `data/features/final/dmd/windows_10s.csv` (coluna `uso`) e `results/exp5_dmd_balanco_janelas.csv`. Principal = `alerta` × `sonolento` (y); `s5_sem_sinal` e `alerta_s1` são só de sensibilidade; `referencia` nunca é avaliada.

| Sujeito | `alerta` | `sonolento` | `s5_sem_sinal` | `alerta_s1` | `referencia` | `fronteira` | `s3_nao_usada` | No principal (y) | Fração alerta no principal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gA_1 | 8 | 26 | 9 | 6 | 4 | 0 | 0 | 34 | 23,5% |
| gA_5 | 8 | 25 | 9 | 12 | 4 | 0 | 0 | 33 | 24,2% |
| gB_10 | 5 | 59 | 15 | 2 | 9 | 0 | 0 | 64 | 7,8% |
| gB_6 | 5 | 20 | 15 | 6 | 4 | 0 | 0 | 25 | 20,0% |
| gB_7 | 7 | 28 | 7 | 6 | 4 | 0 | 0 | 35 | 20,0% |
| gB_9 | 8 | 25 | 9 | 9 | 3 | 0 | 0 | 33 | 24,2% |
| gC_13 | 7 | 22 | 13 | 6 | 4 | 0 | 0 | 29 | 24,1% |
| gC_14 | 2 | 21 | 15 | 2 | 4 | 0 | 0 | 23 | 8,7% |
| gE_28 | 3 | 0 | 0 | 7 | 2 | 0 | 0 | 0 | — |
| gE_29 | 6 | 30 | 5 | 6 | 3 | 0 | 0 | 36 | 16,7% |
| gF_23 | 15 | 57 | 20 | 26 | 9 | 0 | 0 | 72 | 20,8% |
| gZ_33 | 5 | 61 | 10 | 7 | 5 | 0 | 0 | 66 | 7,6% |
| gZ_36 | 6 | 25 | 10 | 8 | 4 | 0 | 0 | 31 | 19,4% |
| gZ_37 | 5 | 24 | 9 | 6 | 3 | 0 | 0 | 29 | 17,2% |
| **total** | 90 | 423 | 146 | 109 | 62 | 0 | 0 | 510 | 17,1% |

gE_28 não tem s5: suas janelas `alerta` ficam com y = NaN (fora do treino pareado), como decidido; a referência dele é a s2 (não há s3).

**Por sessão** (cada linha é um arquivo; só as colunas com janelas):

| Sujeito | Sessão | Gravação | Janelas | Por uso |
|---|---|---|---:|---|
| gA_1 | s1 | 08/03 09h31 | 6 | alerta_s1 6 |
| gA_1 | s2 | 08/03 09h21 | 8 | alerta 8 |
| gA_1 | s3 | 14/03 14h31 | 4 | referencia 4 |
| gA_1 | s5 | 14/03 14h26 | 35 | sonolento 26, s5_sem_sinal 9 |
| gA_5 | s1 | 08/03 10h57 | 12 | alerta_s1 12 |
| gA_5 | s2 | 08/03 10h46 | 8 | alerta 8 |
| gA_5 | s3 | 13/03 09h10 | 4 | referencia 4 |
| gA_5 | s5 | 13/03 09h06 | 34 | sonolento 25, s5_sem_sinal 9 |
| gB_10 | s1 | 11/03 15h24 | 2 | alerta_s1 2 |
| gB_10 | s2 | 11/03 15h15 | 5 | alerta 5 |
| gB_10 | s3 | 12/03 10h40 | 4 | referencia 4 |
| gB_10 | s3 | 13/03 14h22 | 5 | referencia 5 |
| gB_10 | s5 | 12/03 10h35 | 39 | sonolento 31, s5_sem_sinal 8 |
| gB_10 | s5 | 13/03 14h17 | 35 | sonolento 28, s5_sem_sinal 7 |
| gB_6 | s1 | 11/03 13h55 | 6 | alerta_s1 6 |
| gB_6 | s2 | 11/03 13h46 | 5 | alerta 5 |
| gB_6 | s3 | 13/03 13h41 | 4 | referencia 4 |
| gB_6 | s5 | 13/03 13h37 | 35 | sonolento 20, s5_sem_sinal 15 |
| gB_7 | s1 | 11/03 14h22 | 6 | alerta_s1 6 |
| gB_7 | s2 | 11/03 14h12 | 7 | alerta 7 |
| gB_7 | s3 | 13/03 13h59 | 4 | referencia 4 |
| gB_7 | s5 | 13/03 13h55 | 35 | sonolento 28, s5_sem_sinal 7 |
| gB_9 | s1 | 07/03 16h36 | 9 | alerta_s1 9 |
| gB_9 | s2 | 07/03 16h21 | 8 | alerta 8 |
| gB_9 | s3 | 07/03 16h29 | 3 | referencia 3 |
| gB_9 | s5 | 07/03 16h31 | 34 | sonolento 25, s5_sem_sinal 9 |
| gC_13 | s1 | 04/03 10h26 | 6 | alerta_s1 6 |
| gC_13 | s2 | 04/03 10h11 | 7 | alerta 7 |
| gC_13 | s3 | 12/03 10h06 | 4 | referencia 4 |
| gC_13 | s5 | 12/03 10h03 | 35 | sonolento 22, s5_sem_sinal 13 |
| gC_14 | s1 | 04/03 11h56 | 2 | alerta_s1 2 |
| gC_14 | s2 | 04/03 11h48 | 2 | alerta 2 |
| gC_14 | s3 | 12/03 09h23 | 4 | referencia 4 |
| gC_14 | s5 | 12/03 09h18 | 36 | sonolento 21, s5_sem_sinal 15 |
| gE_28 | s1 | 15/03 10h23 | 7 | alerta_s1 7 |
| gE_28 | s2 | 15/03 10h12 | 5 | alerta 3, referencia 2 |
| gE_29 | s1 | 15/03 13h58 | 6 | alerta_s1 6 |
| gE_29 | s2 | 15/03 13h42 | 6 | alerta 6 |
| gE_29 | s3 | 15/03 13h54 | 3 | referencia 3 |
| gE_29 | s5 | 15/03 13h51 | 35 | sonolento 30, s5_sem_sinal 5 |
| gF_23 | s1 | 04/03 16h21 | 12 | alerta_s1 12 |
| gF_23 | s1 | 05/03 09h23 | 9 | alerta_s1 9 |
| gF_23 | s1 | 05/03 10h17 | 5 | alerta_s1 5 |
| gF_23 | s2 | 04/03 16h09 | 11 | alerta 11 |
| gF_23 | s2 | 05/03 10h09 | 4 | alerta 4 |
| gF_23 | s3 | 11/03 09h31 | 5 | referencia 5 |
| gF_23 | s3 | 14/03 13h54 | 4 | referencia 4 |
| gF_23 | s5 | 11/03 10h19 | 40 | sonolento 29, s5_sem_sinal 11 |
| gF_23 | s5 | 14/03 13h49 | 37 | sonolento 28, s5_sem_sinal 9 |
| gZ_33 | s1 | 08/04 10h08 | 6 | alerta_s1 6 |
| gZ_33 | s1 | 08/04 16h18 | 1 | alerta_s1 1 |
| gZ_33 | s2 | 08/04 09h59 | 3 | alerta 3 |
| gZ_33 | s2 | 08/04 16h09 | 2 | alerta 2 |
| gZ_33 | s3 | 04/04 09h33 | 3 | referencia 3 |
| gZ_33 | s3 | 04/04 15h06 | 2 | referencia 2 |
| gZ_33 | s5 | 04/04 09h29 | 35 | sonolento 31, s5_sem_sinal 4 |
| gZ_33 | s5 | 04/04 15h02 | 36 | sonolento 30, s5_sem_sinal 6 |
| gZ_36 | s1 | 09/04 10h55 | 8 | alerta_s1 8 |
| gZ_36 | s2 | 09/04 10h39 | 6 | alerta 6 |
| gZ_36 | s3 | 09/04 10h53 | 4 | referencia 4 |
| gZ_36 | s5 | 09/04 10h49 | 35 | sonolento 25, s5_sem_sinal 10 |
| gZ_37 | s1 | 08/04 15h56 | 6 | alerta_s1 6 |
| gZ_37 | s2 | 08/04 15h45 | 5 | alerta 5 |
| gZ_37 | s3 | 29/04 12h10 | 3 | referencia 3 |
| gZ_37 | s5 | 29/04 12h06 | 33 | sonolento 24, s5_sem_sinal 9 |

### 4.4 Sensibilidade ao comprimento da janela (5, 10 e 15 s; passo 5 s)

| Janela | Janelas | Alerta (y=0) | Sonolento (y=1) | s5 sem sinal | Alerta s1 | Referência | Sujeitos com 2 classes | Referência na s2 (fallback) | Safe_drive ancoradas | Safe_drive na grade global |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 5 s | 1.220 | 237 | 308 | 277 | 275 | 113 | 13 | gE_28 | 635 | 442 |
| 10 s | 830 | 87 | 423 | 146 | 109 | 62 | 13 | gE_28 | 261 | 155 |
| 15 s | 627 | 8 | 466 | 87 | 34 | 17 | 5 | gA_1, gA_5, gB_6, gB_7, gB_9, gC_13, gC_14, gE_28, gE_29, gZ_33, gZ_36, gZ_37 | 74 | 39 |

Com 15 s quase não há janela safe_drive (os trechos têm 10–26 s): a s3 deixa de ter 3 janelas em quase todos os sujeitos, a referência passa para a s2 e sobram poucas janelas alerta — a sensibilidade de 15 s no DMD é pouco informativa. Com 5 s o balanço melhora, mas as janelas são curtas demais para fechamentos longos e taxa de piscadas.

### 4.5 Janelas de referência por sujeito

| Sujeito | 5 s | 10 s | 15 s |
|---|---:|---:|---:|
| gA_1 | 7 (s3) | 4 (s3) | 1 (s2) |
| gA_5 | 7 (s3) | 4 (s3) | 0 (s2) |
| gB_10 | 15 (s3) | 9 (s3) | 4 (s3) |
| gB_6 | 7 (s3) | 4 (s3) | 1 (s2) |
| gB_7 | 7 (s3) | 4 (s3) | 1 (s2) |
| gB_9 | 6 (s3) | 3 (s3) | 1 (s2) |
| gC_13 | 7 (s3) | 4 (s3) | 1 (s2) |
| gC_14 | 7 (s3) | 4 (s3) | 0 (s2) |
| gE_28 | 5 (s2) | 2 (s2) | 1 (s2) |
| gE_29 | 6 (s3) | 3 (s3) | 1 (s2) |
| gF_23 | 15 (s3) | 9 (s3) | 3 (s3) |
| gZ_33 | 13 (s3) | 5 (s3) | 1 (s2) |
| gZ_36 | 6 (s3) | 4 (s3) | 1 (s2) |
| gZ_37 | 5 (s3) | 3 (s3) | 1 (s2) |

Fonte: `data/features/final/dmd/windows_referencia_<L>s.csv` (média e desvio de referência de cada característica). Em 10 s, 11 dos 14 sujeitos têm < 5 janelas de referência (o z exige ≥ 5); com ≥ 5: gB_10, gF_23, gZ_33.

### 4.6 Alinhamento anotação ↔ `*_rgb_face.mp4`

**s5 (sonolência, 16 sessões)** — `results/exp5_dmd_alinhamento.csv`, mesmo método da rodada preliminar: nº de quadros = `total_frames` em 16/16; `frame_shift` = 0 em 16/16; melhor defasagem entre o fechamento anotado e −EAR de −3 a 2 quadros, com ganho de correlação sobre o lag 0 de no máximo 0,015. A verificação independente da rodada preliminar (±300 quadros e inspeção visual; `results/exp5_dmd_alinhamento_verificacao.csv`) continua válida (mesmos vídeos).

**Distração (s1/s2/s3, 49 sessões)** — `results/exp5_dmd_alinhamento_distracao.csv`. A anotação de distração **não tem estado do olho nem piscada**; o sinal ocular disponível é o olhar (`gaze_on_road/not_looking_road`). Conferências: (i) contagem de quadros: nº de quadros do vídeo = `total_frames` em 49/49, `frame_shift` da câmera do rosto = 0 em 49/49, e o fim das ações passa do último quadro em no máximo 2 quadro(s) (11 sessões; a s5 tem a mesma convenção); (ii) correlação cruzada do indicador de olhar fora da estrada com o **desvio da pose da cabeça** (|yaw| e |pitch| padronizados) e com **−EAR**; (iii) da fala anotada (`talking/talking`) com a **atividade da boca** (DP móvel de 0,5 s do MAR).

| Par | Sessões com o evento | Sessões com r ≥ 0,2 | r no lag 0 (mediana) | Melhor lag (±30), sessões com r ≥ 0,2 | Melhor lag (±300) | Ganho máx. de r sobre o lag 0 |
|---|---:|---:|---:|---:|---:|---:|
| olhar fora da estrada × desvio da pose | 48 | 40 | 0,393 | 0–18 | 0–18 | 0,042 |
| fala × atividade da boca | 37 | 37 | 0,552 | −11 a 1 | −11 a 1 | 0,180 |
| olhar fora da estrada × −EAR | 48 | 13 | 0,031 | −1 a 1 | −1 a 1 | 0,016 |

A pose da cabeça e a boca acompanham a anotação com defasagem de poucos quadros (a 29,76 fps, 3 quadros ≈ 0,1 s): olhar × pose com mediana de 2 quadros (38 de 40 sessões entre 0 e 5; fora disso: gE_28 s2 18 quadros (r = 0,21); gZ_37 s3 16 quadros (r = 0,50)) — a pose atrasa um pouco porque o olhar muda antes da cabeça, e o DP móvel centrado adianta a fala. O −EAR só acompanha o olhar em 13 das 48 sessões (r ≥ 0,2; nelas o melhor lag vai de −1 a 1 quadros); nas demais a correlação é fraca ou negativa (20 com r < 0 no lag 0). Não há sinal de deslocamento: **índice da anotação = índice do quadro do vídeo do rosto também nas sessões de distração**. As janelas alerta dependem só dos trechos safe_drive, que duram segundos — um erro de poucos quadros não mudaria o rótulo.

### 4.7 Validação do detector de piscadas (s5) com o limiar novo

| Limiar | P piscadas | R piscadas | F1 piscadas | F1 fech. ≥ 0,5 s | Acur. bal. por quadro | Kappa por quadro | R bocejo |
|---|---:|---:|---:|---:|---:|---:|---:|
| final (limiar da s3 do sujeito) | 0,811 | 0,775 | 0,749 | 0,711 | 0,900 | 0,773 | 0,347 |
| preliminar (limiar dos 30 s iniciais da própria s5) | 0,806 | 0,788 | 0,766 | 0,728 | 0,903 | 0,798 | 0,347 |

Médias por sessão (16 sessões s5; `results/exp5_dmd_validacao_detector.csv`, a linha “preliminar” vem da rodada descartada de 27/09). Sensibilidade ao fator k (× a mesma mediana de referência; `results/exp5_dmd_sensibilidade_limiar.csv`): F1 de piscadas k = 0,60: 0,533, k = 0,65: 0,610, k = 0,70: 0,690, k = 0,75: 0,749, k = 0,80: 0,775, k = 0,85: 0,772, k = 0,90: 0,747. k = 0,75 mantido (pré-especificado).

## 5. UTA-RLDD

### 5.1 Janelas de 10 s por fold e classe

| Fold | Sujeitos | Vídeos lógicos | Arquivos | Janelas | Avaliáveis KSS 0 | KSS 5 | KSS 10 | Referência | Fronteira |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Fold 1 | 12 | 36 | 36 | 4.260 | 954 | 1.419 | 1.400 | 463 | 24 |
| Fold 2 | 12 | 36 | 36 | 4.441 | 962 | 1.483 | 1.507 | 465 | 24 |
| Fold 3 | 12 | 36 | 37 | 4.631 | 1.042 | 1.572 | 1.483 | 510 | 24 |
| Fold 4 | 12 | 36 | 36 | 4.469 | 975 | 1.469 | 1.530 | 471 | 24 |
| Fold 5 | 12 | 36 | 37 | 4.654 | 973 | 1.525 | 1.661 | 471 | 24 |
| **total** | 60 | 180 | 182 | 22.455 | 4.906 | 7.468 | 7.581 | 2.380 | 120 |

B3 (principal): 4.906 alerta × 7.581 sonolento; B1: 4.906 × 15.049; B2: 12.374 × 7.581. As partes dos vídeos divididos geram janelas próprias (nenhuma atravessa o corte) e compartilham a coluna `video` (vídeo lógico) — para métricas por vídeo, agrupar por `video`, não por `arquivo`.

### 5.2 Referência por sujeito (último terço do vídeo 0) e limiar

Janelas de referência por sujeito em 10 s: mínimo 29, mediana 39, máximo 58 (60 sujeitos; `data/features/final/uta/windows_referencia_10s.csv`, uma linha por sujeito). As janelas avaliadas do vídeo 0 terminam antes do corte de 2/3; a janela que o cruza é `fronteira`. O limiar de EAR passou a sair do mesmo último terço: a mediana de referência mudou por um fator de 0,955 a 1,071 (mediana 0,999) em relação ao vídeo 0 inteiro. Em 5 s / 15 s: 5 s: 30–59; 15 s: 28–57.

Janelas de referência (10 s) por sujeito:

| Fold | Pasta: nº de janelas de referência |
|---|---|
| Fold 1 | 01: 38; 02: 38; 03: 44; 04: 39; 05: 39; 06: 39; 07: 38; 08: 39; 09: 39; 10: 34; 11: 40; 12: 36 |
| Fold 2 | 13: 39; 14: 39; 15: 41; 16: 29; 17: 38; 18: 38; 19: 38; 20: 41; 21: 39; 22: 38; 23: 37; 24: 48 |
| Fold 3 | 25: 38; 26: 39; 27: 42; 28: 38; 29: 41; 30: 44; 31: 44; 32: 58; 33: 37; 34: 49; 35: 40; 36: 40 |
| Fold 4 | 37: 44; 38: 38; 39: 41; 40: 38; 41: 40; 42: 38; 43: 38; 44: 39; 45: 39; 46: 38; 47: 38; 48: 40 |
| Fold 5 | 49: 39; 50: 39; 51: 39; 52: 38; 53: 38; 54: 40; 55: 45; 56: 39; 57: 40; 58: 38; 59: 38; 60: 38 |

### 5.3 z-score por sujeito: onde é estimável (janelas de 10 s)

Fração das unidades de referência (UTA e DMD: sujeitos; NITYMED: vídeos) em que o z de cada característica é definido (≥ 5 janelas de referência e desvio > 0). Fonte: `data/features/final/<base>/windows_referencia_10s.csv`.

| Característica | UTA-RLDD (60) | DMD (14) | NITYMED (126) |
|---|---:|---:|---:|
| `perclos` | 97% | 21% | 17% |
| `taxa_piscadas_min` | 97% | 21% | 16% |
| `dur_media_piscada_s` | 83% | 21% | 12% |
| `dur_max_fechamento_s` | 97% | 21% | 17% |
| `n_fechamentos_05s` | 25% | 7% | 7% |
| `ear_media` | 100% | 21% | 17% |
| `ear_dp` | 100% | 21% | 17% |
| `amp_media` | 85% | 21% | 12% |
| `vel_media` | 85% | 21% | 12% |
| `frac_bocejo` | 2% | 0% | 0% |
| `taxa_rosto` | 5% | 21% | 9% |

Mesmo com mais janelas de referência (10 s em vez de 30 s), o z só existe em TODAS as unidades da UTA para `ear_media`, `ear_dp`; `n_fechamentos_05s`, `frac_bocejo`, `taxa_rosto` ficam indefinidas na maioria dos sujeitos (desvio 0: quase sempre constantes numa referência alerta) e, no DMD, 11 dos 14 sujeitos têm < 5 janelas de referência. **Decisão mantida: centragem por sujeito (`_c`) como principal, padronização global ajustada só no treino; `_z` disponível onde existe.**

### 5.4 Mapeamento sujeito oficial ↔ pasta nos 5 folds

Fontes: `results/exp5_uta_mapeamento_oficial.csv` (um sujeito oficial por linha: pasta, classe, motivo, escores, checagem de duração, K, piscadas descartadas), `results/exp5_uta_mapeamento_escores.csv` (sujeito × vídeo × pasta candidata), `results/exp5_uta_mapeamento_sensibilidade.csv`, `data/features/final/uta/oficial_piscadas_fold<k>.csv`; método em `src/features/uta_oficial.py`.

Método (extensão do da rodada preliminar): a Frequency oficial, gravada com 4 casas decimais, devolve o quadro (NF, em quadros com rosto do dlib) de cada piscada oficial; no vídeo certo esses instantes caem a uma defasagem quase constante dos vales do nosso EAR. O escore principal agora é **coerente**: uma trajetória de defasagens por bloco de 25 piscadas, escolhida por programação dinâmica com a restrição física de que a defasagem só cai devagar (dlib perdendo quadros: ≤ 300 quadros por bloco) e quase não sobe (≤ 10), menos o acaso (instantes deslocados 1/3 e 2/3 do vídeo); soma dos 3 vídeos. Vídeos divididos: vales das partes concatenados em ordem. A série oficial do vídeo 10 de F3_32 termina em NF = 9.483 quadros com rosto, além dos 8.710 quadros da parte 1: a contagem do original seguiu na parte 2 (não recomeçou). Classes: **confirmado** = escore ≥ 0,30, máximo da linha e da coluna; **provável** = sem coincidência forte, mas determinado pela ordem de listagem (pastas em ordem crescente, compatível com todos os confirmados do fold) ou pela única pasta restante compatível com a duração (NF da última piscada oficial ≤ nº de quadros); **indeterminado** = o resto.

Calibração do limiar 0,30: nos 47 sujeitos confirmados, as 517 combinações com as outras pastas do fold têm escore máximo 0,257 (percentil 99: 0,185), e os pares confirmados, mínimo 0,413 (mediana 0,941). O escore por blocos independentes da rodada preliminar escolheria a mesma pasta em 46 dos 47 confirmados (difere em Fold 3 oficial 5: F3_34 em vez de F3_31), mas separa pior: máximo entre as combinações não adotadas 0,463 contra mínimo −0,051 nos pares confirmados. Sensibilidade aos parâmetros (defasagem mínima −1.500/−6.000 quadros, queda máxima 150/600 por bloco): mapeamento idêntico em todos os folds.

| Fold | Sujeitos oficiais | Pastas | Confirmados | Prováveis | Indeterminados | Pastas sem sujeito oficial atribuído |
|---|---:|---:|---:|---:|---:|---|
| Fold 1 | 11 | 12 | 11 | 0 | 0 | F1_12 |
| Fold 2 | 12 | 12 | 9 | 3 | 0 | — |
| Fold 3 | 12 | 12 | 7 | 1 | 4 | F3_25, F3_27, F3_29, F3_34 |
| Fold 4 | 12 | 12 | 10 | 2 | 0 | — |
| Fold 5 | 11 | 12 | 10 | 1 | 0 | F5_49 |

Pares não confirmados e motivo:

| Fold | Sujeito oficial | Pasta | Classe | Maior escore (pasta) | Motivo |
|---|---:|---|---|---:|---|
| Fold 2 | 2 | F2_15 | provavel | −0,14 (F2_15) | ordem de listagem (pastas em ordem crescente, como os confirmados do fold) |
| Fold 2 | 8 | F2_21 | provavel | 0,09 (F2_23) | ordem de listagem (pastas em ordem crescente, como os confirmados do fold) |
| Fold 2 | 10 | F2_23 | provavel | 0,13 (F2_23) | ordem de listagem (pastas em ordem crescente, como os confirmados do fold) |
| Fold 3 | 0 | — | indeterminado | 0,13 (F3_36) | sem coincidência forte; pastas restantes compatíveis com a duração: F3_25;F3_27 |
| Fold 3 | 1 | — | indeterminado | 0,19 (F3_27) | sem coincidência forte; pastas restantes compatíveis com a duração: F3_27;F3_34 |
| Fold 3 | 3 | F3_26 | provavel | 0,12 (F3_26) | única pasta restante compatível com a duração |
| Fold 3 | 8 | — | indeterminado | −0,25 (F3_26) | sem coincidência forte; pastas restantes compatíveis com a duração: — (série oficial mal recuperada) |
| Fold 3 | 11 | — | indeterminado | 0,11 (F3_27) | sem coincidência forte; pastas restantes compatíveis com a duração: — (série oficial mal recuperada) |
| Fold 4 | 6 | F4_43 | provavel | 0,26 (F4_43) | ordem de listagem (pastas em ordem crescente, como os confirmados do fold) |
| Fold 4 | 7 | F4_44 | provavel | 0,23 (F4_44) | ordem de listagem (pastas em ordem crescente, como os confirmados do fold) |
| Fold 5 | 2 | F5_52 | provavel | 0,08 (F5_50) | ordem de listagem (pastas em ordem crescente, como os confirmados do fold); ATENÇÃO: NF_final da série oficial passa do nº de quadros da pasta em 2 vídeo(s) (cópia dos autores possivelmente mais longa) |

**Pasta ausente no Fold 5: F5_49.** Os 10 sujeitos oficiais confirmados do Fold 5 (índices 0, 1, 3, 4, 5, 6, 7, 8, 9, 10) correspondem às pastas F5_50, F5_51, F5_53, F5_54, F5_55, F5_56, F5_57, F5_58, F5_59, F5_60, na mesma ordem crescente; o sujeito oficial 2 (maior escore 0,08, sem coincidência forte com nenhuma pasta) fica, pela ordem, em F5_52. Ressalva: a última piscada oficial dele (NF = 11.932 / 18.264 / 18.410 nos vídeos 0 / 5 / 10) passa do nº de quadros de F5_52 em 2 vídeo(s) (NF/quadros até 1,018) e do de F5_49 em 1 (até 1,019) — explicação possível: cópia dos autores um pouco mais longa; a identificação é pela ordem, não pelas piscadas. **Cautela:** nos 133 vídeos com série oficial confiável dos pares confirmados (todos os folds), a razão NF/quadros nunca passa de 1 (máximo 0,9993; 0 acima de 1), de modo que a violação acima é atípica: o par oficial 2 ↔ F5_52 se apoia só na ordem de listagem e deve ficar fora da análise principal (só na sensibilidade `confirmado` + `provavel`). No Fold 1, a pasta ausente continua sendo F1_12.

**Fold 3 — ordem não crescente e defeitos do arquivo oficial:** (i) os 7 confirmados (oficiais 2, 4, 5, 6, 7, 9, 10 → F3_28, F3_30, F3_31, F3_32, F3_33, F3_35, F3_36) seguem pasta = índice + 26, o que não cabe numa listagem crescente de 12 pastas com 12 sujeitos oficiais; a ordem não pode completar o mapeamento; (ii) no sujeito oficial 11 os vídeos 5 e 10 têm sequências idênticas no arquivo oficial; (iii) a Frequency oficial não é recuperada de forma confiável (em nenhum dos 3 vídeos há NF inteiro em todas as piscadas com poucas piscadas 'descartadas') nos sujeitos oficiais 8, 11. Indeterminados: oficiais 0, 1, 8, 11 (pastas restantes: F3_25, F3_27, F3_29, F3_34); oficial 3 → F3_26 provável (única pasta restante compatível com a duração).

**Uso recomendado:** comparações com as características oficiais (Exp. 5a) com os pares `confirmado` (principal) e `confirmado` + `provavel` (sensibilidade); nunca com os indeterminados. Os experimentos com as características oficiais (Exp. 1 e 2) não usam o mapeamento e não mudam.

### 5.5 Sequências de 30 piscadas (passo 2)

| Fold | Sujeitos | Vídeos lógicos com sequência | Seq. KSS 0 | Seq. KSS 5 | Seq. KSS 10 | Sequências | Piscadas | Oficial (teste do fold): 0 / 5 / 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fold 1 | 12 | 36 | 137 | 586 | 565 | 1.288 | 3.736 | 228 / 494 / 547 |
| Fold 2 | 12 | 35 | 340 | 695 | 926 | 1.961 | 5.280 | 465 / 835 / 743 |
| Fold 3 | 12 | 35 | 267 | 697 | 883 | 1.847 | 4.920 | 449 / 771 / 976 |
| Fold 4 | 12 | 36 | 98 | 755 | 1.080 | 1.933 | 4.888 | 211 / 584 / 667 |
| Fold 5 | 12 | 36 | 179 | 737 | 898 | 1.814 | 4.920 | 295 / 810 / 586 |
| **total** | 60 | 178 | 1.021 | 3.470 | 4.352 | 8.843 | 23.744 | 1.648 / 3.494 / 3.519 |

**Vídeos sem nenhuma piscada detectada** (2 de 180; nenhuma sequência): `F2_15_10` (KSS 10; limiar 0,204, EAR mínimo 0,214, percentil 1 0,258); `F3_27_5` (KSS 5; limiar 0,198, EAR mínimo 0,236, percentil 1 0,276). O EAR nunca cruza o limiar calibrado no último terço do vídeo 0 do sujeito — o nível de olho aberto subiu nesses vídeos (efeito inverso ao de F1_03_10; calibração por sujeito, não erro de código). Nas janelas de 10 s esses vídeos têm PERCLOS e taxa de piscadas 0 em todas as janelas; o Exp. 5b deve reportar isso (e, por vídeo, a métrica de sequências da UTA cobre só os vídeos com sequência).

Vídeos divididos concatenados: `F3_32_10` = F3_32_10_p1 + F3_32_10_p2 (180 piscadas, 76 sequências); `F5_49_10` = F5_49_10_p1 + F5_49_10_p2 (285 piscadas, 128 sequências). Fontes: `results/exp5_sequencias_resumo.csv` (uma linha por vídeo lógico), `data/features/final/uta/sequences_normalizacao.csv`; variante com o primeiro terço como referência em `data/features/final/uta/sequences_primeiro_terco.npz`.

Concordância vídeo a vídeo com as características oficiais (Spearman entre as medianas por vídeo das características normalizadas; só pares confirmados; `results/exp5_uta_validacao_oficial_correlacoes.csv` — a análise completa é do Exp. 5a):

| Vídeos | Característica | limiar | segundos | estendida |
|---|---|---:|---:|---:|
| todos os vídeos (141) | amplitude | −0,03 | −0,03 | 0,34 |
| todos os vídeos (141) | duracao | 0,23 | 0,15 | 0,18 |
| todos os vídeos (141) | frequencia | 0,55 | 0,53 | 0,55 |
| todos os vídeos (141) | velocidade | 0,22 | 0,21 | 0,52 |
| só vídeos 5 e 10 (94) | amplitude | 0,00 | 0,00 | 0,32 |
| só vídeos 5 e 10 (94) | duracao | 0,23 | 0,13 | 0,25 |
| só vídeos 5 e 10 (94) | frequencia | 0,47 | 0,46 | 0,47 |
| só vídeos 5 e 10 (94) | velocidade | 0,30 | 0,28 | 0,56 |

## 6. NITYMED

Janelas de 10 s: 979 de 126 vídeos — 529 `exemplo`, 200 `referencia` (primeiro terço), 250 `fronteira`. 63 vídeos (Microsleep 1, Yawning 62) não têm janela inteira no primeiro terço (vídeo < 30 s) e usam a pseudo-referência (o terço inteiro como uma janela). Sem classe alerta nem rótulo por janela: serve para taxa de disparo, captura e qualidade de sinal (PP3). Com características brutas (principal do Exp. 6) a referência não entra no cálculo; `avaliavel` exclui referência e fronteira para manter a mesma regra das variantes normalizadas. Por categoria: Microsleep 438 janelas (273 avaliáveis); Yawning 541 janelas (256 avaliáveis).

## 7. Limitações e pendências

1. **DMD — alerta de outra sessão de gravação**: a referência (s3) foi gravada no mesmo dia da s5 em 13 de 13 sujeitos e a s2 (alerta), em outro dia em 10 de 13 (seção 4.1). O Exp. 5b deve considerar esse fator (ex.: sensibilidade restrita aos sujeitos com s2 no mesmo dia da s5, ou sem normalização).
2. **Metadados `setup` fora da convenção** (seção 4.1): gB_10 s1 (11/03 15h24): 'Car Stopped'; gB_6 s2 (11/03 13h46): 'Car Moving'; gZ_33 s1 (08/04 10h08): 'Car Stopped'; gZ_33 s1 (08/04 16h18): 'Car Stopped'. As janelas foram mantidas pela sessão, conforme a regra.
3. **Sensibilidade de 15 s no DMD** quase sem alerta e com referência na s2 (seção 4.4).
4. **Mapeamento**: 4 sujeitos oficiais indeterminados (Fold 3, com defeitos no arquivo oficial) e pares prováveis identificados só pela ordem ou pela duração (seção 5.4).
5. **Limiar de EAR de outra sessão/trecho**: sessões com nível de olho aberto diferente do da referência têm PERCLOS e contagem de piscadas deslocados (seções 4.2 e 5.2); é a calibração por sujeito prevista, não um erro.
6. `results/sinais_resumo.md` (regenerado em 06/10/2026) cobre todas as sessões e folds; as contagens deste relatório estão na seção 2 e em `results/exp5_contagens_sinais.csv`.

