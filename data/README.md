# Dados (nada aqui é versionado)

Esta pasta é preenchida localmente; o `.gitignore` exclui tudo, menos este arquivo. O repositório é
público: **nunca** faça commit de vídeos, quadros, sinais por quadro ou recortes das bases. Papel de cada
base, licenças e citações obrigatórias: `README.md`, seção 3. Espaço usado nesta máquina (conferido em
26/10/2026 com `du -sh`): 112 GB (UTA-RLDD) + 27 GB (DMD) + 1,3 GB (NITYMED) + 0,5 GB (MRL + CEW, já
extraídos) + cerca de 1 GB de derivados (sinais, características e caches).

## Estrutura esperada

```
data/
├── uta_rldd_blinks/           características oficiais da UTA-RLDD (20 .npy, 40 MB)      -> Exp. 1, 2, 5a
├── uta_rldd_videos/Fold{1..5}/<01..60>/{0,5,10}.<mp4|mov|...> + Fold<N>/.ok             -> Exp. 5a, 5b, 6
├── dmd/<grupo>/<sujeito>/s{1,2,3,5}/*_rgb_face.mp4 + *_rgb_ann_<pacote>.json             -> Exp. 5b, 6, app
│      (+ README-drowsiness.md e README-distraction.md oficiais)
├── nitymed/DSM_Dataset-HDTV720/{Microsleep,Yawning}/{Female,Male}/HDTV720/*.mp4          -> Exp. 6
│      (+ index.csv, ficha gerada pelo Exp. 6)
├── mrl_eye/mrlEyes_2018_01/s0001..s0037/ (+ index.csv gerado)                            -> Exp. 3
├── cew/cew_eyes_24x24/, cew/cew_faces_100x100/                                           -> Exp. 3
├── cache/                     recortes 48x48 (*.npz) + face_landmarker (gerados)
├── signals/<base>/*.npz       sinais por quadro (EAR, MAR, pose), gerados no .venv-mp
│      app_tasks/{dmd,uta}/, app_headless/  sinais e registros da validação do app (só números)
└── features/final/<base>/     piscadas, janelas de 10 s (5/15 s), sequências (gerados)
```

A rodada preliminar (`features/<base>/`, janelas de 30 s) saiu em 06/10/2026 e não é usada por nenhum script.

## Como obter cada base

### UTA-RLDD: características oficiais (Exp. 1, 2 e 5a)

Os autores distribuem as 5 partições oficiais (por sujeito) com as características de piscada já
extraídas no repositório [rezaghoddoosian/Early-Drowsiness-Detection](https://github.com/rezaghoddoosian/Early-Drowsiness-Detection):
`Blinks_30_Fold{1..5}.npy`, `Labels_30_Fold{1..5}.npy` (treino = os outros 4 folds),
`BlinksTest_30_Fold{1..5}.npy`, `LabelsTest_30_Fold{1..5}.npy` (teste). Cada amostra é uma sequência de 30
piscadas × 4 características, na ordem [0] frequência, [1] amplitude, [2] duração, [3] velocidade de
abertura (conferida contra os arquivos; `models/blink_temporal.json`), com z-score por sujeito.
Os arquivos têm **58** sujeitos (os folds 1 e 5 têm 11; o artigo da base descreve 60).

```bash
.venv/bin/python src/baseline/download_data.py        # ~1 min, 40 MB -> data/uta_rldd_blinks/
```

### UTA-RLDD: vídeos (Exp. 5a, 5b e 6)

10 zips (`Fold{1..5}_part{1,2}.zip`, ~111 GB) no Google Drive oficial (link na
[página da base](https://sites.google.com/view/utarldd/home)). **O Drive aplica cota por rajada a
downloads anônimos** ("Quota exceeded ... too many users"): baixe pelo navegador, com uma conta Google
logada, para `~/Downloads`, e deixe o vigia extrair cada fold quando as duas partes estiverem completas
(o Chrome só renomeia para `.zip` ao terminar, então nada é extraído pela metade):

```bash
bash tools/watch_downloads.sh                  # loop: tools/prepare_data.py --uta a cada 2 min até os 5 folds terem .ok
.venv/bin/python tools/prepare_data.py --uta             # uma passada (extrai o que já está em ~/Downloads)
.venv/bin/python tools/prepare_data.py --uta --baixar    # tenta também o download anônimo, gentil, com retomada
```

Cada fold extraído é conferido (contagem de vídeos) e recebe `Fold<N>/.ok`; os zips são apagados para
liberar disco. Não usamos espelhos nem cópias do Kaggle (a proposta promete resultados só por canais
oficiais). **Vídeos divididos:** `Fold3/32` e `Fold5/49` têm o vídeo sonolento em duas partes
(`10_1`, `10_2`); cada parte vira `F<n>_<suj>_10_p<k>` e as métricas por vídeo agrupam pelo vídeo lógico.
Autorização de imagem: só 36 dos 60 sujeitos; **nenhum quadro é publicado**.

### DMD: Driver Monitoring Dataset, da Vicomtech (Exp. 5b, 6 e validação do app)

1. Preencha os formulários do pacote de **sonolência** (sessão s5) e do pacote de **distração** (sessões
   s1, s2 e s3, com os trechos "Safe Driving" usados como classe alerta) em https://dmd.vicomtech.org/
   (maiores de 18, uso acadêmico). Chega um e-mail com links assinados, **válidos por 48 h**.
2. No Outlook, abra o e-mail "Vicomtech Dataset Request | ..." → "..." → **Baixar** (gera um `.eml` em `~/Downloads`).
3. Baixe e extraia (só o vídeo da câmera do rosto e a anotação OpenLABEL de cada pacote; o `.tar.gz` é apagado):

```bash
.venv/bin/python tools/download_dmd.py                      # usa o .eml mais recente da Vicomtech em ~/Downloads
.venv/bin/python tools/download_dmd.py caminho/email.eml    # ou um .eml específico
.venv/bin/python tools/prepare_data.py --dmd                # reextrai pacotes já baixados em data/dmd/
```

Conteúdo usado: 13 sujeitos com s5 (gA/1, gA/5, gB/6, gB/7, gB/9, gB/10, gC/13, gC/14, gE/29, gF/23,
gZ/33, gZ/36, gZ/37; 16 sessões, pois gB/10, gF/23 e gZ/33 têm manhã e tarde) + as sessões de distração
dos mesmos sujeitos e de gE/28 (sem s5). Sonolência **encenada** ("These states were acted"), carro
parado na s5; na s1 o carro anda de verdade. Licença **CC BY-NC-ND 4.0**: nenhum quadro no artigo ou no pôster.

### NITYMED (Exp. 6)

Kaggle [`nikospetrellis/nitymed`](https://www.kaggle.com/datasets/nikospetrellis/nitymed) (versão 720p,
conta gratuita; o IEEE DataPort exige assinatura). Descompacte em `data/nitymed/` (fica
`data/nitymed/DSM_Dataset-HDTV720/...`). Vieram **126** vídeos (19 Microsleep + 107 Yawning); a ficha oficial
diz 130 (21 Microsleep). A divergência está registrada em `results/exp6_nitymed_ficha.md`. Condução noturna real;
a autenticidade de bocejos e microssonos não é documentada. Licença da cópia do Kaggle: **CC BY-ND 4.0**; citar
o dataset **e** um artigo dos autores (`petrellis2021dsd` ou `petrellis2021vlsisoc`).

### MRL Eye e CEW (Exp. 3)

- **MRL Eye Dataset** (Fusek, ISVC 2018): download livre em http://mrl.cs.vsb.cz/eyedataset; descompacte
  `mrlEyes_2018_01.zip` em `data/mrl_eye/mrlEyes_2018_01/` (pastas `s0001` a `s0037`). O índice com os
  metadados do nome do arquivo é gerado em `data/mrl_eye/index.csv`. Os recortes do MRL podem ir para
  figuras: `results/figures/amostras_mrl.{png,pdf}` mostra exemplos por óculos e iluminação.
- **CEW, Closed Eyes in the Wild** (Song et al., 2014), página do grupo PARNEC/NUAA: descompacte em
  `data/cew/cew_eyes_24x24/dataset_B_Eye_Images/` (recortes 24x24) e
  `data/cew/cew_faces_100x100/dataset_B_FacialImages/` (faces 100x100 + `EyeCoordinatesInfo_*.txt`).
  **Só pesquisa; redistribuição proibida.** Por isso nenhuma figura do repositório mostra recortes do CEW.
- Os caches 48x48 (`data/cache/*.npz`) são gerados por `.venv/bin/python src/pipeline_b/datasets.py`.

### NTHU-DDD

Não obtido (termo de licença com assinatura de chefia institucional; prazo de 20/09 vencido). Substituído
por DMD + NITYMED: `docs/TROCA_DATASETS.md`.

## Derivados (gerados pelos scripts; ver `src/features/README.md`)

| Pasta | Gerada por | Ambiente |
|---|---|---|
| `signals/{dmd,nitymed,uta}/` | `src/features/extract_signals.py --dataset <base>` | `.venv-mp` |
| `features/final/<base>/` | `bash src/features/run_features.sh` | `.venv` |
| `signals/app_tasks/dmd/` | `tools/app_validacao/extrair_sinais_app.py` | `.venv-mp` |
| `signals/app_tasks/uta/` (cerca de 80 MB, 48 vídeos dos folds 2 e 4; teste independente do app) | `tools/app_validacao/extrair_sinais_app.py --uta-folds 2 4 --uta-rotulos 0 10`, lido por `tools/app_validacao/validar_motor_uta.mjs` | `.venv-mp` |
| `signals/app_headless/` | `tools/app_validacao/rodar_app_video.mjs` (Chrome headless, mudo) | node |
| `cache/` | `src/pipeline_b/datasets.py` | `.venv` |
