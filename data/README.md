# Dados

Nenhum dado é versionado neste repositório. Esta pasta é preenchida localmente.

## Blink features do UTA-RLDD (baseline)

As *blink features* pré-extraídas e as 5 partições oficiais (divisão por sujeito) são
distribuídas pelos próprios autores do UTA-RLDD no repositório
[rezaghoddoosian/Early-Drowsiness-Detection](https://github.com/rezaghoddoosian/Early-Drowsiness-Detection)
(CVPRW 2019). São 20 arquivos `.npy`:

- `Blinks_30_Fold{1..5}.npy` / `Labels_30_Fold{1..5}.npy` — treino (4 folds restantes)
- `BlinksTest_30_Fold{1..5}.npy` / `LabelsTest_30_Fold{1..5}.npy` — teste (o fold em questão)

Cada amostra é uma sequência de 30 piscadas com 4 características por piscada
(duração, amplitude, velocidade de abertura do olho e frequência), rotulada em
3 classes (alerta / vigilância baixa / sonolento).

Para baixar tudo (~alguns MB) em `data/uta_rldd_blinks/`:

```bash
python src/baseline/download_data.py
```

## Vídeos completos (fases futuras)

- **UTA-RLDD** (~111 GB): https://sites.google.com/view/utarldd/home
- **NTHU-DDD**: acesso mediante formulário institucional aos autores
- **MRL Eye**: http://mrl.cs.vsb.cz/eyedataset
