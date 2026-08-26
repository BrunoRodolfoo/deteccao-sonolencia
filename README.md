# Detecção de Sonolência em Dispositivos Móveis

**Avaliação da Generalização entre Sonolência Atuada e Real**

Projeto Transformador II — Bacharelado em Ciência da Computação, PUCPR (Turma 8ºA, Equipe 14)

André Luís Oliveira Ferreira da Silva · Bruno Rodolfo da Silveira · Bruno Sandoval Ribeiro · Paulo Henrique Perin

## O projeto

Investigamos quanto do desempenho reportado na literatura de detecção de sonolência
por vídeo se mantém quando o modelo é (i) avaliado em sonolência **real** (UTA-RLDD)
em vez de **atuada** (NTHU-DDD) e (ii) restrito à execução local em um smartphone
comum (TensorFlow Lite, quantização int8). Detalhes no documento de planejamento
(Entrega 01).

## Baseline (Entrega 01)

Dois artefatos executáveis:

### 1. Reprodução do protocolo oficial do UTA-RLDD

Classificadores clássicos (scikit-learn) sobre as *blink features* e as 5 partições
oficiais (divisão por sujeito) disponibilizadas por
[Ghoddoosian et al. (CVPRW 2019)](https://github.com/rezaghoddoosian/Early-Drowsiness-Detection).

```bash
pip install -r requirements.txt
python src/baseline/download_data.py   # ~40 MB de features (não os 111 GB de vídeo)
python src/baseline/train.py
```

Resultado (3 classes, acurácia média nos 5 folds — tabela completa em
[results/baseline_uta_rldd.md](results/baseline_uta_rldd.md)):

| Modelo | Acurácia média |
|---|---|
| **RandomForest (estatísticas das piscadas)** | **54,7% ± 9,0%** |
| SVM-RBF (estatísticas das piscadas) | 53,0% ± 3,8% |
| Referência: HM-LSTM (Ghoddoosian et al.) | 65,2% |
| Referência: julgamento humano | 57,8% |

Logs de execução versionados em `results/logs/`; modelos treinados em `models/`.

### 2. Protótipo em tempo real (webcam)

MediaPipe Face Mesh → EAR/PERCLOS/MAR → alerta com amortecimento temporal:

```bash
python src/realtime/demo_webcam.py
```

O limiar de EAR é calibrado automaticamente nos 3 primeiros segundos
(mantenha os olhos abertos). Tecla `q` encerra.

## Estrutura

```
src/realtime/    demo webcam + métricas (EAR, MAR, PERCLOS)
src/baseline/    download dos dados, folds oficiais, treino e avaliação
results/         tabela de resultados e logs de execução
models/          classificadores treinados (.pkl)
data/            dados locais (não versionados — ver data/README.md)
```

## Aviso

Protótipo de pesquisa. Não diagnostica distúrbios do sono, não é dispositivo
veicular certificado e não substitui o descanso previsto em lei. Todo o
processamento ocorre localmente, sem armazenamento ou transmissão de vídeo.
