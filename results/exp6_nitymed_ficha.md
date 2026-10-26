# Ficha do dataset NITYMED (cópia usada no TCC)

Gerado por `src/experiments/exp6_noite.py` (rodada final) em 27/09/2026 a partir dos arquivos em disco (`data/nitymed/`), dos sinais extraídos (`data/signals/nitymed/index.csv`), das medidas por vídeo do Exp. 6 (`results/exp6_noite_videos.csv`) e da luminância medida (`results/exp6_luminancia_videos.csv`). Índice por vídeo: `data/nitymed/index.csv` (inclui SHA-256 de cada arquivo). A rodada preliminar (janelas de 30 s) foi descartada; seus arquivos foram removidos em 06/10/2026.

## 1. Identificação

| Campo | Valor | Fonte |
|---|---|---|
| Nome | NITYMED — NIght Time Yawning-Microsleep-Eyeblink-driver Distraction | Kaggle `nikospetrellis/nitymed`; IEEE DataPort |
| Autores | Nikos Petrellis, Nikolaos Voros, Christos Antonopoulos, Georgios Keramidas, Panagiotis Christakos, Panagiotis Mousouliotis | DataCite (DOIs 10.21227/85xe-3f88 e 10.34740/KAGGLE/DSV/3921886) |
| Cópia usada | Kaggle, versão 1 ("Initial release", 09/07/2022), arquivo `nitymed_720p.zip` (HDTV720) | API do Kaggle, consultada em 27/09/2026 (rodada preliminar) |
| Conteúdo declarado | 130 vídeos: "the drivers yawn 3 times in each video lasting approximately 15-25 seconds (107 videos)" e "the drivers talk, look around and have microsleeps in videos lasting approximately 2 minutes (21 videos)"; 21 motoristas (11 homens, 10 mulheres); Patras, Grécia; carros reais em movimento, à noite; mp4, 25 quadros/s, sem áudio; 720p (Kaggle) e 1080p (site da universidade) | página do IEEE DataPort (lida pelo orquestrador em 27/09/2026) e descrição do Kaggle |
| Iluminação declarada | "natural, com um leve reforço pelas luzes internas mais fracas do carro, para simular a iluminação de uma avenida"; a maioria dos vídeos foi gravada numa estrada escura e pouco movimentada | descrição no Kaggle |
| Autenticidade dos comportamentos | **NÃO documentada**: a fonte primária não diz se bocejos e microssonos foram espontâneos ou encenados; o número fixo de 3 bocejos por vídeo sugere execução sob instrução. Enquadramento: condução noturna REAL com comportamentos de autenticidade não documentada — serve para a condição NOITE da PP3 (captura, sinal, disparo), **não** como evidência de sonolência real para a PP1 | página do IEEE DataPort (27/09/2026) |

## 2. Licença e citação

- **Licença da cópia usada (Kaggle): CC BY-ND 4.0** (Attribution-NoDerivatives 4.0 International) — campo `licenseName` da API do Kaggle.
- A página do IEEE DataPort não traz licença padronizada. Como a cópia baixada é a do Kaggle, vale a **BY-ND**.
- CC BY-ND permite reproduzir quadros **sem modificação** e com atribuição (recortes, anotações sobrepostas ou borrões são modificações). **Nenhuma figura com rosto foi gerada**: a decisão de publicar quadros é da equipe. As figuras do Exp. 6 só têm números e séries temporais.
- Exigência dos autores: "free to be used provided that one or more of the following papers are cited". Metadados conferidos em 27/09/2026 (Crossref/DataCite, rodada preliminar; nenhuma consulta externa nesta rodada):

| Chave no `paper/refs.bib` | Tipo | Referência | DOI | Conferido em |
|---|---|---|---|---|
| `petrellis2021dsd` (presente no .bib) | artigo (exigido: opção 3 da lista dos autores) | N. Petrellis, S. Zogas, P. Christakos, G. Keramidas, P. Mousouliotis, N. Voros, C. Antonopoulos. High Speed Implementation of the Deformable Shape Tracking Face Alignment Algorithm. 2021 24th Euromicro Conference on Digital System Design (DSD), Palermo, 1–3 set. 2021, pp. 174–177. | 10.1109/DSD53832.2021.00035 | Crossref |
| `petrellis2021vlsisoc` (presente no .bib) | artigo (opção 1 da lista dos autores) | N. Petrellis, P. Christakos, S. Zogas, P. Mousouliotis, G. Keramidas, N. Voros, C. Antonopoulos. Challenges Towards Hardware Acceleration of the Deformable Shape Tracking Application. 2021 IFIP/IEEE 29th Int. Conf. on Very Large Scale Integration (VLSI-SoC), Singapura, out. 2021, pp. 1–4. | 10.1109/VLSI-SoC53125.2021.9606999 | Crossref |
| `(não incluída no .bib)` | artigo (opção 2 da lista dos autores) | N. Petrellis, S. Zogas, P. Christakos, P. Mousouliotis, G. Keramidas, N. Voros, C. Antonopoulos. Software Acceleration of the Deformable Shape Tracking Application. ESSE 2021 (2nd European Symposium on Software Engineering), Larissa, ACM, 2021, pp. 51–57. | 10.1145/3501774.3501782 | Crossref |
| `petrellis2022kaggle` (presente no .bib) | conjunto de dados — cópia usada (Kaggle, opção 4) | N. Petrellis, N. Voros, C. Antonopoulos, G. Keramidas, P. Christakos, P. Mousouliotis. NITYMED [Data set]. Kaggle, 2022. | 10.34740/KAGGLE/DSV/3921886 | DataCite |
| `petrellis2022` (presente no .bib) | conjunto de dados — registro no IEEE DataPort (já existia no .bib) | N. Petrellis, N. Voros, C. Antonopoulos, G. Keramidas, P. Christakos, P. Mousouliotis. NITYMED. IEEE DataPort, 2022. | 10.21227/85xe-3f88 | DataCite |

Recomendação: citar o conjunto de dados (`petrellis2022`, IEEE DataPort, ou `petrellis2022kaggle`, a cópia usada) e **pelo menos um** dos artigos (`petrellis2021dsd` ou `petrellis2021vlsisoc`). Divergência registrada: a lista do Kaggle diz VLSI-SoC "4–8 October 2021"; o Crossref registra o evento de 4 a 7/10/2021 — o .bib segue o Crossref.

## 3. Estrutura em disco

```
data/nitymed/
├── nitymed_720p.zip                      (694.690.346 bytes)
└── DSM_Dataset-HDTV720/
    ├── Microsleep/{Female,Male}/HDTV720/P10xxxxx_720.mp4
    └── Yawning/{Female,Male}/HDTV720/P10xxxxx_720.mp4
```

- 126 arquivos .mp4 em disco; 126 arquivos no .zip; todos os do disco estão no .zip: sim; tamanhos conferem: sim.
- Rótulo = pasta (categoria); gênero = pasta. **Não há identificador de motorista** nos nomes (P10xxxxx = número do arquivo da câmera): vídeos do mesmo motorista não podem ser agrupados — qualquer estatística por vídeo trata como independentes vídeos que não são.
- Contagem real × declarada: 126 × 130 vídeos; Microssono 19 × 21; Bocejo 107 × 107. O total de bytes do Kaggle (API: `totalBytes` = 697.049.508) coincide com a soma dos 126 arquivos do .zip (697.049.508 bytes): a diferença para 130 já está na versão publicada, não é falha de download.

## 4. Contagens por categoria e gênero

| Categoria | Gênero | Vídeos | Duração total (min) | Duração mediana (s) | Mín–máx (s) | Quadros | Tamanho (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| Microsleep | Female | 8 | 15,3 | 124,4 | 14,0–146,6 | 22.958 | 87,2 |
| Microsleep | Male | 11 | 23,6 | 125,8 | 123,8–146,0 | 35.370 | 133,8 |
| Yawning | Female | 59 | 31,7 | 26,4 | 14,8–132,0 | 47.566 | 258,7 |
| Yawning | Male | 48 | 27,1 | 32,6 | 18,8–60,4 | 40.590 | 217,3 |
| **Total** | | 126 | 97,7 | 30,2 | 14,0–146,6 | 146.484 | 697,0 |

Vídeos de microssono com menos de 60 s (fora do "~2 min" declarado): `P1043114_720.mp4` (14,0 s).

Duração dos vídeos de bocejo × declarada (15–25 s): 37 de 107 dentro; 1 abaixo; 69 acima (mediana 28,2 s); 4 com mais de 60 s.

Bocejos detectados nos nossos sinais (MAR > 0,5 sustentado ≥ 1 s, a definição de `frac_bocejo`, contados no vídeo inteiro): mediana 2 por vídeo de bocejo (fora os suspeitos abaixo); 21 dos 107 vídeos de bocejo sem nenhum; 42 com 3 ou mais (declarado: 3 por vídeo).

**Possível erro de pasta na versão publicada**: 3 vídeos da pasta Yawning têm duração de vídeo de microssono (> 90 s) e **nenhum bocejo detectado** no vídeo inteiro: `Yawning/Female/P1042777_720.mp4` (124,7 s), `Yawning/Female/P1043088_720.mp4` (132,0 s), `Yawning/Female/P1043107_720.mp4` (126,8 s). Pela duração e pela ausência de bocejo, parecem vídeos de microssono na pasta errada (com eles, o Microssono teria 22 vídeos × 21 declarados). NÃO confirmado visualmente nem pelos autores: os rótulos de pasta foram mantidos, e o efeito nas taxas de disparo está em `results/exp6_noite.md` (sensibilidade da seção 5).

## 5. Formato

| Campo | Valor |
|---|---|
| Resolução | 1280×720 (126 vídeos) |
| fps (contêiner) | 24,00 (1 vídeo); 25,00 (125 vídeos); fora de 25: `P1042749_720.mp4` |
| Codec (OpenCV, FOURCC) | h264 (126) |
| Taxa de bits | mediana 1,04 Mbit/s (mín 0,58, máx 1,53) |
| Quadros decodificados = quadros do contêiner | 126 de 126 vídeos |
| Janelas de 10 s (passo 5 s) | 979, das quais 529 avaliáveis (fora do primeiro terço, que é a referência; 0 vídeos com janela parcial); vídeos sem nenhuma janela avaliável: 12 |
| Taxa de detecção de rosto (Face Mesh) | mediana por vídeo 1,0000, mín 0,8737 |
| Luminância média do rosto (0–255) | mediana por vídeo 82,9 (mín 68,9, máx 99,4) |

## 6. Uso no TCC

- Só na **PP3 (condição noite)**: detecção de rosto, qualidade do sinal de EAR e taxa de disparo dos detectores (Exp. 6, `results/exp6_noite.md`).
- **Não** serve como evidência de sonolência real para a PP1 (autenticidade não documentada) e **não** permite acurácia: não há classe alerta nem anotação quadro a quadro.
- A referência de normalização e do limiar de EAR no NITYMED é o primeiro terço do próprio vídeo (não existe vídeo alerta) — desvio documentado no Exp. 5.
