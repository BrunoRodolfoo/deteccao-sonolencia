# Fatos verificados sobre os datasets (pt-BR)

Verificação em 20/09/2026. Fonte de cada fato entre colchetes. "Não confirmado" = não conseguimos ler a fonte primária; não usar o número no artigo sem conferir.

## UTA-RLDD (Real-Life Drowsiness Dataset, Univ. of Texas at Arlington) — **sonolência real**
Fonte: texto integral de ghoddoosian2019 (CVPRW 2019) e página oficial https://sites.google.com/view/utarldd/home.
- **60 participantes** saudáveis, > 18 anos; 51 homens, 9 mulheres; etnias: 10 caucasianos, 5 hispânicos não brancos, 30 indo-arianos/dravídicos, 8 do Oriente Médio, 7 do Leste Asiático; idade 20–59 (média 25, DP 6). [paper, §3.1]
- **180 vídeos** RGB (3 por participante), ~10 min cada, **~30 h** no total, **111,3 GB**. [paper, §3.1, §3.3]
- Autogravados pelo participante com celular ou webcam, em casa ou na universidade, a até um braço de distância, ambos os olhos visíveis; **taxa de quadros sempre < 30 fps**; resoluções variadas (câmeras pessoais). [paper, §3.1–3.3]
- Óculos em **21 dos 180** vídeos; barba/pelos faciais consideráveis em **72 dos 180**. [paper, §3.1]
- **Rótulos pela KSS (autoatribuídos)**: **alerta = KSS 1–3** (sem sinais de sonolência); **pouca vigilância = KSS 6–7** (alguns sinais, sem esforço para se manter acordado); **sonolento = KSS 8–9** (precisa lutar ativamente contra o sono). Níveis 4–5 não são usados. Rótulos numéricos 0 / 5 / 10. [paper, §3.2, Tabela 1]
- Partição oficial: 60 sujeitos divididos aleatoriamente em **5 folds de 12** para validação cruzada por sujeito. [paper, §3.3]
- Julgamento humano: 4 voluntários por fold (20 no total: 8 mulheres, 12 homens; 3 graduação, 17 pós; idade 26,1 ± 2,9) anotaram os vídeos mudos com um número 0–10. VA humana média **57,8 %**. [paper, §3.4, §5.3]
- **Autorização de publicação de imagem: 36 dos 60 participantes** permitiram a publicação de seus rostos em artigos e trabalhos futuros. [página oficial]
- **Acesso**: download livre via link do Google Drive na página oficial, sem formulário; exige citar ghoddoosian2019. [página oficial]
- Hardware/velocidade do baseline: 35–80 fps (quadros de 568×320 a 1920×1080) em Intel Xeon E3-1270 V2 @ 3,5 GHz, 16 GB RAM. [paper, §5.4]
- Normalização das características: o paper diz "primeiro terço" do vídeo de alerta; o código `Preprocessing.py` usa o último terço (`[-bunch_size:]`). Registrar essa discrepância no artigo.
- Nas features públicas usadas neste TCC aparecem **174 vídeos = 58 sujeitos** (2 sujeitos ausentes); folds de teste com 33/36/36/36/33 vídeos. [verificação local, ver tarefa]

## NTHU-DDD (National Tsing Hua University Driver Drowsiness Detection) — **sonolência atuada**
Fontes: página oficial http://cv.cs.nthu.edu.tw/php/callforpaper/datasets/DDD/ (lida via curl), yu2019 (arXiv), lyu2018, wijnands2020, jabbar2018, weng2016.
- **36 sujeitos** de etnias diferentes, com/sem óculos/óculos escuros, sentados em cadeira jogando um jogo de direção com volante e pedais, **instruídos por um experimentador** a executar expressões (direção normal, bocejo, piscar lento, adormecer, gargalhar) sob iluminação de dia e de noite. **Total ≈ 9,5 h.** [página oficial; yu2019]
- **Treino: 18 sujeitos**, **5 cenários** (BareFace, Glasses, Night_BareFace, Night_Glasses, Sunglasses); sequências de bocejo e de piscar lento com aceno ~1 min; sequências combinadas (sintomas de sonolência / ações não relacionadas) ~1,5 min. Yu 2019 contabiliza 360 vídeos / 722.223 quadros; lyu2018: 10 homens e 8 mulheres. [página oficial; yu2019; lyu2018]
- **Avaliação + teste: 90 vídeos dos outros 18 sujeitos**, com estados misturados. Avaliação = **4 sujeitos, 20 vídeos, 173.259 quadros** (anotados); teste = 14 pessoas, **não público / sem anotação**. Wijnands 2020: treino 8,5 h, avaliação 1,5 h. [página oficial; yu2019; wijnands2020]
- Câmera com **iluminação infravermelha ativa**; **640×480, AVI**; **Night_BareFace e Night_Glasses a 15 fps; BareFace, Glasses e Sunglasses a 30 fps**. [página oficial]
- Anotações por quadro para sonolência, olhos, cabeça e boca. [lyu2018; yu2019]
- **Acesso: preencher o "Dataset License Agreement" e enviar por e-mail** (assunto "Dataset on Driver Drowsiness Detection from Video") para cvlablai636@my.nthu.edu.tw. A página não detalha quem deve assinar; a exigência de assinatura de chefia relatada pela equipe não consta na página pública (registrar como experiência da equipe). Citação obrigatória: weng2016. [página oficial]
- Acurácias publicadas no conjunto de avaliação: Park 2016 73,06 %; MSTN 85,52 % (82,61 % no teste, segundo o resumo e a Tabela 4 de massoz2018); HTDBN 84,82 % e Huynh 2016 87,46 % (ambos conforme Tabela 4 de massoz2018, fonte secundária); Yu 2019 0,762; LMDF 90,05 %; two-stream 94,46 % (RICAI 2020); Wijnands 2020 (móvel, out-of-sample) 73,9 %. Ver `related_work_notes.md`.

## MRL Eye Dataset (Media Research Lab, VSB-TU Ostrava) — **estado do olho, infravermelho**
Fontes: fusek2018 (artigo de referência); florez2026 e rahman2020 (estatísticas); página oficial do dataset ("MRL Eye Dataset | MRL", mrl.cs.vsb.cz/eyedataset; a página descreve os campos do nome de arquivo, os 37 sujeitos (33 homens, 4 mulheres), as 84.898 imagens e os três sensores; conferido também nos arquivos locais).
- **84.898 imagens** de olhos em infravermelho, **37 sujeitos (33 homens, 4 mulheres)**; 42.951 abertos / 41.947 fechados. [rahman2020, Tabela I; florez2026]
- Sensores: **Intel RealSense RS 300 (640×480), IDS Imaging (1280×1024), Aptina (752×480)**; imagens em tons de cinza de tamanho variável. [documentação do dataset; rahman2020 cita os três sensores]
- Nome do arquivo `sXXXX_YYYYY_G_GL_ES_R_L_SS.png`: sujeito, id da imagem, gênero (0 homem/1 mulher), óculos (0/1), estado do olho (0 fechado/1 aberto), reflexos (0 nenhum/1 pequeno/2 grande), iluminação (0 ruim/1 boa), sensor (01 RealSense/02 IDS/03 Aptina). [documentação do dataset; conferido nos arquivos locais em `data/mrl_eye/`]
- Acesso livre (download direto; também espelhado no Kaggle). Citação: fusek2018 (ISVC 2018, LNCS 11241, pp. 433–444).
- Uso cross-dataset na literatura: rahman2020 (MRL→CEW 90,94 %, MRL→ZJU 90,12 %); florez2026 (MRL→RT-BENE zero-shot Macro-F1 0,21–0,78 contra 0,96–0,98 intra-MRL, partição por sujeito).

## CEW — Closed Eyes In The Wild (NUAA) — **estado do olho, RGB, in-the-wild**
Fonte: página oficial http://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/ClosedEyeDatabases.html (responde 412 ao curl; lida via https em 20/09/2026, todos os números abaixo conferidos) e song2014.
- **2.423 sujeitos**: **1.192 com olhos fechados** (coletados da internet) e **1.231 com olhos abertos** (do LFW). [página oficial]
- Formatos: faces originais com fundo (~20 MB), **faces alinhadas 100×100** (~7,6 MB) com coordenadas dos olhos, e **recortes de olhos 24×24** (~2,6 MB; 4.846 recortes: 2.462 abertos, 2.384 fechados segundo rahman2020). [página oficial; rahman2020]
- Variações: iluminação, borrão, oclusão, disfarce. **Uso apenas para pesquisa, proibida redistribuição** sem permissão; contato x.tan / f.song @nuaa.edu.cn. Citação: song2014 (Pattern Recognition 47(9):2825–2838). [página oficial]

## DROZY — ULg Multimodality Drowsiness Database (Univ. de Liège)
Fonte: massoz2016 (WACV 2016, DOI 10.1109/WACV.2016.7477715) e ficha ORBi https://orbi.uliege.be/handle/2268/191620.
- Base multimodal para sonolência: imagens de **intensidade NIR e de profundidade** da face, sinais fisiológicos, testes PVT e KSS; exemplos de uso preveem velocidade de reação. [ORBi/abstract]
- **Acesso mediante License Agreement** (arquivo `DROZY-LicenseAgreement-20180718.pdf`, ~2,46 GB); portal http://www.drozy.ulg.ac.be. [ORBi]
- **14 sujeitos, 3 testes (sessões PVT) por sujeito**, 5 testes perdidos por falha de backup; KSS 1–9 por teste; NIR intensidade e profundidade 512×424 (16 bits) a 30 fps (testes 2–3 dos sujeitos 1–8 a 15 fps); PSG a 512 Hz (EEG Fz, Cz, C3, entre outros); 68 marcos faciais 2D/3D por quadro. [site oficial drozy.ulg.ac.be, lido em 20/09/2026] Número de sujeitos não conferido no PDF do WACV (o resumo não traz o número).

## YawDD — Yawning Detection Dataset (Univ. of Ottawa)
Fonte: abtahi2014 (MMSys 2014, pp. 24–28, DOI 10.1145/2557642.2563678) e ficha IEEE DataPort (DOI 10.21227/e1qm-hb90).
- Dois conjuntos: câmera no espelho retrovisor (**322 vídeos**) e câmera no painel (**29 vídeos**); homens e mulheres, com/sem óculos/óculos escuros, etnias variadas; iluminação natural variável. [DataPort]
- **640×480, 30 fps, RGB 24 bits, AVI sem áudio**; classes: direção normal (silêncio), falando/cantando, bocejando. [DataPort]
- **Acesso aberto** no IEEE DataPort (login gratuito); arquivo "Participants Information" indica quais vídeos permitem capturas públicas. [DataPort]

## DMD — Driver Monitoring Dataset (Vicomtech)
Fonte: ortega2020 (ECCV 2020 WS, LNCS 12538, DOI 10.1007/978-3-030-66823-5_23) e https://dmd.vicomtech.org/.
- **37 motoristas, 41 h** de vídeo **RGB, profundidade e IR** de **3 câmeras** (face, corpo, mãos; Intel RealSense D415 face/mãos e D435 corpo), em direção real e simulada; cenários de distração, olhar, sonolência (bocejo com/sem mão, microssono, direção sonolenta), mãos no volante e contexto. ~27 % mulheres, 73 % homens, idade média 30. [paper/abstract; site]
- **Acesso livre para fins acadêmicos** (links diretos, maiores de 18), licença **CC BY-NC-ND 4.0**; a versão pública atual inclui **apenas RGB** (revisão de licença/tamanho). [site]

## FL3D — Frame Level Driver Drowsiness Detection (Kaggle) **[não confirmado]**
- Dataset no Kaggle (usuário matjazmuc = Matjaz Muc) para classificar o estado do motorista por quadro (alerta, microssono, bocejo). A página fica atrás de reCAPTCHA para clientes automatizados; o resumo indexado (20/09/2026) informa **53.379 arquivos JSON, 628 MB**. **Origem dos vídeos, ano, licença e contagem de quadros por classe não foram confirmadas na página**. Não citar números no artigo sem abrir a página no navegador.

## NITYMED — Night Time Yawning-Microsleep-Eyeblink-driver Distraction (Univ. do Peloponeso)
Fonte: ficha IEEE DataPort https://ieee-dataport.org/documents/nitymed (DOI 10.21227/85xe-3f88); também em Kaggle e https://datasets.esdalab.ece.uop.gr/.
- **130 vídeos** gravados em Patras (Grécia), em carros reais à noite, **21 motoristas (11 homens, 10 mulheres)** com cabelo/barba/óculos variados; **25 fps**, mudos; **720p (1280×720, ~700 MB, Kaggle)** e **1080p (1920×1080, ~1,6 GB, site da universidade)**. [DataPort]
- Categorias: **bocejo (107 vídeos, 15–25 s, 3 bocejos cada)** e **microssono (21 vídeos, ~2 min, falando/olhando ao redor/microssonos)**. [DataPort]
- Acesso: no DataPort exige assinatura; versão 720p livre no Kaggle. Pede citar um dos artigos do grupo (Petrellis). Autoria completa/ano: não confirmados.

## Resumo de acesso
| Dataset | Tipo | Acesso |
|---|---|---|
| UTA-RLDD | real, RGB, celular/webcam | livre (Google Drive), citar ghoddoosian2019 |
| NTHU-DDD | atuado, IR 640×480 | termo de licença por e-mail (não obtido pela equipe) |
| MRL Eye | olhos IR, 37 suj. | livre |
| CEW | olhos/faces RGB in-the-wild | livre para pesquisa, sem redistribuição |
| DROZY | multimodal NIR+fisiológico | termo de licença |
| YawDD | bocejo RGB | aberto (IEEE DataPort, login) |
| DMD | multimodal, 37 motoristas | livre acadêmico, CC BY-NC-ND 4.0 (só RGB público) |
| NITYMED | noturno RGB | Kaggle livre (720p) / DataPort assinatura |
| FL3D | quadros rotulados (Kaggle) | página só no navegador (reCAPTCHA); detalhes não confirmados |
