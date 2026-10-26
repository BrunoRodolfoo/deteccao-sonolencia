# Verificação das fontes brasileiras (Introdução do artigo)

Verificação em 20/09/2026, a partir das URLs do planejamento da Entrega 01 (`project_planning.tex`).

## 1. ABRAMET / Agência Brasil (maio de 2026)

**URL do planejamento:** https://agenciabrasil.ebc.com.br/saude/noticia/2026-05/transtornos-de-saude-causaram-28-dos-sinistros-em-rodovias-do-pais

- **Resolve?** Sim (HTTP 200). Título: "Transtornos de saúde causaram 28% dos sinistros em rodovias do país". Repórter: Paula Laboissière. Data: **12/05/2026**.
- **Números batem?** Sim. A matéria informa que o levantamento da Associação Brasileira de Medicina de Tráfego (Abramet), com base em registros da PRF, associa **1.206.491 sinistros** a problemas de saúde do condutor, ou **27,8 % das 4.339.762 ocorrências** registradas em rodovias federais **entre 2014 e 2024** (arredondado para 28 % no título). Outros números da mesma matéria: 49 % (2.144.175) ligados ao comportamento do condutor ("fator humano"); 8 % à via; 7 % à manutenção do veículo; 4 % ao ambiente; fator humano + saúde ≈ 80 % das ocorrências.
- **O que a categoria "saúde" inclui (citação literal):** "ausência de reação, sono, falta de atenção, transtornos mentais, mal súbito e uso de substâncias". Portanto o número **não isola a sonolência**; o texto do planejamento já faz essa ressalva corretamente.
- Reproduções que confirmam os mesmos números: Portal do Trânsito ("Quase 1 em cada 3 sinistros nas rodovias tem relação com a saúde do motorista"), Medicina S/A, Amazonas Atual.

**Redação sugerida (pt-BR):**
> Levantamento da Associação Brasileira de Medicina de Tráfego (ABRAMET) sobre os registros da Polícia Rodoviária Federal associa 1.206.491 sinistros em rodovias federais entre 2014 e 2024 — 27,8 % das 4.339.762 ocorrências do período — a condições de saúde do condutor, categoria que agrega ausência de reação, sono, falta de atenção, transtornos mentais, mal súbito e uso de substâncias, sem isolar a sonolência [abramet2026].

## 2. PRF — Anuário Estatístico de 2025 (maio de 2026)

**URL do planejamento:** https://www.gov.br/prf/pt-br/noticias/nacionais/2026/maio/prf-divulga-anuario-estatistico-de-2025-com-aumento-de-infracoes-e-reducao-de-mortes-nas-rodovias-federais

- **Resolve?** A URL existe e aparece indexada com o título exato, mas o portal gov.br devolveu **HTTP 401** a todos os clientes automatizados usados aqui (WebFetch e curl com user-agent de navegador). Deve abrir normalmente em navegador; **recomenda-se conferir manualmente no navegador antes da entrega** e manter a data de acesso. Observação: a Agência Gov (agenciagov.ebc.com.br) está com conteúdo suspenso por legislação eleitoral, então não serve como espelho no momento.
- **Números batem?** Sim, conferidos em duas reproduções integrais da notícia da PRF:
  - ABTLP — "PRF divulga anuário estatístico de 2025 com aumento de infrações e redução de mortes nas rodovias federais" (reprodução do texto oficial): https://www.abtlp.org.br/index.php/prf-divulga-anuario-estatistico-de-2025-com-aumento-de-infracoes-e-reducao-de-mortes-nas-rodovias-federais/
  - Portal do Trânsito, 26/05/2026, "Anuário da PRF revela mais de 6 mil mortes em rodovias federais em 2025": https://www.portaldotransito.com.br/noticias/fiscalizacao-e-legislacao/estatisticas/anuario-da-prf-revela-mais-de-6-mil-mortes-em-rodovias-federais-em-2025/ — citações literais: "o país ainda registrou **72.529 sinistros, 83.550 pessoas feridas e 6.043 mortes** nas BRs federais ao longo do último ano" e "Já a '**reação tardia ou ineficiente**' apareceu em **10.799 sinistros**".
- **Atenção à interpretação:** "reação tardia ou ineficiente do condutor" é uma das causas registradas pela PRF; **não é sinônimo de sonolência** (a PRF tem a causa específica "condutor dormindo", cujo número não consta na notícia). Não afirmar que os 10.799 sinistros foram causados por sono.

**Redação sugerida (pt-BR):**
> O Anuário Estatístico da PRF referente a 2025 registra 72.529 sinistros, 83.550 feridos e 6.043 mortes em rodovias federais; a causa "reação tardia ou ineficiente do condutor" — compatível com, mas não restrita a, estados de sonolência — aparece em 10.799 desses sinistros [prf2026].

Se o professor exigir fonte primária acessível por máquina, o próprio anuário (PDF/painel de dados abertos da PRF em https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos) pode substituir a notícia; não foi possível baixá-lo pelo mesmo bloqueio 401.

## 3. Reverificação (revisão crítica, 20/09/2026)
- Agência Brasil: HTTP 200; os valores 1.206.491, 27,8 %, 4.339.762, a autora (Paula Laboissière), a data (12/05/2026) e a lista literal "ausência de reação, sono, falta de atenção, transtornos mentais, mal súbito e uso de substâncias" aparecem na página.
- gov.br/PRF: continua HTTP 401 para clientes automatizados. Portal do Trânsito (datePublished 2026-05-26) e ABTLP: HTTP 200, ambos com 72.529 sinistros, 83.550 feridos, 6.043 mortes e "reação tardia ou ineficiente do condutor" em 10.799 sinistros (a ABTLP lista também "ausência de reação do condutor" com 11.469).

## 4. Ajustes no .bib
- `abramet2026`: mantida a URL; autora pessoal (Paula Laboissière) no campo `author` e "Agência Brasil" em `howpublished`; data (12/05/2026) e nota de que a categoria agrega sono e outros fatores.
- `prf2026`: mantida a URL oficial; adicionada nota de que responde 401 a clientes automatizados e de que os números foram conferidos nas reproduções ABTLP e Portal do Trânsito (26/05/2026). Atualizar "acesso em" para a data em que a equipe abrir a página no navegador.
