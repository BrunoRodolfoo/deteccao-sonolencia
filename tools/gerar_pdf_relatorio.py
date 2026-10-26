"""Gera docs/ENTREGA02_relatorio.pdf a partir de docs/ENTREGA02_relatorio.md (markdown-it -> HTML -> Chrome headless mudo).

Uso (na raiz do repositório, no .venv):  .venv/bin/python tools/gerar_pdf_relatorio.py
Rodar de novo sempre que o relatório .md mudar (por exemplo, depois de um benchmark novo no celular).
O rodapé usa a data da linha "Versão de dd/mm/aaaa" do relatório.
"""
import html
import tempfile
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from markdown_it import MarkdownIt
import pymupdf

REPO = Path(__file__).resolve().parents[1]
TMP = Path(tempfile.mkdtemp(prefix="relatorio_pdf_"))
MD = REPO / "docs" / "ENTREGA02_relatorio.md"
OUT = REPO / "docs" / "ENTREGA02_relatorio.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

TITULO = "Entrega 02: Codificação, Metodologia Implementada e Resultados"
SUBTITULO = "Detecção de Sonolência em Dispositivos Móveis: Avaliação da Generalização entre Sonolência Atuada e Real"
AUTORES = ("André Luís Oliveira Ferreira da Silva, Bruno Rodolfo da Silveira, "
           "Bruno Sandoval Ribeiro, Paulo Henrique Perin")

texto = MD.read_text(encoding="utf-8")
m = re.search(r"[Vv]ersão de (\d{2}/\d{2}/\d{4})", texto)
VERSAO = m.group(1) if m else time.strftime("%d/%m/%Y")
md = MarkdownIt("commonmark", {"html": False, "typographer": False}).enable("table")
corpo = md.render(texto)

# URL do repositório clicável (o markdown-it do .venv não tem o plugin de linkify)
url = "https://github.com/BrunoRodolfoo/deteccao-sonolencia"
corpo = corpo.replace(url, f'<a href="{url}">{url}</a>', 1)

CSS = r"""
@page {
  size: A4;
  margin: 17mm 16mm 18mm 16mm;
  @bottom-left { content: "Equipe 14 · Entrega 02 · versão de __VERSAO__"; font: 7.5pt "Helvetica Neue", Arial, sans-serif; color: #666; }
  @bottom-right { content: "página " counter(page) " de " counter(pages); font: 7.5pt "Helvetica Neue", Arial, sans-serif; color: #666; }
}
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 9.6pt; line-height: 1.42; color: #111; margin: 0; }
h1 { font-size: 16.5pt; line-height: 1.2; margin: 0 0 5pt 0; }
h2 { font-size: 12.5pt; margin: 15pt 0 5pt 0; padding-bottom: 2pt; border-bottom: 0.7pt solid #888; break-after: avoid; }
h3 { font-size: 10.4pt; margin: 11pt 0 4pt 0; break-after: avoid; }
p { margin: 0 0 5pt 0; orphans: 3; widows: 3; }
ul, ol { margin: 2pt 0 6pt 0; padding-left: 16pt; }
li { margin: 1pt 0; }
li > ul { margin-top: 1pt; margin-bottom: 2pt; }
a { color: #111; text-decoration: none; }
code { font-family: Menlo, "SF Mono", Consolas, monospace; font-size: 0.86em; background: #f1f1f1; padding: 0 1.5pt; border-radius: 2pt; overflow-wrap: anywhere; }
pre { background: #f6f6f6; border: 0.5pt solid #c8c8c8; padding: 5pt 7pt; margin: 4pt 0 7pt 0; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 7.6pt; line-height: 1.35; }
blockquote { margin: 7pt 0 9pt 0; padding: 5pt 9pt; border-left: 2.2pt solid #777; background: #f4f4f4; }
blockquote p { margin: 0 0 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 8pt 0; font-size: 7.9pt; line-height: 1.3; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { border: 0.5pt solid #b8b8b8; padding: 2.2pt 4pt; vertical-align: top; text-align: left; }
th { background: #e9e9e9; font-weight: 600; }
td code, th code { font-size: 0.92em; overflow-wrap: break-word; }
a[href^="http"] { white-space: nowrap; }
strong { font-weight: 650; }
.capa { margin-bottom: 8pt; }
"""

pagina = f"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>{html.escape(TITULO)}</title>
<meta name="author" content="{html.escape(AUTORES)}">
<style>{CSS.replace("__VERSAO__", VERSAO)}</style>
</head>
<body>
{corpo}
</body>
</html>
"""
html_path = TMP / "ENTREGA02_relatorio.html"
html_path.write_text(pagina, encoding="utf-8")

pdf_tmp = TMP / "ENTREGA02_relatorio_chrome.pdf"
if pdf_tmp.exists():
    pdf_tmp.unlink()
perfil = TMP / "chrome-perfil"
cmd = [CHROME, "--headless", "--mute-audio", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
       f"--user-data-dir={perfil}", "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
       f"--print-to-pdf={pdf_tmp}", html_path.as_uri()]
proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
t0 = time.time()
while time.time() - t0 < 120:
    if proc.poll() is not None:
        break
    if pdf_tmp.exists() and pdf_tmp.stat().st_size > 0 and time.time() - t0 > 8:
        time.sleep(2)
        break
    time.sleep(0.5)
if proc.poll() is None:  # o Chrome headless às vezes grava o PDF e não termina
    os.killpg(proc.pid, signal.SIGTERM)
    time.sleep(1)
if not pdf_tmp.exists() or pdf_tmp.stat().st_size == 0:
    sys.exit("o Chrome não gerou o PDF")

doc = pymupdf.open(pdf_tmp)
doc.set_metadata({
    "title": TITULO,
    "author": AUTORES,
    "subject": SUBTITULO + ". Projeto Transformador II, Ciência da Computação, PUCPR, Turma 8ºA, Equipe 14. "
               f"Orientador: Prof. Júlio César Nievola. Versão de {VERSAO}.",
    "keywords": "detecção de sonolência; sonolência atuada; sonolência real; UTA-RLDD; DMD; NITYMED; Entrega 02; PUCPR",
    "creator": "docs/ENTREGA02_relatorio.md",
    "producer": "",
})
doc.save(TMP / "ENTREGA02_relatorio_final.pdf", garbage=3, deflate=True)
doc.close()
shutil.copyfile(TMP / "ENTREGA02_relatorio_final.pdf", OUT)

final = pymupdf.open(OUT)
print("páginas:", len(final))
print("metadados:", {k: v for k, v in final.metadata.items() if v})
txt = "".join(p.get_text() for p in final)
print("travessões no texto:", txt.count("—"), "| meias-riscas:", txt.count("–"))
prev = TMP / "previas"
if prev.exists():
    shutil.rmtree(prev)
prev.mkdir()
for i, p in enumerate(final):
    p.get_pixmap(dpi=80).save(prev / f"p{i + 1:02d}.png")
final.close()
print("prévias em", prev)
shutil.rmtree(TMP / "chrome-perfil", ignore_errors=True)
