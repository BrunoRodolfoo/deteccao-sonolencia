"""Confere as figuras do artigo: imprime largura x altura (cm) e o menor tamanho de fonte (pt) de cada PDF.

Uso: python render.py [--png PASTA] [arquivo.pdf ...]   (sem arquivos: todos os fig_*.pdf desta pasta)
Com --png, grava também uma prévia em PNG a 300 dpi na PASTA indicada; esta pasta guarda só o que o artigo
inclui. O menor tamanho de fonte vale para o artigo porque as figuras entram nele sem redução."""
import argparse
import glob
from pathlib import Path

import pymupdf

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("pdfs", nargs="*", help="PDFs a conferir (padrão: fig_*.pdf)")
ap.add_argument("--png", type=Path, default=None, metavar="PASTA", help="grava prévias PNG de 300 dpi nesta pasta")
args = ap.parse_args()
for f in args.pdfs or sorted(glob.glob("fig_*.pdf")):
    p = pymupdf.open(f)[0]
    tamanhos = [s["size"] for b in p.get_text("dict")["blocks"] for linha in b.get("lines", [])
                for s in linha["spans"] if s["text"].strip()]
    msg = f"{f}: {p.rect.width / 72 * 2.54:.2f} x {p.rect.height / 72 * 2.54:.2f} cm"
    if tamanhos:
        msg += f"; menor fonte {min(tamanhos):.1f} pt"
    if args.png:
        args.png.mkdir(parents=True, exist_ok=True)
        out = args.png / f"{Path(f).stem}.png"
        p.get_pixmap(dpi=300).save(out)
        msg += f" -> {out}"
    print(msg)
