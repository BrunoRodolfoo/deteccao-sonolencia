#!/usr/bin/env python
"""Gera as figuras do PÔSTER FINAL em poster/assets/ (PNG 300 dpi, já no tamanho físico do pôster).

Nenhum número é digitado aqui: todos vêm de poster/numeros.json (gerado por
src/experiments/make_paper_numbers.py a partir de results/, conferido por conferir_numeros.py).
Os valores plotados são o campo `bruto` x `escala` e o IC é `ic_bruto` x `escala` de cada entrada.

Figuras (tamanho em cm = tamanho no pôster, lido de poster/layout.py; texto em pt reais):
  fig_poster_metodo.png      diagrama do método: câmera e MediaPipe, sinais (olhos, bocejo, cabeça),
                             experimentos (Face Mesh, 468 pontos; janelas de 10 s; RF e RL) e
                             app v2.1 (Face Landmarker, 478 pontos; nível de 0 a 100)
  fig_poster_exp5b.png       PP1 (Exp. 5b): perda de AUC ao treinar com sono encenado (B − C; RF e RL;
                             normalização principal e as outras três) e quanto o modelo do DMD reconhece
                             das janelas alertas e sonolentas reais
  qr_repositorio.png         QR do repositório

Regras conferidas antes de salvar cada figura (o script para com erro se falharem):
  * nenhum texto com menos de 20 pt (FS_MIN);
  * nenhum texto encostado em outro nem fora da figura;
  * no diagrama, cada texto cabe na sua caixa.
Cores: cinzas e o vermelho do template; verde e amarelo só no semáforo da caixa do app.

Uso:  python poster/make_assets.py   (depois: python poster/build_poster.py --pdf)
"""
import importlib.util
import json
import pathlib
import sys

sys.dont_write_bytecode = True   # não cria __pycache__ ao carregar make_figures.py e layout.py

ROOT = pathlib.Path(__file__).resolve().parents[1]
POSTER = ROOT / "poster"
ASSETS = POSTER / "assets"
SRC = ROOT / "src" / "experiments" / "make_figures.py"

spec = importlib.util.spec_from_file_location("make_figures", SRC)
mf = importlib.util.module_from_spec(spec)
sys.modules["make_figures"] = mf
spec.loader.exec_module(mf)
plt = mf.plt
from matplotlib.patches import Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402

# Geometria compartilhada com build_poster.py
spec_b = importlib.util.spec_from_file_location("poster_layout", POSTER / "layout.py")
LY = importlib.util.module_from_spec(spec_b)
spec_b.loader.exec_module(LY)

NUM = json.loads((POSTER / "numeros.json").read_text(encoding="utf-8"))
E = {e["nome"]: e for e in NUM["entradas"]}
V = NUM["valores"]


def val(nome):
    e = E[nome]
    return e["bruto"] * e["escala"]


def ic(nome):
    e = E[nome]
    lo, hi = e["ic_bruto"]
    return lo * e["escala"], hi * e["escala"]


CM = 1 / 2.54
FS = 20          # todo texto das figuras (pt); o pôster pede no mínimo 20
FS_MIN = 20
LS = 1.1         # espaçamento entre linhas do texto de várias linhas
# Paleta: cinzas + vermelho do template (mesmos tons do build_poster.py)
PRETO = "#1A1A1A"
CINZA_ESC = "#4D4D4D"
CINZA = "#808080"
CINZA_CLA = "#B3B3B3"
GRADE = "#E6E6E6"
RED = "#D52128"
VERDE = "#2E9E5B"       # só no semáforo do app
AMARELO = "#E0A100"     # só no semáforo do app


def setup():
    mf.setup_style()
    plt.rcParams.update({
        "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS,
        "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
        "axes.edgecolor": CINZA, "xtick.color": CINZA, "ytick.color": CINZA,
        "xtick.labelcolor": PRETO, "ytick.labelcolor": PRETO, "grid.color": GRADE,
        "axes.linewidth": 1.2, "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "xtick.major.size": 5, "ytick.major.size": 5, "grid.linewidth": 1.0,
        "figure.dpi": 300,       # as conferências medem o texto na mesma resolução da imagem salva
    })


def conferir(fig, nome, folga_cm=0.1):
    """Para com erro se algum texto tiver menos de FS_MIN pt, sair da figura ou encostar em outro."""
    folga_px = folga_cm / 2.54 * fig.dpi
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    caixas = []
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        if t.get_fontsize() < FS_MIN:
            raise SystemExit(f"{nome}: texto {t.get_text()!r} com {t.get_fontsize()} pt (< {FS_MIN})")
        bb = t.get_window_extent(r)
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            raise SystemExit(f"{nome}: texto {t.get_text()!r} sai da figura ({bb})")
        caixas.append((t.get_text(), bb))
    for i, (ta, a) in enumerate(caixas):
        for tb, b in caixas[i + 1:]:
            if (a.x0 < b.x1 + folga_px and b.x0 < a.x1 + folga_px and
                    a.y0 < b.y1 + folga_px and b.y0 < a.y1 + folga_px):
                raise SystemExit(f"{nome}: textos encostados: {ta!r} e {tb!r}")
    return r


def save(fig, name):
    out = ASSETS / f"{name}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(out).size
    print(f"{name}: {w}x{h} px = {w / 300 * 2.54:.2f} x {h / 300 * 2.54:.2f} cm -> {out}")


# ------------------------------------------------------------------------------------------------
# 1. Diagrama do método (texto em pt reais; coordenadas em cm)
# ------------------------------------------------------------------------------------------------
def caixa(ax, x, y, w, h, ec=CINZA_ESC, lw=1.4):
    ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=ec, lw=lw, zorder=2))


def seta(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), zorder=3,
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.28", lw=1.6, color=CINZA_ESC,
                                shrinkA=0, shrinkB=0))


def fig_metodo():
    W, H = LY.DIAG_W, LY.DIAG_H
    fig = plt.figure(figsize=(W * CM, H * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    LH = FS * LS / 72 * 2.54     # altura de uma linha de texto (cm)
    px, py = 0.25, 0.13          # margem interna das caixas (cm)
    seta_w = 0.55                # espaço das setas entre as faixas
    lado = 0.04
    dentro = []                  # (texto, caixa) para conferir que cada texto cabe na sua caixa

    def txt(x, y, s, cx, **kw):
        t = ax.text(x, y, s, fontsize=FS, color=kw.pop("color", PRETO), linespacing=LS, zorder=4, **kw)
        dentro.append((t, cx))
        return t

    def larg(t):
        fig.canvas.draw()
        return t.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted()).width

    # faixa 1: captura (centrada na altura)
    w1 = 3.7
    c1 = (lado, lado, w1, H - 2 * lado)
    caixa(ax, *c1)
    xm = lado + w1 / 2
    y0 = H / 2 + 2.5 * LH
    txt(xm, y0, "Câmera", c1, ha="center", va="top", weight="bold")
    seta(ax, xm, y0 - LH - 0.05, xm, y0 - 2 * LH + 0.1)
    txt(xm, y0 - 2 * LH, "MediaPipe\n(pontos\ndo rosto)", c1, ha="center", va="top")

    # faixa 3: saídas; faixa 2: sinais que o app mostra, com a validação de cada um (numeros.json)
    w3 = 8.15
    x3 = W - lado - w3
    x2 = lado + w1 + seta_w
    w2 = x3 - seta_w - x2
    sinais = [
        ("Olhos", "fechados ≥ 0,5 s", f"κ = {V['AppKappaOffline']}, revocação {V['AppFechMeioSRevocacao']}"),
        ("Bocejo", "boca aberta ≥ 1,5 s", f"revocação {V['AppBocejoRevocacaoSemMao']} sem mão"),
        ("Cabeça", "queda ou cabeceio", "revocação não medida"),
    ]
    gap = (H - 2 * lado - 3 * (2 * py + 2 * LH)) / 2
    assert gap >= 0.1, f"diagrama baixo demais para os sinais (folga {gap:.2f} cm)"
    hb = 2 * py + 2 * LH
    centros = []
    for i, (nome, regra, valid) in enumerate(sinais):
        y = H - lado - (i + 1) * hb - i * gap
        cx = (x2, y, w2, hb)
        caixa(ax, *cx)
        t = txt(x2 + px, y + hb - py, nome, cx, ha="left", va="top", weight="bold")
        txt(x2 + px + larg(t) + 0.22, y + hb - py, regra, cx, ha="left", va="top")
        txt(x2 + px, y + hb - py - LH, valid, cx, ha="left", va="top", color=CINZA_ESC)
        centros.append(y + hb / 2)
    seta(ax, lado + w1, H / 2, x2, H / 2)

    # saída 1: experimentos (análise offline, Exp. 5b e 6): Face Mesh do MediaPipe, 468 pontos
    h_out = 2 * py + 3 * LH
    y_exp = H - lado - h_out
    ce = (x3, y_exp, w3, h_out)
    caixa(ax, *ce)
    txt(x3 + px, y_exp + h_out - py, "Experimentos", ce, ha="left", va="top", weight="bold")
    txt(x3 + px, y_exp + h_out - py - LH, "Face Mesh, 468 pontos\njanelas de 10 s, RF e RL", ce, ha="left",
        va="top")
    # saída 2: app no celular: Face Landmarker do MediaPipe, 478 pontos; nível de 0 a 100 (semáforo)
    y_app = lado
    ca = (x3, y_app, w3, h_out)
    assert y_app + h_out <= y_exp - 0.25, "diagrama baixo demais para as duas saídas"
    caixa(ax, *ca)
    t = txt(x3 + px, y_app + h_out - py, f"App v{V['AppVersao']}", ca, ha="left", va="top", weight="bold")
    txt(x3 + px, y_app + h_out - py - LH, "Face Landmarker,\n478 pontos", ca, ha="left", va="top")
    # semáforo ao lado do título (limiares do app: AppLimiarAmarelo / AppLimiarVermelho)
    am, vm = int(V["AppLimiarAmarelo"]), int(V["AppLimiarVermelho"])
    bx = x3 + px + larg(t) + 0.35
    bw = x3 + w3 - px - bx
    bh = LH - 0.04
    by = y_app + h_out - py - bh
    for a, b, cor in ((0, am, VERDE), (am, vm, AMARELO), (vm, 100, RED)):
        ax.add_patch(Rectangle((bx + bw * a / 100, by), bw * (b - a) / 100, bh, fc=cor, ec="white", lw=1.5,
                               zorder=3))
    for lim, cor in ((am, PRETO), (vm, "white")):
        txt(bx + bw * lim / 100 + 0.12, by + bh / 2, str(lim), ca, ha="left", va="center", color=cor)

    xa = x2 + w2
    seta(ax, xa, centros[0], x3, y_exp + h_out / 2)
    seta(ax, xa, centros[2], x3, y_app + h_out / 2)

    # cada texto cabe na sua caixa (com 0,1 cm de folga dos lados)
    r = conferir(fig, "fig_poster_metodo", folga_cm=0.025)
    inv = ax.transData.inverted()
    for t, (x, y, w, h) in dentro:
        bb = t.get_window_extent(r).transformed(inv)
        if bb.x0 < x + 0.1 or bb.x1 > x + w - 0.1 or bb.y0 < y + 0.03 or bb.y1 > y + h - 0.03:
            raise SystemExit(f"fig_poster_metodo: {t.get_text()!r} não cabe na caixa "
                             f"({bb.x0:.2f}..{bb.x1:.2f} x {bb.y0:.2f}..{bb.y1:.2f} em {x:.2f}+{w:.2f}, {y:.2f}+{h:.2f})")
    save(fig, "fig_poster_metodo")


# ------------------------------------------------------------------------------------------------
# 2. PP1 (Exp. 5b): queda B − C da AUC por modelo e normalização + acertos do modelo do DMD
# ------------------------------------------------------------------------------------------------
QUEDAS = {   # (modelo, normalização) -> macro da queda B − C da AUC por janela (pontos, com IC)
    ("RF", "principal"): "ExpCincoBQuedaAUCJanela",
    ("RF", "outra referência"): "ExpCincoBRefAltQuedaAUCJanela",       # outra referência na UTA-RLDD
    ("RF", "sem centragem"): "ExpCincoBSemNormQuedaAUCJanela",
    ("RF", "referência do DMD"): "ExpCincoBRefSDoisQuedaAUCJanela",    # referência do DMD na s2
    ("RL", "principal"): "ExpCincoBLRQuedaAUCJanela",
    ("RL", "outra referência"): "ExpCincoBLRRefAltQuedaAUCJanela",
    ("RL", "sem centragem"): "ExpCincoBLRSemNormQuedaAUCJanela",
    ("RL", "referência do DMD"): "ExpCincoBLRRefSDoisQuedaAUCJanela",
}


def fig_exp5b():
    """Dois gráficos empilhados: em cima, a perda de AUC ao treinar com sono encenado (B − C) no RF e na
    RL, na normalização principal e nas outras três (outra referência na UTA-RLDD, sem centragem e
    referência do DMD na s2); embaixo, quanto o modelo do DMD (C) reconhece das janelas alertas e
    sonolentas reais (limiar 0,5)."""
    W, H = LY.F5B_W, LY.F5B_H
    fig = plt.figure(figsize=(W * CM, H * CM))
    LH = FS * LS / 72 * 2.54
    # --- em cima: perda B − C ---
    fig.text(0.1 / W, 1 - 0.05 / H, "Perda ao treinar com sono\nencenado (pontos de AUC)", fontsize=FS, weight="bold",
             color=PRETO, ha="left", va="top", linespacing=LS)
    topo = 0.05 + 2 * LH + 0.2              # cm a partir do alto
    base_barras = 2 * 0.8 + 0.2              # duas linhas de barras + folga
    titulo2 = LH + 0.12
    ticks = 2 * LH + 0.4
    h_ax = H - topo - ticks - 0.25 - titulo2 - base_barras
    assert h_ax >= 2.6, f"figura do PP1 baixa demais ({h_ax:.2f} cm para o gráfico de cima)"
    ax_x0 = 1.1
    ax = fig.add_axes([ax_x0 / W, (H - topo - h_ax) / H, (W - ax_x0 - 0.15) / W, h_ax / H])
    pos = {("RF", "principal"): -0.24, ("RL", "principal"): 0.24,
           ("RF", "outra referência"): 0.78, ("RF", "sem centragem"): 0.92, ("RF", "referência do DMD"): 1.06,
           ("RL", "outra referência"): 1.26, ("RL", "sem centragem"): 1.4, ("RL", "referência do DMD"): 1.54}
    estilo = {"RF": ("o", PRETO), "RL": ("s", CINZA)}
    for (modelo, norm), nome in QUEDAS.items():
        v, (lo, hi) = val(nome), ic(nome)
        mk, cor = estilo[modelo]
        ax.errorbar(pos[(modelo, norm)], v, yerr=[[v - lo], [hi - v]], fmt=mk, ms=10, color=cor, mec="white",
                    mew=1.0, elinewidth=2.2, capsize=4, capthick=2.2, zorder=4)
    ax.axhline(0, color=CINZA_ESC, lw=1.4, zorder=2)
    ax.set_ylim(-3, 17.5)
    ax.set_yticks([0, 10])
    ax.set_xlim(-0.8, 1.75)
    ax.set_xticks([0, 1.16])
    ax.set_xticklabels(["a principal", "as outras três"])
    ax.set_xlabel("normalização por pessoa", fontsize=FS, labelpad=7)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.grid(True, axis="y", color=GRADE, linewidth=1.0)
    for modelo, sinal in (("RF", -1), ("RL", 1)):
        nome = QUEDAS[(modelo, "principal")]
        ax.text(pos[(modelo, "principal")] + sinal * 0.11, val(nome), modelo, ha="right" if sinal < 0 else "left",
                va="center", fontsize=FS, weight="bold", color=estilo[modelo][1])

    # --- embaixo: o modelo do DMD na UTA-RLDD ---
    y_t2 = base_barras + titulo2              # cm a partir de baixo (topo do título de baixo)
    fig.text(0.1 / W, y_t2 / H, "Modelo do DMD reconhece", fontsize=FS, weight="bold", color=PRETO,
             ha="left", va="top")
    bx = fig.add_axes([0, 0, 1, base_barras / H])
    bx.set_xlim(0, W)
    bx.set_ylim(0, base_barras)
    bx.axis("off")
    x0, larg = 3.4, W - 3.4 - 2.4        # início da barra (depois do rótulo) e comprimento de 100 %
    for k, (rot, nome, cor) in enumerate((("alerta", "ExpCincoBCEspecJanela", CINZA_CLA),
                                          ("sono real", "ExpCincoBCSensJanela", RED))):
        yc = base_barras - 0.1 - 0.8 * k - 0.8 / 2
        v, (lo, hi) = val(nome), ic(nome)
        bx.barh(yc, larg * v / 100, left=x0, height=0.5, color=cor, zorder=3)
        bx.errorbar(x0 + larg * v / 100, yc, xerr=[[larg * (v - lo) / 100], [larg * (hi - v) / 100]], fmt="none",
                    ecolor=PRETO, elinewidth=2.0, capsize=4, capthick=2.0, zorder=4)
        bx.text(0.1, yc, rot, ha="left", va="center", fontsize=FS, color=PRETO)
        bx.text(x0 + larg * hi / 100 + 0.25, yc, f"{V[nome]}%", ha="left", va="center", fontsize=FS, weight="bold",
                color=PRETO)
    conferir(fig, "fig_poster_exp5b")
    # os rótulos não podem cobrir os pontos e os IC do gráfico de cima
    r = fig.canvas.get_renderer()
    marcas = [c.get_window_extent(r) for c in ax.collections + ax.lines if c.get_visible()]
    for t in ax.texts:
        tb = t.get_window_extent(r)
        for m in marcas:
            if m.width > 0.6 * fig.bbox.width:
                continue    # linhas de grade e do zero
            if tb.x0 < m.x1 and m.x0 < tb.x1 and tb.y0 < m.y1 and m.y0 < tb.y1:
                raise SystemExit(f"fig_poster_exp5b: rótulo {t.get_text()!r} cobre um ponto do gráfico")
    save(fig, "fig_poster_exp5b")


def qr_repositorio():
    import qrcode
    img = qrcode.make(REPO_URL_QR, border=1)
    img.save(ASSETS / "qr_repositorio.png")
    print(f"qr_repositorio.png -> {REPO_URL_QR}")


REPO_URL_QR = "https://github.com/BrunoRodolfoo/deteccao-sonolencia"

if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    setup()
    qr_repositorio()
    fig_metodo()
    fig_exp5b()
