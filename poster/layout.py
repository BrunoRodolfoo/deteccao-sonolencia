"""Geometria do pôster final (cm), compartilhada por make_assets.py (tamanho das figuras) e
build_poster.py (posição de cada caixa no .pptx e no .html). Template: 50,8 x 28,575 cm, 3 colunas.

As caixas de TEXTO são empilhadas pelo build_poster.py com a altura medida de cada uma (quebra de linha
feita com a Arial do sistema); aqui ficam só as posições fixas: colunas, figuras, tabela e cabeçalho.
"""
SLIDE_W, SLIDE_H = 50.8, 28.575

# Caixas brancas do template (x, y, largura, altura) e raio dos cantos (o mesmo do template)
CARDS = [(0.68, 0.48, 12.49, 5.47), (13.73, 0.46, 23.39, 5.47), (37.62, 0.48, 12.49, 5.47),
         (0.68, 6.69, 12.49, 21.39), (13.73, 6.69, 23.39, 21.39), (37.62, 6.69, 12.49, 21.39)]
CARD_R = 0.93

C1_X, C1_W = 1.15, 11.6
C2_X, C2_W = 14.25, 22.35
C3_X, C3_W = 38.0, 11.65
COL_TOP, COL_BOTTOM = 6.69, 28.08
BOTTOM = COL_BOTTOM - 0.3          # limite inferior do conteúdo

Y_SEC_TITLE = 7.05                 # títulos das colunas (28 pt, uma linha)
H_SEC_TITLE = 1.1
Y_BODY = Y_SEC_TITLE + H_SEC_TITLE + 0.1

# ---- Cabeçalho ---------------------------------------------------------------------------------
TITLE_X, TITLE_W = 13.83, 23.19     # título (32 pt) e endereço do repositório, centrados na caixa do meio
AUT_X, AUT_W = 38.15, 8.45          # autores
QR_SIDE = 2.9                       # QR do repositório, à direita dos autores
QR_X = 50.11 - 0.45 - QR_SIDE

# ---- Coluna 2 --------------------------------------------------------------------------------
DIAG_Y = Y_BODY
DIAG_W = C2_W
DIAG_H = 6.0
TAB_Y = DIAG_Y + DIAG_H + 0.35
TAB_ROW_H = 0.82
TAB_COLS = [6.6, 8.2, 7.55]            # base | o que é | papel (soma = C2_W)
TAB_ROWS = 4
LOW_Y = TAB_Y + TAB_ROWS * TAB_ROW_H + 0.35

F5B_W = 11.0                           # figura do PP1 (os rótulos ficam no próprio gráfico)
F5B_Y = LOW_Y
F5B_H = round(BOTTOM - F5B_Y, 2)

R_X = C2_X + F5B_W + 0.5               # textos à direita da figura (vazamento e app)
R_W = round(C2_X + C2_W - R_X, 2)
