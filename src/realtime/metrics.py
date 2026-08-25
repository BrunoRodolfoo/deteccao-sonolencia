"""Métricas faciais para detecção de sonolência (Pipeline A).

EAR  — Eye Aspect Ratio (Soukupová & Čech, CVWW 2016)
MAR  — Mouth Aspect Ratio (bocejo)
PERCLOS — percentagem de tempo com olhos fechados em janela deslizante
"""

from collections import deque

import numpy as np

# Índices dos landmarks do MediaPipe Face Mesh (468 pontos).
# Ordem p1..p6 da fórmula do EAR: canto externo, 2 pontos da pálpebra
# superior, canto interno, 2 pontos da pálpebra inferior.
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Boca: cantos (61, 291) e par vertical interno (13, 14).
MOUTH_CORNERS = (61, 291)
MOUTH_VERTICAL = (13, 14)


def _dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b)))


def eye_aspect_ratio(pts):
    """EAR de um olho a partir dos 6 pontos [p1..p6].

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Olho aberto ~0.25-0.35; fechado tende a ~0.
    """
    p1, p2, p3, p4, p5, p6 = pts
    horiz = _dist(p1, p4)
    if horiz < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horiz)


def ear_from_landmarks(landmarks, frame_w, frame_h):
    """EAR médio dos dois olhos a partir dos landmarks do Face Mesh."""

    def pts(idxs):
        return [(landmarks[i].x * frame_w, landmarks[i].y * frame_h) for i in idxs]

    left = eye_aspect_ratio(pts(LEFT_EYE))
    right = eye_aspect_ratio(pts(RIGHT_EYE))
    return (left + right) / 2.0


def mar_from_landmarks(landmarks, frame_w, frame_h):
    """MAR: abertura vertical da boca normalizada pela largura."""

    def pt(i):
        return (landmarks[i].x * frame_w, landmarks[i].y * frame_h)

    horiz = _dist(pt(MOUTH_CORNERS[0]), pt(MOUTH_CORNERS[1]))
    if horiz < 1e-6:
        return 0.0
    return _dist(pt(MOUTH_VERTICAL[0]), pt(MOUTH_VERTICAL[1])) / horiz


class Perclos:
    """PERCLOS em janela deslizante de `window_s` segundos.

    Proporção de quadros com olho fechado (EAR abaixo do limiar) na janela.
    Valores sustentados acima de ~0.15 indicam sonolência na literatura.
    """

    def __init__(self, window_s=60.0):
        self.window_s = window_s
        self._events = deque()  # (timestamp, olho_fechado: bool)

    def update(self, t, closed):
        self._events.append((t, bool(closed)))
        while self._events and t - self._events[0][0] > self.window_s:
            self._events.popleft()
        return self.value

    @property
    def value(self):
        if not self._events:
            return 0.0
        return sum(c for _, c in self._events) / len(self._events)
