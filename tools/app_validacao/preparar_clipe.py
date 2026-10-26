#!/usr/bin/env python3
"""Converte um trecho de vídeo das bases em .y4m (câmera falsa do Chrome) para validar a PWA.

    .venv/bin/python tools/app_validacao/preparar_clipe.py VIDEO SAIDA.y4m --inicio 0 --duracao 180 \
        [--preto-inicio 6] [--preto-fim 3] [--pitch "t0,t1,t2,t3,graus;..."] [--roll "t0,t1,t2,t3,graus;..."]

- 30 fps, 640x480 (paisagem) ou 480x640 (vídeo em retrato, como o celular em pé no suporte); o quadro é
  reduzido SEM distorcer e completado com faixas pretas (nada é cortado do rosto).
- --preto-inicio: segundos de tela preta ANTES do trecho. Servem para sincronizar: o primeiro quadro com
  rosto no app deve cair em t = preto-inicio no relógio do vídeo (conferido em cada execução).
- --pitch / --roll: teste SINTÉTICO de cabeça (tempos relativos ao início do TRECHO, em s). Cada
  episódio "t0,t1,t2,t3,g" sobe de 0 a g graus entre t0 e t1, fica em g até t2 e volta a 0 em t3.
  pitch = rotação 3D do plano da imagem em torno do eixo horizontal (com a cabeça caindo para a frente, a
  testa se aproxima da câmera e o queixo se afasta: topo ampliado, base reduzida; projeção em perspectiva,
  focal = largura) mais um deslocamento para baixo de 0,15*altura*sen(ângulo); roll = rotação no plano
  (cabeça tombando para o lado). É uma aproximação planar: o ângulo que o app mede é registrado e relatado.
Os .y4m contêm rosto: ficam SÓ numa pasta temporária, nunca no repositório; apague após o uso.
Usa o ffmpeg estático do imageio-ffmpeg (uv pip install imageio-ffmpeg no .venv).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: str) -> tuple[int, int]:
    """Largura e altura já com a rotação do contêiner aplicada (o ffmpeg auto-rotaciona)."""
    out = subprocess.run([FF, "-hide_banner", "-i", path], capture_output=True, text=True).stderr
    m = re.search(r"Video:.*?, (\d{2,5})x(\d{2,5})", out)
    w, h = int(m.group(1)), int(m.group(2))
    rot = re.search(r"rotation of (-?[\d.]+) degrees", out)
    if rot and abs(abs(float(rot.group(1))) - 90) < 1:
        w, h = h, w
    return w, h


def ramp_expr(eps: list[list[float]], tvar: str) -> str:
    """Expressão do ffmpeg para o ângulo (graus) em função do tempo, episódios trapezoidais."""
    parts = []
    for t0, t1, t2, t3, g in eps:
        up = f"{g}*({tvar}-{t0})/{max(t1 - t0, 1e-3)}"
        down = f"{g}*({t3}-{tvar})/{max(t3 - t2, 1e-3)}"
        parts.append(f"if(between({tvar},{t0},{t1}),{up},if(between({tvar},{t1},{t2}),{g},if(between({tvar},{t2},{t3}),{down},0)))")
    return "+".join(parts) if parts else "0"


def parse_eps(s: str | None) -> list[list[float]]:
    if not s:
        return []
    return [[float(x) for x in ep.split(",")] for ep in s.split(";") if ep.strip()]


def perspective_filter(eps, W, H, sign: int = 1) -> str:
    """Rotação 3D do plano da imagem em torno do eixo horizontal central (pitch), por quadro.
    sign=+1: topo aproxima-se da câmera (testa para a frente); sign=-1: topo afasta-se."""
    tv = "(in/30)"  # número do quadro -> segundos (30 fps)
    th = f"(({ramp_expr(eps, tv)})*PI/180)"
    f = W  # focal em pixels
    coords = {}
    for name, (X, Y) in {"0": (-W / 2, -H / 2), "1": (W / 2, -H / 2), "2": (-W / 2, H / 2), "3": (W / 2, H / 2)}.items():
        # cabeça caindo para a frente: o topo (Y<0) aproxima-se da câmera (Z<0) e a base afasta-se: Z = Y*sin(th)
        z = f"({sign * Y}*sin({th}))"
        coords[f"x{name}"] = f"{W / 2}+{f}*({X})/({f}+{z})"
        coords[f"y{name}"] = f"{H / 2}+{f}*({Y}*cos({th}))/({f}+{z})+{0.15 * H}*sin({th})"
    opts = ":".join(f"{k}='{v}'" for k, v in coords.items())
    return f"perspective={opts}:sense=destination:eval=frame:interpolation=linear"


def squash_filter(eps, W, H) -> str:
    """Achatamento vertical (altura x cos(ângulo)) + deslocamento para baixo (0,15*altura*sen), por quadro:
    a projeção ortográfica de um rosto plano que gira para baixo."""
    tv = "(in/30)"
    th = f"(({ramp_expr(eps, tv)})*PI/180)"
    c = {}
    for name, (X, Y) in {"0": (-W / 2, -H / 2), "1": (W / 2, -H / 2), "2": (-W / 2, H / 2), "3": (W / 2, H / 2)}.items():
        c[f"x{name}"] = f"{W / 2 + X}"
        c[f"y{name}"] = f"{H / 2}+({Y})*cos({th})+{0.15 * H}*sin({th})"
    return "perspective=" + ":".join(f"{k}='{v}'" for k, v in c.items()) + ":sense=destination:eval=frame:interpolation=linear"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("saida")
    ap.add_argument("--inicio", type=float, default=0.0)
    ap.add_argument("--duracao", type=float, required=True)
    ap.add_argument("--preto-inicio", type=float, default=6.0)
    ap.add_argument("--preto-fim", type=float, default=3.0)
    ap.add_argument("--pitch", default=None)
    ap.add_argument("--roll", default=None)
    ap.add_argument("--pitch-longe", default=None, help="como --pitch, mas com o topo afastando-se da câmera")
    ap.add_argument("--achatar", default=None, help="episódios de achatamento vertical + descida (ver squash_filter)")
    a = ap.parse_args()
    w, h = probe(a.video)
    W, H = (480, 640) if h > w else (640, 480)
    vf = [f"fps=30", f"scale={W}:{H}:force_original_aspect_ratio=decrease", f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black", "setsar=1"]
    pe, re_ = parse_eps(a.pitch), parse_eps(a.roll)
    if pe:
        vf.append(perspective_filter(pe, W, H))
    pl, sq = parse_eps(a.pitch_longe), parse_eps(a.achatar)
    if pl:
        vf.append(perspective_filter(pl, W, H, sign=-1))
    if sq:
        vf.append(squash_filter(sq, W, H))
    if re_:
        vf.append(f"rotate=a='({ramp_expr(re_, 't')})*PI/180':fillcolor=black")
    vf.append("format=yuv420p")
    fc = (
        f"color=black:s={W}x{H}:r=30:d={a.preto_inicio},format=yuv420p,setsar=1[a];"
        f"[0:v]{','.join(vf)}[b];"
        f"color=black:s={W}x{H}:r=30:d={a.preto_fim},format=yuv420p,setsar=1[c];"
        f"[a][b][c]concat=n=3:v=1:a=0[v]"
    )
    cmd = [FF, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(a.inicio), "-t", str(a.duracao), "-i", a.video,
           "-filter_complex", fc, "-map", "[v]", "-an", "-pix_fmt", "yuv420p", "-f", "yuv4mpegpipe", a.saida]
    subprocess.run(cmd, check=True)
    print(json.dumps({"saida": a.saida, "largura": W, "altura": H, "origem": [w, h], "inicio_s": a.inicio, "duracao_s": a.duracao,
                      "preto_inicio_s": a.preto_inicio, "preto_fim_s": a.preto_fim, "pitch": pe, "pitch_longe": pl, "achatar": sq, "roll": re_}))


if __name__ == "__main__":
    sys.exit(main())
