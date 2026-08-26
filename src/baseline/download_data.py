"""Baixa as blink features e partições oficiais do UTA-RLDD (CVPRW 2019).

Fonte: https://github.com/rezaghoddoosian/Early-Drowsiness-Detection
Destino: data/uta_rldd_blinks/ (~40 MB, 20 arquivos .npy)
"""

from pathlib import Path

import requests

BASE = ("https://raw.githubusercontent.com/rezaghoddoosian/"
        "Early-Drowsiness-Detection/master")
DEST = Path(__file__).resolve().parents[2] / "data" / "uta_rldd_blinks"

FILES = [f"{kind}_30_Fold{k}.npy"
         for kind in ("Blinks", "BlinksTest", "Labels", "LabelsTest")
         for k in range(1, 6)]


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        out = DEST / name
        if out.exists():
            print(f"[ok] {name} (já existe)")
            continue
        r = requests.get(f"{BASE}/{name}", timeout=60)
        r.raise_for_status()
        out.write_bytes(r.content)
        print(f"[baixado] {name} ({len(r.content) / 1024:.0f} KB)")
    print(f"\nConcluído em {DEST}")


if __name__ == "__main__":
    main()
