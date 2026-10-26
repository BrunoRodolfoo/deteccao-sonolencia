"""Carregamento dos datasets de estado ocular (MRL Eye e CEW).

Convenções comuns a todo o pipeline B:
  * rótulo 0 = olho FECHADO, 1 = olho ABERTO (mesma codificação do MRL Eye);
  * imagens em escala de cinza, 48x48, float32 em [0, 1] (na hora do treino);
    o cache em disco guarda uint8 para economizar espaço.

Datasets
--------
MRL Eye (Fusek, ISVC 2018): 84.898 recortes infravermelhos de olhos de 37
sujeitos. O nome do arquivo codifica os metadados:
  sXXXX_YYYYY_G_GL_ES_R_L_SS.png
  sujeito, id da imagem, gênero (0 homem/1 mulher), óculos (0/1),
  estado (0 fechado/1 aberto), reflexos (0 nenhum/1 pequeno/2 grande),
  iluminação (0 ruim/1 boa), sensor (01 RealSense/02 IDS/03 Aptina).

CEW — Closed Eyes in the Wild (Song et al., Pattern Recognition 2014): 2.423
faces (1.192 de olhos fechados, da internet; 1.231 abertos, do LFW), RGB.
Usamos dois tipos de recorte:
  * "patch": os recortes 24x24 oficiais (dataset_B_Eye_Images), ampliados
    para 48x48; nome closed_eye_0104.jpg_face_2_R.jpg -> face de origem
    "closed_eye_0104.jpg_face_2", lado "R";
  * "crop": recortes que nós mesmos extraímos das faces 100x100
    (dataset_B_FacialImages) com as coordenadas de EyeCoordinatesInfo_*.txt:
    caixa quadrada centrada no olho com lado = CEW_CROP_FRAC * largura da face
    (0,35 * 100 = 35 px, arredondado para 36 px para ficar par), depois
    redimensionada para 48x48. Esse recorte inclui mais contexto (sobrancelha,
    pele ao redor) e se aproxima mais do enquadramento do MRL Eye do que o
    patch 24x24, que é muito justo ao olho.

O formato da linha em EyeCoordinatesInfo_*.txt é
  <arquivo> x_olho_esq y_olho_esq x_olho_dir y_olho_dir
(esquerdo/direito do ponto de vista da imagem: o primeiro par é o olho que
aparece à esquerda, com x menor).
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MRL_DIR = DATA / "mrl_eye" / "mrlEyes_2018_01"
MRL_INDEX = DATA / "mrl_eye" / "index.csv"
CEW_EYES = DATA / "cew" / "cew_eyes_24x24" / "dataset_B_Eye_Images"
CEW_FACES = DATA / "cew" / "cew_faces_100x100" / "dataset_B_FacialImages"
CACHE = DATA / "cache"

IMG_SIZE = 48
CEW_CROP_FRAC = 0.35  # lado da caixa quadrada em fração da largura da face
SENSOR_NAMES = {1: "RealSense", 2: "IDS", 3: "Aptina"}

_MRL_RE = re.compile(
    r"^s(?P<subject>\d{4})_(?P<image>\d{5})_(?P<gender>\d)_(?P<glasses>\d)_"
    r"(?P<state>\d)_(?P<reflections>\d)_(?P<lighting>\d)_(?P<sensor>\d{2})\.png$"
)


# --------------------------------------------------------------------------- #
# Índices (DataFrames)
# --------------------------------------------------------------------------- #
def build_mrl_index(force: bool = False) -> pd.DataFrame:
    """Percorre data/mrl_eye e devolve um DataFrame com os metadados de cada
    imagem, com cache em data/mrl_eye/index.csv."""
    if MRL_INDEX.exists() and not force:
        df = pd.read_csv(MRL_INDEX)
        return df
    rows = []
    for subj_dir in sorted(MRL_DIR.glob("s*")):
        if not subj_dir.is_dir():
            continue
        for name in sorted(os.listdir(subj_dir)):
            m = _MRL_RE.match(name)
            if not m:
                continue
            d = {k: int(v) for k, v in m.groupdict().items()}
            d["path"] = str((subj_dir / name).relative_to(ROOT))
            rows.append(d)
    df = pd.DataFrame(rows)
    df = df[["path", "subject", "image", "gender", "glasses", "state",
             "reflections", "lighting", "sensor"]]
    MRL_INDEX.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MRL_INDEX, index=False)
    return df


_CEW_PATCH_RE = re.compile(r"^(?P<face>.+)_(?P<side>[LR])\.jpg$")


def build_cew_patch_index() -> pd.DataFrame:
    """Índice dos recortes 24x24 oficiais do CEW.

    Colunas: path, face (id da face de origem), side (L/R), state (0/1)."""
    rows = []
    for folder, state in [("closedLeftEyes", 0), ("closedRightEyes", 0),
                          ("openLeftEyes", 1), ("openRightEyes", 1)]:
        for name in sorted(os.listdir(CEW_EYES / folder)):
            m = _CEW_PATCH_RE.match(name)
            if not m:
                continue  # ignora Thumbs.db etc.
            rows.append({
                "path": str((CEW_EYES / folder / name).relative_to(ROOT)),
                "face": m.group("face"), "side": m.group("side"), "state": state,
            })
    return pd.DataFrame(rows)


def build_cew_crop_index() -> pd.DataFrame:
    """Índice dos recortes extraídos das faces 100x100 do CEW (um por olho).

    Colunas: path (da face), face (id = nome do arquivo sem extensão), side,
    cx, cy (centro do olho na face), state."""
    rows = []
    for txt, folder, state in [("EyeCoordinatesInfo_ClosedFace.txt", "ClosedFace", 0),
                               ("EyeCoordinatesInfo_OpenFace.txt", "OpenFace", 1)]:
        with open(CEW_FACES / txt, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 5:
                    continue
                fname, x1, y1, x2, y2 = parts
                p = CEW_FACES / folder / fname
                if not p.exists():
                    continue
                face_id = fname[:-4] if fname.lower().endswith(".jpg") else fname
                for side, (cx, cy) in [("L", (int(x1), int(y1))), ("R", (int(x2), int(y2)))]:
                    rows.append({"path": str(p.relative_to(ROOT)), "face": face_id,
                                 "side": side, "cx": cx, "cy": cy, "state": state})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Leitura e pré-processamento de imagens
# --------------------------------------------------------------------------- #
def preprocess_gray(img: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Escala de cinza + redimensionamento para size x size (uint8)."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def _read_gray(path: str) -> np.ndarray:
    img = cv2.imread(str(ROOT / path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return preprocess_gray(img)


def _read_cew_crop(args) -> np.ndarray:
    path, cx, cy = args
    img = cv2.imread(str(ROOT / path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    side = int(round(CEW_CROP_FRAC * w))
    side += side % 2  # lado par (36 px para faces de 100 px)
    half = side // 2
    # preenchimento por replicação para olhos próximos da borda
    pad = cv2.copyMakeBorder(img, half, half, half, half, cv2.BORDER_REPLICATE)
    x0, y0 = cx, cy  # deslocados por 'half' após o padding
    crop = pad[y0:y0 + side, x0:x0 + side]
    return preprocess_gray(crop)


def _load_many(fn, items, workers: int | None = None) -> np.ndarray:
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers) as ex:
        arrs = list(ex.map(fn, items, chunksize=256))
    return np.stack(arrs).astype(np.uint8)


def to_float(x_uint8: np.ndarray) -> np.ndarray:
    """uint8 (N,48,48) -> float32 (N,48,48,1) em [0,1]."""
    return (x_uint8.astype(np.float32) / 255.0)[..., None]


# --------------------------------------------------------------------------- #
# Carregadores com cache .npz
# --------------------------------------------------------------------------- #
def load_mrl(force: bool = False):
    """Retorna (X uint8 (N,48,48), df de metadados) do MRL Eye completo."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / "mrl_48.npz"
    df = build_mrl_index(force)
    if cache.exists() and not force:
        z = np.load(cache)
        if len(z["x"]) == len(df):
            return z["x"], df
    x = _load_many(_read_gray, df["path"].tolist())
    np.savez_compressed(cache, x=x)
    return x, df


def load_cew(kind: str = "patch", force: bool = False):
    """Retorna (X uint8 (N,48,48), df) do CEW.

    kind = "patch": recortes 24x24 oficiais ampliados para 48x48;
    kind = "crop": recortes 36x36 extraídos das faces 100x100 -> 48x48."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"cew_{kind}_48.npz"
    if kind == "patch":
        df = build_cew_patch_index()
    elif kind == "crop":
        df = build_cew_crop_index()
    else:
        raise ValueError(kind)
    if cache.exists() and not force:
        z = np.load(cache)
        if len(z["x"]) == len(df):
            return z["x"], df
    if kind == "patch":
        x = _load_many(_read_gray, df["path"].tolist())
    else:
        x = _load_many(_read_cew_crop, list(zip(df["path"], df["cx"], df["cy"])))
    np.savez_compressed(cache, x=x)
    return x, df


# --------------------------------------------------------------------------- #
# Estatísticas
# --------------------------------------------------------------------------- #
def _count_table(df: pd.DataFrame, col: str, names: dict | None = None) -> str:
    t = df.groupby([col, "state"]).size().unstack(fill_value=0)
    t.columns = ["fechado (0)", "aberto (1)"]
    t["total"] = t.sum(axis=1)
    t["% aberto"] = (100 * t["aberto (1)"] / t["total"]).round(1)
    if names:
        t.index = [f"{i} ({names.get(i, i)})" for i in t.index]
    return t.to_markdown()


def write_dataset_stats(out_path: Path):
    mrl = build_mrl_index()
    patch = build_cew_patch_index()
    crop = build_cew_crop_index()
    L = []
    L.append("# Exp. 3 — Estatísticas dos datasets de estado ocular\n")
    L.append("Rótulo 0 = fechado, 1 = aberto. Gerado por `src/pipeline_b/datasets.py`.\n")
    L.append("## MRL Eye Dataset (Fusek, ISVC 2018)\n")
    L.append(f"- Imagens: **{len(mrl)}** | sujeitos: **{mrl.subject.nunique()}** "
             f"({mrl.groupby('subject').gender.first().eq(0).sum()} homens, "
             f"{mrl.groupby('subject').gender.first().eq(1).sum()} mulheres)")
    L.append(f"- Fechado: {int((mrl.state == 0).sum())} ({100 * (mrl.state == 0).mean():.1f}%) | "
             f"aberto: {int((mrl.state == 1).sum())} ({100 * (mrl.state == 1).mean():.1f}%)")
    L.append(f"- Nível de chance (classe majoritária): {100 * max(mrl.state.mean(), 1 - mrl.state.mean()):.1f}%\n")
    for col, title, names in [("glasses", "Óculos", {0: "não", 1: "sim"}),
                              ("lighting", "Iluminação", {0: "ruim", 1: "boa"}),
                              ("reflections", "Reflexos", {0: "nenhum", 1: "pequeno", 2: "grande"}),
                              ("sensor", "Sensor", SENSOR_NAMES),
                              ("gender", "Gênero", {0: "homem", 1: "mulher"})]:
        L.append(f"### Por {title.lower()}\n")
        L.append(_count_table(mrl, col, names) + "\n")
    L.append("### Por sujeito\n")
    s = mrl.groupby("subject").agg(imagens=("path", "size"), fechado=("state", lambda v: int((v == 0).sum())),
                                    aberto=("state", "sum"), oculos=("glasses", "mean"),
                                    ilum_boa=("lighting", "mean"), reflexo_medio=("reflections", "mean"),
                                    sensores=("sensor", lambda v: ",".join(str(u) for u in sorted(v.unique()))))
    s["oculos"] = (100 * s["oculos"]).round(0).astype(int)
    s["ilum_boa"] = (100 * s["ilum_boa"]).round(0).astype(int)
    s["reflexo_medio"] = s["reflexo_medio"].round(2)
    s = s.rename(columns={"oculos": "% óculos", "ilum_boa": "% ilum. boa", "reflexo_medio": "reflexo médio"})
    L.append(s.to_markdown() + "\n")
    L.append("## CEW — Closed Eyes in the Wild (Song et al., 2014)\n")
    L.append(f"- Recortes 24x24 oficiais (\"patch\"): **{len(patch)}** imagens de "
             f"**{patch.face.nunique()}** faces; fechado {int((patch.state == 0).sum())}, "
             f"aberto {int((patch.state == 1).sum())}; chance "
             f"{100 * max(patch.state.mean(), 1 - patch.state.mean()):.1f}%")
    L.append(f"- Recortes extraídos das faces 100x100 (\"crop\", caixa de "
             f"{int(round(CEW_CROP_FRAC * 100)) + int(round(CEW_CROP_FRAC * 100)) % 2} px): **{len(crop)}** "
             f"olhos de **{crop.face.nunique()}** faces; fechado {int((crop.state == 0).sum())}, "
             f"aberto {int((crop.state == 1).sum())}")
    L.append(f"- Faces com coordenadas: fechadas {crop[crop.state == 0].face.nunique()}, "
             f"abertas {crop[crop.state == 1].face.nunique()}\n")
    L.append("| Dataset | Domínio | Sujeitos/faces | Imagens | Fechado | Aberto |\n|---|---|---|---|---|---|")
    L.append(f"| MRL Eye | infravermelho, laboratório | {mrl.subject.nunique()} sujeitos | {len(mrl)} | "
             f"{int((mrl.state == 0).sum())} | {int((mrl.state == 1).sum())} |")
    L.append(f"| CEW (patch 24x24) | RGB, in-the-wild | {patch.face.nunique()} faces | {len(patch)} | "
             f"{int((patch.state == 0).sum())} | {int((patch.state == 1).sum())} |")
    L.append(f"| CEW (crop 36x36 da face) | RGB, in-the-wild | {crop.face.nunique()} faces | {len(crop)} | "
             f"{int((crop.state == 0).sum())} | {int((crop.state == 1).sum())} |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out_path


if __name__ == "__main__":
    import time
    t0 = time.time()
    x, df = load_mrl()
    print(f"MRL: {x.shape} em {time.time() - t0:.0f}s")
    for k in ("patch", "crop"):
        t0 = time.time()
        x, df = load_cew(k)
        print(f"CEW {k}: {x.shape} em {time.time() - t0:.0f}s")
    p = write_dataset_stats(ROOT / "results" / "exp3_dataset_stats.md")
    print("Estatísticas em", p)
