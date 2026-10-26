"""Prepara os dados de vídeo do projeto: baixa, extrai e limpa.

UTA-RLDD (Ghoddoosian et al., 2019)
    10 zips no Google Drive oficial (Fold{1..5}_part{1,2}.zip, ~111 GB no total).
    Para cada fold: obtém as duas partes, extrai em data/uta_rldd_videos/FoldN/<sujeito>/{0,5,10}.<ext>,
    confere a contagem de vídeos, grava FoldN/.ok e apaga os zips para liberar disco.

    O Drive aplica uma cota a arquivos muito acessados ("Quota exceeded ... too many users").
    Por isso há duas fontes, nesta ordem:
      1. zips baixados pelo navegador em ~/Downloads (conta Google logada tem cota própria);
      2. download anônimo numa conexão única com retomada (tools/prepare_data.py --uta --baixar).

DMD (Vicomtech)
    .tar.gz dos pacotes de sonolência (sessão s5) e de distração (sessões s1-s4, que trazem os
    trechos anotados "Safe Driving" usados como classe alerta), baixados por links assinados que
    expiram em 48 h. Extrai SÓ o vídeo da câmera do rosto e a anotação (..._rgb_face.mp4 e
    ..._rgb_ann_<pacote>.json) em data/dmd/<grupo>/<sujeito>/<sessão>/ e apaga o .tar.gz.
    Sujeitos com participação dupla (manhã e tarde) têm sessões repetidas.

Uso:
    python tools/prepare_data.py --dmd                 # extrai o DMD já baixado em data/dmd/
    python tools/prepare_data.py --uta                 # extrai folds cujos zips já estão disponíveis
    python tools/prepare_data.py --uta --baixar        # idem, tentando baixar do Drive o que faltar
Reinicializável: pula o que já foi feito.
"""
import argparse
import glob
import os
import shutil
import subprocess
import tarfile
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTA = os.path.join(ROOT, "data", "uta_rldd_videos")
DMD = os.path.join(ROOT, "data", "dmd")
DOWNLOADS = os.path.join(os.path.expanduser("~"), "Downloads")
VIDEO_EXT = (".mp4", ".mov", ".MOV", ".MP4", ".m4v")
ZIP_MAGIC = bytes([0x50, 0x4B, 0x03, 0x04])

UTA_IDS = {
    "Fold1_part1.zip": "16x8afSOwfbdGAjUbfRFvwGj1ie8Lf5Do", "Fold1_part2.zip": "1qPoJ06jeznytP_BGJ25hxdQy7eYTTjyk",
    "Fold2_part1.zip": "12uVMVLv_kRXzQb4QDdru9RK3G9hzfpPQ", "Fold2_part2.zip": "1nYGUpqE3x5ZeRLgDeMlUN4K_ccPjM2pe",
    "Fold3_part1.zip": "1-JhE5TfkNyr4UbQk6PSkkpI7z-SChkWS", "Fold3_part2.zip": "1LZU5KfJkFMj2pIkGwb3RjHDazoUfxSYQ",
    "Fold4_part1.zip": "1IYKdrjvawf90bG_cfeoWA4rZSymllFbD", "Fold4_part2.zip": "1zJlASBwABBgVSULQLiia0R_Eqar8_B3D",
    "Fold5_part1.zip": "16w30wEgcAbaUr4gx9o4sl_2TbIunD5yA", "Fold5_part2.zip": "1loOQE1-hoNM5dIOjXHI2Ss5NjjoPZcX6",
}
UTA_FOLDER = "https://drive.google.com/drive/folders/1d_QwgpMXnLY_FmLYXDn7TLcw0D-svXEl"


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def free_gb():
    return shutil.disk_usage(ROOT).free / 1e9


# --------------------------------------------------------------------------- download do Drive
def _drive_url(fid):
    return f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"


def drive_download(fid, out, tries=4, wait_s=300):
    """Download anônimo numa conexão única, com retomada ("-C -").

    A cota do Drive é checada no início de cada pedido; uma conexão aceita segue até o fim.
    Se o pedido receber a página de cota (HTML, poucos KB), apaga-a e espera antes de tentar de novo.
    Não insiste de forma agressiva: tentativas repetidas prolongam o bloqueio.
    """
    for t in range(tries):
        if os.path.exists(out) and zipfile.is_zipfile(out):
            return True
        if os.path.exists(out) and os.path.getsize(out) < 1_000_000:
            os.remove(out)  # página HTML de cota de uma tentativa anterior
        before = os.path.getsize(out) if os.path.exists(out) else 0
        subprocess.run(["curl", "-sS", "-L", "-C", "-", "-m", "7200", "-o", out, _drive_url(fid)],
                       capture_output=True)
        if os.path.exists(out) and zipfile.is_zipfile(out):
            return True
        after = os.path.getsize(out) if os.path.exists(out) else 0
        if after - before < 50_000_000:
            # Nada (ou quase nada) veio: é a cota. Espera SEMPRE antes de tentar de novo;
            # pedidos em sequência rápida só prolongam o bloqueio.
            if os.path.exists(out) and os.path.getsize(out) < 1_000_000:
                os.remove(out)
            log(f"   [cota] {os.path.basename(out)} bloqueado pelo Drive; nova tentativa em {wait_s // 60} min")
            time.sleep(wait_s)
        # se avançou bastante e caiu no meio, o laço retoma com "-C -" sem esperar
    return False


def _obtain(name, allow_download):
    """Garante data/uta_rldd_videos/<name> válido. Retorna True se disponível."""
    z = os.path.join(UTA, name)
    if os.path.exists(z) and zipfile.is_zipfile(z):
        return True
    stem = name[:-4]  # "Fold2_part1"
    # Variações de nome que o navegador e o Drive geram: "Fold2_part1 (1).zip", "Cópia de Fold2_part1.zip".
    cands = sorted(glob.glob(os.path.join(DOWNLOADS, f"*{stem}*.zip")), key=os.path.getmtime, reverse=True)
    dl = next((c for c in cands if zipfile.is_zipfile(c)), None)
    if dl:
        if os.path.exists(z):
            os.remove(z)  # parcial antigo
        shutil.move(dl, z)
        log(f"[movido] {name} de ~/Downloads")
        return True
    if allow_download:
        log(f"[baixando] {name} (livre: {free_gb():.0f} GB)")
        return drive_download(UTA_IDS[name], z)
    return False


# --------------------------------------------------------------------------- UTA-RLDD
def uta_fold(n, allow_download):
    dest = os.path.join(UTA, f"Fold{n}")
    marker = os.path.join(dest, ".ok")
    if os.path.exists(marker):
        log(f"[ok] Fold{n} já extraído ({open(marker).read().strip()})")
        return True
    names = [f"Fold{n}_part1.zip", f"Fold{n}_part2.zip"]
    missing = [nm for nm in names if not _obtain(nm, allow_download)]
    if missing:
        log(f"[pendente] Fold{n}: faltam {', '.join(missing)}")
        return False
    # Espaço necessário = soma dos tamanhos descompactados; exige folga de 15 GB depois da extração.
    need = 0
    for nm in names:
        with zipfile.ZipFile(os.path.join(UTA, nm)) as zf:
            need += sum(i.file_size for i in zf.infolist())
    need /= 1e9
    if free_gb() - need < 15:
        log(f"[aguardando espaço] Fold{n}: extração precisa de {need:.0f} GB e há {free_gb():.0f} GB livres")
        return False
    os.makedirs(dest, exist_ok=True)
    for name in names:
        log(f"[extraindo] {name}")
        with zipfile.ZipFile(os.path.join(UTA, name)) as zf:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.endswith(VIDEO_EXT):
                    continue
                parts = info.filename.split("/")  # FoldN_partK/<sujeito>/<rotulo>.<ext>
                out = os.path.join(dest, parts[-2], parts[-1])
                os.makedirs(os.path.dirname(out), exist_ok=True)
                if os.path.exists(out) and os.path.getsize(out) == info.file_size:
                    continue
                with zf.open(info) as src, open(out, "wb") as dst:
                    shutil.copyfileobj(src, dst, 16 << 20)
    vids = [p for p in glob.glob(os.path.join(dest, "*", "*")) if p.endswith(VIDEO_EXT)]
    subs = sorted({os.path.basename(os.path.dirname(p)) for p in vids})
    log(f"[conferência] Fold{n}: {len(subs)} sujeitos, {len(vids)} vídeos")
    if len(vids) < 30:
        log(f"[ERRO] Fold{n} com poucos vídeos; zips mantidos para investigação")
        return False
    with open(marker, "w") as fh:
        fh.write(f"{len(subs)} sujeitos, {len(vids)} vídeos\n")
    for name in names:
        os.remove(os.path.join(UTA, name))
        log(f"[removido] {name}")
    return True


# --------------------------------------------------------------------------- DMD
def extract_dmd_package(tgz):
    """Extrai de um .tar.gz do DMD só o vídeo do rosto e a anotação; apaga o .tar.gz se deu certo.

    Retorna True se extraiu (ou se o arquivo já não existe). Um .tar.gz truncado (download
    interrompido) gera erro de leitura: o arquivo é mantido para o download retomar.
    """
    base = os.path.basename(tgz)
    if not os.path.exists(tgz):
        return True
    log(f"[extraindo] {base}")
    got = 0
    try:
        with tarfile.open(tgz) as tf:
            for m in tf.getmembers():
                if not (m.name.endswith("_rgb_face.mp4") or ("_rgb_ann_" in m.name and m.name.endswith(".json"))):
                    continue
                rel = m.name.split("dmd/", 1)[-1]  # <grupo>/<sujeito>/<sessão>/<arquivo>
                out = os.path.join(DMD, rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                if not (os.path.exists(out) and os.path.getsize(out) == m.size):
                    with tf.extractfile(m) as src, open(out, "wb") as dst:
                        shutil.copyfileobj(src, dst, 16 << 20)
                got += 1
    except (tarfile.TarError, EOFError, OSError) as e:
        log(f"[ERRO] {base}: arquivo incompleto ou corrompido ({type(e).__name__}); .tar.gz mantido")
        return False
    # Cada sessão gera 2 arquivos (rosto + anotação); várias sessões ou participação dupla geram mais pares.
    if got >= 2 and got % 2 == 0:
        os.remove(tgz)
        log(f"[ok] {base}: {got // 2} sessão(ões) extraída(s); .tar.gz removido")
        return True
    log(f"[ERRO] {base}: {got} arquivos (esperava pares rosto+anotação); .tar.gz mantido")
    return False


def dmd_all():
    for tgz in sorted(glob.glob(os.path.join(DMD, "dmd-dataset-*.tar.gz"))):
        extract_dmd_package(tgz)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uta", action="store_true", help="extrai os folds da UTA-RLDD disponíveis")
    ap.add_argument("--baixar", action="store_true", help="com --uta: tenta baixar do Drive o que faltar")
    ap.add_argument("--dmd", action="store_true", help="extrai o DMD já baixado em data/dmd/")
    a = ap.parse_args()
    os.makedirs(UTA, exist_ok=True)
    if a.dmd:
        dmd_all()
    if a.uta:
        done = [n for n in range(1, 6) if uta_fold(n, a.baixar)]
        log(f"Folds da UTA-RLDD prontos: {done}")
        if len(done) < 5:
            log(f"Para completar, baixe os zips que faltam em {UTA_FOLDER} para ~/Downloads e rode de novo.")
    log(f"FIM DA PREPARAÇÃO (livre: {free_gb():.0f} GB)")
