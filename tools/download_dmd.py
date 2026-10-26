"""Baixa os pacotes do DMD a partir do e-mail da Vicomtech salvo como .eml.

O portal da Vicomtech (https://dmd.vicomtech.org/) envia, após o formulário, um e-mail com links
assinados (AWS S3, válidos por 48 h), um por arquivo. O Outlook embrulha cada link num
"safelinks"; este script abre o .eml, desembrulha os links e baixa os .tar.gz em paralelo,
com retomada, confere a integridade e extrai só o vídeo do rosto e a anotação de cada pacote
(usa extract_dmd_package de tools/prepare_data.py), apagando o .tar.gz em seguida.

Como obter o .eml: no Outlook, abra o e-mail "Vicomtech Dataset Request | ...", clique em
"..." (mais ações) > Baixar. O arquivo vai para ~/Downloads.

Uso:
    python tools/download_dmd.py                       # usa o .eml mais recente da Vicomtech em ~/Downloads
    python tools/download_dmd.py caminho/do/email.eml  # ou um .eml específico
"""
import email
import email.policy
import glob
import html
import os
import re
import subprocess
import sys
import time
import urllib.parse as up
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "dmd")


def links_from_eml(path):
    msg = email.message_from_bytes(open(path, "rb").read(), policy=email.policy.default)
    body = ""
    for part in msg.walk():
        if part.get_content_type() in ("text/html", "text/plain"):
            try:
                body += part.get_content()
            except Exception:
                pass
    body = html.unescape(body).replace("=\n", "").replace("\r", "")
    files = {}
    for u in re.findall(r"https?://[^\s\"'<>)]+", body):
        if "safelinks.protection.outlook.com" in u:
            q = up.parse_qs(up.urlparse(u).query)
            u = q["url"][0] if "url" in q else u
        if "datasets.vicomtech.org" not in u:
            continue
        name = up.unquote(up.urlparse(u).path.split("/")[-1])
        files.setdefault(name, u)
    return files


def _already_extracted(name):
    """True se o pacote já foi extraído antes (o .tar.gz some depois da extração)."""
    m = re.match(r"dmd-dataset-(\w+)-(g[A-Z])-(\d+)\.tar\.gz$", name)
    if not m:
        return False
    kind, group, subj = m.groups()
    sessions = ["s5"] if kind == "drowsiness" else ["s1", "s2", "s3", "s4"]
    return any(glob.glob(os.path.join(DEST, group, subj, s, "*_rgb_face.mp4")) for s in sessions)


def _gzip_ok(path):
    """True se o .tar.gz está completo e íntegro (gzip -t lê o arquivo inteiro)."""
    return os.path.exists(path) and subprocess.run(["gzip", "-t", path], capture_output=True).returncode == 0


def fetch(item, tries=6):
    """Baixa com retomada, confere a integridade do gzip e extrai rosto + anotação.

    O servidor da Vicomtech derruba conexões HTTP/2 longas ("stream was not closed cleanly");
    por isso força HTTP/1.1 e repete retomando de onde parou. A integridade é conferida ANTES de
    cada tentativa: um arquivo que já chegou completo é extraído sem novo download.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from prepare_data import extract_dmd_package
    name, url = item
    out = os.path.join(DEST, name)
    if not os.path.exists(out) and _already_extracted(name):
        print(f"[ok] {name}: já extraído", flush=True)
        return True
    t0 = time.time()
    for t in range(tries + 1):
        if _gzip_ok(out):
            size = os.path.getsize(out) / 1e9
            ok = extract_dmd_package(out)
            print(f"[{'ok' if ok else 'ERRO'}] {name}: {size:.2f} GB em {(time.time() - t0) / 60:.1f} min", flush=True)
            return ok
        if t == tries:
            break
        r = subprocess.run(["curl", "-sS", "-L", "--http1.1", "-C", "-", "--retry", "5", "--retry-delay", "10",
                            "--retry-all-errors", "--max-time", "7200", "-o", out, url], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"   [tentativa {t + 1}] {name}: {r.stderr.strip()[:120]}; retomando", flush=True)
            time.sleep(15)
    print(f"[ERRO] {name}: falhou após {tries} tentativas", flush=True)
    return False


def main():
    if len(sys.argv) > 1:
        eml = sys.argv[1]
    else:
        cands = sorted(glob.glob(os.path.expanduser("~/Downloads/*Vicomtech*.eml")), key=os.path.getmtime)
        if not cands:
            sys.exit("Nenhum .eml da Vicomtech em ~/Downloads. Salve o e-mail (… > Baixar) e rode de novo.")
        eml = cands[-1]
    files = links_from_eml(eml)
    os.makedirs(DEST, exist_ok=True)
    readme = {n: u for n, u in files.items() if n.lower().startswith("readme")}
    tars = {n: u for n, u in files.items() if n.endswith(".tar.gz")}
    print(f"E-mail: {os.path.basename(eml)} — {len(tars)} pacotes, {len(readme)} README", flush=True)
    for n, u in readme.items():
        subprocess.run(["curl", "-sS", "-L", "-o", os.path.join(DEST, n), u])
    with ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(fetch, tars.items()))
    print(f"\nConcluído: {sum(res)}/{len(res)} pacotes baixados e extraídos em data/dmd/", flush=True)
    print("FIM DO DOWNLOAD DMD", flush=True)


if __name__ == "__main__":
    main()
