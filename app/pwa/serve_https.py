#!/usr/bin/env python3
"""Servidor HTTPS local para testar a PWA no celular (mesma rede Wi-Fi).

A câmera (getUserMedia) só funciona em contexto seguro (HTTPS ou localhost),
por isso este script gera um certificado autoassinado com o `openssl` do
macOS e serve a pasta app/pwa na porta 8443.

Uso:
    python3 app/pwa/serve_https.py [--port 8443] [--cert-dir DIR]

Depois abra no Chrome do celular:  https://<IP-do-Mac>:8443/
e aceite o aviso do certificado (Avançado -> Ir para ... (não seguro)).

Sem internet no celular, o app precisa das dependências em app/pwa/vendor/ (MediaPipe, WASM,
face_landmarker.task, TensorFlow.js), baixadas por tools/baixar_dependencias_app.sh; o servidor
confere a pasta ao subir e, se faltar algo, avisa e oferece baixar.
"""

import argparse
import http.server
import json
import re
import os
import socket
import ssl
import subprocess
import sys
from pathlib import Path

PWA_DIR = Path(__file__).resolve().parent
# Certificado autoassinado fica em app/.pwa-cert (FORA da pasta servida, para a chave privada
# não ficar acessível por HTTP); gerado na primeira execução; não versionar.
DEFAULT_CERT_DIR = str(PWA_DIR.parent / ".pwa-cert")
# Resultados do benchmark enviados pelo celular (POST /api/benchmark) vão para results/phone/ do repositório.
PHONE_DIR = PWA_DIR.parent.parent / "results" / "phone"
MAX_UPLOAD = 50 * 1024 * 1024  # 50 MB: um benchmark de 60 s tem poucas centenas de KB
# Dependências locais do app (./vendor/): baixa e confere pelo SHA-256 (sem a pasta, o app usa a CDN).
VENDOR_SCRIPT = PWA_DIR.parent.parent / "tools" / "baixar_dependencias_app.sh"


def local_ip():
    """IP local preferencial (en0 no macOS); cai para o truque do socket UDP."""
    try:
        out = subprocess.run(["ipconfig", "getifaddr", "en0"], capture_output=True, text=True, timeout=5)
        ip = out.stdout.strip()
        if ip:
            return ip
    except Exception:
        pass
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def check_vendor(vendor: Path) -> bool:
    """Confere a pasta vendor/ com tools/baixar_dependencias_app.sh --conferir (sem rede). Se faltar algo,
    avisa e, num terminal interativo, oferece baixar agora. Devolve True se a pasta ficou completa."""
    cmd = ["bash", str(VENDOR_SCRIPT), "--pasta", str(vendor)]
    try:
        if subprocess.run(cmd + ["--conferir"], capture_output=True, text=True, timeout=120).returncode == 0:
            return True
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[vendor] não foi possível conferir {vendor} ({exc})")
        return False
    print(f"AVISO: {vendor} vazia ou incompleta. O app vai buscar o MediaPipe e o TensorFlow.js na")
    print("       internet (CDN) e NÃO funciona num celular sem internet (ex.: na PUC, sem dados móveis).")
    print("       Para baixar (~36 MB, uma vez): bash tools/baixar_dependencias_app.sh")
    if sys.stdin.isatty():
        try:
            resp = input("Baixar agora? [s/N] ").strip().lower()
        except EOFError:
            resp = ""
        if resp in ("s", "sim", "y", "yes"):
            return subprocess.run(cmd).returncode == 0
    return False


def ensure_cert(cert_dir: Path, ip: str):
    """Gera cert.pem/key.pem autoassinados (RSA 2048, 825 dias) se não existirem."""
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert = cert_dir / "cert.pem"
    key = cert_dir / "key.pem"
    if cert.exists() and key.exists():
        return cert, key
    san = f"subjectAltName=IP:{ip},IP:127.0.0.1,DNS:localhost"
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256",
        "-keyout", str(key), "-out", str(cert), "-days", "825",
        "-subj", f"/CN={ip}/O=PUCPR Equipe 14/OU=Sonolencia PWA",
        "-addext", san,
    ]
    print("[cert] gerando certificado autoassinado:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return cert, key


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serve app/pwa com MIME corretos e sem cache (desenvolvimento)."""

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".mjs": "text/javascript",
        ".webmanifest": "application/manifest+json",
        ".json": "application/json",
        ".wasm": "application/wasm",
        ".tflite": "application/octet-stream",
        ".svg": "image/svg+xml",
        ".md": "text/markdown; charset=utf-8",
    }

    directory_override = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(self.directory_override or PWA_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        # Permite Wake Lock / câmera / tela cheia no próprio documento
        self.send_header("Permissions-Policy", "camera=(self), screen-wake-lock=(self), fullscreen=(self)")
        super().end_headers()

    def do_POST(self):
        """Recebe o resultado do benchmark do app (JSON com stamp, summary e csv) e grava em results/phone/."""
        if self.path.rstrip("/") != "/api/benchmark":
            self.send_error(404, "use POST /api/benchmark")
            return
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_UPLOAD:
            self.send_error(413, "corpo vazio ou grande demais")
            return
        try:
            data = json.loads(self.rfile.read(n))
            summary, csv = data["summary"], data["csv"]
            if not isinstance(summary, dict) or summary.get("tipo") != "benchmark_pwa" or not isinstance(csv, str):
                raise ValueError("formato inesperado")
            stamp = re.sub(r"[^0-9A-Za-zT-]", "", str(data.get("stamp", "")))[:40] or "sem-data"
        except Exception as exc:
            self.send_error(400, f"JSON inválido: {exc}")
            return
        PHONE_DIR.mkdir(parents=True, exist_ok=True)
        base = PHONE_DIR / f"benchmark_pwa_{stamp}"
        (base.with_suffix(".json")).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (PHONE_DIR / f"benchmark_pwa_{stamp}_frames.csv").write_text(csv, encoding="utf-8")
        saved = [f"results/phone/benchmark_pwa_{stamp}.json", f"results/phone/benchmark_pwa_{stamp}_frames.csv"]
        lat = summary.get("latencia_ms", {}).get("total_por_quadro", {})
        print(f"\n>>> BENCHMARK RECEBIDO do celular ({summary.get('aparelho', {}).get('userAgent', '?')[:60]}...)")
        print(f"    {summary.get('quadros')} quadros, FPS médio {summary.get('fps_medio', 0):.1f}, "
              f"latência total p50 {lat.get('p50', 0):.1f} ms / p95 {lat.get('p95', 0):.1f} ms")
        print(f"    salvo em: {', '.join(saved)}\n")
        body = json.dumps({"salvo_em": saved}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8443)
    ap.add_argument("--cert-dir", default=DEFAULT_CERT_DIR)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--dir", default=None, help="pasta a servir (padrão: app/pwa)")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    if args.dir:
        Handler.directory_override = Path(args.dir).resolve()

    ip = local_ip()
    cert, key = ensure_cert(Path(args.cert_dir), ip)

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))

    httpd = http.server.ThreadingHTTPServer((args.bind, args.port), Handler)
    httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)

    served = Handler.directory_override or PWA_DIR
    models = sorted(p.name for p in (served / "models").glob("*.tflite")) if (served / "models").exists() else []
    vendor_ok = check_vendor(served / "vendor")
    print("=" * 70)
    print("PWA de sonolência: servidor HTTPS")
    print(f"  Pasta servida : {served}")
    print(f"  Certificado   : {cert}")
    print(f"  Modelos .tflite em app/pwa/models: {models or 'NENHUM (modo degradado: só landmarks)'}")
    print(f"  Dependências  : {'vendor/ completa (funciona sem internet)' if vendor_ok else 'CDN (o celular precisa de internet)'}")
    print()
    print("  URLs:")
    print(f"    neste Mac  : https://localhost:{args.port}/")
    print(f"    no celular : https://{ip}:{args.port}/")
    print()
    print("  No celular (iPhone ou Galaxy S23 Ultra, mesma rede Wi-Fi), no Chrome:")
    print(f"   1. Abra https://{ip}:{args.port}/")
    print("   2. Na tela 'Sua conexão não é particular' toque em Avançado -> 'Ir para", ip, "(não seguro)'.")
    print("   3. Toque em 'Iniciar viagem' e permita a câmera quando o Chrome pedir; a calibração (10 s olhando")
    print("      para a frente) começa sozinha. 'Recalibrar' (barra de baixo da viagem) refaz a calibração.")
    print("   4. Na tela 'Em viagem', role até 'Detalhes técnicos' e toque em 'Benchmark 60 s'. O resultado vai sozinho para este Mac,")
    print(f"      em {PHONE_DIR} (aparece aqui como 'BENCHMARK RECEBIDO'). Repita 3 vezes.")
    print("   5. Se o Chrome bloquear a câmera por certificado inválido, use chrome://flags ->")
    print(f"      'Insecure origins treated as secure' e adicione https://{ip}:{args.port} (ou use o GitHub Pages).")
    print()
    print("  Alternativa sem certificado: publicar app/pwa no GitHub Pages (HTTPS válido):")
    print("    - Settings -> Pages -> Branch main, pasta /docs (copie app/pwa para docs/) ou use a branch gh-pages;")
    print("    - copie também os .tflite/.json para docs/models/;")
    print("    - abra https://<usuario>.github.io/<repo>/ no celular.")
    print("  Ctrl+C encerra.")
    print("=" * 70)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nencerrado")


if __name__ == "__main__":
    main()
