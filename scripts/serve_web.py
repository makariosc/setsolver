"""Serve the browser app (web/) for local development.

    uv run scripts/serve_web.py                 # http://127.0.0.1:8000
    uv run scripts/serve_web.py --host 0.0.0.0  # reachable from a phone on the same network

Sends cross-origin isolation headers so ONNX Runtime Web can use multi-threaded
WASM, and disables caching so model/code edits show up on reload.
"""

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "web"


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm",
                      ".onnx": "application/octet-stream", ".json": "application/json"}

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "credentialless")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    server = ThreadingHTTPServer((a.host, a.port), functools.partial(Handler, directory=str(WEB)))
    print(f"Set Solver on http://{a.host}:{a.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
