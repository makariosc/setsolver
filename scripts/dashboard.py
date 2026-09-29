"""Training dashboard: a tiny local web server (stdlib only) that charts every
Ultralytics run under runs/ and shows live progress of the one in flight.

    .venv-train/bin/python scripts/dashboard.py                      # http://127.0.0.1:8765
    .venv-train/bin/python scripts/dashboard.py --host <gpu-box-ip>   # reachable from your LAN/tailnet

Reads, on every refresh:
- runs/**/results.csv   per-epoch losses/metrics (written by Ultralytics after each epoch)
- logs/*.log            the newest training log, for progress within the current epoch
- nvidia-smi            GPU utilisation / memory / temperature
- preds/*/summary.json   detector outputs on real photos (scripts/predict_detector.py),
                        shown on the /predictions page
Images from run and prediction directories are served under /files/.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).with_name("dashboard.html")
PRED_PAGE = Path(__file__).with_name("predictions.html")

# "  55/60      10.1G   0.0847 ...  640: 69% ━━━ 217/313 5.8it/s 37.5s<16.5s"
PROGRESS = re.compile(
    r"^\s*(?P<epoch>\d+)/(?P<epochs>\d+)\s+(?P<mem>[\d.]+G)\s.*?(?P<pct>\d+)%\S*\s+\S*\s*"
    r"(?P<batch>\d+)/(?P<batches>\d+)\s+(?P<rate>[\d.]+(?:it/s|s/it))\s+(?P<elapsed>[\d.hms:]+)(?:<(?P<eta>[\d.hms:]+))?"
)


def read_runs(runs_dir: Path) -> list[dict]:
    runs = []
    for csv_path in sorted(runs_dir.rglob("results.csv")):
        with csv_path.open() as f:
            rows = list(csv.reader(f))
        run_dir = csv_path.parent
        pf = run_dir / "progress.json"
        live = pf.exists() and time.time() - pf.stat().st_mtime < 90
        # list a run once it has a finished epoch, or while it's live in its first epoch
        if not rows or (len(rows) < 2 and not live):
            continue
        header = [h.strip() for h in rows[0]]
        data = {h: [] for h in header}
        for r in rows[1:]:
            for h, v in zip(header, r):
                try:
                    data[h].append(float(v))
                except ValueError:
                    data[h].append(None)
        args = {}
        args_file = run_dir / "args.yaml"
        if args_file.exists():  # flat "key: value" lines are all we need
            for line in args_file.read_text().splitlines():
                if ": " in line and not line.startswith(" "):
                    k, v = line.split(": ", 1)
                    args[k] = v
        images = sorted(p.name for p in run_dir.glob("*.jpg")) + sorted(p.name for p in run_dir.glob("*.png"))
        progress = None
        if live:
            try:
                progress = json.loads(pf.read_text())
            except json.JSONDecodeError:
                pass  # being rewritten
        runs.append({
            "progress": progress,
            "name": run_dir.name,
            "path": str(run_dir.relative_to(ROOT)),
            "mtime": max(csv_path.stat().st_mtime, pf.stat().st_mtime if live else 0),
            "columns": data,
            "epochs_planned": int(args["epochs"]) if args.get("epochs", "").isdigit() else None,
            "model": args.get("model"),
            "close_mosaic": int(args["close_mosaic"]) if args.get("close_mosaic", "").isdigit() else None,
            "images": images,
        })
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def read_live(logs_dir: Path) -> dict | None:
    logs = sorted(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not logs:
        return None
    log = logs[0]
    age = time.time() - log.stat().st_mtime
    with log.open("rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 20000))
        tail = f.read().decode("utf-8", "replace")
    # Progress bars redraw with \r; take the last complete-looking update.
    for chunk in reversed(re.split(r"[\r\n]", tail)):
        m = PROGRESS.search(re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", chunk))
        if m:
            d = m.groupdict()
            return {
                "log": log.name,
                "age_s": round(age, 1),
                "active": age < 90,
                "epoch": int(d["epoch"]),
                "epochs": int(d["epochs"]),
                "batch": int(d["batch"]),
                "batches": int(d["batches"]),
                "rate": d["rate"],
                "eta_epoch": d["eta"],
                "gpu_mem": d["mem"],
            }
    return {"log": log.name, "age_s": round(age, 1), "active": False}


_gpu_cache: dict = {"t": 0.0, "v": None}


def read_gpu() -> dict | None:
    if time.time() - _gpu_cache["t"] < 4:
        return _gpu_cache["v"]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        name, util, used, total, temp = [s.strip() for s in out.split(",")]
        v = {"name": name, "util": int(util), "mem_used": int(used), "mem_total": int(total), "temp": int(temp)}
    except Exception:
        v = None
    _gpu_cache.update(t=time.time(), v=v)
    return v


def read_predictions(preds_dir: Path) -> list[dict]:
    out = []
    for f in sorted(preds_dir.glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue  # being written right now
        d["name"] = f.parent.name
        d["path"] = str(f.parent.relative_to(ROOT))
        out.append(d)
    return out


class Handler(BaseHTTPRequestHandler):
    runs_dir: Path
    logs_dir: Path
    preds_dir: Path

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, body: bytes, ctype: str, status: HTTPStatus = HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/data":
            body = json.dumps({
                "runs": read_runs(self.runs_dir),
                "live": read_live(self.logs_dir),
                "gpu": read_gpu(),
                "now": time.time(),
            }).encode()
            return self._send(body, "application/json")
        if path == "/predictions":
            return self._send(PRED_PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/api/predictions":
            return self._send(json.dumps(read_predictions(self.preds_dir)).encode(), "application/json")
        if path.startswith("/files/"):
            target = (ROOT / path[len("/files/"):]).resolve()
            # only images inside the runs/ or preds/ directories, never elsewhere
            allowed = any(d in target.parents for d in (self.runs_dir, self.preds_dir))
            if allowed and target.is_file() and target.suffix in (".jpg", ".png"):
                ctype = "image/jpeg" if target.suffix == ".jpg" else "image/png"
                return self._send(target.read_bytes(), ctype)
        self._send(b"not found", "text/plain", HTTPStatus.NOT_FOUND)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--logs", type=Path, default=ROOT / "logs")
    ap.add_argument("--preds", type=Path, default=ROOT / "preds")
    a = ap.parse_args()
    Handler.runs_dir = a.runs.resolve()
    Handler.logs_dir = a.logs.resolve()
    Handler.preds_dir = a.preds.resolve()
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"dashboard on http://{a.host}:{a.port}  (runs: {Handler.runs_dir}, logs: {Handler.logs_dir})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
