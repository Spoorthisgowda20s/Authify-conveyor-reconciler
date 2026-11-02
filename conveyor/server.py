"""One process that serves the review page and runs the pipeline stages as background jobs.

    python -m conveyor.server --port 8000
"""
import argparse
import gc
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
WEB = os.path.join(ROOT, "web")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")

JOB = dict(state="idle", stage="", progress=0.0, run="", log=[], started=0.0, error="")
JOB_LOCK = threading.Lock()
_stores, _lines = {}, {}


def run_dir(name):
    if not NAME_RE.match(name or ""):
        raise ValueError("bad run name")
    return os.path.join(DATA, name)


def _exec(stage, args, weight_lo, weight_hi):
    """Run `python -m conveyor.<stage> ...`, translating its PROGRESS lines into JOB progress."""
    JOB["stage"] = stage
    p = subprocess.Popen([sys.executable, "-m", "conveyor." + stage] + args, cwd=ROOT,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def pump(stream):
        for line in stream:
            line = line.rstrip()
            if line.startswith("PROGRESS"):
                JOB["progress"] = weight_lo + (weight_hi - weight_lo) * float(line.split()[1])
            elif line and line != "DONE":
                JOB["log"].append(line); JOB["log"] = JOB["log"][-40:]
    th = [threading.Thread(target=pump, args=(p.stderr,), daemon=True), threading.Thread(target=pump, args=(p.stdout,), daemon=True)]
    for t in th:
        t.start()
    rc = p.wait()
    for t in th:
        t.join()
    if rc != 0:
        raise RuntimeError(f"{stage} failed (exit {rc}); see log")


def pipeline(run, seed, duration, stride, do_generate):
    d = run_dir(run)
    try:
        if do_generate:
            _exec("generate", ["--out", d, "--seed", str(seed), "--duration", str(duration)], 0.0, 0.45)
        _exec("track", ["--data", d, "--stride", str(stride)], 0.45 if do_generate else 0.0, 0.9)
        _exec("evaluate", ["--data", d], 0.9, 1.0)
        JOB.update(state="done", stage="", progress=1.0)
    except Exception as e:  # noqa
        JOB.update(state="error", error=str(e))
    finally:
        _stores.pop(run, None); _lines.pop(run, None)


def start_job(run, seed, duration, stride, do_generate):
    with JOB_LOCK:
        if JOB["state"] == "running":
            return False
        JOB.update(state="running", stage="", progress=0.0, run=run, log=[], started=time.time(), error="")
        _stores.pop(run, None); _lines.pop((run, "tracks.jsonl"), None); gc.collect()   # Windows: release open files
    threading.Thread(target=pipeline, args=(run, seed, duration, stride, do_generate), daemon=True).start()
    return True


def have(run, fn):
    return os.path.exists(os.path.join(run_dir(run), fn))


def jl(run, fn):
    """Lines of a .jsonl file, cached (reloaded if the file changed)."""
    path = os.path.join(run_dir(run), fn)
    m = os.path.getmtime(path)
    key = (run, fn)
    if key not in _lines or _lines[key][0] != m:
        with open(path, "rb") as f:
            _lines[key] = (m, f.read().split(b"\n"))
    return _lines[key][1]


def frame_bytes(run, i):
    from .common import FrameStore
    path = os.path.join(run_dir(run), "frames_idx.npy")
    m = os.path.getmtime(path)
    if run not in _stores or _stores[run][0] != m:
        _stores[run] = (m, FrameStore(run_dir(run)))
    st = _stores[run][1]
    if not 0 <= i < len(st):
        raise IndexError
    return st.raw(i)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", cache=False):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, open(os.path.join(WEB, "index.html"), "rb").read(), "text/html; charset=utf-8")
            if u.path == "/api/state":
                runs = sorted(d for d in os.listdir(DATA) if os.path.exists(os.path.join(DATA, d, "frames_idx.npy"))) if os.path.isdir(DATA) else []
                run = q.get("run") or ("default" if "default" in runs else (runs[0] if runs else "default"))
                st = dict(runs=runs, run=run, job=JOB)
                if run in runs:
                    st.update(has_tracks=have(run, "tracks.jsonl") and have(run, "track_meta.json"), has_metrics=have(run, "metrics.json"))
                return self._send(200, st)
            run = q.get("run", "default")
            if u.path == "/api/meta":
                gt = json.load(open(os.path.join(run_dir(run), "ground_truth.json")))["summary"]
                cal = json.load(open(os.path.join(run_dir(run), "calibration.json")))
                return self._send(200, dict(calibration=cal, gt=gt))
            if u.path == "/api/frame":
                return self._send(200, frame_bytes(run, int(q["i"])), "image/jpeg", cache=True)
            if u.path in ("/api/tracks", "/api/gt"):
                fn = "tracks.jsonl" if u.path == "/api/tracks" else "gt_frames.jsonl"
                lines = jl(run, fn)
                a, b = int(q.get("start", 0)), int(q.get("end", 0))
                if fn == "tracks.jsonl":   # records carry their own frame number (stride may skip frames)
                    out = b"[" + b",".join(l for l in lines if l and a <= json.loads(l)["f"] < b) + b"]"
                else:
                    out = b"[" + b",".join(lines[a:b]) + b"]"
                return self._send(200, out)
            if u.path == "/api/metrics":
                return self._send(200, open(os.path.join(run_dir(run), "metrics.json"), "rb").read())
            return self._send(404, dict(error="not found"))
        except (FileNotFoundError, IndexError):
            return self._send(404, dict(error="not available yet"))
        except Exception as e:  # noqa
            return self._send(400, dict(error=str(e)))

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/api/run":
                run = body.get("run", "default"); run_dir(run)
                ok = start_job(run, int(body.get("seed", 1)), float(body.get("duration", 300)),
                               int(body.get("stride", 1)), bool(body.get("generate", True)))
                return self._send(200 if ok else 409, dict(started=ok, job=JOB))
            return self._send(404, dict(error="not found"))
        except Exception as e:  # noqa
            return self._send(400, dict(error=str(e)))


def serve(port, auto_seed=1, auto_duration=300, auto=True, host="127.0.0.1"):
    os.makedirs(DATA, exist_ok=True)
    if auto and not have("default", "metrics.json"):
        have_video = have("default", "frames_idx.npy") and have("default", "ground_truth.json")
        start_job("default", auto_seed, auto_duration, 1, not have_video)
    srv = ThreadingHTTPServer((host, port), H)
    srv.daemon_threads = True
    print(f"Conveyor reconciler running at http://{host}:{port}/", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--duration", type=float, default=300)
    ap.add_argument("--no-auto", action="store_true", help="do not generate/track the default dataset on start")
    a = ap.parse_args()
    serve(a.port, a.seed, a.duration, not a.no_auto, a.host)


if __name__ == "__main__":
    main()
