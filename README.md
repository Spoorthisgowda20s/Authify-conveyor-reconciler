# Conveyor-count reconciler

Synthesises a top-down conveyor video (with exact ground truth), tracks parcels, counts each one once,
buckets it small/medium/large, and shows everything in a browser page that compares the count with the truth.

## Run it (one command)

**Windows (cmd or PowerShell), in the project folder:**

```bat
start.bat
```

**Mac / Linux:**

```bash
./start.sh            # Linux / macOS: creates .venv, installs numpy + opencv-python-headless, starts the server
```
```bat
start.bat             :: Windows 11 (cmd or PowerShell): same thing
```
If you already have numpy and opencv: `python3 run.py` (`python run.py` on Windows).

Wait for the line `Conveyor reconciler running at http://127.0.0.1:8000/`, then open **http://127.0.0.1:8000/** in a browser.
Do not double-click `web/index.html`: the page needs the server behind it. On the first start the server itself generates the default 5-minute video
(~1,235 parcels, 4 surges, 2 stalls), runs the tracker and scores it (about 1.5 minutes; progress bar on the page).
Later starts reuse `data/default`.

Prerequisites: Python 3.9+ (tested on 3.13 on Linux; `start.bat` is untested on Windows), `venv`+`pip` for the start scripts. No GPU, no other services. Port: `--port N`.

## What to do on the page

* **Count vs ground truth** table: whole video, steady stretches, surges, stall, occluded parcels.
* **Review**: input frame left, tracker decision right. Play, ±1, ±10 frame steps, slider, arrow keys.
  Jump to next count event / ground-truth overlap / belt stall / wrong result. Toggle ground-truth boxes (magenta,
  with visible %), the predicted box (dashed) and raw detected blobs.
* **Contact sheet**: seven frames around the current one, all with overlays; click one to jump there.
* **Drive the system**: change seed / duration / stride and press *Generate video + track + score*
  (any run name; results appear in the *Dataset* box).

## Command line

```bash
python3 -m conveyor.generate --out data/s1 --seed 1 --duration 300     # video + ground_truth.json (+ per-frame truth)
python3 -m conveyor.track    --data data/s1 [--stride 1]                # tracker -> tracks.jsonl, count_events.json
python3 -m conveyor.evaluate --data data/s1                             # metrics.json + printed summary
python3 -m conveyor.sweep --seeds 2 3 4                                 # held-out seeds x strides -> data/sweep/sweep.json
python3 -m conveyor.generate --out data/hour --seed 5 --duration 3600   # a full simulated hour (~11k parcels, ~2 GB)
python3 -m unittest discover tests                                      # smoke test
```

## Layout

`conveyor/generate.py` video + truth · `track.py` detector/tracker/counter · `evaluate.py` scoring ·
`server.py` stdlib HTTP server and job runner · `web/index.html` the page · `DECISIONS.md` reasoning ·
`RESULTS.md` measured numbers · `VIDEO_SCRIPT.md` talking points.
