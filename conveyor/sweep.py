"""Held-out evaluation + the sampling-rate (stride) sweep, written to RESULTS.md.

    python -m conveyor.sweep --seeds 2 3 4 --duration 300

Seed 1 is the development seed (thresholds were tuned while watching it); the numbers
reported come from seeds the tracker was never tuned on.
"""
import argparse
import json
import os
import subprocess
import sys

from . import evaluate, generate, track


def ensure(seed, duration, root):
    d = os.path.join(root, f"seed{seed}")
    if not os.path.exists(os.path.join(d, "ground_truth.json")):
        generate.main(["--out", d, "--seed", str(seed), "--duration", str(duration)])
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--duration", type=float, default=300)
    ap.add_argument("--root", default="data/sweep")
    ap.add_argument("--strides", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8])
    ap.add_argument("--out", default="RESULTS.md")
    a = ap.parse_args(argv)
    rows = []
    for seed in a.seeds:
        d = ensure(seed, a.duration, a.root)
        for stride in a.strides:
            track.run(d, stride=stride)
            r = evaluate.evaluate(d)
            rows.append(dict(seed=seed, stride=stride, total=r["total"], steady=r["steady"], surge=r["surge"], stall=r["stall"],
                             occ=r["occluded_3plus_frames"], heavy=r["heavily_occluded_vis_le_30pct"], buckets=r["buckets"]["accuracy_pct"],
                             ident=r["identity"], timing=r["timing"]))
    json.dump(rows, open(os.path.join(a.root, "sweep.json"), "w"))
    print("wrote", os.path.join(a.root, "sweep.json"), file=sys.stderr)


if __name__ == "__main__":
    main()
