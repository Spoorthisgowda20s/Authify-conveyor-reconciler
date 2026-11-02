"""Synthesise a top-down conveyor video plus exact ground truth.

    python -m conveyor.generate --out data/default --seed 1 --duration 300

Everything the tracker must survive is planned here, up front, from the seed:

* steady belt  ~2 parcels/s at 8 px/frame, surges ~6 parcels/s at 12 px/frame
  (the belt runs faster in a surge, as the brief says), 20-40 s each;
* belt stalls (speed ramps to exactly 0 for 1-2 s while the camera keeps running),
  at least one in a steady stretch and one inside a surge;
* planned overlaps during surges that last 3-5 frames: a "lateral" nudge
  (a side-by-side twin slides sideways into its neighbour) and an "x-slip"
  (a follower catches up with the parcel ahead). The later parcel is on top.

Parcel positions are analytic (belt distance since entry + an optional offset bump),
so ground truth is exact: spawn frame, line-crossing frame, bucket, true box and
visible fraction per frame (visible fraction is measured from the rendered z-buffer).
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from .common import FrameWriter, save_json

FPS = 30
W, H = 640, 200
BELT_TOP, BELT_BOT = 16, 184
COUNT_X = 440  # virtual counting line (centre of a parcel crossing it = one count)

V_STEADY, V_SURGE = 8.0, 12.0  # px / frame
RATE_STEADY, RATE_SURGE = 2.0, 6.0  # parcels / s targets

# w,h ranges chosen so the *area* ranges overlap slightly at the class borders:
# buckets are distinguishable but not trivial.
SPECS = {
    "small": dict(w=(26, 44), h=(20, 32)),
    "medium": dict(w=(40, 66), h=(30, 48)),
    "large": dict(w=(60, 90), h=(46, 68)),
}
BUCKETS = ["small", "medium", "large"]
BUCKET_P = [0.4, 0.4, 0.2]
MEAN_W = sum(p * (SPECS[b]["w"][0] + SPECS[b]["w"][1]) / 2 for b, p in zip(BUCKETS, BUCKET_P))

PALETTE = np.array([
    (100, 150, 190), (120, 170, 210), (80, 115, 150),   # kraft (BGR)
    (225, 225, 220), (150, 150, 155), (190, 140, 90),   # white, grey, blue bag
    (110, 190, 120), (90, 90, 200),                      # green, red
], dtype=np.float32)
BELT_COLOR = np.array([68, 62, 58], dtype=np.float32)


class Parcel:
    def __init__(self, pid, bucket, w, h, y0, s, f0):
        self.id, self.bucket, self.w, self.h, self.y0, self.s, self.f0 = pid, bucket, w, h, y0, s, f0
        self.xev = None  # (start_frame, W, amplitude) raised-cosine x offset
        self.yev = None  # (start_frame, W, signed amplitude) raised-cosine y offset
        self.kind = "plain"
        self.color = None
        self.tex_off = (0, 0)


def bump(ev, f):
    if ev is None:
        return 0.0
    f0, w, amp = ev
    t = f - f0
    if t < 0 or t > w:
        return 0.0
    return amp * 0.5 * (1.0 - np.cos(2 * np.pi * t / w))


def overlap_window(depth, gap, frames_wanted):
    """Pick the bump width so the offset exceeds `gap` for ~frames_wanted frames."""
    best, best_err = None, 1e9
    for w in range(4, 16):
        n = sum(1 for t in range(w + 1) if depth * 0.5 * (1 - np.cos(2 * np.pi * t / w)) > gap)
        if abs(n - frames_wanted) < best_err:
            best, best_err = w, abs(n - frames_wanted)
    return best


def plan_phases(rng, nframes, drain):
    """Alternating steady / surge phases, then stalls placed inside them."""
    end_spawn = nframes - drain
    phases, t = [], int(rng.uniform(18, 26) * FPS)
    phases.append([0, t, "steady"])
    while t < end_spawn:
        d = int(rng.uniform(20, 40) * FPS)
        if t + d > end_spawn:
            d = end_spawn - t
        if d < 20 * FPS:
            phases.append([t, nframes, "steady"]); t = nframes; break
        phases.append([t, t + d, "surge"]); t += d
        d2 = int(rng.uniform(25, 45) * FPS)
        phases.append([t, min(t + d2, nframes), "steady"]); t += d2
    if phases[-1][1] < nframes:
        phases[-1][1] = nframes
    surge_i = [i for i, p in enumerate(phases) if p[2] == "surge"]
    steady_i = [i for i, p in enumerate(phases) if p[2] == "steady" and p[1] - p[0] > 8 * FPS and p[1] <= end_spawn]
    n_stall = max(2, int((nframes / FPS) // 150))
    # one stall inside a surge (hardest), one in a steady stretch, then alternate
    pick = []
    if surge_i:
        pick.append(surge_i[min(1, len(surge_i) - 1)])
    if steady_i:
        pick.append(steady_i[min(1, len(steady_i) - 1)])
    k = 0
    while len(pick) < n_stall and (surge_i or steady_i):
        pool = (surge_i if k % 2 == 0 else steady_i) or (surge_i + steady_i)
        pick.append(pool[(k // 2 + 2) % len(pool)]); k += 1
    stalls = []
    for pi in pick:
        a, b, _ = phases[pi]
        for _try in range(20):
            L = int(rng.uniform(1.0, 2.2) * FPS)
            s0 = int(rng.uniform(a + 4 * FPS, max(a + 4 * FPS + 1, b - 4 * FPS - L)))
            if all(abs(s0 - o[0]) > 8 * FPS for o in stalls) and s0 + L < end_spawn:
                stalls.append([s0, s0 + L]); break
    stalls.sort()
    return phases, stalls


def speed_profile(rng, nframes, phases, stalls):
    target = np.zeros(nframes)
    for a, b, m in phases:
        target[a:b] = V_SURGE if m == "surge" else V_STEADY
    for a, b in stalls:
        target[a:b] = 0.0
    v = np.zeros(nframes)
    cur = V_STEADY
    for f in range(nframes):
        d = target[f] - cur
        cur += float(np.clip(d, -1.2, 0.5))  # brakes hard, accelerates gently
        v[f] = max(cur, 0.0) * (1 + rng.normal(0, 0.01) if cur > 0.1 else 1)
    v[v < 0.02] = 0.0
    return v


def plan_parcels(rng, nframes, drain, phases, v, s_belt):
    """Sequential layout in belt-distance space; returns Parcel list (spawn order)."""
    surge = np.zeros(nframes, bool)
    for a, b, m in phases:
        if m == "surge":
            surge[a:b] = True
    parcels, pid, s_next = [], 0, 0.0
    end_spawn = nframes - drain
    prev = None
    while True:
        f0 = int(np.searchsorted(s_belt, s_next))
        if f0 >= end_spawn:
            break
        in_surge = bool(surge[f0])

        def make(s_entry, twin_of=None):
            nonlocal pid
            pid += 1
            b = int(rng.choice(3, p=BUCKET_P))
            sp = SPECS[BUCKETS[b]]
            w = float(rng.uniform(*sp["w"])); h = float(rng.uniform(*sp["h"]))
            p = Parcel(pid, BUCKETS[b], w, h, 0.0, s_entry, int(np.searchsorted(s_belt, s_entry)))
            base = PALETTE[rng.integers(len(PALETTE))]
            p.color = np.clip(base * rng.uniform(0.85, 1.1, 3) * 0.5 + base * 0.5, 0, 255).astype(np.float32)
            p.tex_off = (int(rng.integers(256)), int(rng.integers(256)))
            return p

        lo, hi = BELT_TOP + 4, BELT_BOT - 4
        r = rng.random()
        if in_surge and r < 0.14 and prev is not None:          # lateral twin pair
            a, b = make(s_next), make(s_next + rng.uniform(-0.25, 0.25) * 40)
            a.y0 = rng.uniform(lo, lo + 4 + rng.uniform(0, 14))
            gy = rng.uniform(2, 4)
            b.y0 = a.y0 + a.h + gy
            if b.y0 + b.h > hi:                                  # does not fit: demote to plain
                b.y0 = hi - b.h
            m = rng.uniform(0.22, 0.5) * min(a.h, b.h)
            gap_now = max(b.y0 - (a.y0 + a.h), 0.5)
            depth = gap_now + m
            wlen = overlap_window(depth, gap_now, int(rng.integers(3, 6)))
            b.yev = (b.f0 + int(rng.integers(22, 34)), wlen, -depth)  # B moves up into A
            a.kind, b.kind = "twin", "twin_nudge"
            parcels += [a, b]
            wmax = max(a.w, b.w)
        else:
            p = make(s_next)
            if in_surge and prev is not None and r < 0.14 + 0.24 and prev.kind != "twin":
                p.kind = "x_slip"
                # same lane as predecessor (centre within a quarter of the smaller height)
                cy = prev.y0 + prev.h / 2 + rng.uniform(-0.25, 0.25) * min(p.h, prev.h)
                p.y0 = float(np.clip(cy - p.h / 2, lo, hi - p.h))
                p.s = prev.s + prev.w + rng.uniform(2, 4)
                p.f0 = int(np.searchsorted(s_belt, p.s))
                m = rng.uniform(0.22, 0.5) * min(p.w, prev.w)
                gap0 = p.s - (prev.s + prev.w)
                depth = gap0 + m
                wlen = overlap_window(depth, gap0, int(rng.integers(3, 6)))
                p.xev = (p.f0 + int(rng.integers(22, 34)), wlen, depth)
            else:
                p.y0 = float(rng.uniform(lo, hi - p.h))
            parcels.append(p)
            prev = p
            wmax = p.w
            s_entry = p.s
            gap = (3 + rng.exponential(MEAN_W / 6.0 * 0.9 + 0.1)) if in_surge else (20 + rng.exponential(MEAN_W * 1.1))
            s_next = s_entry + wmax + gap
            continue
        prev = parcels[-1]
        gap = (3 + rng.exponential(MEAN_W / 6.0 * 0.9 + 0.1)) if in_surge else (20 + rng.exponential(MEAN_W * 1.1))
        s_next = s_next + wmax + gap
    return parcels


def pos(p, f, s_belt):
    x = s_belt[f] - p.s - p.w + bump(p.xev, f)
    y = p.y0 + bump(p.yev, f)
    return x, y


def build_background(rng):
    bg = np.empty((H, W, 3), np.float32)
    bg[:] = (30, 30, 34)
    bg[BELT_TOP:BELT_BOT] = BELT_COLOR
    low = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 6) * 16
    fine = rng.normal(0, 1.8, (H, W, 1)).astype(np.float32)
    bg += low[..., None] + fine
    for x in range(0, W, 64):                       # seams (static: see DECISIONS.md)
        bg[BELT_TOP:BELT_BOT, x:x + 2] -= 9
    bg[BELT_TOP - 2:BELT_TOP] = (40, 170, 210)       # rail edge lines
    bg[BELT_BOT:BELT_BOT + 2] = (40, 170, 210)
    return bg


def render(parcels, v, s_belt, nframes, outdir, rng, progress=True):
    bg = build_background(rng)
    tex = rng.normal(0, 4.5, (256, 256)).astype(np.float32)
    noise_pool = [rng.normal(0, 1.6, (H, W, 3)).astype(np.float32) for _ in range(6)]
    writer = FrameWriter(outdir)
    gtf = open(os.path.join(outdir, "gt_frames.jsonl"), "w")
    by_f0 = sorted(parcels, key=lambda p: p.f0)
    active, nxt = [], 0
    cross_frame = {}
    vis_hist = {p.id: [] for p in parcels}
    t0 = time.time()
    for f in range(nframes):
        while nxt < len(by_f0) and by_f0[nxt].f0 <= f:
            active.append(by_f0[nxt]); nxt += 1
        img = bg + noise_pool[f % 6] + rng.normal(0, 0.4)
        idmap = np.zeros((H, W), np.int32)
        own = {}
        rec = {}
        keep = []
        for p in sorted(active, key=lambda q: q.id):
            x, y = pos(p, f, s_belt)
            if x >= W:
                continue
            keep.append(p)
            cx0, cx1 = max(int(np.floor(x)), 0), min(int(np.ceil(x + p.w)), W)
            cy0, cy1 = max(int(np.floor(y)), 0), min(int(np.ceil(y + p.h)), H)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            cols, rows = np.arange(cx0, cx1), np.arange(cy0, cy1)
            ax = np.clip(np.minimum(cols + 1, x + p.w) - np.maximum(cols, x), 0, 1)
            ay = np.clip(np.minimum(rows + 1, y + p.h) - np.maximum(rows, y), 0, 1)
            alpha = (ay[:, None] * ax[None, :])[..., None].astype(np.float32)
            shade = (1.0 + 0.10 * (0.5 - (rows - y) / p.h))[:, None, None]
            tx = tex[np.ix_((rows - cy0 + p.tex_off[0]) % 256, (cols - cx0 + p.tex_off[1]) % 256)][..., None]
            patch = p.color[None, None, :] * shade + tx
            reg = img[cy0:cy1, cx0:cx1]
            img[cy0:cy1, cx0:cx1] = reg * (1 - alpha) + patch * alpha
            solid = alpha[..., 0] > 0.5
            idmap[cy0:cy1, cx0:cx1][solid] = p.id
            own[p.id] = int(solid.sum())
            rec[p.id] = (x, y)
            if p.id not in cross_frame and x + p.w / 2 >= COUNT_X:
                cross_frame[p.id] = f
        active = keep
        counts = np.bincount(idmap.ravel(), minlength=1)
        out = []
        for pid_, (x, y) in rec.items():
            p = parcels[pid_ - 1]
            vis = counts[pid_] / own[pid_] if pid_ < len(counts) and own[pid_] > 0 else 0.0
            vis_hist[pid_].append((f, vis))
            out.append([pid_, round(x, 1), round(y, 1), round(p.w, 1), round(p.h, 1), round(float(vis), 3)])
        gtf.write(json.dumps(out, separators=(",", ":")) + "\n")
        writer.write(np.clip(img, 0, 255).astype(np.uint8))
        if progress and f % 100 == 0:
            print(f"PROGRESS {f / nframes:.3f}", file=sys.stderr, flush=True)
    gtf.close()
    writer.close()
    return cross_frame, vis_hist


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/default")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--duration", type=float, default=300, help="seconds of video (3600 = a full simulated hour)")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    nframes = int(a.duration * FPS)
    drain = 6 * FPS
    phases, stalls = plan_phases(rng, nframes, drain)
    v = speed_profile(rng, nframes, phases, stalls)
    s_belt = np.cumsum(v)
    parcels = plan_parcels(rng, nframes, drain, phases, v, s_belt)
    cross, vis_hist = render(parcels, v, s_belt, nframes, a.out, rng)

    # frames in which the belt is truly stopped
    stopped = v < 0.05
    stall_iv, f = [], 0
    while f < nframes:
        if stopped[f]:
            g = f
            while g < nframes and stopped[g]:
                g += 1
            if g - f >= 10:
                stall_iv.append([f, g])
            f = g
        else:
            f += 1

    gt_parcels = []
    for p in parcels:
        vh = vis_hist[p.id]
        low = [f for f, vis in vh if vis < 0.9]
        # longest run of frames with the parcel noticeably covered
        run = best = 0
        prev_f = None
        for f_, vis in vh:
            if vis < 0.9:
                run = run + 1 if prev_f == f_ - 1 else 1
                best = max(best, run)
                prev_f = f_
            else:
                prev_f = None; run = 0
        gt_parcels.append(dict(
            id=p.id, bucket=p.bucket, w=round(p.w, 1), h=round(p.h, 1), kind=p.kind,
            spawn_frame=p.f0, cross_frame=cross.get(p.id),
            min_visible=round(min(vis for _, vis in vh), 3) if vh else 1.0,
            covered_frames=len(low), covered_run=best))
    mode_at = np.array(["steady"] * nframes, dtype=object)
    for s0, s1, m in phases:
        mode_at[s0:s1] = m
    crossed = [g for g in gt_parcels if g["cross_frame"] is not None]
    summary = dict(
        seed=a.seed, fps=FPS, width=W, height=H, frames=nframes, duration_s=a.duration,
        count_line_x=COUNT_X, belt_top=BELT_TOP, belt_bottom=BELT_BOT,
        spawned=len(gt_parcels), crossed=len(crossed),
        by_bucket={b: sum(1 for g in crossed if g["bucket"] == b) for b in BUCKETS},
        phases=[dict(start=s0, end=s1, mode=m, crossed=sum(1 for g in crossed if s0 <= g["cross_frame"] < s1))
                for s0, s1, m in phases],
        stalls=stall_iv,
        speed=[round(float(x), 2) for x in v[::5]],
    )
    save_json(os.path.join(a.out, "ground_truth.json"), dict(summary=summary, parcels=gt_parcels))
    # calibration = what a real installation knows (geometry, line); NOT the truth
    save_json(os.path.join(a.out, "calibration.json"), dict(
        fps=FPS, width=W, height=H, count_line_x=COUNT_X, belt_top=BELT_TOP, belt_bottom=BELT_BOT, frames=nframes))
    n_ov = sum(1 for g in gt_parcels if g["covered_run"] >= 1)
    runs = [g["covered_run"] for g in gt_parcels if g["covered_run"] >= 1]
    print(f"generated {nframes} frames, {len(parcels)} parcels spawned, {len(crossed)} crossed the line", file=sys.stderr)
    print(f"  per bucket {summary['by_bucket']}; surges={sum(1 for p in phases if p[2]=='surge')}; stalls={stall_iv}", file=sys.stderr)
    print(f"  parcels partly covered (vis<0.9) at some point: {n_ov}; covered-run length frames: "
          f"min={min(runs) if runs else 0} median={int(np.median(runs)) if runs else 0} max={max(runs) if runs else 0}", file=sys.stderr)
    print("DONE", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
