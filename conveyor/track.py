"""Detect, track and count parcels.

    python -m conveyor.track --data data/s1 [--stride 1]

Pipeline per processed frame:
 1. DETECT   background model (temporal median) -> |frame-bg| threshold -> connected blobs.
 2. PREDICT  every track moves with ONE shared belt velocity (estimated each frame from
             cleanly-observed parcels) -- parcels on a belt move together, so a hidden
             parcel can be advanced without seeing it.
 3. ASSOCIATE each track picks the blob covering most of its predicted box.
             - blob with exactly one track and a consistent size -> clean match, update from the blob
             - blob with >=2 tracks -> MERGED (overlap/touching): tracks coast on the belt prediction
             - blob with no track (or an unexplained residual left over inside a blob) -> new tentative track
             - track with no blob -> MISSED, coast, dropped after max_coast frames
 4. COUNT    a confirmed track adds one to the total, once, the first time its centre
             crosses the counting line while it is supported by a blob.
Bucket (small/medium/large) comes from the box area measured while the parcel was
isolated and fully inside the frame.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

from .common import FrameStore, load_json, save_json

CLEAN, MERGED, MISSED, TENT = 0, 1, 2, 3

DEFAULTS = dict(
    stride=1,            # process every n-th frame (the sampling-rate knob)
    diff_thr=45,         # sum over BGR of |frame - background|
    min_blob=90,         # px^2
    cov_min=0.30,        # fraction of predicted box that must lie in a blob
    size_tol=0.30,       # blob w/h vs track w/h for a clean update
    new_min_area=150,
    new_min_side=7,
    dup_iou=0.55,
    max_coast=30,
    confirm_hits=3,
    v_alpha=0.7,
    bucket_cuts=(1340, 3064),   # area cut points small|medium|large (Bayes-optimal for the class size ranges)
    edge_bias=1.1,       # detected boxes are ~1.1 px larger per side-pair than the parcel (soft edges); subtracted before area
    bg_samples=61,
    max_parcel_w=96,     # installation knowledge: largest parcel the depot handles (px at this camera)
    max_parcel_h=72,
    entry_zone=80,       # new tracks may only be born this close to the entry edge (px)
    cut_thr=30,          # colour step (L1 over BGR) that counts as a boundary between two parcels
)


def bucket_of(area, cuts):
    return 0 if area < cuts[0] else (1 if area < cuts[1] else 2)


def iou(a, b):
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = min(ax1, bx1) - max(a[0], b[0]); ih = min(ay1, by1) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


class Track:
    def __init__(self, tid, x, y, w, h, f, entering):
        self.id, self.x, self.y, self.w, self.h = tid, x, y, w, h
        self.px, self.py = x, y
        self.vy = 0.0
        self.hits, self.miss = 1, 0
        self.confirmed = False
        self.counted = False
        self.state = TENT
        self.sizes = []
        self.born = f
        self.entering = entering
        self.last_edges = None   # (frame, left|None, right|None)
        self.blob = 0
        self.cov = None

    def size(self):
        if self.sizes:
            a = np.array(self.sizes[-8:])
            return float(np.median(a[:, 0])), float(np.median(a[:, 1]))
        return self.w, self.h


def estimate_background(store, n):
    idx = np.linspace(0, len(store) - 1, n).astype(int)
    stack = np.stack([store.get(int(i)) for i in idx])
    return np.median(stack, axis=0).astype(np.uint8)


class Tracker:
    def __init__(self, cal, cfg):
        self.cal, self.cfg = cal, cfg
        self.W, self.H = cal["width"], cal["height"]
        self.L = cal["count_line_x"]
        self.tracks, self.next_id = [], 1
        self.v = 0.0
        self.last_f = None
        self.total = [0, 0, 0]
        self.events = []
        self.k3 = np.ones((3, 3), np.uint8)
        self.k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # ---------------------------------------------------------------- detection
    def detect(self, img, bg):
        c = self.cfg
        d = cv2.absdiff(img, bg).astype(np.uint16).sum(axis=2)
        mask = (d > c["diff_thr"]).astype(np.uint8)
        mask[:int(self.cal["belt_top"]) - 1] = 0
        mask[int(self.cal["belt_bottom"]) + 1:] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.k3)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        valid = np.zeros(n, bool)
        valid[1:] = stats[1:, cv2.CC_STAT_AREA] >= c["min_blob"]
        return n, labels, stats, valid

    def coverage(self, labels, n, x, y, w, h):
        x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
        x1, y1 = min(int(round(x + w)), self.W), min(int(round(y + h)), self.H)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = labels[y0:y1, x0:x1]
        return np.bincount(crop.ravel(), minlength=n)[:n] / crop.size

    # ---------------------------------------------------------------- one step
    def step(self, f, img, bg):
        c = self.cfg
        n, labels, stats, valid = self.detect(img, bg)
        dt = 1 if self.last_f is None else f - self.last_f
        self.last_f = f
        W, H = self.W, self.H
        blobs = [(int(stats[j, 0]), int(stats[j, 1]), int(stats[j, 2]), int(stats[j, 3]), int(stats[j, 4]), j)
                 for j in range(1, n) if valid[j]]

        # PREDICT: all parcels ride the same belt, so one shared velocity advances every track
        for t in self.tracks:
            t.px, t.py = t.x + self.v * dt, t.y + t.vy * dt
            vx = max(t.px, 0.0)
            cov = self.coverage(labels, n, vx, t.py, t.px + t.w - vx, t.h)
            t.blob = 0
            if cov is not None:
                cov = cov * valid
                j = int(np.argmax(cov))
                if j > 0 and cov[j] >= c["cov_min"]:
                    t.blob = j
        members = {}
        for t in self.tracks:
            if t.blob:
                members.setdefault(t.blob, []).append(t)

        vmeas, births, handled = [], [], set()
        for (bx, by, bw, bh, area, j) in blobs:
            tr = members.get(j, [])
            in_entry = bx < c["entry_zone"]
            pieces = None
            if in_entry:   # near the entry edge, look for several parcels inside one blob
                pieces = self._split(img, labels[by:by + bh, bx:bx + bw] == j, bx, by, force=True)
            if pieces is not None and len(pieces) > 1:
                self._composite(f, dt, pieces, tr, vmeas, handled, births)
                continue
            touch_l, touch_r = bx <= 0, bx + bw >= W
            if len(tr) == 1:
                t = tr[0]
                tw, th = t.size() if t.sizes else (t.w, t.h)
                ok_h = abs(bh - th) <= c["size_tol"] * th + 2 or not t.sizes
                ok_w = touch_l or touch_r or abs(bw - tw) <= c["size_tol"] * tw + 2 or not t.sizes
                if ok_h and ok_w and bw <= c["max_parcel_w"] + 4 and bh <= c["max_parcel_h"] + 4:
                    self._clean_update(t, f, dt, bx, by, bw, bh, touch_l, touch_r, vmeas)
                    handled.add(t.id)
                    continue
            if tr:                     # overlap / touching: tracks coast on the belt prediction
                for t in tr:
                    t.state, t.miss = MERGED, 0
                    t.hits += 1
                    handled.add(t.id)
            elif in_entry:             # nothing explains this blob and it is at the entry edge: new parcel
                births.append((float(bx), float(by), float(bw), float(bh), bx <= 0))
        # global belt velocity from cleanly-observed parcels
        if vmeas:
            self.v = (1 - c["v_alpha"]) * self.v + c["v_alpha"] * float(np.median(vmeas))
            if self.v < 0.05:
                self.v = 0.0

        # COAST everything that was not updated from a blob
        for t in self.tracks:
            if t.id in handled and t.state == CLEAN:
                continue
            t.x, t.y = t.px, t.py
            t.vy *= 0.85
            if t.id not in handled:
                t.state = MISSED
                t.miss += 1
        # BIRTHS (entry zone only)
        for (x, y, w, h, entering) in births:
            if x > c["entry_zone"] or (x > 1 and (w < 20 or h < 16)):
                continue
            if any(iou((x, y, w, h), (max(t.x, 0.0), t.y, t.x + t.w - max(t.x, 0.0), t.h)) > 0.3 for t in self.tracks):
                continue
            self.tracks.append(Track(self.next_id, x, y, w, h, f, entering))
            self.next_id += 1
        # CONFIRM / PRUNE
        for t in self.tracks:
            if not t.confirmed and t.hits >= c["confirm_hits"] and t.state != MISSED:
                t.confirmed = True
        keep = [t for t in self.tracks
                if not (not t.confirmed and t.miss >= 2) and not (t.miss > c["max_coast"] or t.x >= W)]
        keep.sort(key=lambda t: -t.hits)           # duplicate suppression keeps the established track
        out = []
        for t in keep:
            if any(iou((t.x, t.y, t.w, t.h), (u.x, u.y, u.w, u.h)) > c["dup_iou"] for u in out):
                continue
            out.append(t)
        self.tracks = out

        # COUNT: once per track, when its centre crosses the line while supported by a blob
        new_events = []
        for t in self.tracks:
            if t.confirmed and not t.counted and t.state in (CLEAN, MERGED) and t.x + t.w / 2 >= self.L:
                t.counted = True
                w_, h_ = t.size()
                b = bucket_of((w_ - c["edge_bias"]) * (h_ - c["edge_bias"]), c["bucket_cuts"])
                self.total[b] += 1
                self.events.append(dict(frame=f, id=t.id, bucket=b, w=round(w_, 1), h=round(h_, 1),
                                        cx=round(t.x + t.w / 2, 1), cy=round(t.y + t.h / 2, 1), state=int(t.state)))
                new_events.append([t.id, b])
        return dict(
            f=f, v=round(self.v, 2),
            tr=[[t.id, round(t.x, 1), round(t.y, 1), round(t.w, 1), round(t.h, 1),
                 3 if not t.confirmed else int(t.state), int(t.counted)] for t in self.tracks],
            pr=[[t.id, round(t.px, 1), round(t.py, 1), round(t.w, 1), round(t.h, 1)] for t in self.tracks],
            bl=[[b[0], b[1], b[2], b[3]] for b in blobs],
            cnt=[sum(self.total)] + self.total, ev=new_events)

    def _composite(self, f, dt, pieces, tr, vmeas, handled, births):
        """One blob, several parcels: match visible pieces to tracks by IoU, birth the rest."""
        cand = []
        for t in tr:
            vx = max(t.px, 0.0)
            cand.append((t, (vx, t.py, t.px + t.w - vx, t.h)))
        pairs = sorted(((iou(p, vb), pi, ti) for pi, p in enumerate(pieces) for ti, (_, vb) in enumerate(cand)), reverse=True)
        used_p, used_t = set(), set()
        for sc, pi, ti in pairs:
            if sc < 0.3 or pi in used_p or ti in used_t:
                continue
            used_p.add(pi); used_t.add(ti)
            x, y, w, h = pieces[pi]
            self._clean_update(cand[ti][0], f, dt, int(x), int(y), int(w), int(h), x <= 0, False, vmeas)
            handled.add(cand[ti][0].id)
        for ti, (t, _) in enumerate(cand):
            if ti not in used_t:
                t.state, t.miss = MERGED, 0
                t.hits += 1
                handled.add(t.id)
        for pi, (x, y, w, h) in enumerate(pieces):
            if pi not in used_p:
                births.append((float(x), float(y), float(w), float(h), x <= 0))

    def _clean_update(self, t, f, dt, bx, by, bw, bh, touch_l, touch_r, vmeas):
        left = None if touch_l else float(bx)
        right = None if touch_r else float(bx + bw)
        # belt speed measurement from an edge visible in both consecutive observations
        if t.last_edges is not None and t.last_edges[0] == f - dt:
            _, pl, pr_ = t.last_edges
            if left is not None and pl is not None:
                vmeas.append((left - pl) / dt)
            elif touch_l and pr_ is not None:
                vmeas.append((float(bx + bw) - pr_) / dt)
            elif right is not None and pr_ is not None:
                vmeas.append((right - pr_) / dt)
        t.last_edges = (f, left, right if not touch_l else float(bx + bw))
        if touch_l:                       # entering: right edge known, width not yet
            t.entering = True
            t.w = max(t.w, float(bx + bw))          # visible width is a lower bound on the width
            t.x = bx + bw - t.w
            t.y, t.h = float(by), float(bh)
        elif touch_r:                     # leaving
            t.x, t.y = float(bx), float(by)
        else:
            t.entering = False
            t.x, t.y, t.w, t.h = float(bx), float(by), float(bw), float(bh)
            t.sizes.append((bw, bh))
        t.vy = 0.5 * t.vy + 0.5 * (by - t.py) / dt if t.hits > 1 else 0.0
        t.state = CLEAN
        t.hits += 1
        t.miss = 0

    # ---- split a blob into visible parcel rectangles along colour boundaries
    def _split(self, img, m, ox, oy, depth=0, force=False):
        c = self.cfg
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            return []
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        m = m[y0:y1, x0:x1]; ox += x0; oy += y0
        h, w = m.shape
        fill = m.sum() / (w * h)
        if not (force and depth == 0):
            if (w <= c["max_parcel_w"] and h <= c["max_parcel_h"] and fill >= 0.88) or depth >= 3:
                return [(ox, oy, w, h)]
        if w < 16 and h < 16:
            return [(ox, oy, w, h)]
        C = img[oy:oy + h, ox:ox + w].astype(np.float32)
        me = cv2.erode(m.astype(np.uint8), self.k5).astype(bool)
        best = (0.0, 1, 0)
        for axis in (1, 0):
            Cm, mm = (C, me) if axis == 1 else (C.transpose(1, 0, 2), me.T)
            s_, p_ = self._best_cut(Cm, mm)
            if s_ > best[0]:
                best = (s_, axis, p_)
        if best[0] < c["cut_thr"]:
            return [(ox, oy, w, h)]
        _, axis, p = best
        parts = [(m[:, :p], 0, 0), (m[:, p:], p, 0)] if axis == 1 else [(m[:p, :], 0, 0), (m[p:, :], 0, p)]
        out = []
        for pm, dx, dy in parts:
            if pm.sum() >= c["min_blob"]:
                out += self._split(img, pm, ox + dx, oy + dy, depth + 1)
        return out if out else [(ox, oy, w, h)]

    @staticmethod
    def _best_cut(C, m):
        """Strongest vertical colour step: mean |left band - right band| over rows where both exist."""
        h, w = m.shape
        n = w - 11
        if n <= 0:
            return 0.0, w // 2
        Lb = (C[:, 0:n] + C[:, 1:n + 1] + C[:, 2:n + 2]) / 3
        Rb = (C[:, 5:5 + n] + C[:, 6:6 + n] + C[:, 7:7 + n]) / 3
        valid = (m[:, 0:n] & m[:, 1:n + 1] & m[:, 2:n + 2] & m[:, 5:5 + n] & m[:, 6:6 + n] & m[:, 7:7 + n])
        nv = valid.sum(0)
        d = (np.abs(Lb - Rb).sum(-1) * valid).sum(0) / np.maximum(nv, 1)
        d[nv < 8] = 0
        d[:4] = 0                          # each side of a cut must keep >= 8 px
        i = int(np.argmax(d))
        return float(d[i]), i + 4          # cut sits between columns i+4+... (middle of the 3-px gap)


def run(data_dir, out_dir=None, **over):
    cfg = dict(DEFAULTS); cfg.update(over)
    out_dir = out_dir or data_dir
    store = FrameStore(data_dir)
    cal = load_json(os.path.join(data_dir, "calibration.json"))
    t0 = time.time()
    bg = estimate_background(store, cfg["bg_samples"])
    t_bg = time.time() - t0
    trk = Tracker(cal, cfg)
    n = len(store)
    fout = open(os.path.join(out_dir, "tracks.jsonl"), "w")
    t_dec = t_trk = 0.0
    processed = 0
    t_all = time.time()
    for f in range(0, n, cfg["stride"]):
        a = time.perf_counter()
        img = store.get(f)
        b = time.perf_counter()
        rec = trk.step(f, img, bg)
        c_ = time.perf_counter()
        t_dec += b - a; t_trk += c_ - b
        processed += 1
        fout.write(json.dumps(rec, separators=(",", ":")) + "\n")
        if f % 300 == 0:
            print(f"PROGRESS {f / n:.3f}", file=sys.stderr, flush=True)
    fout.close()
    wall = time.time() - t_all
    save_json(os.path.join(out_dir, "count_events.json"), trk.events)
    meta = dict(config=cfg, frames=n, processed=processed, total=sum(trk.total), by_bucket=trk.total,
                timing=dict(background_s=round(t_bg, 2), decode_ms_per_processed=round(1000 * t_dec / processed, 3),
                            track_ms_per_processed=round(1000 * t_trk / processed, 3),
                            wall_s=round(wall, 2), video_s=n / cal["fps"],
                            realtime_factor=round((n / cal["fps"]) / wall, 1)))
    save_json(os.path.join(out_dir, "track_meta.json"), meta)
    print(f"tracked: total={sum(trk.total)} by_bucket={trk.total} wall={wall:.1f}s "
          f"({meta['timing']['realtime_factor']}x real time)", file=sys.stderr)
    print("DONE", file=sys.stderr, flush=True)
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/default")
    ap.add_argument("--out", default=None)
    for k, v in DEFAULTS.items():
        if isinstance(v, (int, float)):
            ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=v)
    a = ap.parse_args(argv)
    over = {k: getattr(a, k) for k, v in DEFAULTS.items() if isinstance(v, (int, float))}
    run(a.data, a.out, **over)


if __name__ == "__main__":
    main()
