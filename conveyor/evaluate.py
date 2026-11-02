"""Score a tracker run against the generator's ground truth.

    python -m conveyor.evaluate --data data/s1

Every count event is attributed to the ground-truth parcel nearest its centre at
that frame. Per ground-truth parcel: 0 events = missed, 1 = correct, >=2 = double
count. Events near no parcel are spurious. Errors are reported per phase type.
"""
import argparse
import json
import os
import sys

import numpy as np

from .common import load_json, save_json
from .track import iou

BN = ["small", "medium", "large"]


def phase_of(f, phases, stalls):
    for a, b in stalls:
        if a <= f < b:
            return "stall"
    for p in phases:
        if p["start"] <= f < p["end"]:
            return p["mode"]
    return "steady"


def evaluate(data_dir):
    gt = load_json(os.path.join(data_dir, "ground_truth.json"))
    S, P = gt["summary"], gt["parcels"]
    events = load_json(os.path.join(data_dir, "count_events.json"))
    meta = load_json(os.path.join(data_dir, "track_meta.json"))
    gtf = [json.loads(l) for l in open(os.path.join(data_dir, "gt_frames.jsonl"))]
    phases, stalls = S["phases"], S["stalls"]
    byid = {p["id"]: p for p in P}
    n = S["frames"]

    attrib = {p["id"]: [] for p in P}
    spurious = []
    for e in events:
        f = e["frame"]
        best, bd = None, 1e9
        for r in gtf[f]:
            cx, cy = r[1] + r[3] / 2, r[2] + r[4] / 2
            d = np.hypot(cx - e["cx"], cy - e["cy"])
            if d < bd:
                best, bd = r[0], d
        # tolerance: ride within the parcel's own extent (+ a stride's worth of belt travel)
        if best is not None and bd <= 0.6 * max(byid[best]["w"], byid[best]["h"]) + 14 * meta["config"]["stride"]:
            attrib[best].append(e)
        else:
            spurious.append(e)

    def tally(sel):
        gtn = len(sel)
        once = sum(1 for p in sel if len(attrib[p["id"]]) == 1)
        missed = sum(1 for p in sel if len(attrib[p["id"]]) == 0)
        dup = sum(len(attrib[p["id"]]) - 1 for p in sel if len(attrib[p["id"]]) > 1)
        return gtn, once, missed, dup

    result = {}
    for mode in ("steady", "surge"):
        sel = [p for p in P if p["cross_frame"] is not None and phase_of(p["cross_frame"], phases, stalls) == mode]
        gtn, once, missed, dup = tally(sel)
        spur = sum(1 for e in spurious if phase_of(e["frame"], phases, stalls) == mode)
        pred = sum(len(attrib[p["id"]]) for p in sel) + spur
        result[mode] = dict(gt=gtn, counted=pred, error_pct=round(100 * (pred - gtn) / max(gtn, 1), 2),
                            exactly_once=once, missed=missed, double_counted=dup, spurious=spur)
    # stall: the count must not increase while the belt is stopped
    stall_events = [e for e in events if any(a <= e["frame"] < b for a, b in stalls)]
    result["stall"] = dict(intervals=stalls, count_events_during_stall=len(stall_events),
                           frames_stopped=sum(b - a for a, b in stalls))
    sel_all = [p for p in P if p["cross_frame"] is not None]
    gtn, once, missed, dup = tally(sel_all)
    spur = len(spurious)
    pred = len(events)
    result["total"] = dict(gt=S["crossed"], spawned=S["spawned"], counted=pred, error_pct=round(100 * (pred - S["crossed"]) / S["crossed"], 2),
                           exactly_once=once, missed=missed, double_counted=dup, spurious=spur,
                           exactly_once_pct=round(100 * once / gtn, 2))
    # occlusion subsets
    occ = [p for p in sel_all if p["covered_run"] >= 3]
    heavy = [p for p in sel_all if p["min_visible"] <= 0.3 and p["covered_run"] >= 2]
    for name, sel in (("occluded_3plus_frames", occ), ("heavily_occluded_vis_le_30pct", heavy)):
        g, o, m, d = tally(sel)
        result[name] = dict(parcels=g, exactly_once=o, missed=m, double_counted=d,
                            exactly_once_pct=round(100 * o / g, 2) if g else None)
    # buckets on parcels counted exactly once
    conf = np.zeros((3, 3), int)
    for p in sel_all:
        if len(attrib[p["id"]]) == 1:
            conf[BN.index(p["bucket"]), attrib[p["id"]][0]["bucket"]] += 1
    result["buckets"] = dict(confusion_rows_truth_cols_pred=conf.tolist(), labels=BN,
                             accuracy_pct=round(100 * np.trace(conf) / max(conf.sum(), 1), 2),
                             pred_totals=meta["by_bucket"], gt_totals=[S["by_bucket"][b] for b in BN])
    # identity: match confirmed tracks to GT boxes per processed frame
    tracks = [json.loads(l) for l in open(os.path.join(data_dir, "tracks.jsonl"))]
    last = {}
    switches = 0
    frag = {}
    discrepancies = []
    for rec in tracks:
        f = rec["f"]
        used = set()
        for tr in rec["tr"]:
            if tr[5] == 3:
                continue
            best, bi = None, 0.3
            for r in gtf[f]:
                if r[0] in used or r[5] < 0.25:
                    continue
                i = iou((tr[1], tr[2], tr[3], tr[4]), (r[1], r[2], r[3], r[4]))
                if i > bi:
                    best, bi = r[0], i
            if best is None:
                continue
            used.add(best)
            if best in last and last[best] != tr[0]:
                switches += 1
                discrepancies.append(dict(frame=f, kind="id_switch", gt=best, was=last[best], now=tr[0]))
            last[best] = tr[0]
            frag.setdefault(best, set()).add(tr[0])
    result["identity"] = dict(id_switches=switches,
                              parcels_with_multiple_ids=sum(1 for v in frag.values() if len(v) > 1))
    for p in sel_all:
        k = len(attrib[p["id"]])
        if k == 0:
            discrepancies.append(dict(frame=p["cross_frame"], kind="missed", gt=p["id"]))
        elif k > 1:
            discrepancies.append(dict(frame=attrib[p["id"]][-1]["frame"], kind="double", gt=p["id"]))
    for e in spurious:
        discrepancies.append(dict(frame=e["frame"], kind="spurious", track=e["id"]))
    discrepancies.sort(key=lambda d: d["frame"])
    result["discrepancies"] = discrepancies
    result["timing"] = meta["timing"]
    result["config"] = meta["config"]
    # cumulative curves (every 5 frames) for the page's timeline
    cum_gt = np.zeros(n + 1, int)
    for p in sel_all:
        cum_gt[p["cross_frame"] + 1:] += 1
    cum_pred = np.zeros(n + 1, int)
    for e in events:
        cum_pred[e["frame"] + 1:] += 1
    result["timeline"] = dict(step=5, gt=cum_gt[::5].tolist(), pred=cum_pred[::5].tolist())
    # per-GT-parcel overlap list for the "jump to overlap" button
    result["overlaps"] = [dict(frame=p["cross_frame"] - 30, gt=p["id"], kind=p["kind"], min_visible=p["min_visible"])
                          for p in P if p["covered_run"] >= 3 and p["cross_frame"] is not None]
    result["gt_crossings"] = sorted([p["cross_frame"], BN.index(p["bucket"])] for p in sel_all)
    result["count_events"] = [[e["frame"], e["id"], e["bucket"]] for e in events]
    result["gt_phases"] = phases
    result["gt_stalls"] = stalls
    save_json(os.path.join(data_dir, "metrics.json"), result)
    return result


def summarize(r):
    lines = []
    t = r["total"]
    lines.append(f"TOTAL  gt={t['gt']} counted={t['counted']} error={t['error_pct']:+.2f}%  exactly-once={t['exactly_once_pct']}%"
                 f"  missed={t['missed']} double={t['double_counted']} spurious={t['spurious']}")
    for m in ("steady", "surge"):
        s = r[m]
        lines.append(f"{m:7s} gt={s['gt']:4d} counted={s['counted']:4d} error={s['error_pct']:+.2f}%  once={s['exactly_once']} missed={s['missed']} double={s['double_counted']} spurious={s['spurious']}")
    lines.append(f"stall   count events while belt stopped = {r['stall']['count_events_during_stall']} (stopped frames {r['stall']['frames_stopped']})")
    for k in ("occluded_3plus_frames", "heavily_occluded_vis_le_30pct"):
        s = r[k]
        lines.append(f"{k}: parcels={s['parcels']} once={s['exactly_once_pct']}% missed={s['missed']} double={s['double_counted']}")
    b = r["buckets"]
    lines.append(f"bucket accuracy={b['accuracy_pct']}%  pred={b['pred_totals']} gt={b['gt_totals']}")
    lines.append(f"identity: switches={r['identity']['id_switches']} multi-id parcels={r['identity']['parcels_with_multiple_ids']}")
    lines.append(f"timing: {r['timing']['track_ms_per_processed']} ms track + {r['timing']['decode_ms_per_processed']} ms decode per processed frame; {r['timing']['realtime_factor']}x real time")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/default")
    a = ap.parse_args(argv)
    print(summarize(evaluate(a.data)))


if __name__ == "__main__":
    main()
