"""Turn data/sweep/sweep.json (+ the dev seed) into RESULTS.md.   python3 -m conveyor.report"""
import json
import os

from . import evaluate


def pooled(rows, key):
    g = sum(r[key]["gt"] for r in rows); c = sum(r[key]["counted"] for r in rows)
    return g, c, 100 * (c - g) / g


def main():
    rows = json.load(open("data/sweep/sweep.json"))
    L = ["# Results (measured, not estimated)", "",
         "All numbers come from `python3 -m conveyor.sweep` (seeds 2, 3, 4, 300 s each, never used while tuning) and the development seed 1.",
         "Error = (counted - true) / true, where *true* is the generator's own log of parcels that crossed the counting line.", ""]
    dev = os.path.join("data", "s1", "metrics.json")
    L += ["## Stride 1 (the chosen setting), per seed", "",
          "| seed | true | counted | error | steady err | surge err | exactly once | missed | double | spurious | count events during stall | ID switches | bucket accuracy |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    def line(name, t, s, u, st, idn, b):
        return (f"| {name} | {t['gt']} | {t['counted']} | {t['error_pct']:+.2f}% | {s['error_pct']:+.2f}% | {u['error_pct']:+.2f}% | {t['exactly_once_pct']}% | "
                f"{t['missed']} | {t['double_counted']} | {t['spurious']} | {st['count_events_during_stall']} | {idn['id_switches']} | {b}% |")
    if os.path.exists(dev):
        m = json.load(open(dev))
        L.append(line("1 (dev)", m["total"], m["steady"], m["surge"], m["stall"], m["identity"], m["buckets"]["accuracy_pct"]))
    s1 = [r for r in rows if r["stride"] == 1]
    for r in s1:
        L.append(line(r["seed"], r["total"], r["steady"], r["surge"], r["stall"], r["ident"], r["buckets"]))
    g, c, e = pooled(s1, "total"); gs, cs, es = pooled(s1, "steady"); gu, cu, eu = pooled(s1, "surge")
    L += ["", f"**Held-out pooled (seeds 2-4):** {c} counted vs {g} true = **{e:+.2f}%**; steady stretches {cs} vs {gs} = **{es:+.2f}%**; "
          f"surges {cu} vs {gu} = **{eu:+.2f}%**. Net error is small partly because misses and double counts cancel; "
          f"exactly-once is {sum(r['total']['exactly_once'] for r in s1) / sum(r['total']['gt'] for r in s1) * 100:.2f}% "
          f"({sum(r['total']['missed'] for r in s1)} missed, {sum(r['total']['double_counted'] for r in s1)} double, {sum(r['total']['spurious'] for r in s1)} spurious of {g}).", ""]
    oc = sum(r["occ"]["parcels"] for r in s1); oo = sum(r["occ"]["exactly_once"] for r in s1)
    hc = sum(r["heavy"]["parcels"] for r in s1); ho = sum(r["heavy"]["exactly_once"] for r in s1)
    L += [f"Parcels partly covered for 3+ frames: {oo}/{oc} counted exactly once ({100 * oo / oc:.1f}%). "
          f"Parcels that were at some point >=70% covered: {ho}/{hc} ({100 * ho / hc:.1f}%). "
          "The generator's overlaps last 3-5 frames, so *occluded for most of the crossing* is **not** exercised; that optional case is not claimed.", ""]
    L += ["## Sampling-rate sweep (process every n-th frame), pooled over seeds 2-4", "",
          "| stride | effective fps at 30 fps input | total err | steady err | surge err | exactly once | missed | double | count events in stall | ID switches | tracker ms / processed frame | x real time |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for st in sorted({r["stride"] for r in rows}):
        rs = [r for r in rows if r["stride"] == st]
        g, c, e = pooled(rs, "total"); _, _, es = pooled(rs, "steady"); _, _, eu = pooled(rs, "surge")
        once = 100 * sum(r["total"]["exactly_once"] for r in rs) / g
        L.append(f"| {st} | {30 / st:g} | {e:+.2f}% | {es:+.2f}% | {eu:+.2f}% | {once:.2f}% | {sum(r['total']['missed'] for r in rs)} | "
                 f"{sum(r['total']['double_counted'] for r in rs)} | {sum(r['stall']['count_events_during_stall'] for r in rs)} | "
                 f"{sum(r['ident']['id_switches'] for r in rs)} | {sum(r['timing']['track_ms_per_processed'] for r in rs) / len(rs):.2f} | "
                 f"{sum(r['timing']['realtime_factor'] for r in rs) / len(rs):.1f} |")
    L += ["", "Timings are from a 2-core cloud container, JPEG decode included in 'x real time' (wall clock of the whole run vs video length)."]
    open("RESULTS.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
