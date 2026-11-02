# Results (measured, not estimated)

All numbers come from `python3 -m conveyor.sweep` (seeds 2, 3, 4, 300 s each, never used while tuning) and the development seed 1.
Error = (counted - true) / true, where *true* is the generator's own log of parcels that crossed the counting line.

## Stride 1 (the chosen setting), per seed

| seed | true | counted | error | steady err | surge err | exactly once | missed | double | spurious | count events during stall | ID switches | bucket accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 (dev) | 1235 | 1227 | -0.65% | +0.00% | -0.91% | 98.46% | 14 | 6 | 0 | 0 | 2 | 97.29% |
| 2 | 1217 | 1212 | -0.41% | +0.00% | -0.56% | 98.85% | 10 | 4 | 1 | 0 | 1 | 97.76% |
| 3 | 1193 | 1185 | -0.67% | -0.30% | -0.82% | 97.82% | 17 | 9 | 0 | 0 | 8 | 97.17% |
| 4 | 1049 | 1045 | -0.38% | -0.27% | -0.44% | 97.81% | 15 | 11 | 0 | 0 | 9 | 97.17% |

**Held-out pooled (seeds 2-4):** 3442 counted vs 3459 true = **-0.49%**; steady stretches 1033 vs 1035 = **-0.19%**; surges 2409 vs 2424 = **-0.62%**. Net error is small partly because misses and double counts cancel; exactly-once is 98.18% (42 missed, 24 double, 1 spurious of 3459).

Parcels partly covered for 3+ frames: 362/378 counted exactly once (95.8%). Parcels that were at some point >=70% covered: 56/61 (91.8%). The generator's overlaps last 3-5 frames, so *occluded for most of the crossing* is **not** exercised; that optional case is not claimed.

## Sampling-rate sweep (process every n-th frame), pooled over seeds 2-4

| stride | effective fps at 30 fps input | total err | steady err | surge err | exactly once | missed | double | count events in stall | ID switches | tracker ms / processed frame | x real time |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 30 | -0.49% | -0.19% | -0.62% | 98.18% | 42 | 24 | 0 | 18 | 3.89 | 7.0 |
| 2 | 15 | -0.66% | -0.19% | -0.87% | 98.01% | 46 | 23 | 0 | 8 | 3.81 | 15.2 |
| 3 | 10 | -0.38% | -0.29% | -0.41% | 97.86% | 45 | 32 | 1 | 13 | 3.81 | 22.7 |
| 4 | 7.5 | -0.81% | -0.29% | -1.03% | 97.63% | 55 | 27 | 0 | 22 | 3.91 | 29.5 |
| 6 | 5 | -1.53% | -2.90% | -0.95% | 95.63% | 106 | 53 | 13 | 54 | 3.87 | 44.7 |
| 8 | 3.75 | -12.29% | -11.40% | -12.67% | 84.30% | 491 | 66 | 13 | 241 | 3.90 | 59.0 |

Timings are from a 2-core cloud container, JPEG decode included in 'x real time' (wall clock of the whole run vs video length).
