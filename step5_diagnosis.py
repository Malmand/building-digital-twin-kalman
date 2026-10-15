"""Step 5: fault diagnosis with the twin, on real data with injected faults.

1. Thresholds. The CUSUM threshold (on 3-hour-ahead residuals of the twin with fixed parameters) is set
   on July 2019 so that the data give at most one alarm per 30 days; the cooling-effectiveness limit
   (from the tracking twin) is fixed at 0.8. (April and May are not used: their residuals are biased,
   the building being operated differently from the June fortnight the models were fitted on.)
2. False alarms. The twin runs over August to December 2019 as recorded; every alarm there counts as
   false (the data may contain real faults, so this is an upper bound).
3. Injected faults, at random weekday mornings in August to December, one at a time:
       bias      the temperature sensor reads 1 K (or 2 K) high from the onset
       drift     the sensor drifts upwards by 0.5 K per day
       frozen    the sensor repeats its last value for 6 h
       cooling   the units remove 30 % less heat per kW: the measured AC power is scaled by 1/0.7
                 while the temperatures are as recorded (same cooling, more power)
   Detection counts if the detector meant for that fault alarms within the window after the onset.
Results: results/diagnosis_false_alarms.csv, results/diagnosis_injected.csv.

    python step5_diagnosis.py [--zones F7z1] [--n 12]
"""
import argparse
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from twin import data as D
from twin import diagnosis as dg
from twin.runner import load_model, segments, selected_models
from twin.twin import DigitalTwin

OUT = Path(__file__).parent / "results"
TUNE = ("2019-07-01", "2019-07-31 23:55")
TEST = ("2019-08-01", "2019-12-31 23:55")
CUSUM_GRID = (5, 8, 10, 15, 20, 30, 40, 60)
CUSUM_DRIFT = 1.0        # half of a 2-sigma shift: tuned to offsets of about 1 K
FAULTS = {"bias 1 K": 2 * 288, "bias 2 K": 2 * 288, "drift": 4 * 288, "frozen": 72, "cooling": 10 * 288}  # windows (steps)
DETECTOR = {"bias 1 K": "cusum", "bias 2 K": "cusum", "drift": "cusum", "frozen": "frozen", "cooling": "cooling"}
WARMUP = 7 * 288  # one week to settle the twin


def settings(zone):
    s = pd.read_csv(OUT / "twin_settings.csv").set_index("zone")
    return float(s.loc[zone, "q_kappa"]), float(s.loc[zone, "q_dq"]), float(s.loc[zone, "dq_tau_h"])


def run(model, U, Y, qk, qd, tau):
    y0 = Y[~np.isnan(Y)][0]
    tw = DigitalTwin(model, y0, q_kappa=qk, q_dq=qd, dq_tau_h=tau)
    o = tw.run(U, Y)
    return o


def detect(o, U, Y, cusum_h, model):
    return {"cusum": dg.cusum(dg.hstep_residuals(model, U, Y), drift=CUSUM_DRIFT, threshold=cusum_h)[0],
            "frozen": dg.frozen(Y),
            "cooling": dg.cooling(o["kappa"], U[:, 1])}


def events(alarms, gap=72):
    """Merge alarms closer than `gap` steps (6 h) into one event."""
    out, last = [], -10 ** 9
    for a in sorted(alarms):
        if a - last > gap:
            out.append(a)
        last = a
    return out


def alarm_rate(model, df, qk, qd, tau, cusum_h):
    U, Y = D.arrays(df)
    counts, days = {"cusum": 0, "frozen": 0, "cooling": 0}, 0.0
    for a, b in segments(D.complete_inputs(df)):
        if np.isnan(Y[a:b]).all():
            continue
        o = run(model, U[a:b], Y[a:b], qk, qd, tau)
        for k, v in detect(o, U[a:b], Y[a:b], cusum_h, model).items():
            counts[k] += len(events(v))
        days += np.sum(~np.isnan(Y[a:b])) / 288
    return counts, days


def inject(kind, U, Y, k0):
    U, Y = U.copy(), Y.copy()
    n = len(Y)
    if kind.startswith("bias"):
        Y[k0:] += float(kind.split()[1])
    elif kind == "drift":
        Y[k0:] += 0.5 * np.arange(n - k0) / 288
    elif kind == "frozen":
        last = Y[k0 - 1]
        Y[k0:k0 + 72] = last
    elif kind == "cooling":
        U[k0:, 1] /= 0.7  # same heat removed, 1/0.7 more power
    return U, Y


def run_zone(args):
    zone, n_inj, seed = args
    model = load_model(zone)
    qk, qd, tau = settings(zone)
    tune = D.zone_frame(zone, *TUNE)
    cusum_h = CUSUM_GRID[-1]
    for h in CUSUM_GRID:
        c, days = alarm_rate(model, tune, qk, qd, tau, h)
        if days > 0 and c["cusum"] / days * 30 <= 1.0:  # lowest threshold with <= 1 alarm per 30 days
            cusum_h = h
            break
    test = D.zone_frame(zone, *TEST)
    c, days = alarm_rate(model, test, qk, qd, tau, cusum_h)
    fa = dict(zone=zone, cusum_threshold=cusum_h, days=round(days, 1),
              **{f"{k}_alarms_per_30d": (v / days * 30 if days else np.nan) for k, v in c.items()})

    rng = np.random.default_rng(seed)
    U, Y = D.arrays(test)
    ok = D.complete_inputs(test).to_numpy()
    idx = test.index
    rows = []
    for kind, win in FAULTS.items():
        cands = [k for k in range(WARMUP, len(Y) - win)
                 # onsets at 10:00 on weekdays
                 if idx[k].hour == 10 and idx[k].minute == 0 and idx[k].dayofweek < 5
                 and ok[k - WARMUP:k + win].all() and np.mean(~np.isnan(Y[k - WARMUP:k + win])) > 0.9]
        for k0 in rng.choice(cands, size=min(n_inj, len(cands)), replace=False) if cands else []:
            sl = slice(k0 - WARMUP, k0 + win)
            Ui, Yi = inject(kind, U[sl], Y[sl], WARMUP)
            o = run(model, Ui, Yi, qk, qd, tau)
            al = detect(o, Ui, Yi, cusum_h, model)
            first = dg.first_after(al[DETECTOR[kind]], WARMUP, WARMUP + win)
            before = len(events([a for v in al.values() for a in v if a < WARMUP]))
            rows.append(dict(zone=zone, fault=kind, onset=str(idx[k0]), detected=first is not None,
                             delay_h=(first - WARMUP) / 12 if first is not None else np.nan,
                             alarms_before_onset=before))
    det = pd.DataFrame(rows).groupby("fault")["detected"].mean().round(2).to_dict() if rows else {}
    logging.info("%s: CUSUM h=%s, false alarms/30d %s, detected %s", zone, cusum_h,
                 {k: round(v, 2) for k, v in fa.items() if k.endswith("30d")}, det)
    return fa, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT / "step5.log"), logging.StreamHandler(sys.stdout)])
    zones = a.zones.split(",") if a.zones else sorted(selected_models())
    fas, rows = [], []
    with Pool(a.workers) as pool:
        for fa, r in pool.imap_unordered(run_zone, [(z, a.n, i) for i, z in enumerate(zones)]):
            fas.append(fa)
            rows += r
    pd.DataFrame(fas).to_csv(OUT / "diagnosis_false_alarms.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "diagnosis_injected.csv", index=False)


if __name__ == "__main__":
    main()
