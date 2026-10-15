"""Step 6: flexibility of the air-conditioning, tested on the real building first.

A. Validation on natural experiments. Whenever a zone's AC power drops by more than 80 % and stays down
   for at least 2 h (the units switching off, or most of them),
   the twin (synchronised on the data up to that moment) predicts the temperature rise from the recorded
   inputs; the prediction is compared with the measurement 15 min to 2 h later, including whether the
   measurement falls inside the twin's 95 % band.
B. Demand-response scenarios. On every weekday from July to December at 14:00, the twin is asked what
   happens if the units are switched off for 30, 60 or 120 min, or run at half power for 2 h, instead of
   what was recorded. Outputs: electricity shed (kWh), the extra temperature rise, and the probability
   that the zone exceeds 26 C, from the twin's own uncertainty.

Results: results/flex_validation.csv, results/flex_scenarios.csv.

    python step6_flexibility.py [--zones F7z1]
"""
import argparse
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from twin import data as D
from twin.runner import load_model, segments, selected_models
from twin.twin import DigitalTwin

OUT = Path(__file__).parent / "results"
PERIOD = ("2019-07-01", "2019-12-31 23:55")
WARMUP = 3 * 288  # three days
CHECK = {3: "15min", 6: "30min", 12: "1h", 24: "2h"}
T_LIMIT = 26.0
SCEN = {"off 30 min": (6, 0.0), "off 60 min": (12, 0.0), "off 120 min": (24, 0.0), "half power 120 min": (24, 0.5)}
EVAL = 36         # steps evaluated after the start of a scenario (3 h, to include the recovery)


def settings(zone):
    s = pd.read_csv(OUT / "twin_settings.csv").set_index("zone")
    return float(s.loc[zone, "q_kappa"]), float(s.loc[zone, "q_dq"]), float(s.loc[zone, "dq_tau_h"])


def shutoff_events(P, Y, p_on=2.0, keep=0.2, off_steps=24):
    """Sharp, lasting cuts in AC power: the zone drew more than p_on kW shortly before, and from the
    first step below `keep` times that level it stays below for 2 h (some zones keep a few units on at
    night, so the cut need not be to zero)."""
    ev = []
    for k in range(WARMUP, len(P) - off_steps):
        before = P[k - 12:k - 3].mean()          # 45 to 15 min earlier: the 5-min means ramp down over a few steps
        if before > p_on and P[k - 1] >= keep * before and (P[k:k + off_steps] < keep * before).all() \
                and np.mean(~np.isnan(Y[k:k + off_steps + 1])) > 0.8:
            ev.append(k)
    return ev


def run_zone(zone):
    model = load_model(zone)
    qk, qd, tau = settings(zone)
    df = D.zone_frame(zone, *PERIOD)
    U, Y = D.arrays(df)
    ok = D.complete_inputs(df).to_numpy()
    idx = df.index
    val, scen = [], []
    for a, b in segments(ok):
        y0 = Y[a:b][~np.isnan(Y[a:b])]
        if not len(y0):
            continue
        tw = DigitalTwin(model, y0[0], q_kappa=qk, q_dq=qd, dq_tau_h=tau)
        ev = set(e + a for e in shutoff_events(U[a:b, 1], Y[a:b]))
        for k in range(a, b - EVAL):
            # the twin has assimilated everything up to k-1 and now predicts k onwards
            if k in ev:
                mean, std = tw.forecast(U[k:k + 24])        # mean[j] is the prediction for k+j+1
                y_start = Y[k - 1] if not np.isnan(Y[k - 1]) else tw.x[0]
                for j, lab in CHECK.items():
                    y = Y[k + j]
                    if np.isnan(y):
                        continue
                    val.append(dict(zone=zone, start=str(idx[k]), hour=idx[k].hour, after=lab,
                                    measured_rise=y - y_start, predicted_rise=mean[j - 1] - y_start,
                                    error=y - mean[j - 1], inside95=abs(y - mean[j - 1]) <= 1.96 * std[j - 1]))
            t = idx[k]
            # weekday 14:00 with units running
            if t.hour == 14 and t.minute == 0 and t.dayofweek < 5 and U[k:k + 24, 1].mean() > 1.0:
                base, base_sd = tw.forecast(U[k:k + EVAL])
                for name, (dur, frac) in SCEN.items():
                    Us = U[k:k + EVAL].copy()
                    Us[:dur, 1] *= frac
                    m, sd = tw.forecast(Us)
                    p_exceed = norm.sf(T_LIMIT, loc=m, scale=sd)  # Gaussian forecast assumed
                    scen.append(dict(zone=zone, day=str(t.date()), scenario=name,
                                     shed_kWh=float((U[k:k + dur, 1] - Us[:dur, 1]).sum() * D.DT_H),
                                     shed_kW_mean=float((U[k:k + dur, 1] - Us[:dur, 1]).mean()),
                                     T_start=float(tw.x[0]), max_extra_rise=float(np.max(m - base)),
                                     T_peak=float(np.max(m)), p_exceed_max=float(np.max(p_exceed)),
                                     base_p_exceed_max=float(np.max(norm.sf(T_LIMIT, loc=base, scale=base_sd)))))
            tw.step(Y[k], U[k])  # assimilate k only after both checks
    logging.info("%s: %d shut-off events, %d scenario days", zone, len({v['start'] for v in val}),
                 len({s['day'] for s in scen}))
    return val, scen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT / "step6.log"), logging.StreamHandler(sys.stdout)])
    zones = a.zones.split(",") if a.zones else sorted(selected_models())
    val, scen = [], []
    with Pool(a.workers) as pool:
        for v, s in pool.imap_unordered(run_zone, zones):
            val += v
            scen += s
    pd.DataFrame(val).to_csv(OUT / "flex_validation.csv", index=False)
    pd.DataFrame(scen).to_csv(OUT / "flex_scenarios.csv", index=False)


if __name__ == "__main__":
    main()
