"""Step 3: the twin in operation, July to December 2019.

Each zone's model (selected in step 2) is run as a digital twin over six months it was not fitted on,
in two versions:
    fixed     Kalman filter with the identified parameters only
    tracking  extended Kalman filter that also tracks the cooling effectiveness (kappa, a slow random
              walk) and an extra heat gain (dq, mean-reverting)
The noise intensities and the time constant of dq are chosen by maximum likelihood on the training
fortnight (a grid search). Outputs:
results/twin_monthly.csv (forecast error by month, horizon and version), results/twin_daily.csv
(daily kappa and dq), results/twin_settings.csv.

    python step3_twin.py [--zones F7z1]
"""
import argparse
import itertools
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from twin import data as D
from twin.forecast import rmse
from twin.runner import load_model, run_twin, segments, selected_models
from twin.twin import DigitalTwin

OUT = Path(__file__).parent / "results"
TRAIN = ("2019-06-03", "2019-06-16 23:55")
PERIOD = ("2019-07-01", "2019-12-31 23:55")
GRID_KAPPA = (1e-7, 1e-6, 1e-5)
GRID_DQ = (0.002, 0.02, 0.2)
GRID_TAU = (1.0, 4.0, 24.0, np.inf)       # h; inf = random walk


def loglik(model, df, q_kappa, q_dq, tau):
    U, Y = D.arrays(df)
    ll = 0.0
    for a, b in segments(D.complete_inputs(df)):
        y0 = Y[a:b][~np.isnan(Y[a:b])]
        tw = DigitalTwin(model, y0[0], q_kappa=q_kappa, q_dq=q_dq, dq_tau_h=tau)
        o = tw.run(U[a:b], Y[a:b])
        e, z = o["innov"], o["z"]
        ok = ~np.isnan(e) & (np.abs(z) > 0)
        S = (e[ok] / z[ok]) ** 2
        ll += -0.5 * np.sum(np.log(2 * np.pi * S) + z[ok] ** 2)
    return ll


def run_zone(zone):
    model = load_model(zone)
    tr = D.zone_frame(zone, *TRAIN)
    grid = {g: loglik(model, tr, *g) for g in itertools.product(GRID_KAPPA, GRID_DQ, GRID_TAU)}
    qk, qd, tau = max(grid, key=grid.get)
    df = D.zone_frame(zone, *PERIOD)
    rows, daily = [], []
    for version, track in (("fixed", False), ("tracking", True)):
        o = run_twin(model, df, track=track, q_kappa=qk, q_dq=qd, dq_tau_h=tau)
        month = o.index.to_period("M")
        for h, lab in ((12, "1h"), (72, "6h")):
            err = o["Ti"] - o[f"fc{h}"]
            inside = (err.abs() <= 1.96 * o[f"fc{h}_std"]).where(err.notna())
            for mth, e in err.groupby(month):
                e = e.dropna()
                rows.append(dict(zone=zone, model=model.key, version=version, horizon=lab, month=str(mth),
                                 n=len(e), rmse=rmse(e.to_numpy()), bias=float(e.mean()) if len(e) else np.nan,
                                 coverage95=float(inside[month == mth].mean())))
        if track:
            d = o[["kappa", "dq"]].resample("D").mean()
            d["zone"] = zone
            daily.append(d)
    logging.info("%s (%s): q_kappa %.0e, q_dq %.3f, tau %.0f h", zone, model.key, qk, qd, tau)
    return zone, rows, daily, dict(zone=zone, model=model.key, q_kappa=qk, q_dq=qd, dq_tau_h=tau)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT / "step3.log"), logging.StreamHandler(sys.stdout)])
    zones = a.zones.split(",") if a.zones else sorted(selected_models())
    rows, daily, settings = [], [], []
    with Pool(a.workers) as pool:
        for zone, r, d, s in pool.imap_unordered(run_zone, zones):
            rows += r
            daily += d
            settings.append(s)
    pd.DataFrame(rows).to_csv(OUT / "twin_monthly.csv", index=False)
    pd.concat(daily).to_csv(OUT / "twin_daily.csv")
    pd.DataFrame(settings).to_csv(OUT / "twin_settings.csv", index=False)


if __name__ == "__main__":
    main()
