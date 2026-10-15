"""Step 4: physics, machine learning, or both?

Three forecasters of the zone temperature 1 h and 6 h ahead, all given the same information:
    twin     the Kalman-filter twin (fixed parameters), conditional on the planned AC power and weather
    hybrid   the twin plus a gradient-boosting model of its own forecast error
    ML only  gradient boosting predicting the temperature directly
Features available at the forecast origin: measured and filtered temperature, the twin's extra-gain
estimate, lighting and plug power now, the hour and weekday of the target, the planned AC power and
the weather at the target time. Training: April to mid-June 2019. Test: July to December 2019.
Results: results/hybrid.csv.

    python step4_hybrid.py [--zones F7z1]
"""
import argparse
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from twin import data as D
from twin.forecast import rmse
from twin.runner import load_model, run_twin, selected_models

OUT = Path(__file__).parent / "results"
TRAIN = ("2019-04-01", "2019-06-16 23:55")
TEST = ("2019-07-01", "2019-12-31 23:55")


def dataset(zone, model, period, h):
    df = D.zone_frame(zone, *period)
    o = run_twin(model, df, track=False, horizons=(h,), stride=3)
    tgt = o.index[o[f"fc{h}"].notna() & o["Ti"].notna()]
    pos = df.index.get_indexer(tgt)
    org = pos - h
    X = pd.DataFrame({
        "Ti_now": df["Ti"].to_numpy()[org],
        "Ti_filt_now": o["Ti_f"].to_numpy()[org],
        "P_int_now": df["P_int"].to_numpy()[org],
        "P_ac_now": df["P_ac"].to_numpy()[org],
        "P_ac_target": df["P_ac"].to_numpy()[pos],
        "P_ac_mean_horizon": np.array([df["P_ac"].to_numpy()[a + 1:b + 1].mean() for a, b in zip(org, pos)]),
        "T_out_target": df["T_out"].to_numpy()[pos],
        "G_target": df["G"].to_numpy()[pos],
        "hour_target": tgt.hour + tgt.minute / 60,
        "weekday_target": tgt.dayofweek,
        "twin_forecast": o[f"fc{h}"].to_numpy()[pos],
    }, index=tgt)
    y = df["Ti"].to_numpy()[pos]
    return X, y


def run_zone(zone):
    model = load_model(zone)
    rows = []
    for h, lab in ((12, "1h"), (72, "6h")):
        Xtr, ytr = dataset(zone, model, TRAIN, h)
        Xte, yte = dataset(zone, model, TEST, h)
        if len(Xtr) < 500 or len(Xte) < 500:
            continue
        feats_ml = [c for c in Xtr.columns if c != "twin_forecast" and c != "Ti_filt_now"]
        gb = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0, random_state=0)
        res = HistGradientBoostingRegressor(**gb).fit(Xtr, ytr - Xtr["twin_forecast"])
        hyb = Xte["twin_forecast"] + res.predict(Xte)
        ml = HistGradientBoostingRegressor(**gb).fit(Xtr[feats_ml], ytr).predict(Xte[feats_ml])
        persist = Xte["Ti_now"].to_numpy()
        for name, pred in (("persistence", persist), ("twin", Xte["twin_forecast"].to_numpy()),
                           ("hybrid", hyb.to_numpy()), ("ML only", ml)):
            ok = ~np.isnan(pred)
            rows.append(dict(zone=zone, horizon=lab, method=name, n_train=len(Xtr), n_test=int(ok.sum()),
                             rmse=rmse(yte[ok] - pred[ok])))
    logging.info("%s: %s", zone, "; ".join(f"{r['horizon']} {r['method']} {r['rmse']:.2f}" for r in rows))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT / "step4.log"), logging.StreamHandler(sys.stdout)])
    zones = a.zones.split(",") if a.zones else sorted(selected_models())
    rows = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(run_zone, zones):
            rows += r
    pd.DataFrame(rows).to_csv(OUT / "hybrid.csv", index=False)


if __name__ == "__main__":
    main()
