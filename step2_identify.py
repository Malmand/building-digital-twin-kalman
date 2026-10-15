"""Step 2: identify grey-box models of every air-conditioned zone at three levels of detail.

For each zone, models M1 (1R1C), M2 (2R2C) and M3 (4R3C) are estimated by maximum likelihood on two
weeks of June 2019 and tested on the next two weeks, which the fit never sees. Model selection uses
likelihood-ratio tests, AIC and BIC on the training data, and forecast errors 5 min to 6 h ahead on
the test data. Results: results/identification.csv, results/params.json (resumable).

    python step2_identify.py [--zones F7z1,F4z2] [--workers 2]
"""
import argparse
import json
import logging
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from twin import data as D
from twin.forecast import acf, filter_zone, kstep_errors, rmse
from twin.rc import RCModel

TRAIN = ("2019-06-03", "2019-06-16 23:55")
TEST = ("2019-06-17", "2019-06-30 23:55")
HORIZONS = {1: "5min", 12: "1h", 36: "3h", 72: "6h"}
OUT = Path(__file__).parent / "results"
MIN_COVERAGE = 0.7  # share of Ti readings required


def run_zone(zone):
    t0 = time.time()
    tr, te = D.zone_frame(zone, *TRAIN), D.zone_frame(zone, *TEST)
    cov_tr, cov_te = tr["Ti"].notna().mean(), te["Ti"].notna().mean()
    if cov_tr < MIN_COVERAGE or cov_te < MIN_COVERAGE or not D.complete_inputs(tr).all() or not D.complete_inputs(te).all():
        return zone, [dict(zone=zone, model="-", skipped=f"coverage train {cov_tr:.2f}, test {cov_te:.2f}")], {}
    Utr, Ytr = D.arrays(tr)
    Ute, Yte = D.arrays(te)
    rows, params = [], {}
    for key in ("M1", "M2", "M3"):
        m = RCModel(key).fit(Utr, Ytr, starts=4)
        res = filter_zone(m, Ute, Yte)
        errs = kstep_errors(m, Ute, Yte, horizons=tuple(HORIZONS), res=res)
        r = acf(res.normalised_innov, 12)
        rows.append(dict(zone=zone, model=key, n_params=m.n_params, nll=m.nll, aic=m.aic, bic=m.bic,
                         coverage_train=round(cov_tr, 3), coverage_test=round(cov_te, 3),
                         **{f"rmse_{v}": rmse(errs[h]) for h, v in HORIZONS.items()},
                         acf_max_lag1_12=float(np.max(np.abs(r[1:]))),
                         tau_h=";".join(f"{t:.2f}" for t in m.time_constants_h()),
                         at_bound=";".join(m.at_bounds())))
        params[key] = m.p
    for a, b in (("M1", "M2"), ("M2", "M3")):
        ra = next(r for r in rows if r["model"] == a)
        rb = next(r for r in rows if r["model"] == b)
        stat = 2 * (ra["nll"] - rb["nll"])  # likelihood-ratio statistic, nested models
        rb[f"lr_p_vs_{a}"] = float(chi2.sf(stat, rb["n_params"] - ra["n_params"]))
    logging.info("%s done in %.0f s: 6 h RMSE %s", zone, time.time() - t0,
                 ", ".join(f"{r['model']} {r['rmse_6h']:.2f}" for r in rows))
    return zone, rows, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zones")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(OUT / "step2.log"), logging.StreamHandler(sys.stdout)])
    csv, js = OUT / "identification.csv", OUT / "params.json"
    done = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    params = json.loads(js.read_text()) if js.exists() else {}
    todo = [z for z in (a.zones.split(",") if a.zones else D.zones()) if done.empty or z not in set(done["zone"])]
    logging.info("%d zones to identify", len(todo))
    with Pool(a.workers) as pool:
        for zone, rows, p in pool.imap_unordered(run_zone, todo):
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(csv, index=False)
            if p:
                params[zone] = p
                js.write_text(json.dumps(params, indent=1))


if __name__ == "__main__":
    main()
