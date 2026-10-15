"""Step 1: what the data contain and how they are cleaned.

Writes results/data_quality.csv (one row per zone: availability and every cleaning action over the
period used, April to December 2019) and results/data_checks.json (time alignment of the weather and
basic plausibility). Figures are made by make_figures.py.

    python step1_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from twin import data as D

OUT = Path(__file__).parent / "results"
PERIOD = ("2019-04-01", "2019-12-31 23:55")


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for z in D.zones():
        rep = {}
        df = D.zone_frame(z, *PERIOD, report=rep)
        occ = (df.index.hour >= 8) & (df.index.hour < 18) & (df.index.dayofweek < 5)
        rows.append(dict(zone=z, **rep, Ti_available=round(df["Ti"].notna().mean(), 3),
                         Ti_mean_occupied=round(df.loc[occ, "Ti"].mean(), 2),
                         P_ac_mean_occupied_kW=round(df.loc[occ, "P_ac"].mean(), 2),
                         P_ac_max_kW=round(df["P_ac"].max(), 1),
                         P_int_mean_occupied_kW=round(df.loc[occ, "P_int"].mean(), 2)))
    q = pd.DataFrame(rows)
    q.to_csv(OUT / "data_quality.csv", index=False)

    # time alignment: the 5-min ERA5 irradiance against a clear-sky model at the site (local time)
    w = D.weather().loc[PERIOD[0]:PERIOD[1]]
    cs = pvlib.location.Location(13.7384, 100.5327, tz="Asia/Bangkok").get_clearsky(
        w.index.tz_localize("Asia/Bangkok"))["ghi"].to_numpy() / 1000
    lags = range(-12, 13)
    corr = [np.corrcoef(np.roll(w["G"].to_numpy(), k)[24:-24], cs[24:-24])[0, 1] for k in lags]
    best = list(lags)[int(np.argmax(corr))] * D.DT_MIN
    checks = dict(
        irradiance_best_shift_min=best,
        irradiance_corr_at_best=round(float(max(corr)), 4),
        T_out_range=[float(w["T_out"].min()), float(w["T_out"].max())],
        zones=len(q), period=list(PERIOD),
        Ti_available_median=float(q["Ti_available"].median()),
        cleaning_totals={k: int(q[k].sum()) for k in q.columns if k.startswith(("Ti_", "P_")) and q[k].dtype.kind == "i"},
    )
    assert abs(best) <= 15, f"weather looks shifted by {best} min"
    (OUT / "data_checks.json").write_text(json.dumps(checks, indent=1))
    print(q.to_string())
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
