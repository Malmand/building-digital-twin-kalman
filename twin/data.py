"""Loading and cleaning the CU-BEMS zone data and the matching weather.

A zone is identified as "F<floor>z<zone>", e.g. "F7z1". For each zone the twin uses::

    Ti      measured air temperature (degC), sensor S1
    RH      relative humidity (%), lux: illuminance, both for diagnostics only
    P_ac    total electrical power of the zone's air-conditioning units (kW)
    P_int   lighting + plug power (kW), the measurable part of the internal gains

and the outdoor conditions T_out (degC) and G (global horizontal irradiance, kW/m2) from ERA5.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
DT_MIN = 5
DT_H = DT_MIN / 60


@lru_cache(maxsize=1)
def raw_table():
    return pd.read_csv(DATA / "cubems_5min.csv.gz", index_col=0, parse_dates=True)


@lru_cache(maxsize=1)
def weather():
    """ERA5 (Open-Meteo archive, local time). Radiation values are means over the preceding hour,
    so they are placed at the middle of that hour before interpolating to 5 min."""
    w = pd.read_csv(DATA / "weather_era5.csv.gz", parse_dates=["time"]).set_index("time")
    idx = pd.date_range(w.index.min(), w.index.max(), freq=f"{DT_MIN}min")
    T = w["temperature_2m"].reindex(idx).interpolate(limit_area="inside")
    rad = w[["shortwave_radiation", "direct_radiation", "diffuse_radiation"]].copy()
    rad.index = rad.index - pd.Timedelta("30min")  # hourly means -> mid-hour
    rad = rad.reindex(rad.index.union(idx)).interpolate(limit_area="inside").reindex(idx).clip(lower=0)
    rh = w["relative_humidity_2m"].reindex(idx).interpolate(limit_area="inside")
    return pd.DataFrame({"T_out": T, "G": rad["shortwave_radiation"] / 1000, "G_dir": rad["direct_radiation"] / 1000,
                         "G_dif": rad["diffuse_radiation"] / 1000, "RH_out": rh})


def zones():
    """Zones that have both air-conditioning and a temperature sensor."""
    cols = raw_table().columns
    out = []
    for c in cols:
        if c.endswith("_S1(degC)"):
            floor, z = c.split("_")[:2]
            if any(k.startswith(f"{floor}_{z}_AC") for k in cols):
                out.append(f"F{floor[5:]}{z}")  # 'Floor7', 'z1' -> 'F7z1'
    return out


def _prefix(zone):
    f, z = zone[1:].split("z")
    return f"Floor{f}_z{z}"


def spikes(x, jump=1.0):
    """Isolated one-sample spikes: the value jumps by more than `jump` K from both neighbours, in
    opposite directions. A step (the air-conditioning switching on, which cools a zone by several K
    within 10 min) is kept, because the temperature does not come back."""
    d_prev = x - x.shift(1)
    d_next = x - x.shift(-1)
    # up then down (or down then up)
    bad = (d_prev.abs() > jump) & (d_next.abs() > jump) & (np.sign(d_prev) == np.sign(d_next))
    return x.mask(bad), int(bad.sum())


def frozen(x, min_steps=24):
    """Flag runs of identical values lasting at least min_steps (2 h at 5 min): a stuck sensor."""
    same = x.diff().eq(0)
    grp = (~same).cumsum()  # one group per run of equal values
    run = same.groupby(grp).transform("sum") + 1
    return run >= min_steps


def zone_frame(zone, start=None, end=None, clean=True, report=None):
    """Tidy 5-min frame for one zone, with outdoor conditions; cleaning steps are counted in report."""
    t = raw_table()
    p = _prefix(zone)
    ac = [c for c in t.columns if c.startswith(f"{p}_AC")]
    gains = [c for c in (f"{p}_Light(kW)", f"{p}_Plug(kW)") if c in t]
    df = pd.DataFrame({
        "Ti": t[f"{p}_S1(degC)"],
        "RH": t[f"{p}_S1(RH%)"],
        "lux": t[f"{p}_S1(lux)"],
        "P_ac": t[ac].sum(axis=1, min_count=len(ac)),  # NaN if any unit missing
        "P_int": t[gains].sum(axis=1, min_count=len(gains)),
    })
    df = df.join(weather(), how="left")
    if start or end:
        df = df.loc[start:end]
    if clean:
        rep = report if report is not None else {}
        rep["rows"] = len(df)
        rep["Ti_missing"] = int(df["Ti"].isna().sum())
        bad = (df["Ti"] < 12) | (df["Ti"] > 40)  # physically implausible indoors
        rep["Ti_out_of_range"] = int(bad.sum())
        df.loc[bad, "Ti"] = np.nan
        fr = frozen(df["Ti"])
        rep["Ti_frozen"] = int(fr.sum())
        df.loc[fr, "Ti"] = np.nan
        df["Ti"], rep["Ti_spikes"] = spikes(df["Ti"])
        for c in ("P_ac", "P_int"):
            neg = df[c] < 0
            rep[f"{c}_negative"] = int(neg.sum())
            df.loc[neg, c] = np.nan
            rep[f"{c}_missing"] = int(df[c].isna().sum())
            df[c] = df[c].interpolate(limit=6, limit_area="inside")     # bridge gaps up to 30 min
    df["one"] = 1.0
    return df


def complete_inputs(df, cols=("T_out", "P_ac", "P_int", "G")):
    return df[list(cols)].notna().all(axis=1)


def arrays(df):
    """U (n, 5) in rc.INPUTS order and Y (n,) for the filter."""
    U = df[["T_out", "P_ac", "P_int", "G", "one"]].to_numpy(float)
    Y = df["Ti"].to_numpy(float)
    return np.ascontiguousarray(U), np.ascontiguousarray(Y)
