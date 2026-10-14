"""Condense the CU-BEMS building data and fetch matching Bangkok weather (run by the GitHub Actions job
in .github/workflows/fetch_data.yml; it can also be run locally).

Inputs (downloaded by the workflow into raw/):
  CU-BEMS, Chulalongkorn University, Bangkok: 14 CSV files, one per floor and year, 1-min data,
  1 Jul 2018 - 31 Dec 2019 (Pipattanasomporn et al., Scientific Data 7, 241, 2020; figshare
  10.6084/m9.figshare.11726517, CC BY 4.0).

Outputs (data/):
  cubems_5min.csv.gz      every numeric column averaged to 5 min (rounded to sensor resolution)
  cubems_quality.json     per column, from the 1-min data: share missing, longest frozen run, range
  weather_era5.csv.gz     hourly ERA5 reanalysis at the building (Open-Meteo archive, CC BY 4.0):
                          temperature, humidity, global/direct/diffuse irradiance, cloud cover
  weather_isd.csv.gz      hourly measured temperature at Bangkok stations (NOAA ISD, public domain)

    python tools/prepare_data.py raw data
"""
import gzip
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

MD5 = {  # figshare record 11726517
    "2018Floor1.csv": "3ae4902c359a74167963dff758406487", "2018Floor2.csv": "c0fb626a0149c29fd00a0d0d06e3472d",
    "2018Floor3.csv": "e1f2385af5dabefd63b1f98caf44fe18", "2018Floor4.csv": "1397b807889337451bc0deba79dd5333",
    "2018Floor5.csv": "b394e07b6f9245becc0dbc34544334c8", "2018Floor6.csv": "e4074802fd7973a87e620a9ea1a381b6",
    "2018Floor7.csv": "9ff3859e4b0fd74e142b81ddfaac50a9", "2019Floor1.csv": "e51d1cbf5642c53656e6a4e6d997ff18",
    "2019Floor2.csv": "2fa6b68e788afaf55f4395a6b918092b", "2019Floor3.csv": "f34d43850bcb042355495b0275b5428c",
    "2019Floor4.csv": "4b9d738c05555edcaadd9173c7b2e1cc", "2019Floor5.csv": "2422f2d7864ac6889c2ad2d68a086a28",
    "2019Floor6.csv": "4f2446b38b566f2c1a07bf0e44587b42",
    "2019Floor7.csv": "05b7920ed3dc3c478fbc172571ef8cca",
}
LAT, LON = 13.7384, 100.5327          # Chulalongkorn University, Faculty of Engineering
ISD_STATIONS = {"48455099999": "Bangkok Metropolis", "48456099999": "Don Mueang"}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decimals(col):
    c = col.lower()
    if "degc" in c:
        return 2
    if "rh" in c:
        return 1
    if "lux" in c:
        return 0
    return 3                                           # kW


def longest_constant_run(x):
    """Longest run (in samples) of identical non-missing values: a frozen sensor."""
    v = x.to_numpy()
    ok = ~np.isnan(v)
    same = np.r_[False, (v[1:] == v[:-1]) & ok[1:] & ok[:-1]]
    if not same.any():
        return 1
    idx = np.flatnonzero(~same)
    runs = np.diff(np.r_[idx, len(v)])
    return int(runs.max())


def condense(raw, out):
    frames, quality = [], {}
    for name in sorted(MD5):
        path = raw / name
        got = md5(path)
        if got != MD5[name]:
            raise SystemExit(f"{name}: md5 {got} does not match the figshare record")
        df = pd.read_csv(path, low_memory=False)
        tcol = df.columns[0]
        df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
        df = df.dropna(subset=[tcol]).set_index(tcol).sort_index()
        df = df[~df.index.duplicated(keep="first")]
        floor = name[4:].replace(".csv", "")              # Floor1 ... Floor7
        df = df.apply(pd.to_numeric, errors="coerce")
        df.columns = [f"{floor}_{c}" for c in df.columns]
        full = pd.date_range(df.index.min().floor("D"), df.index.max().ceil("D") - pd.Timedelta("1min"), freq="1min")
        df = df.reindex(full)
        for c in df.columns:
            s = df[c]
            q = quality.setdefault(c, {"rows": 0, "missing": 0, "longest_constant_min": 0, "min": np.inf, "max": -np.inf})
            q["rows"] += int(len(s))
            q["missing"] += int(s.isna().sum())
            q["longest_constant_min"] = max(q["longest_constant_min"], longest_constant_run(s))
            if s.notna().any():
                q["min"] = float(min(q["min"], s.min()))
                q["max"] = float(max(q["max"], s.max()))
        frames.append(df.resample("5min").mean())
        print(f"{name}: {len(df)} rows, {df.shape[1]} columns, {df.index.min()} to {df.index.max()}", flush=True)
    by_floor = {}
    for f in frames:
        key = f.columns[0].split("_")[0]
        by_floor.setdefault(key, []).append(f)
    wide = pd.concat([pd.concat(v).sort_index() for v in by_floor.values()], axis=1).sort_index()
    wide = wide[~wide.index.duplicated(keep="first")]
    for c in wide.columns:
        wide[c] = wide[c].round(decimals(c))
    wide.index.name = "time"
    wide.to_csv(out / "cubems_5min.csv.gz", compression={"method": "gzip", "compresslevel": 9})
    for q in quality.values():
        q["share_missing"] = round(q.pop("missing") / max(q["rows"], 1), 4)
        for k in ("min", "max"):
            q[k] = None if not np.isfinite(q[k]) else round(q[k], 3)
    (out / "cubems_quality.json").write_text(json.dumps(quality, indent=1))
    print(f"5-min table: {wide.shape}, {wide.index.min()} to {wide.index.max()}", flush=True)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "building-digital-twin research script"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def weather(out):
    q = ("https://archive-api.open-meteo.com/v1/archive?"
         f"latitude={LAT}&longitude={LON}&start_date=2018-06-30&end_date=2020-01-01"
         "&hourly=temperature_2m,relative_humidity_2m,shortwave_radiation,direct_radiation,"
         "diffuse_radiation,cloud_cover,wind_speed_10m&timezone=Asia%2FBangkok")
    js = json.loads(fetch(q))
    w = pd.DataFrame(js["hourly"]).rename(columns={"time": "time"})
    w["time"] = pd.to_datetime(w["time"])
    w.to_csv(out / "weather_era5.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 9})
    print(f"ERA5: {len(w)} hours", flush=True)

    rows = []
    for sid, sname in ISD_STATIONS.items():
        for year in (2018, 2019):
            try:
                txt = fetch(f"https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{sid}.csv").decode()
            except Exception as e:                    # noqa: BLE001
                print(f"ISD {sid} {year}: {e!r}", flush=True)
                continue
            d = pd.read_csv(io.StringIO(txt), usecols=["DATE", "TMP", "DEW"], dtype=str)
            t = d["TMP"].str.split(",", expand=True)
            ok = t[1].isin(["1", "5", "A", "C", "I", "M", "P", "R", "U"]) & (t[0] != "+9999")
            d = pd.DataFrame({"time_utc": pd.to_datetime(d["DATE"]), "T_air": t[0].astype(float) / 10})[ok]
            d["station"] = sname
            rows.append(d)
    if rows:
        isd = pd.concat(rows)
        isd["time"] = isd["time_utc"] + pd.Timedelta(hours=7)          # Bangkok, no daylight saving
        isd = isd.drop(columns="time_utc").sort_values(["station", "time"])
        isd.to_csv(out / "weather_isd.csv.gz", index=False, compression={"method": "gzip", "compresslevel": 9})
        print(f"ISD: {len(isd)} observations", flush=True)


if __name__ == "__main__":
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    condense(raw, out)
    try:
        weather(out)
    except Exception as e:                            # noqa: BLE001  building data are kept even if this fails
        print(f"weather download failed: {e!r}", flush=True)
