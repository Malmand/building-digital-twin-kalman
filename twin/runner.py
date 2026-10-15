"""Running the twin over long records: model loading, segments with complete inputs, forecasts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .rc import RCModel
from .twin import DigitalTwin

RESULTS = Path(__file__).resolve().parents[1] / "results"


def selected_models():
    """Model chosen for each zone in step 2: lowest BIC among the identified ones."""
    ident = pd.read_csv(RESULTS / "identification.csv")
    ident = ident[ident["model"].isin(["M1", "M2", "M3"])]
    return ident.loc[ident.groupby("zone")["bic"].idxmin()].set_index("zone")["model"].to_dict()


def load_model(zone, key=None):
    params = json.loads((RESULTS / "params.json").read_text())
    key = key or selected_models()[zone]
    m = RCModel(key)
    m.p = params[zone][key]
    return m


def segments(ok, min_len=288):
    """Start/end indices of runs where the inputs are complete (at least one day long)."""
    ok = np.asarray(ok, bool)
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def run_twin(model, df, track=True, q_kappa=1e-6, q_dq=0.02, dq_tau_h=4.0, horizons=(12, 72), stride=6):
    """Run the twin over every complete-input segment of df. Returns a frame with the filtered
    outputs and, for each horizon h, the forecast made h steps earlier for this time (mean, std)."""
    U, Y = D.arrays(df)
    out = pd.DataFrame(index=df.index)
    cols = {f: np.full(len(df), np.nan) for f in ("Ti_f", "Ti_std", "kappa", "dq", "innov", "z")}
    fc = {h: (np.full(len(df), np.nan), np.full(len(df), np.nan)) for h in horizons}
    H = max(horizons)
    for a, b in segments(D.complete_inputs(df)):
        y0 = Y[a:b][~np.isnan(Y[a:b])]
        if not len(y0):
            continue
        tw = DigitalTwin(model, y0[0], q_kappa=q_kappa, q_dq=q_dq, dq_tau_h=dq_tau_h, track=track)
        for k in range(a, b):
            o = tw.step(Y[k], U[k])
            cols["Ti_f"][k], cols["Ti_std"][k], cols["kappa"][k] = o.Ti, o.Ti_std, o.kappa
            cols["dq"][k], cols["innov"][k], cols["z"][k] = o.dq, o.innov, o.z
            if (k - a) % stride == 0 and k + H < b:
                # information up to and including Y[k]; the state now predicts k+1, and the
                # planned inputs U[k+1:] carry the forecast further
                mean, std = tw.forecast(U[k + 1:k + H])
                mean = np.r_[tw.x[0], mean]
                std = np.r_[np.sqrt(tw.P[0, 0] + tw.r), std]
                for h in horizons:
                    fc[h][0][k + h], fc[h][1][k + h] = mean[h - 1], std[h - 1]
    for f, v in cols.items():
        out[f] = v
    for h in horizons:
        out[f"fc{h}"], out[f"fc{h}_std"] = fc[h]
    out["Ti"] = Y
    return out
