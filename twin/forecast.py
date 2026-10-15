"""Forecast evaluation: filter with the measurements up to time k, then predict k+h with the known
inputs only (what the twin would do in operation, with weather and planned set-points as inputs)."""
import numpy as np

from .kalman import KalmanFilter


def filter_zone(model, U, Y):
    Ad, Bd, Qd, r = model.matrices()
    nx = Ad.shape[0]
    y0 = Y[~np.isnan(Y)][0]
    kf = KalmanFilter(Ad, Bd, np.eye(nx)[0], Qd, r, np.full(nx, y0), np.eye(nx))
    return kf.run(U, Y)


def kstep_errors(model, U, Y, horizons=(1, 12, 36, 72), stride=3, res=None):
    """Errors of h-step-ahead forecasts started from the filtered state every `stride` steps.
    Returns {h: array of errors (measured - predicted)}."""
    Ad, Bd, _, _ = model.matrices()
    res = res or filter_zone(model, U, Y)
    H = max(horizons)
    out = {h: [] for h in horizons}
    for k in range(0, len(Y) - H, stride):
        x = res.x_filt[k].copy()
        for j in range(1, H + 1):
            x = Ad @ x + Bd @ U[k + j - 1]
            if j in out and not np.isnan(Y[k + j]):
                out[j].append(Y[k + j] - x[0])
    return {h: np.array(v) for h, v in out.items()}


def rmse(e):
    return float(np.sqrt(np.mean(np.square(e)))) if len(e) else np.nan


def acf(x, lags=36):
    x = x[~np.isnan(x)]
    x = x - x.mean()
    d = np.dot(x, x)
    return np.array([np.dot(x[:-k or None], x[k:]) / d for k in range(lags + 1)])
