"""Fault detection from the twin's outputs.

Three detectors, each aimed at one kind of fault:

    cusum     two-sided CUSUM on standardised 3-hour-ahead forecast residuals of the twin with fixed
              parameters: a sensor that jumps or drifts, or any lasting change the model cannot explain.
              Multi-step residuals are used because a filter corrects itself towards the measurements
              within a few steps, so one-step innovations hide a constant sensor offset; and the
              fixed twin is used because the tracking twin would absorb the offset as an extra gain
    frozen    the measured temperature repeats the same value for an hour while the twin expects it
              to move: a stuck sensor or a stalled logger
    cooling   the tracked cooling effectiveness kappa stays below a limit for a day: the units draw
              power but remove less heat than identified

Thresholds are set on fault-free data (false alarms per month) and then tested on injected faults.
"""
from dataclasses import dataclass

import numpy as np

from .twin import DigitalTwin


@dataclass
class Alarm:
    k: int          # sample index
    kind: str
    detail: str = ""


def hstep_residuals(model, U, Y, h=36, stride=3, free_floating=False, p_off=0.1):
    """Standardised residuals y[k+h] - forecast made at k, every `stride` steps (NaN elsewhere),
    from a twin with fixed parameters. With free_floating, only forecasts over which the
    air-conditioning stays off count (not used in the study: several zones run units at night)."""
    y0 = Y[~np.isnan(Y)][0]
    tw = DigitalTwin(model, y0, track=False)  # fixed twin: an offset cannot be absorbed
    z = np.full(len(Y), np.nan)
    for k in range(len(Y)):
        tw.step(Y[k], U[k])
        if k % stride == 0 and k + h < len(Y):
            mean, std = tw.forecast(U[k + 1:k + h])  # mean[-1] is the forecast for k+h
            if not np.isnan(Y[k + h]) and (not free_floating or (U[k:k + h + 1, 1] < p_off).all()):
                z[k + h] = (Y[k + h] - mean[-1]) / std[-1]
    return z


def cusum(z, drift=0.5, threshold=10.0, reset=True):
    """Two-sided CUSUM; z is a sequence of normalised innovations (NaN skipped). Returns alarm indices
    and the statistics. After an alarm the sums restart, so a persisting fault alarms again later."""
    gp = gn = 0.0
    alarms, sp, sn = [], np.zeros(len(z)), np.zeros(len(z))
    for k, v in enumerate(z):
        if not np.isnan(v):
            gp = max(0.0, gp + v - drift)  # upward shift
            gn = max(0.0, gn - v - drift)  # downward shift
            if gp > threshold or gn > threshold:
                alarms.append(k)
                if reset:
                    gp = gn = 0.0
        sp[k], sn[k] = gp, gn
    return alarms, sp, sn


def frozen(y, steps=12, min_expected_change=0.3, forecast=None):
    """Indices where y has repeated exactly for `steps` samples. If a forecast is given, only runs
    over which the twin expected a change of at least min_expected_change K count."""
    alarms, run = [], 1
    for k in range(1, len(y)):
        if not np.isnan(y[k]) and y[k] == y[k - 1]:
            run += 1
        else:
            run = 1
        if run == steps:
            if forecast is None or np.ptp(forecast[k - steps + 1:k + 1]) >= min_expected_change:
                alarms.append(k)
    return alarms


def cooling(kappa, P_ac, limit=0.8, window=288, min_on_steps=24, p_on=0.5):
    """Indices (checked hourly) where, over the last `window` samples (one day), the cooling
    effectiveness averaged over the samples with the air-conditioning on is below `limit`; at least
    `min_on_steps` (2 h) of operation are needed for a verdict."""
    on = P_ac > p_on
    alarms = []
    for k in range(window, len(kappa), 12):  # check every hour
        sl = slice(k - window, k)
        if on[sl].sum() >= min_on_steps and np.mean(kappa[sl][on[sl]]) < limit:
            alarms.append(k)
    return alarms


def first_after(alarms, k0, k1):
    """First alarm in [k0, k1), or None."""
    for a in alarms:
        if k0 <= a < k1:
            return a
    return None
