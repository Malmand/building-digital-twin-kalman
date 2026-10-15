"""Kalman filters for the building twin.

All filters work on discrete linear(ised) state-space models

    x[k+1] = A x[k] + B u[k] + w[k],   w ~ N(0, Q)
    y[k]   = C x[k] + v[k],            v ~ N(0, R)

and accept missing measurements (NaN in y): the update step is then skipped and the state is only
propagated, which is how the twin keeps running through sensor drop-outs.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class FilterResult:
    x_pred: np.ndarray       # (n, nx) one-step-ahead predicted states
    P_pred: np.ndarray       # (n, nx, nx)
    x_filt: np.ndarray       # (n, nx) filtered states
    P_filt: np.ndarray
    innov: np.ndarray        # (n,) innovations y - C x_pred (NaN where y is missing)
    S: np.ndarray            # (n,) innovation variances
    loglik: float            # Gaussian log-likelihood of the observed innovations

    @property
    def normalised_innov(self):
        return self.innov / np.sqrt(self.S)


class KalmanFilter:
    """Linear Kalman filter for a single measured output (the zone air temperature)."""

    def __init__(self, A, B, C, Q, R, x0, P0):
        self.A, self.B = np.asarray(A, float), np.asarray(B, float)
        self.C = np.asarray(C, float).reshape(1, -1)
        self.Q, self.R = np.asarray(Q, float), float(R)
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()

    def predict(self, u):
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x, self.P

    def update(self, y):
        """Returns (innovation, innovation variance); does nothing but report if y is NaN."""
        S = (self.C @ self.P @ self.C.T).item() + self.R
        if np.isnan(y):
            return np.nan, S
        e = y - (self.C @ self.x).item()
        K = (self.P @ self.C.T).ravel() / S
        self.x = self.x + K * e
        # Joseph form: keeps P symmetric and positive definite over long runs
        I_KC = np.eye(len(self.x)) - np.outer(K, self.C)
        self.P = I_KC @ self.P @ I_KC.T + self.R * np.outer(K, K)
        return e, S

    def run(self, U, Y):
        """Filter a whole record. U: (n, nu) inputs applied from step k to k+1; Y: (n,) measurements.
        Convention: at step k the filter first updates with Y[k], then predicts k+1 with U[k]."""
        n, nx = len(Y), len(self.x)
        xp, Pp = np.empty((n, nx)), np.empty((n, nx, nx))
        xf, Pf = np.empty((n, nx)), np.empty((n, nx, nx))
        innov, S = np.full(n, np.nan), np.empty(n)
        ll = 0.0
        for k in range(n):
            xp[k], Pp[k] = self.x, self.P
            innov[k], S[k] = self.update(Y[k])
            if not np.isnan(innov[k]):
                ll -= 0.5 * (np.log(2 * np.pi * S[k]) + innov[k] ** 2 / S[k])
            xf[k], Pf[k] = self.x, self.P
            self.predict(U[k])
        return FilterResult(xp, Pp, xf, Pf, innov, S, ll)


def rts_smoother(res: FilterResult, A):
    """Rauch-Tung-Striebel smoother: best estimate of the hidden temperatures using the whole record
    (offline analysis; the live twin uses the filter)."""
    n = len(res.x_filt)
    xs, Ps = res.x_filt.copy(), res.P_filt.copy()
    for k in range(n - 2, -1, -1):
        G = res.P_filt[k] @ A.T @ np.linalg.inv(res.P_pred[k + 1])
        xs[k] = res.x_filt[k] + G @ (xs[k + 1] - res.x_pred[k + 1])
        Ps[k] = res.P_filt[k] + G @ (Ps[k + 1] - res.P_pred[k + 1]) @ G.T
    return xs, Ps


class ExtendedKalmanFilter:
    """EKF for a state augmented with slowly drifting parameters (random walks).

    f(x, u) -> next state, F(x, u) -> its Jacobian; the measurement is linear, y = C x + v.
    Used to track the effective cooling capacity of the air-conditioning units online.
    """

    def __init__(self, f, F, C, Q, R, x0, P0):
        self.f, self.F = f, F
        self.C = np.asarray(C, float).reshape(1, -1)
        self.Q, self.R = np.asarray(Q, float), float(R)
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()

    def predict(self, u):
        J = self.F(self.x, u)
        self.x = self.f(self.x, u)
        self.P = J @ self.P @ J.T + self.Q

    update = KalmanFilter.update

    def run(self, U, Y):
        n, nx = len(Y), len(self.x)
        xf, Pf = np.empty((n, nx)), np.empty((n, nx, nx))
        innov, S = np.full(n, np.nan), np.empty(n)
        for k in range(n):
            innov[k], S[k] = self.update(Y[k])
            xf[k], Pf[k] = self.x, self.P
            self.predict(U[k])
        return xf, Pf, innov, S
