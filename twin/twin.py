"""The online digital twin of one zone.

Each 5-min step the twin receives the new measurements and inputs, corrects its state with the Kalman
filter, and moves forward. Besides the air temperature it estimates two hidden quantities, as random
walks in an extended Kalman filter:

    kappa  cooling effectiveness: 1 when the air-conditioning cools as identified, lower when it
           removes less heat per kW (dirty filters, refrigerant loss)
    dq     extra heat gain in the zone (kW, in model units) not explained by the inputs: occupants,
           doors left open, a neighbouring zone. Modelled as mean-reverting (time constant dq_tau_h),
           so a forecast lets a disturbance fade instead of carrying it on for ever

It also provides forecasts with uncertainty for any planned input sequence (a demand-response event,
a set-point change), starting from its current estimate, and the innovations used by the diagnosis.
Memory is constant (a few small arrays); one step costs a handful of 5x5 matrix products.
"""
from dataclasses import dataclass

import numpy as np

from .rc import RCModel, continuous, discretise


@dataclass
class StepOutput:
    Ti: float            # filtered air temperature
    Ti_std: float
    kappa: float
    dq: float
    innov: float         # measured - predicted (NaN if the measurement is missing)
    z: float             # normalised innovation


class DigitalTwin:
    def __init__(self, model: RCModel, Ti0, q_kappa=1e-6, q_dq=0.02, dq_tau_h=4.0, track=True):
        self.m = model
        Ad, Bd, Qd, r = model.matrices()
        self.Ad, self.Bd, self.r = Ad, Bd, r
        nx = Ad.shape[0]
        self.nx = nx
        # discrete response of the states to 1 kW of extra heat into the air node
        A, _ = continuous(model.spec, model.p)
        e = np.zeros((nx, 1))
        e[0, 0] = 1.0 / model.p["Ci"]
        _, b_air = discretise(A, e, model.dt)
        self.b_air = b_air[:, 0]
        self.track = track
        self.phi = float(np.exp(-model.dt / dq_tau_h)) if np.isfinite(dq_tau_h) else 1.0
        n = nx + 2
        self.x = np.r_[np.full(nx, Ti0), 1.0, 0.0]
        self.P = np.diag(np.r_[np.full(nx, 1.0), 0.01, 1.0])
        self.Q = np.zeros((n, n))
        self.Q[:nx, :nx] = Qd
        if track:
            self.Q[nx, nx], self.Q[nx + 1, nx + 1] = q_kappa, q_dq
        else:
            self.P[nx:, nx:] = 0.0
        self.C = np.zeros(n)
        self.C[0] = 1.0

    # --- dynamics of the augmented state ---
    def _f(self, z, u):
        x, kappa, dq = z[:self.nx], z[self.nx], z[self.nx + 1]
        uu = u.copy()
        uu[1] = kappa * u[1]
        return np.r_[self.Ad @ x + self.Bd @ uu + self.b_air * dq, kappa, self.phi * dq]

    def _F(self, z, u):
        n = self.nx + 2
        J = np.eye(n)
        J[:self.nx, :self.nx] = self.Ad
        J[:self.nx, self.nx] = self.Bd[:, 1] * u[1]
        J[:self.nx, self.nx + 1] = self.b_air
        J[self.nx + 1, self.nx + 1] = self.phi
        return J

    # --- one step of operation ---
    def update(self, y):
        S = self.P[0, 0] + self.r
        if np.isnan(y):
            return np.nan, S
        e = y - self.x[0]
        K = self.P[:, 0] / S
        self.x = self.x + K * e
        IKC = np.eye(len(self.x)) - np.outer(K, self.C)
        self.P = IKC @ self.P @ IKC.T + self.r * np.outer(K, K)
        return e, S

    def predict(self, u):
        J = self._F(self.x, u)
        self.x = self._f(self.x, u)
        self.P = J @ self.P @ J.T + self.Q

    def step(self, y, u):
        """Assimilate the measurement y taken at this time, then advance with the inputs u applied
        until the next step."""
        e, S = self.update(y)
        out = StepOutput(self.x[0], float(np.sqrt(self.P[0, 0])), self.x[self.nx], self.x[self.nx + 1],
                         e, e / np.sqrt(S) if not np.isnan(e) else np.nan)
        self.predict(u)
        return out

    def run(self, U, Y):
        rows = [self.step(Y[k], U[k]) for k in range(len(Y))]
        return {f: np.array([getattr(r, f) for r in rows]) for f in StepOutput.__dataclass_fields__}

    # --- what-if forecasts from the current estimate ---
    def forecast(self, U, extra_heat=None):
        """Mean and standard deviation of Ti over a planned input sequence U (h, 5), starting from
        the current state; extra_heat (h,) adds known heat to the air node (kW, model units)."""
        x, P = self.x.copy(), self.P.copy()
        mean, std = np.empty(len(U)), np.empty(len(U))
        for k in range(len(U)):
            J = self._F(x, U[k])
            x = self._f(x, U[k])
            if extra_heat is not None:
                x[:self.nx] += self.b_air * extra_heat[k]
            P = J @ P @ J.T + self.Q
            mean[k], std[k] = x[0], np.sqrt(P[0, 0] + self.r)
        return mean, std

    def state(self):
        return self.x.copy(), self.P.copy()

    def set_state(self, x, P):
        self.x, self.P = x.copy(), P.copy()
