"""Grey-box RC models of one air-conditioned zone, at three levels of detail, and their identification
from measured data by maximum likelihood (the prediction-error method with a Kalman filter, as in
Bacher and Madsen, Energy and Buildings 43, 2011).

Units: temperatures in degC, heat flows in kW, capacities in kWh/K, resistances in K/kW, time in h.

Inputs u (columns, in this order)::

    T_out   outdoor air temperature (degC)
    P_ac    electrical power of the zone's air-conditioning units (kW)
    P_int   electrical power of lighting and plug loads in the zone (kW): internal gains
    G       global horizontal irradiance (kW/m2)
    one     constant 1, for heat flows the inputs do not explain (occupants, neighbouring zones)

Models (states, parameters)::

    M1  1R1C: Ti.                Ci dTi/dt = (To - Ti)/Rio - eta P_ac + a_int P_int + a_sol G + q0
    M2  2R2C: Ti, Tm.            adds a thermal mass Cm coupled to the air by Rim; sun heats the mass
    M3  4R3C: Ti, Tm, Te.        adds the envelope Te between inside and outside (Rie, Reo) and a direct
                                 path through windows Rw

Cooling and gains are effective: eta is the heat removed per kW of electricity (an apparent COP
including the share of each unit's capacity that reaches this zone), a_sol an effective solar area.

Scale. Temperature data alone cannot fix the absolute size of the heat flows: multiplying every
capacity, heat gain and the cooling by the same factor (and dividing the resistances by it) gives
exactly the same temperatures. One value has to be set from outside, and here it is the apparent
COP: eta = ETA_REF. Capacities, resistances and gains are therefore in units scaled to a COP of 3;
time constants, forecasts and all temperature results do not depend on this choice.
"""
from dataclasses import dataclass, field

import numpy as np
from numba import njit
from scipy.linalg import expm
from scipy.optimize import minimize

INPUTS = ["T_out", "P_ac", "P_int", "G", "one"]
ETA_REF = 3.0      # apparent COP that fixes the scale (see module docstring)

# Physical search ranges. Without them the likelihood can drift to degenerate models (for example a
# random walk with no thermal meaning: tiny resistance, huge capacity and no measurement noise).
BOUNDS = {
    "Ci": (0.05, 50), "Cm": (0.5, 500), "Ce": (0.5, 500),                 # kWh/K
    "Rio": (0.005, 5), "Rim": (0.001, 2), "Rie": (0.002, 5), "Reo": (0.002, 5), "Rw": (0.01, 20),  # K/kW
    "eta": (0.3, 6.0),            # kW heat removed per kW electricity
    "a_int": (0.0, 2.0),          # share of lighting and plug power that heats the zone (can exceed 1: occupants)
    "a_sol": (1e-3, 200.0),       # m2
    "q0": (-20.0, 20.0),          # kW
    "q_Ti": (1e-5, 10.0), "q_Tm": (1e-6, 5.0), "q_Te": (1e-6, 5.0),     # K2/h
    "r": (1e-4, 0.25),            # K2: sensor noise, at least its 0.01 K resolution
}


@dataclass
class ModelSpec:
    name: str
    states: list
    params: list            # physical parameters, all positive except q0
    noise: list             # process-noise intensities, one per state, plus measurement noise "r"
    guess: dict = field(default_factory=dict)


SPECS = {
    "M1": ModelSpec("M1 (1R1C)", ["Ti"], ["Ci", "Rio", "a_int", "a_sol", "q0"], ["q_Ti", "r"],
                    dict(Ci=2.0, Rio=0.5, eta=2.5, a_int=0.8, a_sol=5.0, q0=0.0, q_Ti=0.05, r=0.01)),
    "M2": ModelSpec("M2 (2R2C)", ["Ti", "Tm"], ["Ci", "Cm", "Rio", "Rim", "a_int", "a_sol", "q0"],
                    ["q_Ti", "q_Tm", "r"],
                    dict(Ci=0.5, Cm=10.0, Rio=0.5, Rim=0.1, eta=2.5, a_int=0.8, a_sol=5.0, q0=0.0,
                         q_Ti=0.05, q_Tm=0.005, r=0.01)),
    "M3": ModelSpec("M3 (4R3C)", ["Ti", "Tm", "Te"],
                    ["Ci", "Cm", "Ce", "Rim", "Rie", "Reo", "Rw", "a_int", "a_sol", "q0"],
                    ["q_Ti", "q_Tm", "q_Te", "r"],
                    dict(Ci=0.5, Cm=10.0, Ce=5.0, Rim=0.1, Rie=0.3, Reo=0.3, Rw=2.0, eta=2.5, a_int=0.8,
                         a_sol=5.0, q0=0.0, q_Ti=0.05, q_Tm=0.005, q_Te=0.005, r=0.01)),
}


def continuous(spec: ModelSpec, p: dict):
    """Continuous-time A, B (inputs as in INPUTS)."""
    p = {"eta": ETA_REF, **p}
    if spec.states == ["Ti"]:
        A = np.array([[-1 / (p["Ci"] * p["Rio"])]])
        B = np.array([[1 / p["Rio"], -p["eta"], p["a_int"], p["a_sol"], p["q0"]]]) / p["Ci"]
    elif spec.states == ["Ti", "Tm"]:
        Ci, Cm, Rio, Rim = p["Ci"], p["Cm"], p["Rio"], p["Rim"]
        A = np.array([[-(1 / Rio + 1 / Rim) / Ci, 1 / (Rim * Ci)],
                      [1 / (Rim * Cm), -1 / (Rim * Cm)]])
        B = np.array([[1 / (Rio * Ci), -p["eta"] / Ci, p["a_int"] / Ci, 0.0, p["q0"] / Ci],
                      [0.0, 0.0, 0.0, p["a_sol"] / Cm, 0.0]])
    else:
        Ci, Cm, Ce = p["Ci"], p["Cm"], p["Ce"]
        Rim, Rie, Reo, Rw = p["Rim"], p["Rie"], p["Reo"], p["Rw"]
        A = np.array([[-(1 / Rim + 1 / Rie + 1 / Rw) / Ci, 1 / (Rim * Ci), 1 / (Rie * Ci)],
                      [1 / (Rim * Cm), -1 / (Rim * Cm), 0.0],
                      [1 / (Rie * Ce), 0.0, -(1 / Rie + 1 / Reo) / Ce]])
        B = np.array([[1 / (Rw * Ci), -p["eta"] / Ci, p["a_int"] / Ci, 0.0, p["q0"] / Ci],
                      [0.0, 0.0, 0.0, p["a_sol"] / Cm, 0.0],
                      [1 / (Reo * Ce), 0.0, 0.0, 0.0, 0.0]])
    return A, B


def discretise(A, B, dt):
    """Zero-order-hold discretisation through the matrix exponential of the augmented system."""
    nx, nu = B.shape
    M = np.zeros((nx + nu, nx + nu))
    M[:nx, :nx], M[:nx, nx:] = A, B
    E = expm(M * dt)
    return E[:nx, :nx], E[:nx, nx:]


@njit(cache=True)
def _kf_loglik(Ad, Bd, Qd, r, U, Y, x0, P0):
    """Gaussian log-likelihood of the innovations (measured state: the first one). Missing y are NaN."""
    nx = Ad.shape[0]
    x = x0.copy()
    P = P0.copy()
    ll = 0.0
    nobs = 0
    for k in range(Y.shape[0]):
        y = Y[k]
        if not np.isnan(y):
            S = P[0, 0] + r
            e = y - x[0]
            K = P[:, 0] / S
            x = x + K * e
            P = P - np.outer(K, P[0, :])
            P = 0.5 * (P + P.T)
            ll -= 0.5 * (np.log(2 * np.pi * S) + e * e / S)
            nobs += 1
        x = Ad @ x + Bd @ U[k]
        P = Ad @ P @ Ad.T + Qd
    return ll, nobs


class RCModel:
    """One zone model at a given level of detail; parameters are estimated with fit()."""

    def __init__(self, key, dt_h=5 / 60):
        self.key, self.spec, self.dt = key, SPECS[key], dt_h
        self.p = dict(self.spec.guess)

    # parameter vector: log of positive parameters, q0 as is
    def _names(self):
        return self.spec.params + self.spec.noise

    LINEAR = ("q0", "a_int")      # can be zero, so not log-transformed

    def _to_vec(self, p):
        return np.array([p[n] if n in self.LINEAR else np.log(p[n]) for n in self._names()])

    def _from_vec(self, v):
        return {n: (float(x) if n in self.LINEAR else float(np.exp(x))) for n, x in zip(self._names(), v)}

    def _bounds(self):
        return [BOUNDS[n] if n in self.LINEAR else tuple(np.log(BOUNDS[n])) for n in self._names()]

    def matrices(self, p=None):
        p = p or self.p
        A, B = continuous(self.spec, p)
        Ad, Bd = discretise(A, B, self.dt)
        Qd = np.diag([p[f"q_{s}"] for s in self.spec.states]) * self.dt
        c = np.ascontiguousarray
        return c(Ad), c(Bd), c(Qd), p["r"]

    def loglik(self, U, Y, p=None):
        Ad, Bd, Qd, r = self.matrices(p)
        nx = len(self.spec.states)
        y0 = Y[~np.isnan(Y)][0]
        ll, _ = _kf_loglik(Ad, Bd, Qd, r, U, Y, np.full(nx, y0), np.eye(nx))
        return ll

    def fit(self, U, Y, starts=4, seed=0, maxiter=400):
        """Maximum likelihood over several starting points (log-normal perturbations of the guess)."""
        rng = np.random.default_rng(seed)
        best = None
        v0 = self._to_vec(self.spec.guess)

        def nll(v):
            try:
                val = -self.loglik(U, Y, self._from_vec(v))
            except (np.linalg.LinAlgError, ValueError, OverflowError):
                return 1e12
            return val if np.isfinite(val) else 1e12

        for s in range(starts):
            b = self._bounds()
            v = v0 + (rng.normal(0, 0.7, len(v0)) if s else 0)
            v = np.clip(v, [lo for lo, _ in b], [hi for _, hi in b])
            res = minimize(nll, v, method="L-BFGS-B", bounds=b, options={"maxiter": maxiter})
            if best is None or res.fun < best.fun:
                best = res
        self.p = self._from_vec(best.x)
        self.nll = float(best.fun)
        self.n_obs = int(np.sum(~np.isnan(Y)))
        return self

    def at_bounds(self, tol=0.01):
        """Parameters that ended on a bound of their search range (a sign the data cannot pin them)."""
        out = []
        for n, (lo, hi) in zip(self._names(), self._bounds()):
            v = self.p[n] if n in self.LINEAR else np.log(self.p[n])
            if abs(v - lo) <= tol * (hi - lo) or abs(v - hi) <= tol * (hi - lo):
                out.append(n)
        return out

    @property
    def n_params(self):
        return len(self._names())

    @property
    def aic(self):
        return 2 * self.nll + 2 * self.n_params

    @property
    def bic(self):
        return 2 * self.nll + np.log(self.n_obs) * self.n_params

    def time_constants_h(self):
        A, _ = continuous(self.spec, self.p)
        ev = np.linalg.eigvals(A).real
        return sorted(-1 / ev)

    def simulate(self, U, x0):
        """Open-loop simulation (no measurements) from state x0; returns the states."""
        Ad, Bd, _, _ = self.matrices()
        x = np.asarray(x0, float).copy()
        out = np.empty((len(U), len(x)))
        for k in range(len(U)):
            out[k] = x
            x = Ad @ x + Bd @ U[k]
        return out
