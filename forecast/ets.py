"""Exponential smoothing (ETS) forecasters, implemented from scratch with numpy + scipy.optimize.

Three classic members of the family, each a special case of additive Holt-Winters:

    simple (SES)         level only                      y_hat(t+h) = l_t
    holt                 level + linear trend            y_hat(t+h) = l_t + h * b_t
    holt_winters_add     level + trend + additive season y_hat(t+h) = l_t + h * b_t + s_{t+h-m}

Update equations (Hyndman & Athanasopoulos, "Forecasting: Principles and Practice", ch. 8):

    l_t = alpha * (y_t - s_{t-m}) + (1 - alpha) * (l_{t-1} + b_{t-1})
    b_t = beta  * (l_t - l_{t-1}) + (1 - beta)  * b_{t-1}
    s_t = gamma * (y_t - l_{t-1} - b_{t-1}) + (1 - gamma) * s_{t-m}

The smoothing weights (alpha, beta, gamma) are chosen by minimising the sum of squared one-step-ahead
errors on the training data (L-BFGS-B inside [0, 1], started from the best point of a small grid so
the fit is deterministic). Initial states use the standard heuristic: level = mean of the first
season, trend = average change between the first two seasons, season = first season minus that
trend line (all placed at time -1, just before the first observation).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize

KINDS = ("simple", "holt", "holt_winters_add")
_GRID = (0.05, 0.2, 0.5, 0.8)
_EPS = 1e-4  # keep weights strictly inside (0, 1)


def initial_states(y: np.ndarray, trend: bool, m: int | None) -> tuple[float, float, np.ndarray]:
    """Heuristic starting states *at time -1* (just before the first observation), so the very first
    one-step forecast l + b + s already lines up with y[0]."""
    if m:
        slope = float((y[m:2 * m].mean() - y[:m].mean()) / m) if trend else 0.0
        center = (m - 1) / 2                      # the first season's mean sits at its middle
        mean = float(y[:m].mean())
        level = mean - slope * (center + 1)
        season = y[:m] - (mean + slope * (np.arange(m) - center))   # detrended, sums to 0
    else:
        slope = float(y[1] - y[0]) if trend else 0.0
        level = float(y[0]) - slope
        season = np.zeros(0)
    return level, slope, np.asarray(season, dtype=float)


def smooth(y, alpha: float, beta: float = 0.0, gamma: float = 0.0, *, trend: bool = False,
           m: int | None = None, init: tuple | None = None):
    """Run the smoothing recursions over y.

    Returns (sse of one-step-ahead errors, final level, final slope, last m seasonal states)."""
    y = np.asarray(y, dtype=float)
    level, slope, season = init if init is not None else initial_states(y, trend, m)
    s = list(season)
    sse = 0.0
    for t, yt in enumerate(y):
        st = s[t] if m else 0.0                  # s_{t-m}
        err = yt - (level + slope + st)          # one-step-ahead error (forecast made at t-1)
        sse += err * err
        new_level = alpha * (yt - st) + (1 - alpha) * (level + slope)
        if m:
            s.append(gamma * (yt - level - slope) + (1 - gamma) * st)
        if trend:
            slope = beta * (new_level - level) + (1 - beta) * slope
        level = new_level
    return sse, level, slope, (np.asarray(s[-m:]) if m else np.zeros(0))


class ETSForecaster:
    """Exponential smoothing forecaster with the same fit/predict interface as the other models."""

    def __init__(self, kind: str = "holt_winters_add", season_length: int = 12):
        if kind not in KINDS:
            raise ValueError(f"unknown ETS kind {kind!r}; choose from {KINDS}")
        self.kind, self.m = kind, season_length

    @property
    def trend(self) -> bool:
        return self.kind != "simple"

    @property
    def seasonal(self) -> bool:
        return self.kind == "holt_winters_add"

    def _n_params(self) -> int:
        return 1 + self.trend + self.seasonal

    def fit(self, y: pd.Series) -> "ETSForecaster":
        need = 2 * self.m if self.seasonal else (3 if self.trend else 2)
        if len(y) < need:
            raise ValueError(f"{self.kind} needs at least {need} observations, got {len(y)}")
        v = y.to_numpy(dtype=float)
        if not np.isfinite(v).all():
            raise ValueError("series contains NaN or inf")
        m = self.m if self.seasonal else None
        init = initial_states(v, self.trend, m)
        k = self._n_params()

        def unpack(p):
            p = list(p) + [0.0] * (3 - k)
            return p[0], p[1], p[2]

        def objective(p):
            a, b, g = unpack(p)
            return smooth(v, a, b, g, trend=self.trend, m=m, init=init)[0]

        start = min(itertools.product(_GRID, repeat=k), key=objective)
        res = minimize(objective, np.array(start), method="L-BFGS-B", bounds=[(_EPS, 1 - _EPS)] * k)
        best = res.x if res.fun <= objective(start) else np.array(start)
        self.alpha_, self.beta_, self.gamma_ = (float(x) for x in unpack(best))
        self.sse_, self.level_, self.slope_, self.season_ = smooth(
            v, self.alpha_, self.beta_, self.gamma_, trend=self.trend, m=m, init=init)
        self.n_obs_ = len(v)
        self.last_date_ = y.index[-1]
        return self

    @property
    def params(self) -> dict:
        p = {"alpha": round(self.alpha_, 4)}
        if self.trend:
            p["beta"] = round(self.beta_, 4)
        if self.seasonal:
            p["gamma"] = round(self.gamma_, 4)
        return p

    def predict(self, horizon: int) -> pd.Series:
        h = np.arange(1, horizon + 1)
        vals = self.level_ + h * self.slope_
        if self.seasonal:
            # season_ holds the last m seasonal states in time order; step h reuses s_{n+h-m}
            vals = vals + self.season_[(h - 1) % self.m]
        idx = pd.date_range(self.last_date_, periods=horizon + 1, freq="MS")[1:]
        return pd.Series(vals, index=idx)
