"""Prediction intervals with split conformal calibration (no extra model, no distribution assumption).

Idea in plain words: before we trust a forecast range, we check how wrong the model *actually* was.
1. Re-fit the model at several past cut-off points ("origins") and forecast `horizon` steps ahead.
2. For each step ahead h, collect the relative errors |actual - pred| / |pred| across origins.
3. The band at step h is pred * (1 ± q_h), where q_h is the conformal quantile of those errors.

Because errors are measured per step ahead, the band widens naturally the further out we forecast.
Errors are relative (not absolute) so growing series (e.g. e-commerce) get bands that scale with level.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


def conformal_quantile(scores, level: float) -> float:
    """Finite-sample conformal quantile: the ceil((n+1)*level)-th smallest score.

    Gives >= `level` coverage when future errors are exchangeable with the calibration errors.
    Raises if there are too few scores to reach the requested level honestly."""
    if not 0 < level < 1:
        raise ValueError("level must be between 0 and 1")
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = math.ceil((n + 1) * level)
    if n == 0 or k > n:
        need = max(1, math.ceil(level / (1 - level)))
        raise ValueError(f"need at least {need} calibration errors for a {level:.0%} interval, got {n}")
    return float(s[k - 1])


@dataclass(frozen=True)
class Calibration:
    level: float
    rel_q: tuple[float, ...]  # relative half-width per step ahead (index 0 = 1 month ahead)
    n_origins: int

    @property
    def horizon(self) -> int:
        return len(self.rel_q)

    def to_dict(self) -> dict:
        return {"level": self.level, "rel_q": [round(q, 6) for q in self.rel_q], "n_origins": self.n_origins}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        return cls(level=float(d["level"]), rel_q=tuple(float(q) for q in d["rel_q"]), n_origins=int(d["n_origins"]))

    def apply(self, pred: pd.Series) -> pd.DataFrame:
        """Return a frame with columns point, lower, upper (same index as `pred`)."""
        h = len(pred)
        if h > self.horizon:
            raise ValueError(f"calibrated for {self.horizon} steps, asked for {h}")
        q = np.asarray(self.rel_q[:h])
        p = pred.to_numpy(dtype=float)
        half = np.abs(p) * q
        return pd.DataFrame({"point": p, "lower": p - half, "upper": p + half}, index=pred.index)


def calibrate(y: pd.Series, make_model, horizon: int, level: float = 0.8, n_origins: int = 12) -> Calibration:
    """Collect out-of-sample errors from the last `n_origins` cut-offs (one month apart) and
    turn them into a per-step conformal half-width. Only uses data inside `y`, so calibrating on
    the training window never peeks at the holdout."""
    last_cut = len(y) - horizon
    first_cut = last_cut - n_origins + 1
    if first_cut < 1:
        raise ValueError(f"series of {len(y)} points is too short for {n_origins} origins x {horizon} steps")
    errors = np.empty((n_origins, horizon))
    for i, cut in enumerate(range(first_cut, last_cut + 1)):
        train, test = y.iloc[:cut], y.iloc[cut: cut + horizon]
        pred = make_model().fit(train).predict(horizon).to_numpy(dtype=float)
        errors[i] = np.abs(test.to_numpy(dtype=float) - pred) / np.maximum(np.abs(pred), 1e-9)
    q = [conformal_quantile(errors[:, h], level) for h in range(horizon)]
    # Uncertainty should not shrink further out: enforce non-decreasing width.
    q = np.maximum.accumulate(q)
    return Calibration(level=level, rel_q=tuple(float(x) for x in q), n_origins=n_origins)


def coverage(actual, lower, upper) -> float:
    """Share of actuals that fell inside [lower, upper] (0..1)."""
    a, lo, hi = (np.asarray(x, dtype=float) for x in (actual, lower, upper))
    if not (len(a) == len(lo) == len(hi)) or len(a) == 0:
        raise ValueError("actual, lower and upper must be non-empty and the same length")
    return float(np.mean((a >= lo) & (a <= hi)))
