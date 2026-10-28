"""Feature engineering shared by training AND inference (one code path = no train/serve skew)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_features(values: np.ndarray, dates: pd.DatetimeIndex, lags, windows) -> pd.DataFrame:
    """Row i holds features available *before* observing values[i] (no leakage)."""
    s = pd.Series(values, index=dates, dtype=float)
    X = pd.DataFrame(index=dates)
    for L in lags:
        X[f"lag_{L}"] = s.shift(L)
    for w in windows:
        X[f"roll_mean_{w}"] = s.shift(1).rolling(w).mean()
    X["month_sin"] = np.sin(2 * np.pi * (dates.month - 1) / 12)
    X["month_cos"] = np.cos(2 * np.pi * (dates.month - 1) / 12)
    X["t"] = np.arange(len(dates), dtype=float)
    return X
