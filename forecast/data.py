"""Sample monthly series. Synthetic but realistic (trend + seasonality + promos/shocks + noise),
seeded so results are reproducible. Swap `load_series` for your warehouse query in production."""
from __future__ import annotations

import numpy as np
import pandas as pd

SERIES = {
    "retail_sales": "Monthly retail sales ($K): growth trend, strong Nov/Dec holiday peak",
    "ecommerce_orders": "Monthly e-commerce orders: fast growth, multiplicative seasonality",
    "energy_demand": "Monthly electricity demand (GWh): summer + winter double peak, flat trend",
}


def load_series(name: str, n_months: int = 96, start: str = "2018-01-01") -> pd.Series:
    if name not in SERIES:
        raise KeyError(f"unknown series {name!r}; choose from {sorted(SERIES)}")
    rng = np.random.default_rng(sum(map(ord, name)))  # stable across processes (hash() is salted)
    t = np.arange(n_months)
    m = t % 12
    if name == "retail_sales":
        season = np.array([-8, -10, -2, 0, 3, 2, 1, 4, 0, 3, 18, 34])
        y = 400 + 2.2 * t + season[m] * (1 + t / 150) + rng.normal(0, 6, n_months)
    elif name == "ecommerce_orders":
        season = 1 + 0.25 * np.sin(2 * np.pi * (m - 2) / 12) + np.where(m == 10, 0.35, 0)
        y = 1000 * np.exp(0.012 * t) * season * rng.normal(1, 0.03, n_months)
    else:
        season = 30 * np.cos(2 * np.pi * (m - 0.5) / 6) + 10 * np.cos(2 * np.pi * m / 12)
        y = 520 + season + rng.normal(0, 8, n_months)
    idx = pd.date_range(start, periods=n_months, freq="MS")
    return pd.Series(np.round(y, 2), index=idx, name=name)


def train_test_split(y: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    if horizon <= 0 or horizon >= len(y):
        raise ValueError("horizon must be between 1 and len(y)-1")
    return y.iloc[:-horizon], y.iloc[-horizon:]
