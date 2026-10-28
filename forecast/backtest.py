"""Rolling-origin backtesting: the honest way to estimate forecast error."""
from __future__ import annotations

import numpy as np
import pandas as pd


def mape(actual, pred) -> float:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    if np.any(a == 0):
        raise ValueError("MAPE undefined when actuals contain 0")
    return float(np.mean(np.abs((a - p) / a)) * 100)


def rmse(actual, pred) -> float:
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def backtest(y: pd.Series, make_model, horizon: int = 12, folds: int = 3) -> dict:
    """Train on y[:cut], forecast `horizon`, score, slide the origin forward by `horizon`."""
    if len(y) <= horizon * folds:
        raise ValueError("series too short for requested folds/horizon")
    results = []
    for k in range(folds, 0, -1):
        cut = len(y) - k * horizon
        train, test = y.iloc[:cut], y.iloc[cut: cut + horizon]
        pred = make_model().fit(train).predict(len(test))
        results.append({"origin": str(train.index[-1].date()), "mape": mape(test, pred), "rmse": rmse(test, pred)})
    return {
        "folds": results,
        "mape": float(np.mean([r["mape"] for r in results])),
        "rmse": float(np.mean([r["rmse"] for r in results])),
    }
