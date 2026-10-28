from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .config import ModelConfig
from .features import make_features


class SeasonalNaive:
    """Baseline: next value = value one season ago. Any real model must beat this."""

    def __init__(self, season_length: int = 12):
        self.m = season_length

    def fit(self, y: pd.Series) -> "SeasonalNaive":
        if len(y) < self.m:
            raise ValueError("need at least one full season of history")
        self.last_season_ = y.values[-self.m:]
        self.last_date_ = y.index[-1]
        return self

    def predict(self, horizon: int) -> pd.Series:
        vals = np.array([self.last_season_[i % self.m] for i in range(horizon)])
        idx = pd.date_range(self.last_date_, periods=horizon + 1, freq="MS")[1:]
        return pd.Series(vals, index=idx)


class GBMForecaster:
    """Gradient boosting on lag/rolling/calendar features, forecasting recursively.

    Trained on first differences of the target so the trees (which can't extrapolate)
    still capture trend; predictions are integrated back to levels.
    """

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        self.cfg = cfg

    def fit(self, y: pd.Series) -> "GBMForecaster":
        c = self.cfg
        need = max(c.lags) + max(c.rolling_windows) + 2
        if len(y) < need:
            raise ValueError(f"need at least {need} observations, got {len(y)}")
        self.history_ = y.astype(float).copy()
        d = y.diff().dropna()
        X = make_features(d.values, d.index, c.lags, c.rolling_windows)
        mask = X.notna().all(axis=1)
        self.model_ = GradientBoostingRegressor(
            n_estimators=c.n_estimators, learning_rate=c.learning_rate,
            max_depth=c.max_depth, random_state=c.random_state,
        ).fit(X[mask], d[mask])
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, horizon: int) -> pd.Series:
        c = self.cfg
        y = self.history_
        diffs = list(y.diff().dropna().values)
        idx = pd.date_range(y.index[0], periods=len(y) + horizon, freq="MS")
        level, out = y.values[-1], []
        for _ in range(horizon):
            dates = idx[1: len(diffs) + 2]
            X = make_features(np.append(diffs, np.nan), dates, c.lags, c.rolling_windows).iloc[[-1]]
            step = float(self.model_.predict(X[self.feature_names_])[0])
            diffs.append(step)
            level += step
            out.append(level)
        return pd.Series(out, index=idx[-horizon:])


def build_model(cfg: ModelConfig):
    if cfg.name == "seasonal_naive":
        return SeasonalNaive(cfg.season_length)
    if cfg.name == "gbm_lags":
        return GBMForecaster(cfg)
    raise ValueError(f"unknown model {cfg.name!r}")
