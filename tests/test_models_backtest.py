import numpy as np
import pandas as pd
import pytest

from forecast.backtest import backtest, mape, rmse
from forecast.config import ModelConfig, load_config
from forecast.data import load_series
from forecast.models import GBMForecaster, SeasonalNaive, build_model


def test_metrics():
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)
    assert rmse([1, 2, 3], [1, 2, 3]) == 0
    with pytest.raises(ValueError):
        mape([0, 1], [0, 1])


def test_seasonal_naive_repeats_last_season():
    y = pd.Series(np.arange(24.0), index=pd.date_range("2020-01-01", periods=24, freq="MS"))
    p = SeasonalNaive(12).fit(y).predict(14)
    assert list(p.values[:12]) == list(range(12, 24)) and p.values[12] == 12
    assert p.index[0] == pd.Timestamp("2022-01-01")


def test_seasonal_naive_needs_full_season():
    with pytest.raises(ValueError):
        SeasonalNaive(12).fit(pd.Series([1.0] * 5, index=pd.date_range("2020-01-01", periods=5, freq="MS")))


def test_gbm_forecast_shape_dates_and_determinism():
    y = load_series("retail_sales")
    p1 = GBMForecaster().fit(y).predict(6)
    p2 = GBMForecaster().fit(y).predict(6)
    assert len(p1) == 6 and p1.index[0] == y.index[-1] + pd.offsets.MonthBegin()
    np.testing.assert_allclose(p1.values, p2.values)
    assert np.isfinite(p1.values).all()


def test_gbm_rejects_short_history():
    with pytest.raises(ValueError, match="at least"):
        GBMForecaster().fit(load_series("retail_sales").iloc[:20])


def test_gbm_beats_seasonal_naive_on_trending_series():
    y = load_series("retail_sales")
    model = backtest(y, lambda: GBMForecaster(), 12, 3)
    base = backtest(y, lambda: SeasonalNaive(12), 12, 3)
    assert model["mape"] < base["mape"]
    assert len(model["folds"]) == 3


def test_backtest_rejects_too_short_series():
    y = load_series("retail_sales").iloc[:30]
    with pytest.raises(ValueError):
        backtest(y, lambda: SeasonalNaive(12), 12, 3)


def test_build_model_and_config_versioning():
    assert isinstance(build_model(ModelConfig(name="seasonal_naive")), SeasonalNaive)
    with pytest.raises(ValueError):
        build_model(ModelConfig(name="prophet"))
    a, b = ModelConfig(), ModelConfig(learning_rate=0.1)
    assert a.version_hash() == ModelConfig().version_hash() != b.version_hash()
    cfg = load_config()
    assert cfg.model.lags == (1, 2, 3, 6, 12) and cfg.max_horizon == 24
