from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from forecast.backtest import backtest, smape
from forecast.config import ModelConfig, load_config
from forecast.data import load_series
from forecast.ets import ETSForecaster, smooth
from forecast.intervals import calibrate, coverage
from forecast.models import CANDIDATES, SeasonalNaive, build_model
from forecast.selection import select_model

IDX = lambda n: pd.date_range("2020-01-01", periods=n, freq="MS")  # noqa: E731


def test_smape_metric():
    assert smape([100, 100], [100, 100]) == 0
    assert smape([100], [50]) == pytest.approx(2 * 50 / 150 * 100)
    assert smape([0, 0], [0, 0]) == 0  # defined where MAPE is not
    assert smape([100], [0]) == pytest.approx(200)


def test_ses_on_constant_series_forecasts_the_constant():
    y = pd.Series([7.0] * 20, index=IDX(20))
    p = ETSForecaster("simple").fit(y).predict(5)
    np.testing.assert_allclose(p.values, 7.0)
    assert p.index[0] == pd.Timestamp("2021-09-01")


def test_holt_extrapolates_a_straight_line_exactly():
    y = pd.Series(10 + 3.0 * np.arange(30), index=IDX(30))
    m = ETSForecaster("holt").fit(y)
    np.testing.assert_allclose(m.predict(4).values, 10 + 3.0 * np.arange(30, 34), atol=1e-6)
    assert m.sse_ == pytest.approx(0, abs=1e-12)


def test_holt_winters_recovers_trend_plus_additive_season():
    t = np.arange(60)
    season = np.array([5, -3, 0, 2, 8, -6, -4, 1, 3, -2, 6, -10], float)
    y = pd.Series(100 + 0.5 * t + season[t % 12], index=IDX(60))
    p = ETSForecaster("holt_winters_add", 12).fit(y).predict(15)
    tf = np.arange(60, 75)
    np.testing.assert_allclose(p.values, 100 + 0.5 * tf + season[tf % 12], atol=1e-6)


def test_smoothing_recursion_matches_hand_computation():
    # SES with alpha=0.5 from level y0=10: levels 10 -> 11 -> 10.5, errors 0, 2, -1
    sse, level, slope, _ = smooth([10.0, 12.0, 10.0], 0.5)
    assert sse == pytest.approx(0 + 4 + 1) and level == pytest.approx(10.5) and slope == 0


def test_ets_fit_is_deterministic_bounded_and_validates_input():
    y = load_series("energy_demand")
    a, b = ETSForecaster().fit(y), ETSForecaster().fit(y)
    np.testing.assert_allclose(a.predict(24).values, b.predict(24).values)
    assert all(0 < v < 1 for v in a.params.values()) and set(a.params) == {"alpha", "beta", "gamma"}
    with pytest.raises(ValueError, match="at least 24"):
        ETSForecaster("holt_winters_add", 12).fit(y.iloc[:20])
    with pytest.raises(ValueError, match="unknown ETS kind"):
        ETSForecaster("multiplicative")
    with pytest.raises(ValueError, match="NaN"):
        ETSForecaster("simple").fit(pd.Series([1.0, np.nan, 2.0], index=IDX(3)))


def test_build_model_knows_all_candidates_and_refuses_auto():
    for name in CANDIDATES:
        m = build_model(ModelConfig(name=name)).fit(load_series("retail_sales"))
        assert len(m.predict(3)) == 3
    with pytest.raises(ValueError, match="auto"):
        build_model(ModelConfig(name="auto"))


def test_holt_winters_beats_baseline_on_seasonal_series():
    y = load_series("energy_demand")
    hw = backtest(y, lambda: ETSForecaster("holt_winters_add"), 12, 3)
    base = backtest(y, lambda: SeasonalNaive(12), 12, 3)
    assert hw["mape"] < base["mape"] and "smape" in hw


def test_selection_ranks_all_candidates_and_picks_lowest_error():
    cfg = load_config()
    sel = select_model(load_series("retail_sales"), cfg)
    board = sel["leaderboard"]
    assert {r["model"] for r in board} == set(cfg.candidates) and sel["metric"] == "mape"
    assert [r["mape"] for r in board] == sorted(r["mape"] for r in board)
    assert sel["winner"] == board[0]["model"]
    sm = select_model(load_series("retail_sales"), cfg, metric="smape")
    assert [r["smape"] for r in sm["leaderboard"]] == sorted(r["smape"] for r in sm["leaderboard"])


def test_selection_prefers_simple_model_on_pure_seasonal_pattern_and_validates():
    # A repeating pattern: seasonal naive is perfect (0 error) and listed first, so it wins the tie.
    y = pd.Series(np.tile(np.arange(1.0, 13.0), 6), index=IDX(72))
    cfg = replace(load_config(), candidates=("seasonal_naive", "ets_simple"))
    assert select_model(y, cfg)["winner"] == "seasonal_naive"
    with pytest.raises(ValueError, match="unknown candidate"):
        select_model(y, cfg, candidates=("prophet",))
    with pytest.raises(ValueError, match="metric"):
        select_model(y, cfg, metric="r2")


def test_prediction_intervals_work_for_holt_winters():
    y = load_series("ecommerce_orders")
    train, test = y.iloc[:-12], y.iloc[-12:]
    make = lambda: ETSForecaster("holt_winters_add")  # noqa: E731
    cal = calibrate(train, make, horizon=12, level=0.8, n_origins=12)
    band = cal.apply(make().fit(train).predict(12))
    assert (band.lower <= band.point).all() and (band.point <= band.upper).all()
    assert coverage(test, band.lower, band.upper) >= 0.6
