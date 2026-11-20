import numpy as np
import pandas as pd
import pytest

from forecast.data import load_series
from forecast.intervals import Calibration, calibrate, conformal_quantile, coverage
from forecast.models import GBMForecaster, SeasonalNaive


def test_conformal_quantile_uses_finite_sample_rank():
    scores = np.arange(1, 11)  # n=10
    assert conformal_quantile(scores, 0.8) == 9  # ceil(11 * 0.8) = 9th smallest
    assert conformal_quantile(scores, 0.5) == 6  # ceil(11 * 0.5) = 6th smallest


def test_conformal_quantile_refuses_dishonest_levels():
    with pytest.raises(ValueError, match="at least 19"):
        conformal_quantile(np.arange(10), 0.95)  # 10 errors cannot back a 95% claim
    with pytest.raises(ValueError):
        conformal_quantile([1.0, 2.0], 1.0)


def test_band_contains_point_and_widens_with_horizon():
    y = load_series("retail_sales")
    cal = calibrate(y, lambda: SeasonalNaive(12), horizon=12, level=0.8, n_origins=12)
    assert cal.horizon == 12 and cal.n_origins == 12
    assert all(b >= a for a, b in zip(cal.rel_q, cal.rel_q[1:]))  # never narrower further out
    band = cal.apply(SeasonalNaive(12).fit(y).predict(12))
    assert list(band.columns) == ["point", "lower", "upper"]
    assert (band.lower <= band.point).all() and (band.point <= band.upper).all()
    assert (band.upper - band.lower).iloc[-1] > 0


def test_perfect_model_gets_zero_width_band():
    y = pd.Series(np.tile(np.arange(1.0, 13.0), 4), index=pd.date_range("2020-01-01", periods=48, freq="MS"))
    cal = calibrate(y, lambda: SeasonalNaive(12), horizon=6, level=0.8, n_origins=10)
    assert max(cal.rel_q) == pytest.approx(0.0)


def test_calibration_round_trips_and_guards_horizon():
    cal = Calibration(level=0.8, rel_q=(0.05, 0.1), n_origins=12)
    assert Calibration.from_dict(cal.to_dict()) == cal
    pred = pd.Series([100.0, 200.0], index=pd.date_range("2026-01-01", periods=2, freq="MS"))
    band = cal.apply(pred)
    assert band.lower.tolist() == pytest.approx([95.0, 180.0]) and band.upper.tolist() == pytest.approx([105.0, 220.0])
    with pytest.raises(ValueError, match="calibrated for 2"):
        cal.apply(pd.concat([pred, pred]))


def test_calibrate_rejects_too_short_series():
    y = load_series("retail_sales").iloc[:30]
    with pytest.raises(ValueError, match="too short"):
        calibrate(y, lambda: SeasonalNaive(12), horizon=24, n_origins=12)


def test_coverage_counts_actuals_inside_band():
    assert coverage([1, 5, 10], [0, 0, 0], [2, 4, 10]) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        coverage([], [], [])


def test_gbm_holdout_coverage_is_honest_on_stable_series():
    """Calibrate on the training window only, then check the band on 12 unseen months."""
    y = load_series("energy_demand")
    train, test = y.iloc[:-12], y.iloc[-12:]
    cal = calibrate(train, lambda: GBMForecaster(), horizon=12, level=0.8, n_origins=12)
    band = cal.apply(GBMForecaster().fit(train).predict(12))
    assert coverage(test, band.lower, band.upper) >= 0.6
