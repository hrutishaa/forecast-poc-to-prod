import numpy as np
import pandas as pd
import pytest

from forecast.data import SERIES, load_series, train_test_split
from forecast.features import make_features


@pytest.mark.parametrize("name", sorted(SERIES))
def test_series_are_reproducible_monthly_and_positive(name):
    a, b = load_series(name), load_series(name)
    pd.testing.assert_series_equal(a, b)
    assert len(a) == 96 and a.index.freqstr == "MS" and (a > 0).all()


def test_unknown_series_raises():
    with pytest.raises(KeyError):
        load_series("nope")


def test_train_test_split_sizes_and_validation():
    y = load_series("retail_sales")
    tr, te = train_test_split(y, 12)
    assert len(tr) == 84 and len(te) == 12 and tr.index[-1] < te.index[0]
    with pytest.raises(ValueError):
        train_test_split(y, 0)


def test_features_do_not_leak_current_value():
    vals = np.arange(30, dtype=float)
    X = make_features(vals, pd.date_range("2020-01-01", periods=30, freq="MS"), [1, 12], [3])
    assert X["lag_1"].iloc[20] == vals[19]
    assert X["lag_12"].iloc[20] == vals[8]
    assert X["roll_mean_3"].iloc[20] == pytest.approx(vals[17:20].mean())
    assert X["lag_1"].isna().iloc[0]
