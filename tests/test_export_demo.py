import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_export_demo_payload(tmp_path):
    spec = importlib.util.spec_from_file_location("export_demo", ROOT / "scripts" / "export_demo.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    p = mod.main(tmp_path / "data.json")
    s = p["series"]["retail_sales"]
    assert len(s["holdout"]["gbm_lags"]) == p["max_horizon"] == len(s["future_dates"])
    assert s["cutoff_index"] == len(s["values"]) - p["max_horizon"] - 1
    assert (tmp_path / "data.json").stat().st_size > 1000
    assert p["interval_level"] == 0.8
    for key in ("holdout_band", "future_band"):
        b = s[key][s["selection"]["future"]["winner"]]
        assert len(b["lower"]) == len(b["upper"]) == p["max_horizon"]
        assert all(lo <= hi for lo, hi in zip(b["lower"], b["upper"]))
    assert set(p["models"]) == {"seasonal_naive", "gbm_lags", "ets_simple", "ets_holt", "ets_hw_add"}
    assert set(s["holdout"]) == set(p["models"]) and "smape" in s["rolling_backtest"]["ets_hw_add"]
    for view in ("holdout", "future"):
        sel = s["selection"][view]
        assert sel["winner"] == sel["leaderboard"][0]["model"]
    assert s["model_version"].startswith(s["selection"]["future"]["winner"] + "-")
