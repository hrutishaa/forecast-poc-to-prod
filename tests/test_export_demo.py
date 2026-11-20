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
        b = s[key]["gbm_lags"]
        assert len(b["lower"]) == len(b["upper"]) == p["max_horizon"]
        assert all(lo <= hi for lo, hi in zip(b["lower"], b["upper"]))
