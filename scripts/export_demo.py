"""Precompute everything the static live demo needs into docs/data.json.

The browser never runs a model: it reads these results, so the demo is free to host
and loads instantly. CI re-runs this script to prove it still works."""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecast.backtest import backtest  # noqa: E402
from forecast.config import load_config  # noqa: E402
from forecast.data import SERIES, load_series  # noqa: E402
from forecast.models import build_model  # noqa: E402


def r(xs):
    return [round(float(x), 2) for x in xs]


def main(out: Path = ROOT / "docs" / "data.json") -> dict:
    cfg = load_config()
    H = cfg.max_horizon
    models = {"seasonal_naive": replace(cfg.model, name="seasonal_naive"), "gbm_lags": cfg.model}
    payload = {"max_horizon": H, "model_version": f"{cfg.model.name}-{cfg.model.version_hash()}", "series": {}}
    for name, desc in SERIES.items():
        y = load_series(name)
        train = y.iloc[:-H]
        entry = {"description": desc, "dates": [str(d.date()) for d in y.index], "values": r(y.values),
                 "cutoff_index": len(train) - 1, "holdout": {}, "future": {}, "future_dates": None, "rolling_backtest": {}}
        for key, mc in models.items():
            entry["holdout"][key] = r(build_model(mc).fit(train).predict(H).values)
            fut = build_model(mc).fit(y).predict(H)
            entry["future"][key] = r(fut.values)
            entry["future_dates"] = [str(d.date()) for d in fut.index]
            bt = backtest(y, lambda mc=mc: build_model(mc), cfg.horizon, cfg.folds)
            entry["rolling_backtest"][key] = {"mape": round(bt["mape"], 2), "rmse": round(bt["rmse"], 2)}
        payload["series"][name] = entry
    out.write_text(json.dumps(payload, separators=(",", ":")))
    return payload


if __name__ == "__main__":
    p = main()
    print(f"wrote docs/data.json for {len(p['series'])} series")
