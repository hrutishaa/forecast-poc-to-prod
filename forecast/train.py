"""Train pipeline: python -m forecast.train [--series retail_sales] [--model auto|ets_hw_add|...]"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

from . import registry
from .backtest import backtest
from .config import load_config
from .data import SERIES, load_series
from .intervals import calibrate
from .models import CANDIDATES, build_model
from .selection import select_model


def resolve_model(y, cfg, model: str | None = None):
    """Return (ModelConfig to ship, selection report or None). 'auto' = lowest backtest error wins."""
    name = model or cfg.model.name
    if name == "auto":
        sel = select_model(y, cfg)
        return replace(cfg.model, name=sel["winner"]), sel
    return replace(cfg.model, name=name), None


def train(series: str, cfg=None, root=registry.ARTIFACTS, model: str | None = None) -> dict:
    cfg = cfg or load_config()
    y = load_series(series)
    mc, sel = resolve_model(y, cfg, model)
    model_bt = backtest(y, lambda: build_model(mc), cfg.horizon, cfg.folds)
    base_bt = backtest(y, lambda: build_model(replace(cfg.model, name="seasonal_naive")), cfg.horizon, cfg.folds)
    metrics = {"model": model_bt, "baseline": base_bt,
               "beats_baseline": model_bt["mape"] < base_bt["mape"]}
    if sel is not None:
        metrics["selection"] = sel
    fitted = build_model(mc).fit(y)
    version = f"{mc.name}-{mc.version_hash()}"
    calib = calibrate(y, lambda: build_model(mc), cfg.max_horizon, cfg.interval_level, cfg.interval_origins)
    registry.save(fitted, series, version, metrics, root, intervals=calib.to_dict())
    return {"series": series, "model_name": mc.name, "version": version, **metrics, "intervals": calib.to_dict()}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--series", choices=sorted(SERIES), default=None, help="default: all")
    p.add_argument("--model", choices=("auto", *CANDIDATES), default=None, help="default: [model] name in config.toml")
    a = p.parse_args(argv)
    for s in [a.series] if a.series else sorted(SERIES):
        r = train(s, model=a.model)
        print(json.dumps({k: r[k] for k in ("series", "model_name", "version", "beats_baseline")} |
                         {"mape": round(r["model"]["mape"], 2), "smape": round(r["model"]["smape"], 2),
                          "baseline_mape": round(r["baseline"]["mape"], 2)}))


if __name__ == "__main__":
    main()
