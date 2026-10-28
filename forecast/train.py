"""Train pipeline: python -m forecast.train --series retail_sales"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

from . import registry
from .backtest import backtest
from .config import load_config
from .data import SERIES, load_series
from .models import build_model


def train(series: str, cfg=None, root=registry.ARTIFACTS) -> dict:
    cfg = cfg or load_config()
    y = load_series(series)
    model_bt = backtest(y, lambda: build_model(cfg.model), cfg.horizon, cfg.folds)
    base_bt = backtest(y, lambda: build_model(replace(cfg.model, name="seasonal_naive")), cfg.horizon, cfg.folds)
    metrics = {"model": model_bt, "baseline": base_bt,
               "beats_baseline": model_bt["mape"] < base_bt["mape"]}
    model = build_model(cfg.model).fit(y)
    version = f"{cfg.model.name}-{cfg.model.version_hash()}"
    registry.save(model, series, version, metrics, root)
    return {"series": series, "version": version, **metrics}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--series", choices=sorted(SERIES), default=None, help="default: all")
    a = p.parse_args(argv)
    for s in [a.series] if a.series else sorted(SERIES):
        r = train(s)
        print(json.dumps({k: r[k] for k in ("series", "version", "beats_baseline")} |
                         {"mape": round(r["model"]["mape"], 2), "baseline_mape": round(r["baseline"]["mape"], 2)}))


if __name__ == "__main__":
    main()
