"""Automatic model selection: backtest every candidate on the same folds, ship the lowest error.

Same idea as the baseline gate, generalised: every candidate (seasonal naive, GBM, three ETS
variants) gets the identical rolling-origin backtest, and the one with the lowest mean error
(MAPE by default, or sMAPE) is chosen for that series. Ties go to the earlier (simpler) candidate.

CLI:  python -m forecast.selection [--series retail_sales] [--metric smape]
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from .backtest import backtest
from .config import Config, load_config
from .data import SERIES, load_series
from .models import CANDIDATES, build_model

METRICS = ("mape", "smape", "rmse")


def select_model(y: pd.Series, cfg: Config, candidates=None, metric: str | None = None) -> dict:
    metric = metric or cfg.selection_metric
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
    candidates = tuple(candidates or cfg.candidates)
    unknown = [c for c in candidates if c not in CANDIDATES]
    if unknown or not candidates:
        raise ValueError(f"unknown candidate(s) {unknown}; choose from {CANDIDATES}")
    board = []
    for name in candidates:
        mc = replace(cfg.model, name=name)
        bt = backtest(y, lambda mc=mc: build_model(mc), cfg.horizon, cfg.folds)
        board.append({"model": name, "mape": bt["mape"], "smape": bt["smape"], "rmse": bt["rmse"]})
    order = {c: i for i, c in enumerate(candidates)}
    board.sort(key=lambda r: (r[metric], order[r["model"]]))
    return {"metric": metric, "winner": board[0]["model"], "leaderboard": board,
            "folds": cfg.folds, "horizon": cfg.horizon}


def main(argv=None):
    p = argparse.ArgumentParser(description="Backtest all candidate models and pick one per series")
    p.add_argument("--series", choices=sorted(SERIES), default=None, help="default: all")
    p.add_argument("--metric", choices=METRICS, default=None, help="default: [selection] metric in config.toml")
    a = p.parse_args(argv)
    cfg = load_config()
    for s in [a.series] if a.series else sorted(SERIES):
        sel = select_model(load_series(s), cfg, metric=a.metric)
        print(f"\n{s}  (rolling backtest {sel['folds']} folds x {sel['horizon']} months, ranked by {sel['metric']})")
        print(f"  {'model':<18}{'MAPE %':>9}{'sMAPE %':>9}{'RMSE':>10}")
        for r in sel["leaderboard"]:
            tag = "  <- selected" if r["model"] == sel["winner"] else ""
            print(f"  {r['model']:<18}{r['mape']:>9.2f}{r['smape']:>9.2f}{r['rmse']:>10.2f}{tag}")


if __name__ == "__main__":
    main()
