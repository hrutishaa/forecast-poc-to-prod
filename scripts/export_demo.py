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
from forecast.intervals import calibrate  # noqa: E402
from forecast.models import build_model  # noqa: E402
from forecast.selection import select_model  # noqa: E402


def r(xs):
    return [round(float(x), 2) for x in xs]


def _board(sel):
    return [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()} for row in sel["leaderboard"]]


def main(out: Path = ROOT / "docs" / "data.json") -> dict:
    cfg = load_config()
    H = cfg.max_horizon
    models = {name: replace(cfg.model, name=name) for name in cfg.candidates}
    payload = {"max_horizon": H, "interval_level": cfg.interval_level, "models": list(models),
               "selection_metric": cfg.selection_metric, "series": {}}
    for name, desc in SERIES.items():
        y = load_series(name)
        train = y.iloc[:-H]
        # Two honest selections: on the training window only (what the backtest view evaluates on
        # unseen months) and on the full history (what production ships for the future view).
        sel_train, sel_full = select_model(train, cfg), select_model(y, cfg)
        win_full = models[sel_full["winner"]]
        entry = {"description": desc, "dates": [str(d.date()) for d in y.index], "values": r(y.values),
                 "cutoff_index": len(train) - 1, "holdout": {}, "future": {}, "future_dates": None, "rolling_backtest": {},
                 "holdout_band": {}, "future_band": {},
                 "selection": {"holdout": {"winner": sel_train["winner"], "leaderboard": _board(sel_train)},
                               "future": {"winner": sel_full["winner"], "leaderboard": _board(sel_full)}},
                 "model_version": f"{win_full.name}-{win_full.version_hash()}"}
        for key, mc in models.items():
            make = lambda mc=mc: build_model(mc)  # noqa: E731
            hold = make().fit(train).predict(H)
            entry["holdout"][key] = r(hold.values)
            # Holdout band is calibrated on the training window only, so its coverage is an honest test.
            hb = calibrate(train, make, H, cfg.interval_level, cfg.interval_origins).apply(hold)
            entry["holdout_band"][key] = {"lower": r(hb.lower), "upper": r(hb.upper)}
            fut = make().fit(y).predict(H)
            entry["future"][key] = r(fut.values)
            fb = calibrate(y, make, H, cfg.interval_level, cfg.interval_origins).apply(fut)
            entry["future_band"][key] = {"lower": r(fb.lower), "upper": r(fb.upper)}
            entry["future_dates"] = [str(d.date()) for d in fut.index]
            bt = backtest(y, make, cfg.horizon, cfg.folds)
            entry["rolling_backtest"][key] = {"mape": round(bt["mape"], 2), "smape": round(bt["smape"], 2),
                                              "rmse": round(bt["rmse"], 2)}
        payload["series"][name] = entry
    out.write_text(json.dumps(payload, separators=(",", ":")))
    return payload


if __name__ == "__main__":
    p = main()
    print(f"wrote docs/data.json for {len(p['series'])} series")
    for name, s in p["series"].items():
        print(f"  {name}: holdout pick={s['selection']['holdout']['winner']}, production pick={s['selection']['future']['winner']}")
