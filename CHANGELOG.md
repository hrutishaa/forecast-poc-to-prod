# Changelog

## [0.3.0] - 2026-11-21
### Added
- `forecast/ets.py`: exponential smoothing written from scratch: simple (SES), Holt linear trend and additive
  Holt-Winters, with smoothing weights fitted by minimising one-step-ahead squared error (grid start + L-BFGS-B).
- `forecast/selection.py`: automatic model selection. All 5 candidates get the same rolling backtest and the lowest
  MAPE (or sMAPE/RMSE) wins per series. CLI: `python -m forecast.selection`.
- sMAPE metric in every backtest.
- `[model] name = "auto"` + `[selection]` section in `config.toml`; `python -m forecast.train --model ...` to pin a model.
- The registry stores the selection leaderboard with the model; the API adds `GET /models/{series}` and an optional
  `model` field on `POST /forecast` (default `auto`), and the response now includes `model`.
- Live demo: all 5 models, the auto-selected model (chosen on the training window only) drawn with its 80% band,
  a sMAPE column and a selection leaderboard.
- 14 new tests (43 total).
### Changed
- Holt-Winters additive is auto-selected for all three demo series (backtest MAPE 0.95% / 4.54% / 1.54% for
  retail / e-commerce / energy, vs 1.34% / 8.73% / 1.94% for GBM). Production version is now `ets_hw_add-705b00b4ca`.

## [0.2.0] - 2026-11-19
### Added
- `forecast/intervals.py`: split conformal prediction intervals (per-step relative errors from 12 past origins,
  finite-sample conformal quantile, non-decreasing width), plus a `coverage` metric.
- `[intervals]` section in `config.toml` (`level = 0.8`, `n_origins = 12`).
- Training saves the interval calibration with each registered model; `/forecast` returns `lower`, `upper`
  and `interval_level` for every point.
- Live demo: shaded 80% band for the GBM model in both views and a holdout coverage KPI.
- 9 new tests (29 total).

## [0.1.0] - 2026-10-28
### Added
- MVP: seasonal-naive baseline, GBM with lag features, rolling-origin backtests, config-hash versioning,
  file registry, FastAPI `/forecast`, Docker, CI and the GitHub Pages live demo.
