# Changelog

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
