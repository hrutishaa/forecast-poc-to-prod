# Roadmap

## v0.1: MVP ✅
- [x] Seasonal-naive baseline + gradient boosting with lag/rolling/calendar features
- [x] Rolling-origin backtests (MAPE, RMSE), `beats_baseline` gate
- [x] Config-hash model versioning + file-based registry
- [x] FastAPI `/forecast` with validation, Dockerfile, container smoke test in CI
- [x] Live demo on GitHub Pages using precomputed results

## Next
- [ ] Prediction intervals (quantile GBM or conformal) and showing them in the demo
- [ ] Add ETS / Holt-Winters (statsmodels) as a third candidate, with automatic model selection per series
- [ ] Real public dataset (e.g. US Census retail sales) behind the same `load_series` interface
- [ ] Data validation on load (missing months, outliers) with clear errors
- [ ] Drift monitoring: compare live errors with backtest errors and alert when they diverge
- [ ] Scheduled retraining workflow that opens a PR when a new version beats the current one
- [ ] MLflow tracking as an optional registry backend
- [ ] Deploy the API to a free tier (Render/Fly) and link it from the demo
