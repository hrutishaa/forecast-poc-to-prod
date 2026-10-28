# Forecast: PoC → Production

[![CI](https://github.com/hrutishaa/forecast-poc-to-prod/actions/workflows/ci.yml/badge.svg)](https://github.com/hrutishaa/forecast-poc-to-prod/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**🔴 Live demo: https://hrutishaa.github.io/forecast-poc-to-prod/** (pick a series, drag the horizon, compare baseline vs model against actuals)

Most forecasting projects stop at a notebook with a nice chart. This repo goes further: it takes that notebook to something you can deploy, with
**a baseline you have to beat → leak-free features → rolling backtests → versioned config → model registry → FastAPI `/forecast` → Docker → CI**.

| Series (monthly, 96 pts) | Seasonal-naive MAPE | GBM + lags MAPE | Error reduction |
|---|---|---|---|
| retail_sales | 4.61% | **1.34%** | −71% |
| ecommerce_orders | 13.16% | **8.73%** | −34% |
| energy_demand | 1.95% | **1.94%** | ≈0%: on a stable seasonal series the baseline is already hard to beat, and the backtest shows that honestly |

*3-fold rolling-origin backtest, 12-month horizon. These numbers are recomputed in CI.*

## Architecture

```mermaid
flowchart LR
    subgraph PoC
      NB[notebooks/01_poc_exploration.ipynb]
    end
    NB -->|promote what worked| PKG
    subgraph PKG[forecast/ package]
      D[data.py<br/>load_series] --> F[features.py<br/>lags, rolling means, calendar<br/>shared by train + serve]
      F --> M[models.py<br/>SeasonalNaive baseline<br/>GBMForecaster recursive]
      M --> B[backtest.py<br/>rolling-origin MAPE / RMSE]
      CFG[config.toml] -->|version = hash of config| T[train.py]
      B --> T --> R[(registry<br/>artifacts/series/version)]
    end
    R --> API[FastAPI<br/>POST /forecast]
    API --> DK[Docker image]
    T --> EX[scripts/export_demo.py] --> GP[GitHub Pages<br/>live demo]
    CI[GitHub Actions] -.tests, train, export, docker smoke test.-> PKG
```

### Design decisions
- **Baseline first.** Seasonal naive gets the same backtest as the model. If the model doesn't beat it, the model doesn't ship (`beats_baseline` is saved in the model's metadata).
- **No train/serve skew.** Training and recursive inference both call the same `make_features`, and a unit test checks that no feature can see the value it's predicting.
- **Trees can't extrapolate trends,** so the GBM learns *first differences* and the predictions are added back up to levels.
- **Versioning by config hash.** `gbm_lags-d7811b8175` means "this exact config". Retraining with the same config overwrites the same version, and any config change creates a new version alongside the old one, which makes rollback easy.
- **Free, instant demo.** The static page reads results that the pipeline precomputed (`docs/data.json`), so there are no servers or keys to keep running.

## Quickstart

```bash
git clone https://github.com/hrutishaa/forecast-poc-to-prod && cd forecast-poc-to-prod
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest                                                  # 20 tests
python -m forecast.train                                # backtest + register all series
uvicorn forecast.api:app --reload                       # http://localhost:8000/docs
```

```bash
curl -X POST localhost:8000/forecast -H 'Content-Type: application/json' \
  -d '{"series": "retail_sales", "horizon": 12}'
```

With Docker:

```bash
docker build -t forecast . && docker run -p 8000:8000 forecast
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | liveness + app version |
| GET | `/series` | available series |
| POST | `/forecast` | `{"series": str, "horizon": 1..24}` → dated forecast + `model_version` |

## Project layout

```
forecast/      config, data, features, models, backtest, registry, train, api
tests/         20 pytest tests (leakage, metrics, models, registry, API, demo export)
notebooks/     the original PoC notebook (the "before" picture)
scripts/       export_demo.py → docs/data.json
docs/          static live demo (GitHub Pages)
config.toml    the one place to change model hyperparameters
```

The sample data is synthetic: seeded trend, seasonality and noise, so every result can be reproduced. To use real data, replace `load_series` with a warehouse query.

See [ROADMAP.md](ROADMAP.md) for what's next.

## License

MIT © 2026 Hrutishaa Nanduti
