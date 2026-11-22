"""FastAPI service. Run: uvicorn forecast.api:app --port 8000"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__, registry
from .config import load_config
from .data import SERIES, load_series
from .intervals import Calibration, calibrate
from .models import build_model
from .selection import select_model
from .train import resolve_model

CFG = load_config()
app = FastAPI(title="forecast-poc-to-prod", version=__version__)

ModelName = Literal["auto", "seasonal_naive", "gbm_lags", "ets_simple", "ets_holt", "ets_hw_add"]


class ForecastRequest(BaseModel):
    series: str = Field(examples=["retail_sales"])
    horizon: int = Field(12, ge=1, le=CFG.max_horizon)
    model: ModelName = Field("auto", description="'auto' = the model that won the backtest for this series")


class Point(BaseModel):
    date: str
    value: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    series: str
    model: str
    model_version: str
    horizon: int
    interval_level: float
    forecast: list[Point]


def _calibrate(series: str, mc) -> Calibration:
    return calibrate(load_series(series), lambda: build_model(mc), CFG.max_horizon,
                     CFG.interval_level, CFG.interval_origins)


@lru_cache(maxsize=None)
def get_model(series: str, model: str = "auto"):
    """Returns (fitted model, model name, version, interval calibration).

    'auto' prefers the registered artifact (trained + selected by forecast.train); without one
    (dev mode) or for an explicitly requested model, fit from config and calibrate on the fly."""
    if model == "auto":
        try:
            fitted, meta = registry.load_latest(series)
            mc = replace(CFG.model, name=meta["version"].rsplit("-", 1)[0])
            calib = Calibration.from_dict(meta["intervals"]) if "intervals" in meta else _calibrate(series, mc)
            return fitted, mc.name, meta["version"], calib
        except FileNotFoundError:
            pass
    mc, _ = resolve_model(load_series(series), CFG, model)
    fitted = build_model(mc).fit(load_series(series))
    return fitted, mc.name, f"{mc.name}-{mc.version_hash()}", _calibrate(series, mc)


@lru_cache(maxsize=None)
def _selection(series: str) -> dict:
    try:
        _, meta = registry.load_latest(series)
        if "selection" in meta["metrics"]:
            return meta["metrics"]["selection"]
    except FileNotFoundError:
        pass
    return select_model(load_series(series), CFG)


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/series")
def list_series():
    return SERIES


@app.get("/models/{series}")
def model_leaderboard(series: str):
    """Backtest leaderboard used by automatic model selection for this series."""
    if series not in SERIES:
        raise HTTPException(404, f"unknown series {series!r}")
    return {"series": series, **_selection(series)}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if req.series not in SERIES:
        raise HTTPException(404, f"unknown series {req.series!r}")
    model, name, version, calib = get_model(req.series, req.model)
    band = calib.apply(model.predict(req.horizon))
    return ForecastResponse(
        series=req.series, model=name, model_version=version, horizon=req.horizon, interval_level=calib.level,
        forecast=[Point(date=str(d.date()), value=round(r.point, 2), lower=round(r.lower, 2), upper=round(r.upper, 2))
                  for d, r in band.iterrows()])
