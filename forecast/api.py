"""FastAPI service. Run: uvicorn forecast.api:app --port 8000"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__, registry
from .config import load_config
from .data import SERIES, load_series
from .intervals import Calibration, calibrate
from .models import build_model

CFG = load_config()
app = FastAPI(title="forecast-poc-to-prod", version=__version__)


class ForecastRequest(BaseModel):
    series: str = Field(examples=["retail_sales"])
    horizon: int = Field(12, ge=1, le=CFG.max_horizon)


class Point(BaseModel):
    date: str
    value: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    series: str
    model_version: str
    horizon: int
    interval_level: float
    forecast: list[Point]


def _calibrate(series: str) -> Calibration:
    return calibrate(load_series(series), lambda: build_model(CFG.model), CFG.max_horizon,
                     CFG.interval_level, CFG.interval_origins)


@lru_cache(maxsize=None)
def get_model(series: str):
    """Prefer the registered artifact; fall back to fitting from config (dev mode).
    Returns (model, version, interval calibration)."""
    try:
        model, meta = registry.load_latest(series)
        calib = Calibration.from_dict(meta["intervals"]) if "intervals" in meta else _calibrate(series)
        return model, meta["version"], calib
    except FileNotFoundError:
        m = build_model(CFG.model).fit(load_series(series))
        return m, f"{CFG.model.name}-{CFG.model.version_hash()}", _calibrate(series)


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/series")
def list_series():
    return SERIES


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if req.series not in SERIES:
        raise HTTPException(404, f"unknown series {req.series!r}")
    model, version, calib = get_model(req.series)
    band = calib.apply(model.predict(req.horizon))
    return ForecastResponse(
        series=req.series, model_version=version, horizon=req.horizon, interval_level=calib.level,
        forecast=[Point(date=str(d.date()), value=round(r.point, 2), lower=round(r.lower, 2), upper=round(r.upper, 2))
                  for d, r in band.iterrows()])
