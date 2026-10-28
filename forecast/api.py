"""FastAPI service. Run: uvicorn forecast.api:app --port 8000"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__, registry
from .config import load_config
from .data import SERIES, load_series
from .models import build_model

CFG = load_config()
app = FastAPI(title="forecast-poc-to-prod", version=__version__)


class ForecastRequest(BaseModel):
    series: str = Field(examples=["retail_sales"])
    horizon: int = Field(12, ge=1, le=CFG.max_horizon)


class Point(BaseModel):
    date: str
    value: float


class ForecastResponse(BaseModel):
    series: str
    model_version: str
    horizon: int
    forecast: list[Point]


@lru_cache(maxsize=None)
def get_model(series: str):
    """Prefer the registered artifact; fall back to fitting from config (dev mode)."""
    try:
        model, meta = registry.load_latest(series)
        return model, meta["version"]
    except FileNotFoundError:
        m = build_model(CFG.model).fit(load_series(series))
        return m, f"{CFG.model.name}-{CFG.model.version_hash()}"


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
    model, version = get_model(req.series)
    pred = model.predict(req.horizon)
    return ForecastResponse(series=req.series, model_version=version, horizon=req.horizon,
                            forecast=[Point(date=str(d.date()), value=round(float(v), 2)) for d, v in pred.items()])
