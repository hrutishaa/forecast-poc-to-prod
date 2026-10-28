from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelConfig:
    name: str = "gbm_lags"
    season_length: int = 12
    lags: tuple[int, ...] = (1, 2, 3, 6, 12)
    rolling_windows: tuple[int, ...] = (3, 12)
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 3
    random_state: int = 42

    def version_hash(self) -> str:
        """Short, deterministic hash: same config -> same model version id."""
        blob = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:10]


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    folds: int = 3
    horizon: int = 12
    max_horizon: int = 24


def load_config(path: str | Path = ROOT / "config.toml") -> Config:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    m = raw.get("model", {})
    for k in ("lags", "rolling_windows"):
        if k in m:
            m[k] = tuple(m[k])
    bt, sv = raw.get("backtest", {}), raw.get("serving", {})
    return Config(model=ModelConfig(**m), folds=bt.get("folds", 3), horizon=bt.get("horizon", 12),
                  max_horizon=sv.get("max_horizon", 24))
