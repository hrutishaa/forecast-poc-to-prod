"""Minimal model registry: artifacts/<series>/<version>/{model.pkl, meta.json}.

version = hash of the model config, so a retrain with identical config is idempotent and
any config change produces a new, side-by-side version (easy rollback)."""
from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT

ARTIFACTS = ROOT / "artifacts"


def save(model, series: str, version: str, metrics: dict, root: Path = ARTIFACTS) -> Path:
    d = root / series / version
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    meta = {"series": series, "version": version, "metrics": metrics,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    (root / series / "LATEST").write_text(version)
    return d


def load_latest(series: str, root: Path = ARTIFACTS):
    """Only load artifacts you produced yourself: pickle is not safe for untrusted files."""
    version = (root / series / "LATEST").read_text().strip()
    d = root / series / version
    with open(d / "model.pkl", "rb") as fh:
        model = pickle.load(fh)
    return model, json.loads((d / "meta.json").read_text())
