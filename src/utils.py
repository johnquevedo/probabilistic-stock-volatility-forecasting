from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


@dataclass(slots=True)
class ProjectPaths:
    root: Path
    artifacts_dir: Path
    cache_dir: Path


def project_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration from disk."""
    config_path = Path(path) if path else project_root() / "configs" / "default.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_paths(config: dict[str, Any]) -> ProjectPaths:
    """Resolve and create project directories needed by the pipeline."""
    root = project_root()
    artifacts_dir = root / config["artifacts"]["dir"]
    cache_dir = root / config["data"]["cache_dir"]
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return ProjectPaths(root=root, artifacts_dir=artifacts_dir, cache_dir=cache_dir)


def select_device(device_name: str = "auto") -> torch.device:
    """Select a compute device, preferring CUDA when configured and available."""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def business_day_range(start: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    """Return a business-day index used for forward forecasts."""
    return pd.bdate_range(start=start, periods=periods)


def safe_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling z-score while guarding against zero variance."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std().replace(0.0, np.nan)
    return (series - mean) / std


def json_ready_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert numpy scalar values so metrics can be serialized to JSON."""
    converted: list[dict[str, Any]] = []
    for record in records:
        converted.append(
            {
                key: (float(value) if isinstance(value, (np.floating, np.integer)) else value)
                for key, value in record.items()
            }
        )
    return converted
