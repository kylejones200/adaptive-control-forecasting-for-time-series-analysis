"""Shared utilities for the adaptive control forecasting demo."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(config_path: Path | None = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def ensure_output_dir(config: dict) -> Path:
    out = Path(config.get("output", {}).get("figures_dir", "images"))
    out.mkdir(parents=True, exist_ok=True)
    return out
