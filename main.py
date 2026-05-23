#!/usr/bin/env python3
"""Adaptive control forecasting demos for time series analysis."""

from __future__ import annotations

import logging
from pathlib import Path

from src import ensure_output_dir, load_config
from src.adaptive_smoothing import (
    adaptive_exponential_smoothing,
    adaptive_holt_winters,
    fixed_exponential_smoothing,
    mean_absolute_error,
    saft_forecast,
)
from src.plotting import (
    plot_adaptive_smoothing_comparison,
    plot_holt_winters_adaptive,
    plot_saft_demo,
)
from src.synthetic import generate_regime_shift_series, generate_seasonal_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_adaptive_smoothing_demo(config: dict, output_dir: Path) -> None:
    """Compare fixed and adaptive exponential smoothing on a regime-shift series."""
    data_cfg = config["data"]
    smooth_cfg = config["adaptive_smoothing"]

    y = generate_regime_shift_series(
        n_periods=data_cfg["n_periods"],
        seed=data_cfg["seed"],
        regime_change_at=data_cfg["regime_change_at"],
        stable_noise=data_cfg["stable_noise"],
        volatile_noise=data_cfg["volatile_noise"],
    )

    fixed = fixed_exponential_smoothing(y, alpha=smooth_cfg["alpha_init"])
    adaptive = adaptive_exponential_smoothing(
        y,
        alpha_init=smooth_cfg["alpha_init"],
        gamma=smooth_cfg["gamma"],
        alpha_min=smooth_cfg["alpha_min"],
        alpha_max=smooth_cfg["alpha_max"],
    )

    fixed_mae = mean_absolute_error(y, fixed)
    adaptive_mae = mean_absolute_error(y, adaptive["forecasts"])
    logger.info("Fixed exponential smoothing MAE: %.3f", fixed_mae)
    logger.info("Adaptive exponential smoothing MAE: %.3f", adaptive_mae)

    plot_adaptive_smoothing_comparison(
        y,
        fixed,
        adaptive,
        regime_change_at=data_cfg["regime_change_at"],
        output_path=output_dir / "adaptive_exponential_smoothing.png",
    )


def run_saft_demo(config: dict, output_dir: Path) -> None:
    """Demonstrate SAFT model switching on a regime-shift series."""
    data_cfg = config["data"]
    saft_cfg = config["saft"]

    y = generate_regime_shift_series(
        n_periods=data_cfg["n_periods"],
        seed=data_cfg["seed"],
        regime_change_at=data_cfg["regime_change_at"],
        stable_noise=data_cfg["stable_noise"],
        volatile_noise=data_cfg["volatile_noise"],
    )

    result = saft_forecast(
        y,
        stable_alpha=saft_cfg["stable_alpha"],
        responsive_alpha=saft_cfg["responsive_alpha"],
        error_window=saft_cfg["error_window"],
        spike_threshold=saft_cfg["spike_threshold"],
    )
    saft_mae = mean_absolute_error(y, result["forecasts"])
    logger.info("SAFT MAE: %.3f", saft_mae)
    logger.info(
        "SAFT responsive mode used %.1f%% of steps",
        100.0 * result["modes"].mean(),
    )

    plot_saft_demo(
        y,
        result,
        regime_change_at=data_cfg["regime_change_at"],
        output_path=output_dir / "saft_model_switching.png",
    )


def run_holt_winters_demo(config: dict, output_dir: Path) -> None:
    """Demonstrate adaptive Holt-Winters on seasonal synthetic data."""
    data_cfg = config["data"]
    hw_cfg = config["holt_winters"]
    season_length = hw_cfg["season_length"]

    y = generate_seasonal_series(
        n_periods=max(data_cfg["n_periods"], 2 * season_length + 1),
        season_length=season_length,
        seed=data_cfg["seed"],
    )

    result = adaptive_holt_winters(
        y,
        season_length=season_length,
        alpha_init=hw_cfg["alpha_init"],
        beta_init=hw_cfg["beta_init"],
        gamma_init=hw_cfg["gamma_init"],
        learning_rate=hw_cfg["learning_rate"],
        alpha_min=config["adaptive_smoothing"]["alpha_min"],
        alpha_max=config["adaptive_smoothing"]["alpha_max"],
    )
    hw_mae = mean_absolute_error(y, result["forecasts"])
    logger.info("Adaptive Holt-Winters MAE: %.3f", hw_mae)

    plot_holt_winters_adaptive(
        y,
        result,
        output_path=output_dir / "adaptive_holt_winters.png",
    )


def main() -> None:
    """Run all adaptive forecasting demos."""
    config = load_config()
    output_dir = ensure_output_dir(config)

    run_adaptive_smoothing_demo(config, output_dir)
    run_saft_demo(config, output_dir)
    run_holt_winters_demo(config, output_dir)

    logger.info("Analysis complete. Figures saved to %s", output_dir)


if __name__ == "__main__":
    main()
