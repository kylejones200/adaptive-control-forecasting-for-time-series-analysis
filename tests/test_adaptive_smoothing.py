"""Unit tests for adaptive forecasting methods."""

from __future__ import annotations

import numpy as np
import pytest

from src.adaptive_smoothing import (
    adaptive_exponential_smoothing,
    adaptive_holt_winters,
    fixed_exponential_smoothing,
    mean_absolute_error,
    saft_forecast,
)
from src.synthetic import generate_regime_shift_series, generate_seasonal_series


def test_adaptive_exponential_smoothing_returns_expected_keys() -> None:
    y = generate_regime_shift_series(n_periods=40, seed=1)
    result = adaptive_exponential_smoothing(y)
    assert set(result) == {"forecasts", "levels", "alphas", "errors"}
    assert len(result["forecasts"]) == len(y)


def test_adaptive_alpha_responds_to_regime_change() -> None:
    y = generate_regime_shift_series(
        n_periods=80,
        seed=2,
        regime_change_at=40,
        stable_noise=0.2,
        volatile_noise=3.0,
    )
    result = adaptive_exponential_smoothing(y, gamma=0.2)
    early_alpha = result["alphas"][20:40].mean()
    late_alpha = result["alphas"][50:70].mean()
    assert late_alpha > early_alpha


def test_adaptive_beats_fixed_on_regime_shift() -> None:
    y = generate_regime_shift_series(
        n_periods=100,
        seed=3,
        regime_change_at=50,
        stable_noise=0.3,
        volatile_noise=2.5,
    )
    fixed = fixed_exponential_smoothing(y, alpha=0.2)
    adaptive = adaptive_exponential_smoothing(y, gamma=0.15)
    assert mean_absolute_error(y, adaptive["forecasts"]) <= mean_absolute_error(y, fixed)


def test_saft_switches_to_responsive_mode() -> None:
    y = generate_regime_shift_series(
        n_periods=80,
        seed=4,
        regime_change_at=40,
        stable_noise=0.2,
        volatile_noise=3.0,
    )
    result = saft_forecast(
        y,
        stable_alpha=0.1,
        responsive_alpha=0.7,
        error_window=5,
        spike_threshold=1.2,
    )
    assert result["modes"][50:].mean() > result["modes"][10:30].mean()


def test_adaptive_holt_winters_requires_enough_data() -> None:
    y = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="Need at least"):
        adaptive_holt_winters(y, season_length=12)


def test_adaptive_holt_winters_on_seasonal_data() -> None:
    y = generate_seasonal_series(n_periods=60, season_length=12, seed=5)
    result = adaptive_holt_winters(y, season_length=12)
    assert len(result["forecasts"]) == len(y)
    assert result["alphas"].min() >= 0.05
    assert result["alphas"].max() <= 0.95
