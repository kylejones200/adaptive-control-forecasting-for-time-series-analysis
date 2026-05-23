"""Synthetic series with a regime change for adaptive method demos."""

from __future__ import annotations

import numpy as np


def generate_regime_shift_series(
    n_periods: int = 120,
    *,
    seed: int = 42,
    regime_change_at: int = 60,
    stable_noise: float = 0.5,
    volatile_noise: float = 2.5,
) -> np.ndarray:
    """Generate a series with stable then volatile regimes.

    The first segment follows a smooth upward trend with low noise.
    After the regime change, noise increases and the local slope shifts.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_periods, dtype=float)
    change_at = min(regime_change_at, max(1, n_periods // 2))

    level = 100.0 + 0.3 * t
    noise = np.empty(n_periods)
    noise[:change_at] = rng.normal(0.0, stable_noise, change_at)
    noise[change_at:] = rng.normal(0.0, volatile_noise, n_periods - change_at)

    slope_shift = np.zeros(n_periods)
    slope_shift[change_at:] = 0.5 * (t[change_at:] - change_at)

    return level + slope_shift + noise


def generate_seasonal_series(
    n_periods: int = 120,
    *,
    season_length: int = 12,
    seed: int = 42,
) -> np.ndarray:
    """Generate a seasonal series for Holt-Winters demos."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_periods, dtype=float)
    trend = 50.0 + 0.4 * t
    seasonal = 8.0 * np.sin(2.0 * np.pi * t / season_length)
    noise = rng.normal(0.0, 1.0, n_periods)
    return trend + seasonal + noise
