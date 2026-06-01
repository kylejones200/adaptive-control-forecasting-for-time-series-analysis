"""Adaptive control forecasting methods for time series."""

from __future__ import annotations

import numpy as np


def _clip_alpha(alpha: float, alpha_min: float, alpha_max: float) -> float:
    return float(np.clip(alpha, alpha_min, alpha_max))


def adaptive_exponential_smoothing(
    y: np.ndarray,
    *,
    alpha_init: float = 0.2,
    gamma: float = 0.1,
    alpha_min: float = 0.05,
    alpha_max: float = 0.95,
) -> dict[str, np.ndarray]:
    """Trigg-Leach style adaptive exponential smoothing.

    The smoothing parameter alpha_t adapts to forecast errors: it rises when
    errors increase (more responsive) and falls when errors stabilize.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        raise ValueError("Series must not be empty.")

    forecasts = np.zeros(n)
    levels = np.zeros(n)
    alphas = np.zeros(n)
    errors = np.zeros(n)

    level = y[0]
    pe = abs(y[1] - y[0]) if n > 1 else 0.0
    fe = pe
    alphas[0] = alpha_init
    levels[0] = level
    forecasts[0] = level

    for t in range(1, n):
        alpha = pe / (pe + fe) if (pe + fe) > 0 else alpha_init
        alpha = _clip_alpha(alpha, alpha_min, alpha_max)

        forecast = level
        forecasts[t] = forecast
        error = y[t] - forecast
        errors[t] = error
        level = level + alpha * error
        pe = gamma * abs(error) + (1.0 - gamma) * pe
        fe = gamma * abs(y[t] - level) + (1.0 - gamma) * fe

        alphas[t] = alpha
        levels[t] = level

    return {
        "forecasts": forecasts,
        "levels": levels,
        "alphas": alphas,
        "errors": errors,
    }


def fixed_exponential_smoothing(y: np.ndarray, *, alpha: float = 0.2) -> np.ndarray:
    """Simple exponential smoothing with a fixed alpha for comparison."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    forecasts = np.zeros(n)
    level = y[0]
    forecasts[0] = level
    for t in range(1, n):
        forecast = level
        forecasts[t] = forecast
        level = level + alpha * (y[t] - level)
    return forecasts


def adaptive_holt_winters(
    y: np.ndarray,
    *,
    season_length: int = 12,
    alpha_init: float = 0.2,
    beta_init: float = 0.05,
    gamma_init: float = 0.1,
    learning_rate: float = 0.05,
    alpha_min: float = 0.05,
    alpha_max: float = 0.95,
) -> dict[str, np.ndarray]:
    """Additive Holt-Winters with error-driven parameter updates.

    Alpha, beta, and gamma each move toward more responsive values when
    one-step-ahead errors are large.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2 * season_length:
        raise ValueError(
            f"Need at least {2 * season_length} observations for season_length={season_length}."
        )

    forecasts = np.zeros(n)
    alphas = np.full(n, alpha_init)
    betas = np.full(n, beta_init)
    gammas = np.full(n, gamma_init)

    level = y[:season_length].mean()
    trend = (y[season_length : 2 * season_length].mean() - level) / season_length
    seasonals = np.array(
        [y[i] - level for i in range(season_length)],
        dtype=float,
    )

    alpha, beta, gamma = alpha_init, beta_init, gamma_init

    for t in range(n):
        s_idx = t % season_length
        forecast = level + trend + seasonals[s_idx]
        forecasts[t] = forecast

        if t == 0:
            continue

        error = y[t] - forecast
        scale = max(abs(y[t]), 1.0)
        adjustment = learning_rate * abs(error) / scale

        alpha = _clip_alpha(alpha + adjustment, alpha_min, alpha_max)
        beta = _clip_alpha(beta + adjustment * 0.5, 0.01, 0.3)
        gamma = _clip_alpha(gamma + adjustment * 0.5, 0.01, 0.5)

        prev_level = level
        level = alpha * (y[t] - seasonals[s_idx]) + (1.0 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1.0 - beta) * trend
        seasonals[s_idx] = gamma * (y[t] - level) + (1.0 - gamma) * seasonals[s_idx]

        alphas[t] = alpha
        betas[t] = beta
        gammas[t] = gamma

    return {
        "forecasts": forecasts,
        "alphas": alphas,
        "betas": betas,
        "gammas": gammas,
    }


def saft_forecast(
    y: np.ndarray,
    *,
    stable_alpha: float = 0.15,
    responsive_alpha: float = 0.6,
    error_window: int = 5,
    spike_threshold: float = 1.5,
) -> dict[str, np.ndarray]:
    """Self-Adaptive Forecasting Technique (SAFT) with model switching.

    Monitors rolling forecast error. When errors spike, switches to a more
    responsive smoothing model; when errors stabilize, returns to stable mode.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        raise ValueError("Series must not be empty.")

    forecasts = np.zeros(n)
    errors = np.zeros(n)
    modes = np.zeros(n, dtype=int)  # 0 = stable, 1 = responsive
    alphas = np.zeros(n)

    level = y[0]
    forecasts[0] = level
    alphas[0] = stable_alpha
    abs_errors: list[float] = []

    for t in range(1, n):
        recent = abs_errors[-error_window:] if abs_errors else [0.0]
        baseline = float(np.mean(recent)) if recent else 0.0
        last_error = abs_errors[-1] if abs_errors else 0.0

        responsive = baseline > 0 and last_error > spike_threshold * baseline
        alpha = responsive_alpha if responsive else stable_alpha
        modes[t] = 1 if responsive else 0
        alphas[t] = alpha

        forecast = level
        forecasts[t] = forecast
        error = y[t] - forecast
        errors[t] = error
        level = level + alpha * error
        abs_errors.append(abs(error))

    return {
        "forecasts": forecasts,
        "errors": errors,
        "modes": modes,
        "alphas": alphas,
    }


def mean_absolute_error(actual: np.ndarray, forecast: np.ndarray) -> float:
    """One-step-ahead MAE, skipping the first observation."""
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if len(actual) != len(forecast):
        raise ValueError("actual and forecast must have the same length.")
    return float(np.mean(np.abs(actual[1:] - forecast[1:])))
