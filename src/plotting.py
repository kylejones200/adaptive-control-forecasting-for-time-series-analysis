"""Plotting utilities for adaptive forecasting demos."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _apply_clean_style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def plot_adaptive_smoothing_comparison(
    y: np.ndarray,
    fixed_forecast: np.ndarray,
    adaptive_result: dict[str, np.ndarray],
    *,
    regime_change_at: int | None = None,
    output_path: Path,
    title: str = "Fixed vs Adaptive Exponential Smoothing",
) -> None:
    """Plot observed data, fixed and adaptive forecasts, and alpha over time."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1, ax2 = axes
    ax1.plot(y, color="#444444", linewidth=1.5, label="Observed")
    ax1.plot(fixed_forecast, color="#d62728", linewidth=1.5, label="Fixed α")
    ax1.plot(
        adaptive_result["forecasts"],
        color="#1f77b4",
        linewidth=1.8,
        label="Adaptive α",
    )
    if regime_change_at is not None:
        ax1.axvline(
            regime_change_at,
            color="#888888",
            linestyle=":",
            linewidth=1.2,
            label="Regime change",
        )
    ax1.set_title(title)
    ax1.set_ylabel("Value")
    ax1.legend(frameon=False, loc="upper left")
    _apply_clean_style(ax1)

    ax2.plot(adaptive_result["alphas"], color="#1f77b4", linewidth=1.8)
    ax2.set_ylabel("α")
    ax2.set_xlabel("Time")
    if regime_change_at is not None:
        ax2.axvline(regime_change_at, color="#888888", linestyle=":", linewidth=1.2)
    _apply_clean_style(ax2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_saft_demo(
    y: np.ndarray,
    saft_result: dict[str, np.ndarray],
    *,
    regime_change_at: int | None = None,
    output_path: Path,
) -> None:
    """Plot SAFT forecasts, errors, and model mode switches."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    ax1, ax2, ax3 = axes
    ax1.plot(y, color="#444444", linewidth=1.5, label="Observed")
    ax1.plot(saft_result["forecasts"], color="#2ca02c", linewidth=1.8, label="SAFT forecast")
    if regime_change_at is not None:
        ax1.axvline(regime_change_at, color="#888888", linestyle=":", linewidth=1.2)
    ax1.set_title("SAFT: Self-Adaptive Forecasting with Model Switching")
    ax1.set_ylabel("Value")
    ax1.legend(frameon=False, loc="upper left")
    _apply_clean_style(ax1)

    ax2.plot(np.abs(saft_result["errors"]), color="#d62728", linewidth=1.2)
    ax2.set_ylabel("|Error|")
    _apply_clean_style(ax2)

    ax3.step(
        np.arange(len(saft_result["modes"])),
        saft_result["modes"],
        where="post",
        color="#9467bd",
        linewidth=1.5,
    )
    ax3.set_yticks([0, 1])
    ax3.set_yticklabels(["Stable", "Responsive"])
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Mode")
    if regime_change_at is not None:
        ax3.axvline(regime_change_at, color="#888888", linestyle=":", linewidth=1.2)
    _apply_clean_style(ax3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_holt_winters_adaptive(
    y: np.ndarray,
    hw_result: dict[str, np.ndarray],
    *,
    output_path: Path,
) -> None:
    """Plot adaptive Holt-Winters forecasts and parameter traces."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax1, ax2 = axes
    ax1.plot(y, color="#444444", linewidth=1.5, label="Observed")
    ax1.plot(hw_result["forecasts"], color="#ff7f0e", linewidth=1.8, label="Adaptive HW")
    ax1.set_title("Adaptive Holt-Winters Forecast")
    ax1.set_ylabel("Value")
    ax1.legend(frameon=False, loc="upper left")
    _apply_clean_style(ax1)

    ax2.plot(hw_result["alphas"], label="α (level)", linewidth=1.5)
    ax2.plot(hw_result["betas"], label="β (trend)", linewidth=1.5)
    ax2.plot(hw_result["gammas"], label="γ (seasonal)", linewidth=1.5)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Parameter value")
    ax2.legend(frameon=False, loc="upper right")
    _apply_clean_style(ax2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
