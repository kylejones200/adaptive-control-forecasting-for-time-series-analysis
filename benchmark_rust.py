#!/usr/bin/env python3
"""Python vs Rust benchmark — showcase adaptive smoothing speed."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from adaptive_smoothing import adaptive_exponential_smoothing  # noqa: E402
from synthetic import generate_regime_shift_series  # noqa: E402


def bench_python(y: np.ndarray, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        adaptive_exponential_smoothing(y)
    return time.perf_counter() - start


def bench_rust(y: np.ndarray, iterations: int) -> float:
    import adaptive_forecast_rs

    return adaptive_forecast_rs.bench_adaptive_py(y, iterations)


def fmt_speedup(python_s: float, rust_s: float) -> str:
    if rust_s <= 0:
        return "n/a"
    return f"{python_s / rust_s:.1f}x"


def main() -> None:
    parser = argparse.ArgumentParser(description="Python vs Rust adaptive smoothing benchmark")
    parser.add_argument("--n", type=int, default=50_000, help="series length")
    parser.add_argument("--iterations", type=int, default=500, help="repeat count")
    args = parser.parse_args()

    y = generate_regime_shift_series(n_periods=args.n, seed=42)

    print("Adaptive exponential smoothing — Python vs Rust")
    print(f"  series length : {args.n:>12,}")
    print(f"  iterations    : {args.iterations:>12,}")
    print()

    # Warmup
    adaptive_exponential_smoothing(y[:1000])
    try:
        import adaptive_forecast_rs  # noqa: F401

        adaptive_forecast_rs.adaptive_exponential_smoothing_py(y[:1000])
    except ImportError:
        print("Rust extension not built. Run:")
        print("  cd rust && maturin develop --release -m adaptive-py/Cargo.toml")
        print()
        print("Running Python-only timing...")
        py_s = bench_python(y, args.iterations)
        total = args.n * args.iterations
        print(f"  Python total  : {py_s:>10.3f} s")
        print(f"  Python        : {total / py_s / 1e6:>10.2f} M forecasts/s")
        return

    py_s = bench_python(y, args.iterations)
    rs_s = bench_rust(y, args.iterations)

    total = args.n * args.iterations
    print(f"  {'':14}  {'total (s)':>10}  {'M forecasts/s':>14}")
    print(f"  {'Python':14}  {py_s:>10.3f}  {total / py_s / 1e6:>14.2f}")
    print(f"  {'Rust (PyO3)':14}  {rs_s:>10.3f}  {total / rs_s / 1e6:>14.2f}")
    print()
    print(f"  Speedup       : {fmt_speedup(py_s, rs_s)}")
    print()

    # Correctness check
    py_out = adaptive_exponential_smoothing(y[:500])
    rs_out = adaptive_forecast_rs.adaptive_exponential_smoothing_py(y[:500])
    np.testing.assert_allclose(py_out["forecasts"], rs_out["forecasts"], rtol=1e-12)
    np.testing.assert_allclose(py_out["alphas"], rs_out["alphas"], rtol=1e-12)
    print("  Correctness   : Python and Rust outputs match (rtol=1e-12)")


if __name__ == "__main__":
    main()
