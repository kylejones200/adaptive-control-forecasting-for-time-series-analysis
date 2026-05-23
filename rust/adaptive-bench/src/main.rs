//! Standalone benchmark — no Python required.
//!
//! ```bash
//! cargo run --release -p adaptive-bench -- 100_000 50_000
//! ```

use adaptive_core::adaptive_exponential_smoothing;
use std::env;
use std::time::Instant;

fn generate_regime_shift(n: usize, seed: u64) -> Vec<f64> {
  // Simple LCG — deterministic, no external deps.
    let mut state = seed;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (state >> 33) as f64 / (1u64 << 31) as f64
    };

    let change_at = n / 2;
    (0..n)
        .map(|t| {
            let level = 100.0 + 0.3 * t as f64;
            let noise_std = if t < change_at { 0.5 } else { 2.5 };
            let u1 = next();
            let u2 = next();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            level + z * noise_std
        })
        .collect()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10_000);

    let y = generate_regime_shift(n, 42);
    println!("Rust adaptive exponential smoothing benchmark");
    println!("  series length : {n:>12}");
    println!("  iterations    : {iterations:>12}");
    println!();

    // Warmup
    for _ in 0..3 {
        let _ = adaptive_exponential_smoothing(&y, 0.2, 0.1, 0.05, 0.95).unwrap();
    }

    let start = Instant::now();
    let mut last_alpha = 0.0;
    for _ in 0..iterations {
        let result = adaptive_exponential_smoothing(&y, 0.2, 0.1, 0.05, 0.95).unwrap();
        last_alpha = *result.alphas.last().unwrap_or(&0.0);
    }
    let elapsed = start.elapsed();

    let total_forecasts = n as u64 * iterations as u64;
    let per_iter_us = elapsed.as_secs_f64() * 1e6 / iterations as f64;
    let throughput = total_forecasts as f64 / elapsed.as_secs_f64();

    println!("  total time    : {:>10.3} s", elapsed.as_secs_f64());
    println!("  per iteration : {:>10.1} µs", per_iter_us);
    println!(
        "  throughput    : {:>10.2} M forecasts/s",
        throughput / 1_000_000.0
    );
  // Prevent LLVM from optimizing away the computation.
    println!("  checksum      : {last_alpha:.6}");
}
