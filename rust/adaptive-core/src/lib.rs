//! Adaptive control forecasting kernels — memory-safe, allocation-aware Rust.

use std::fmt;

/// Trigg-Leach style adaptive exponential smoothing result.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveResult {
    pub forecasts: Vec<f64>,
    pub levels: Vec<f64>,
    pub alphas: Vec<f64>,
    pub errors: Vec<f64>,
}

/// Self-Adaptive Forecasting Technique (SAFT) result.
#[derive(Debug, Clone, PartialEq)]
pub struct SaftResult {
    pub forecasts: Vec<f64>,
    pub errors: Vec<f64>,
    pub modes: Vec<u8>,
    pub alphas: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ForecastError {
    EmptySeries,
    LengthMismatch { expected: usize, got: usize },
    InsufficientData { need: usize, got: usize },
}

impl fmt::Display for ForecastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySeries => write!(f, "Series must not be empty."),
            Self::LengthMismatch { expected, got } => {
                write!(f, "actual and forecast must have the same length ({expected} != {got}).")
            }
            Self::InsufficientData { need, got } => {
                write!(f, "Need at least {need} observations, got {got}.")
            }
        }
    }
}

impl std::error::Error for ForecastError {}

#[inline]
fn clip_alpha(alpha: f64, alpha_min: f64, alpha_max: f64) -> f64 {
    alpha.clamp(alpha_min, alpha_max)
}

/// Trigg-Leach style adaptive exponential smoothing.
///
/// Alpha adapts to forecast errors: rises when errors increase, falls when stable.
pub fn adaptive_exponential_smoothing(
    y: &[f64],
    alpha_init: f64,
    gamma: f64,
    alpha_min: f64,
    alpha_max: f64,
) -> Result<AdaptiveResult, ForecastError> {
    let n = y.len();
    if n == 0 {
        return Err(ForecastError::EmptySeries);
    }

    let mut forecasts = vec![0.0; n];
    let mut levels = vec![0.0; n];
    let mut alphas = vec![0.0; n];
    let mut errors = vec![0.0; n];

    let mut level = y[0];
    let mut pe = if n > 1 { (y[1] - y[0]).abs() } else { 0.0 };
    let mut fe = pe;

    alphas[0] = alpha_init;
    levels[0] = level;
    forecasts[0] = level;

    for t in 1..n {
        let mut alpha = if pe + fe > 0.0 {
            pe / (pe + fe)
        } else {
            alpha_init
        };
        alpha = clip_alpha(alpha, alpha_min, alpha_max);

        let forecast = level;
        forecasts[t] = forecast;
        let error = y[t] - forecast;
        errors[t] = error;
        level += alpha * error;
        pe = gamma * error.abs() + (1.0 - gamma) * pe;
        fe = gamma * (y[t] - level).abs() + (1.0 - gamma) * fe;

        alphas[t] = alpha;
        levels[t] = level;
    }

    Ok(AdaptiveResult {
        forecasts,
        levels,
        alphas,
        errors,
    })
}

/// SAFT: switches between stable and responsive smoothing based on error spikes.
pub fn saft_forecast(
    y: &[f64],
    stable_alpha: f64,
    responsive_alpha: f64,
    error_window: usize,
    spike_threshold: f64,
) -> Result<SaftResult, ForecastError> {
    let n = y.len();
    if n == 0 {
        return Err(ForecastError::EmptySeries);
    }

    let mut forecasts = vec![0.0; n];
    let mut errors = vec![0.0; n];
    let mut modes = vec![0u8; n];
    let mut alphas = vec![0.0; n];
    let mut abs_errors: Vec<f64> = Vec::with_capacity(n);

    let mut level = y[0];
    forecasts[0] = level;
    alphas[0] = stable_alpha;

    for t in 1..n {
        let recent: &[f64] = if abs_errors.is_empty() {
            &[0.0]
        } else {
            let start = abs_errors.len().saturating_sub(error_window);
            &abs_errors[start..]
        };
        let baseline = recent.iter().sum::<f64>() / recent.len() as f64;
        let last_error = abs_errors.last().copied().unwrap_or(0.0);

        let responsive = baseline > 0.0 && last_error > spike_threshold * baseline;
        let alpha = if responsive {
            responsive_alpha
        } else {
            stable_alpha
        };
        modes[t] = u8::from(responsive);
        alphas[t] = alpha;

        let forecast = level;
        forecasts[t] = forecast;
        let error = y[t] - forecast;
        errors[t] = error;
        level += alpha * error;
        abs_errors.push(error.abs());
    }

    Ok(SaftResult {
        forecasts,
        errors,
        modes,
        alphas,
    })
}

/// One-step-ahead MAE, skipping the first observation.
pub fn mean_absolute_error(actual: &[f64], forecast: &[f64]) -> Result<f64, ForecastError> {
    if actual.len() != forecast.len() {
        return Err(ForecastError::LengthMismatch {
            expected: actual.len(),
            got: forecast.len(),
        });
    }
    if actual.len() <= 1 {
        return Ok(0.0);
    }
    let mae = actual[1..]
        .iter()
        .zip(&forecast[1..])
        .map(|(a, f)| (a - f).abs())
        .sum::<f64>()
        / (actual.len() - 1) as f64;
    Ok(mae)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_series_errors() {
        assert_eq!(
            adaptive_exponential_smoothing(&[], 0.2, 0.1, 0.05, 0.95),
            Err(ForecastError::EmptySeries)
        );
    }

    #[test]
    fn constant_series_zero_error() {
        let y = vec![7.0; 5];
        let result = adaptive_exponential_smoothing(&y, 0.3, 0.1, 0.05, 0.95).unwrap();
        assert!(result.errors.iter().all(|e| e.abs() < 1e-12));
        assert!(result.forecasts.iter().all(|f| (*f - 7.0).abs() < 1e-12));
    }

    #[test]
    fn mae_length_mismatch() {
        let err = mean_absolute_error(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert_eq!(
            err,
            ForecastError::LengthMismatch {
                expected: 2,
                got: 1
            }
        );
    }
}
