//! PyO3 bindings — call Rust forecasting kernels from Python with zero-copy NumPy I/O.

use adaptive_core::{adaptive_exponential_smoothing, saft_forecast, ForecastError};
use numpy::{PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn map_err(err: ForecastError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

#[pyfunction]
#[pyo3(signature = (y, alpha_init=0.2, gamma=0.1, alpha_min=0.05, alpha_max=0.95))]
fn adaptive_exponential_smoothing_py(
    py: Python<'_>,
    y: PyReadonlyArray1<f64>,
    alpha_init: f64,
    gamma: f64,
    alpha_min: f64,
    alpha_max: f64,
) -> PyResult<Py<PyDict>> {
    let slice = y.as_slice()?;
    let result = adaptive_exponential_smoothing(slice, alpha_init, gamma, alpha_min, alpha_max)
        .map_err(map_err)?;

    let dict = PyDict::new(py);
    dict.set_item("forecasts", result.forecasts.to_pyarray(py))?;
    dict.set_item("levels", result.levels.to_pyarray(py))?;
    dict.set_item("alphas", result.alphas.to_pyarray(py))?;
    dict.set_item("errors", result.errors.to_pyarray(py))?;
    Ok(dict.into())
}

#[pyfunction]
#[pyo3(signature = (
    y,
    stable_alpha=0.15,
    responsive_alpha=0.6,
    error_window=5,
    spike_threshold=1.5
))]
fn saft_forecast_py(
    py: Python<'_>,
    y: PyReadonlyArray1<f64>,
    stable_alpha: f64,
    responsive_alpha: f64,
    error_window: usize,
    spike_threshold: f64,
) -> PyResult<Py<PyDict>> {
    let slice = y.as_slice()?;
    let result = saft_forecast(
        slice,
        stable_alpha,
        responsive_alpha,
        error_window,
        spike_threshold,
    )
    .map_err(map_err)?;

    let dict = PyDict::new(py);
    dict.set_item("forecasts", result.forecasts.to_pyarray(py))?;
    dict.set_item("errors", result.errors.to_pyarray(py))?;
    dict.set_item("modes", result.modes.to_pyarray(py))?;
    dict.set_item("alphas", result.alphas.to_pyarray(py))?;
    Ok(dict.into())
}

#[pyfunction]
fn rust_info() -> &'static str {
    "adaptive_forecast_rs: Trigg-Leach smoothing compiled with Rust + PyO3"
}

/// Benchmark helper exposed to Python for apples-to-apples timing.
#[pyfunction]
#[pyo3(signature = (y, iterations=10_000))]
fn bench_adaptive_py(y: PyReadonlyArray1<f64>, iterations: usize) -> PyResult<f64> {
    let slice = y.as_slice()?.to_vec();
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        adaptive_exponential_smoothing(&slice, 0.2, 0.1, 0.05, 0.95)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    }
    Ok(start.elapsed().as_secs_f64())
}

#[pymodule]
fn adaptive_forecast_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adaptive_exponential_smoothing_py, m)?)?;
    m.add_function(wrap_pyfunction!(saft_forecast_py, m)?)?;
    m.add_function(wrap_pyfunction!(rust_info, m)?)?;
    m.add_function(wrap_pyfunction!(bench_adaptive_py, m)?)?;
    Ok(())
}
