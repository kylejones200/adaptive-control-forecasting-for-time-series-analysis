# Adaptive Control Forecasting for Time Series Analysis

Published: 2025-02-27  
Medium: [Adaptive Control Forecasting for Time Series Analysis](https://medium.com/@kyle-t-jones/adaptive-control-forecasting-for-time-series-analysis-129a0afc972a)

## Business context

Forecasting time series data presents challenges. Patterns shift due to economic cycles, technological change, or external shocks. Traditional methods assume fixed parameters and fail when systems change. Adaptive control forecasting adjusts model parameters based on new data.

This repo demonstrates:

- **Adaptive exponential smoothing** — Trigg-Leach style α adaptation
- **Adaptive Holt-Winters** — error-driven updates to α, β, and γ
- **SAFT** — Self-Adaptive Forecasting Technique with model switching

For Kalman filtering and state-space models, see the dedicated companion repo linked below.

## Project structure

```
.
├── README.md
├── article.md              # Original Medium article export
├── main.py                 # Run all demos
├── config.yaml             # Demo parameters
├── src/
│   ├── adaptive_smoothing.py
│   ├── synthetic.py
│   └── plotting.py
├── tests/
└── images/                 # Generated figures (created by main.py)
```

## Quick start

```bash
uv sync          # or: pip install -e .
python main.py
```

Figures are written to `images/`:

- `adaptive_exponential_smoothing.png` — fixed vs adaptive α on a regime-shift series
- `saft_model_switching.png` — SAFT stable/responsive mode switching
- `adaptive_holt_winters.png` — seasonal series with adapting parameters

## Related repos

- [State Space Models and Kalman Filtering for Time Series Analysis](../state-space-models-and-kalman-filtering-for-time-series-analysis) — full Kalman filtering walkthrough with structural models and forecasting
- [Exponential Smoothing for Time Series Forecasting in Python](../exponential-smoothing-for-time-series-forecasting-in-python) — fixed-parameter exponential smoothing and ETS

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).
