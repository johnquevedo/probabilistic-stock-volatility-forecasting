# Probabilistic Stock Volatility Forecasting with Gaussian Processes

This project forecasts 5-day ahead realized volatility for liquid equities and ETFs using Gaussian Process Regression with calibrated predictive intervals. It includes a walk-forward backtest pipeline, baseline comparisons, anomaly and regime alerts, and a Streamlit dashboard for analytics.

## Problem statement

Daily equity volatility is heteroskedastic and regime-dependent. Point forecasts alone are not enough for decision-making, so this project models realized volatility as a probabilistic forecast problem:

- Target: future realized volatility over a rolling window, defaulting to 5 trading days.
- Model: exact Gaussian Process regression in GPyTorch over lagged volatility and calendar features.
- Output: predictive mean, uncertainty intervals, calibration diagnostics, and alerting signals.

## Methodology

1. Download daily OHLCV data from `yfinance` and cache it locally in `data/cache/`.
2. Compute log returns and rolling realized volatility:

```text
r_t = log(C_t / C_{t-1})
RV_t = sqrt(sum_{i=0..k-1} r_{t-i}^2)
```

3. Build features from:
   - `RV_t ... RV_{t-L}` lag stack
   - return EWMs and rolling return volatility
   - rolling volume z-score
   - day-of-week and month cyclic encodings
4. Run a walk-forward backtest over the last 2 years using a rolling training window.
5. Fit an exact GP with `GaussianLikelihood` and `ScaleKernel(RBF + Periodic)` on standardized train-only features.
6. Compare against persistence and EWMA baselines.
7. Save predictions, metrics, and forward forecasts into `artifacts/` for the dashboard.

## Repo structure

```text
src/
  data.py
  baselines.py
  eval.py
  utils.py
  models/gp.py
  dashboard/app.py
configs/default.yaml
tests/
artifacts/
data/cache/
```

## Setup

Python 3.11 is the target runtime.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Run the backtest

Default tickers are `SPY`, `QQQ`, and `AAPL`.

```bash
python -m src.eval --tickers SPY QQQ AAPL --horizon 5 --window 750 --out artifacts/results.parquet
```

Artifacts produced:

- `artifacts/results.parquet`: walk-forward predictions, intervals, residuals, alerts, and latency stats
- `artifacts/future_forecasts.parquet`: next 5 trading day forecasts with intervals
- `artifacts/metrics_summary.json`: per-ticker backtest metrics
- `data/cache/*.parquet`: cached raw daily OHLCV history

## Launch the dashboard

```bash
streamlit run src/dashboard/app.py
```

## Dashboard guide

The dashboard includes five sections:

1. Overview: latest forecast and 90% interval for the selected ticker.
2. Forecasts: actual versus predicted realized volatility with interval shading plus next 5 trading day forecast table.
3. Calibration: expected versus observed coverage and rolling 90% coverage.
4. Alerts: anomalies where actual RV breaches the 95% interval or the standardized residual is large, plus regime indicators.
5. Performance: GP versus persistence and EWMA latency and error comparisons.

## Metrics

- RMSE / MAE for mean forecasts
- 50 / 90 / 95 interval coverage
- Average interval width
- Simple Gaussian CRPS-style approximation
- Average fit and predict latency

## Notes on runtime

- Exact GP training is kept CPU-friendly by using a rolling train window capped at around 750 rows by default.
- If you increase the window well above 1000 observations, training latency will rise sharply because exact GP inference scales cubically.
- CUDA is supported automatically when available, but not required.

## Screenshots

Add screenshots here after running the dashboard:

- `docs/screenshots/overview.png`
- `docs/screenshots/calibration.png`

## Tests

```bash
pytest
```
