# Probabilistic Stock Volatility Forecasting with Gaussian Processes

Forecast 5-day-ahead realized volatility for liquid tickers (`SPY`, `QQQ`, `AAPL`) using Gaussian Process Regression with predictive uncertainty, walk-forward evaluation, and dashboard-based monitoring.

Live app: https://probabilistic-stock-volatility-forecasting.streamlit.app/

## What this project does

- Downloads daily OHLCV history from `yfinance` and caches it locally.
- Engineers realized-volatility features (`RV` lags, return stats, calendar cyclic features, volume z-score).
- Trains rolling-window exact GPs (`PyTorch` + `GPyTorch`) for walk-forward backtests.
- Produces mean forecasts with 50/90/95% prediction intervals.
- Flags anomalies and regime alerts from interval breaches and standardized residuals.
- Serves analytics in a `Streamlit` dashboard.

## Model setup

Target definitions:

```text
r_t = log(C_t / C_{t-1})
RV_t = sqrt(sum_{i=0..k-1} r_{t-i}^2)
```

- Forecast target: `RV_{t+5}` (next-week realized volatility proxy)
- Kernel: `ScaleKernel(RBF + Periodic)`
- Likelihood: `GaussianLikelihood`
- Training: rolling window (default `W=750`), Adam + early stopping
- Baselines: persistence and EWMA

## Latest backtest snapshot (2-year walk-forward)

| Ticker | N preds | GP RMSE | GP MAE | Persistence RMSE | EWMA RMSE | Coverage@90 |
|---|---:|---:|---:|---:|---:|---:|
| SPY | 503 | 0.0103 | 0.0062 | 0.0140 | 0.0146 | 0.980 |
| QQQ | 503 | 0.0117 | 0.0081 | 0.0163 | 0.0165 | 0.968 |
| AAPL | 503 | 0.0198 | 0.0133 | 0.0258 | 0.0240 | 0.899 |

Artifacts are committed in `artifacts/` so the deployed dashboard does not need retraining.

## Repository structure

```text
src/
  data.py             # data pull, cache, features
  baselines.py        # persistence + EWMA forecasts
  eval.py             # rolling backtest, metrics, artifact export CLI
  utils.py            # config + path utilities
  models/gp.py        # exact GP model, train, inference
  dashboard/app.py    # Streamlit app
configs/default.yaml
tests/
artifacts/
```

## Local setup

Python `3.11` or `3.12`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run evaluation

```bash
python -m src.eval --tickers SPY QQQ AAPL --horizon 5 --window 750 --out artifacts/results.parquet
```

Outputs:

- `artifacts/results.parquet`
- `artifacts/future_forecasts.parquet`
- `artifacts/metrics_summary.json`

## Run dashboard

```bash
streamlit run src/dashboard/app.py
```

## Tests

```bash
pytest
```
