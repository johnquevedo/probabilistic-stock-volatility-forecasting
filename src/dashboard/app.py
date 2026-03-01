from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils import load_config, project_root, resolve_paths


def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_config()
    paths = resolve_paths(config)
    prediction_path = paths.artifacts_dir / config["artifacts"]["predictions_file"]
    metrics_path = paths.artifacts_dir / config["artifacts"]["metrics_file"]
    future_path = paths.artifacts_dir / config["artifacts"]["future_file"]
    if not prediction_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Run `python -m src.eval` first to generate artifacts.")

    predictions = pd.read_parquet(prediction_path)
    future = pd.read_parquet(future_path) if future_path.exists() else pd.DataFrame()
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = pd.DataFrame(json.load(handle))
    return predictions, metrics, future


def interval_figure(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["target_date"], y=frame["actual"], name="Actual RV", line=dict(color="#1b4332")))
    fig.add_trace(go.Scatter(x=frame["target_date"], y=frame["upper_90"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(
        go.Scatter(
            x=frame["target_date"],
            y=frame["lower_90"],
            fill="tonexty",
            fillcolor="rgba(64,145,108,0.18)",
            line=dict(width=0),
            name="90% interval",
            hoverinfo="skip",
        )
    )
    fig.add_trace(go.Scatter(x=frame["target_date"], y=frame["pred_mean"], name="GP mean", line=dict(color="#40916c")))
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=430, template="plotly_white")
    return fig


def coverage_figure(frame: pd.DataFrame) -> go.Figure:
    expected = [0.5, 0.9, 0.95]
    observed = [
        float(((frame["actual"] >= frame["lower_50"]) & (frame["actual"] <= frame["upper_50"])).mean()),
        float(((frame["actual"] >= frame["lower_90"]) & (frame["actual"] <= frame["upper_90"])).mean()),
        float(((frame["actual"] >= frame["lower_95"]) & (frame["actual"] <= frame["upper_95"])).mean()),
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=expected, y=expected, name="Ideal", mode="lines", line=dict(dash="dash", color="#6c757d")))
    fig.add_trace(go.Scatter(x=expected, y=observed, name="Observed", mode="lines+markers", line=dict(color="#d62828")))
    fig.update_layout(xaxis_title="Expected coverage", yaxis_title="Observed coverage", height=320, template="plotly_white")
    return fig


def rolling_coverage_figure(frame: pd.DataFrame) -> go.Figure:
    covered = ((frame["actual"] >= frame["lower_90"]) & (frame["actual"] <= frame["upper_90"])).astype(float)
    rolling = covered.rolling(30).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frame["target_date"], y=rolling, name="30-day rolling 90% coverage", line=dict(color="#003049")))
    fig.add_hline(y=0.9, line_dash="dash", line_color="#6c757d")
    fig.update_layout(height=320, template="plotly_white", margin=dict(l=20, r=20, t=30, b=20))
    return fig


def main() -> None:
    st.set_page_config(page_title="GP Volatility Forecasting", layout="wide")
    st.title("Probabilistic Stock Volatility Forecasting")
    st.caption("Gaussian Process regression with predictive intervals, calibration views, and anomaly alerts.")

    predictions, metrics, future = load_artifacts()
    predictions["target_date"] = pd.to_datetime(predictions["target_date"])
    predictions["forecast_origin"] = pd.to_datetime(predictions["forecast_origin"])
    if not future.empty:
        future["forecast_date"] = pd.to_datetime(future["forecast_date"])

    tickers = sorted(predictions["ticker"].unique())
    selected_ticker = st.sidebar.selectbox("Ticker", tickers)
    horizon = st.sidebar.number_input("Horizon (days)", value=load_config()["model"]["horizon"], min_value=1, max_value=20)
    window_days = st.sidebar.slider("Days shown", min_value=60, max_value=504, value=252, step=21)

    ticker_frame = predictions[predictions["ticker"] == selected_ticker].sort_values("target_date").tail(window_days)
    ticker_metrics = metrics[metrics["ticker"] == selected_ticker]
    future_frame = future[future["ticker"] == selected_ticker].sort_values("forecast_date")

    latest = ticker_frame.iloc[-1]
    col1, col2, col3 = st.columns(3)
    col1.metric("Latest forecast", f"{latest['pred_mean']:.4f}")
    col2.metric("90% interval", f"[{latest['lower_90']:.4f}, {latest['upper_90']:.4f}]")
    col3.metric("Latest actual", f"{latest['actual']:.4f}")

    st.subheader("Overview")
    st.plotly_chart(interval_figure(ticker_frame), use_container_width=True)

    st.subheader("Forecasts")
    if not future_frame.empty:
        display_future = future_frame.copy()
        display_future["horizon_days"] = range(1, len(display_future) + 1)
        st.dataframe(display_future[["forecast_date", "horizon_days", "pred_mean", "lower_90", "upper_90"]], use_container_width=True)
    else:
        st.info("No future forecast artifact found.")

    st.subheader("Calibration")
    cal_col1, cal_col2 = st.columns(2)
    cal_col1.plotly_chart(coverage_figure(ticker_frame), use_container_width=True)
    cal_col2.plotly_chart(rolling_coverage_figure(ticker_frame), use_container_width=True)

    st.subheader("Alerts")
    alerts = ticker_frame[ticker_frame["anomaly_flag"] | ticker_frame["regime_alert"]].copy()
    if alerts.empty:
        st.info("No alerts in the selected range.")
    else:
        st.dataframe(
            alerts[["target_date", "actual", "pred_mean", "std_residual", "interval_breach_95", "regime_score", "regime_alert"]],
            use_container_width=True,
        )

    st.subheader("Performance")
    st.dataframe(ticker_metrics, use_container_width=True)
    perf_fig = go.Figure()
    metric_row = ticker_metrics.iloc[0]
    perf_fig.add_bar(name="RMSE", x=["GP", "Persistence", "EWMA"], y=[metric_row["rmse"], metric_row["baseline_rmse_persistence"], metric_row["baseline_rmse_ewma"]])
    perf_fig.update_layout(height=320, template="plotly_white", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(perf_fig, use_container_width=True)

    st.caption(f"Artifacts loaded from {Path(project_root() / 'artifacts')}")
    st.caption(f"Configured horizon: {horizon} trading days")


if __name__ == "__main__":
    main()
