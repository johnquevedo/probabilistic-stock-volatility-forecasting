from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.baselines import ewma_forecast, persistence_forecast
from src.data import (
    download_ohlcv,
    feature_columns,
    make_feature_frame,
    next_feature_row,
    standardize_split,
    train_test_mask,
)
from src.models.gp import fit_exact_gp, predict_distribution
from src.utils import business_day_range, json_ready_metrics, load_config, resolve_paths, select_device


def normal_quantile(level: float) -> float:
    """Return a z-score for supported central interval levels."""
    lookup = {
        0.5: 0.67448975,
        0.9: 1.64485363,
        0.95: 1.95996398,
    }
    if level not in lookup:
        raise ValueError(f"Unsupported coverage level {level}")
    return lookup[level]


def compute_metrics(frame: pd.DataFrame, coverage_levels: list[float]) -> dict[str, float]:
    """Compute forecast accuracy, calibration, and latency metrics."""
    if frame.empty:
        raise ValueError("No predictions available to score.")

    metrics: dict[str, float] = {
        "rmse": float(np.sqrt(np.mean((frame["actual"] - frame["pred_mean"]) ** 2))),
        "mae": float(np.mean(np.abs(frame["actual"] - frame["pred_mean"]))),
        "baseline_rmse_persistence": float(np.sqrt(np.mean((frame["actual"] - frame["persistence_pred"]) ** 2))),
        "baseline_rmse_ewma": float(np.sqrt(np.mean((frame["actual"] - frame["ewma_pred"]) ** 2))),
        "baseline_mae_persistence": float(np.mean(np.abs(frame["actual"] - frame["persistence_pred"]))),
        "baseline_mae_ewma": float(np.mean(np.abs(frame["actual"] - frame["ewma_pred"]))),
        "avg_fit_time_sec": float(frame["fit_time_sec"].mean()),
        "avg_predict_time_sec": float(frame["predict_time_sec"].mean()),
        "avg_interval_width_90": float((frame["upper_90"] - frame["lower_90"]).mean()),
    }

    residual_std = frame["pred_std"].replace(0.0, np.nan)
    z = (frame["actual"] - frame["pred_mean"]) / residual_std
    metrics["crps_gaussian_approx"] = float(np.nanmean(np.abs(z) * frame["pred_std"]))

    for level in coverage_levels:
        lower = frame[f"lower_{int(level * 100)}"]
        upper = frame[f"upper_{int(level * 100)}"]
        metrics[f"coverage_{int(level * 100)}"] = float(((frame["actual"] >= lower) & (frame["actual"] <= upper)).mean())
        metrics[f"avg_width_{int(level * 100)}"] = float((upper - lower).mean())

    return metrics


def add_alerts(frame: pd.DataFrame) -> pd.DataFrame:
    """Annotate anomaly and regime-shift style alerts from forecast residuals."""
    output = frame.copy()
    output["residual"] = output["actual"] - output["pred_mean"]
    output["std_residual"] = output["residual"] / output["pred_std"].replace(0.0, np.nan)
    output["interval_breach_95"] = output["actual"] > output["upper_95"]
    output["anomaly_flag"] = output["interval_breach_95"] | (output["std_residual"].abs() > 2.5)
    regime_base = output["pred_mean"].rolling(20).mean()
    regime_scale = output["pred_mean"].rolling(20).std().replace(0.0, np.nan)
    output["regime_score"] = (output["pred_mean"] - regime_base) / regime_scale
    output["residual_shift"] = output["residual"].rolling(10).mean() / output["residual"].rolling(20).std().replace(0.0, np.nan)
    output["regime_alert"] = output["regime_score"].abs() > 1.5
    return output


def walk_forward_backtest(
    ticker: str,
    feature_frame: pd.DataFrame,
    config: dict[str, Any],
    device: torch.device,
) -> pd.DataFrame:
    """Run a rolling-window walk-forward GP backtest for one ticker."""
    model_cfg = config["model"]
    eval_cfg = config["eval"]
    cov_levels = eval_cfg["coverage_levels"]
    feature_names = feature_columns(feature_frame)
    periodic_dim = feature_names.index("time_idx") if "time_idx" in feature_names else None
    backtest_mask = train_test_mask(feature_frame.index, eval_cfg["backtest_years"])
    backtest_indices = feature_frame.index[backtest_mask]
    results: list[dict[str, Any]] = []

    persistence = persistence_forecast(feature_frame)
    ewma = ewma_forecast(feature_frame, decay=eval_cfg["ewma_decay"])

    for current_idx in backtest_indices:
        loc = feature_frame.index.get_loc(current_idx)
        train_start = max(0, loc - model_cfg["train_window"])
        train_slice = feature_frame.iloc[train_start:loc]
        test_row = feature_frame.iloc[[loc]]
        if len(train_slice) < max(100, model_cfg["train_window"] // 3):
            continue

        train_x_raw = train_slice[feature_names]
        test_x_raw = test_row[feature_names]
        train_x, test_x, _, _ = standardize_split(train_x_raw, test_x_raw)
        train_y = train_slice["target"]

        train_tensor = torch.tensor(train_x.to_numpy(), dtype=torch.float32)
        test_tensor = torch.tensor(test_x.to_numpy(), dtype=torch.float32)
        target_tensor = torch.tensor(train_y.to_numpy(), dtype=torch.float32)

        fit_start = time.perf_counter()
        fit = fit_exact_gp(
            train_x=train_tensor,
            train_y=target_tensor,
            epochs=model_cfg["epochs"],
            lr=model_cfg["lr"],
            patience=model_cfg["early_stopping_patience"],
            min_improvement=model_cfg["min_improvement"],
            use_periodic_kernel=model_cfg["use_periodic_kernel"],
            periodic_dim=periodic_dim,
            device=device,
        )
        fit_time = time.perf_counter() - fit_start

        predict_start = time.perf_counter()
        mean, std = predict_distribution(fit.model, fit.likelihood, test_tensor, device=device)
        predict_time = time.perf_counter() - predict_start

        record: dict[str, Any] = {
            "ticker": ticker,
            "forecast_origin": current_idx,
            "target_date": pd.Timestamp(test_row["target_date"].iloc[0]),
            "actual": float(test_row["target"].iloc[0]),
            "pred_mean": float(mean[0]),
            "pred_std": float(std[0]),
            "fit_time_sec": fit_time,
            "predict_time_sec": predict_time,
            "train_loss": fit.train_loss,
            "epochs_run": fit.epochs_run,
            "persistence_pred": float(persistence.loc[current_idx]),
            "ewma_pred": float(ewma.loc[current_idx]),
        }
        for level in cov_levels:
            z_value = normal_quantile(level)
            record[f"lower_{int(level * 100)}"] = float(mean[0] - z_value * std[0])
            record[f"upper_{int(level * 100)}"] = float(mean[0] + z_value * std[0])
        results.append(record)

    if not results:
        return pd.DataFrame()
    return add_alerts(pd.DataFrame(results))


def forecast_next_days(
    ticker: str,
    feature_frame: pd.DataFrame,
    config: dict[str, Any],
    device: torch.device,
) -> pd.DataFrame:
    """Generate recursive forecasts for the next configured business days."""
    model_cfg = config["model"]
    cov_levels = config["eval"]["coverage_levels"]
    feature_names = feature_columns(feature_frame)
    periodic_dim = feature_names.index("time_idx") if "time_idx" in feature_names else None
    train_slice = feature_frame.iloc[-model_cfg["train_window"] :].copy()

    train_x_raw = train_slice[feature_names]
    train_y = train_slice["target"]
    train_x, _, mean, std = standardize_split(train_x_raw, train_x_raw.iloc[[-1]])

    fit = fit_exact_gp(
        train_x=torch.tensor(train_x.to_numpy(), dtype=torch.float32),
        train_y=torch.tensor(train_y.to_numpy(), dtype=torch.float32),
        epochs=model_cfg["epochs"],
        lr=model_cfg["lr"],
        patience=model_cfg["early_stopping_patience"],
        min_improvement=model_cfg["min_improvement"],
        use_periodic_kernel=model_cfg["use_periodic_kernel"],
        periodic_dim=periodic_dim,
        device=device,
    )

    history = train_slice.copy()
    future_dates = business_day_range(feature_frame.index.max() + pd.offsets.BDay(1), periods=model_cfg["horizon"])
    forecasts: list[dict[str, Any]] = []

    for step, next_date in enumerate(future_dates, start=1):
        base_rv = float(history["rv_lag_0"].iloc[-1] if step == 1 else forecasts[-1]["pred_mean"])
        next_row = next_feature_row(history, next_date, predicted_rv=base_rv, feature_names=feature_names)
        next_features = pd.DataFrame([next_row], index=[next_date])
        next_scaled = (next_features - mean) / std
        mean_pred, std_pred = predict_distribution(
            fit.model,
            fit.likelihood,
            torch.tensor(next_scaled.to_numpy(), dtype=torch.float32),
            device=device,
        )
        record: dict[str, Any] = {
            "ticker": ticker,
            "forecast_date": next_date,
            "pred_mean": float(mean_pred[0]),
            "pred_std": float(std_pred[0]),
        }
        for level in cov_levels:
            z_value = normal_quantile(level)
            record[f"lower_{int(level * 100)}"] = float(mean_pred[0] - z_value * std_pred[0])
            record[f"upper_{int(level * 100)}"] = float(mean_pred[0] + z_value * std_pred[0])
        forecasts.append(record)

        appended = next_features.copy()
        appended["target"] = np.nan
        appended["target_date"] = pd.NaT
        appended["actual_close"] = np.nan
        history = pd.concat([history, appended], axis=0)
        history.iloc[-1, history.columns.get_loc("rv")] = float(mean_pred[0])
        history.iloc[-1, history.columns.get_loc("rv_lag_0")] = float(mean_pred[0])

    return pd.DataFrame(forecasts)


def run_for_ticker(ticker: str, config: dict[str, Any], refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Execute data loading, backtest, scoring, and future forecasting for one ticker."""
    paths = resolve_paths(config)
    device = select_device(config["model"]["device"])
    raw = download_ohlcv(
        ticker=ticker,
        cache_dir=paths.cache_dir,
        period=config["data"]["period"],
        interval=config["data"]["interval"],
        refresh=refresh,
    )
    feature_frame = make_feature_frame(
        raw,
        horizon=config["model"]["horizon"],
        rv_window=config["data"]["rv_window"],
        max_lag=config["data"]["max_lag"],
        volume_z_window=config["data"]["volume_z_window"],
    )
    predictions = walk_forward_backtest(ticker, feature_frame, config, device)
    metrics = compute_metrics(predictions, config["eval"]["coverage_levels"])
    metrics.update({"ticker": ticker, "n_predictions": int(len(predictions))})
    future = forecast_next_days(ticker, feature_frame, config, device)
    return predictions, metrics, future


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluation entrypoint."""
    parser = argparse.ArgumentParser(description="Walk-forward GP volatility forecasting evaluation.")
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "AAPL"])
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for end-to-end evaluation and artifact export."""
    args = parse_args()
    config = load_config(args.config)
    if args.horizon is not None:
        config["model"]["horizon"] = args.horizon
    if args.window is not None:
        config["model"]["train_window"] = args.window

    paths = resolve_paths(config)
    prediction_frames: list[pd.DataFrame] = []
    future_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    for ticker in args.tickers:
        predictions, metrics, future = run_for_ticker(ticker, config, refresh=args.refresh)
        prediction_frames.append(predictions)
        metric_rows.append(metrics)
        future_frames.append(future)

    prediction_output = pd.concat(prediction_frames, ignore_index=True)
    future_output = pd.concat(future_frames, ignore_index=True)
    out_path = Path(args.out) if args.out else paths.artifacts_dir / config["artifacts"]["predictions_file"]
    metrics_path = paths.artifacts_dir / config["artifacts"]["metrics_file"]
    future_path = paths.artifacts_dir / config["artifacts"]["future_file"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_output.to_parquet(out_path, index=False)
    future_output.to_parquet(future_path, index=False)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(json_ready_metrics(metric_rows), handle, indent=2)

    print(f"Saved predictions to {out_path}")
    print(f"Saved future forecasts to {future_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
