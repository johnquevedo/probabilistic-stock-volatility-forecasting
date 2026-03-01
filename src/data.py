from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


def cache_path(cache_dir: Path, ticker: str, period: str, interval: str) -> Path:
    """Return the cache file path for a ticker download."""
    safe_ticker = ticker.replace("^", "")
    return cache_dir / f"{safe_ticker}_{period}_{interval}.parquet"


def download_ohlcv(
    ticker: str,
    cache_dir: Path,
    period: str = "10y",
    interval: str = "1d",
    refresh: bool = False,
) -> pd.DataFrame:
    """Load daily OHLCV history from cache or yfinance."""
    path = cache_path(cache_dir, ticker, period, interval)
    if path.exists() and not refresh:
        return pd.read_parquet(path)

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns=str).sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_parquet(path)
    return df


def make_feature_frame(
    price_df: pd.DataFrame,
    horizon: int = 5,
    rv_window: int = 5,
    max_lag: int = 10,
    volume_z_window: int = 20,
) -> pd.DataFrame:
    """Build lagged volatility and calendar features plus the forward RV target."""
    df = price_df.copy()
    close = np.log(df["Close"]).rename("log_close")
    returns = close.diff().rename("log_return")
    rv = np.sqrt(returns.pow(2).rolling(rv_window).sum()).rename("rv")

    features = pd.DataFrame(index=df.index)
    features["rv"] = rv
    features["ret_ewm_5"] = returns.ewm(span=5, adjust=False).mean()
    features["ret_ewm_20"] = returns.ewm(span=20, adjust=False).mean()
    features["ret_std_20"] = returns.rolling(20).std()
    features["volume_z"] = (
        (df["Volume"] - df["Volume"].rolling(volume_z_window).mean())
        / df["Volume"].rolling(volume_z_window).std()
    )
    features["time_idx"] = np.arange(len(features), dtype=float)

    for lag in range(max_lag + 1):
        features[f"rv_lag_{lag}"] = rv.shift(lag)

    dow = features.index.dayofweek
    month = features.index.month
    features["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    features["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)
    features["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    features["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    features["target"] = rv.shift(-horizon)
    features["target_date"] = features.index.to_series().shift(-horizon)
    features["actual_close"] = df["Close"]

    return features.dropna().copy()


def train_test_mask(index: pd.DatetimeIndex, backtest_years: int = 2) -> pd.Series:
    """Return a boolean mask for the backtest portion of the series."""
    cutoff = index.max() - pd.DateOffset(years=backtest_years)
    return pd.Series(index >= cutoff, index=index)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return model feature columns, excluding labels and metadata."""
    excluded = {"target", "target_date", "actual_close"}
    return [column for column in frame.columns if column not in excluded]


def standardize_split(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Standardize train and test features using train-only moments."""
    mean = train_x.mean()
    std = train_x.std().replace(0.0, 1.0).fillna(1.0)
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


def next_feature_row(
    history: pd.DataFrame,
    next_date: pd.Timestamp,
    predicted_rv: float,
    feature_names: list[str],
) -> dict[str, Any]:
    """Construct the next recursive forecast feature row from the latest history."""
    row = history.iloc[-1].copy()
    rv_lags = [name for name in feature_names if name.startswith("rv_lag_")]
    lag_map = {name: float(row[name]) for name in rv_lags}
    for lag_name in sorted(rv_lags, key=lambda item: int(item.rsplit("_", 1)[-1]), reverse=True):
        lag_idx = int(lag_name.rsplit("_", 1)[-1])
        if lag_idx == 0:
            lag_map[lag_name] = predicted_rv
        else:
            lag_map[lag_name] = float(row.get(f"rv_lag_{lag_idx - 1}", predicted_rv))

    dow = next_date.dayofweek
    month = next_date.month
    next_row = {
        "rv": predicted_rv,
        "ret_ewm_5": float(row["ret_ewm_5"]),
        "ret_ewm_20": float(row["ret_ewm_20"]),
        "ret_std_20": float(row["ret_std_20"]),
        "volume_z": float(row["volume_z"]),
        "time_idx": float(row["time_idx"]) + 1.0,
        "dow_sin": float(np.sin(2 * np.pi * dow / 5.0)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 5.0)),
        "month_sin": float(np.sin(2 * np.pi * month / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * month / 12.0)),
    }
    next_row.update(lag_map)
    return {name: next_row[name] for name in feature_names}
