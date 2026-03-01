from __future__ import annotations

import numpy as np
import pandas as pd


def persistence_forecast(frame: pd.DataFrame) -> pd.Series:
    """Use current realized volatility as the future forecast."""
    return frame["rv_lag_0"].rename("persistence_pred")


def ewma_forecast(frame: pd.DataFrame, decay: float = 0.94) -> pd.Series:
    """Forecast volatility using an EWMA of squared realized volatility."""
    squared_rv = frame["rv_lag_0"].pow(2)
    ewma_var = squared_rv.ewm(alpha=1.0 - decay, adjust=False).mean()
    return np.sqrt(ewma_var).rename("ewma_pred")
