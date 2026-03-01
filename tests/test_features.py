from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import make_feature_frame


def test_make_feature_frame_builds_target_and_lags() -> None:
    index = pd.bdate_range("2024-01-01", periods=40)
    close = np.linspace(100, 120, len(index))
    volume = np.linspace(1_000_000, 1_500_000, len(index))
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Adj Close": close,
            "Volume": volume,
        },
        index=index,
    )

    features = make_feature_frame(frame, horizon=5, rv_window=5, max_lag=3, volume_z_window=5)
    assert "target" in features.columns
    assert "rv_lag_3" in features.columns
    assert "dow_sin" in features.columns
    assert features["target"].notna().all()
