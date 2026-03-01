from __future__ import annotations

import pandas as pd

from src.eval import add_alerts, compute_metrics


def test_compute_metrics_and_alerts() -> None:
    frame = pd.DataFrame(
        {
            "actual": [0.1, 0.2, 0.3],
            "pred_mean": [0.11, 0.19, 0.29],
            "pred_std": [0.02, 0.02, 0.02],
            "lower_50": [0.09, 0.17, 0.27],
            "upper_50": [0.13, 0.21, 0.31],
            "lower_90": [0.07, 0.15, 0.25],
            "upper_90": [0.15, 0.23, 0.33],
            "lower_95": [0.06, 0.14, 0.24],
            "upper_95": [0.16, 0.24, 0.34],
            "persistence_pred": [0.12, 0.18, 0.31],
            "ewma_pred": [0.11, 0.2, 0.28],
            "fit_time_sec": [1.0, 1.0, 1.0],
            "predict_time_sec": [0.01, 0.01, 0.01],
        }
    )
    metrics = compute_metrics(frame, [0.5, 0.9, 0.95])
    assert metrics["rmse"] > 0
    assert 0.0 <= metrics["coverage_90"] <= 1.0

    alerted = add_alerts(frame)
    assert "anomaly_flag" in alerted.columns
