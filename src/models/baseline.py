"""Baseline persistence forecasting models and evaluation metrics."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate genuine MAE, RMSE, and R2 evaluation metrics.

    Args:
        y_true: Ground truth observed values (1D array).
        y_pred: Predicted values (1D array).

    Returns:
        Dictionary with 'mae', 'rmse', and 'r2' scores rounded to 4 decimals.
    """
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()

    if len(yt) != len(yp):
        raise ValueError(f"Array length mismatch: y_true has {len(yt)}, y_pred has {len(yp)}.")

    if len(yt) == 0:
        raise ValueError("Cannot calculate metrics on empty arrays.")

    mae = float(mean_absolute_error(yt, yp))
    mse = float(mean_squared_error(yt, yp))
    rmse = float(np.sqrt(mse))

    # R2 can be undefined/negative on poorly fitting models
    try:
        r2 = float(r2_score(yt, yp))
    except Exception:
        r2 = float("nan")

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


class PersistenceBaseline:
    """Persistence (naïve) weather forecasting baseline.

    Under persistence forecasting, the next day's temperature is predicted
    to be identical to the most recently observed day's temperature:
        y_hat[t] = y[t-1]
    """

    @staticmethod
    def predict_holdout(
        train_series: pd.Series,
        test_series: pd.Series,
    ) -> Tuple[np.ndarray, dict[str, float]]:
        """Generate persistence predictions for the holdout test period.

        For test[0], the prediction is train[-1].
        For test[k] (k > 0), the prediction is test[k - 1].

        Args:
            train_series: Chronological training observations.
            test_series: Chronological test observations.

        Returns:
            Tuple of (predictions_array, metrics_dict).
        """
        if len(test_series) == 0:
            raise ValueError("Test series must not be empty.")

        last_train_val = float(train_series.iloc[-1])
        test_vals = test_series.values

        predictions = np.empty_like(test_vals, dtype=float)
        predictions[0] = last_train_val
        if len(test_vals) > 1:
            predictions[1:] = test_vals[:-1]

        metrics = calculate_metrics(test_vals, predictions)
        return predictions, metrics

    @staticmethod
    def forecast_7_days(
        last_observed_value: float,
        last_date: pd.Timestamp,
    ) -> pd.Series:
        """Produce 7-day persistence forecast (holding the last observed temperature constant).

        Args:
            last_observed_value: Most recently recorded temperature in Celsius.
            last_date: Timestamp of the last observation.

        Returns:
            pd.Series of 7 daily forecasts indexed by subsequent dates.
        """
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=7,
            freq="D",
        )
        return pd.Series([float(last_observed_value)] * 7, index=future_dates, name="baseline_forecast_celsius")
