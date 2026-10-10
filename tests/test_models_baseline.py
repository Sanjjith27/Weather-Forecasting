"""Unit tests for metrics calculation and the Persistence baseline forecaster."""

import numpy as np
import pandas as pd
import pytest

from src.models.baseline import PersistenceBaseline, calculate_metrics


def test_calculate_metrics_values() -> None:
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 19.0, 28.0])

    # Errors: |+2|, |-1|, |-2| -> sum = 5 / 3 = 1.6667
    # Squared errors: 4, 1, 4 -> sum = 9 / 3 = 3.0 -> sqrt = 1.7321
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(1.6667, abs=1e-3)
    assert metrics["rmse"] == pytest.approx(1.7321, abs=1e-3)
    assert np.isfinite(metrics["r2"])


def test_calculate_metrics_perfect_predictions() -> None:
    y = np.array([5.0, 10.0, 15.0, 20.0])
    metrics = calculate_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_persistence_holdout_predictions() -> None:
    train_dates = pd.date_range("2023-01-01", periods=3, freq="D")
    test_dates = pd.date_range("2023-01-04", periods=3, freq="D")

    train_series = pd.Series([10.0, 11.0, 12.0], index=train_dates)
    test_series = pd.Series([13.0, 14.0, 15.0], index=test_dates)

    preds, metrics = PersistenceBaseline.predict_holdout(train_series, test_series)

    # Persistence predictions:
    # Day 1 of test predicts last train value (12.0)
    # Day 2 of test predicts previous test value (13.0)
    # Day 3 of test predicts previous test value (14.0)
    np.testing.assert_array_equal(preds, [12.0, 13.0, 14.0])
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)


def test_persistence_7_days_forecast() -> None:
    last_val = 22.5
    last_date = pd.Timestamp("2023-05-10")

    forecast = PersistenceBaseline.forecast_7_days(last_val, last_date)
    assert len(forecast) == 7
    assert (forecast == 22.5).all()
    assert forecast.index[0] == pd.Timestamp("2023-05-11")
    assert forecast.index[-1] == pd.Timestamp("2023-05-17")
