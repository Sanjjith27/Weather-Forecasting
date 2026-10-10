"""Unit tests for the end-to-end LSTM modeling and forecasting pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.pipeline import ModelTrainingResult, train_and_forecast_pipeline


def _make_realistic_weather_series(n_days: int = 80) -> pd.Series:
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rng = np.random.RandomState(42)
    # Seasonal wave with noise around 20°C
    temps = 20.0 + 8.0 * np.sin(np.linspace(0, 2 * np.pi, n_days)) + rng.normal(0, 0.4, n_days)
    return pd.Series(temps, index=dates, name="temperature_celsius")


def test_end_to_end_training_and_forecast_pipeline(tmp_path: Path) -> None:
    series = _make_realistic_weather_series(80)

    result = train_and_forecast_pipeline(
        data=series,
        location_name="San Francisco",
        lookback=12,
        test_ratio=0.2,
        lstm_units=16,
        dense_units=8,
        epochs=3,
        batch_size=8,
        patience=2,
        models_dir=tmp_path / "models",
        use_cache=True,
        verbose=0,
    )

    assert isinstance(result, ModelTrainingResult)
    assert result.location_name == "San Francisco"
    assert result.lookback == 12
    assert not result.is_cached

    # Metrics
    assert "mae" in result.metrics_lstm
    assert "rmse" in result.metrics_lstm
    assert "mae" in result.metrics_baseline
    assert "rmse" in result.metrics_baseline
    assert result.metrics_lstm["mae"] > 0.0
    assert result.metrics_lstm["rmse"] > 0.0

    # Test predictions vs true values alignment
    assert len(result.y_test_pred) == len(result.y_test_true)
    assert len(result.test_dates) == len(result.y_test_true)
    assert np.all(np.isfinite(result.y_test_pred))

    # 7-day forecast verification
    assert len(result.forecast_7_days) == 7
    assert np.all(np.isfinite(result.forecast_7_days.values))
    # Correct future dates
    assert result.forecast_7_days.index[0] == series.index[-1] + pd.Timedelta(days=1)
    assert result.forecast_7_days.index[-1] == series.index[-1] + pd.Timedelta(days=7)

    # Summary
    summary = result.comparison_summary()
    assert summary["location"] == "San Francisco"
    assert "mae_improvement" in summary
    assert "rmse_improvement" in summary

    # Verify second execution hits cache
    cached_result = train_and_forecast_pipeline(
        data=series,
        location_name="San Francisco",
        lookback=12,
        test_ratio=0.2,
        lstm_units=16,
        dense_units=8,
        models_dir=tmp_path / "models",
        use_cache=True,
        verbose=0,
    )
    assert cached_result.is_cached
    assert cached_result.signature == result.signature
    np.testing.assert_allclose(cached_result.y_test_pred, result.y_test_pred, atol=1e-4)
