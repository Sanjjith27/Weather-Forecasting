"""Unit tests for UI charts and component rendering logic."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.ui.charts import create_forecast_comparison_chart, create_historical_chart
from src.ui.styles import LIGHT_THEME_CSS


def test_styles_defined() -> None:
    assert isinstance(LIGHT_THEME_CSS, str)
    assert ".stApp" in LIGHT_THEME_CSS
    assert ".metric-card" in LIGHT_THEME_CSS
    assert ".forecast-card" in LIGHT_THEME_CSS


def test_create_historical_chart() -> None:
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "temperature_celsius": [15.0 + i * 0.2 for i in range(30)],
            "is_interpolated": [False] * 29 + [True],
        },
        index=dates,
    )

    fig = create_historical_chart(df, location_name="Berlin")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1  # Base line trace + optional interpolated marker trace
    assert "Berlin" in fig.layout.title.text


def test_create_forecast_comparison_chart() -> None:
    all_dates = pd.date_range("2023-01-01", periods=60, freq="D")
    historical_df = pd.DataFrame(
        {"temperature_celsius": np.linspace(10.0, 20.0, 60)},
        index=all_dates,
    )

    test_dates = all_dates[45:]
    y_test_true = np.linspace(17.0, 20.0, 15)
    y_test_pred = np.linspace(16.8, 19.8, 15)

    future_dates = pd.date_range("2023-03-02", periods=7, freq="D")
    forecast_7d = pd.Series([20.1 + i * 0.1 for i in range(7)], index=future_dates)

    fig = create_forecast_comparison_chart(
        historical_df=historical_df,
        test_dates=test_dates,
        y_test_true=y_test_true,
        y_test_pred=y_test_pred,
        forecast_7d=forecast_7d,
        location_name="London",
    )

    assert isinstance(fig, go.Figure)
    # Check trace count (train history, test actual, test pred, 7-day forecast)
    assert len(fig.data) == 4
    trace_names = [t.name for t in fig.data]
    assert "Training History (Actual)" in trace_names
    assert "Holdout Test (Actual)" in trace_names
    assert "Holdout Test (LSTM Pred)" in trace_names
    assert "7-Day Future Forecast (LSTM)" in trace_names
