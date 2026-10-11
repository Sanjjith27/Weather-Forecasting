"""Unit tests for UI charts and component rendering logic."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.ui.charts import create_forecast_comparison_chart, create_historical_chart
from src.ui.components import (
    render_disclaimer,
    render_evaluation_metrics,
    render_footer,
    render_forecast_cards,
    render_header,
    render_lstm_explanation,
    render_summary_cards,
)
from src.ui.styles import LIGHT_THEME_CSS


def test_styles_defined() -> None:
    assert isinstance(LIGHT_THEME_CSS, str)
    assert ".stApp" in LIGHT_THEME_CSS
    assert ".metric-card" in LIGHT_THEME_CSS
    assert ".forecast-card" in LIGHT_THEME_CSS
    assert ".brand-eyebrow" in LIGHT_THEME_CSS
    assert ".methodology-grid" in LIGHT_THEME_CSS
    assert ".dashboard-footer" in LIGHT_THEME_CSS
    assert "tabular-nums" in LIGHT_THEME_CSS


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


@patch("streamlit.markdown")
def test_render_header_and_footer(mock_markdown: MagicMock) -> None:
    render_header("Munich, Germany", pd.Timestamp("2023-12-31"))
    assert mock_markdown.called
    header_html = mock_markdown.call_args[0][0]
    assert "Munich, Germany" in header_html
    assert "Meteorological Intelligence Platform" in header_html

    mock_markdown.reset_mock()
    render_footer()
    assert mock_markdown.called
    footer_html = mock_markdown.call_args[0][0]
    assert "Open-Meteo" in footer_html
    assert "GitHub Repository" in footer_html


@patch("streamlit.columns")
@patch("streamlit.markdown")
def test_render_summary_and_forecast_cards(mock_markdown: MagicMock, mock_columns: MagicMock) -> None:
    # Dynamically mock columns context based on argument
    mock_columns.side_effect = lambda n: [MagicMock() for _ in range(n)]

    summary = {
        "mean_temp_c": 12.5,
        "min_temp_c": 1.2,
        "max_temp_c": 28.4,
        "samples": 365,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "interpolated_samples": 0,
    }
    render_summary_cards(summary)
    assert mock_markdown.called

    dates = pd.date_range("2024-01-01", periods=7, freq="D")
    forecast_7d = pd.Series([10.5, 11.0, 11.2, 11.5, 12.0, 12.1, 12.4], index=dates)
    render_forecast_cards(forecast_7d)
    assert mock_markdown.called


@patch("streamlit.columns")
@patch("streamlit.markdown")
def test_render_evaluation_metrics_and_explanation(mock_markdown: MagicMock, mock_columns: MagicMock) -> None:
    mock_cols = [MagicMock() for _ in range(4)]
    mock_columns.return_value = mock_cols

    metrics_lstm = {"mae": 2.25, "rmse": 2.95}
    metrics_baseline = {"mae": 1.85, "rmse": 2.40}
    render_evaluation_metrics(metrics_lstm, metrics_baseline, is_cached=True)
    assert mock_markdown.called

    with patch("streamlit.expander") as mock_expander:
        mock_expander.return_value.__enter__.return_value = MagicMock()
        mock_expander.return_value.__exit__.return_value = None
        render_lstm_explanation(lookback=14)
        render_disclaimer()
        assert mock_markdown.called
