"""Plotly interactive chart builders for historical observations and LSTM forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _apply_light_theme_layout(fig: go.Figure, title_text: str, y_title: str = "Temperature (°C)") -> None:
    """Apply consistent clean, precision light aesthetic to Plotly figures."""
    fig.update_layout(
        title={
            "text": title_text,
            "font": {"family": "Plus Jakarta Sans, -apple-system, sans-serif", "size": 15, "color": "#0f172a"},
            "x": 0.01,
            "xanchor": "left",
        },
        template="plotly_white",
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font={"family": "Plus Jakarta Sans, -apple-system, sans-serif", "color": "#334155", "size": 12},
        margin={"l": 45, "r": 25, "t": 55, "b": 40},
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "#0f172a",
            "font": {"family": "Plus Jakarta Sans, sans-serif", "size": 12, "color": "#f8fafc"},
            "bordercolor": "#334155",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
            "font": {"size": 11, "color": "#475569"},
        },
        xaxis={
            "showgrid": True,
            "gridcolor": "#f1f5f9",
            "gridwidth": 1,
            "linecolor": "#cbd5e1",
            "linewidth": 1,
            "tickformat": "%b %d",
            "hoverformat": "%a, %b %d, %Y",
            "tickfont": {"size": 11, "color": "#64748b"},
        },
        yaxis={
            "title": {"text": y_title, "font": {"size": 12, "color": "#475569"}},
            "showgrid": True,
            "gridcolor": "#f1f5f9",
            "gridwidth": 1,
            "linecolor": "#cbd5e1",
            "linewidth": 1,
            "ticksuffix": "°C",
            "tickfont": {"size": 11, "color": "#64748b"},
            "zeroline": True,
            "zerolinecolor": "#e2e8f0",
            "zerolinewidth": 1,
        },
    )


def create_historical_chart(df: pd.DataFrame, location_name: str) -> go.Figure:
    """Build interactive historical temperature timeline chart.

    Args:
        df: DataFrame indexed by DatetimeIndex with 'temperature_celsius'.
        location_name: Name of the geocoded city or coordinates.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["temperature_celsius"],
            name="Daily Mean Temp",
            mode="lines",
            line={"color": "#0284c7", "width": 2},
            fill="tozeroy",
            fillcolor="rgba(2, 132, 199, 0.05)",
            hovertemplate="%{y:.1f}°C<extra>Historical Observation</extra>",
        )
    )

    # Highlight any interpolated days if present
    if "is_interpolated" in df.columns and df["is_interpolated"].any():
        interp_subset = df[df["is_interpolated"]]
        fig.add_trace(
            go.Scatter(
                x=interp_subset.index,
                y=interp_subset["temperature_celsius"],
                name="Interpolated Observations",
                mode="markers",
                marker={"color": "#d97706", "size": 6, "symbol": "circle-open", "line": {"width": 1.5}},
                hovertemplate="%{y:.1f}°C (Interpolated)<extra></extra>",
            )
        )

    _apply_light_theme_layout(
        fig,
        title_text=f"Historical Daily Mean Temperature — {location_name}",
        y_title="Mean Temperature (°C)",
    )

    return fig


def create_forecast_comparison_chart(
    historical_df: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    forecast_7d: pd.Series,
    location_name: str,
) -> go.Figure:
    """Build a comprehensive chart distinguishing ground truth, LSTM estimates, and 7-day forecast.

    Args:
        historical_df: Full historical DataFrame.
        test_dates: DatetimeIndex of the holdout test period.
        y_test_true: Actual holdout values in °C.
        y_test_pred: LSTM holdout estimates in °C.
        forecast_7d: 7-day recursive forecast Series in °C.
        location_name: Name of the location.

    Returns:
        Plotly Figure.
    """
    fig = go.Figure()

    # 1. Historical Observations (pre-test training history)
    split_date = test_dates[0]
    train_history = historical_df.loc[historical_df.index < split_date]

    if not train_history.empty:
        fig.add_trace(
            go.Scatter(
                x=train_history.index,
                y=train_history["temperature_celsius"],
                name="Training History (Actual)",
                mode="lines",
                line={"color": "#94a3b8", "width": 1.5},
                hovertemplate="%{y:.1f}°C<extra>Train History</extra>",
            )
        )

    # 2. Holdout Test Set: Actual Ground Truth
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=y_test_true,
            name="Holdout Test (Actual)",
            mode="lines",
            line={"color": "#0f172a", "width": 2.2},
            hovertemplate="%{y:.1f}°C<extra>Holdout Actual</extra>",
        )
    )

    # 3. Holdout Test Set: LSTM Predictions
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=y_test_pred,
            name="Holdout Test (LSTM Pred)",
            mode="lines",
            line={"color": "#0284c7", "width": 2.2, "dash": "dash"},
            hovertemplate="%{y:.1f}°C<extra>LSTM Prediction</extra>",
        )
    )

    # 4. 7-Day Future Forecast
    # Prepend last known test observation to connect lines smoothly
    connect_date = historical_df.index[-1]
    connect_val = float(historical_df["temperature_celsius"].iloc[-1])

    forecast_x = [connect_date] + list(forecast_7d.index)
    forecast_y = [connect_val] + list(forecast_7d.values)

    fig.add_trace(
        go.Scatter(
            x=forecast_x,
            y=forecast_y,
            name="7-Day Future Forecast (LSTM)",
            mode="lines+markers",
            line={"color": "#d97706", "width": 2.5},
            marker={"size": 6, "color": "#d97706"},
            hovertemplate="%{y:.1f}°C<extra>Future Forecast</extra>",
        )
    )

    # Add vertical divider and shaded region for the forecast window
    fig.add_vrect(
        x0=forecast_7d.index[0] - pd.Timedelta(hours=12),
        x1=forecast_7d.index[-1] + pd.Timedelta(hours=12),
        fillcolor="rgba(217, 119, 6, 0.07)",
        layer="below",
        line_width=0,
        annotation_text="7-Day Forecast Horizon",
        annotation_position="top left",
        annotation_font={"size": 10, "color": "#b45309", "family": "Plus Jakarta Sans, sans-serif"},
    )

    _apply_light_theme_layout(
        fig,
        title_text=f"Historical Observations vs. LSTM Forecast — {location_name}",
        y_title="Temperature (°C)",
    )

    return fig
