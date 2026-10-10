"""Reusable UI components for the Weather Forecasting Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_header(location_name: str, latest_date: pd.Timestamp | str) -> None:
    """Render top page heading with selected city and latest observation timestamp."""
    date_str = latest_date.strftime("%B %d, %Y") if isinstance(latest_date, pd.Timestamp) else str(latest_date)
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <div class="main-title">Weather Forecasting Using LSTM</div>
            <div class="sub-title">
                Selected Location: <strong>{location_name}</strong> &nbsp;|&nbsp;
                Latest Available Observation: <strong>{date_str}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_cards(summary: dict) -> None:
    """Render historical meteorological summary cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Mean Temperature</div>
                <div class="metric-value">{summary.get('mean_temp_c', 0.0):.1f}°C</div>
                <div class="metric-sub">Over {summary.get('samples', 0)} daily records</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Observed Range</div>
                <div class="metric-value">{summary.get('min_temp_c', 0.0):.1f}° to {summary.get('max_temp_c', 0.0):.1f}°</div>
                <div class="metric-sub">Min to Max (°C)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Data Span</div>
                <div class="metric-value">{summary.get('samples', 0)} Days</div>
                <div class="metric-sub">{summary.get('start_date', '')} to {summary.get('end_date', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        interp = summary.get("interpolated_samples", 0)
        badge_text = f"{interp} interpolated" if interp > 0 else "100% measured"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Data Quality</div>
                <div class="metric-value">{badge_text}</div>
                <div class="metric-sub">Open-Meteo Archive API</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_forecast_cards(forecast_7d: pd.Series) -> None:
    """Render seven distinct cards showing dates and forecasted temperatures in Celsius."""
    st.markdown('<div class="section-title">7-Day Temperature Forecast (LSTM)</div>', unsafe_allow_html=True)

    cols = st.columns(7)
    for i, (date, temp) in enumerate(forecast_7d.items()):
        day_name = date.strftime("%a")
        date_formatted = date.strftime("%b %d")
        with cols[i]:
            st.markdown(
                f"""
                <div class="forecast-card">
                    <div class="forecast-day">{day_name}</div>
                    <div class="forecast-date">{date_formatted}</div>
                    <div class="forecast-temp">{temp:.1f}°C</div>
                    <div class="forecast-badge">Day +{i + 1}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_evaluation_metrics(
    metrics_lstm: dict[str, float],
    metrics_baseline: dict[str, float],
    is_cached: bool,
) -> None:
    """Render honest holdout evaluation metrics comparing LSTM against the persistence baseline."""
    st.markdown('<div class="section-title">Model Performance & Baseline Benchmarking</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    lstm_mae = metrics_lstm.get("mae", 0.0)
    base_mae = metrics_baseline.get("mae", 0.0)
    mae_diff = base_mae - lstm_mae

    lstm_rmse = metrics_lstm.get("rmse", 0.0)
    base_rmse = metrics_baseline.get("rmse", 0.0)
    rmse_diff = base_rmse - lstm_rmse

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">LSTM MAE (Holdout)</div>
                <div class="metric-value">{lstm_mae:.2f}°C</div>
                <div class="metric-sub">Mean Absolute Error</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">LSTM RMSE (Holdout)</div>
                <div class="metric-value">{lstm_rmse:.2f}°C</div>
                <div class="metric-sub">Root Mean Squared Error</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Persistence Baseline MAE</div>
                <div class="metric-value">{base_mae:.2f}°C</div>
                <div class="metric-sub">Naïve y(t) = y(t-1)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        improvement_text = f"+{mae_diff:.2f}°C vs baseline" if mae_diff >= 0 else f"{mae_diff:.2f}°C vs baseline"
        cache_status = "Cached Model" if is_cached else "Trained Model"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Model Status</div>
                <div class="metric-value">{cache_status}</div>
                <div class="metric-sub">MAE delta: {improvement_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_lstm_explanation(lookback: int = 14) -> None:
    """Render educational collapsible explanation of LSTM mechanics."""
    with st.expander("ℹ️ How the LSTM Temperature Forecasting Model Works", expanded=False):
        st.markdown(
            f"""
            <div class="explanation-card">
                <h4>Recurrent Neural Network Architecture</h4>
                <p>
                    Long Short-Term Memory (LSTM) networks are a specialized class of recurrent neural networks
                    designed to learn long-range temporal dependencies without suffering from vanishing gradients.
                </p>
                <ul>
                    <li><strong>Sliding-Window Lookback:</strong> The model ingests a rolling context window of
                        <strong>{lookback} consecutive daily observations</strong> to forecast the subsequent day's temperature.</li>
                    <li><strong>Memory & Gating Mechanism:</strong> Each LSTM cell maintains a cell state regulated by three gates:
                        the <em>forget gate</em> (discarding irrelevant historical noise), the <em>input gate</em> (assimilating new seasonal shifts),
                        and the <em>output gate</em> (producing the hidden activation).</li>
                    <li><strong>Recursive Multi-Step Forecasting:</strong> To predict 7 days ahead, the model predicts day <em>t+1</em>,
                        rolls the window forward by dropping the oldest observation, and feeds its own prediction back as input.</li>
                    <li><strong>Leakage Prevention:</strong> Min-Max normalization is fitted exclusively on training observations,
                        and evaluations are strictly performed on un-shuffled chronological holdout test sets.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_disclaimer() -> None:
    """Render academic and experimental disclaimer banner."""
    st.markdown(
        """
        <div class="disclaimer-banner">
            <strong>⚠️ Experimental Academic Model Disclaimer:</strong>
            These predictions are generated by a univariate LSTM recurrent neural network developed for
            academic research and experimentation. They are based purely on historical temperature sequence
            extrapolation without exogenous atmospheric physics (barometric pressure gradients, wind velocity,
            or moisture advection) and <strong>do not guarantee reliable meteorological forecasts</strong>.
            Please consult national weather services (e.g., NOAA, Met Office, DWD, IMD) for official forecasts.
        </div>
        """,
        unsafe_allow_html=True,
    )
