"""Reusable UI components for the Weather Forecasting Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st


def render_header(location_name: str, latest_date: pd.Timestamp | str) -> None:
    """Render top page heading with selected city and latest observation timestamp."""
    date_str = latest_date.strftime("%B %d, %Y") if isinstance(latest_date, pd.Timestamp) else str(latest_date)
    st.markdown(
        f"""
        <div class="header-container">
            <div class="brand-eyebrow">
                <span class="brand-dot"></span>
                Meteorological Intelligence Platform
            </div>
            <h1 class="main-title">Weather Forecasting Using LSTM</h1>
            <div class="sub-title">
                <span class="sub-title-item">Location: <strong>{location_name}</strong></span>
                <span class="sub-title-divider">•</span>
                <span class="sub-title-item">Latest Observation: <strong>{date_str}</strong></span>
                <span class="sub-title-divider">•</span>
                <span class="sub-title-item">Data Source: <strong>Open-Meteo Archive API</strong></span>
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
                <div class="metric-label">Mean Daily Temperature</div>
                <div class="metric-value">{summary.get('mean_temp_c', 0.0):.1f}°C</div>
                <div class="metric-sub">Across {summary.get('samples', 0)} daily observations</div>
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
                <div class="metric-sub">Recorded Min to Max (°C)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Observation Window</div>
                <div class="metric-value">{summary.get('samples', 0)} Days</div>
                <div class="metric-sub">{summary.get('start_date', '')} to {summary.get('end_date', '')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        interp = summary.get("interpolated_samples", 0)
        if interp > 0:
            quality_html = f'<span class="status-pill status-pill-info">{interp} interpolated</span>'
        else:
            quality_html = '<span class="status-pill status-pill-success">100% Measured</span>'

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Data Integrity</div>
                <div class="metric-value" style="font-size: 1.35rem; padding-top: 0.25rem;">{quality_html}</div>
                <div class="metric-sub">Verified daily continuity</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_forecast_cards(forecast_7d: pd.Series) -> None:
    """Render seven distinct cards showing dates and forecasted temperatures in Celsius."""
    st.markdown(
        """
        <div class="section-header-wrap">
            <h3 class="section-title">7-Day Temperature Forecast</h3>
            <span class="section-caption">Autoregressive multi-step recursive rollout in Celsius (°C)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    st.markdown(
        """
        <div class="section-header-wrap">
            <h3 class="section-title">Model Evaluation & Benchmark</h3>
            <span class="section-caption">Chronological holdout validation vs. Persistence Baseline (ŷ(t) = y(t-1))</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)

    lstm_mae = metrics_lstm.get("mae", 0.0)
    base_mae = metrics_baseline.get("mae", 0.0)
    mae_diff = base_mae - lstm_mae

    lstm_rmse = metrics_lstm.get("rmse", 0.0)
    base_rmse = metrics_baseline.get("rmse", 0.0)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">LSTM Holdout MAE</div>
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
                <div class="metric-label">LSTM Holdout RMSE</div>
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
                <div class="metric-sub">Naïve 1-day lag reference (RMSE: {base_rmse:.2f}°C)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        cache_status_html = (
            '<span class="status-pill status-pill-success">Cached Weights</span>'
            if is_cached
            else '<span class="status-pill status-pill-info">Freshly Trained</span>'
        )
        diff_sign = "+" if mae_diff >= 0 else ""
        diff_html = f'<span class="delta-tag delta-tag-neutral">Δ MAE: {diff_sign}{mae_diff:.2f}°C vs baseline</span>'

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Execution Provenance</div>
                <div class="metric-value" style="font-size: 1.35rem; padding-top: 0.25rem;">{cache_status_html}</div>
                <div class="metric-sub">{diff_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_lstm_explanation(lookback: int = 14) -> None:
    """Render educational collapsible explanation of LSTM mechanics."""
    with st.expander("Methodology & Architecture: How the LSTM Operates", expanded=False):
        st.markdown(
            f"""
            <div class="methodology-grid">
                <div class="methodology-card">
                    <h4>1. Lookback Window</h4>
                    <p>
                        The model samples a sliding context of <strong>{lookback} consecutive daily observations</strong>
                        $X_t = [x_{{t-{lookback}}}, \\dots, x_{{t-1}}]^T$ to predict target temperature $y_t$.
                    </p>
                </div>
                <div class="methodology-card">
                    <h4>2. Recurrent Cell Gating</h4>
                    <p>
                        Forget, input, and output gates modulate long-term memory across sequential steps,
                        filtering noise and preserving seasonal momentum without vanishing gradients.
                    </p>
                </div>
                <div class="methodology-card">
                    <h4>3. 7-Day Recursive Rollout</h4>
                    <p>
                        Future horizons are generated autoregressively: predicted step $t+1$ is appended to the
                        input buffer to forecast $t+2$, iterating for 7 consecutive days.
                    </p>
                </div>
                <div class="methodology-card">
                    <h4>4. Leakage-Free Validation</h4>
                    <p>
                        Features are scaled strictly using training partition parameters. Shuffling is prohibited;
                        evaluations use genuine chronological holdout observations.
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_disclaimer() -> None:
    """Render academic and experimental disclaimer banner."""
    st.markdown(
        """
        <div class="disclaimer-banner">
            <strong>Experimental Academic Notice:</strong>
            These predictions are generated by an experimental univariate LSTM recurrent neural network
            developed for academic modeling and demonstration. Because predictions rely solely on autoregressive
            temperature sequences without incorporating physical atmospheric variables (barometric pressure,
            wind vectors, front advection), they <strong>do not guarantee meteorological accuracy</strong>.
            For operational weather planning, always consult official national weather services.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render restrained, professional attribution footer."""
    st.markdown(
        """
        <div class="dashboard-footer">
            <div>
                Weather Forecasting Using LSTM &bull; Built with TensorFlow, Keras & Streamlit
            </div>
            <div>
                Data courtesy of <a href="https://open-meteo.com" target="_blank" rel="noopener noreferrer">Open-Meteo</a> &bull;
                <a href="https://github.com/Sanjjith27/Weather-Forecasting" target="_blank" rel="noopener noreferrer">GitHub Repository</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
