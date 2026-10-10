"""Weather Forecasting Using LSTM - Streamlit Application Entry Point."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from src.data.exceptions import GeocodingError, WeatherPipelineError
from src.data.pipeline import WeatherDataset, fetch_weather_by_city
from src.models.exceptions import InsufficientDataError, ModelEngineError
from src.models.pipeline import ModelTrainingResult, train_and_forecast_pipeline
from src.ui.charts import create_forecast_comparison_chart, create_historical_chart
from src.ui.components import (
    render_disclaimer,
    render_evaluation_metrics,
    render_forecast_cards,
    render_header,
    render_lstm_explanation,
    render_summary_cards,
)
from src.ui.styles import LIGHT_THEME_CSS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default curated cities for quick navigation
POPULAR_CITIES = [
    "London",
    "Tokyo",
    "New York",
    "Paris",
    "Berlin",
    "Chennai",
    "Sydney",
    "Toronto",
    "Custom...",
]


@st.cache_data(show_spinner=False, ttl=3600)
def get_cached_weather_data(city_name: str, start_str: str, end_str: str):
    """Retrieve and cache historical weather dataset for a city."""
    return fetch_weather_by_city(
        city_name=city_name,
        start_date=start_str,
        end_date=end_str,
        missing_strategy="interpolate",
        min_samples=30,
    )


@st.cache_data(show_spinner=False)
def get_cached_forecast_result(
    df: pd.DataFrame,
    location_name: str,
    lookback: int,
) -> ModelTrainingResult:
    """Train or load cached LSTM model and generate predictions and 7-day forecast."""
    return train_and_forecast_pipeline(
        data=df,
        location_name=location_name,
        lookback=lookback,
        test_ratio=0.2,
        epochs=30,
        batch_size=16,
        patience=5,
        use_cache=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Weather Forecasting Using LSTM",
        page_icon="⛅",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject custom styling
    st.markdown(LIGHT_THEME_CSS, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # Sidebar Controls
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown("### ⚙️ Location & Parameters")

        selected_preset = st.selectbox("Select City", options=POPULAR_CITIES, index=0)
        if selected_preset == "Custom...":
            city_query = st.text_input("Enter City Name", value="Munich", placeholder="e.g. Amsterdam, Chicago")
        else:
            city_query = selected_preset

        st.markdown("---")
        st.markdown("### 📅 Historical Observation Window")

        # Open-Meteo historical archive typically has a 3-5 day latency
        max_available_date = datetime.now().date() - timedelta(days=5)
        default_start_date = max_available_date - timedelta(days=365)

        start_date = st.date_input(
            "Start Date",
            value=default_start_date,
            max_value=max_available_date - timedelta(days=60),
        )
        end_date = st.date_input(
            "End Date",
            value=max_available_date,
            min_value=start_date + timedelta(days=60),
            max_value=max_available_date,
        )

        st.markdown("---")
        st.markdown("### 🧠 Model Hyperparameters")
        lookback_days = st.slider("Lookback Window (Days)", min_value=7, max_value=28, value=14, step=1)

        fetch_button = st.button("🔄 Update Forecast", use_container_width=True, type="primary")

    # --------------------------------------------------------------------------
    # Data Ingestion & Model Execution
    # --------------------------------------------------------------------------
    city_name = (city_query or "").strip()
    if not city_name:
        st.warning("Please enter a valid city name in the sidebar.")
        return

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    try:
        with st.spinner(f"Retrieving verified historical weather observations for {city_name}..."):
            dataset, location = get_cached_weather_data(city_name, start_str, end_str)

    except GeocodingError as err:
        st.error(f"📍 **Location Not Found:** {err}")
        st.info("Check city spelling or try selecting one of the preset locations in the sidebar.")
        return
    except WeatherPipelineError as err:
        st.error(f"🌐 **Data Retrieval Failed:** {err}")
        st.info("The weather service could not be reached or returned an unexpected format. Please retry shortly.")
        return
    except Exception as err:
        st.error(f"An unexpected error occurred while fetching observations: {err}")
        return

    # Train or load model
    try:
        with st.spinner(f"Computing LSTM sequence predictions and baseline benchmarking ({lookback_days}-day lookback)..."):
            forecast_result = get_cached_forecast_result(
                df=dataset.df,
                location_name=location.display_name(),
                lookback=lookback_days,
            )

    except InsufficientDataError as err:
        st.error(f"⚠️ **Insufficient History:** {err}")
        st.info("Extend the date range in the sidebar to ensure sufficient observations for training and holdout validation.")
        return
    except ModelEngineError as err:
        st.error(f"🧠 **Model Training Issue:** {err}")
        return
    except Exception as err:
        st.error(f"An unexpected error occurred during model forecasting: {err}")
        return

    # --------------------------------------------------------------------------
    # Dashboard View
    # --------------------------------------------------------------------------
    summary = dataset.summary()
    latest_date = dataset.df.index.max()

    # 1. Header with city and latest date
    render_header(location_name=location.display_name(), latest_date=latest_date)

    # 2. Historical Summary KPI cards
    render_summary_cards(summary)

    # 3. 7-Day Forecast Cards
    render_forecast_cards(forecast_result.forecast_7_days)

    # 4. Interactive Forecast & Historical Comparison Chart
    st.markdown('<div class="section-title">Forecasting Trajectory & Holdout Validation</div>', unsafe_allow_html=True)
    comparison_fig = create_forecast_comparison_chart(
        historical_df=dataset.df,
        test_dates=forecast_result.test_dates,
        y_test_true=forecast_result.y_test_true,
        y_test_pred=forecast_result.y_test_pred,
        forecast_7d=forecast_result.forecast_7_days,
        location_name=location.display_name(),
    )
    st.plotly_chart(comparison_fig, use_container_width=True)

    # 5. Model Evaluation Metrics vs Baseline
    render_evaluation_metrics(
        metrics_lstm=forecast_result.metrics_lstm,
        metrics_baseline=forecast_result.metrics_baseline,
        is_cached=forecast_result.is_cached,
    )

    # 6. Detailed Historical Timeline Chart
    with st.expander("📈 View Full Historical Daily Observation Timeline", expanded=False):
        history_fig = create_historical_chart(dataset.df, location.display_name())
        st.plotly_chart(history_fig, use_container_width=True)

    # 7. Model Explanation Section
    render_lstm_explanation(lookback=lookback_days)

    # 8. Disclaimer
    render_disclaimer()


if __name__ == "__main__":
    main()
