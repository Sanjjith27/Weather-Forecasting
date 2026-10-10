"""High-level weather data pipeline coordinating geocoding, retrieval, and preprocessing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.data.client import HistoricalWeatherClient
from src.data.geocoding import LocationResult, geocode_city
from src.data.preprocessing import MissingHandlingStrategy, preprocess_weather_data
from src.data.validation import validate_api_response_schema

logger = logging.getLogger(__name__)


@dataclass
class WeatherDataset:
    """Encapsulates a processed historical weather dataset with provenance metadata."""

    df: pd.DataFrame
    location_name: str
    latitude: float
    longitude: float
    start_date: str
    end_date: str
    variable: str = "temperature_2m_mean"
    unit: str = "°C"

    @property
    def sample_count(self) -> int:
        return len(self.df)

    def summary(self) -> dict[str, float | str | int]:
        """Compute basic summary statistics of the temperature time series."""
        temp_series = self.df["temperature_celsius"]
        return {
            "location": self.location_name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "samples": len(self.df),
            "start_date": self.df.index.min().strftime("%Y-%m-%d"),
            "end_date": self.df.index.max().strftime("%Y-%m-%d"),
            "mean_temp_c": round(float(temp_series.mean()), 2),
            "min_temp_c": round(float(temp_series.min()), 2),
            "max_temp_c": round(float(temp_series.max()), 2),
            "interpolated_samples": int(self.df["is_interpolated"].sum()),
        }


def fetch_weather_by_coordinates(
    latitude: float,
    longitude: float,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    location_name: str = "Custom Coordinates",
    daily_variable: str = "temperature_2m_mean",
    temperature_unit: str = "celsius",
    timezone: str = "UTC",
    missing_strategy: MissingHandlingStrategy = "interpolate",
    min_samples: int = 30,
    max_missing_ratio: float = 0.05,
    max_consecutive_missing: int = 2,
    enable_cache: bool = True,
    cache_dir: Optional[Path | str] = None,
    timeout: float = 10.0,
    session: Optional[requests.Session] = None,
    client: Optional[HistoricalWeatherClient] = None,
) -> WeatherDataset:
    """Fetch, validate, and preprocess historical weather data for given coordinates.

    Args:
        latitude: Geographical latitude.
        longitude: Geographical longitude.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        location_name: Label for metadata tracking.
        daily_variable: Daily observation variable name.
        temperature_unit: Temperature unit ('celsius').
        timezone: Timezone string.
        missing_strategy: Missing observations policy ('interpolate', 'drop', 'error').
        min_samples: Minimum required observations.
        max_missing_ratio: Maximum proportion of missing observations.
        max_consecutive_missing: Max consecutive missing observations for interpolation.
        enable_cache: Whether to use disk caching.
        cache_dir: Custom cache directory path.
        timeout: Network timeout in seconds.
        session: Custom requests session.
        client: Optional pre-configured HistoricalWeatherClient.

    Returns:
        WeatherDataset instance containing validated DataFrame and metadata.
    """
    weather_client = client or HistoricalWeatherClient(
        timeout=timeout,
        cache_dir=cache_dir,
        enable_cache=enable_cache,
        session=session,
    )

    raw_payload = weather_client.fetch_historical_weather_raw(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        daily_variable=daily_variable,
        temperature_unit=temperature_unit,
        timezone=timezone,
    )

    dates, values = validate_api_response_schema(raw_payload, daily_variable=daily_variable)

    clean_df = preprocess_weather_data(
        dates=dates,
        temperatures=values,
        strategy=missing_strategy,
        min_samples=min_samples,
        max_missing_ratio=max_missing_ratio,
        max_consecutive_missing=max_consecutive_missing,
    )

    start_str = clean_df.index.min().strftime("%Y-%m-%d")
    end_str = clean_df.index.max().strftime("%Y-%m-%d")

    return WeatherDataset(
        df=clean_df,
        location_name=location_name,
        latitude=latitude,
        longitude=longitude,
        start_date=start_str,
        end_date=end_str,
        variable=daily_variable,
        unit="°C",
    )


def fetch_weather_by_city(
    city_name: str,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    missing_strategy: MissingHandlingStrategy = "interpolate",
    min_samples: int = 30,
    max_missing_ratio: float = 0.05,
    max_consecutive_missing: int = 2,
    enable_cache: bool = True,
    cache_dir: Optional[Path | str] = None,
    timeout: float = 10.0,
    session: Optional[requests.Session] = None,
) -> tuple[WeatherDataset, LocationResult]:
    """Geocode a city and fetch its historical weather time series in one unified call.

    Args:
        city_name: Name of the city to geocode and retrieve.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        missing_strategy: Missing observations policy ('interpolate', 'drop', 'error').
        min_samples: Minimum required observations.
        max_missing_ratio: Maximum proportion of missing observations.
        max_consecutive_missing: Max consecutive missing observations for interpolation.
        enable_cache: Whether to use disk caching.
        cache_dir: Custom cache directory path.
        timeout: Network timeout in seconds.
        session: Custom requests session.

    Returns:
        Tuple of (WeatherDataset, LocationResult).
    """
    location = geocode_city(city_name, timeout=timeout, session=session)
    logger.info("Geocoded '%s' to %s (lat=%f, lon=%f)", city_name, location.display_name(), location.latitude, location.longitude)

    dataset = fetch_weather_by_coordinates(
        latitude=location.latitude,
        longitude=location.longitude,
        start_date=start_date,
        end_date=end_date,
        location_name=location.display_name(),
        missing_strategy=missing_strategy,
        min_samples=min_samples,
        max_missing_ratio=max_missing_ratio,
        max_consecutive_missing=max_consecutive_missing,
        enable_cache=enable_cache,
        cache_dir=cache_dir,
        timeout=timeout,
        session=session,
    )

    return dataset, location
