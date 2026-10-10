"""Validation logic for weather API responses and time-series dataframes."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.data.exceptions import DataValidationError

logger = logging.getLogger(__name__)

# Meteorological physical plausibility bounds for surface temperatures in Celsius
MIN_PHYSICAL_TEMPERATURE_C = -90.0  # Coldest recorded temperature on Earth: ~ -89.2°C (Vostok Station)
MAX_PHYSICAL_TEMPERATURE_C = 60.0   # Hottest recorded temperature on Earth: ~ 56.7°C (Death Valley)
EXPECTED_CELSIUS_UNITS = {"°C", "C", "celsius"}


def validate_api_response_schema(
    payload: dict[str, Any],
    daily_variable: str = "temperature_2m_mean",
) -> tuple[list[str], list[Any]]:
    """Validate raw Open-Meteo archive JSON response schema and extract time/value series.

    Args:
        payload: Decoded JSON response from the Open-Meteo archive API.
        daily_variable: Name of the daily meteorological parameter.

    Returns:
        Tuple of (dates_list, values_list).

    Raises:
        DataValidationError: If structure is missing required fields, units mismatch, or lengths differ.
    """
    if not isinstance(payload, dict):
        raise DataValidationError(f"Invalid API response: expected JSON object, got {type(payload)}.")

    # Check for API error response message
    if payload.get("error") is True:
        reason = payload.get("reason", "Unknown API error")
        raise DataValidationError(f"API returned an error: {reason}")

    # Validate daily block
    daily = payload.get("daily")
    if not isinstance(daily, dict):
        raise DataValidationError("API response missing 'daily' dictionary block.")

    # Validate unit specification
    daily_units = payload.get("daily_units", {})
    if isinstance(daily_units, dict) and daily_variable in daily_units:
        unit = str(daily_units[daily_variable]).strip()
        if unit not in EXPECTED_CELSIUS_UNITS:
            raise DataValidationError(
                f"Unexpected unit for '{daily_variable}': expected Celsius ({EXPECTED_CELSIUS_UNITS}), got '{unit}'."
            )

    # Validate time series and parameter series
    if "time" not in daily:
        raise DataValidationError("API response daily block missing 'time' field.")
    if daily_variable not in daily:
        raise DataValidationError(f"API response daily block missing '{daily_variable}' field.")

    times = daily["time"]
    values = daily[daily_variable]

    if not isinstance(times, list) or not isinstance(values, list):
        raise DataValidationError("Daily 'time' and variable values must both be JSON lists.")

    if len(times) == 0:
        raise DataValidationError("API response returned an empty dataset (0 observations).")

    if len(times) != len(values):
        raise DataValidationError(
            f"Mismatched array lengths: 'time' has {len(times)} entries, "
            f"but '{daily_variable}' has {len(values)} entries."
        )

    return times, values


def validate_date_sequence(dates: Sequence[str]) -> pd.DatetimeIndex:
    """Validate that date strings are valid, unique, and strictly chronological.

    Args:
        dates: Sequence of date strings in YYYY-MM-DD format.

    Returns:
        Validated pd.DatetimeIndex.

    Raises:
        DataValidationError: If dates are invalid, duplicated, or not strictly increasing.
    """
    try:
        dt_index = pd.to_datetime(dates, format="%Y-%m-%d", errors="raise")
    except Exception as err:
        raise DataValidationError(f"Failed to parse dates into ISO YYYY-MM-DD format: {err}") from err

    # Check for duplicates
    if dt_index.has_duplicates:
        duplicated = dt_index[dt_index.duplicated()].tolist()
        raise DataValidationError(f"Dataset contains duplicate dates: {duplicated[:5]}")

    # Check strictly monotonic increasing order
    if not dt_index.is_monotonic_increasing:
        raise DataValidationError("Dataset dates are not in strictly increasing chronological order.")

    return dt_index


def validate_weather_dataframe(
    df: pd.DataFrame,
    temperature_column: str = "temperature_celsius",
    min_samples: int = 30,
    max_missing_ratio: float = 0.05,
    min_temp: float = MIN_PHYSICAL_TEMPERATURE_C,
    max_temp: float = MAX_PHYSICAL_TEMPERATURE_C,
) -> None:
    """Validate dataset length, missingness ratio, variance, and physical plausibility.

    Args:
        df: Processed DataFrame indexed by DatetimeIndex with temperature measurements.
        temperature_column: Column name containing temperature measurements.
        min_samples: Minimum required observations for LSTM time series analysis.
        max_missing_ratio: Maximum allowable proportion of missing observations (0.0 to 1.0).
        min_temp: Lower physical temperature limit in Celsius.
        max_temp: Upper physical temperature limit in Celsius.

    Raises:
        DataValidationError: If any quality check fails.
    """
    if df.empty:
        raise DataValidationError("Weather dataset is empty.")

    if temperature_column not in df.columns:
        raise DataValidationError(f"Missing required column '{temperature_column}' in DataFrame.")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("DataFrame must have a pd.DatetimeIndex.")

    # Validate dataset length
    total_samples = len(df)
    if total_samples < min_samples:
        raise DataValidationError(
            f"Insufficient dataset length: {total_samples} samples found, "
            f"but at least {min_samples} samples are required for LSTM modeling."
        )

    # Missing value check
    series = df[temperature_column]
    missing_count = series.isna().sum()
    missing_ratio = missing_count / total_samples

    if missing_ratio > max_missing_ratio:
        raise DataValidationError(
            f"Missing observation ratio {missing_ratio:.2%} ({missing_count}/{total_samples}) "
            f"exceeds maximum allowed threshold of {max_missing_ratio:.2%}."
        )

    # Physical temperature sanity bounds on available observations
    valid_values = series.dropna()
    if valid_values.empty:
        raise DataValidationError("All temperature observations are missing/NaN.")

    lowest = valid_values.min()
    highest = valid_values.max()

    if lowest < min_temp or highest > max_temp:
        raise DataValidationError(
            f"Temperature observations out of physical bounds [{min_temp}°C, {max_temp}°C]: "
            f"observed range is [{lowest:.2f}°C, {highest:.2f}°C]."
        )

    # Flatline / constant value check (sensor freeze or corrupt mock)
    if len(valid_values) >= 10:
        std_dev = valid_values.std()
        if np.isclose(std_dev, 0.0) or std_dev < 1e-6:
            raise DataValidationError(
                f"Zero variance detected in temperature data (std={std_dev:.6f}). "
                "Constant flatline readings indicate corrupted or frozen sensor observations."
            )
