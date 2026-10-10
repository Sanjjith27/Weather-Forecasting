"""Unit tests for response validation, date sequences, and quality checks."""

import pandas as pd
import pytest

from src.data.exceptions import DataValidationError
from src.data.validation import (
    validate_api_response_schema,
    validate_date_sequence,
    validate_weather_dataframe,
)


def test_validate_api_response_schema_valid() -> None:
    payload = {
        "daily_units": {"time": "iso8601", "temperature_2m_mean": "°C"},
        "daily": {
            "time": ["2023-01-01", "2023-01-02"],
            "temperature_2m_mean": [12.5, 14.2],
        },
    }
    times, values = validate_api_response_schema(payload)
    assert times == ["2023-01-01", "2023-01-02"]
    assert values == [12.5, 14.2]


def test_validate_api_response_schema_missing_daily() -> None:
    payload = {"latitude": 52.52}
    with pytest.raises(DataValidationError, match="missing 'daily' dictionary"):
        validate_api_response_schema(payload)


def test_validate_api_response_schema_missing_variable() -> None:
    payload = {
        "daily_units": {"time": "iso8601"},
        "daily": {"time": ["2023-01-01"]},
    }
    with pytest.raises(DataValidationError, match="missing 'temperature_2m_mean'"):
        validate_api_response_schema(payload, daily_variable="temperature_2m_mean")


def test_validate_api_response_schema_mismatched_lengths() -> None:
    payload = {
        "daily_units": {"temperature_2m_mean": "°C"},
        "daily": {
            "time": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "temperature_2m_mean": [12.5],
        },
    }
    with pytest.raises(DataValidationError, match="Mismatched array lengths"):
        validate_api_response_schema(payload)


def test_validate_api_response_schema_invalid_unit() -> None:
    payload = {
        "daily_units": {"temperature_2m_mean": "°F"},
        "daily": {
            "time": ["2023-01-01"],
            "temperature_2m_mean": [65.0],
        },
    }
    with pytest.raises(DataValidationError, match="Unexpected unit.*expected Celsius"):
        validate_api_response_schema(payload)


def test_validate_api_response_schema_error_flag() -> None:
    payload = {"error": True, "reason": "Date range exceeded limits"}
    with pytest.raises(DataValidationError, match="API returned an error"):
        validate_api_response_schema(payload)


def test_validate_date_sequence_valid() -> None:
    dates = ["2023-01-01", "2023-01-02", "2023-01-03"]
    dt_index = validate_date_sequence(dates)
    assert len(dt_index) == 3
    assert dt_index.is_monotonic_increasing


def test_validate_date_sequence_non_chronological() -> None:
    dates = ["2023-01-03", "2023-01-01", "2023-01-02"]
    with pytest.raises(DataValidationError, match="strictly increasing chronological order"):
        validate_date_sequence(dates)


def test_validate_date_sequence_duplicates() -> None:
    dates = ["2023-01-01", "2023-01-02", "2023-01-02"]
    with pytest.raises(DataValidationError, match="contains duplicate dates"):
        validate_date_sequence(dates)


def test_validate_date_sequence_invalid_format() -> None:
    dates = ["01/01/2023", "02/01/2023"]
    with pytest.raises(DataValidationError, match="Failed to parse dates"):
        validate_date_sequence(dates)


def test_validate_weather_dataframe_insufficient_samples() -> None:
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({"temperature_celsius": range(10)}, index=dates)

    with pytest.raises(DataValidationError, match="Insufficient dataset length"):
        validate_weather_dataframe(df, min_samples=30)


def test_validate_weather_dataframe_excessive_missing() -> None:
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    temps = [20.0] * 30 + [None] * 10
    df = pd.DataFrame({"temperature_celsius": temps}, index=dates)

    with pytest.raises(DataValidationError, match="Missing observation ratio.*exceeds"):
        validate_weather_dataframe(df, min_samples=30, max_missing_ratio=0.10)


def test_validate_weather_dataframe_physical_bounds_exceeded() -> None:
    dates = pd.date_range("2023-01-01", periods=35, freq="D")
    # Impossibly hot temperature
    temps = [20.0] * 34 + [85.0]
    df = pd.DataFrame({"temperature_celsius": temps}, index=dates)

    with pytest.raises(DataValidationError, match="out of physical bounds"):
        validate_weather_dataframe(df, min_samples=30)


def test_validate_weather_dataframe_flatline() -> None:
    dates = pd.date_range("2023-01-01", periods=35, freq="D")
    # All identical temperatures (zero variance)
    df = pd.DataFrame({"temperature_celsius": [22.0] * 35}, index=dates)

    with pytest.raises(DataValidationError, match="Zero variance detected"):
        validate_weather_dataframe(df, min_samples=30)
