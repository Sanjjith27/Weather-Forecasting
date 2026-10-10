"""Unit tests for missing value handling and preprocessing."""

import pandas as pd
import pytest

from src.data.exceptions import DataValidationError
from src.data.preprocessing import (
    create_weather_dataframe,
    handle_missing_observations,
    preprocess_weather_data,
)


def test_create_weather_dataframe() -> None:
    dates = ["2023-01-01", "2023-01-02", "2023-01-03"]
    temps = [10.5, 11.2, 12.0]
    df = create_weather_dataframe(dates, temps)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "date"
    assert "temperature_celsius" in df.columns
    assert (df["temperature_celsius"] == temps).all()
    assert (df["is_interpolated"] == False).all()  # noqa: E712


def test_handle_missing_no_missing() -> None:
    dates = ["2023-01-01", "2023-01-02", "2023-01-03"]
    temps = [10.5, 11.2, 12.0]
    df = create_weather_dataframe(dates, temps)

    cleaned = handle_missing_observations(df)
    assert cleaned["temperature_celsius"].isna().sum() == 0
    assert (cleaned["is_interpolated"] == False).all()  # noqa: E712


def test_handle_missing_strategy_error() -> None:
    dates = ["2023-01-01", "2023-01-02", "2023-01-03"]
    temps = [10.5, None, 12.0]
    df = create_weather_dataframe(dates, temps)

    with pytest.raises(DataValidationError, match="missing observations"):
        handle_missing_observations(df, strategy="error")


def test_handle_missing_strategy_drop() -> None:
    dates = ["2023-01-01", "2023-01-02", "2023-01-03"]
    temps = [10.5, None, 12.0]
    df = create_weather_dataframe(dates, temps)

    cleaned = handle_missing_observations(df, strategy="drop", max_missing_ratio=0.5)
    assert len(cleaned) == 2
    assert cleaned.index.tolist() == [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-03")]
    assert cleaned["temperature_celsius"].isna().sum() == 0


def test_handle_missing_strategy_interpolate_success() -> None:
    dates = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
    # 2023-01-02 is missing between 10.0 and 12.0 -> interpolated to 11.0
    temps = [10.0, None, 12.0, 13.0]
    df = create_weather_dataframe(dates, temps)

    cleaned = handle_missing_observations(
        df, strategy="interpolate", max_consecutive_missing=2, max_missing_ratio=0.5
    )
    assert cleaned["temperature_celsius"].isna().sum() == 0
    assert cleaned.loc["2023-01-02", "temperature_celsius"] == pytest.approx(11.0)
    assert cleaned.loc["2023-01-02", "is_interpolated"] == True  # noqa: E712
    assert cleaned.loc["2023-01-01", "is_interpolated"] == False  # noqa: E712


def test_handle_missing_strategy_interpolate_exceeding_gap() -> None:
    dates = ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
    # 3 consecutive missing days with max_consecutive_missing=2
    temps = [10.0, None, None, None, 14.0]
    df = create_weather_dataframe(dates, temps)

    with pytest.raises(DataValidationError, match="Refusing to interpolate across large gaps"):
        handle_missing_observations(
            df, strategy="interpolate", max_consecutive_missing=2, max_missing_ratio=0.8
        )


def test_handle_missing_strategy_interpolate_boundary_nan() -> None:
    dates = ["2023-01-01", "2023-01-02", "2023-01-03"]
    # Missing at the very beginning
    temps = [None, 11.0, 12.0]
    df = create_weather_dataframe(dates, temps)

    with pytest.raises(DataValidationError, match="Cannot reliably interpolate boundary"):
        handle_missing_observations(df, strategy="interpolate", max_missing_ratio=0.5)


def test_preprocess_weather_data_end_to_end() -> None:
    dates = pd.date_range("2023-01-01", periods=35, freq="D").strftime("%Y-%m-%d").tolist()
    temps = [15.0 + (i % 5) for i in range(1, 36)]
    temps[10] = None  # Single missing day

    df = preprocess_weather_data(dates, temps, min_samples=30)
    assert len(df) == 35
    assert df["temperature_celsius"].isna().sum() == 0
    assert df.loc["2023-01-11", "is_interpolated"] == True  # noqa: E712
