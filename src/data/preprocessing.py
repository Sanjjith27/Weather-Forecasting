"""Preprocessing and missing observation handling for historical weather datasets."""

from __future__ import annotations

import logging
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from src.data.exceptions import DataValidationError
from src.data.validation import (
    validate_date_sequence,
    validate_weather_dataframe,
)

logger = logging.getLogger(__name__)

MissingHandlingStrategy = Literal["interpolate", "drop", "error"]


def create_weather_dataframe(
    dates: Sequence[str],
    temperatures: Sequence[float | None],
    column_name: str = "temperature_celsius",
) -> pd.DataFrame:
    """Construct an initial typed DataFrame from dates and raw temperature values.

    Args:
        dates: Sequence of date strings (YYYY-MM-DD).
        temperatures: Sequence of temperature values (numeric or None/NaN).
        column_name: Target column name for temperatures.

    Returns:
        pd.DataFrame with DatetimeIndex named 'date' and float column.

    Raises:
        DataValidationError: If dates cannot be parsed or lengths do not match.
    """
    if len(dates) != len(temperatures):
        raise DataValidationError(
            f"Dates ({len(dates)}) and temperatures ({len(temperatures)}) must have identical lengths."
        )

    dt_index = validate_date_sequence(dates)
    dt_index.name = "date"

    series = pd.to_numeric(temperatures, errors="coerce")
    df = pd.DataFrame({column_name: series}, index=dt_index)
    df["is_interpolated"] = False

    return df


def handle_missing_observations(
    df: pd.DataFrame,
    column: str = "temperature_celsius",
    strategy: MissingHandlingStrategy = "interpolate",
    max_consecutive_missing: int = 2,
    max_missing_ratio: float = 0.05,
) -> pd.DataFrame:
    """Handle missing observations in the time-series without inventing synthetic data.

    Policies:
    - 'error': Reject datasets with any missing readings.
    - 'drop': Drop rows containing NaN observations without filling values.
    - 'interpolate': Linearly interpolate isolated missing days (up to max_consecutive_missing)
      between surrounding measured readings. Explicitly marks 'is_interpolated' as True.
      Rejects datasets if consecutive missing days exceed the threshold or total missing ratio
      exceeds max_missing_ratio.

    Args:
        df: Input DataFrame with DatetimeIndex and temperature measurements.
        column: Column containing temperature observations.
        strategy: 'interpolate', 'drop', or 'error'.
        max_consecutive_missing: Maximum consecutive missing days allowed for interpolation.
        max_missing_ratio: Maximum total proportion of missing days allowed.

    Returns:
        DataFrame with handled missing values.

    Raises:
        DataValidationError: If missing values violate policy constraints.
    """
    if column not in df.columns:
        raise DataValidationError(f"Column '{column}' not found in DataFrame.")

    total_count = len(df)
    missing_mask = df[column].isna()
    missing_count = int(missing_mask.sum())

    if missing_count == 0:
        return df.copy()

    if strategy == "error":
        missing_dates = df.index[missing_mask].strftime("%Y-%m-%d").tolist()
        raise DataValidationError(
            f"Dataset contains {missing_count} missing observations on dates: {missing_dates[:5]}"
        )

    missing_ratio = missing_count / total_count
    if missing_ratio > max_missing_ratio:
        raise DataValidationError(
            f"Missing observation ratio {missing_ratio:.2%} ({missing_count}/{total_count}) "
            f"exceeds maximum allowed threshold of {max_missing_ratio:.2%}."
        )

    if strategy == "drop":
        logger.info("Dropping %d missing observation rows under 'drop' strategy.", missing_count)
        return df.dropna(subset=[column]).copy()

    if strategy == "interpolate":
        # Check consecutive missing streak length
        # Consecutive group helper using cumsum
        streak_lengths = missing_mask.groupby((~missing_mask).cumsum()).sum()
        max_streak = int(streak_lengths.max()) if not streak_lengths.empty else 0

        if max_streak > max_consecutive_missing:
            raise DataValidationError(
                f"Dataset contains {max_streak} consecutive missing days, exceeding the "
                f"maximum allowed limit of {max_consecutive_missing} consecutive missing days for interpolation. "
                "Refusing to interpolate across large gaps."
            )

        # Cannot interpolate if the first or last observation is missing
        if missing_mask.iloc[0] or missing_mask.iloc[-1]:
            raise DataValidationError(
                "Cannot reliably interpolate boundary observations (first or last observation is NaN). "
                "Boundary extrapolation is disallowed."
            )

        result_df = df.copy()
        result_df[column] = result_df[column].interpolate(method="linear", limit=max_consecutive_missing)
        result_df.loc[missing_mask, "is_interpolated"] = True

        logger.info(
            "Linearly interpolated %d isolated missing observation(s) (<= %d consecutive).",
            missing_count,
            max_consecutive_missing,
        )
        return result_df

    raise DataValidationError(f"Unknown missing handling strategy: '{strategy}'.")


def preprocess_weather_data(
    dates: Sequence[str],
    temperatures: Sequence[float | None],
    column_name: str = "temperature_celsius",
    strategy: MissingHandlingStrategy = "interpolate",
    min_samples: int = 30,
    max_missing_ratio: float = 0.05,
    max_consecutive_missing: int = 2,
) -> pd.DataFrame:
    """End-to-end preprocessing pipeline for raw weather observations.

    Transforms date strings and values into a validated, chronologically sorted,
    clean DataFrame ready for downstream time-series modeling.

    Args:
        dates: Sequence of date strings in YYYY-MM-DD format.
        temperatures: Sequence of temperature values.
        column_name: Name of target temperature column.
        strategy: Strategy for missing value handling ('interpolate', 'drop', 'error').
        min_samples: Minimum required observation count for LSTM processing.
        max_missing_ratio: Maximum proportion of missing observations.
        max_consecutive_missing: Maximum consecutive missing days for interpolation.

    Returns:
        Validated pd.DataFrame indexed by date.

    Raises:
        DataValidationError: If any validation or preprocessing step fails.
    """
    raw_df = create_weather_dataframe(dates, temperatures, column_name=column_name)

    clean_df = handle_missing_observations(
        raw_df,
        column=column_name,
        strategy=strategy,
        max_consecutive_missing=max_consecutive_missing,
        max_missing_ratio=max_missing_ratio,
    )

    validate_weather_dataframe(
        clean_df,
        temperature_column=column_name,
        min_samples=min_samples,
        max_missing_ratio=max_missing_ratio,
    )

    return clean_df
