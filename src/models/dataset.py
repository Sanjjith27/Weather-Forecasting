"""Dataset preparation, chronological splitting, scaling, and sliding-window generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.models.exceptions import InsufficientDataError

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesDataset:
    """Encapsulates prepared time series data for training, validation, and testing."""

    X_train: np.ndarray
    y_train: np.ndarray
    train_dates: pd.DatetimeIndex
    X_test: np.ndarray
    y_test: np.ndarray
    test_dates: pd.DatetimeIndex
    raw_train_series: pd.Series
    raw_test_series: pd.Series
    scaler: MinMaxScaler
    lookback: int

    @property
    def total_train_sequences(self) -> int:
        return int(len(self.X_train))

    @property
    def total_test_sequences(self) -> int:
        return int(len(self.X_test))


def validate_history_length(
    total_samples: int,
    lookback: int,
    test_size: int,
    min_train_samples: int = 30,
) -> None:
    """Validate that available observations are sufficient for lookback, training, and testing.

    Args:
        total_samples: Total number of observations in dataset.
        lookback: Number of past time steps used as input.
        test_size: Number of observations reserved for holdout testing.
        min_train_samples: Minimum required training targets.

    Raises:
        InsufficientDataError: If observation count is insufficient.
    """
    min_required = lookback + min_train_samples + test_size
    if total_samples < min_required:
        raise InsufficientDataError(
            f"Insufficient time-series history: available samples ({total_samples}) is less than "
            f"the minimum required ({min_required} samples = lookback {lookback} + "
            f"min train targets {min_train_samples} + test holdout {test_size})."
        )


def chronological_split(
    series: pd.Series,
    test_ratio: float = 0.2,
    min_test_samples: int = 7,
) -> Tuple[pd.Series, pd.Series]:
    """Split time-series observations chronologically without random shuffling.

    Args:
        series: Pandas Series indexed by DatetimeIndex in chronological order.
        test_ratio: Proportion of observations reserved for testing (0.0 < test_ratio < 1.0).
        min_test_samples: Minimum count of test samples.

    Returns:
        Tuple of (train_series, test_series).
    """
    total = len(series)
    test_count = max(min_test_samples, int(total * test_ratio))
    split_idx = total - test_count

    train_series = series.iloc[:split_idx]
    test_series = series.iloc[split_idx:]

    return train_series, test_series


def create_sliding_windows(
    data: np.ndarray,
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create input window sequences and corresponding 1-step targets.

    Args:
        data: 1D or 2D array of time series values of length N.
        lookback: Sequence window length L.

    Returns:
        X: Array of shape (N - lookback, lookback, 1)
        y: Array of shape (N - lookback, 1)
    """
    flat = np.asarray(data).reshape(-1, 1)
    n_samples = len(flat)

    if n_samples <= lookback:
        raise InsufficientDataError(
            f"Cannot create sliding windows of lookback {lookback} with only {n_samples} samples."
        )

    X_list = []
    y_list = []

    for i in range(n_samples - lookback):
        X_list.append(flat[i : i + lookback])
        y_list.append(flat[i + lookback])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    return X, y


def prepare_time_series_dataset(
    series: pd.Series,
    lookback: int = 14,
    test_ratio: float = 0.2,
    min_train_samples: int = 30,
) -> TimeSeriesDataset:
    """Full data preparation pipeline for LSTM training.

    1. Validates historical length against lookback and split constraints.
    2. Chronologically splits into train and test series (never shuffled).
    3. Fits MinMaxScaler strictly on training observations to prevent data leakage.
    4. Transforms training and test observations using the fitted scaler.
    5. Forms sliding-window sequences for training and aligned holdout testing.

    Args:
        series: Time-series Series indexed by pd.DatetimeIndex.
        lookback: Window size in days.
        test_ratio: Proportion of series reserved for holdout testing.
        min_train_samples: Minimum training sequences required.

    Returns:
        TimeSeriesDataset dataclass with ready-to-train numpy arrays and date indices.
    """
    total = len(series)
    test_count = max(7, int(total * test_ratio))
    validate_history_length(
        total_samples=total,
        lookback=lookback,
        test_size=test_count,
        min_train_samples=min_train_samples,
    )

    train_series, test_series = chronological_split(
        series, test_ratio=test_ratio, min_test_samples=test_count
    )

    # Fit scaler strictly on training observations
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    train_vals = train_series.values.reshape(-1, 1)
    test_vals = test_series.values.reshape(-1, 1)

    scaler.fit(train_vals)
    scaled_train = scaler.transform(train_vals)
    scaled_test = scaler.transform(test_vals)

    # Create training windows
    X_train, y_train = create_sliding_windows(scaled_train, lookback)
    train_dates = train_series.index[lookback:]

    # For testing, prepend the last `lookback` observations from training
    # to evaluate all test observations without losing the first `lookback` test points
    context_prefix = scaled_train[-lookback:]
    combined_test_input = np.vstack([context_prefix, scaled_test])

    X_test, y_test = create_sliding_windows(combined_test_input, lookback)
    test_dates = test_series.index

    if len(y_test) != len(test_series):
        raise ValueError(
            f"Test sequence length ({len(y_test)}) does not match test series length ({len(test_series)})."
        )

    return TimeSeriesDataset(
        X_train=X_train,
        y_train=y_train,
        train_dates=train_dates,
        X_test=X_test,
        y_test=y_test,
        test_dates=test_dates,
        raw_train_series=train_series,
        raw_test_series=test_series,
        scaler=scaler,
        lookback=lookback,
    )
