"""Unit tests for time-series dataset preparation, splitting, and window generation."""

import numpy as np
import pandas as pd
import pytest

from src.models.dataset import (
    chronological_split,
    create_sliding_windows,
    prepare_time_series_dataset,
    validate_history_length,
)
from src.models.exceptions import InsufficientDataError


def _create_sample_series(n_days: int = 100) -> pd.Series:
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    # Realistic seasonal variation with daily noise
    values = 15.0 + 10.0 * np.sin(np.linspace(0, 3 * np.pi, n_days)) + np.random.RandomState(42).normal(0, 0.5, n_days)
    return pd.Series(values, index=dates, name="temperature_celsius")


def test_chronological_split_no_shuffle() -> None:
    series = _create_sample_series(50)
    train, test = chronological_split(series, test_ratio=0.2, min_test_samples=7)

    assert len(train) + len(test) == len(series)
    assert train.index.max() < test.index.min()
    # Confirm exact chronological order preserved
    assert train.index.is_monotonic_increasing
    assert test.index.is_monotonic_increasing


def test_validate_history_insufficient_data() -> None:
    with pytest.raises(InsufficientDataError, match="Insufficient time-series history"):
        validate_history_length(
            total_samples=30,
            lookback=14,
            test_size=10,
            min_train_samples=30,
        )


def test_create_sliding_windows_shapes() -> None:
    data = np.arange(20, dtype=float)
    lookback = 5
    X, y = create_sliding_windows(data, lookback=lookback)

    # 20 - 5 = 15 sequences
    assert X.shape == (15, 5, 1)
    assert y.shape == (15, 1)

    # First sequence: [0, 1, 2, 3, 4] -> target [5]
    np.testing.assert_array_equal(X[0].ravel(), [0, 1, 2, 3, 4])
    assert y[0, 0] == 5.0


def test_prepare_time_series_dataset_alignment() -> None:
    series = _create_sample_series(90)
    lookback = 14
    dataset = prepare_time_series_dataset(series, lookback=lookback, test_ratio=0.2)

    assert dataset.lookback == 14
    assert dataset.total_train_sequences > 0
    assert dataset.total_test_sequences > 0

    # Ensure test sequences match holdout test series count
    assert dataset.total_test_sequences == len(dataset.raw_test_series)
    assert len(dataset.test_dates) == len(dataset.raw_test_series)
    assert dataset.test_dates.equals(dataset.raw_test_series.index)

    # Ensure scaler was fitted on train observations only
    train_min = dataset.raw_train_series.min()
    train_max = dataset.raw_train_series.max()
    assert dataset.scaler.data_min_[0] == pytest.approx(train_min)
    assert dataset.scaler.data_max_[0] == pytest.approx(train_max)
