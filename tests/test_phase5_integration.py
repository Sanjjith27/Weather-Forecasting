"""Phase 5 comprehensive end-to-end integration and verification test suite."""

import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.exceptions import DataValidationError, GeocodingError, WeatherDataAPIError
from src.data.pipeline import fetch_weather_by_city
from src.models.baseline import PersistenceBaseline, calculate_metrics
from src.models.dataset import chronological_split, prepare_time_series_dataset
from src.models.exceptions import InsufficientDataError
from src.models.pipeline import train_and_forecast_pipeline


def _make_sample_dataset_series(n_days: int = 120) -> pd.Series:
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rng = np.random.RandomState(42)
    # Seasonal temperature signal in Celsius
    temps = 18.0 + 9.0 * np.sin(np.linspace(0, 3 * np.pi, n_days)) + rng.normal(0, 0.4, n_days)
    return pd.Series(temps, index=dates, name="temperature_celsius")


# --------------------------------------------------------------------------
# 1. Geocoding & Historical Data Retrieval Integration
# --------------------------------------------------------------------------
def test_geocoding_and_historical_pipeline_mocked(tmp_path: Path) -> None:
    mock_geocode = {
        "results": [
            {
                "name": "Amsterdam",
                "latitude": 52.3676,
                "longitude": 4.9041,
                "country": "Netherlands",
                "admin1": "North Holland",
            }
        ]
    }
    dates = pd.date_range("2023-01-01", periods=45, freq="D").strftime("%Y-%m-%d").tolist()
    temps = [10.0 + (i % 6) * 1.2 for i in range(45)]
    mock_archive = {
        "latitude": 52.37,
        "longitude": 4.90,
        "daily_units": {"time": "iso8601", "temperature_2m_mean": "°C"},
        "daily": {"time": dates, "temperature_2m_mean": temps},
    }

    def mock_get(url: str, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "geocoding-api" in url:
            resp.json.return_value = mock_geocode
        else:
            resp.json.return_value = mock_archive
        return resp

    with patch("requests.get", side_effect=mock_get):
        dataset, location = fetch_weather_by_city(
            city_name="Amsterdam",
            start_date="2023-01-01",
            end_date="2023-02-14",
            cache_dir=tmp_path / "cache",
        )

        assert location.name == "Amsterdam"
        assert location.latitude == pytest.approx(52.3676)
        assert len(dataset.df) == 45
        assert dataset.unit == "°C"
        assert dataset.df.index.is_monotonic_increasing


# --------------------------------------------------------------------------
# 2. Chronological Split & Leakage-Free Scaling
# --------------------------------------------------------------------------
def test_chronological_splitting_no_shuffling_and_leakage_free() -> None:
    series = _make_sample_dataset_series(100)
    lookback = 14

    dataset = prepare_time_series_dataset(series, lookback=lookback, test_ratio=0.2)

    # 1. Chronological boundary check: train strictly precedes test
    assert dataset.train_dates.max() < dataset.test_dates.min()

    # 2. No shuffling: indices are strictly monotonic increasing
    assert dataset.train_dates.is_monotonic_increasing
    assert dataset.test_dates.is_monotonic_increasing

    # 3. Leakage verification: Scaler min and max must match train partition ONLY
    raw_train = dataset.raw_train_series
    raw_test = dataset.raw_test_series

    assert dataset.scaler.data_min_[0] == pytest.approx(raw_train.min())
    assert dataset.scaler.data_max_[0] == pytest.approx(raw_train.max())

    # Confirm test minimum was not used in fitting scaler if test min < train min
    if raw_test.min() < raw_train.min():
        assert dataset.scaler.data_min_[0] != pytest.approx(raw_test.min())


# --------------------------------------------------------------------------
# 3. Genuine Holdout Metrics Calculation
# --------------------------------------------------------------------------
def test_genuine_holdout_metrics_computation() -> None:
    y_true = np.array([15.2, 16.5, 17.1, 14.8, 15.9])
    y_pred = np.array([15.0, 16.0, 17.5, 15.1, 15.5])

    metrics = calculate_metrics(y_true, y_pred)

    expected_mae = float(np.mean(np.abs(y_true - y_pred)))
    expected_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    assert metrics["mae"] == pytest.approx(expected_mae, abs=1e-4)
    assert metrics["rmse"] == pytest.approx(expected_rmse, abs=1e-4)
    assert np.isfinite(metrics["r2"])


# --------------------------------------------------------------------------
# 4. 7-Day Recursive Forecasting Validity
# --------------------------------------------------------------------------
def test_7_day_forecast_finite_and_celsius(tmp_path: Path) -> None:
    series = _make_sample_dataset_series(90)

    result = train_and_forecast_pipeline(
        data=series,
        location_name="Rome",
        lookback=14,
        test_ratio=0.2,
        epochs=3,
        batch_size=8,
        models_dir=tmp_path / "models",
        use_cache=False,
        verbose=0,
    )

    forecast_7d = result.forecast_7_days
    assert len(forecast_7d) == 7
    # Verify all predictions are finite float values
    assert np.all(np.isfinite(forecast_7d.values))
    # Verify within plausible temperature ranges
    assert (forecast_7d.values > -50.0).all() and (forecast_7d.values < 60.0).all()

    # Verify 7 consecutive calendar dates starting the day after the last observation
    last_known_date = series.index[-1]
    expected_dates = pd.date_range(last_known_date + pd.Timedelta(days=1), periods=7, freq="D")
    assert forecast_7d.index.equals(expected_dates)


# --------------------------------------------------------------------------
# 5. Caching & Retraining Elimination
# --------------------------------------------------------------------------
def test_caching_eliminates_unnecessary_retraining(tmp_path: Path) -> None:
    series = _make_sample_dataset_series(80)
    models_dir = tmp_path / "models"

    # Run 1: Cold start, model is trained and saved
    start_time = time.time()
    res1 = train_and_forecast_pipeline(
        data=series,
        location_name="Madrid",
        lookback=12,
        epochs=3,
        batch_size=8,
        models_dir=models_dir,
        use_cache=True,
        verbose=0,
    )
    cold_duration = time.time() - start_time
    assert not res1.is_cached

    # Run 2: Hot cache, loads persisted artifacts without retraining
    start_time = time.time()
    res2 = train_and_forecast_pipeline(
        data=series,
        location_name="Madrid",
        lookback=12,
        epochs=3,
        batch_size=8,
        models_dir=models_dir,
        use_cache=True,
        verbose=0,
    )
    hot_duration = time.time() - start_time
    assert res2.is_cached
    assert hot_duration < cold_duration
    # Bitwise/numerical identity of holdout predictions
    np.testing.assert_allclose(res1.y_test_pred, res2.y_test_pred, atol=1e-5)


# --------------------------------------------------------------------------
# 6. Failure Modes & Exception Handling
# --------------------------------------------------------------------------
def test_failure_modes_and_exceptions() -> None:
    # Insufficient observations
    short_series = pd.Series([20.0] * 20, index=pd.date_range("2023-01-01", periods=20, freq="D"))
    with pytest.raises(InsufficientDataError, match="Insufficient time-series history"):
        train_and_forecast_pipeline(short_series, lookback=14, test_ratio=0.2)

    # Empty city search
    with pytest.raises(GeocodingError, match="non-empty string"):
        fetch_weather_by_city("", "2023-01-01", "2023-02-01")


# --------------------------------------------------------------------------
# 7. Git Exclusion Audit
# --------------------------------------------------------------------------
def test_git_exclusion_rules() -> None:
    test_paths = [
        ".env",
        ".env.local",
        "secrets.toml",
        ".streamlit/secrets.toml",
        "venv/",
        ".venv/",
        "models/saved_model.keras",
        "models/weights.h5",
        "models/scaler.joblib",
        "data/raw_weather.csv",
        "__pycache__/test.pyc",
    ]

    cmd = ["git", "check-ignore", "-v"] + test_paths
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    # Confirm every pattern produced an ignore match in the output
    ignored_lines = result.stdout.strip().splitlines()
    assert len(ignored_lines) == len(test_paths)
