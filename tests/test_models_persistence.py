"""Unit tests for model persistence, serialization, and signature caching."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.models.lstm_model import WeatherLSTMModel
from src.models.persistence import (
    compute_model_signature,
    load_cached_artifacts,
    save_model_artifacts,
)


def test_compute_model_signature_determinism() -> None:
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    series1 = pd.Series([20.0] * 10, index=dates)
    series2 = pd.Series([20.0] * 10, index=dates)

    sig1 = compute_model_signature(series1, "Berlin", 14, 32, 16, 0.005, 42)
    sig2 = compute_model_signature(series2, "Berlin", 14, 32, 16, 0.005, 42)
    assert sig1 == sig2

    # Different location or parameters produce different signature
    sig3 = compute_model_signature(series1, "Tokyo", 14, 32, 16, 0.005, 42)
    assert sig1 != sig3


def test_save_and_load_cached_artifacts(tmp_path: Path) -> None:
    lookback = 10
    model_wrapper = WeatherLSTMModel(lookback=lookback, lstm_units=16, dense_units=8)
    model_wrapper.build()

    scaler = MinMaxScaler()
    scaler.fit(np.array([[10.0], [30.0]]))

    metadata = {"location": "London", "mae": 1.25}
    signature = "test_sig_123"

    saved_dir = save_model_artifacts(
        lstm_wrapper=model_wrapper,
        scaler=scaler,
        metadata=metadata,
        signature=signature,
        base_dir=tmp_path,
    )

    assert saved_dir.exists()
    assert (saved_dir / "model.keras").exists()
    assert (saved_dir / "scaler.joblib").exists()
    assert (saved_dir / "metadata.json").exists()

    loaded = load_cached_artifacts(signature, base_dir=tmp_path)
    assert loaded is not None

    loaded_model, loaded_scaler, loaded_meta = loaded
    assert loaded_meta["location"] == "London"
    assert loaded_meta["mae"] == 1.25
    assert loaded_scaler.data_min_[0] == scaler.data_min_[0]
    assert loaded_scaler.data_max_[0] == scaler.data_max_[0]

    # Verify predictions from loaded model are identical
    dummy_input = np.ones((1, lookback, 1), dtype=np.float32)
    orig_pred = model_wrapper.model.predict(dummy_input, verbose=0)
    loaded_pred = loaded_model.model.predict(dummy_input, verbose=0)
    np.testing.assert_allclose(orig_pred, loaded_pred, atol=1e-5)


def test_load_cached_artifacts_miss(tmp_path: Path) -> None:
    loaded = load_cached_artifacts("non_existent_signature", base_dir=tmp_path)
    assert loaded is None
