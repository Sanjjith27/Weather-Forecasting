"""Model serialization, artifact persistence, and dataset-matched caching."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

from src.models.exceptions import ModelPersistenceError
from src.models.lstm_model import WeatherLSTMModel

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path("models")


def compute_model_signature(
    series: pd.Series,
    location_name: str,
    lookback: int,
    lstm_units: int,
    dense_units: int,
    learning_rate: float,
    seed: int,
) -> str:
    """Compute a deterministic hash fingerprint representing dataset and model config.

    Ensures cached models strictly match both the underlying time series data
    and hyperparameter configuration.
    """
    total = len(series)
    start_str = series.index[0].strftime("%Y-%m-%d") if total > 0 else ""
    end_str = series.index[-1].strftime("%Y-%m-%d") if total > 0 else ""
    mean_val = round(float(series.mean()), 4) if total > 0 else 0.0

    meta = {
        "location": location_name.strip().lower(),
        "total_samples": total,
        "start_date": start_str,
        "end_date": end_str,
        "mean_val": mean_val,
        "lookback": lookback,
        "lstm_units": lstm_units,
        "dense_units": dense_units,
        "lr": learning_rate,
        "seed": seed,
    }

    serialized = json.dumps(meta, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def save_model_artifacts(
    lstm_wrapper: WeatherLSTMModel,
    scaler: MinMaxScaler,
    metadata: dict[str, Any],
    signature: str,
    base_dir: Path | str = DEFAULT_MODELS_DIR,
) -> Path:
    """Save trained Keras model, fitted scaler, and metadata to disk.

    Args:
        lstm_wrapper: Trained WeatherLSTMModel instance.
        scaler: Fitted MinMaxScaler instance.
        metadata: Summary metrics and training parameters.
        signature: Unique dataset/config fingerprint.
        base_dir: Base directory for models storage (defaults to 'models/').

    Returns:
        Path to the saved artifact directory.

    Raises:
        ModelPersistenceError: If saving fails.
    """
    if lstm_wrapper.model is None:
        raise ModelPersistenceError("Cannot save model: Keras Sequential model is not built or trained.")

    target_dir = Path(base_dir) / signature
    try:
        target_dir.mkdir(parents=True, exist_ok=True)

        model_path = target_dir / "model.keras"
        lstm_wrapper.model.save(model_path)

        scaler_path = target_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)

        meta_path = target_dir / "metadata.json"
        metadata["saved_at"] = datetime.now().isoformat()
        metadata["signature"] = signature
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info("Saved model artifacts to %s", target_dir)
        return target_dir
    except Exception as err:
        raise ModelPersistenceError(f"Failed to save model artifacts to {target_dir}: {err}") from err


def load_cached_artifacts(
    signature: str,
    base_dir: Path | str = DEFAULT_MODELS_DIR,
) -> Optional[Tuple[WeatherLSTMModel, MinMaxScaler, dict[str, Any]]]:
    """Load previously trained model and scaler if matching signature exists.

    Args:
        signature: Dataset/config signature hash.
        base_dir: Base directory where models are persisted.

    Returns:
        Tuple of (WeatherLSTMModel, MinMaxScaler, metadata) or None if cache miss.
    """
    target_dir = Path(base_dir) / signature
    model_path = target_dir / "model.keras"
    scaler_path = target_dir / "scaler.joblib"
    meta_path = target_dir / "metadata.json"

    if not (model_path.exists() and scaler_path.exists() and meta_path.exists()):
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        keras_model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        lstm_wrapper = WeatherLSTMModel(
            lookback=metadata.get("lookback", 14),
            lstm_units=metadata.get("lstm_units", 32),
            dense_units=metadata.get("dense_units", 16),
            dropout_rate=metadata.get("dropout_rate", 0.1),
            learning_rate=metadata.get("learning_rate", 0.005),
            seed=metadata.get("seed", 42),
        )
        lstm_wrapper.model = keras_model

        logger.info("Loaded cached model artifacts from %s", target_dir)
        return lstm_wrapper, scaler, metadata
    except Exception as err:
        logger.warning("Error loading cached model artifacts from %s: %s", target_dir, err)
        return None
