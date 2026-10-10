"""High-level forecasting engine pipeline coordinating preparation, training, and forecasting."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.pipeline import WeatherDataset
from src.models.baseline import PersistenceBaseline
from src.models.dataset import TimeSeriesDataset, prepare_time_series_dataset
from src.models.lstm_model import WeatherLSTMModel
from src.models.persistence import (
    compute_model_signature,
    load_cached_artifacts,
    save_model_artifacts,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelTrainingResult:
    """Encapsulates all evaluation metrics, predictions, and 7-day forecasts."""

    location_name: str
    signature: str
    lookback: int
    train_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    y_test_true: np.ndarray
    y_test_pred: np.ndarray
    baseline_test_pred: np.ndarray
    metrics_lstm: dict[str, float]
    metrics_baseline: dict[str, float]
    forecast_7_days: pd.Series
    is_cached: bool
    training_duration_seconds: float
    model_artifact_path: Optional[str] = None

    def comparison_summary(self) -> dict[str, float | str | bool]:
        """Summarize performance comparison between LSTM and Persistence baseline."""
        lstm_mae = self.metrics_lstm["mae"]
        base_mae = self.metrics_baseline["mae"]
        lstm_rmse = self.metrics_lstm["rmse"]
        base_rmse = self.metrics_baseline["rmse"]

        mae_improvement = round(base_mae - lstm_mae, 4)
        rmse_improvement = round(base_rmse - lstm_rmse, 4)

        return {
            "location": self.location_name,
            "lookback_days": self.lookback,
            "lstm_mae": lstm_mae,
            "baseline_mae": base_mae,
            "mae_improvement": mae_improvement,
            "lstm_rmse": lstm_rmse,
            "baseline_rmse": base_rmse,
            "rmse_improvement": rmse_improvement,
            "lstm_beats_baseline_mae": lstm_mae < base_mae,
            "is_cached": self.is_cached,
        }


def train_and_forecast_pipeline(
    data: WeatherDataset | pd.DataFrame | pd.Series,
    location_name: str = "Target Location",
    lookback: int = 14,
    test_ratio: float = 0.2,
    lstm_units: int = 32,
    dense_units: int = 16,
    dropout_rate: float = 0.1,
    learning_rate: float = 0.005,
    epochs: int = 35,
    batch_size: int = 16,
    patience: int = 5,
    seed: int = 42,
    use_cache: bool = True,
    models_dir: Path | str = Path("models"),
    verbose: int = 0,
) -> ModelTrainingResult:
    """Execute end-to-end training, holdout evaluation, and 7-day recursive forecasting.

    Args:
        data: WeatherDataset, DataFrame with 'temperature_celsius', or Series.
        location_name: Descriptive name of the location.
        lookback: Input window size in days.
        test_ratio: Proportion of series reserved for holdout testing.
        lstm_units: Units in LSTM layer.
        dense_units: Units in intermediate dense layer.
        dropout_rate: Dropout rate.
        learning_rate: Adam optimizer learning rate.
        epochs: Maximum training epochs.
        batch_size: Batch size.
        patience: Early stopping patience.
        seed: Random seed.
        use_cache: Whether to load/save matching cached model artifacts.
        models_dir: Directory where models are stored.
        verbose: Keras training verbosity.

    Returns:
        ModelTrainingResult with genuine holdout evaluation and 7-day forecast.
    """
    start_time = time.time()

    # Extract clean temperature series
    if isinstance(data, WeatherDataset):
        series = data.df["temperature_celsius"]
        loc_name = data.location_name or location_name
    elif isinstance(data, pd.DataFrame):
        if "temperature_celsius" not in data.columns:
            raise ValueError("DataFrame must contain 'temperature_celsius' column.")
        series = data["temperature_celsius"]
        loc_name = location_name
    elif isinstance(data, pd.Series):
        series = data
        loc_name = location_name
    else:
        raise TypeError(f"Unsupported data type: {type(data)}. Expected WeatherDataset, DataFrame, or Series.")

    series = series.dropna().sort_index()

    # Compute cache signature
    signature = compute_model_signature(
        series=series,
        location_name=loc_name,
        lookback=lookback,
        lstm_units=lstm_units,
        dense_units=dense_units,
        learning_rate=learning_rate,
        seed=seed,
    )

    # 1. Prepare data splits and sliding windows
    dataset: TimeSeriesDataset = prepare_time_series_dataset(
        series=series,
        lookback=lookback,
        test_ratio=test_ratio,
        min_train_samples=30,
    )

    # 2. Compute Persistence baseline on holdout test set
    base_preds, base_metrics = PersistenceBaseline.predict_holdout(
        train_series=dataset.raw_train_series,
        test_series=dataset.raw_test_series,
    )

    # 3. Check for cached model
    cached_obj = load_cached_artifacts(signature, base_dir=models_dir) if use_cache else None
    is_cached = False
    artifact_path: Optional[str] = None

    if cached_obj is not None:
        lstm_model, scaler, meta = cached_obj
        is_cached = True
        logger.info("Using cached LSTM model for %s (%s)", loc_name, signature)
    else:
        # Build and train genuine LSTM model
        lstm_model = WeatherLSTMModel(
            lookback=lookback,
            lstm_units=lstm_units,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            seed=seed,
        )
        lstm_model.train(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            validation_split=0.15,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            verbose=verbose,
        )
        scaler = dataset.scaler

    # 4. Evaluate on holdout test set
    y_test_true, y_test_pred, lstm_metrics = lstm_model.evaluate_holdout(
        X_test=dataset.X_test,
        y_test=dataset.y_test,
        scaler=scaler,
    )

    # 5. Produce 7-day recursive forecast
    last_date = series.index[-1]
    forecast_7d = lstm_model.forecast_7_days(
        recent_observations=series.values,
        scaler=scaler,
        last_date=last_date,
    )

    # 6. Save model artifacts if newly trained
    if not is_cached and use_cache:
        meta_to_save = {
            "location": loc_name,
            "lookback": lookback,
            "lstm_units": lstm_units,
            "dense_units": dense_units,
            "learning_rate": learning_rate,
            "seed": seed,
            "metrics_lstm": lstm_metrics,
            "metrics_baseline": base_metrics,
            "training_samples": len(dataset.X_train),
            "test_samples": len(dataset.X_test),
        }
        saved_dir = save_model_artifacts(
            lstm_wrapper=lstm_model,
            scaler=scaler,
            metadata=meta_to_save,
            signature=signature,
            base_dir=models_dir,
        )
        artifact_path = str(saved_dir)

    duration = round(time.time() - start_time, 2)

    return ModelTrainingResult(
        location_name=loc_name,
        signature=signature,
        lookback=lookback,
        train_dates=dataset.train_dates,
        test_dates=dataset.test_dates,
        y_test_true=y_test_true,
        y_test_pred=y_test_pred,
        baseline_test_pred=base_preds,
        metrics_lstm=lstm_metrics,
        metrics_baseline=base_metrics,
        forecast_7_days=forecast_7d,
        is_cached=is_cached,
        training_duration_seconds=duration,
        model_artifact_path=artifact_path,
    )
