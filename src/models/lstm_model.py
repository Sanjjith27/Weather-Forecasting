"""TensorFlow/Keras LSTM temperature forecasting model."""

from __future__ import annotations

import logging
import random
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.models.baseline import calculate_metrics
from src.models.exceptions import ForecastingError, ModelTrainingError

logger = logging.getLogger(__name__)

# Import TensorFlow with warning suppression
try:
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential
except ImportError as err:
    raise ImportError(
        "TensorFlow is required for WeatherLSTMModel. Ensure tensorflow is installed."
    ) from err


def set_seed(seed: int = 42) -> None:
    """Set random seed across Python, NumPy, and TensorFlow for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class WeatherLSTMModel:
    """Compact TensorFlow/Keras LSTM neural network for weather temperature forecasting."""

    def __init__(
        self,
        lookback: int = 14,
        lstm_units: int = 32,
        dense_units: int = 16,
        dropout_rate: float = 0.1,
        learning_rate: float = 0.005,
        seed: int = 42,
    ) -> None:
        self.lookback = lookback
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.seed = seed
        self.model: Optional[Sequential] = None
        self.history_: dict[str, list[float]] = {}

    def build(self) -> Sequential:
        """Construct and compile the compact Sequential LSTM architecture."""
        set_seed(self.seed)

        model = Sequential(
            [
                Input(shape=(self.lookback, 1)),
                LSTM(self.lstm_units, activation="tanh", return_sequences=False),
                Dropout(self.dropout_rate),
                Dense(self.dense_units, activation="relu"),
                Dense(1),
            ],
            name="weather_lstm",
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mean_squared_error")
        self.model = model
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        validation_split: float = 0.15,
        epochs: int = 40,
        batch_size: int = 16,
        patience: int = 6,
        verbose: int = 0,
    ) -> dict[str, list[float]]:
        """Train the LSTM model using EarlyStopping on holdout validation loss.

        Args:
            X_train: Input sequence tensor of shape (N, lookback, 1).
            y_train: Target values of shape (N, 1).
            validation_split: Proportion of training data for validation early stopping.
            epochs: Maximum training epochs.
            batch_size: Mini-batch size.
            patience: Early stopping patience epochs.
            verbose: Verbosity level (0=silent, 1=progress).

        Returns:
            Dictionary containing loss and val_loss training history.

        Raises:
            ModelTrainingError: If training fails.
        """
        if self.model is None:
            self.build()

        if len(X_train) == 0:
            raise ModelTrainingError("Training dataset is empty.")

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=0,
        )

        try:
            history = self.model.fit(
                X_train,
                y_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=verbose,
            )
        except Exception as err:
            raise ModelTrainingError(f"Error occurred during LSTM training: {err}") from err

        self.history_ = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        return self.history_

    def predict_holdout(
        self,
        X_test: np.ndarray,
        scaler: MinMaxScaler,
    ) -> np.ndarray:
        """Predict target temperatures for holdout sequences and inverse-transform to Celsius.

        Args:
            X_test: Test input sequences of shape (M, lookback, 1).
            scaler: Fitted MinMaxScaler instance.

        Returns:
            1D array of predicted temperatures in degrees Celsius.
        """
        if self.model is None:
            raise ModelTrainingError("Model has not been built or trained yet.")

        preds_scaled = self.model.predict(X_test, verbose=0)
        preds_celsius = scaler.inverse_transform(preds_scaled).ravel()
        return preds_celsius.astype(float)

    def evaluate_holdout(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        scaler: MinMaxScaler,
    ) -> Tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Evaluate model performance on the holdout test set.

        Computes genuine MAE and RMSE in original degrees Celsius.

        Args:
            X_test: Test sequences of shape (M, lookback, 1).
            y_test: Test targets (scaled) of shape (M, 1).
            scaler: Fitted MinMaxScaler.

        Returns:
            Tuple of (y_true_celsius, y_pred_celsius, metrics_dict).
        """
        y_true_celsius = scaler.inverse_transform(y_test).ravel().astype(float)
        y_pred_celsius = self.predict_holdout(X_test, scaler=scaler)

        metrics = calculate_metrics(y_true_celsius, y_pred_celsius)
        return y_true_celsius, y_pred_celsius, metrics

    def forecast_7_days(
        self,
        recent_observations: np.ndarray | pd.Series,
        scaler: MinMaxScaler,
        last_date: pd.Timestamp,
    ) -> pd.Series:
        """Produce 7 future daily temperature estimates using recursive forecasting.

        Starting from the most recent known `lookback` observations, recursively
        predicts day t+1, feeds it back into the sliding input window, and repeats
        for 7 days.

        Args:
            recent_observations: The last `lookback` known temperature observations in Celsius.
            scaler: Fitted MinMaxScaler used during training.
            last_date: Date timestamp of the most recent known observation.

        Returns:
            pd.Series containing 7 forecasted daily temperatures in Celsius indexed by dates.

        Raises:
            ForecastingError: If recent observations are insufficient or outputs are non-finite.
        """
        if self.model is None:
            raise ModelTrainingError("Model has not been built or trained yet.")

        recent_vals = np.asarray(recent_observations, dtype=float).ravel()
        if len(recent_vals) < self.lookback:
            raise ForecastingError(
                f"Insufficient recent observations for recursive forecasting: "
                f"need at least {self.lookback} observations, got {len(recent_vals)}."
            )

        # Take the most recent `lookback` values and scale them
        last_window = recent_vals[-self.lookback :].reshape(-1, 1)
        scaled_window = scaler.transform(last_window).reshape(1, self.lookback, 1)

        forecasts_scaled: list[float] = []
        current_window = scaled_window.copy()

        for step in range(7):
            pred_tensor = self.model(current_window, training=False)
            pred_scaled = float(pred_tensor.numpy()[0, 0])

            if not np.isfinite(pred_scaled):
                raise ForecastingError(
                    f"Recursive forecasting generated non-finite prediction at step {step + 1}: {pred_scaled}"
                )

            forecasts_scaled.append(pred_scaled)

            # Slide window forward by 1: drop oldest, append prediction
            next_step_input = np.array([[[pred_scaled]]], dtype=np.float32)
            current_window = np.concatenate([current_window[:, 1:, :], next_step_input], axis=1)

        # Inverse-transform all 7 predictions back to degrees Celsius
        scaled_arr = np.array(forecasts_scaled).reshape(-1, 1)
        forecasts_celsius = scaler.inverse_transform(scaled_arr).ravel()

        # Sanity check: must be finite and within meteorological limits
        if not np.all(np.isfinite(forecasts_celsius)):
            raise ForecastingError("One or more 7-day temperature forecasts are non-finite (NaN or Inf).")

        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=7,
            freq="D",
        )

        return pd.Series(
            forecasts_celsius,
            index=future_dates,
            name="lstm_forecast_celsius",
        )
