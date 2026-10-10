"""Unit tests for the TensorFlow/Keras LSTM model architecture and forecasting."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.models.exceptions import ForecastingError, ModelTrainingError
from src.models.lstm_model import WeatherLSTMModel, set_seed


def _generate_synthetic_sequences(n_samples: int = 40, lookback: int = 10) -> tuple:
    np.random.seed(42)
    # Scaled sine wave in [0, 1]
    t = np.linspace(0, 4 * np.pi, n_samples + lookback)
    signal = (np.sin(t) + 1.0) / 2.0  # Range [0, 1]

    X = []
    y = []
    for i in range(n_samples):
        X.append(signal[i : i + lookback].reshape(lookback, 1))
        y.append([signal[i + lookback]])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def test_lstm_build_architecture() -> None:
    set_seed(42)
    model_wrapper = WeatherLSTMModel(lookback=14, lstm_units=32, dense_units=16)
    keras_model = model_wrapper.build()

    assert keras_model.input_shape == (None, 14, 1)
    assert keras_model.output_shape == (None, 1)
    assert len(keras_model.layers) == 4  # LSTM, Dropout, Dense(16), Dense(1)


def test_lstm_train_and_predict_holdout() -> None:
    set_seed(42)
    lookback = 10
    X, y = _generate_synthetic_sequences(n_samples=40, lookback=lookback)

    # Train/test split
    X_train, y_train = X[:30], y[:30]
    X_test, y_test = X[30:], y[30:]

    scaler = MinMaxScaler()
    # Dummy scaler fitted on range [10.0, 30.0] Celsius
    scaler.fit(np.array([[10.0], [30.0]]))

    model = WeatherLSTMModel(lookback=lookback, lstm_units=16, dense_units=8)
    history = model.train(X_train, y_train, epochs=4, batch_size=8, verbose=0)

    assert "loss" in history
    assert len(history["loss"]) > 0

    # Test holdout evaluation
    y_true_c, y_pred_c, metrics = model.evaluate_holdout(X_test, y_test, scaler=scaler)
    assert len(y_pred_c) == len(X_test)
    assert len(y_true_c) == len(y_test)
    assert np.all(np.isfinite(y_pred_c))
    assert metrics["mae"] >= 0.0
    assert metrics["rmse"] >= 0.0


def test_lstm_forecast_7_days_finite_celsius() -> None:
    set_seed(42)
    lookback = 10
    X, y = _generate_synthetic_sequences(n_samples=30, lookback=lookback)

    scaler = MinMaxScaler()
    scaler.fit(np.array([[10.0], [30.0]]))

    model = WeatherLSTMModel(lookback=lookback, lstm_units=16, dense_units=8)
    model.train(X, y, epochs=3, batch_size=8, verbose=0)

    recent_temps = np.linspace(15.0, 25.0, lookback)
    last_date = pd.Timestamp("2023-06-30")

    forecast = model.forecast_7_days(recent_temps, scaler=scaler, last_date=last_date)

    assert len(forecast) == 7
    assert isinstance(forecast, pd.Series)
    assert np.all(np.isfinite(forecast.values))
    # All predicted values should be within plausible temperature bounds
    assert (forecast.values > -50.0).all() and (forecast.values < 60.0).all()
    # Check date progression
    assert forecast.index[0] == pd.Timestamp("2023-07-01")
    assert forecast.index[-1] == pd.Timestamp("2023-07-07")


def test_lstm_forecast_insufficient_history() -> None:
    model = WeatherLSTMModel(lookback=14)
    model.build()

    scaler = MinMaxScaler()
    scaler.fit(np.array([[0.0], [100.0]]))

    short_history = [20.0, 21.0, 22.0]  # only 3 samples, need 14
    with pytest.raises(ForecastingError, match="Insufficient recent observations"):
        model.forecast_7_days(short_history, scaler=scaler, last_date=pd.Timestamp.now())
