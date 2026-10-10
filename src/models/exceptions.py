"""Custom exceptions for the LSTM modeling and forecasting engine."""

from src.data.exceptions import WeatherPipelineError


class ModelEngineError(WeatherPipelineError):
    """Base exception for modeling and forecasting errors."""


class InsufficientDataError(ModelEngineError):
    """Raised when available time-series history is insufficient for lookback and training."""


class ModelTrainingError(ModelEngineError):
    """Raised when model compilation or training fails."""


class ForecastingError(ModelEngineError):
    """Raised when multi-step recursive forecasting fails or generates non-finite outputs."""


class ModelPersistenceError(ModelEngineError):
    """Raised when saving or loading model artifacts and state fails."""
