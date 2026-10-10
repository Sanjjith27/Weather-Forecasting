"""Custom exceptions for the weather data pipeline."""


class WeatherPipelineError(Exception):
    """Base exception for all weather pipeline errors."""


class GeocodingError(WeatherPipelineError):
    """Raised when geocoding fails or location is not found."""


class WeatherDataAPIError(WeatherPipelineError):
    """Raised when the Open-Meteo weather API call fails (network, status code, timeout)."""


class DataValidationError(WeatherPipelineError):
    """Raised when weather data fails validation checks (schema, length, units, quality)."""


class CacheError(WeatherPipelineError):
    """Raised when caching read/write operations fail."""
