"""Weather data ingestion, validation, and preprocessing package."""

from src.data.client import HistoricalWeatherClient, fetch_historical_raw
from src.data.exceptions import (
    CacheError,
    DataValidationError,
    GeocodingError,
    WeatherDataAPIError,
    WeatherPipelineError,
)
from src.data.geocoding import LocationResult, geocode_city, search_locations
from src.data.pipeline import (
    WeatherDataset,
    fetch_weather_by_city,
    fetch_weather_by_coordinates,
)
from src.data.preprocessing import (
    create_weather_dataframe,
    handle_missing_observations,
    preprocess_weather_data,
)
from src.data.validation import (
    validate_api_response_schema,
    validate_date_sequence,
    validate_weather_dataframe,
)

__all__ = [
    "CacheError",
    "DataValidationError",
    "GeocodingError",
    "HistoricalWeatherClient",
    "LocationResult",
    "WeatherDataAPIError",
    "WeatherDataset",
    "WeatherPipelineError",
    "create_weather_dataframe",
    "fetch_historical_raw",
    "fetch_weather_by_city",
    "fetch_weather_by_coordinates",
    "geocode_city",
    "handle_missing_observations",
    "preprocess_weather_data",
    "search_locations",
    "validate_api_response_schema",
    "validate_date_sequence",
    "validate_weather_dataframe",
]
