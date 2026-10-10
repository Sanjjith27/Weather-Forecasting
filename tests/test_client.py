"""Unit tests for the historical weather API client and disk caching."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

from src.data.client import HistoricalWeatherClient
from src.data.exceptions import WeatherDataAPIError


def _sample_valid_response() -> dict:
    return {
        "latitude": 52.52,
        "longitude": 13.41,
        "generationtime_ms": 1.2,
        "utc_offset_seconds": 0,
        "timezone": "UTC",
        "timezone_abbreviation": "UTC",
        "elevation": 38.0,
        "daily_units": {
            "time": "iso8601",
            "temperature_2m_mean": "°C",
        },
        "daily": {
            "time": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "temperature_2m_mean": [5.1, 4.3, 6.2],
        },
    }


def test_fetch_historical_weather_success(tmp_path: Path) -> None:
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = _sample_valid_response()
    mock_session.get.return_value = mock_response

    client = HistoricalWeatherClient(
        cache_dir=tmp_path / "cache",
        enable_cache=True,
        session=mock_session,
    )

    data = client.fetch_historical_weather_raw(
        latitude=52.52,
        longitude=13.41,
        start_date="2023-01-01",
        end_date="2023-01-03",
    )

    assert data["daily"]["temperature_2m_mean"] == [5.1, 4.3, 6.2]
    mock_session.get.assert_called_once()


def test_fetch_historical_weather_caching(tmp_path: Path) -> None:
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = _sample_valid_response()
    mock_session.get.return_value = mock_response

    cache_dir = tmp_path / "cache"
    client = HistoricalWeatherClient(
        cache_dir=cache_dir,
        enable_cache=True,
        session=mock_session,
    )

    # First call: hits the network
    data1 = client.fetch_historical_weather_raw(
        latitude=52.52,
        longitude=13.41,
        start_date="2023-01-01",
        end_date="2023-01-03",
    )
    assert mock_session.get.call_count == 1

    # Second call: loads from cache without network call
    data2 = client.fetch_historical_weather_raw(
        latitude=52.52,
        longitude=13.41,
        start_date="2023-01-01",
        end_date="2023-01-03",
    )
    assert mock_session.get.call_count == 1
    assert data1 == data2


def test_fetch_historical_weather_cache_disabled(tmp_path: Path) -> None:
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = _sample_valid_response()
    mock_session.get.return_value = mock_response

    client = HistoricalWeatherClient(
        cache_dir=tmp_path / "cache",
        enable_cache=False,
        session=mock_session,
    )

    client.fetch_historical_weather_raw(52.52, 13.41, "2023-01-01", "2023-01-03")
    client.fetch_historical_weather_raw(52.52, 13.41, "2023-01-01", "2023-01-03")
    assert mock_session.get.call_count == 2


def test_fetch_historical_weather_http_error(tmp_path: Path) -> None:
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = '{"error": true, "reason": "Latitude must be in range of -90 to 90"}'
    mock_session.get.return_value = mock_response

    client = HistoricalWeatherClient(cache_dir=tmp_path / "cache", session=mock_session)

    with pytest.raises(WeatherDataAPIError, match="HTTP status 400"):
        client.fetch_historical_weather_raw(52.52, 13.41, "2023-01-01", "2023-01-03")


def test_fetch_historical_weather_timeout(tmp_path: Path) -> None:
    mock_session = MagicMock()
    mock_session.get.side_effect = requests.exceptions.Timeout("Connection timed out")

    client = HistoricalWeatherClient(cache_dir=tmp_path / "cache", session=mock_session)

    with pytest.raises(WeatherDataAPIError, match="timed out"):
        client.fetch_historical_weather_raw(52.52, 13.41, "2023-01-01", "2023-01-03")


def test_fetch_historical_weather_invalid_coords(tmp_path: Path) -> None:
    client = HistoricalWeatherClient(cache_dir=tmp_path / "cache")

    with pytest.raises(WeatherDataAPIError, match="Latitude must be between -90 and 90"):
        client.fetch_historical_weather_raw(120.0, 13.41, "2023-01-01", "2023-01-03")

    with pytest.raises(WeatherDataAPIError, match="Longitude must be between -180 and 180"):
        client.fetch_historical_weather_raw(52.0, 200.0, "2023-01-01", "2023-01-03")


def test_fetch_historical_weather_invalid_date_order(tmp_path: Path) -> None:
    client = HistoricalWeatherClient(cache_dir=tmp_path / "cache")

    with pytest.raises(WeatherDataAPIError, match="less than or equal to end_date"):
        client.fetch_historical_weather_raw(52.0, 13.0, "2023-05-10", "2023-05-01")


def test_fetch_historical_weather_corrupt_cache(tmp_path: Path) -> None:
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = _sample_valid_response()
    mock_session.get.return_value = mock_response

    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    # Write a broken file in cache
    (cache_dir / "broken.json").write_text("NOT_JSON")

    client = HistoricalWeatherClient(cache_dir=cache_dir, session=mock_session)
    data = client.fetch_historical_weather_raw(52.52, 13.41, "2023-01-01", "2023-01-03")
    assert data["daily"]["temperature_2m_mean"] == [5.1, 4.3, 6.2]
