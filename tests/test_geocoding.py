"""Unit tests for the geocoding module."""

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.data.exceptions import GeocodingError
from src.data.geocoding import LocationResult, geocode_city, search_locations


def test_search_locations_success() -> None:
    mock_payload = {
        "results": [
            {
                "id": 2950159,
                "name": "Berlin",
                "latitude": 52.52437,
                "longitude": 13.41053,
                "country": "Germany",
                "country_code": "DE",
                "admin1": "State of Berlin",
                "timezone": "Europe/Berlin",
            },
            {
                "id": 5146743,
                "name": "Berlin",
                "latitude": 39.79123,
                "longitude": -74.92906,
                "country": "United States",
                "country_code": "US",
                "admin1": "New Jersey",
                "timezone": "America/New_York",
            },
        ]
    }

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_payload

    with patch("requests.get", return_value=mock_response) as mock_get:
        results = search_locations("Berlin", count=2)
        assert len(results) == 2

        first = results[0]
        assert first.name == "Berlin"
        assert first.latitude == 52.52437
        assert first.longitude == 13.41053
        assert first.country == "Germany"
        assert first.display_name() == "Berlin, State of Berlin, Germany"

        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["name"] == "Berlin"
        assert kwargs["params"]["count"] == 2


def test_search_locations_empty_results() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"generationtime_ms": 0.1}

    with patch("requests.get", return_value=mock_response):
        results = search_locations("NonExistentCityXYZ")
        assert results == []


def test_search_locations_empty_query() -> None:
    with pytest.raises(GeocodingError, match="non-empty string"):
        search_locations("   ")


def test_geocode_city_top_result() -> None:
    mock_payload = {
        "results": [
            {
                "name": "Tokyo",
                "latitude": 35.6895,
                "longitude": 139.6917,
                "country": "Japan",
                "admin1": "Tokyo",
            }
        ]
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_payload

    with patch("requests.get", return_value=mock_response):
        result = geocode_city("Tokyo")
        assert isinstance(result, LocationResult)
        assert result.name == "Tokyo"
        assert result.latitude == 35.6895
        assert result.longitude == 139.6917


def test_geocode_city_not_found() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"results": []}

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(GeocodingError, match="No location found"):
            geocode_city("InvalidCityName12345")


def test_geocoding_http_error() -> None:
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(GeocodingError, match="HTTP status 500"):
            geocode_city("London")


def test_geocoding_timeout() -> None:
    with patch("requests.get", side_effect=requests.exceptions.Timeout("Connection timed out")):
        with pytest.raises(GeocodingError, match="timed out"):
            geocode_city("Paris", timeout=5.0)


def test_geocoding_malformed_entry() -> None:
    mock_payload = {
        "results": [
            {
                "name": "CorruptCity",
                # missing latitude and longitude
            }
        ]
    }
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = mock_payload

    with patch("requests.get", return_value=mock_response):
        with pytest.raises(GeocodingError, match="Invalid geocoding response entry"):
            search_locations("CorruptCity")
