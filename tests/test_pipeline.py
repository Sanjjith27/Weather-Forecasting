"""End-to-end unit tests for the weather data pipeline with mocked APIs."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.pipeline import (
    WeatherDataset,
    fetch_weather_by_city,
    fetch_weather_by_coordinates,
)


def _generate_mock_archive_payload(count: int = 35) -> dict:
    dates = [f"2023-01-{i:02d}" if i <= 31 else f"2023-02-{(i - 31):02d}" for i in range(1, count + 1)]
    temps = [20.0 + (i % 6) * 1.5 for i in range(1, count + 1)]
    return {
        "latitude": 13.08,
        "longitude": 80.27,
        "generationtime_ms": 0.4,
        "utc_offset_seconds": 0,
        "timezone": "UTC",
        "timezone_abbreviation": "UTC",
        "elevation": 10.0,
        "daily_units": {
            "time": "iso8601",
            "temperature_2m_mean": "°C",
        },
        "daily": {
            "time": dates,
            "temperature_2m_mean": temps,
        },
    }


def test_fetch_weather_by_city_end_to_end(tmp_path: Path) -> None:
    mock_geocode_payload = {
        "results": [
            {
                "name": "Chennai",
                "latitude": 13.0827,
                "longitude": 80.2707,
                "country": "India",
                "country_code": "IN",
                "admin1": "Tamil Nadu",
                "timezone": "Asia/Kolkata",
            }
        ]
    }
    mock_archive_payload = _generate_mock_archive_payload(40)

    def mock_get(url: str, **kwargs):
        resp = MagicMock()
        resp.status_code = 200
        if "geocoding-api" in url:
            resp.json.return_value = mock_geocode_payload
        else:
            resp.json.return_value = mock_archive_payload
        return resp

    with patch("requests.get", side_effect=mock_get):
        dataset, location = fetch_weather_by_city(
            city_name="Chennai",
            start_date="2023-01-01",
            end_date="2023-02-09",
            cache_dir=tmp_path / "cache",
            min_samples=30,
        )

        assert isinstance(dataset, WeatherDataset)
        assert dataset.location_name == "Chennai, Tamil Nadu, India"
        assert dataset.sample_count == 40
        assert location.name == "Chennai"
        assert location.country == "India"
        assert "temperature_celsius" in dataset.df.columns
        assert dataset.df["temperature_celsius"].isna().sum() == 0

        summary = dataset.summary()
        assert summary["location"] == "Chennai, Tamil Nadu, India"
        assert summary["samples"] == 40
        assert summary["interpolated_samples"] == 0
        assert summary["min_temp_c"] > 0


def test_fetch_weather_by_coordinates_end_to_end(tmp_path: Path) -> None:
    mock_archive_payload = _generate_mock_archive_payload(35)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_archive_payload

    with patch("requests.Session.get", return_value=mock_resp):
        dataset = fetch_weather_by_coordinates(
            latitude=13.08,
            longitude=80.27,
            start_date="2023-01-01",
            end_date="2023-02-04",
            cache_dir=tmp_path / "cache",
            min_samples=30,
        )

        assert dataset.sample_count == 35
        assert dataset.latitude == 13.08
        assert dataset.longitude == 80.27
        assert dataset.unit == "°C"
