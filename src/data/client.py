"""Historical weather client for the Open-Meteo Historical Weather API."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import requests

from src.data.exceptions import CacheError, WeatherDataAPIError

logger = logging.getLogger(__name__)

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_CACHE_DIR = Path(".cache") / "weather_api"


def _format_date(val: str | date | datetime) -> str:
    """Validate and format a date value into YYYY-MM-DD."""
    if isinstance(val, (date, datetime)):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, str):
        try:
            parsed = datetime.strptime(val.strip(), "%Y-%m-%d")
            return parsed.strftime("%Y-%m-%d")
        except ValueError as err:
            raise WeatherDataAPIError(
                f"Invalid date format '{val}'. Expected YYYY-MM-DD."
            ) from err
    raise WeatherDataAPIError(f"Unsupported date type: {type(val)}. Expected str or date.")


def _compute_cache_key(params: dict[str, Any]) -> str:
    """Generate a deterministic hash key for API request parameters."""
    normalized = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class HistoricalWeatherClient:
    """Client for retrieving historical meteorological observations from Open-Meteo."""

    def __init__(
        self,
        base_url: str = OPEN_METEO_ARCHIVE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        cache_dir: Optional[Path | str] = DEFAULT_CACHE_DIR,
        enable_cache: bool = True,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.session = session or requests.Session()

        if self.enable_cache and self.cache_dir:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except OSError as err:
                logger.warning("Failed to create cache directory %s: %s", self.cache_dir, err)

    def _read_cache(self, cache_key: str) -> Optional[dict[str, Any]]:
        """Attempt to read a cached JSON response."""
        if not self.enable_cache or not self.cache_dir:
            return None
        cache_file = self.cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as err:
            logger.warning("Error reading cache file %s: %s", cache_file, err)
            return None

    def _write_cache(self, cache_key: str, data: dict[str, Any]) -> None:
        """Write JSON response to disk cache."""
        if not self.enable_cache or not self.cache_dir:
            return
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as err:
            logger.warning("Failed to write cache file %s: %s", cache_file, err)

    def fetch_historical_weather_raw(
        self,
        latitude: float,
        longitude: float,
        start_date: str | date | datetime,
        end_date: str | date | datetime,
        daily_variable: str = "temperature_2m_mean",
        temperature_unit: str = "celsius",
        timezone: str = "UTC",
        use_cache: Optional[bool] = None,
    ) -> dict[str, Any]:
        """Fetch raw historical daily weather data from the Open-Meteo Archive API.

        Args:
            latitude: Latitude (-90.0 to 90.0).
            longitude: Longitude (-180.0 to 180.0).
            start_date: Starting date (inclusive, YYYY-MM-DD).
            end_date: Ending date (inclusive, YYYY-MM-DD).
            daily_variable: Daily meteorological parameter to fetch.
            temperature_unit: Unit for temperature ('celsius').
            timezone: Timezone identifier (default 'UTC').
            use_cache: Override instance cache setting if provided.

        Returns:
            Dictionary response from the Open-Meteo archive API.

        Raises:
            WeatherDataAPIError: If coordinates, dates, or network request fail.
        """
        # Validate coordinates
        if not (-90.0 <= latitude <= 90.0):
            raise WeatherDataAPIError(f"Latitude must be between -90 and 90. Got: {latitude}")
        if not (-180.0 <= longitude <= 180.0):
            raise WeatherDataAPIError(f"Longitude must be between -180 and 180. Got: {longitude}")

        start_str = _format_date(start_date)
        end_str = _format_date(end_date)
        if start_str > end_str:
            raise WeatherDataAPIError(
                f"start_date ({start_str}) must be less than or equal to end_date ({end_str})."
            )

        params: dict[str, Any] = {
            "latitude": round(latitude, 4),
            "longitude": round(longitude, 4),
            "start_date": start_str,
            "end_date": end_str,
            "daily": daily_variable,
            "temperature_unit": temperature_unit,
            "timezone": timezone or "UTC",
        }

        should_cache = self.enable_cache if use_cache is None else use_cache
        cache_key = _compute_cache_key(params)

        if should_cache:
            cached_data = self._read_cache(cache_key)
            if cached_data is not None:
                logger.debug("Loaded historical weather data from cache (%s)", cache_key)
                return cached_data

        try:
            response = self.session.get(
                self.base_url,
                params=params,
                timeout=self.timeout,
                headers={"Accept": "application/json"},
            )
        except requests.exceptions.Timeout as err:
            raise WeatherDataAPIError(
                f"Historical weather API request timed out after {self.timeout} seconds."
            ) from err
        except requests.exceptions.RequestException as err:
            raise WeatherDataAPIError(
                f"Historical weather API request failed: {err}"
            ) from err

        if response.status_code != 200:
            raise WeatherDataAPIError(
                f"Open-Meteo archive API returned HTTP status {response.status_code}: {response.text}"
            )

        try:
            payload = response.json()
        except Exception as err:
            raise WeatherDataAPIError(
                f"Failed to parse Open-Meteo archive response as JSON: {err}"
            ) from err

        if should_cache:
            self._write_cache(cache_key, payload)

        return payload


def fetch_historical_raw(
    latitude: float,
    longitude: float,
    start_date: str | date | datetime,
    end_date: str | date | datetime,
    daily_variable: str = "temperature_2m_mean",
    temperature_unit: str = "celsius",
    timezone: str = "UTC",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    cache_dir: Optional[Path | str] = DEFAULT_CACHE_DIR,
    enable_cache: bool = True,
    session: Optional[requests.Session] = None,
) -> dict[str, Any]:
    """Convenience function to fetch historical weather using default client configuration."""
    client = HistoricalWeatherClient(
        timeout=timeout,
        cache_dir=cache_dir,
        enable_cache=enable_cache,
        session=session,
    )
    return client.fetch_historical_weather_raw(
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        daily_variable=daily_variable,
        temperature_unit=temperature_unit,
        timezone=timezone,
    )
