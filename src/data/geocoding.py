"""Geocoding module using the Open-Meteo Geocoding API."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional

import requests

from src.data.exceptions import GeocodingError

logger = logging.getLogger(__name__)

GEOCODING_BASE_URL = "https://geocoding-api.open-meteo.com/v1/search"
DEFAULT_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class LocationResult:
    """Represents a geocoded location candidate."""

    name: str
    latitude: float
    longitude: float
    country: str = ""
    country_code: str = ""
    admin1: str = ""
    timezone: str = "UTC"

    @classmethod
    def from_api_dict(cls, data: dict[str, Any]) -> LocationResult:
        """Construct a LocationResult from an Open-Meteo geocoding item."""
        try:
            return cls(
                name=str(data.get("name", "")).strip(),
                latitude=float(data["latitude"]),
                longitude=float(data["longitude"]),
                country=str(data.get("country", "")).strip(),
                country_code=str(data.get("country_code", "")).strip(),
                admin1=str(data.get("admin1", "")).strip(),
                timezone=str(data.get("timezone", "UTC")).strip() or "UTC",
            )
        except (KeyError, TypeError, ValueError) as err:
            raise GeocodingError(
                f"Invalid geocoding response entry: missing or invalid coordinate fields. Raw: {data}"
            ) from err

    def display_name(self) -> str:
        """Return human-readable location description (City, Admin1, Country)."""
        parts = [self.name]
        if self.admin1 and self.admin1 != self.name:
            parts.append(self.admin1)
        if self.country:
            parts.append(self.country)
        return ", ".join(parts)


def search_locations(
    query: str,
    count: int = 5,
    language: str = "en",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    session: Optional[requests.Session] = None,
) -> List[LocationResult]:
    """Search for matching geographical locations for a given city query.

    Args:
        query: Name of the city or municipality to search.
        count: Maximum number of search candidates to return (1-100).
        language: Language code for results (default 'en').
        timeout: Network request timeout in seconds.
        session: Optional custom requests Session.

    Returns:
        List of LocationResult instances matching the query.

    Raises:
        GeocodingError: If query is invalid, network fails, or status code is non-200.
    """
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise GeocodingError("City query must be a non-empty string.")

    params = {
        "name": cleaned_query,
        "count": max(1, min(count, 100)),
        "language": language,
        "format": "json",
    }

    client = session or requests
    try:
        response = client.get(
            GEOCODING_BASE_URL,
            params=params,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
    except requests.exceptions.Timeout as err:
        raise GeocodingError(
            f"Geocoding request for '{cleaned_query}' timed out after {timeout} seconds."
        ) from err
    except requests.exceptions.RequestException as err:
        raise GeocodingError(
            f"Geocoding request for '{cleaned_query}' failed: {err}"
        ) from err

    if response.status_code != 200:
        raise GeocodingError(
            f"Geocoding API returned HTTP status {response.status_code}: {response.text}"
        )

    try:
        payload = response.json()
    except Exception as err:
        raise GeocodingError(
            f"Failed to parse geocoding response as JSON: {err}"
        ) from err

    results_data = payload.get("results")
    if not results_data or not isinstance(results_data, list):
        return []

    return [LocationResult.from_api_dict(item) for item in results_data]


def geocode_city(
    query: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    session: Optional[requests.Session] = None,
) -> LocationResult:
    """Geocode a city name and return the top matching location.

    Args:
        query: City or locality name to look up.
        timeout: Network timeout in seconds.
        session: Optional custom requests Session.

    Returns:
        The highest-ranked LocationResult.

    Raises:
        GeocodingError: If the city cannot be found or the request fails.
    """
    candidates = search_locations(query=query, count=1, timeout=timeout, session=session)
    if not candidates:
        raise GeocodingError(f"No location found matching city query '{query.strip()}'.")
    return candidates[0]
