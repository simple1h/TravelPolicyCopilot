from __future__ import annotations

import pytest
import httpx

from src.config import settings
from src.domain.models import TripRequest
from src.tools.travel_api import TravelApiClient, TravelApiError


def _trip_request() -> TripRequest:
    return TripRequest(
        origin_city="Казань",
        destination_city="Москва",
        departure_date="2026-09-01",
        return_date="2026-09-02",
        disallowed_time_windows=[],
        transport_preferences=["flight", "train"],
    )


def _require_mock_api_ready() -> None:
    health_url = f"{settings.mock_travel_api_url.rstrip('/')}/health"
    try:
        response = httpx.get(health_url, timeout=1.5)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Mock Travel API is not reachable: {exc}")
    if payload.get("service") != "mock-travel-api":
        pytest.skip("Mock Travel API is not updated to FastAPI implementation.")


def test_travel_api_client_gets_options_via_http():
    _require_mock_api_ready()
    client = TravelApiClient(settings.mock_travel_api_url, timeout_seconds=2.0)
    trip_request = _trip_request()
    try:
        options = client.search_options(trip_request)
    except TravelApiError as exc:  # pragma: no cover - environment dependent
        if exc.code in {"TRAVEL_API_UNAVAILABLE", "TRAVEL_API_TIMEOUT"}:
            pytest.skip(f"Mock Travel API is not available: {exc}")
        raise

    assert len(options) >= 2
    assert options[0].option_id.startswith("OPT-")
    assert options[0].total_price > 0


def test_travel_api_client_handles_5xx():
    _require_mock_api_ready()
    client = TravelApiClient(settings.mock_travel_api_url, timeout_seconds=2.0)
    trip_request = _trip_request()
    try:
        with pytest.raises(TravelApiError) as exc_info:
            client.search_options(trip_request, scenario_mode="error_5xx")
    except TravelApiError as exc:  # pragma: no cover - environment dependent
        if exc.code in {"TRAVEL_API_UNAVAILABLE", "TRAVEL_API_TIMEOUT"}:
            pytest.skip(f"Mock Travel API is not available: {exc}")
        raise

    assert exc_info.value.code == "TRAVEL_API_5XX"
    assert exc_info.value.retryable is True


def test_travel_api_client_handles_4xx():
    _require_mock_api_ready()
    client = TravelApiClient(settings.mock_travel_api_url, timeout_seconds=2.0)
    trip_request = _trip_request()
    try:
        with pytest.raises(TravelApiError) as exc_info:
            client.search_options(trip_request, scenario_mode="error_4xx")
    except TravelApiError as exc:  # pragma: no cover - environment dependent
        if exc.code in {"TRAVEL_API_UNAVAILABLE", "TRAVEL_API_TIMEOUT"}:
            pytest.skip(f"Mock Travel API is not available: {exc}")
        raise

    assert exc_info.value.code == "TRAVEL_API_4XX"
    assert exc_info.value.retryable is False


def test_travel_api_client_handles_timeout():
    _require_mock_api_ready()
    client = TravelApiClient(settings.mock_travel_api_url, timeout_seconds=0.2)
    trip_request = _trip_request()
    try:
        with pytest.raises(TravelApiError) as exc_info:
            client.search_options(trip_request, scenario_mode="timeout")
    except TravelApiError as exc:  # pragma: no cover - environment dependent
        if exc.code == "TRAVEL_API_UNAVAILABLE":
            pytest.skip(f"Mock Travel API is not available: {exc}")
        raise

    assert exc_info.value.code == "TRAVEL_API_TIMEOUT"
    assert exc_info.value.retryable is True
