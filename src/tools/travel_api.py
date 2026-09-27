from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import httpx

from src.config import settings
from src.domain.models import TravelOption, TripRequest


TravelApiErrorCode = Literal[
    "TRAVEL_API_TIMEOUT",
    "TRAVEL_API_4XX",
    "TRAVEL_API_5XX",
    "TRAVEL_API_INVALID_RESPONSE",
    "TRAVEL_API_UNAVAILABLE",
]


@dataclass
class TravelApiError(RuntimeError):
    code: TravelApiErrorCode
    message: str
    retryable: bool
    status_code: int | None = None

    def __str__(self) -> str:
        return self.message


class TravelApiClient:
    def __init__(self, base_url: str, *, timeout_seconds: float) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def _build_payload(
        self,
        trip_request: TripRequest,
        *,
        scenario_mode: str = "normal",
    ) -> dict:
        return {
            "origin_city": trip_request.origin_city,
            "destination_city": trip_request.destination_city,
            "departure_date": trip_request.departure_date.isoformat() if trip_request.departure_date else None,
            "return_date": trip_request.return_date.isoformat() if trip_request.return_date else None,
            "meeting_datetime": trip_request.meeting_datetime.isoformat() if trip_request.meeting_datetime else None,
            "earliest_departure_hour": trip_request.earliest_departure_hour,
            "latest_arrival_hour": trip_request.latest_arrival_hour,
            "max_segments": trip_request.max_segments,
            "disallowed_time_windows": trip_request.disallowed_time_windows,
            "transport_preferences": trip_request.transport_preferences,
            "hotel_required": trip_request.hotel_required,
            "scenario_mode": scenario_mode,
        }

    def search_options(
        self,
        trip_request: TripRequest,
        *,
        scenario_mode: str = "normal",
    ) -> list[TravelOption]:
        payload = self._build_payload(trip_request, scenario_mode=scenario_mode)
        if not payload["origin_city"] or not payload["destination_city"]:
            raise TravelApiError(
                code="TRAVEL_API_INVALID_RESPONSE",
                message="Search request must include origin_city and destination_city.",
                retryable=False,
            )
        if not payload["departure_date"] or not payload["return_date"]:
            raise TravelApiError(
                code="TRAVEL_API_INVALID_RESPONSE",
                message="Search request must include departure_date and return_date.",
                retryable=False,
            )

        try:
            with httpx.Client(timeout=self._timeout_seconds) as client:
                response = client.post(f"{self._base_url}/v1/options/search", json=payload)
        except httpx.TimeoutException as exc:
            raise TravelApiError(
                code="TRAVEL_API_TIMEOUT",
                message="Mock Travel API request timed out.",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise TravelApiError(
                code="TRAVEL_API_UNAVAILABLE",
                message=f"Mock Travel API unavailable: {type(exc).__name__}",
                retryable=True,
            ) from exc

        if 400 <= response.status_code < 500:
            raise TravelApiError(
                code="TRAVEL_API_4XX",
                message=f"Mock Travel API rejected request with status {response.status_code}.",
                retryable=False,
                status_code=response.status_code,
            )
        if response.status_code >= 500:
            raise TravelApiError(
                code="TRAVEL_API_5XX",
                message=f"Mock Travel API failed with status {response.status_code}.",
                retryable=True,
                status_code=response.status_code,
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise TravelApiError(
                code="TRAVEL_API_INVALID_RESPONSE",
                message="Mock Travel API returned non-JSON response.",
                retryable=True,
                status_code=response.status_code,
            ) from exc

        options_raw = body.get("options")
        if not isinstance(options_raw, list):
            raise TravelApiError(
                code="TRAVEL_API_INVALID_RESPONSE",
                message="Mock Travel API response has invalid options payload.",
                retryable=True,
                status_code=response.status_code,
            )
        options = [TravelOption.model_validate(item) for item in options_raw]
        for option in options:
            if option.outbound_departure.date().isoformat() != payload["departure_date"]:
                raise TravelApiError(
                    code="TRAVEL_API_INVALID_RESPONSE",
                    message="Mock Travel API returned outbound date mismatch.",
                    retryable=True,
                    status_code=response.status_code,
                )
            if option.inbound_departure.date().isoformat() != payload["return_date"]:
                raise TravelApiError(
                    code="TRAVEL_API_INVALID_RESPONSE",
                    message="Mock Travel API returned inbound date mismatch.",
                    retryable=True,
                    status_code=response.status_code,
                )
        return options


_default_client = TravelApiClient(
    base_url=settings.mock_travel_api_url,
    timeout_seconds=settings.request_timeout_seconds,
)


def search_travel_options_http(
    trip_request: TripRequest,
    *,
    scenario_mode: str = "normal",
) -> list[TravelOption]:
    return _default_client.search_options(trip_request, scenario_mode=scenario_mode)
