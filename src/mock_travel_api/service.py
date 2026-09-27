from __future__ import annotations

import os
from datetime import UTC, date

from src.mock_travel_api.catalog import DEFAULT_RETURN_OFFSETS, build_catalog
from src.mock_travel_api.schemas import TravelOffer, TravelOptionsSearchRequest

_CATALOG: list[TravelOffer] | None = None
_CATALOG_CACHE_KEY: tuple[date, date | None, int, tuple[int, ...]] | None = None
_DEFAULT_HORIZON_DAYS = 90


def _resolve_catalog_base_date() -> date | None:
    raw = os.getenv("MOCK_TRAVEL_CATALOG_BASE_DATE", "").strip()
    if not raw:
        return None
    return date.fromisoformat(raw)


def _resolve_horizon_days() -> int:
    raw = os.getenv("MOCK_TRAVEL_CATALOG_HORIZON_DAYS", "").strip()
    if not raw:
        return _DEFAULT_HORIZON_DAYS
    return max(7, int(raw))


def _resolve_return_offsets() -> tuple[int, ...]:
    raw = os.getenv("MOCK_TRAVEL_RETURN_OFFSETS", "").strip()
    if not raw:
        return DEFAULT_RETURN_OFFSETS
    offsets = tuple(
        sorted({max(1, int(part.strip())) for part in raw.split(",") if part.strip()})
    )
    return offsets or DEFAULT_RETURN_OFFSETS


def get_catalog() -> list[TravelOffer]:
    global _CATALOG
    global _CATALOG_CACHE_KEY

    today = date.today()
    base_date = _resolve_catalog_base_date()
    horizon_days = _resolve_horizon_days()
    return_offsets = _resolve_return_offsets()
    cache_key = (today, base_date, horizon_days, return_offsets)
    if _CATALOG is None or _CATALOG_CACHE_KEY != cache_key:
        _CATALOG = build_catalog(
            base_date=base_date,
            horizon_days=horizon_days,
            return_offsets=return_offsets,
        )
        _CATALOG_CACHE_KEY = cache_key
    return _CATALOG


def get_catalog_date_range() -> tuple[date | None, date | None]:
    catalog = get_catalog()
    if not catalog:
        return None, None
    outbound_days = [offer.outbound_departure.date() for offer in catalog]
    return min(outbound_days), max(outbound_days)


def get_catalog_size() -> int:
    return len(get_catalog())


def reset_catalog_cache() -> None:
    global _CATALOG
    global _CATALOG_CACHE_KEY
    _CATALOG = None
    _CATALOG_CACHE_KEY = None


def _is_disallowed_hour(hour: int, disallowed_windows: list[str]) -> bool:
    for window in disallowed_windows:
        try:
            start_raw, end_raw = window.split("-", maxsplit=1)
            start = int(start_raw.split(":", maxsplit=1)[0])
            end = int(end_raw.split(":", maxsplit=1)[0])
        except Exception:
            continue
        if start <= hour < end:
            return True
    return False


def _matches_date(offer: TravelOffer, departure_date: date, return_date: date) -> bool:
    outbound_day = offer.outbound_departure.date()
    inbound_day = offer.inbound_departure.date()
    return outbound_day == departure_date and inbound_day == return_date


def _supports_accessibility(offer: TravelOffer, requirements: list[str]) -> bool:
    if not requirements:
        return True
    if "wheelchair_access" in requirements:
        # Simple MVP heuristic: direct daytime flights are considered accessible.
        return offer.transport_type == "flight" and offer.segments == 1 and offer.outbound_departure.hour >= 7
    return True


def search_offers(payload: TravelOptionsSearchRequest) -> list[TravelOffer]:
    catalog = get_catalog()
    options = [
        offer
        for offer in catalog
        if offer.origin_city == payload.origin_city and offer.destination_city == payload.destination_city
    ]

    options = [
        offer for offer in options if _matches_date(offer, payload.departure_date, payload.return_date)
    ]

    if payload.transport_preferences:
        options = [offer for offer in options if offer.transport_type in payload.transport_preferences]

    if payload.allowed_fare_classes:
        options = [offer for offer in options if offer.fare_class in payload.allowed_fare_classes]

    if payload.max_segments is not None:
        options = [offer for offer in options if offer.segments <= payload.max_segments]

    if payload.earliest_departure_hour is not None:
        options = [
            offer
            for offer in options
            if offer.outbound_departure.hour >= payload.earliest_departure_hour
        ]

    if payload.latest_arrival_hour is not None:
        options = [
            offer
            for offer in options
            if offer.outbound_arrival.hour <= payload.latest_arrival_hour
        ]

    if payload.meeting_datetime is not None:
        meeting_deadline = payload.meeting_datetime
        if meeting_deadline.tzinfo is None:
            meeting_deadline = meeting_deadline.replace(tzinfo=UTC)
        options = [offer for offer in options if offer.outbound_arrival <= meeting_deadline]

    if payload.disallowed_time_windows:
        options = [
            offer
            for offer in options
            if not _is_disallowed_hour(offer.outbound_departure.hour, payload.disallowed_time_windows)
        ]

    if payload.hotel_required is False:
        normalized_without_hotel: list[TravelOffer] = []
        for offer in options:
            hotel_price = offer.hotel_nightly_price or 0
            normalized_without_hotel.append(
                offer.model_copy(
                    update={
                        "hotel_name": None,
                        "hotel_nightly_price": None,
                        "total_price": max(0, offer.total_price - hotel_price),
                    }
                )
            )
        options = normalized_without_hotel

    if payload.accessibility_requirements:
        options = [offer for offer in options if _supports_accessibility(offer, payload.accessibility_requirements)]

    return options
