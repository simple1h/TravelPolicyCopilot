from __future__ import annotations

from datetime import date

from src.mock_travel_api.catalog import build_catalog
from src.mock_travel_api.schemas import TravelOptionsSearchRequest
from src.mock_travel_api.service import get_catalog, reset_catalog_cache, search_offers


def test_build_catalog_from_today_excludes_past_dates(monkeypatch):
    import datetime as dt

    class FakeDate(dt.date):
        @classmethod
        def today(cls):
            return dt.date(2026, 9, 10)

    monkeypatch.setattr("src.mock_travel_api.catalog.date", FakeDate)
    catalog = build_catalog(horizon_days=30)
    outbound_days = {offer.outbound_departure.date() for offer in catalog}
    assert min(outbound_days) == dt.date(2026, 9, 10)
    assert dt.date(2026, 9, 9) not in outbound_days


def test_build_catalog_with_explicit_base_includes_past_for_tests():
    catalog = build_catalog(base_date=date(2026, 9, 1), horizon_days=5)
    outbound_days = sorted({offer.outbound_departure.date() for offer in catalog})
    assert outbound_days[0] == date(2026, 9, 1)
    assert len(outbound_days) == 5


def test_get_catalog_refreshes_when_day_changes(monkeypatch):
    import datetime as dt

    reset_catalog_cache()

    class FakeDate(dt.date):
        current = dt.date(2026, 9, 3)

        @classmethod
        def today(cls):
            return cls.current

    monkeypatch.setattr("src.mock_travel_api.service.date", FakeDate)
    monkeypatch.setattr("src.mock_travel_api.catalog.date", FakeDate)
    monkeypatch.delenv("MOCK_TRAVEL_CATALOG_BASE_DATE", raising=False)
    first = get_catalog()
    first_from = min(offer.outbound_departure.date() for offer in first)
    assert first_from == dt.date(2026, 9, 3)

    FakeDate.current = dt.date(2026, 9, 5)
    second = get_catalog()
    second_from = min(offer.outbound_departure.date() for offer in second)
    assert second_from == dt.date(2026, 9, 5)
    reset_catalog_cache()


def test_build_catalog_generates_multiple_return_offsets():
    catalog = build_catalog(base_date=date(2026, 9, 20), horizon_days=1, return_offsets=(1, 3))
    kazan_moscow = [
        offer
        for offer in catalog
        if offer.origin_city == "Казань" and offer.destination_city == "Москва"
    ]
    inbound_days = {offer.inbound_departure.date() for offer in kazan_moscow}
    assert inbound_days == {date(2026, 9, 21), date(2026, 9, 23)}


def test_search_finds_kazan_moscow_with_three_day_return(monkeypatch):
    catalog = build_catalog(base_date=date(2026, 9, 1), horizon_days=30, return_offsets=(1, 2, 3, 5, 7))
    monkeypatch.setattr("src.mock_travel_api.service.get_catalog", lambda: catalog)
    payload = TravelOptionsSearchRequest(
        origin_city="Казань",
        destination_city="Москва",
        departure_date=date(2026, 9, 20),
        return_date=date(2026, 9, 23),
        earliest_departure_hour=6,
        latest_arrival_hour=17,
    )
    options = search_offers(payload)

    assert len(options) >= 1
    assert all(offer.outbound_departure.date() == date(2026, 9, 20) for offer in options)
    assert all(offer.inbound_departure.date() == date(2026, 9, 23) for offer in options)
    assert all(offer.outbound_departure.hour >= 6 for offer in options)
    assert all(offer.outbound_arrival.hour <= 17 for offer in options)
