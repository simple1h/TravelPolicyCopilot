from __future__ import annotations

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from src.mock_travel_api.main import app
from src.mock_travel_api.service import get_catalog_size, reset_catalog_cache


@pytest.fixture(autouse=True)
def fixed_demo_catalog(monkeypatch):
    monkeypatch.setenv("MOCK_TRAVEL_CATALOG_BASE_DATE", "2026-09-01")
    monkeypatch.setenv("MOCK_TRAVEL_CATALOG_HORIZON_DAYS", "30")
    reset_catalog_cache()
    yield
    reset_catalog_cache()


client = TestClient(app)


def test_health_returns_catalog_size():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["catalog_size"] >= 100
    assert get_catalog_size() >= 100


def test_search_returns_multiple_options_for_happy_path():
    response = client.post(
        "/v1/options/search",
        json={
            "origin_city": "Казань",
            "destination_city": "Москва",
            "departure_date": "2026-09-01",
            "return_date": "2026-09-02",
            "transport_preferences": ["flight", "train"],
            "allowed_fare_classes": ["economy", "premium_economy", "business"],
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_options"] >= 2
    assert len(payload["options"]) >= 2


def test_search_supports_empty_and_error_scenarios():
    empty_response = client.post(
        "/v1/options/search",
        json={
            "origin_city": "Казань",
            "destination_city": "Москва",
            "departure_date": "2026-09-01",
            "return_date": "2026-09-02",
            "scenario_mode": "empty",
        },
    )
    assert empty_response.status_code == 200
    assert empty_response.json()["total_options"] == 0

    error_4xx = client.post(
        "/v1/options/search",
        json={
            "origin_city": "Казань",
            "destination_city": "Москва",
            "departure_date": "2026-09-01",
            "return_date": "2026-09-02",
            "scenario_mode": "error_4xx",
        },
    )
    assert error_4xx.status_code == 422

    error_5xx = client.post(
        "/v1/options/search",
        json={
            "origin_city": "Казань",
            "destination_city": "Москва",
            "departure_date": "2026-09-01",
            "return_date": "2026-09-02",
            "scenario_mode": "error_5xx",
        },
    )
    assert error_5xx.status_code == 503


def test_search_applies_meeting_deadline_and_hotel_requirement():
    response = client.post(
        "/v1/options/search",
        json={
            "origin_city": "Казань",
            "destination_city": "Санкт-Петербург",
            "departure_date": "2026-09-01",
            "return_date": "2026-09-02",
            "meeting_datetime": "2026-09-01T10:00:00+00:00",
            "hotel_required": False,
            "max_segments": 2,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["options"]
    deadline = datetime.fromisoformat("2026-09-01T10:00:00+00:00")
    for offer in payload["options"]:
        arrival = datetime.fromisoformat(offer["outbound_arrival"].replace("Z", "+00:00"))
        assert arrival <= deadline
        assert offer["hotel_name"] is None
        assert offer["hotel_nightly_price"] is None
