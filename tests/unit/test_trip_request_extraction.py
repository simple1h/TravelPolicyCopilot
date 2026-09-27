from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from src.agent.nodes import parse_request_node, parse_trip_request_with_llm
from src.llm.gateway import LLMGatewayError


class _FakeGateway:
    def __init__(self, content: str | None = None, *, should_raise: bool = False) -> None:
        self._content = content
        self._should_raise = should_raise

    def complete(self, **_kwargs):
        if self._should_raise:
            raise LLMGatewayError(
                code="LLM_PROVIDER_ERROR",
                message="provider failed",
                retryable=True,
            )
        return SimpleNamespace(
            content=self._content,
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                estimated_cost_usd=0.00001,
            ),
        )


def test_structured_extraction_happy_path(monkeypatch):
    fake_gateway = _FakeGateway(
        content=(
            '{"origin_city":"Казань","destination_city":"Санкт-Петербург",'
            '"departure_date":"2026-09-18","return_date":"2026-09-19",'
            '"meeting_datetime":null,"disallowed_time_windows":[],'
            '"transport_preferences":["flight"],"hotel_required":true,'
            '"comment":"встреча"}'
        )
    )
    monkeypatch.setattr("src.agent.nodes.get_default_llm_gateway", lambda: fake_gateway)

    trip, usage = parse_trip_request_with_llm("Нужна поездка из Казани в СПБ 18-19 сентября")

    assert trip.origin_city == "Казань"
    assert trip.destination_city == "Санкт-Петербург"
    assert str(trip.departure_date) == "2026-09-18"
    assert str(trip.return_date) == "2026-09-19"
    assert trip.hotel_required is True
    assert usage["total_tokens"] == 15


def test_structured_extraction_ambiguous_keeps_unknown_fields(monkeypatch):
    fake_gateway = _FakeGateway(
        content=(
            '{"origin_city":null,"destination_city":"Москва",'
            '"departure_date":null,"return_date":null,'
            '"meeting_datetime":null,"disallowed_time_windows":[],'
            '"transport_preferences":[],"hotel_required":true,'
            '"comment":"завтра в москву"}'
        )
    )
    monkeypatch.setattr("src.agent.nodes.get_default_llm_gateway", lambda: fake_gateway)

    trip, usage = parse_trip_request_with_llm("Нужно завтра в Москву")

    assert trip.destination_city == "Москва"
    assert trip.origin_city is None
    assert trip.departure_date is None
    assert trip.return_date is None
    assert usage["calls"] == 1


def test_parse_request_falls_back_on_invalid_llm_output(monkeypatch):
    fake_gateway = _FakeGateway(content="not-a-json")
    monkeypatch.setattr("src.agent.nodes.get_default_llm_gateway", lambda: fake_gateway)

    state = {"last_user_message": "Нужна командировка в СПБ", "trip_request": None}
    result = parse_request_node(state)

    assert result["trip_request"] is not None
    assert result["trip_request"].destination_city == "Санкт-Петербург"


def test_structured_extraction_normalizes_nullable_fields(monkeypatch):
    fake_gateway = _FakeGateway(
        content=(
            '{"origin_city":"Казань","destination_city":"Москва",'
            '"departure_date":"2026-09-18","return_date":"2026-09-19",'
            '"meeting_datetime":null,"disallowed_time_windows":null,'
            '"transport_preferences":null,"hotel_required":null,'
            '"comment":null}'
        )
    )
    monkeypatch.setattr("src.agent.nodes.get_default_llm_gateway", lambda: fake_gateway)

    trip, _usage = parse_trip_request_with_llm("Я хочу в командировку в Москву")

    assert trip.disallowed_time_windows == []
    assert trip.transport_preferences == []
    assert trip.hotel_required is True


def test_structured_extraction_canonicalizes_city_aliases(monkeypatch):
    fake_gateway = _FakeGateway(
        content=(
            '{"origin_city":"казани","destination_city":"Питер",'
            '"departure_date":"2026-09-18","return_date":"2026-09-19",'
            '"meeting_datetime":null,"disallowed_time_windows":[],"transport_preferences":[],'
            '"hotel_required":true,"comment":null}'
        )
    )
    monkeypatch.setattr("src.agent.nodes.get_default_llm_gateway", lambda: fake_gateway)

    trip, _usage = parse_trip_request_with_llm("Я хочу в командировку в Питер из Казани")

    assert trip.origin_city == "Казань"
    assert trip.destination_city == "Санкт-Петербург"


def test_parse_request_enriches_outdated_llm_dates(monkeypatch):
    fake_gateway = _FakeGateway(
        content=(
            '{"origin_city":"Казань","destination_city":"Санкт-Петербург",'
            '"departure_date":"2023-09-18","return_date":"2023-09-19",'
            '"meeting_datetime":null,"disallowed_time_windows":[],"transport_preferences":[],'
            '"hotel_required":true,"comment":null}'
        )
    )
    monkeypatch.setattr("src.agent.nodes.get_default_llm_gateway", lambda: fake_gateway)

    state = {
        "last_user_message": (
            "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
            "на встречу к 15:00. Вернуться на следующий день."
        ),
        "trip_request": None,
    }
    result = parse_request_node(state)

    assert result["trip_request"] is not None
    assert result["trip_request"].departure_date.year == date.today().year
