from __future__ import annotations

from datetime import date, datetime
import types

import pytest

from src.agent.service import bootstrap_state, resume_agent_step, run_agent_step
from src.config import settings
from src.domain.models import EmployeeProfile, TravelOption, TripRequest
from src.rag.retrieval import RagUnavailableError, retrieve_policy_chunks
from src.tools.stubs import BackendDependencyError, search_travel_options
from src.tools.travel_api import TravelApiError


def _profile() -> EmployeeProfile:
    return EmployeeProfile(
        employee_id="EMP-TEST",
        employee_category="standard",
        cost_center="CC-TEST",
        grade="G6",
        home_city="Kazan",
        policy_tier="default",
        max_hotel_per_night=7000,
        max_flight_class="economy",
        requires_preapproval_over_amount=45000,
        accessibility_requirements=[],
    )


def test_strict_mode_raises_on_travel_api_error(monkeypatch):
    monkeypatch.setattr(settings, "backend_mode", "strict")

    def _raise_http(*args, **kwargs):
        _ = (args, kwargs)
        raise TravelApiError(code="TRAVEL_API_UNAVAILABLE", message="offline", retryable=True)

    monkeypatch.setattr("src.tools.stubs.search_travel_options_http", _raise_http)
    trip_request = TripRequest(
        origin_city="Казань",
        destination_city="Москва",
        departure_date=date(2026, 9, 18),
        return_date=date(2026, 9, 19),
    )

    with pytest.raises(BackendDependencyError) as exc_info:
        search_travel_options(trip_request)
    assert exc_info.value.code == "TRAVEL_API_UNAVAILABLE"


def test_demo_fallback_returns_options_when_travel_api_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "backend_mode", "demo_fallback")

    def _raise_http(*args, **kwargs):
        _ = (args, kwargs)
        raise TravelApiError(code="TRAVEL_API_UNAVAILABLE", message="offline", retryable=True)

    monkeypatch.setattr("src.tools.stubs.search_travel_options_http", _raise_http)
    trip_request = TripRequest(
        origin_city="Казань",
        destination_city="Москва",
        departure_date=date(2026, 9, 18),
        return_date=date(2026, 9, 19),
    )

    options = search_travel_options(trip_request)
    assert options
    assert options[0].option_id == "OPT-001"


def test_strict_mode_raises_when_qdrant_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "backend_mode", "strict")
    monkeypatch.setattr(settings, "embedding_provider", "router.ai")
    monkeypatch.setattr(settings, "embedding_api_key", "test-key")
    monkeypatch.setattr(
        "src.rag.retrieval.embed_texts",
        lambda texts, is_query, trace_id=None: (
            [[0.01] * settings.embedding_dimension for _ in texts],
            types.SimpleNamespace(
                provider="router.ai",
                model="baai/bge-m3",
                mode="remote",
                input_tokens=10,
                estimated_cost_usd=0.0,
                latency_ms=1,
                cost_known=True,
            ),
        ),
    )
    monkeypatch.setattr("src.rag.retrieval._retrieve_from_qdrant", lambda *args, **kwargs: None)

    with pytest.raises(RagUnavailableError) as exc_info:
        retrieve_policy_chunks("test", _profile())
    assert exc_info.value.code == "RAG_UNAVAILABLE"


def test_strict_mode_requires_external_embedding_provider(monkeypatch):
    monkeypatch.setattr(settings, "backend_mode", "strict")
    monkeypatch.setattr(settings, "embedding_provider", "local_hash")
    monkeypatch.setattr(settings, "embedding_api_key", "")

    with pytest.raises(RagUnavailableError) as exc_info:
        retrieve_policy_chunks("test", _profile())
    assert exc_info.value.code == "EMBEDDING_PROVIDER_UNAVAILABLE"


def test_confirm_does_not_show_result_when_repository_fails(monkeypatch):
    monkeypatch.setattr(settings, "backend_mode", "strict")
    monkeypatch.setattr(settings, "embedding_provider", "router.ai")
    monkeypatch.setattr(settings, "embedding_api_key", "test-key")
    monkeypatch.setattr(
        "src.rag.retrieval.embed_texts",
        lambda texts, is_query, trace_id=None: (
            [[0.01] * settings.embedding_dimension for _ in texts],
            types.SimpleNamespace(
                provider="router.ai",
                model="baai/bge-m3",
                mode="remote",
                input_tokens=10,
                estimated_cost_usd=0.0,
                latency_ms=1,
                cost_known=True,
            ),
        ),
    )
    monkeypatch.setattr(
        "src.rag.retrieval._retrieve_from_qdrant",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "src.tools.service.search_travel_options",
        lambda trip_request: [
            TravelOption(
                option_id="OPT-TST",
                transport_type="flight",
                outbound_departure=datetime(2026, 9, 18, 9, 0),
                outbound_arrival=datetime(2026, 9, 18, 11, 10),
                inbound_departure=datetime(2026, 9, 19, 18, 30),
                inbound_arrival=datetime(2026, 9, 19, 20, 40),
                carrier="TestAir",
                fare_class="economy",
                total_price=32000,
                segments=2,
                refundable=True,
            )
        ],
    )
    monkeypatch.setattr("src.tools.stubs.employee_repository.get_by_id", lambda employee_id: _profile())

    def _raise_write(*args, **kwargs):
        _ = (args, kwargs)
        raise RuntimeError("db down")

    monkeypatch.setattr(
        "src.tools.service.approval_request_repository.create_idempotent",
        _raise_write,
    )

    message = (
        "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
        "на встречу к 15:00. Вернуться на следующий день."
    )
    state = run_agent_step(bootstrap_state(employee_id="EMP-001", user_message=message))
    selected_option_id = state["ranked_options"][0].option_id
    failed_write_state = resume_agent_step(
        state,
        selected_option_id=selected_option_id,
        confirmation_status="confirmed",
        error=None,
    )

    assert failed_write_state["current_step"] == "end"
    assert failed_write_state.get("approval_request_id") is None
    assert failed_write_state["error"].code == "APPROVAL_WRITE_UNAVAILABLE"
