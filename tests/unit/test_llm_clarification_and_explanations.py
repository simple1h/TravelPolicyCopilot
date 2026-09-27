from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.agent.nodes import ask_clarification_node, rank_options_node
from src.domain.models import BudgetCheckResult, PolicyChunkRef, TravelOption, TripRequest
from src.llm.gateway import LLMGatewayError


def test_ask_clarification_uses_llm_question(monkeypatch):
    monkeypatch.setattr("src.agent.nodes.interrupt", None)

    def _fake_builder(**_kwargs):
        return (
            "Нужны уточнения: укажите город отправления и дату выезда.",
            {
                "calls": 1,
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
                "estimated_cost_usd": 0.00001,
            },
        )

    monkeypatch.setattr("src.agent.nodes._build_clarification_question_with_llm", _fake_builder)
    state = {
        "missing_fields": ["origin_city", "departure_date"],
        "trip_request": TripRequest(destination_city="Москва"),
        "last_user_message": "Нужно в Москву",
        "llm_usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0},
    }

    result = ask_clarification_node(state)

    assert result["clarification_question"].startswith("Нужны уточнения:")
    assert result["llm_usage"]["calls"] == 1
    assert result["llm_usage"]["total_tokens"] == 18


def test_ask_clarification_falls_back_when_llm_fails(monkeypatch):
    monkeypatch.setattr("src.agent.nodes.interrupt", None)

    def _raise_builder(**_kwargs):
        raise LLMGatewayError(code="LLM_PROVIDER_ERROR", message="provider down", retryable=True)

    monkeypatch.setattr("src.agent.nodes._build_clarification_question_with_llm", _raise_builder)
    state = {
        "missing_fields": ["origin_city", "return_date"],
        "trip_request": TripRequest(),
        "last_user_message": "Нужна поездка",
    }

    result = ask_clarification_node(state)

    assert "origin_city" in result["clarification_question"]
    assert "return_date" in result["clarification_question"]


def test_rank_options_enriches_explanation_with_llm(monkeypatch):
    def _fake_explainer(**_kwargs):
        return (
            "Опция OPT-001 соответствует политике и укладывается в бюджет.",
            {
                "calls": 1,
                "prompt_tokens": 9,
                "completion_tokens": 8,
                "total_tokens": 17,
                "estimated_cost_usd": 0.00001,
            },
        )

    monkeypatch.setattr("src.agent.nodes._build_grounded_explanation_with_llm", _fake_explainer)

    option = TravelOption(
        option_id="OPT-001",
        transport_type="flight",
        outbound_departure=datetime(2026, 9, 18, 9, 0, tzinfo=UTC),
        outbound_arrival=datetime(2026, 9, 18, 11, 0, tzinfo=UTC),
        inbound_departure=datetime(2026, 9, 19, 18, 0, tzinfo=UTC),
        inbound_arrival=datetime(2026, 9, 19, 20, 0, tzinfo=UTC),
        carrier="Mock Air",
        fare_class="economy",
        total_price=32000,
        segments=1,
    )
    state = {
        "travel_options": [option],
        "budget_results": [BudgetCheckResult(verdict="within_limit", total_price=32000, threshold=45000, overage=0)],
        "retrieved_policy_chunks": [
            PolicyChunkRef(document_id="POLICY-BUDGET", section_id="B1", source_path="data/policies/budget.md", score=0.9)
        ],
        "employee_profile": None,
        "trip_request": TripRequest(origin_city="Казань", destination_city="Москва"),
        "llm_usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0},
    }

    result = rank_options_node(state)

    assert len(result["ranked_options"]) == 1
    assert "соответствует политике" in result["ranked_options"][0].explanation
    assert result["llm_usage"]["calls"] == 1


def test_rank_options_uses_fallback_when_explainer_fails(monkeypatch):
    def _raise_explainer(**_kwargs):
        raise ValueError("bad json")

    monkeypatch.setattr("src.agent.nodes._build_grounded_explanation_with_llm", _raise_explainer)

    option = TravelOption(
        option_id="OPT-009",
        transport_type="flight",
        outbound_departure=datetime(2026, 9, 18, 9, 0, tzinfo=UTC),
        outbound_arrival=datetime(2026, 9, 18, 11, 0, tzinfo=UTC),
        inbound_departure=datetime(2026, 9, 19, 18, 0, tzinfo=UTC),
        inbound_arrival=datetime(2026, 9, 19, 20, 0, tzinfo=UTC),
        carrier="Mock Air",
        fare_class="economy",
        total_price=34000,
        segments=1,
    )
    state = {
        "travel_options": [option],
        "budget_results": [BudgetCheckResult(verdict="within_limit", total_price=34000, threshold=45000, overage=0)],
        "retrieved_policy_chunks": [
            PolicyChunkRef(document_id="POLICY-TRANSPORT", section_id="T1", source_path="data/policies/transport.md", score=0.87)
        ],
        "employee_profile": None,
        "trip_request": TripRequest(origin_city="Казань", destination_city="Москва"),
        "llm_usage": {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "estimated_cost_usd": 0.0},
    }

    result = rank_options_node(state)

    assert "Политика" in result["ranked_options"][0].explanation
