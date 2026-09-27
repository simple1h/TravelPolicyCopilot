from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import settings
from src.llm.gateway import LLMGatewayError


@pytest.fixture()
def api_client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def disable_live_llm(monkeypatch):
    def _raise_llm_error(message: str, *, trace_id: str | None = None):
        _ = (message, trace_id)
        raise LLMGatewayError(code="LLM_PROVIDER_ERROR", message="offline", retryable=True)

    monkeypatch.setattr("src.agent.nodes.parse_trip_request_with_llm", _raise_llm_error)


@pytest.mark.integration
@pytest.mark.e2e
def test_api_message_incomplete_request_goes_to_clarification(api_client: TestClient):
    response = api_client.post(
        "/v1/agent/message",
        json={
            "employee_id": "EMP-001",
            "message": "Нужно организовать командировку в сентябре.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["current_step"] == "ask_clarification"
    assert "origin_city" in payload["missing_fields"]
    assert "destination_city" in payload["missing_fields"]
    assert "Нужны уточнения" in payload["answer"]


@pytest.mark.integration
@pytest.mark.e2e
def test_api_message_happy_path_returns_recommendations(api_client: TestClient):
    response = api_client.post(
        "/v1/agent/message",
        json={
            "employee_id": "EMP-001",
            "message": (
                "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
                "на встречу к 15:00. Вернуться на следующий день."
            ),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["current_step"] == "present_recommendations"
    assert "Рекомендуем" in payload["answer"]
    assert payload["trace_id"]


@pytest.mark.integration
@pytest.mark.e2e
def test_api_message_external_error_branch_returns_clarification(api_client: TestClient, monkeypatch):
    monkeypatch.setattr(settings, "backend_mode", "strict")
    response = api_client.post(
        "/v1/agent/message",
        json={
            "employee_id": "EMP-001",
            "message": (
                "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
                "на встречу к 15:00. Вернуться на следующий день. [scenario=error_5xx]"
            ),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["current_step"] == "propose_constraint_change"
    assert payload["travel_search_outcome"] == "retryable_external_error"
    assert "повторить поиск" in payload["answer"].lower()


@pytest.mark.integration
@pytest.mark.e2e
def test_api_message_no_options_branch_returns_recovery_step(api_client: TestClient):
    response = api_client.post(
        "/v1/agent/message",
        json={
            "employee_id": "EMP-003",
            "message": (
                "Мне нужно 18 сентября утром попасть из Казани в Москву и вернуться на следующий день. "
                "[scenario=empty]"
            ),
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["current_step"] == "propose_constraint_change"
    assert payload["travel_search_outcome"] == "options_not_found"


@pytest.mark.integration
@pytest.mark.e2e
def test_stateful_api_session_flow_creates_approval_request(api_client: TestClient):
    created = api_client.post(
        "/v1/agent/sessions",
        json={
            "employee_id": "EMP-001",
            "message": (
                "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
                "на встречу к 15:00. Вернуться на следующий день."
            ),
        },
    )
    assert created.status_code == 200
    session = created.json()
    assert session["current_step"] == "present_recommendations"
    assert session["recommendations"]

    selected = api_client.post(
        f"/v1/agent/sessions/{session['session_id']}/actions",
        json={
            "employee_id": "EMP-001",
            "action": "select_option",
            "selected_option_id": session["recommendations"][0]["option_id"],
        },
    )
    assert selected.status_code == 200
    selected_state = selected.json()
    assert selected_state["current_step"] == "request_confirmation"

    confirmed = api_client.post(
        f"/v1/agent/sessions/{session['session_id']}/actions",
        json={
            "employee_id": "EMP-001",
            "action": "confirm",
        },
    )
    assert confirmed.status_code == 200
    final_state = confirmed.json()
    assert final_state["current_step"] == "show_result"
    assert final_state["approval_request_id"]


@pytest.mark.integration
@pytest.mark.e2e
def test_stateful_api_denies_session_access_for_other_employee(api_client: TestClient):
    created = api_client.post(
        "/v1/agent/sessions",
        json={"employee_id": "EMP-001", "message": "Нужна командировка в СПБ"},
    )
    assert created.status_code == 200
    session_id = created.json()["session_id"]

    response = api_client.get(
        f"/v1/agent/sessions/{session_id}",
        params={"employee_id": "EMP-002"},
    )
    assert response.status_code == 403


@pytest.mark.integration
@pytest.mark.e2e
def test_stateful_api_reject_action_does_not_create_request(api_client: TestClient):
    created = api_client.post(
        "/v1/agent/sessions",
        json={
            "employee_id": "EMP-001",
            "message": (
                "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
                "на встречу к 15:00. Вернуться на следующий день."
            ),
        },
    )
    assert created.status_code == 200
    session = created.json()
    selected = api_client.post(
        f"/v1/agent/sessions/{session['session_id']}/actions",
        json={
            "employee_id": "EMP-001",
            "action": "select_option",
            "selected_option_id": session["recommendations"][0]["option_id"],
        },
    )
    assert selected.status_code == 200
    rejected = api_client.post(
        f"/v1/agent/sessions/{session['session_id']}/actions",
        json={"employee_id": "EMP-001", "action": "reject"},
    )
    assert rejected.status_code == 200
    payload = rejected.json()
    assert payload["current_step"] == "request_confirmation"
    assert payload["approval_request_id"] is None
    assert "create_approval_request" not in payload["tool_calls"]


@pytest.mark.integration
@pytest.mark.e2e
def test_stateful_api_recovery_update_constraints_keeps_session(api_client: TestClient):
    created = api_client.post(
        "/v1/agent/sessions",
        json={
            "employee_id": "EMP-003",
            "message": (
                "Мне нужно 18 сентября утром попасть из Казани в Москву и вернуться на следующий день. "
                "[scenario=empty]"
            ),
        },
    )
    assert created.status_code == 200
    session = created.json()
    assert session["current_step"] == "propose_constraint_change"
    session_id = session["session_id"]

    updated = api_client.post(
        f"/v1/agent/sessions/{session_id}/actions",
        json={
            "employee_id": "EMP-003",
            "action": "update_constraints",
            "trip_request_patch": {
                "departure_date": "2026-09-10",
                "return_date": "2026-09-11",
                "max_segments": 2,
            },
        },
    )
    assert updated.status_code == 200
    payload = updated.json()
    assert payload["session_id"] == session_id
    assert payload["current_step"] in {"present_recommendations", "ask_clarification", "propose_constraint_change"}


@pytest.mark.integration
@pytest.mark.e2e
def test_stateful_api_recovery_cancel_ends_flow(api_client: TestClient):
    created = api_client.post(
        "/v1/agent/sessions",
        json={
            "employee_id": "EMP-003",
            "message": (
                "Мне нужно 18 сентября утром попасть из Казани в Москву и вернуться на следующий день. "
                "[scenario=empty]"
            ),
        },
    )
    assert created.status_code == 200
    session_id = created.json()["session_id"]
    cancelled = api_client.post(
        f"/v1/agent/sessions/{session_id}/actions",
        json={"employee_id": "EMP-003", "action": "cancel"},
    )
    assert cancelled.status_code == 200
    payload = cancelled.json()
    assert payload["current_step"] == "end"
    assert "отмен" in payload["answer"].lower()


@pytest.mark.integration
@pytest.mark.e2e
def test_update_constraints_validates_date_order(api_client: TestClient):
    created = api_client.post(
        "/v1/agent/sessions",
        json={
            "employee_id": "EMP-003",
            "message": (
                "Мне нужно 18 сентября утром попасть из Казани в Москву и вернуться на следующий день. "
                "[scenario=empty]"
            ),
        },
    )
    session_id = created.json()["session_id"]
    invalid_patch = api_client.post(
        f"/v1/agent/sessions/{session_id}/actions",
        json={
            "employee_id": "EMP-003",
            "action": "update_constraints",
            "trip_request_patch": {
                "departure_date": "2026-09-20",
                "return_date": "2026-09-19",
            },
        },
    )
    assert invalid_patch.status_code == 400
