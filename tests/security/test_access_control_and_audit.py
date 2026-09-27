from __future__ import annotations

from src.agent.service import bootstrap_state, resume_agent_step, run_agent_step
from src.observability.tracing import sanitize_for_logging


def _message() -> str:
    return (
        "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
        "на встречу к 15:00. Вернуться на следующий день."
    )


def test_unknown_employee_profile_is_denied():
    state = bootstrap_state(employee_id="EMP-UNKNOWN", user_message=_message())
    result = run_agent_step(state)

    assert result["current_step"] == "end"
    assert result["error"] is not None
    assert result["error"].code == "EMPLOYEE_ACCESS_DENIED"


def test_session_access_denied_on_employee_switch():
    state = run_agent_step(bootstrap_state(employee_id="EMP-001", user_message=_message()))
    switched = resume_agent_step(state, employee_id="EMP-002")

    assert switched["current_step"] == "end"
    assert switched["error"] is not None
    assert switched["error"].code == "SESSION_ACCESS_DENIED"


def test_invalid_selected_option_is_rejected():
    state = run_agent_step(bootstrap_state(employee_id="EMP-001", user_message=_message()))
    rejected = resume_agent_step(
        state,
        selected_option_id="OPT-NOT-EXISTS",
        confirmation_status=None,
        error=None,
    )

    assert rejected["current_step"] == "end"
    assert rejected["error"] is not None
    assert rejected["error"].code == "OPTION_NOT_ALLOWED"


def test_audit_events_created_for_confirmed_write():
    state = run_agent_step(bootstrap_state(employee_id="EMP-001", user_message=_message()))
    selected_option_id = state["ranked_options"][0].option_id
    confirmation_state = resume_agent_step(
        state,
        selected_option_id=selected_option_id,
        confirmation_status="confirmed",
        error=None,
    )

    assert confirmation_state["approval_request_id"] is not None
    event_types = {item["event_type"] for item in confirmation_state.get("audit_events", [])}
    assert "confirmation_decision" in event_types
    assert "approval_request_created" in event_types


def test_confirmed_write_blocked_when_policy_fails(monkeypatch):
    from src.domain.models import PolicyCheckResult

    monkeypatch.setattr(
        "src.agent.nodes.evaluate_option",
        lambda option, refs, profile: [
            PolicyCheckResult(rule_id="TRANSPORT_CLASS_LIMIT", verdict="fail", message="blocked", citations=[])
        ],
    )
    state = run_agent_step(bootstrap_state(employee_id="EMP-001", user_message=_message()))
    assert state["current_step"] == "propose_constraint_change"
    assert state["approval_request_id"] is None


def test_sanitize_for_logging_redacts_sensitive_data():
    payload = {
        "message": "api_key=abc123 token:xyz secret=qwerty",
        "password": "top-secret",
        "nested": {"authorization": "Bearer test"},
    }

    sanitized = sanitize_for_logging(payload)
    assert "***REDACTED***" in sanitized["message"]
    assert sanitized["password"] == "***REDACTED***"
    assert sanitized["nested"]["authorization"] == "***REDACTED***"
