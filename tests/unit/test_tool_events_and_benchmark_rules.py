from __future__ import annotations

from scripts.run_benchmark import _extract_path, _is_subsequence, _validate_tool_expectations
from src.agent.service import bootstrap_state, resume_agent_step, run_agent_step


def test_happy_path_generates_structured_tool_events():
    message = (
        "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
        "на встречу к 15:00. Вернуться на следующий день."
    )
    state = run_agent_step(bootstrap_state(employee_id="EMP-001", user_message=message))
    events = state.get("tool_events", [])
    assert events
    for event in events:
        assert event["tool_name"]
        assert event["correlation_id"]
        assert event["session_id"] == state["session_id"]
        assert event["status"] in {"ok", "error"}
        assert isinstance(event["latency_ms"], int)


def test_write_tool_event_contains_confirmation_status():
    message = (
        "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
        "на встречу к 15:00. Вернуться на следующий день."
    )
    state = run_agent_step(bootstrap_state(employee_id="EMP-001", user_message=message))
    selected_option_id = state["ranked_options"][0].option_id
    confirmed = resume_agent_step(
        state,
        selected_option_id=selected_option_id,
        confirmation_status="confirmed",
        error=None,
    )
    write_event = next(event for event in confirmed.get("tool_events", []) if event["tool_name"] == "create_approval_request")
    assert _extract_path(write_event, "request.confirmation_status") == "confirmed"


def test_tool_order_helpers():
    assert _is_subsequence(["a", "c"], ["a", "b", "c"])
    assert not _is_subsequence(["a", "c", "b"], ["a", "b", "c"])
    payload = {"request": {"trip_request_payload": {"origin_city": "Казань"}}}
    assert _extract_path(payload, "request.trip_request_payload.origin_city") == "Казань"


def test_runner_detects_wrong_tool_order():
    errors = _validate_tool_expectations(
        expected_tools={"order": ["a", "b", "c"]},
        tool_calls=["a", "c", "b"],
        tool_events=[],
    )
    assert any("tool order mismatch" in item for item in errors)


def test_runner_detects_wrong_tool_count():
    errors = _validate_tool_expectations(
        expected_tools={"count": {"search_travel_options": 1}},
        tool_calls=["search_travel_options", "search_travel_options"],
        tool_events=[],
    )
    assert any("tool count mismatch" in item for item in errors)


def test_runner_detects_wrong_tool_argument():
    errors = _validate_tool_expectations(
        expected_tools={
            "key_arguments": [
                {"tool": "search_travel_options", "path": "trip_request_payload.origin_city", "equals": "Казань"}
            ]
        },
        tool_calls=["search_travel_options"],
        tool_events=[
            {
                "tool_name": "search_travel_options",
                "request": {"trip_request_payload": {"origin_city": "Москва"}},
            }
        ],
    )
    assert any("key argument mismatch" in item for item in errors)


def test_runner_detects_write_before_confirmation():
    errors = _validate_tool_expectations(
        expected_tools={"write_after_confirmation": True},
        tool_calls=["create_approval_request"],
        tool_events=[
            {
                "tool_name": "create_approval_request",
                "request": {"confirmation_status": "rejected"},
            }
        ],
    )
    assert "write tool executed without confirmed status" in errors
