from src.agent.service import _validate_post_run_guardrails


def test_disallowed_tool_marks_error_and_stops_state():
    state = {
        "current_step": "present_recommendations",
        "tool_calls": ["get_employee_profile", "drop_database"],
    }

    _validate_post_run_guardrails(state)

    assert state["current_step"] == "end"
    assert state["error"].code == "TOOL_NOT_ALLOWED"
