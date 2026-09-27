from src.agent.service import bootstrap_state, run_agent_step


def test_oversized_input_is_blocked_before_graph_run():
    message = "x" * 5001
    state = bootstrap_state(employee_id="EMP-001", user_message=message)

    result = run_agent_step(state)

    assert result["current_step"] == "end"
    assert result["error"] is not None
    assert result["error"].code == "INPUT_TOO_LARGE"
    assert result["approval_request_id"] is None
