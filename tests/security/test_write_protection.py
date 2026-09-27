from src.agent.service import bootstrap_state, resume_agent_step, run_agent_step


def _get_recommendation_state() -> dict:
    message = (
        "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
        "на встречу к 15:00. Вернуться на следующий день."
    )
    return run_agent_step(bootstrap_state(employee_id="EMP-001", user_message=message))


def test_rejected_confirmation_does_not_create_request():
    state = _get_recommendation_state()
    selected_option_id = state["ranked_options"][0].option_id

    confirmation_state = resume_agent_step(
        state,
        selected_option_id=selected_option_id,
        confirmation_status=None,
        error=None,
    )
    result = resume_agent_step(confirmation_state, confirmation_status="rejected", error=None)

    assert result["current_step"] == "request_confirmation"
    assert result.get("approval_request_id") is None
    assert "create_approval_request" not in result.get("tool_calls", [])


def test_confirmed_write_is_idempotent():
    state = _get_recommendation_state()
    selected_option_id = state["ranked_options"][0].option_id

    first = resume_agent_step(
        state,
        selected_option_id=selected_option_id,
        confirmation_status="confirmed",
        error=None,
    )
    second = resume_agent_step(
        state,
        selected_option_id=selected_option_id,
        confirmation_status="confirmed",
        error=None,
    )

    assert first["approval_request_id"] is not None
    assert first["approval_request_id"] == second["approval_request_id"]
