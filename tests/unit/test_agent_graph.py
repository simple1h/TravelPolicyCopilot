from datetime import UTC, date, datetime

from src.agent.nodes import parse_trip_request_from_message
from src.agent.service import bootstrap_state, get_compiled_graph, resume_agent_step, run_agent_step
from src.domain.models import TripRequest


def test_graph_compiles():
    graph = get_compiled_graph()
    assert graph is not None


def test_parse_trip_request_extracts_destination():
    trip = parse_trip_request_from_message("Нужна командировка в СПБ")
    assert trip.destination_city == "Санкт-Петербург"
    assert trip.origin_city is None


def test_run_agent_step_routes_to_clarification():
    state = bootstrap_state(
        employee_id="EMP-001",
        user_message="Нужна командировка в СПБ",
    )
    result = run_agent_step(state)

    assert result["current_step"] == "ask_clarification"
    assert "origin_city" in result["missing_fields"]
    assert result["trip_request"] is not None
    assert result["trip_request"].destination_city == "Санкт-Петербург"


def test_happy_path_reaches_present_recommendations():
    message = (
        "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
        "на встречу к 15:00. Вернуться на следующий день."
    )
    state = bootstrap_state(employee_id="EMP-001", user_message=message)
    result = run_agent_step(state)

    assert result["current_step"] == "present_recommendations"
    assert result["employee_profile"] is not None
    assert len(result["travel_options"]) >= 1
    assert len(result["retrieved_policy_chunks"]) >= 1
    assert len(result["policy_results"]) >= 1
    assert len(result["budget_results"]) >= 1
    assert len(result["ranked_options"]) >= 1
    assert result["ranked_options"][0].is_recommended is True


def test_confirmation_flow_confirmed_creates_request():
    message = (
        "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
        "на встречу к 15:00. Вернуться на следующий день."
    )
    state = run_agent_step(bootstrap_state(employee_id="EMP-001", user_message=message))
    selected_option_id = state["ranked_options"][0].option_id

    confirmation_state = resume_agent_step(
        state,
        selected_option_id=selected_option_id,
        confirmation_status=None,
    )
    assert confirmation_state["current_step"] == "request_confirmation"

    created_state = resume_agent_step(confirmation_state, confirmation_status="confirmed")
    assert created_state["current_step"] == "show_result"
    assert created_state["approval_request_id"] is not None


def test_confirmation_flow_rejected_has_no_write():
    message = (
        "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
        "на встречу к 15:00. Вернуться на следующий день."
    )
    state = run_agent_step(bootstrap_state(employee_id="EMP-001", user_message=message))
    selected_option_id = state["ranked_options"][0].option_id

    confirmation_state = resume_agent_step(
        state,
        selected_option_id=selected_option_id,
        confirmation_status="rejected",
    )
    assert confirmation_state["current_step"] == "request_confirmation"
    assert confirmation_state.get("approval_request_id") is None


def test_confirmation_flow_change_selection_returns_to_recommendations():
    message = (
        "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
        "на встречу к 15:00. Вернуться на следующий день."
    )
    state = run_agent_step(bootstrap_state(employee_id="EMP-001", user_message=message))
    selected_option_id = state["ranked_options"][0].option_id

    confirmation_state = resume_agent_step(
        state,
        selected_option_id=selected_option_id,
        confirmation_status=None,
    )
    changed_state = resume_agent_step(
        confirmation_state,
        confirmation_status="change_selection",
        selected_option_id=None,
    )

    assert changed_state["current_step"] == "present_recommendations"
    assert changed_state.get("approval_request_id") is None


def test_create_request_is_idempotent_for_same_context():
    message = (
        "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
        "на встречу к 15:00. Вернуться на следующий день."
    )
    state = run_agent_step(bootstrap_state(employee_id="EMP-001", user_message=message))
    selected_option_id = state["ranked_options"][0].option_id

    first = resume_agent_step(
        state,
        selected_option_id=selected_option_id,
        confirmation_status="confirmed",
    )
    second = resume_agent_step(
        state,
        selected_option_id=selected_option_id,
        confirmation_status="confirmed",
    )

    assert first["approval_request_id"] == second["approval_request_id"]


def test_resume_does_not_repeat_read_tools():
    message = (
        "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
        "на встречу к 15:00. Вернуться на следующий день."
    )
    state = run_agent_step(bootstrap_state(employee_id="EMP-001", user_message=message))
    initial_calls = list(state.get("tool_calls", []))
    assert "get_employee_profile" in initial_calls
    assert "search_travel_options" in initial_calls
    assert "retrieve_travel_policy" in initial_calls
    assert "check_budget" in initial_calls
    initial_budget_call_count = initial_calls.count("check_budget")
    assert initial_budget_call_count >= 1

    selected_option_id = state["ranked_options"][0].option_id
    resumed = resume_agent_step(
        state,
        selected_option_id=selected_option_id,
        confirmation_status=None,
    )
    assert resumed["current_step"] == "request_confirmation"
    assert resumed.get("tool_calls", []) == initial_calls


def test_meeting_deadline_is_applied_to_recommendations():
    state = run_agent_step(
        bootstrap_state(
            employee_id="EMP-001",
            user_message=(
                "Мне нужно 18 сентября попасть из Казани в Санкт-Петербург "
                "на встречу к 10:00. Вернуться на следующий день."
            ),
        )
    )
    assert state["current_step"] == "present_recommendations"
    assert state["trip_request"] is not None
    meeting_dt = state["trip_request"].meeting_datetime
    assert meeting_dt is not None
    meeting_dt_utc = meeting_dt.replace(tzinfo=UTC) if meeting_dt.tzinfo is None else meeting_dt.astimezone(UTC)
    for option in state["travel_options"]:
        assert option.outbound_arrival.astimezone(UTC) <= meeting_dt_utc


def test_hotel_not_required_removes_hotel_fields(monkeypatch):
    trip = TripRequest(
        origin_city="Казань",
        destination_city="Санкт-Петербург",
        departure_date=date(2026, 9, 18),
        return_date=date(2026, 9, 19),
        meeting_datetime=datetime(2026, 9, 18, 15, 0, tzinfo=UTC),
        hotel_required=False,
    )
    monkeypatch.setattr("src.agent.nodes.parse_trip_request_with_llm", lambda *args, **kwargs: (trip, {"calls": 0}))
    state = run_agent_step(bootstrap_state(employee_id="EMP-001", user_message="тест"))
    assert state["current_step"] == "present_recommendations"
    assert state["travel_options"]
    for option in state["travel_options"]:
        assert option.hotel_name is None
        assert option.hotel_nightly_price is None


def test_no_options_go_to_recovery_branch():
    state = run_agent_step(
        bootstrap_state(
            employee_id="EMP-003",
            user_message=(
                "Мне нужно 18 сентября утром попасть из Казани в Москву и вернуться на следующий день. "
                "[scenario=empty]"
            ),
        )
    )
    assert state["current_step"] == "propose_constraint_change"
    assert state["travel_search_outcome"] == "options_not_found"


def test_all_policy_fail_routes_to_recovery(monkeypatch):
    from src.domain.models import PolicyCheckResult

    monkeypatch.setattr(
        "src.agent.nodes.evaluate_option",
        lambda option, refs, profile: [
            PolicyCheckResult(rule_id="TRANSPORT_CLASS_LIMIT", verdict="fail", message="blocked", citations=[])
        ],
    )
    state = run_agent_step(
        bootstrap_state(
            employee_id="EMP-001",
            user_message=(
                "Мне нужно 18 сентября утром попасть из Казани в Санкт-Петербург "
                "на встречу к 15:00. Вернуться на следующий день."
            ),
        )
    )
    assert state["current_step"] == "propose_constraint_change"
    assert state["travel_search_outcome"] == "options_not_found"
