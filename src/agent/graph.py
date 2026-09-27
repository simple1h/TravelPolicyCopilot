from __future__ import annotations

from src.agent.nodes import (
    ask_clarification_node,
    check_budget_node,
    check_policy_node,
    create_approval_request_node,
    get_employee_profile_node,
    parse_request_node,
    present_recommendations_node,
    propose_constraint_change_node,
    rank_options_node,
    request_confirmation_node,
    retrieve_policy_node,
    search_travel_options_node,
    show_result_node,
    validate_request_node,
)
from src.agent.state import TravelAgentState
from src.observability.tracing import traced_span


def _record_route(state: TravelAgentState, *, source: str, label: str, target: str) -> None:
    transitions = state.setdefault("route_transitions", [])
    transitions.append({"source": source, "label": label, "target": target})
    with traced_span(
        "travelcopilot.agent.route",
        trace_id=state.get("trace_id"),
        input_data={"source": source, "label": label, "target": target},
        metadata={"component": "agent_route"},
    ):
        pass


def route_after_error_check(state: TravelAgentState) -> str:
    source = str(state.get("current_step") or "unknown")
    if state.get("error") is not None or state.get("current_step") == "end":
        _record_route(state, source=source, label="error", target="end")
        return "error"
    ok_target = "parse_request" if source == "ask_clarification" else "search_travel_options"
    _record_route(state, source=source, label="ok", target=ok_target)
    return "ok"


def route_after_validate(state: TravelAgentState) -> str:
    if state.get("error") is not None or state.get("current_step") == "end":
        _record_route(state, source="validate_request", label="error", target="end")
        return "error"
    if state.get("missing_fields"):
        _record_route(state, source="validate_request", label="clarification_required", target="ask_clarification")
        return "clarification_required"
    _record_route(state, source="validate_request", label="request_complete", target="get_employee_profile")
    return "request_complete"


def route_after_search_options(state: TravelAgentState) -> str:
    if state.get("error") is not None or state.get("current_step") == "end":
        _record_route(state, source="search_travel_options", label="error", target="end")
        return "error"
    outcome = state.get("travel_search_outcome")
    if outcome in {"options_found", "options_not_found", "invalid_supplier_request", "retryable_external_error"}:
        _record_route(state, source="search_travel_options", label=str(outcome), target="propose_constraint_change" if outcome != "options_found" else "retrieve_policy")
        return outcome
    if state.get("travel_options"):
        _record_route(state, source="search_travel_options", label="options_found", target="retrieve_policy")
        return "options_found"
    _record_route(state, source="search_travel_options", label="options_not_found", target="propose_constraint_change")
    return "options_not_found"


def route_after_recovery_prompt(state: TravelAgentState) -> str:
    if state.get("error") is not None:
        _record_route(state, source="propose_constraint_change", label="error", target="end")
        return "error"
    action = state.get("recovery_action")
    if action == "update_constraints":
        _record_route(state, source="propose_constraint_change", label="update_constraints", target="validate_request")
        return "update_constraints"
    if action == "retry_search":
        _record_route(state, source="propose_constraint_change", label="retry_search", target="search_travel_options")
        return "retry_search"
    if action == "cancel":
        _record_route(state, source="propose_constraint_change", label="cancel", target="end")
        return "cancel"
    _record_route(state, source="propose_constraint_change", label="awaiting_action", target="end")
    return "awaiting_action"


def route_after_present_recommendations(state: TravelAgentState) -> str:
    if state.get("error") is not None or state.get("current_step") == "end":
        _record_route(state, source="present_recommendations", label="error", target="end")
        return "error"
    ranked = state.get("ranked_options", [])
    if ranked and not any(getattr(item, "is_recommended", False) for item in ranked):
        _record_route(state, source="present_recommendations", label="no_recommendation", target="propose_constraint_change")
        return "no_recommendation"
    if state.get("selected_option_id"):
        _record_route(state, source="present_recommendations", label="selection_made", target="request_confirmation")
        return "selection_made"
    _record_route(state, source="present_recommendations", label="awaiting_selection", target="end")
    return "awaiting_selection"


def route_after_confirmation(state: TravelAgentState) -> str:
    if state.get("error") is not None or state.get("current_step") == "end":
        _record_route(state, source="request_confirmation", label="error", target="end")
        return "error"
    status = state.get("confirmation_status")
    if status is None:
        _record_route(state, source="request_confirmation", label="awaiting_confirmation", target="end")
        return "awaiting_confirmation"
    if status == "confirmed":
        _record_route(state, source="request_confirmation", label="confirmed", target="create_approval_request")
        return "confirmed"
    if status == "change_selection":
        _record_route(state, source="request_confirmation", label="change_selection", target="present_recommendations")
        return "change_selection"
    _record_route(state, source="request_confirmation", label="rejected", target="end")
    return "rejected"


def build_graph():
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.checkpoint.memory import InMemorySaver
    except Exception as exc:  # pragma: no cover - import guard for minimal skeleton
        raise RuntimeError(
            "LangGraph is not installed yet. Install dependencies to compile the graph."
        ) from exc

    graph = StateGraph(TravelAgentState)
    graph.add_node("parse_request", parse_request_node)
    graph.add_node("validate_request", validate_request_node)
    graph.add_node("ask_clarification", ask_clarification_node)
    graph.add_node("get_employee_profile", get_employee_profile_node)
    graph.add_node("search_travel_options", search_travel_options_node)
    graph.add_node("propose_constraint_change", propose_constraint_change_node)
    graph.add_node("retrieve_policy", retrieve_policy_node)
    graph.add_node("check_policy", check_policy_node)
    graph.add_node("check_budget", check_budget_node)
    graph.add_node("rank_options", rank_options_node)
    graph.add_node("present_recommendations", present_recommendations_node)
    graph.add_node("request_confirmation", request_confirmation_node)
    graph.add_node("create_approval_request", create_approval_request_node)
    graph.add_node("show_result", show_result_node)

    graph.add_edge(START, "parse_request")
    graph.add_edge("parse_request", "validate_request")

    graph.add_conditional_edges(
        "validate_request",
        route_after_validate,
        {
            "error": END,
            "clarification_required": "ask_clarification",
            "request_complete": "get_employee_profile",
        },
    )
    graph.add_conditional_edges(
        "ask_clarification",
        route_after_error_check,
        {
            "error": END,
            "ok": "parse_request",
        },
    )

    graph.add_conditional_edges(
        "get_employee_profile",
        route_after_error_check,
        {
            "error": END,
            "ok": "search_travel_options",
        },
    )
    graph.add_conditional_edges(
        "search_travel_options",
        route_after_search_options,
        {
            "error": END,
            "options_found": "retrieve_policy",
            "options_not_found": "propose_constraint_change",
            "invalid_supplier_request": "propose_constraint_change",
            "retryable_external_error": "propose_constraint_change",
        },
    )

    graph.add_conditional_edges(
        "propose_constraint_change",
        route_after_recovery_prompt,
        {
            "error": END,
            "awaiting_action": END,
            "update_constraints": "validate_request",
            "retry_search": "search_travel_options",
            "cancel": END,
        },
    )

    graph.add_edge("retrieve_policy", "check_policy")
    graph.add_edge("check_policy", "check_budget")
    graph.add_edge("check_budget", "rank_options")
    graph.add_edge("rank_options", "present_recommendations")
    graph.add_conditional_edges(
        "present_recommendations",
        route_after_present_recommendations,
        {
            "error": END,
            "no_recommendation": "propose_constraint_change",
            "awaiting_selection": END,
            "selection_made": "request_confirmation",
        },
    )

    graph.add_conditional_edges(
        "request_confirmation",
        route_after_confirmation,
        {
            "error": END,
            "awaiting_confirmation": END,
            "confirmed": "create_approval_request",
            "change_selection": "present_recommendations",
            "rejected": END,
        },
    )

    graph.add_edge("create_approval_request", "show_result")
    graph.add_edge("show_result", END)

    return graph.compile(checkpointer=InMemorySaver())
