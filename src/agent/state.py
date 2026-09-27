from __future__ import annotations

from typing import Literal, TypedDict

from src.domain.models import (
    AgentError,
    BudgetCheckResult,
    EmployeeProfile,
    PolicyCheckResult,
    PolicyChunkRef,
    Recommendation,
    TravelOption,
    TripRequest,
)


AgentStep = Literal[
    "parse_request",
    "validate_request",
    "ask_clarification",
    "get_employee_profile",
    "search_travel_options",
    "retrieve_policy",
    "check_policy",
    "check_budget",
    "rank_options",
    "present_recommendations",
    "request_confirmation",
    "propose_constraint_change",
    "create_approval_request",
    "show_result",
    "end",
]


class TravelAgentState(TypedDict, total=False):
    session_id: str
    thread_id: str
    employee_id: str

    last_user_message: str
    trip_request: TripRequest | None
    missing_fields: list[str]
    clarification_question: str | None

    employee_profile: EmployeeProfile | None
    travel_options: list[TravelOption]

    retrieved_policy_chunks: list[PolicyChunkRef]
    assembled_policy_context: str
    policy_results: list[PolicyCheckResult]
    budget_results: list[BudgetCheckResult]

    ranked_options: list[Recommendation]
    selected_option_id: str | None

    confirmation_status: str | None
    approval_request_id: str | None
    tool_calls: list[str]
    tool_events: list[dict]
    audit_events: list[dict]
    llm_usage: dict[str, float | int]
    embedding_usage: dict[str, float | int | str]
    backend_status: str
    degraded_components: list[str]
    node_transitions: int
    rejected_options: list[dict]
    travel_search_outcome: str | None
    travel_search_error_code: str | None
    recovery_action: str | None
    recovery_advice: str | None
    route_transitions: list[dict]
    prompt_injection_detected: bool

    current_step: AgentStep
    iteration_count: int
    trace_id: str | None
    error: AgentError | None
