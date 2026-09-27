from __future__ import annotations

from functools import lru_cache
from time import perf_counter
from uuid import uuid4

from src.agent.graph import build_graph
from src.agent.security import is_prompt_injection_attempt
from src.agent.state import TravelAgentState
from src.config import get_backend_mode, is_demo_fallback_mode, settings
from src.domain.models import AgentError
from src.observability.tracing import sanitize_for_logging, traced_span, traced_workflow

try:
    from langgraph.types import Command
except Exception:  # pragma: no cover - compatibility fallback
    Command = None

ALLOWED_TOOL_CALLS = {
    "get_employee_profile",
    "search_travel_options",
    "retrieve_travel_policy",
    "check_budget",
    "create_approval_request",
}


@lru_cache(maxsize=1)
def get_compiled_graph():
    return build_graph()


def bootstrap_state(employee_id: str, user_message: str) -> TravelAgentState:
    session_id = str(uuid4())
    backend_mode = get_backend_mode()
    backend_status = "degraded" if is_demo_fallback_mode() else "healthy"
    degraded_components = ["fallback_mode_enabled"] if is_demo_fallback_mode() else []
    return TravelAgentState(
        session_id=session_id,
        thread_id=session_id,
        employee_id=employee_id,
        last_user_message=user_message,
        trip_request=None,
        missing_fields=[],
        clarification_question=None,
        employee_profile=None,
        travel_options=[],
        retrieved_policy_chunks=[],
        assembled_policy_context="",
        policy_results=[],
        budget_results=[],
        ranked_options=[],
        selected_option_id=None,
        confirmation_status=None,
        approval_request_id=None,
        tool_calls=[],
        tool_events=[],
        audit_events=[],
        llm_usage={
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        },
        embedding_usage={
            "calls": 0,
            "input_tokens": 0,
            "estimated_cost_usd": 0.0,
            "latency_ms_total": 0,
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
        },
        backend_status=backend_status,
        degraded_components=degraded_components,
        node_transitions=0,
        rejected_options=[],
        travel_search_outcome=None,
        travel_search_error_code=None,
        recovery_action=None,
        recovery_advice=None,
        route_transitions=[],
        prompt_injection_detected=False,
        current_step="parse_request",
        iteration_count=0,
        trace_id=uuid4().hex,
        error=None,
    )


def _attach_error(state: TravelAgentState, code: str, message: str, *, retryable: bool = False) -> None:
    state["error"] = AgentError(code=code, message=message, retryable=retryable)
    state["current_step"] = "end"


def _apply_preflight_guardrails(state: TravelAgentState) -> TravelAgentState | None:
    message = state.get("last_user_message", "")
    if is_prompt_injection_attempt(message):
        state["prompt_injection_detected"] = True
        state.setdefault("audit_events", []).append(
            {
                "event": "prompt_injection_rejected",
                "employee_id": state.get("employee_id"),
            }
        )

    if len(message) > settings.max_user_chars:
        _attach_error(
            state,
            "INPUT_TOO_LARGE",
            f"Input is too large (>{settings.max_user_chars} chars).",
            retryable=True,
        )
        return state

    if state.get("node_transitions", 0) >= settings.max_agent_steps:
        _attach_error(
            state,
            "STEP_LIMIT_EXCEEDED",
            f"Agent step limit reached ({settings.max_agent_steps}).",
            retryable=True,
        )
        return state

    return None


def _validate_post_run_guardrails(state: TravelAgentState) -> None:
    if state.get("node_transitions", 0) > settings.max_agent_steps:
        _attach_error(
            state,
            "STEP_LIMIT_EXCEEDED",
            f"Agent step limit exceeded ({settings.max_agent_steps}).",
            retryable=True,
        )
        return

    disallowed_tools = sorted(set(state.get("tool_calls", [])) - ALLOWED_TOOL_CALLS)
    if disallowed_tools:
        _attach_error(
            state,
            "TOOL_NOT_ALLOWED",
            f"Disallowed tool calls detected: {', '.join(disallowed_tools)}.",
            retryable=False,
        )


def _enforce_session_access(state: TravelAgentState, updates: dict) -> TravelAgentState | None:
    requested_employee_id = updates.get("employee_id")
    if requested_employee_id is None:
        return None

    current_employee_id = state.get("employee_id")
    if current_employee_id and requested_employee_id != current_employee_id:
        _attach_error(
            state,
            "SESSION_ACCESS_DENIED",
            "Session belongs to a different employee.",
            retryable=False,
        )
        return state
    return None


def _graph_config(state: TravelAgentState) -> dict:
    thread_id = state.get("thread_id") or state.get("session_id")
    return {"configurable": {"thread_id": thread_id}}


def _normalize_interrupt_step(state: TravelAgentState) -> TravelAgentState:
    interrupts = state.get("__interrupt__", [])
    if not interrupts:
        return state

    first = interrupts[0]
    value = getattr(first, "value", None)
    if not isinstance(value, dict):
        return state

    interrupt_type = value.get("type")
    if interrupt_type == "clarification":
        state["current_step"] = "ask_clarification"
    elif interrupt_type == "selection":
        state["current_step"] = "present_recommendations"
    elif interrupt_type == "confirmation":
        state["current_step"] = "request_confirmation"
    elif interrupt_type == "recovery":
        state["current_step"] = "propose_constraint_change"

    return state


def run_agent_step(state: TravelAgentState) -> TravelAgentState:
    blocked_state = _apply_preflight_guardrails(state)
    if blocked_state is not None:
        return blocked_state

    graph = get_compiled_graph()
    started_at = perf_counter()
    invoke_config = _graph_config(state)
    with traced_workflow(
        name="travelcopilot.agent.run",
        trace_id=state.get("trace_id"),
        session_id=state.get("session_id"),
        user_id=state.get("employee_id"),
        input_data={
            "employee_id": state.get("employee_id"),
            "last_user_message": sanitize_for_logging(state.get("last_user_message")),
            "current_step": state.get("current_step"),
        },
    ) as trace:
        with traced_span(
            "travelcopilot.langgraph.invoke",
            trace_id=trace.trace_id,
            metadata={"component": "langgraph"},
        ):
            try:
                result = graph.invoke(state, config=invoke_config)
            except Exception as exc:
                result = dict(state)
                _attach_error(
                    result,
                    "AGENT_RUNTIME_ERROR",
                    f"Runtime failure during graph execution: {type(exc).__name__}",
                    retryable=True,
                )

        result = _normalize_interrupt_step(result)
        _validate_post_run_guardrails(result)
        latency_ms = int((perf_counter() - started_at) * 1000)
        result["trace_id"] = trace.trace_id or result.get("trace_id")
        trace.update(
            output_data={
                "current_step": result.get("current_step"),
                "missing_fields": result.get("missing_fields", []),
                "tool_calls": result.get("tool_calls", []),
                "tool_events": sanitize_for_logging(result.get("tool_events", [])),
                "audit_events": sanitize_for_logging(result.get("audit_events", [])),
                "llm_usage": result.get("llm_usage", {}),
                "embedding_usage": result.get("embedding_usage", {}),
                "backend_status": result.get("backend_status"),
                "degraded_components": result.get("degraded_components", []),
                "travel_search_outcome": result.get("travel_search_outcome"),
                "recovery_action": result.get("recovery_action"),
                "node_transitions": result.get("node_transitions", 0),
                "route_transitions": sanitize_for_logging(result.get("route_transitions", [])),
                "approval_request_id": result.get("approval_request_id"),
                "error": sanitize_for_logging(result.get("error").model_dump()) if result.get("error") else None,
            },
            metadata={"latency_ms": latency_ms},
        )
        return result


def resume_agent_step(state: TravelAgentState, **updates) -> TravelAgentState:
    access_denied = _enforce_session_access(state, updates)
    if access_denied is not None:
        return access_denied

    next_state = dict(state)
    next_state.update(updates)
    next_state.setdefault("thread_id", next_state.get("session_id"))

    blocked_state = _apply_preflight_guardrails(next_state)
    if blocked_state is not None:
        return blocked_state

    graph = get_compiled_graph()
    invoke_config = _graph_config(next_state)
    started_at = perf_counter()
    with traced_workflow(
        name="travelcopilot.agent.resume",
        trace_id=next_state.get("trace_id"),
        session_id=next_state.get("session_id"),
        user_id=next_state.get("employee_id"),
        input_data={
            "employee_id": next_state.get("employee_id"),
            "current_step": next_state.get("current_step"),
            "updates": sanitize_for_logging(sorted(updates.keys())),
        },
    ) as trace:
        with traced_span(
            "travelcopilot.langgraph.resume",
            trace_id=trace.trace_id,
            metadata={"component": "langgraph"},
        ):
            try:
                if Command is None:
                    raise RuntimeError("LangGraph Command API unavailable")
                result = graph.invoke(Command(resume=updates), config=invoke_config)
            except Exception:
                # Fallback path for non-interrupt states or compatibility issues.
                result = run_agent_step(next_state)

        result = _normalize_interrupt_step(result)
        _validate_post_run_guardrails(result)
        latency_ms = int((perf_counter() - started_at) * 1000)
        result["trace_id"] = trace.trace_id or result.get("trace_id")
        trace.update(
            output_data={
                "current_step": result.get("current_step"),
                "missing_fields": result.get("missing_fields", []),
                "tool_calls": result.get("tool_calls", []),
                "tool_events": sanitize_for_logging(result.get("tool_events", [])),
                "audit_events": sanitize_for_logging(result.get("audit_events", [])),
                "llm_usage": result.get("llm_usage", {}),
                "embedding_usage": result.get("embedding_usage", {}),
                "backend_status": result.get("backend_status"),
                "degraded_components": result.get("degraded_components", []),
                "travel_search_outcome": result.get("travel_search_outcome"),
                "recovery_action": result.get("recovery_action"),
                "node_transitions": result.get("node_transitions", 0),
                "route_transitions": sanitize_for_logging(result.get("route_transitions", [])),
                "approval_request_id": result.get("approval_request_id"),
                "error": sanitize_for_logging(result.get("error").model_dump()) if result.get("error") else None,
            },
            metadata={"latency_ms": latency_ms},
        )
        return result
