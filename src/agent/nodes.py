from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo
from uuid import uuid4, uuid5, NAMESPACE_URL

from src.agent.state import TravelAgentState
from src.config import settings
from src.domain.models import ApprovalRequest, AgentError, TripRequest
from src.llm.gateway import LLMGatewayError, get_default_llm_gateway
from src.observability.tracing import traced_span
from src.policy.engine import evaluate_option
from src.policy.scoring import rank_travel_options
from src.rag.context import assemble_policy_context
from src.tools.contracts import (
    CheckBudgetRequest,
    CreateApprovalRequestRequest,
    GetEmployeeProfileRequest,
    RetrieveTravelPolicyRequest,
    SearchTravelOptionsRequest,
    ToolContext,
    ToolEvent,
)
from src.tools.service import (
    ToolExecutionError,
    execute_check_budget,
    execute_create_approval_request,
    execute_get_employee_profile,
    execute_retrieve_travel_policy,
    execute_search_travel_options,
)

try:
    from langgraph.types import interrupt
except Exception:  # pragma: no cover - graceful fallback when langgraph changes
    interrupt = None

_DESTINATION_ALIASES: dict[str, str] = {
    "спб": "Санкт-Петербург",
    "санкт-петербург": "Санкт-Петербург",
    "питер": "Санкт-Петербург",
    "москв": "Москва",
    "казан": "Казань",
}

_MONTH_ALIASES: dict[str, int] = {
    "январ": 1,
    "феврал": 2,
    "март": 3,
    "апрел": 4,
    "ма": 5,
    "июн": 6,
    "июл": 7,
    "август": 8,
    "сентябр": 9,
    "октябр": 10,
    "ноябр": 11,
    "декабр": 12,
}


def _extract_scenario_mode(message: str) -> str | None:
    match = re.search(r"\[scenario=(normal|empty|error_4xx|error_5xx|timeout)\]", message.lower())
    if not match:
        return None
    return match.group(1)


def _record_tool_call(state: TravelAgentState, tool_name: str) -> None:
    calls = state.setdefault("tool_calls", [])
    calls.append(tool_name)


def _record_tool_event(state: TravelAgentState, event: ToolEvent) -> None:
    events = state.setdefault("tool_events", [])
    events.append(event.model_dump(mode="json"))
    _record_tool_call(state, event.tool_name)


def _tool_context(state: TravelAgentState) -> ToolContext:
    session_id = str(state.get("session_id") or "unknown-session")
    return ToolContext(
        correlation_id=uuid4().hex,
        session_id=session_id,
        trace_id=state.get("trace_id"),
        timeout_seconds=settings.request_timeout_seconds,
    )


def _record_audit_event(state: TravelAgentState, event_type: str, **details: object) -> None:
    events = state.setdefault("audit_events", [])
    sanitized: dict[str, object] = {}
    for key, value in details.items():
        key_lower = key.lower()
        if any(secret_key in key_lower for secret_key in ("password", "token", "secret", "api_key")):
            sanitized[key] = "***REDACTED***"
        else:
            sanitized[key] = value
    events.append(
        {
            "event_type": event_type,
            "timestamp": datetime.now(UTC).isoformat(),
            **sanitized,
        }
    )


def _mark_degraded_component(state: TravelAgentState, component: str) -> None:
    degraded = state.setdefault("degraded_components", [])
    if component not in degraded:
        degraded.append(component)
    state["backend_status"] = "degraded"


def _reject_with_error(state: TravelAgentState, code: str, message: str, *, retryable: bool = False) -> TravelAgentState:
    state["error"] = AgentError(code=code, message=message, retryable=retryable)
    state["current_step"] = "end"
    return state


def _enter_step(state: TravelAgentState, step_name: str) -> None:
    source = str(state.get("current_step") or "unknown")
    state["current_step"] = step_name
    state["node_transitions"] = int(state.get("node_transitions", 0)) + 1
    transitions = state.setdefault("route_transitions", [])
    transitions.append({"source": source, "target": step_name, "label": "node_enter"})
    with traced_span(
        "travelcopilot.agent.node",
        trace_id=state.get("trace_id"),
        input_data={"source": source, "target": step_name},
        metadata={"component": "agent_node", "node": step_name},
    ):
        pass


def _has_selected_option_in_recommendations(state: TravelAgentState) -> bool:
    selected_option_id = state.get("selected_option_id")
    if not selected_option_id:
        return False
    return any(item.option_id == selected_option_id for item in state.get("ranked_options", []))


def _hour_in_disallowed_windows(hour: int, windows: list[str]) -> bool:
    for window in windows:
        try:
            start_raw, end_raw = window.split("-", maxsplit=1)
            start_hour = int(start_raw.split(":", maxsplit=1)[0])
            end_hour = int(end_raw.split(":", maxsplit=1)[0])
        except Exception:
            continue
        if start_hour <= hour < end_hour:
            return True
    return False


def _to_utc_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _constraint_reasons_for_option(
    option,
    *,
    trip_request: TripRequest,
    profile,
) -> list[str]:
    reasons: list[str] = []
    meeting_deadline = _to_utc_datetime(trip_request.meeting_datetime)
    if meeting_deadline and option.outbound_arrival.astimezone(UTC) > meeting_deadline:
        reasons.append("meeting_deadline_missed")
    if trip_request.earliest_departure_hour is not None and option.outbound_departure.hour < trip_request.earliest_departure_hour:
        reasons.append("earliest_departure_violation")
    if trip_request.latest_arrival_hour is not None and option.outbound_arrival.hour > trip_request.latest_arrival_hour:
        reasons.append("latest_arrival_violation")
    if trip_request.max_segments is not None and option.segments > trip_request.max_segments:
        reasons.append("max_segments_violation")
    if trip_request.disallowed_time_windows and _hour_in_disallowed_windows(
        option.outbound_departure.hour,
        trip_request.disallowed_time_windows,
    ):
        reasons.append("disallowed_time_window_violation")
    if trip_request.hotel_required is False:
        if option.hotel_name is not None or option.hotel_nightly_price is not None:
            reasons.append("hotel_not_required_violation")
    if profile and "wheelchair_access" in profile.accessibility_requirements:
        if option.transport_type != "flight" or option.segments > 1 or option.outbound_departure.hour < 7:
            reasons.append("accessibility_violation")
    return reasons


def _normalize_city(raw: str) -> str:
    city = raw.strip().capitalize()
    if city.endswith("и") and len(city) > 3:
        return city[:-1] + "ь"
    return city


def parse_trip_request_from_message(message: str) -> TripRequest:
    trip = TripRequest(comment=message)
    lower = message.lower()

    for alias, city in _DESTINATION_ALIASES.items():
        if alias in lower and (" в " in lower or f" {alias}" in lower):
            trip.destination_city = city
            break

    origin_match = re.search(r"из\s+([а-яА-ЯёЁ\-]+)", message, re.IGNORECASE)
    if origin_match:
        trip.origin_city = _normalize_city(origin_match.group(1))

    date_match = re.search(
        r"(\d{1,2})\s+(январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*",
        lower,
    )
    if date_match:
        day = int(date_match.group(1))
        month_key = date_match.group(2)
        month = next(value for key, value in _MONTH_ALIASES.items() if month_key.startswith(key))
        trip.departure_date = date(date.today().year, month, day)

    if "следующ" in lower and trip.departure_date is not None:
        trip.return_date = trip.departure_date + timedelta(days=1)

    if "утром" in lower and trip.earliest_departure_hour is None:
        trip.earliest_departure_hour = 6

    meeting_time_match = re.search(r"к\s*(\d{1,2}):(\d{2})", lower)
    if meeting_time_match and trip.departure_date is not None:
        hour = int(meeting_time_match.group(1))
        minute = int(meeting_time_match.group(2))
        timezone_name = settings.app_timezone.strip() or "UTC"
        tz = UTC
        try:
            tz = ZoneInfo(timezone_name)
        except Exception:
            tz = UTC
        trip.meeting_datetime = datetime(
            year=trip.departure_date.year,
            month=trip.departure_date.month,
            day=trip.departure_date.day,
            hour=hour,
            minute=minute,
            tzinfo=tz,
        )
        trip.latest_arrival_hour = hour

    if "ноч" in lower and not trip.disallowed_time_windows:
        trip.disallowed_time_windows = ["00:00-06:00", "23:00-24:00"]

    scenario_mode = _extract_scenario_mode(message)
    if scenario_mode:
        trip.comment = f"{message} [scenario={scenario_mode}]"

    return trip


def _trip_request_json_instructions() -> str:
    return (
        "Return ONLY strict JSON object with keys: "
        "origin_city, destination_city, departure_date, return_date, "
        "meeting_datetime, earliest_departure_hour, latest_arrival_hour, max_segments, "
        "disallowed_time_windows, transport_preferences, "
        "hotel_required, comment. "
        "Dates must be ISO format YYYY-MM-DD or null. "
        "meeting_datetime must be ISO-8601 datetime or null. "
        "earliest_departure_hour/latest_arrival_hour/max_segments must be integer or null. "
        "Use null for unknown values. "
        "Do not add extra keys."
    )


def _now_context() -> tuple[datetime, str]:
    timezone_name = settings.app_timezone.strip() or "UTC"
    try:
        now_local = datetime.now(ZoneInfo(timezone_name))
    except Exception:
        timezone_name = "UTC"
        now_local = datetime.now(UTC)
    return now_local, timezone_name


def _normalize_trip_request_payload(payload: dict) -> dict:
    normalized = dict(payload)

    list_fields = ("disallowed_time_windows", "transport_preferences")
    for field in list_fields:
        if normalized.get(field) is None:
            normalized[field] = []

    if normalized.get("hotel_required") is None:
        normalized["hotel_required"] = True
    if normalized.get("earliest_departure_hour") is not None:
        normalized["earliest_departure_hour"] = int(normalized["earliest_departure_hour"])
    if normalized.get("latest_arrival_hour") is not None:
        normalized["latest_arrival_hour"] = int(normalized["latest_arrival_hour"])
    if normalized.get("max_segments") is not None:
        normalized["max_segments"] = int(normalized["max_segments"])

    for city_field in ("origin_city", "destination_city"):
        raw_city = normalized.get(city_field)
        if not isinstance(raw_city, str):
            continue
        lower_city = raw_city.strip().lower()
        canonical = None
        for alias, target in _DESTINATION_ALIASES.items():
            if alias in lower_city:
                canonical = target
                break
        if canonical is not None:
            normalized[city_field] = canonical

    return normalized


def parse_trip_request_with_llm(message: str, *, trace_id: str | None = None) -> tuple[TripRequest, dict[str, float | int]]:
    now_local, timezone_name = _now_context()
    gateway = get_default_llm_gateway()
    response = gateway.complete(
        system_prompt=(
            "You extract corporate travel request fields into structured JSON. "
            "Do not invent unknown values."
        ),
        user_prompt=(
            f"Current local datetime: {now_local.isoformat()} ({timezone_name}).\n"
            f"{_trip_request_json_instructions()}\n"
            f"User message:\n{message}"
        ),
        temperature=0.0,
        trace_id=trace_id,
    )
    payload = json.loads(response.content)
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object.")
    payload = _normalize_trip_request_payload(payload)
    trip_request = TripRequest.model_validate(payload)
    usage = {
        "calls": 1,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "estimated_cost_usd": response.usage.estimated_cost_usd,
    }
    return trip_request, usage


def _merge_llm_usage(state: TravelAgentState, usage: dict[str, float | int]) -> None:
    current = state.get("llm_usage") or {
        "calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    state["llm_usage"] = {
        "calls": int(current.get("calls", 0)) + int(usage.get("calls", 0)),
        "prompt_tokens": int(current.get("prompt_tokens", 0)) + int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(current.get("completion_tokens", 0))
        + int(usage.get("completion_tokens", 0)),
        "total_tokens": int(current.get("total_tokens", 0)) + int(usage.get("total_tokens", 0)),
        "estimated_cost_usd": round(
            float(current.get("estimated_cost_usd", 0.0)) + float(usage.get("estimated_cost_usd", 0.0)),
            8,
        ),
    }


def _merge_embedding_usage(state: TravelAgentState, usage: dict[str, object]) -> None:
    current = state.get("embedding_usage") or {
        "calls": 0,
        "input_tokens": 0,
        "estimated_cost_usd": 0.0,
        "latency_ms_total": 0,
        "provider": settings.embedding_provider,
        "model": settings.embedding_model,
    }
    state["embedding_usage"] = {
        "calls": int(current.get("calls", 0)) + 1,
        "input_tokens": int(current.get("input_tokens", 0)) + int(usage.get("input_tokens", 0)),
        "estimated_cost_usd": round(
            float(current.get("estimated_cost_usd", 0.0)) + float(usage.get("estimated_cost_usd", 0.0)),
            8,
        ),
        "latency_ms_total": int(current.get("latency_ms_total", 0)) + int(usage.get("latency_ms", 0)),
        "provider": str(usage.get("provider") or current.get("provider") or ""),
        "model": str(usage.get("model") or current.get("model") or ""),
        "mode": str(usage.get("mode") or current.get("mode") or ""),
    }


def _enrich_trip_request_with_fallback_parser(trip_request: TripRequest, message: str) -> TripRequest:
    fallback_trip = parse_trip_request_from_message(message)
    updates: dict[str, object] = {}
    current_year = date.today().year
    for field in (
        "origin_city",
        "destination_city",
        "departure_date",
        "return_date",
        "meeting_datetime",
        "earliest_departure_hour",
        "latest_arrival_hour",
        "max_segments",
    ):
        llm_value = getattr(trip_request, field)
        fallback_value = getattr(fallback_trip, field)
        if llm_value is None and fallback_value is not None:
            updates[field] = fallback_value
            continue
        if field in {"departure_date", "return_date"} and llm_value is not None and fallback_value is not None:
            if llm_value.year < current_year <= fallback_value.year:
                updates[field] = fallback_value
    if not trip_request.disallowed_time_windows and fallback_trip.disallowed_time_windows:
        updates["disallowed_time_windows"] = fallback_trip.disallowed_time_windows
    if not trip_request.transport_preferences and fallback_trip.transport_preferences:
        updates["transport_preferences"] = fallback_trip.transport_preferences
    if trip_request.comment is None and fallback_trip.comment is not None:
        updates["comment"] = fallback_trip.comment
    scenario_mode = _extract_scenario_mode(message)
    if scenario_mode:
        base_comment = str(updates.get("comment", trip_request.comment or message))
        if f"[scenario={scenario_mode}]" not in base_comment.lower():
            updates["comment"] = f"{base_comment} [scenario={scenario_mode}]".strip()
    if not updates:
        return trip_request
    return trip_request.model_copy(update=updates)


def _fallback_clarification_question(missing_fields: list[str]) -> str:
    if not missing_fields:
        return "Нужны уточнения по параметрам поездки."
    return "Нужны уточнения: " + ", ".join(missing_fields) + "."


def _extract_json_object(raw: str) -> dict[str, object]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("LLM payload is not an object.")
    return parsed


def _build_clarification_question_with_llm(
    *,
    missing_fields: list[str],
    trip_request: TripRequest | None,
    user_message: str,
    trace_id: str | None,
) -> tuple[str, dict[str, float | int]]:
    now_local, timezone_name = _now_context()
    gateway = get_default_llm_gateway()
    response = gateway.complete(
        system_prompt=(
            "You generate one concise Russian clarification question for a corporate travel assistant. "
            "Use only the provided missing fields and known trip data."
        ),
        user_prompt=(
            "Return strict JSON object with key 'question'.\n"
            f"Current local datetime: {now_local.isoformat()} ({timezone_name}).\n"
            f"Missing fields: {missing_fields}\n"
            f"Known trip fields: {trip_request.model_dump(mode='json') if trip_request else {}}\n"
            f"Original user message: {user_message}\n"
            "Question must start with 'Нужны уточнения:' and must not invent unavailable details."
        ),
        temperature=0.0,
        trace_id=trace_id,
        response_format={"type": "json_object"},
    )
    payload = _extract_json_object(response.content)
    question_raw = str(payload.get("question", "")).strip()
    if not question_raw:
        raise ValueError("LLM clarification question is empty.")
    if not question_raw.lower().startswith("нужны уточнения"):
        question_raw = f"Нужны уточнения: {question_raw}"
    usage = {
        "calls": 1,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "estimated_cost_usd": response.usage.estimated_cost_usd,
    }
    return question_raw, usage


def _build_grounded_explanation_with_llm(
    *,
    option_id: str,
    base_explanation: str,
    policy_summary: str,
    budget_summary: str,
    score_total: float,
    citations: list[dict[str, object]],
    trace_id: str | None,
) -> tuple[str, dict[str, float | int]]:
    gateway = get_default_llm_gateway()
    response = gateway.complete(
        system_prompt=(
            "You rewrite recommendation explanation in Russian. "
            "Use only provided facts. Do not invent new policy rules, prices, or sources."
        ),
        user_prompt=(
            "Return strict JSON object with key 'explanation'.\n"
            f"option_id: {option_id}\n"
            f"score_total: {score_total}\n"
            f"base_explanation: {base_explanation}\n"
            f"policy_summary: {policy_summary}\n"
            f"budget_summary: {budget_summary}\n"
            f"citations: {citations}\n"
            "Explanation must mention policy status and budget status, be max 2 sentences."
        ),
        temperature=0.0,
        trace_id=trace_id,
        response_format={"type": "json_object"},
    )
    payload = _extract_json_object(response.content)
    explanation = str(payload.get("explanation", "")).strip()
    if not explanation:
        raise ValueError("LLM explanation is empty.")
    usage = {
        "calls": 1,
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "estimated_cost_usd": response.usage.estimated_cost_usd,
    }
    return explanation, usage


def parse_request_node(state: TravelAgentState) -> TravelAgentState:
    if state.get("error") is not None:
        return state
    _enter_step(state, "parse_request")
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    state.setdefault("missing_fields", [])
    state["clarification_question"] = None
    state.setdefault("travel_options", [])
    state.setdefault("retrieved_policy_chunks", [])
    state.setdefault("assembled_policy_context", "")
    state.setdefault("policy_results", [])
    state.setdefault("budget_results", [])
    state.setdefault("ranked_options", [])
    state.setdefault("tool_calls", [])
    state.setdefault("tool_events", [])
    state.setdefault("audit_events", [])
    state.setdefault(
        "llm_usage",
        {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
        },
    )
    state.setdefault(
        "embedding_usage",
        {
            "calls": 0,
            "input_tokens": 0,
            "estimated_cost_usd": 0.0,
            "latency_ms_total": 0,
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
        },
    )
    state.setdefault("backend_status", "healthy")
    state.setdefault("degraded_components", [])
    state.setdefault("node_transitions", 0)
    state.setdefault("rejected_options", [])
    state.setdefault("travel_search_outcome", None)
    state.setdefault("travel_search_error_code", None)
    state.setdefault("recovery_action", None)
    state.setdefault("recovery_advice", None)
    state.setdefault("route_transitions", [])

    if state.get("trip_request") is None:
        message = state.get("last_user_message", "")
        try:
            trip_request, usage = parse_trip_request_with_llm(message, trace_id=state.get("trace_id"))
            state["trip_request"] = _enrich_trip_request_with_fallback_parser(trip_request, message)
            _merge_llm_usage(state, usage)
        except (LLMGatewayError, json.JSONDecodeError, ValueError, TypeError):
            state["trip_request"] = parse_trip_request_from_message(message)

    return state


def validate_request_node(state: TravelAgentState) -> TravelAgentState:
    if state.get("error") is not None:
        return state
    _enter_step(state, "validate_request")

    trip_request = state.get("trip_request")
    if trip_request is None:
        state["missing_fields"] = ["trip_request"]
        return state

    missing_fields: list[str] = []
    if not trip_request.origin_city:
        missing_fields.append("origin_city")
    if not trip_request.destination_city:
        missing_fields.append("destination_city")
    if not trip_request.departure_date:
        missing_fields.append("departure_date")
    if not trip_request.return_date:
        missing_fields.append("return_date")
    if trip_request.departure_date and trip_request.return_date and trip_request.return_date < trip_request.departure_date:
        state["clarification_question"] = "Нужны уточнения: дата возврата должна быть не раньше даты выезда."
        state["missing_fields"] = ["return_date"]
        _record_audit_event(state, "trip_request_invalid_dates")
        return state
    if trip_request.departure_date and trip_request.departure_date < date.today():
        state["clarification_question"] = "Нужны уточнения: дата выезда в прошлом, укажите новую дату."
        state["missing_fields"] = ["departure_date"]
        _record_audit_event(state, "trip_request_past_departure_date")
        return state
    if trip_request.meeting_datetime and trip_request.departure_date:
        meeting_deadline = _to_utc_datetime(trip_request.meeting_datetime)
        if meeting_deadline and meeting_deadline.date() < trip_request.departure_date:
            state["clarification_question"] = "Нужны уточнения: дедлайн встречи раньше даты выезда."
            state["missing_fields"] = ["meeting_datetime"]
            _record_audit_event(state, "trip_request_meeting_deadline_invalid")
            return state

    state["missing_fields"] = missing_fields
    if not missing_fields:
        state["clarification_question"] = None
    return state


def ask_clarification_node(state: TravelAgentState) -> TravelAgentState:
    if state.get("error") is not None:
        return state
    _enter_step(state, "ask_clarification")
    missing_fields = state.get("missing_fields", [])
    try:
        question, usage = _build_clarification_question_with_llm(
            missing_fields=missing_fields,
            trip_request=state.get("trip_request"),
            user_message=state.get("last_user_message", ""),
            trace_id=state.get("trace_id"),
        )
        state["clarification_question"] = question
        _merge_llm_usage(state, usage)
    except (LLMGatewayError, json.JSONDecodeError, ValueError, TypeError):
        state["clarification_question"] = _fallback_clarification_question(missing_fields)

    if interrupt is not None:
        resume_payload = interrupt(
            {
                "type": "clarification",
                "missing_fields": state.get("missing_fields", []),
                "question": state.get("clarification_question"),
            }
        )
        if isinstance(resume_payload, dict):
            state.update(resume_payload)
    return state


def get_employee_profile_node(state: TravelAgentState) -> TravelAgentState:
    if state.get("error") is not None:
        return state
    _enter_step(state, "get_employee_profile")
    try:
        response, event = execute_get_employee_profile(
            GetEmployeeProfileRequest(employee_id=state["employee_id"]),
            _tool_context(state),
        )
        _record_tool_event(state, event)
        state["employee_profile"] = response.profile
    except ToolExecutionError as exc:
        _record_tool_event(state, exc.tool_event)
        _record_audit_event(
            state,
            "backend_unavailable",
            component=exc.tool_error.component,
            code=exc.tool_error.code,
            retryable=exc.tool_error.retryable,
        )
        if exc.tool_error.code == "TOOL_LOOKUP_ERROR":
            _record_audit_event(
                state,
                "access_denied_employee_profile",
                employee_id=state.get("employee_id"),
            )
            return _reject_with_error(
                state,
                "EMPLOYEE_ACCESS_DENIED",
                "Employee profile is not available for this session.",
                retryable=False,
            )
        return _reject_with_error(
            state,
            exc.tool_error.code,
            exc.tool_error.safe_message,
            retryable=exc.tool_error.retryable,
        )
    return state


def search_travel_options_node(state: TravelAgentState) -> TravelAgentState:
    if state.get("error") is not None:
        return state
    _enter_step(state, "search_travel_options")
    state["travel_search_outcome"] = None
    state["travel_search_error_code"] = None
    state["recovery_action"] = None
    state["recovery_advice"] = None
    try:
        trip_request = state.get("trip_request")
        profile = state.get("employee_profile")
        if trip_request is None:
            return _reject_with_error(
                state,
                "TRAVEL_REQUEST_INVALID",
                "Trip request is required before searching travel options.",
                retryable=False,
            )
        latest_arrival_hour = trip_request.latest_arrival_hour
        if latest_arrival_hour is None and trip_request.meeting_datetime is not None:
            meeting_deadline = _to_utc_datetime(trip_request.meeting_datetime)
            if meeting_deadline is not None:
                latest_arrival_hour = meeting_deadline.hour

        effective_trip = trip_request.model_copy(
            update={
                "latest_arrival_hour": latest_arrival_hour,
                "max_segments": trip_request.max_segments or 2,
            }
        )
        response, event = execute_search_travel_options(
            SearchTravelOptionsRequest(trip_request_payload=effective_trip.model_dump(mode="json")),
            _tool_context(state),
        )
        _record_tool_event(state, event)
        normalized_options = response.options
        if effective_trip.hotel_required is False:
            normalized_options = [
                item.model_copy(
                    update={
                        "hotel_name": None,
                        "hotel_nightly_price": None,
                        "total_price": max(0, item.total_price - int(item.hotel_nightly_price or 0)),
                    }
                )
                for item in response.options
            ]

        accepted_options = []
        rejected_options = []
        for option in normalized_options:
            reasons = _constraint_reasons_for_option(
                option,
                trip_request=effective_trip,
                profile=profile,
            )
            if reasons:
                rejected_options.append({"option_id": option.option_id, "reasons": reasons})
            else:
                accepted_options.append(option)
        state["travel_options"] = accepted_options
        state["rejected_options"] = rejected_options
        if rejected_options:
            _record_audit_event(
                state,
                "travel_options_rejected_by_constraints",
                rejected_count=len(rejected_options),
                reasons=sorted({reason for item in rejected_options for reason in item["reasons"]}),
            )
        state["travel_search_outcome"] = "options_found" if accepted_options else "options_not_found"
    except ToolExecutionError as exc:
        _record_tool_event(state, exc.tool_event)
        error_code = exc.tool_error.code
        state["travel_search_error_code"] = error_code
        _record_audit_event(
            state,
            "travel_search_failed",
            component=exc.tool_error.component,
            code=error_code,
            retryable=exc.tool_error.retryable,
        )
        if error_code in {"TRAVEL_API_4XX", "TRAVEL_API_INVALID_RESPONSE"}:
            state["travel_search_outcome"] = "invalid_supplier_request"
            state["recovery_advice"] = "Проверьте параметры поездки и скорректируйте ограничения."
            return state
        if error_code in {"TRAVEL_API_TIMEOUT", "TRAVEL_API_5XX", "TRAVEL_API_UNAVAILABLE"}:
            state["travel_search_outcome"] = "retryable_external_error"
            state["recovery_advice"] = "Внешний сервис недоступен, попробуйте повторить поиск."
            return state
        return _reject_with_error(state, error_code, exc.tool_error.safe_message, retryable=exc.tool_error.retryable)
    if not state["travel_options"]:
        _record_audit_event(state, "travel_options_empty")
    return state


def propose_constraint_change_node(state: TravelAgentState) -> TravelAgentState:
    if state.get("error") is not None:
        return state
    _enter_step(state, "propose_constraint_change")
    outcome = state.get("travel_search_outcome")
    if outcome == "options_not_found":
        state["recovery_advice"] = (
            "Варианты не найдены. Измените дату, время, транспорт, количество пересадок "
            "или маршрут и повторите поиск."
        )
    elif outcome == "invalid_supplier_request":
        state["recovery_advice"] = (
            "Поставщик отклонил запрос. Уточните параметры и обновите ограничения перед повтором."
        )
    elif outcome == "retryable_external_error":
        state["recovery_advice"] = (
            "Поставщик временно недоступен. Повторите поиск позже или измените параметры."
        )
    else:
        state["recovery_advice"] = "Нужны уточнения по ограничениям поездки."

    if interrupt is not None and state.get("recovery_action") is None:
        resume_payload = interrupt(
            {
                "type": "recovery",
                "travel_search_outcome": outcome,
                "advice": state.get("recovery_advice"),
                "allowed_actions": ["update_constraints", "retry_search", "cancel"],
            }
        )
        if isinstance(resume_payload, dict):
            state.update(resume_payload)

    if state.get("recovery_action") == "cancel":
        _record_audit_event(state, "travel_search_cancelled")
        state["current_step"] = "end"
    return state


def retrieve_policy_node(state: TravelAgentState) -> TravelAgentState:
    if state.get("error") is not None:
        return state
    _enter_step(state, "retrieve_policy")

    trip_request = state.get("trip_request")
    profile = state.get("employee_profile")
    query = "travel policy"
    if trip_request is not None:
        query = f"travel policy {trip_request.origin_city} {trip_request.destination_city}"

    if profile is not None:
        try:
            response, event = execute_retrieve_travel_policy(
                RetrieveTravelPolicyRequest(query=query, employee_profile=profile),
                _tool_context(state),
            )
            _record_tool_event(state, event)
            state["retrieved_policy_chunks"] = response.chunks
            usage = response.diagnostics.get("embedding_usage", {})
            if usage:
                _merge_embedding_usage(state, usage)
        except ToolExecutionError as exc:
            _record_tool_event(state, exc.tool_event)
            _record_audit_event(
                state,
                "backend_unavailable",
                component=exc.tool_error.component,
                code=exc.tool_error.code,
                retryable=exc.tool_error.retryable,
            )
            return _reject_with_error(
                state,
                exc.tool_error.code,
                exc.tool_error.safe_message,
                retryable=exc.tool_error.retryable,
            )
    else:
        return _reject_with_error(
            state,
            "EMPLOYEE_PROFILE_REQUIRED",
            "Employee profile is required before policy retrieval.",
            retryable=False,
        )

    context, safe_chunks = assemble_policy_context(state.get("retrieved_policy_chunks", []))
    state["assembled_policy_context"] = context
    state["retrieved_policy_chunks"] = safe_chunks

    return state


def check_policy_node(state: TravelAgentState) -> TravelAgentState:
    if state.get("error") is not None:
        return state
    _enter_step(state, "check_policy")

    refs = state.get("retrieved_policy_chunks", [])
    profile = state.get("employee_profile")
    policy_results = []
    for option in state.get("travel_options", []):
        policy_results.extend(evaluate_option(option, refs, profile))

    state["policy_results"] = policy_results
    return state


def check_budget_node(state: TravelAgentState) -> TravelAgentState:
    if state.get("error") is not None:
        return state
    _enter_step(state, "check_budget")

    profile = state.get("employee_profile")
    threshold = profile.requires_preapproval_over_amount if profile is not None else 45000

    budget_results = []
    for option in state.get("travel_options", []):
        try:
            response, event = execute_check_budget(
                CheckBudgetRequest(
                    employee_id=state.get("employee_id", "unknown"),
                    total_price=option.total_price,
                    threshold=threshold,
                ),
                _tool_context(state),
            )
            _record_tool_event(state, event)
            budget_results.append(response.result)
        except ToolExecutionError as exc:
            _record_tool_event(state, exc.tool_event)
            return _reject_with_error(
                state,
                exc.tool_error.code,
                exc.tool_error.safe_message,
                retryable=exc.tool_error.retryable,
            )

    state["budget_results"] = budget_results
    return state


def rank_options_node(state: TravelAgentState) -> TravelAgentState:
    if state.get("error") is not None:
        return state
    _enter_step(state, "rank_options")

    travel_options = state.get("travel_options", [])
    budget_results = state.get("budget_results", [])
    citations = state.get("retrieved_policy_chunks", [])
    profile = state.get("employee_profile")
    trip_request = state.get("trip_request")

    policy_results_by_option = {
        option.option_id: evaluate_option(option, citations, profile) for option in travel_options
    }

    budget_results_by_option = {
        option.option_id: budget_results[index]
        for index, option in enumerate(travel_options)
        if index < len(budget_results)
    }

    state["ranked_options"] = rank_travel_options(
        travel_options,
        policy_results_by_option,
        budget_results_by_option,
        citations,
        trip_request=trip_request,
    )
    enriched_recommendations = []
    for index, recommendation in enumerate(state["ranked_options"]):
        if index >= 3:
            enriched_recommendations.append(recommendation)
            continue
        citations_payload = [
            {"document_id": item.document_id, "section_id": item.section_id}
            for item in recommendation.citations
        ]
        try:
            explanation, usage = _build_grounded_explanation_with_llm(
                option_id=recommendation.option_id,
                base_explanation=recommendation.explanation,
                policy_summary=recommendation.policy_summary,
                budget_summary=recommendation.budget_summary,
                score_total=recommendation.score_total,
                citations=citations_payload,
                trace_id=state.get("trace_id"),
            )
            _merge_llm_usage(state, usage)
            enriched_recommendations.append(recommendation.model_copy(update={"explanation": explanation}))
        except (LLMGatewayError, json.JSONDecodeError, ValueError, TypeError):
            fallback_explanation = (
                f"{recommendation.explanation} {recommendation.policy_summary} {recommendation.budget_summary}"
            )
            enriched_recommendations.append(recommendation.model_copy(update={"explanation": fallback_explanation}))
    state["ranked_options"] = enriched_recommendations
    return state


def present_recommendations_node(state: TravelAgentState) -> TravelAgentState:
    if state.get("error") is not None:
        return state
    _enter_step(state, "present_recommendations")
    ranked = state.get("ranked_options", [])
    if ranked and not any(item.is_recommended for item in ranked):
        state["travel_search_outcome"] = "options_not_found"
        state["recovery_advice"] = (
            "Все найденные варианты нарушают обязательные правила политики. "
            "Измените ограничения поездки и повторите поиск."
        )
        _record_audit_event(state, "no_policy_compliant_recommendation")
        return state
    if state.get("selected_option_id") is None and interrupt is not None:
        resume_payload = interrupt(
            {
                "type": "selection",
                "available_options": [item.option_id for item in state.get("ranked_options", [])],
            }
        )
        if isinstance(resume_payload, dict):
            state.update(resume_payload)
    return state


def request_confirmation_node(state: TravelAgentState) -> TravelAgentState:
    if state.get("error") is not None:
        return state
    _enter_step(state, "request_confirmation")
    selected_option_id = state.get("selected_option_id")
    if selected_option_id and not _has_selected_option_in_recommendations(state):
        _record_audit_event(
            state,
            "invalid_option_selection",
            employee_id=state.get("employee_id"),
            selected_option_id=selected_option_id,
        )
        return _reject_with_error(
            state,
            "OPTION_NOT_ALLOWED",
            f"Selected option {selected_option_id} is not available in current recommendations.",
            retryable=False,
        )

    if state.get("confirmation_status") in {"confirmed", "rejected", "change_selection"}:
        _record_audit_event(
            state,
            "confirmation_decision",
            employee_id=state.get("employee_id"),
            selected_option_id=state.get("selected_option_id"),
            decision=state.get("confirmation_status"),
        )

    if state.get("confirmation_status") is None and interrupt is not None:
        resume_payload = interrupt(
            {
                "type": "confirmation",
                "selected_option_id": state.get("selected_option_id"),
                "allowed_actions": ["confirmed", "rejected", "change_selection"],
            }
        )
        if isinstance(resume_payload, dict):
            state.update(resume_payload)
    return state


def create_approval_request_node(state: TravelAgentState) -> TravelAgentState:
    if state.get("error") is not None:
        return state
    _enter_step(state, "create_approval_request")
    if state.get("confirmation_status") != "confirmed":
        _record_audit_event(
            state,
            "write_blocked_missing_confirmation",
            employee_id=state.get("employee_id"),
            selected_option_id=state.get("selected_option_id"),
        )
        return _reject_with_error(
            state,
            "CONFIRMATION_REQUIRED",
            "Cannot create approval request without explicit confirmation.",
            retryable=False,
        )

    selected_option_id = state.get("selected_option_id")
    if not selected_option_id:
        _record_audit_event(
            state,
            "write_blocked_missing_selection",
            employee_id=state.get("employee_id"),
        )
        return _reject_with_error(
            state,
            "SELECTION_REQUIRED",
            "Cannot create approval request without selected option.",
            retryable=False,
        )

    ranked_options = state.get("ranked_options", [])
    selected_recommendation = next(
        (item for item in ranked_options if item.option_id == selected_option_id),
        None,
    )
    if selected_recommendation is None:
        _record_audit_event(
            state,
            "write_blocked_unknown_option",
            employee_id=state.get("employee_id"),
            selected_option_id=selected_option_id,
        )
        return _reject_with_error(
            state,
            "OPTION_NOT_FOUND",
            f"Selected option {selected_option_id} not found in ranked options.",
            retryable=False,
        )

    travel_option = next(
        (item for item in state.get("travel_options", []) if item.option_id == selected_option_id),
        None,
    )
    if travel_option is None:
        _record_audit_event(
            state,
            "write_blocked_unknown_travel_option",
            employee_id=state.get("employee_id"),
            selected_option_id=selected_option_id,
        )
        return _reject_with_error(
            state,
            "TRAVEL_OPTION_NOT_FOUND",
            f"Selected option {selected_option_id} not found in travel options.",
            retryable=False,
        )

    budget_result = next(
        (
            item
            for item in state.get("budget_results", [])
            if item.total_price == travel_option.total_price
        ),
        None,
    )
    if budget_result is None:
        _record_audit_event(
            state,
            "write_blocked_budget_result_missing",
            employee_id=state.get("employee_id"),
            selected_option_id=selected_option_id,
        )
        return _reject_with_error(
            state,
            "BUDGET_RESULT_NOT_FOUND",
            "Budget result not found for selected option.",
            retryable=False,
        )

    policy_results = evaluate_option(
        travel_option,
        selected_recommendation.citations,
        state.get("employee_profile"),
    )
    if any(item.verdict == "fail" for item in policy_results):
        _record_audit_event(
            state,
            "write_blocked_policy_violation",
            employee_id=state.get("employee_id"),
            selected_option_id=selected_option_id,
        )
        return _reject_with_error(
            state,
            "POLICY_VIOLATION",
            "Selected option violates mandatory travel policy rules.",
            retryable=False,
        )
    idempotency_key = str(
        uuid5(
            NAMESPACE_URL,
            f"{state['session_id']}:{state['employee_id']}:{selected_option_id}",
        )
    )
    request_id = f"REQ-{uuid4().hex[:8].upper()}"
    approval_request = ApprovalRequest(
        request_id=request_id,
        employee_id=state["employee_id"],
        trip_request=state["trip_request"],
        selected_option_id=selected_option_id,
        total_price=travel_option.total_price,
        budget_result=budget_result,
        policy_results=policy_results,
        citations=selected_recommendation.citations,
        status="draft_created",
        created_at=datetime.now(UTC),
        idempotency_key=idempotency_key,
    )

    try:
        response, event = execute_create_approval_request(
            CreateApprovalRequestRequest(
                request=approval_request,
                confirmation_status=state.get("confirmation_status"),
            ),
            _tool_context(state),
        )
        _record_tool_event(state, event)
        stored = response.stored
    except ToolExecutionError as exc:
        _record_tool_event(state, exc.tool_event)
        _record_audit_event(
            state,
            "backend_unavailable",
            component=exc.tool_error.component,
            code=exc.tool_error.code,
            retryable=exc.tool_error.retryable,
        )
        return _reject_with_error(
            state,
            exc.tool_error.code,
            exc.tool_error.safe_message,
            retryable=exc.tool_error.retryable,
        )
    _record_audit_event(
        state,
        "approval_request_created",
        employee_id=state.get("employee_id"),
        selected_option_id=selected_option_id,
        request_id=stored.request_id,
    )
    state["approval_request_id"] = stored.request_id
    return state


def show_result_node(state: TravelAgentState) -> TravelAgentState:
    if state.get("error") is not None:
        return state
    _enter_step(state, "show_result")
    return state
