from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI, HTTPException
import httpx

from src.config import is_strict_mode, settings

from src.agent.service import bootstrap_state, run_agent_step
from src.agent.service import resume_agent_step
from src.agent.security import build_prompt_injection_clarification_answer, is_prompt_injection_attempt
from src.api.schemas import (
    AgentMessageRequest,
    AgentMessageResponse,
    AgentSessionActionRequest,
    AgentSessionResponse,
    AgentSessionStartRequest,
    TripRequestPatch,
)
from src.api.session_store import session_store
app = FastAPI(title=settings.app_name, version="0.1.0")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {
        "status": "ok",
        "env": settings.app_env,
        "backend_mode": settings.backend_mode,
    }


@dataclass
class _DependencyStatus:
    name: str
    ok: bool
    detail: str


def _check_postgres() -> _DependencyStatus:
    try:
        import psycopg

        with psycopg.connect(settings.postgres_dsn, connect_timeout=2):
            return _DependencyStatus("postgres", True, "ok")
    except Exception as exc:
        return _DependencyStatus("postgres", False, type(exc).__name__)


def _check_http(name: str, url: str) -> _DependencyStatus:
    try:
        response = httpx.get(url, timeout=2.0)
        if response.status_code >= 400:
            return _DependencyStatus(name, False, f"status={response.status_code}")
        return _DependencyStatus(name, True, "ok")
    except Exception as exc:
        return _DependencyStatus(name, False, type(exc).__name__)


def _check_qdrant() -> _DependencyStatus:
    return _check_http("qdrant", f"{settings.qdrant_url.rstrip('/')}/collections")


def _check_mock_travel_api() -> _DependencyStatus:
    return _check_http("mock_travel_api", f"{settings.mock_travel_api_url.rstrip('/')}/health")


def _required_dependencies() -> list[_DependencyStatus]:
    return [_check_postgres(), _check_qdrant(), _check_mock_travel_api()]


@app.get("/ready")
def readiness_check() -> dict:
    checks = _required_dependencies()
    failed = [item for item in checks if not item.ok]
    payload = {
        "status": "ok" if not failed else "degraded",
        "backend_mode": settings.backend_mode,
        "dependencies": {item.name: {"ok": item.ok, "detail": item.detail} for item in checks},
    }
    if is_strict_mode() and failed:
        raise HTTPException(status_code=503, detail=payload)
    return payload


@app.post("/v1/agent/message", response_model=AgentMessageResponse)
def agent_message(request: AgentMessageRequest) -> AgentMessageResponse:
    session_response = create_session(AgentSessionStartRequest(employee_id=request.employee_id, message=request.message))
    return AgentMessageResponse(
        trace_id=session_response.trace_id,
        current_step=session_response.current_step,
        missing_fields=session_response.missing_fields,
        answer=session_response.answer,
        travel_search_outcome=session_response.travel_search_outcome,
        travel_search_error_code=session_response.travel_search_error_code,
        recovery_advice=session_response.recovery_advice,
        backend_status=session_response.backend_status,
        degraded_components=session_response.degraded_components,
    )


def _build_answer(state: dict) -> str:
    if state.get("error"):
        error = state.get("error")
        code = error.code if hasattr(error, "code") else getattr(error, "get", lambda _x, _y=None: None)("code")
        if code == "INPUT_TOO_LARGE":
            return "Не удалось обработать запрос: сообщение слишком длинное. Заявка не создана. Сократите запрос и повторите попытку."
        return "Не удалось обработать запрос. Проверьте входные данные и повторите попытку."

    if state.get("current_step") == "ask_clarification":
        question = state.get("clarification_question") or ("Нужны уточнения: " + ", ".join(state.get("missing_fields", [])))
        if state.get("prompt_injection_detected") or is_prompt_injection_attempt(
            str(state.get("last_user_message") or "")
        ):
            return build_prompt_injection_clarification_answer(
                missing_fields=list(state.get("missing_fields") or []),
                question=str(question) if question else None,
            )
        return f"{question} Создание заявки без подтверждения невозможно. Источник: [POLICY-GENERAL:SEC-APPROVAL]."

    if state.get("current_step") == "propose_constraint_change":
        advice = str(state.get("recovery_advice") or "Измените ограничения и повторите поиск.")
        outcome = str(state.get("travel_search_outcome") or "")
        if outcome == "retryable_external_error":
            return advice + " Внешний сервис временно недоступен, повторите поиск позже."
        if outcome == "invalid_supplier_request":
            return advice + " Проверьте формат дат и ограничений перед повтором."
        return advice

    if state.get("current_step") == "present_recommendations":
        recommendations = state.get("ranked_options", [])
        if recommendations:
            top = recommendations[0]
            citations = []
            for item in top.citations:
                if item.document_id and item.section_id:
                    citations.append(f"[{item.document_id}:{item.section_id}]")
                elif item.document_id:
                    citations.append(f"[{item.document_id}]")
            suffix = f" Источники: {', '.join(citations)}." if citations else ""
            return (
                f"Найдено вариантов: {len(recommendations)}. "
                f"Рекомендуем {top.option_id} (score={top.score_total}). "
                f"Политика: {top.policy_summary} Бюджет: {top.budget_summary}{suffix}"
            )
        return "Подходящие варианты не найдены."

    if state.get("current_step") == "request_confirmation":
        status = state.get("confirmation_status")
        if status == "rejected":
            return "Создание заявки отменено пользователем."
        return (
            "Попытка создать заявку без подтверждения отклонена. "
            "Подтвердите создание заявки по выбранному варианту. "
            "Без подтверждения заявка не создаётся и запись в базе не выполняется."
        )

    if state.get("current_step") == "show_result":
        request_id = state.get("approval_request_id")
        recommendations = state.get("ranked_options", [])
        selected_option = state.get("selected_option_id")
        selected = next((item for item in recommendations if item.option_id == selected_option), None)
        if selected is None:
            return f"Заявка создана: {request_id}."
        citations = []
        for item in selected.citations:
            if item.document_id and item.section_id:
                citations.append(f"[{item.document_id}:{item.section_id}]")
            elif item.document_id:
                citations.append(f"[{item.document_id}]")
        suffix = f" Источники: {', '.join(citations)}." if citations else ""
        return (
            f"Заявка создана: {request_id}. "
            f"Политика: {selected.policy_summary} "
            f"Бюджет: {selected.budget_summary}{suffix}"
        )

    if state.get("current_step") == "end" and state.get("recovery_action") == "cancel":
        return "Поиск вариантов отменён пользователем."

    return "Запрос принят."


def _available_actions_for_state(state: dict) -> list[str]:
    step = state.get("current_step")
    if step == "ask_clarification":
        return ["submit_clarification"]
    if step == "present_recommendations":
        return ["select_option"]
    if step == "request_confirmation":
        return ["confirm", "reject", "change_selection"]
    if step == "propose_constraint_change":
        return ["update_constraints", "retry_search", "cancel"]
    return []


def _serialize_session_response(session_id: str, state: dict) -> AgentSessionResponse:
    error = state.get("error")
    if error and error.code == "EMPLOYEE_ACCESS_DENIED":
        raise HTTPException(status_code=403, detail="Employee profile access denied.")
    if error and error.code == "SESSION_ACCESS_DENIED":
        raise HTTPException(status_code=403, detail="Session access denied.")

    return AgentSessionResponse(
        session_id=session_id,
        employee_id=state.get("employee_id", ""),
        trace_id=state.get("trace_id", ""),
        current_step=state.get("current_step", ""),
        missing_fields=state.get("missing_fields", []),
        answer=_build_answer(state),
        clarification_question=state.get("clarification_question"),
        trip_request=state.get("trip_request"),
        recommendations=state.get("ranked_options", []),
        selected_option_id=state.get("selected_option_id"),
        approval_request_id=state.get("approval_request_id"),
        available_actions=_available_actions_for_state(state),
        tool_calls=state.get("tool_calls", []),
        tool_events=state.get("tool_events", []),
        audit_events=state.get("audit_events", []),
        travel_search_outcome=state.get("travel_search_outcome"),
        travel_search_error_code=state.get("travel_search_error_code"),
        recovery_advice=state.get("recovery_advice"),
        rejected_options=state.get("rejected_options", []),
        route_transitions=state.get("route_transitions", []),
        backend_status=state.get("backend_status", "healthy"),
        degraded_components=state.get("degraded_components", []),
        error=error,
    )


def _apply_trip_request_patch(state: dict, patch: TripRequestPatch) -> dict:
    current_trip = state.get("trip_request")
    patch_values = patch.model_dump(exclude_unset=True)
    from pydantic import ValidationError
    if current_trip is None:
        from src.domain.models import TripRequest

        current_trip = TripRequest()
    merged_payload = {**current_trip.model_dump(mode="json"), **patch_values}
    try:
        from src.domain.models import TripRequest

        updated_trip = TripRequest.model_validate(merged_payload)
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid trip_request_patch: {exc.errors()[0]['msg']}") from exc
    return {
        "trip_request": updated_trip,
        "missing_fields": [],
        "clarification_question": None,
        "error": None,
    }


@app.post("/v1/agent/sessions", response_model=AgentSessionResponse)
def create_session(request: AgentSessionStartRequest) -> AgentSessionResponse:
    if len(request.message) > settings.max_user_chars:
        raise HTTPException(status_code=400, detail="Input is too large.")

    state = bootstrap_state(employee_id=request.employee_id, user_message=request.message)
    state = run_agent_step(state)
    record = session_store.create(state)
    return _serialize_session_response(record.session_id, record.state)


@app.get("/v1/agent/sessions/{session_id}", response_model=AgentSessionResponse)
def get_session(session_id: str, employee_id: str) -> AgentSessionResponse:
    record = session_store.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if record.employee_id != employee_id:
        raise HTTPException(status_code=403, detail="Session access denied.")
    return _serialize_session_response(session_id, record.state)


@app.post("/v1/agent/sessions/{session_id}/actions", response_model=AgentSessionResponse)
def apply_session_action(session_id: str, request: AgentSessionActionRequest) -> AgentSessionResponse:
    record = session_store.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    if record.employee_id != request.employee_id:
        raise HTTPException(status_code=403, detail="Session access denied.")

    state = record.state
    action = request.action
    updates: dict = {"employee_id": request.employee_id}

    if action == "submit_clarification":
        if request.trip_request_patch is None:
            raise HTTPException(status_code=400, detail="trip_request_patch is required for submit_clarification.")
        updates.update(_apply_trip_request_patch(state, request.trip_request_patch))
    elif action == "update_constraints":
        if request.trip_request_patch is None:
            raise HTTPException(status_code=400, detail="trip_request_patch is required for update_constraints.")
        updates.update(_apply_trip_request_patch(state, request.trip_request_patch))
        updates.update(
            {
                "travel_search_outcome": None,
                "travel_search_error_code": None,
                "recovery_action": "update_constraints",
                "selected_option_id": None,
                "confirmation_status": None,
                "travel_options": [],
                "ranked_options": [],
                "policy_results": [],
                "budget_results": [],
                "retrieved_policy_chunks": [],
                "assembled_policy_context": "",
            }
        )
    elif action == "retry_search":
        updates.update(
            {
                "recovery_action": "retry_search",
                "travel_search_error_code": None,
                "error": None,
            }
        )
    elif action == "cancel":
        updates.update({"recovery_action": "cancel", "error": None})
    elif action == "select_option":
        if not request.selected_option_id:
            raise HTTPException(status_code=400, detail="selected_option_id is required for select_option.")
        updates.update({"selected_option_id": request.selected_option_id, "confirmation_status": None, "error": None})
    elif action == "confirm":
        updates.update({"confirmation_status": "confirmed", "error": None})
    elif action == "reject":
        updates.update({"confirmation_status": "rejected", "error": None})
    elif action == "change_selection":
        updates.update({"confirmation_status": "change_selection", "selected_option_id": None, "error": None})
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported action: {action}")

    next_state = resume_agent_step(state, **updates)
    updated = session_store.update_state(session_id, next_state)
    if updated is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return _serialize_session_response(session_id, updated.state)
