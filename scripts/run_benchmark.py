from __future__ import annotations

import json
import os
import subprocess
import hashlib
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.security import build_prompt_injection_clarification_answer, is_prompt_injection_attempt
from src.agent.service import bootstrap_state, resume_agent_step, run_agent_step
from src.config import is_strict_mode, settings
from src.observability.tracing import (
    get_tracing_diagnostics,
    sanitize_for_logging,
    traced_span,
    traced_workflow,
    verify_trace,
)
from src.persistence.repository import PostgresApprovalRequestRepository
from src.rag.embeddings import resolve_policy_collection_name

CASES_PATH = ROOT / "data" / "benchmark" / "cases.yaml"
REPORT_PATH = ROOT / "data" / "benchmark" / "last_report.json"


@dataclass
class CaseResult:
    case_id: str
    success: bool
    errors: list[str]
    final_step: str
    tool_calls: list[str]
    tool_events: list[dict[str, Any]]
    approval_request_id: str | None
    trace_id: str | None
    trace_url: str | None
    trace_exported: bool
    trace_verified: bool
    trace_verification_mode: str
    latency_ms: int
    llm_calls: int
    llm_total_tokens: int
    llm_estimated_cost_usd: float
    embedding_calls: int
    embedding_input_tokens: int
    embedding_estimated_cost_usd: float
    fallback_used: bool
    strict_preconditions_passed: bool
    backend_sources: dict[str, str]
    db_write_confirmed: bool | None
    tool_call_correct: bool
    answer_text: str
    answer_hash: str
    quality_bucket: str


def _to_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def _is_subsequence(order: list[str], sequence: list[str]) -> bool:
    if not order:
        return True
    index = 0
    for item in sequence:
        if item == order[index]:
            index += 1
            if index == len(order):
                return True
    return False


def _extract_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _strict_preconditions() -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not is_strict_mode():
        errors.append(f"BACKEND_MODE must be strict, got '{settings.backend_mode}'.")
    if settings.embedding_provider.strip().lower() != "router.ai":
        errors.append("EMBEDDING_PROVIDER must be router.ai in strict benchmark.")
    if settings.embedding_model.strip() != "baai/bge-m3":
        errors.append("EMBEDDING_MODEL must be baai/bge-m3 in strict benchmark.")
    if settings.embedding_dimension != 1024:
        errors.append(f"EMBEDDING_DIMENSION must be 1024, got {settings.embedding_dimension}.")
    if not settings.embedding_api_key.strip():
        errors.append("EMBEDDING_API_KEY is required in strict benchmark.")
    return (len(errors) == 0, errors)


def _git_commit_sha() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return None


def _collect_backend_sources(tool_events: list[dict[str, Any]]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for event in tool_events:
        tool_name = str(event.get("tool_name", ""))
        if not tool_name or tool_name in sources:
            continue
        sources[tool_name] = str(event.get("backend_source", "unknown"))
    return sources


def _has_fallback_signals(state: dict[str, Any], tool_events: list[dict[str, Any]]) -> bool:
    if state.get("backend_status") == "degraded":
        return True
    if state.get("degraded_components"):
        return True
    for event in tool_events:
        source = str(event.get("backend_source", "")).lower()
        if "fallback" in source or "local_hash" in source or "local_policy_stub" in source:
            return True
    for option in state.get("travel_options", []) or []:
        option_id = getattr(option, "option_id", None)
        if option_id is None and isinstance(option, dict):
            option_id = option.get("option_id")
        if option_id == "OPT-001":
            return True
    embedding_mode = str((state.get("embedding_usage") or {}).get("mode", "")).lower()
    if embedding_mode == "local":
        return True
    return False


def _validate_required_backend_sources(
    *,
    expected_output: dict[str, Any],
    tool_calls: list[str],
    backend_sources: dict[str, str],
) -> list[str]:
    errors: list[str] = []
    required_by_tool = {
        "get_employee_profile": "postgres",
        "search_travel_options": "http_mock_travel_api",
        "retrieve_travel_policy": "qdrant+router.ai",
        "create_approval_request": "postgres",
    }
    for tool_name in tool_calls:
        if tool_name not in required_by_tool:
            continue
        expected_source = required_by_tool[tool_name]
        actual_source = backend_sources.get(tool_name)
        if actual_source != expected_source:
            errors.append(
                f"backend source mismatch for {tool_name}: expected={expected_source}, got={actual_source or 'unknown'}"
            )
    if expected_output.get("approval_request_created") is True and "create_approval_request" not in tool_calls:
        errors.append("backend source for create_approval_request cannot be confirmed (tool not called)")
    return errors


def _validate_tool_expectations(
    *,
    expected_tools: dict[str, Any],
    tool_calls: list[str],
    tool_events: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    required_tools = expected_tools.get("required", [])
    forbidden_tools = expected_tools.get("forbidden", [])
    tool_calls_set = set(tool_calls)

    for tool in required_tools:
        if tool not in tool_calls_set:
            errors.append(f"required tool not called: {tool}")

    for tool in forbidden_tools:
        if tool in tool_calls_set:
            errors.append(f"forbidden tool called: {tool}")

    expected_order = expected_tools.get("order", [])
    if expected_order and not _is_subsequence(expected_order, tool_calls):
        errors.append(f"tool order mismatch: expected subsequence={expected_order}, got={tool_calls}")

    expected_counts = expected_tools.get("count", {})
    for tool_name, expected_count in expected_counts.items():
        actual_count = tool_calls.count(tool_name)
        if actual_count != int(expected_count):
            errors.append(f"tool count mismatch for {tool_name}: expected={expected_count}, got={actual_count}")

    key_arguments = expected_tools.get("key_arguments", [])
    for rule in key_arguments:
        if not isinstance(rule, dict):
            errors.append("invalid key_arguments rule")
            continue
        tool_name = str(rule.get("tool", ""))
        path = str(rule.get("path", ""))
        has_equals = "equals" in rule
        expected_value = rule.get("equals")
        must_be_not_null = bool(rule.get("not_null", False))
        matched_event = next((event for event in tool_events if event.get("tool_name") == tool_name), None)
        if matched_event is None:
            errors.append(f"key argument check failed: tool event not found for {tool_name}")
            continue
        actual_value = _extract_path(matched_event.get("request", {}), path)
        if must_be_not_null and actual_value in {None, ""}:
            errors.append(f"key argument must be non-null for {tool_name}.{path}")
            continue
        if has_equals and actual_value != expected_value:
            errors.append(
                f"key argument mismatch for {tool_name}.{path}: expected={expected_value}, got={actual_value}"
            )

    if expected_tools.get("write_after_confirmation") is True:
        write_events = [event for event in tool_events if event.get("tool_name") == "create_approval_request"]
        for event in write_events:
            confirmation_status = _extract_path(event.get("request", {}), "confirmation_status")
            if confirmation_status != "confirmed":
                errors.append("write tool executed without confirmed status")
    return errors


def _db_write_confirmed(request_id: str | None) -> bool | None:
    if not request_id:
        return None
    try:
        repository = PostgresApprovalRequestRepository(
            settings.postgres_dsn,
            timeout_seconds=min(settings.request_timeout_seconds, 10.0),
        )
        stored = repository.get_by_request_id(request_id)
        return stored is not None
    except Exception:
        return False


def _is_tool_error(error: str) -> bool:
    markers = (
        "required tool",
        "forbidden tool",
        "tool order mismatch",
        "tool count mismatch",
        "key argument",
        "write tool executed",
        "backend source mismatch",
    )
    return any(marker in error for marker in markers)


def _build_agent_answer(state: dict[str, Any]) -> str:
    if state.get("error"):
        error = state.get("error")
        code = getattr(error, "code", None) or (error.get("code") if isinstance(error, dict) else None)
        if code == "INPUT_TOO_LARGE":
            return (
                "Не удалось обработать запрос: сообщение слишком длинное и превышает допустимый лимит ввода. "
                "Заявка не была создана. Сократите запрос, уберите лишний текст и повторите попытку. "
                "Источник: [POLICY-GENERAL:SEC-INPUT]."
            )
        return "Не удалось обработать запрос. Проверьте входные данные и повторите попытку."

    step = state.get("current_step")
    if step == "ask_clarification":
        question = str(state.get("clarification_question") or "").strip()
        if state.get("prompt_injection_detected") or is_prompt_injection_attempt(
            str(state.get("last_user_message") or "")
        ):
            return build_prompt_injection_clarification_answer(
                missing_fields=list(state.get("missing_fields") or []),
                question=question or None,
            )
        if question:
            return (
                f"{question} Создание заявки без подтверждения невозможно. "
                "Источник: [POLICY-GENERAL:SEC-APPROVAL]."
            )
        return (
            "Нужны уточнения: "
            + ", ".join(state.get("missing_fields", []))
            + ". Создание заявки без подтверждения невозможно. Источник: [POLICY-GENERAL:SEC-APPROVAL]."
        )
    if step == "propose_constraint_change":
        advice = str(state.get("recovery_advice") or "Варианты не найдены. Измените ограничения и повторите поиск.")
        outcome = str(state.get("travel_search_outcome") or "")
        if outcome == "retryable_external_error":
            return (
                advice
                + " Внешний сервис временно недоступен, поэтому варианты не рассчитаны и заявка не создана. "
                "Повторите поиск позже или скорректируйте ограничения поездки."
            )
        if outcome == "invalid_supplier_request":
            return advice + " Проверьте формат дат и ограничений перед повтором."
        return advice
    if step == "present_recommendations":
        ranked = state.get("ranked_options", []) or []
        if ranked:
            top = ranked[0]
            citations = []
            for item in getattr(top, "citations", []) or []:
                doc = getattr(item, "document_id", None)
                sec = getattr(item, "section_id", None)
                if doc and sec:
                    citations.append(f"[{doc}:{sec}]")
                elif doc:
                    citations.append(f"[{doc}]")
            policy_summary = str(getattr(top, "policy_summary", "") or "").strip()
            budget_summary = str(getattr(top, "budget_summary", "") or "").strip()
            parts = [
                f"Найдено вариантов: {len(ranked)}.",
                f"Рекомендуем {top.option_id} (score={top.score_total}).",
            ]
            if policy_summary:
                parts.append(f"Политика: {policy_summary}")
            if budget_summary:
                parts.append(f"Бюджет: {budget_summary}")
            if citations:
                parts.append("Источники: " + ", ".join(citations) + ".")
            return " ".join(parts)
        return "Подходящие варианты не найдены."
    if step == "request_confirmation":
        return (
            "Попытка создать заявку без подтверждения отклонена. Подтвердите создание заявки по выбранному варианту. "
            "Без подтверждения заявка не создаётся и запись в базе не выполняется. "
            "Доступные действия: подтвердить, изменить выбор или отказаться."
        )
    if step == "show_result":
        selected_id = state.get("selected_option_id")
        recommendation = next(
            (item for item in (state.get("ranked_options", []) or []) if getattr(item, "option_id", None) == selected_id),
            None,
        )
        policy_summary = str(getattr(recommendation, "policy_summary", "") or "").strip() if recommendation else ""
        budget_summary = str(getattr(recommendation, "budget_summary", "") or "").strip() if recommendation else ""
        citations = []
        for item in getattr(recommendation, "citations", []) if recommendation else []:
            doc = getattr(item, "document_id", None)
            sec = getattr(item, "section_id", None)
            if doc and sec:
                citations.append(f"[{doc}:{sec}]")
            elif doc:
                citations.append(f"[{doc}]")
        suffix = f" Источники: {', '.join(citations)}." if citations else ""
        summary = " ".join(part for part in [f"Политика: {policy_summary}" if policy_summary else "", f"Бюджет: {budget_summary}" if budget_summary else ""] if part)
        return f"Заявка создана: {state.get('approval_request_id')}. {summary}{suffix}".strip()
    if step == "end" and state.get("recovery_action") == "cancel":
        return "Поиск вариантов отменён пользователем."
    return "Запрос принят."


def _answer_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _quality_bucket(case_id: str) -> str:
    if case_id.startswith("security_"):
        return "security"
    if "no_options" in case_id or "external_api_error" in case_id:
        return "infrastructure"
    if "rag_" in case_id or "retrieval" in case_id:
        return "rag"
    if "tool" in case_id or "idempotency" in case_id:
        return "tooling"
    return "deterministic"


def _apply_action(state: dict[str, Any], action: Any) -> dict[str, Any]:
    if isinstance(action, str):
        if action == "select_top_option":
            ranked = state.get("ranked_options", [])
            if not ranked:
                return state
            return resume_agent_step(state, selected_option_id=ranked[0].option_id, confirmation_status=None, error=None)
        if action == "confirm":
            return resume_agent_step(state, confirmation_status="confirmed", error=None)
        if action == "reject":
            return resume_agent_step(state, confirmation_status="rejected", error=None)
        if action == "change_selection":
            return resume_agent_step(state, confirmation_status="change_selection", selected_option_id=None, error=None)
        return state

    if isinstance(action, dict) and action.get("type") == "update_trip_request":
        trip = state.get("trip_request")
        if trip is None:
            return state

        updates = dict(action.get("values", {}))
        for key in ("departure_date", "return_date"):
            if key in updates and isinstance(updates[key], str):
                updates[key] = _to_iso_date(updates[key])

        updated_trip = trip.model_copy(update=updates)
        return resume_agent_step(state, trip_request=updated_trip, error=None)

    if isinstance(action, dict) and action.get("type") == "set_last_user_message":
        message = str(action.get("value", ""))
        return resume_agent_step(
            state,
            last_user_message=message,
            trip_request=None,
            missing_fields=[],
            error=None,
        )

    if isinstance(action, dict) and action.get("type") == "set_oversized_message":
        length = int(action.get("length", 0))
        fill_char = str(action.get("char", "x")) or "x"
        oversized_message = fill_char[0] * max(length, 0)
        return resume_agent_step(
            state,
            last_user_message=oversized_message,
            trip_request=None,
            missing_fields=[],
            error=None,
        )

    if isinstance(action, dict) and action.get("type") == "recovery_action":
        action_name = str(action.get("action", "")).strip()
        if action_name not in {"update_constraints", "retry_search", "cancel"}:
            return state
        updates: dict[str, Any] = {
            "recovery_action": action_name,
            "travel_search_error_code": None,
            "error": None,
        }
        values = action.get("values", {})
        if action_name == "update_constraints" and isinstance(values, dict):
            trip = state.get("trip_request")
            if trip is not None:
                normalized = dict(values)
                for key in ("departure_date", "return_date"):
                    if key in normalized and isinstance(normalized[key], str):
                        normalized[key] = _to_iso_date(normalized[key])
                updates["trip_request"] = trip.model_copy(update=normalized)
        return resume_agent_step(state, **updates)

    return state


def _run_case(case: dict[str, Any], *, strict_preconditions_passed: bool) -> CaseResult:
    case_id = case["id"]
    input_payload = case["input"]
    expected_output = case.get("expected_output", {})
    expected_tools = case.get("expected_tools", {})
    actions = case.get("actions", [])

    started_at = perf_counter()
    state = bootstrap_state(
        employee_id=input_payload["employee_id"],
        user_message=input_payload["message"],
    )
    with traced_workflow(
        name="travelcopilot.benchmark.case",
        trace_id=state.get("trace_id"),
        session_id=f"benchmark:{case_id}",
        user_id=input_payload["employee_id"],
        input_data={"case_id": case_id, "message": sanitize_for_logging(input_payload["message"])},
    ) as benchmark_trace:
        with traced_span(
            "travelcopilot.benchmark.case.run",
            trace_id=benchmark_trace.trace_id,
            metadata={"case_id": case_id},
        ):
            state = run_agent_step(state)
            for action in actions:
                state = _apply_action(state, action)

        errors: list[str] = []

        expected_status = expected_output.get("status")
        if expected_status and state.get("current_step") != expected_status:
            errors.append(f"status mismatch: expected={expected_status}, got={state.get('current_step')}")

        required_missing = expected_output.get("required_missing_fields", [])
        missing_fields = set(state.get("missing_fields", []))
        for field in required_missing:
            if field not in missing_fields:
                errors.append(f"missing field not detected: {field}")

        if "approval_request_id" in expected_output:
            expected_req_id = expected_output["approval_request_id"]
            if expected_req_id is None and state.get("approval_request_id") is not None:
                errors.append("approval_request_id must be null")

        if expected_output.get("approval_request_created") is True and not state.get("approval_request_id"):
            errors.append("approval request was not created")

        min_recommendations = expected_output.get("min_recommendations")
        if min_recommendations is not None:
            if len(state.get("ranked_options", [])) < int(min_recommendations):
                errors.append("recommendations count is below expected minimum")

        min_retrieved_chunks = expected_output.get("min_retrieved_chunks")
        if min_retrieved_chunks is not None:
            if len(state.get("retrieved_policy_chunks", [])) < int(min_retrieved_chunks):
                errors.append("retrieved policy chunks count is below expected minimum")

        if expected_output.get("require_assembled_policy_context") is True:
            if not str(state.get("assembled_policy_context", "")).strip():
                errors.append("assembled_policy_context must be non-empty")

        max_retrieved_chunk_score = expected_output.get("max_retrieved_chunk_score")
        if max_retrieved_chunk_score is not None:
            chunk_scores = [float(item.score) for item in state.get("retrieved_policy_chunks", [])]
            max_score = max(chunk_scores) if chunk_scores else 0.0
            if max_score > float(max_retrieved_chunk_score):
                errors.append(
                    "max retrieved chunk score exceeded expected upper bound: "
                    f"{max_score:.4f} > {float(max_retrieved_chunk_score):.4f}"
                )

        required_budget_verdicts = expected_output.get("required_budget_verdicts", [])
        actual_budget_verdicts = {item.verdict for item in state.get("budget_results", [])}
        for verdict in required_budget_verdicts:
            if verdict not in actual_budget_verdicts:
                errors.append(f"budget verdict missing: {verdict}")

        required_policy_rule_verdicts = expected_output.get("required_policy_rule_verdicts", [])
        actual_policy_pairs = {(item.rule_id, item.verdict) for item in state.get("policy_results", [])}
        for pair in required_policy_rule_verdicts:
            if not isinstance(pair, dict):
                errors.append("invalid required_policy_rule_verdicts entry")
                continue
            expected_pair = (str(pair.get("rule_id", "")), str(pair.get("verdict", "")))
            if expected_pair not in actual_policy_pairs:
                errors.append(
                    "policy rule/verdict missing: "
                    f"{expected_pair[0]}={expected_pair[1]}"
                )

        tool_calls = state.get("tool_calls", [])
        tool_events = state.get("tool_events", [])
        errors.extend(
            _validate_tool_expectations(
                expected_tools=expected_tools,
                tool_calls=tool_calls,
                tool_events=tool_events,
            )
        )

        if expected_output.get("idempotency_replay_same_request_id") is True:
            replay = state
            replay_again = resume_agent_step(
                replay,
                confirmation_status="confirmed",
                selected_option_id=replay.get("selected_option_id"),
                error=None,
            )
            if replay.get("approval_request_id") != replay_again.get("approval_request_id"):
                errors.append("idempotency replay created a different request id")

        backend_sources = _collect_backend_sources(tool_events)
        errors.extend(
            _validate_required_backend_sources(
                expected_output=expected_output,
                tool_calls=tool_calls,
                backend_sources=backend_sources,
            )
        )
        fallback_used = _has_fallback_signals(state, tool_events)
        if fallback_used:
            errors.append("fallback_used=true in strict benchmark case")

        db_write_confirmed = _db_write_confirmed(state.get("approval_request_id"))
        if expected_output.get("approval_request_created") is True and db_write_confirmed is not True:
            errors.append("approval request is not confirmed by PostgreSQL read")
        tool_call_correct = not any(_is_tool_error(item) for item in errors)
        answer_text = _build_agent_answer(state)
        answer_hash = _answer_hash(answer_text)
        quality_bucket = _quality_bucket(case_id)

        latency_ms = int((perf_counter() - started_at) * 1000)
        llm_usage = state.get("llm_usage", {})
        embedding_usage = state.get("embedding_usage", {})
        llm_calls = int(llm_usage.get("calls", 0))
        llm_total_tokens = int(llm_usage.get("total_tokens", 0))
        llm_estimated_cost_usd = float(llm_usage.get("estimated_cost_usd", 0.0))
        embedding_calls = int(embedding_usage.get("calls", 0))
        embedding_input_tokens = int(embedding_usage.get("input_tokens", 0))
        embedding_estimated_cost_usd = float(embedding_usage.get("estimated_cost_usd", 0.0))
        trace_id = benchmark_trace.trace_id if benchmark_trace.enabled else None
        trace_check = verify_trace(trace_id)
        trace_verified = trace_check.verified
        trace_verification_mode = trace_check.mode
        benchmark_trace.update(
            output_data={
                "case_id": case_id,
                "success": not errors,
                "final_step": state.get("current_step", ""),
                "approval_request_id": state.get("approval_request_id"),
                "llm_calls": llm_calls,
                "llm_total_tokens": llm_total_tokens,
                "llm_estimated_cost_usd": llm_estimated_cost_usd,
                "embedding_calls": embedding_calls,
                "embedding_input_tokens": embedding_input_tokens,
                "embedding_estimated_cost_usd": embedding_estimated_cost_usd,
                "fallback_used": fallback_used,
                "backend_sources": backend_sources,
                "db_write_confirmed": db_write_confirmed,
                "answer_hash": answer_hash,
                "quality_bucket": quality_bucket,
                "trace_verified": trace_verified,
                "trace_verification_mode": trace_verification_mode,
            },
            metadata={"errors": errors, "latency_ms": latency_ms},
        )

    return CaseResult(
        case_id=case_id,
        success=not errors,
        errors=errors,
        final_step=state.get("current_step", ""),
        tool_calls=tool_calls,
        tool_events=tool_events,
        approval_request_id=state.get("approval_request_id"),
        trace_id=trace_id,
        trace_url=None,
        trace_exported=bool(trace_id),
        trace_verified=trace_verified,
        trace_verification_mode=trace_verification_mode,
        latency_ms=latency_ms,
        llm_calls=llm_calls,
        llm_total_tokens=llm_total_tokens,
        llm_estimated_cost_usd=llm_estimated_cost_usd,
        embedding_calls=embedding_calls,
        embedding_input_tokens=embedding_input_tokens,
        embedding_estimated_cost_usd=embedding_estimated_cost_usd,
        fallback_used=fallback_used,
        strict_preconditions_passed=strict_preconditions_passed,
        backend_sources=backend_sources,
        db_write_confirmed=db_write_confirmed,
        tool_call_correct=tool_call_correct,
        answer_text=answer_text,
        answer_hash=answer_hash,
        quality_bucket=quality_bucket,
    )


def _percentile(values: list[int], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = max(0, min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * p))))
    return float(sorted_values[rank])


def run_benchmark(*, output_path: str | Path | None = None) -> dict[str, Any]:
    strict_preconditions_passed, strict_precondition_errors = _strict_preconditions()
    if not strict_preconditions_passed:
        raise RuntimeError("Strict benchmark preconditions failed: " + "; ".join(strict_precondition_errors))

    data = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
    cases = data.get("cases", [])

    run_id = uuid4().hex
    timestamp = datetime.now(UTC).isoformat()
    commit_sha = _git_commit_sha()
    collection_name = resolve_policy_collection_name()

    results = [_run_case(case, strict_preconditions_passed=strict_preconditions_passed) for case in cases]
    successful = sum(1 for item in results if item.success)
    total = len(results)
    success_rate = successful / total if total else 0.0

    langfuse_host = os.getenv("LANGFUSE_HOST", "http://localhost:3000").rstrip("/")
    latencies = [item.latency_ms for item in results]
    avg_latency_ms = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
    latency_p50_ms = round(_percentile(latencies, 0.50), 2)
    latency_p95_ms = round(_percentile(latencies, 0.95), 2)
    trace_coverage = sum(1 for item in results if item.trace_verified)
    trace_coverage_rate = round(trace_coverage / total, 4) if total else 0.0
    cases_without_trace = [item.case_id for item in results if not item.trace_id]
    unverified_trace_cases = [item.case_id for item in results if item.trace_exported and not item.trace_verified]
    accepted_trace_modes = {"read_api", "events_write_ack"}
    invalid_trace_verification_cases = [
        item.case_id
        for item in results
        if item.trace_exported and item.trace_verification_mode not in accepted_trace_modes
    ]
    trace_verification_modes: dict[str, int] = {}
    for item in results:
        if item.trace_exported:
            trace_verification_modes[item.trace_verification_mode] = (
                trace_verification_modes.get(item.trace_verification_mode, 0) + 1
            )
    tool_violation_cases = [
        item.case_id
        for item in results
        if any("tool" in error for error in item.errors)
    ]
    security_cases = [item for item in results if item.case_id.startswith("security_")]
    security_success_rate = (
        round(sum(1 for item in security_cases if item.success) / len(security_cases), 4)
        if security_cases
        else 0.0
    )
    total_llm_calls = sum(item.llm_calls for item in results)
    total_llm_tokens = sum(item.llm_total_tokens for item in results)
    total_llm_cost_usd = round(sum(item.llm_estimated_cost_usd for item in results), 8)
    cost_per_run_usd = round(total_llm_cost_usd / total, 8) if total else 0.0
    total_embedding_calls = sum(item.embedding_calls for item in results)
    total_embedding_input_tokens = sum(item.embedding_input_tokens for item in results)
    total_embedding_cost_usd = round(sum(item.embedding_estimated_cost_usd for item in results), 8)
    tool_call_correct_cases = sum(1 for item in results if item.tool_call_correct)
    tool_call_correctness = round(tool_call_correct_cases / total, 4) if total else 0.0
    fallback_used_cases = [item.case_id for item in results if item.fallback_used]
    deterministic_failures = [item.case_id for item in results if item.quality_bucket == "deterministic" and not item.success]
    infrastructure_failures = [item.case_id for item in results if item.quality_bucket == "infrastructure" and not item.success]
    rag_failures = [item.case_id for item in results if item.quality_bucket == "rag" and not item.success]
    tool_failures = [item.case_id for item in results if item.quality_bucket == "tooling" and not item.success]
    security_failures = [item.case_id for item in results if item.quality_bucket == "security" and not item.success]
    backend_sources_by_tool: dict[str, list[str]] = {}
    for item in results:
        for tool_name, source in item.backend_sources.items():
            backend_sources_by_tool.setdefault(tool_name, [])
            if source not in backend_sources_by_tool[tool_name]:
                backend_sources_by_tool[tool_name].append(source)
    overall_pass = bool(
        strict_preconditions_passed
        and not strict_precondition_errors
        and not fallback_used_cases
        and successful == total
        and trace_coverage_rate >= 1.0
        and not invalid_trace_verification_cases
    )

    for item in results:
        if item.trace_id:
            item.trace_url = f"{langfuse_host}/trace/{item.trace_id}"

    report = {
        "tracing": get_tracing_diagnostics(),
        "run_id": run_id,
        "timestamp": timestamp,
        "commit_sha": commit_sha,
        "backend_mode": settings.backend_mode,
        "embedding": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "dimension": settings.embedding_dimension,
        },
        "qdrant_collection": collection_name,
        "strict_preconditions_passed": strict_preconditions_passed,
        "strict_precondition_errors": strict_precondition_errors,
        "overall_pass": overall_pass,
        "total_cases": total,
        "successful_cases": successful,
        "success_rate": round(success_rate, 4),
        "latency_avg_ms": avg_latency_ms,
        "latency_p50_ms": latency_p50_ms,
        "latency_p95_ms": latency_p95_ms,
        "trace_coverage_rate": trace_coverage_rate,
        "trace_coverage_min": 1.0,
        "cases_without_trace": cases_without_trace,
        "trace_unverified_cases": unverified_trace_cases,
        "trace_verification_modes": trace_verification_modes,
        "invalid_trace_verification_cases": invalid_trace_verification_cases,
        "tool_violation_cases": tool_violation_cases,
        "security_case_count": len(security_cases),
        "security_success_rate": security_success_rate,
        "llm_calls_total": total_llm_calls,
        "llm_tokens_total": total_llm_tokens,
        "llm_estimated_cost_total_usd": total_llm_cost_usd,
        "cost_per_run_usd": cost_per_run_usd,
        "embedding_calls_total": total_embedding_calls,
        "embedding_input_tokens_total": total_embedding_input_tokens,
        "embedding_estimated_cost_total_usd": total_embedding_cost_usd,
        "fallback_used_cases": fallback_used_cases,
        "backend_sources_by_tool": backend_sources_by_tool,
        "tool_call_correctness": tool_call_correctness,
        "failure_breakdown": {
            "deterministic": deterministic_failures,
            "infrastructure": infrastructure_failures,
            "rag": rag_failures,
            "tooling": tool_failures,
            "security": security_failures,
        },
        "results": [
            {
                "id": item.case_id,
                "success": item.success,
                "strict_preconditions_passed": item.strict_preconditions_passed,
                "fallback_used": item.fallback_used,
                "final_step": item.final_step,
                "approval_request_id": item.approval_request_id,
                "db_write_confirmed": item.db_write_confirmed,
                "backend_sources": item.backend_sources,
                "tool_call_correct": item.tool_call_correct,
                "answer_text": item.answer_text,
                "answer_hash": item.answer_hash,
                "quality_bucket": item.quality_bucket,
                "trace_id": item.trace_id,
                "trace_url": item.trace_url,
                "trace_exported": item.trace_exported,
                "trace_verified": item.trace_verified,
                "trace_verification_mode": item.trace_verification_mode,
                "latency_ms": item.latency_ms,
                "llm_calls": item.llm_calls,
                "llm_total_tokens": item.llm_total_tokens,
                "llm_estimated_cost_usd": item.llm_estimated_cost_usd,
                "embedding_calls": item.embedding_calls,
                "embedding_input_tokens": item.embedding_input_tokens,
                "embedding_estimated_cost_usd": item.embedding_estimated_cost_usd,
                "tool_calls": item.tool_calls,
                "tool_events": item.tool_events,
                "errors": item.errors,
            }
            for item in results
        ],
    }

    target = Path(output_path) if output_path is not None else REPORT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run_benchmark()
    print(json.dumps(result, ensure_ascii=False, indent=2))
