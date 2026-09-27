from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import Any

from src.observability.tracing import sanitize_for_logging, traced_span
from src.persistence.repository import approval_request_repository
from src.rag.retrieval import RagUnavailableError, retrieve_policy_chunks_with_diagnostics
from src.tools.contracts import (
    CheckBudgetRequest,
    CheckBudgetResponse,
    CreateApprovalRequestRequest,
    CreateApprovalRequestResponse,
    GetEmployeeProfileRequest,
    GetEmployeeProfileResponse,
    RetrieveTravelPolicyRequest,
    RetrieveTravelPolicyResponse,
    SearchTravelOptionsRequest,
    SearchTravelOptionsResponse,
    ToolContext,
    ToolError,
    ToolEvent,
)
from src.tools.stubs import BackendDependencyError, check_budget, get_employee_profile, search_travel_options


class ToolExecutionError(RuntimeError):
    def __init__(self, tool_error: ToolError, tool_event: ToolEvent) -> None:
        super().__init__(tool_error.safe_message)
        self.tool_error = tool_error
        self.tool_event = tool_event


def _build_tool_event(
    *,
    tool_name: str,
    backend_source: str | None,
    context: ToolContext,
    status: str,
    latency_ms: int,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any] | None = None,
    error: ToolError | None = None,
    retries: int = 0,
) -> ToolEvent:
    return ToolEvent(
        tool_name=tool_name,
        backend_source=backend_source,
        correlation_id=context.correlation_id,
        session_id=context.session_id,
        trace_id=context.trace_id,
        status=status,
        latency_ms=latency_ms,
        retries=retries,
        request=sanitize_for_logging(request_payload),
        response=sanitize_for_logging(response_payload) if response_payload is not None else None,
        error=error,
        timestamp=datetime.now(UTC),
    )


def _run_tool_call(
    *,
    tool_name: str,
    backend_source: str | None,
    context: ToolContext,
    request_payload: dict[str, Any],
    callable_fn,
) -> tuple[Any, ToolEvent]:
    started_at = perf_counter()
    with traced_span(
        f"travelcopilot.tool.{tool_name}",
        trace_id=context.trace_id,
        input_data={
            "correlation_id": context.correlation_id,
            "session_id": context.session_id,
            "request": request_payload,
        },
        metadata={"component": "tool", "tool_name": tool_name},
    ) as span:
        try:
            output = callable_fn()
        except BackendDependencyError as exc:
            latency_ms = int((perf_counter() - started_at) * 1000)
            tool_error = ToolError(
                code=exc.code,
                retryable=exc.retryable,
                safe_message=exc.safe_message,
                component=exc.component,
            )
            event = _build_tool_event(
                tool_name=tool_name,
                backend_source=backend_source,
                context=context,
                status="error",
                latency_ms=latency_ms,
                request_payload=request_payload,
                error=tool_error,
            )
            span.update(output_data={"status": "error", "error_code": tool_error.code}, metadata={"latency_ms": latency_ms})
            raise ToolExecutionError(tool_error, event) from exc
        except RagUnavailableError as exc:
            latency_ms = int((perf_counter() - started_at) * 1000)
            tool_error = ToolError(
                code=exc.code,
                retryable=exc.retryable,
                safe_message=exc.safe_message,
                component=exc.component,
            )
            event = _build_tool_event(
                tool_name=tool_name,
                backend_source=backend_source,
                context=context,
                status="error",
                latency_ms=latency_ms,
                request_payload=request_payload,
                error=tool_error,
            )
            span.update(output_data={"status": "error", "error_code": tool_error.code}, metadata={"latency_ms": latency_ms})
            raise ToolExecutionError(tool_error, event) from exc
        except LookupError as exc:
            latency_ms = int((perf_counter() - started_at) * 1000)
            tool_error = ToolError(
                code="TOOL_LOOKUP_ERROR",
                retryable=False,
                safe_message=str(exc),
                component=tool_name,
            )
            event = _build_tool_event(
                tool_name=tool_name,
                backend_source=backend_source,
                context=context,
                status="error",
                latency_ms=latency_ms,
                request_payload=request_payload,
                error=tool_error,
            )
            span.update(output_data={"status": "error", "error_code": tool_error.code}, metadata={"latency_ms": latency_ms})
            raise ToolExecutionError(tool_error, event) from exc
        except Exception as exc:  # pragma: no cover - defensive fallback
            latency_ms = int((perf_counter() - started_at) * 1000)
            tool_error = ToolError(
                code="TOOL_RUNTIME_ERROR",
                retryable=True,
                safe_message=f"Tool runtime failed: {type(exc).__name__}",
                component=tool_name,
            )
            event = _build_tool_event(
                tool_name=tool_name,
                backend_source=backend_source,
                context=context,
                status="error",
                latency_ms=latency_ms,
                request_payload=request_payload,
                error=tool_error,
            )
            span.update(output_data={"status": "error", "error_code": tool_error.code}, metadata={"latency_ms": latency_ms})
            raise ToolExecutionError(tool_error, event) from exc

    latency_ms = int((perf_counter() - started_at) * 1000)
    response_payload = output.model_dump(mode="json") if hasattr(output, "model_dump") else {"ok": True}
    event = _build_tool_event(
        tool_name=tool_name,
        backend_source=backend_source,
        context=context,
        status="ok",
        latency_ms=latency_ms,
        request_payload=request_payload,
        response_payload=response_payload,
    )
    return output, event


def execute_get_employee_profile(
    request: GetEmployeeProfileRequest,
    context: ToolContext,
) -> tuple[GetEmployeeProfileResponse, ToolEvent]:
    return _run_tool_call(
        tool_name="get_employee_profile",
        backend_source="postgres",
        context=context,
        request_payload=request.model_dump(mode="json"),
        callable_fn=lambda: GetEmployeeProfileResponse(profile=get_employee_profile(request.employee_id)),
    )


def execute_search_travel_options(
    request: SearchTravelOptionsRequest,
    context: ToolContext,
) -> tuple[SearchTravelOptionsResponse, ToolEvent]:
    from src.domain.models import TripRequest

    trip_request = TripRequest.model_validate(request.trip_request_payload)
    return _run_tool_call(
        tool_name="search_travel_options",
        backend_source="http_mock_travel_api",
        context=context,
        request_payload=request.model_dump(mode="json"),
        callable_fn=lambda: SearchTravelOptionsResponse(options=search_travel_options(trip_request)),
    )


def execute_retrieve_travel_policy(
    request: RetrieveTravelPolicyRequest,
    context: ToolContext,
) -> tuple[RetrieveTravelPolicyResponse, ToolEvent]:
    def _retrieve() -> RetrieveTravelPolicyResponse:
        chunks, diagnostics = retrieve_policy_chunks_with_diagnostics(
            request.query,
            request.employee_profile,
            trace_id=context.trace_id,
        )
        return RetrieveTravelPolicyResponse(chunks=chunks, diagnostics=diagnostics)

    return _run_tool_call(
        tool_name="retrieve_travel_policy",
        backend_source="qdrant+router.ai",
        context=context,
        request_payload=request.model_dump(mode="json"),
        callable_fn=_retrieve,
    )


def execute_check_budget(
    request: CheckBudgetRequest,
    context: ToolContext,
) -> tuple[CheckBudgetResponse, ToolEvent]:
    return _run_tool_call(
        tool_name="check_budget",
        backend_source="deterministic_policy_code",
        context=context,
        request_payload=request.model_dump(mode="json"),
        callable_fn=lambda: CheckBudgetResponse(result=check_budget(request.total_price, request.threshold)),
    )


def execute_create_approval_request(
    request: CreateApprovalRequestRequest,
    context: ToolContext,
) -> tuple[CreateApprovalRequestResponse, ToolEvent]:
    if request.confirmation_status != "confirmed":
        tool_error = ToolError(
            code="CONFIRMATION_REQUIRED",
            retryable=False,
            safe_message="Cannot create approval request without explicit confirmation.",
            component="approval_write",
        )
        blocked_event = _build_tool_event(
            tool_name="create_approval_request",
            backend_source="postgres",
            context=context,
            status="blocked",
            latency_ms=0,
            request_payload=request.model_dump(mode="json"),
            error=tool_error,
        )
        raise ToolExecutionError(tool_error, blocked_event)

    def _create() -> CreateApprovalRequestResponse:
        try:
            stored = approval_request_repository.create_idempotent(request.request)
        except Exception as exc:
            raise BackendDependencyError(
                code="APPROVAL_WRITE_UNAVAILABLE",
                safe_message="Approval request storage is unavailable.",
                retryable=True,
                component="postgres",
            ) from exc
        return CreateApprovalRequestResponse(stored=stored)

    return _run_tool_call(
        tool_name="create_approval_request",
        backend_source="postgres",
        context=context,
        request_payload=request.model_dump(mode="json"),
        callable_fn=_create,
    )
