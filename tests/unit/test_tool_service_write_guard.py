from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.domain.models import ApprovalRequest, BudgetCheckResult, PolicyCheckResult, PolicyChunkRef, TripRequest
from src.tools.contracts import CreateApprovalRequestRequest, ToolContext
from src.tools.service import ToolExecutionError, execute_create_approval_request


def _approval_request(idempotency_key: str = "idem-1") -> ApprovalRequest:
    return ApprovalRequest(
        request_id="REQ-TEST1",
        employee_id="EMP-001",
        trip_request=TripRequest(
            origin_city="Казань",
            destination_city="Санкт-Петербург",
            departure_date="2026-09-18",
            return_date="2026-09-19",
        ),
        selected_option_id="OPT-123",
        total_price=42000,
        budget_result=BudgetCheckResult(verdict="within_limit", total_price=42000, threshold=45000, overage=0),
        policy_results=[PolicyCheckResult(rule_id="TRANSPORT_CLASS_LIMIT", verdict="pass", message="ok")],
        citations=[PolicyChunkRef(document_id="POLICY-TRANSPORT", section_id="S1", source_path="x", score=0.9)],
        status="draft_created",
        created_at=datetime.now(UTC),
        idempotency_key=idempotency_key,
    )


def _context() -> ToolContext:
    return ToolContext(correlation_id="corr-12345678", session_id="sess-1", trace_id="trace-1")


@pytest.mark.parametrize("status", [None, "rejected", "change_selection", "unknown"])
def test_write_tool_blocks_without_confirmed(monkeypatch, status):
    calls = {"count": 0}

    class _Repo:
        @staticmethod
        def create_idempotent(request):
            calls["count"] += 1
            return request

    monkeypatch.setattr("src.tools.service.approval_request_repository", _Repo())
    with pytest.raises(ToolExecutionError) as exc_info:
        execute_create_approval_request(
            CreateApprovalRequestRequest(request=_approval_request(), confirmation_status=status),
            _context(),
        )
    assert exc_info.value.tool_error.code == "CONFIRMATION_REQUIRED"
    assert calls["count"] == 0


def test_write_tool_idempotent_confirmed(monkeypatch):
    stored: dict[str, ApprovalRequest] = {}
    calls = {"count": 0}

    class _Repo:
        @staticmethod
        def create_idempotent(request: ApprovalRequest) -> ApprovalRequest:
            calls["count"] += 1
            existing = stored.get(request.idempotency_key)
            if existing is not None:
                return existing
            stored[request.idempotency_key] = request
            return request

    monkeypatch.setattr("src.tools.service.approval_request_repository", _Repo())
    first, _event_one = execute_create_approval_request(
        CreateApprovalRequestRequest(request=_approval_request("idem-42"), confirmation_status="confirmed"),
        _context(),
    )
    second, _event_two = execute_create_approval_request(
        CreateApprovalRequestRequest(request=_approval_request("idem-42"), confirmation_status="confirmed"),
        _context(),
    )
    assert first.stored.request_id == second.stored.request_id
    assert calls["count"] == 2
