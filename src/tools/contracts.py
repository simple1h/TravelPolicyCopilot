from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from src.domain.models import ApprovalRequest, BudgetCheckResult, EmployeeProfile, PolicyChunkRef, TravelOption


class ToolContext(BaseModel):
    correlation_id: str = Field(min_length=8)
    session_id: str = Field(min_length=1)
    trace_id: str | None = None
    timeout_seconds: float = Field(default=8.0, ge=0.1, le=120.0)


class ToolError(BaseModel):
    code: str
    retryable: bool
    safe_message: str
    component: str


class ToolEvent(BaseModel):
    tool_name: str
    backend_source: str | None = None
    correlation_id: str
    session_id: str
    trace_id: str | None = None
    status: str
    latency_ms: int
    retries: int = 0
    request: dict[str, Any] = Field(default_factory=dict)
    response: dict[str, Any] | None = None
    error: ToolError | None = None
    timestamp: datetime


class GetEmployeeProfileRequest(BaseModel):
    employee_id: str = Field(min_length=3)


class GetEmployeeProfileResponse(BaseModel):
    profile: EmployeeProfile


class SearchTravelOptionsRequest(BaseModel):
    trip_request_payload: dict[str, Any]


class SearchTravelOptionsResponse(BaseModel):
    options: list[TravelOption]


class RetrieveTravelPolicyRequest(BaseModel):
    query: str
    employee_profile: EmployeeProfile


class RetrieveTravelPolicyResponse(BaseModel):
    chunks: list[PolicyChunkRef]
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class CheckBudgetRequest(BaseModel):
    employee_id: str = Field(min_length=3)
    total_price: int = Field(ge=0)
    threshold: int = Field(ge=0)


class CheckBudgetResponse(BaseModel):
    result: BudgetCheckResult


class CreateApprovalRequestRequest(BaseModel):
    request: ApprovalRequest
    confirmation_status: str | None


class CreateApprovalRequestResponse(BaseModel):
    stored: ApprovalRequest
