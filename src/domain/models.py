from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


EmployeeCategory = Literal["standard", "manager", "regional", "mobility", "special_budget"]
PolicyVerdict = Literal["pass", "fail", "warning", "requires_approval"]
BudgetVerdict = Literal["within_limit", "over_limit", "requires_approval"]
RequestStatus = Literal["draft_created", "pending_approval", "approved", "rejected"]


class TripRequest(BaseModel):
    origin_city: str | None = None
    destination_city: str | None = None
    departure_date: date | None = None
    return_date: date | None = None
    meeting_datetime: datetime | None = None
    earliest_departure_hour: int | None = Field(default=None, ge=0, le=23)
    latest_arrival_hour: int | None = Field(default=None, ge=0, le=23)
    max_segments: int | None = Field(default=None, ge=1, le=4)
    disallowed_time_windows: list[str] = Field(default_factory=list)
    transport_preferences: list[str] = Field(default_factory=list)
    hotel_required: bool = True
    comment: str | None = None

    @model_validator(mode="after")
    def validate_dates(self) -> "TripRequest":
        if self.departure_date and self.return_date and self.return_date < self.departure_date:
            raise ValueError("return_date must be greater than or equal to departure_date")
        return self


class EmployeeProfile(BaseModel):
    employee_id: str
    employee_category: EmployeeCategory
    cost_center: str
    grade: str
    home_city: str
    policy_tier: str
    max_hotel_per_night: int
    max_flight_class: Literal["economy", "premium_economy", "business"]
    requires_preapproval_over_amount: int
    accessibility_requirements: list[str] = Field(default_factory=list)


class TravelOption(BaseModel):
    option_id: str
    transport_type: Literal["flight", "train"]
    outbound_departure: datetime
    outbound_arrival: datetime
    inbound_departure: datetime
    inbound_arrival: datetime
    carrier: str
    fare_class: str
    hotel_name: str | None = None
    hotel_nightly_price: int | None = None
    currency: Literal["RUB"] = "RUB"
    total_price: int
    segments: int
    refundable: bool = False


class PolicyChunkRef(BaseModel):
    document_id: str
    section_id: str
    source_path: str
    score: float
    document_title: str | None = None
    policy_version: str | None = None
    section_title: str | None = None
    effective_from: date | None = None
    employee_category: str | None = None
    rule_type: str | None = None
    chunk_text: str | None = None


class PolicyCheckResult(BaseModel):
    rule_id: str
    verdict: PolicyVerdict
    message: str
    citations: list[PolicyChunkRef] = Field(default_factory=list)


class BudgetCheckResult(BaseModel):
    verdict: BudgetVerdict
    total_price: int
    threshold: int
    overage: int
    currency: Literal["RUB"] = "RUB"


class Recommendation(BaseModel):
    option_id: str
    score_total: float
    score_price: float
    score_schedule: float
    score_preference: float
    score_policy: float
    is_recommended: bool
    explanation: str
    policy_summary: str
    budget_summary: str
    citations: list[PolicyChunkRef] = Field(default_factory=list)


class ApprovalRequest(BaseModel):
    request_id: str
    employee_id: str
    trip_request: TripRequest
    selected_option_id: str
    total_price: int
    budget_result: BudgetCheckResult
    policy_results: list[PolicyCheckResult]
    citations: list[PolicyChunkRef]
    status: RequestStatus
    created_at: datetime
    idempotency_key: str


class AgentError(BaseModel):
    code: str
    message: str
    retryable: bool = False
