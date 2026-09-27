from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.domain.models import AgentError, Recommendation, TripRequest


class AgentMessageRequest(BaseModel):
    employee_id: str = Field(min_length=3)
    message: str = Field(min_length=1, max_length=4000)


class AgentMessageResponse(BaseModel):
    trace_id: str
    current_step: str
    missing_fields: list[str]
    answer: str
    travel_search_outcome: str | None = None
    travel_search_error_code: str | None = None
    recovery_advice: str | None = None
    backend_status: str = "healthy"
    degraded_components: list[str] = Field(default_factory=list)


class AgentSessionStartRequest(BaseModel):
    employee_id: str = Field(min_length=3)
    message: str = Field(min_length=1, max_length=4000)


class TripRequestPatch(BaseModel):
    origin_city: str | None = None
    destination_city: str | None = None
    departure_date: date | None = None
    return_date: date | None = None
    meeting_datetime: datetime | None = None
    earliest_departure_hour: int | None = Field(default=None, ge=0, le=23)
    latest_arrival_hour: int | None = Field(default=None, ge=0, le=23)
    max_segments: int | None = Field(default=None, ge=1, le=4)
    disallowed_time_windows: list[str] | None = None
    transport_preferences: list[str] | None = None
    hotel_required: bool | None = None
    comment: str | None = None


AgentActionType = Literal[
    "submit_clarification",
    "update_constraints",
    "retry_search",
    "cancel",
    "select_option",
    "confirm",
    "reject",
    "change_selection",
]


class AgentSessionActionRequest(BaseModel):
    employee_id: str = Field(min_length=3)
    action: AgentActionType
    trip_request_patch: TripRequestPatch | None = None
    selected_option_id: str | None = None


class AgentSessionResponse(BaseModel):
    session_id: str
    employee_id: str
    trace_id: str
    current_step: str
    missing_fields: list[str]
    answer: str
    clarification_question: str | None = None
    trip_request: TripRequest | None = None
    recommendations: list[Recommendation] = Field(default_factory=list)
    selected_option_id: str | None = None
    approval_request_id: str | None = None
    available_actions: list[AgentActionType] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    tool_events: list[dict[str, Any]] = Field(default_factory=list)
    audit_events: list[dict[str, Any]] = Field(default_factory=list)
    travel_search_outcome: str | None = None
    travel_search_error_code: str | None = None
    recovery_advice: str | None = None
    rejected_options: list[dict[str, Any]] = Field(default_factory=list)
    route_transitions: list[dict[str, Any]] = Field(default_factory=list)
    backend_status: str = "healthy"
    degraded_components: list[str] = Field(default_factory=list)
    error: AgentError | None = None
