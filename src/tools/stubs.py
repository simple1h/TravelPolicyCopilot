from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import re

from src.config import is_strict_mode, settings
from src.domain.models import BudgetCheckResult, EmployeeProfile, PolicyChunkRef, TravelOption, TripRequest
from src.persistence.repository import employee_repository
from src.tools.travel_api import TravelApiError, search_travel_options_http


_FALLBACK_EMPLOYEE_PROFILES: dict[str, EmployeeProfile] = {
    "EMP-001": EmployeeProfile(
        employee_id="EMP-001",
        employee_category="standard",
        cost_center="CC-001",
        grade="G6",
        home_city="Kazan",
        policy_tier="default",
        max_hotel_per_night=7000,
        max_flight_class="economy",
        requires_preapproval_over_amount=45000,
        accessibility_requirements=[],
    ),
    "EMP-002": EmployeeProfile(
        employee_id="EMP-002",
        employee_category="manager",
        cost_center="CC-002",
        grade="M2",
        home_city="Kazan",
        policy_tier="manager",
        max_hotel_per_night=12000,
        max_flight_class="business",
        requires_preapproval_over_amount=90000,
        accessibility_requirements=[],
    ),
    "EMP-003": EmployeeProfile(
        employee_id="EMP-003",
        employee_category="regional",
        cost_center="CC-REG-01",
        grade="G7",
        home_city="Yekaterinburg",
        policy_tier="regional",
        max_hotel_per_night=8000,
        max_flight_class="economy",
        requires_preapproval_over_amount=50000,
        accessibility_requirements=[],
    ),
    "EMP-004": EmployeeProfile(
        employee_id="EMP-004",
        employee_category="mobility",
        cost_center="CC-004",
        grade="G6",
        home_city="Kazan",
        policy_tier="mobility",
        max_hotel_per_night=9000,
        max_flight_class="premium_economy",
        requires_preapproval_over_amount=60000,
        accessibility_requirements=["wheelchair_access"],
    ),
    "EMP-005": EmployeeProfile(
        employee_id="EMP-005",
        employee_category="special_budget",
        cost_center="CC-SPECIAL-01",
        grade="G8",
        home_city="Kazan",
        policy_tier="special_budget",
        max_hotel_per_night=15000,
        max_flight_class="business",
        requires_preapproval_over_amount=150000,
        accessibility_requirements=[],
    ),
}


@dataclass
class BackendDependencyError(RuntimeError):
    code: str
    safe_message: str
    retryable: bool
    component: str

    def __str__(self) -> str:
        return self.safe_message


def get_employee_profile(employee_id: str) -> EmployeeProfile:
    try:
        profile = employee_repository.get_by_id(employee_id)
    except Exception as exc:
        if is_strict_mode():
            raise BackendDependencyError(
                code="EMPLOYEE_BACKEND_UNAVAILABLE",
                safe_message="Employee profile backend is unavailable.",
                retryable=True,
                component="postgres",
            ) from exc
        profile = None
    if profile is not None:
        return profile
    if is_strict_mode():
        raise LookupError(f"Employee profile is not available for employee_id={employee_id}.")
    fallback = _FALLBACK_EMPLOYEE_PROFILES.get(employee_id)
    if fallback is None:
        raise LookupError(f"Employee profile is not available for employee_id={employee_id}.")
    return fallback


def search_travel_options(trip_request: TripRequest | None = None) -> list[TravelOption]:
    def _fallback() -> list[TravelOption]:
        now = datetime.now(UTC)
        return [
            TravelOption(
                option_id="OPT-001",
                transport_type="flight",
                outbound_departure=now,
                outbound_arrival=now,
                inbound_departure=now,
                inbound_arrival=now,
                carrier="Mock Air",
                fare_class="economy",
                hotel_name="Mock Hotel",
                hotel_nightly_price=5000,
                total_price=32000,
                segments=1,
                refundable=False,
            )
        ]

    if trip_request is None:
        if is_strict_mode():
            raise BackendDependencyError(
                code="TRAVEL_REQUEST_INVALID",
                safe_message="Trip request is required before searching travel options.",
                retryable=False,
                component="travel_api",
            )
        return _fallback()

    scenario_mode = "normal"
    if trip_request.comment:
        match = re.search(r"\[scenario=(normal|empty|error_4xx|error_5xx|timeout)\]", trip_request.comment.lower())
        if match:
            scenario_mode = match.group(1)

    try:
        options = search_travel_options_http(trip_request, scenario_mode=scenario_mode)
        if scenario_mode == "normal":
            return options
        return options
    except TravelApiError as exc:
        if is_strict_mode():
            raise BackendDependencyError(
                code=exc.code,
                safe_message=exc.message,
                retryable=exc.retryable,
                component="travel_api",
            ) from exc
        if scenario_mode != "normal":
            return []
        return _fallback()


def retrieve_travel_policy(query: str = "", employee_context: dict | None = None) -> list[PolicyChunkRef]:
    _ = (query, employee_context)
    return [
        PolicyChunkRef(
            document_id="POLICY-001",
            section_id="SEC-TRANSPORT-1",
            source_path="data/policies/transport.md",
            score=0.88,
        )
    ]


def check_budget(total_price: int, threshold: int) -> BudgetCheckResult:
    if total_price <= threshold:
        return BudgetCheckResult(
            verdict="within_limit",
            total_price=total_price,
            threshold=threshold,
            overage=0,
        )

    overage = total_price - threshold
    return BudgetCheckResult(
        verdict="requires_approval",
        total_price=total_price,
        threshold=threshold,
        overage=overage,
    )
