from __future__ import annotations

from src.domain.models import EmployeeProfile, PolicyCheckResult, PolicyChunkRef, TravelOption

_CLASS_ORDER = {
    "economy": 0,
    "premium_economy": 1,
    "business": 2,
}


def _is_mobility_exception_applicable(profile: EmployeeProfile | None) -> bool:
    if profile is None:
        return False
    if profile.employee_category != "mobility":
        return False
    return "wheelchair_access" in profile.accessibility_requirements


def _transport_class_rule(
    option: TravelOption,
    refs: list[PolicyChunkRef],
    profile: EmployeeProfile | None,
) -> PolicyCheckResult:
    if profile is None:
        return PolicyCheckResult(
            rule_id="TRANSPORT_CLASS_LIMIT",
            verdict="warning",
            message="Employee profile is unavailable, transport class check is partial.",
            citations=refs,
        )

    option_rank = _CLASS_ORDER.get(option.fare_class)
    allowed_rank = _CLASS_ORDER.get(profile.max_flight_class)
    if option_rank is None or allowed_rank is None:
        return PolicyCheckResult(
            rule_id="TRANSPORT_CLASS_LIMIT",
            verdict="fail",
            message=f"Unknown fare class '{option.fare_class}' is not allowed.",
            citations=refs,
        )

    if option_rank > allowed_rank:
        return PolicyCheckResult(
            rule_id="TRANSPORT_CLASS_LIMIT",
            verdict="fail",
            message=(
                f"Fare class '{option.fare_class}' exceeds allowed class "
                f"'{profile.max_flight_class}'."
            ),
            citations=refs,
        )

    return PolicyCheckResult(
        rule_id="TRANSPORT_CLASS_LIMIT",
        verdict="pass",
        message="Fare class is allowed for employee category.",
        citations=refs,
    )


def _hotel_limit_rule(
    option: TravelOption,
    refs: list[PolicyChunkRef],
    profile: EmployeeProfile | None,
) -> PolicyCheckResult:
    rule_suffix = (profile.employee_category if profile is not None else "standard").upper()
    rule_id = f"HOTEL_LIMIT_{rule_suffix}"

    if profile is None:
        return PolicyCheckResult(
            rule_id=rule_id,
            verdict="warning",
            message="Employee profile is unavailable, hotel limit check is partial.",
            citations=refs,
        )
    if option.hotel_nightly_price is None:
        return PolicyCheckResult(
            rule_id=rule_id,
            verdict="pass",
            message="Hotel is not required for this option.",
            citations=refs,
        )
    if option.hotel_nightly_price > profile.max_hotel_per_night:
        return PolicyCheckResult(
            rule_id=rule_id,
            verdict="requires_approval",
            message=(
                f"Hotel nightly price {option.hotel_nightly_price} exceeds "
                f"limit {profile.max_hotel_per_night}."
            ),
            citations=refs,
        )
    return PolicyCheckResult(
        rule_id=rule_id,
        verdict="pass",
        message="Hotel nightly price is within employee limit.",
        citations=refs,
    )


def _preapproval_rule(
    option: TravelOption,
    refs: list[PolicyChunkRef],
    profile: EmployeeProfile | None,
) -> PolicyCheckResult:
    if profile is None:
        return PolicyCheckResult(
            rule_id="PREAPPROVAL_OVER_THRESHOLD",
            verdict="warning",
            message="Employee profile is unavailable, preapproval check is partial.",
            citations=refs,
        )
    if option.total_price > profile.requires_preapproval_over_amount:
        return PolicyCheckResult(
            rule_id="PREAPPROVAL_OVER_THRESHOLD",
            verdict="requires_approval",
            message=(
                f"Total price {option.total_price} exceeds preapproval threshold "
                f"{profile.requires_preapproval_over_amount}."
            ),
            citations=refs,
        )
    return PolicyCheckResult(
        rule_id="PREAPPROVAL_OVER_THRESHOLD",
        verdict="pass",
        message="Total price is within preapproval threshold.",
        citations=refs,
    )


def _night_departure_rule(
    option: TravelOption,
    refs: list[PolicyChunkRef],
    profile: EmployeeProfile | None,
) -> PolicyCheckResult:
    is_night_departure = option.outbound_departure.hour < 6 or option.outbound_departure.hour >= 23
    if not is_night_departure:
        return PolicyCheckResult(
            rule_id="NIGHT_DEPARTURE_RESTRICTION",
            verdict="pass",
            message="Departure time is within regular hours.",
            citations=refs,
        )

    if _is_mobility_exception_applicable(profile):
        return PolicyCheckResult(
            rule_id="NIGHT_DEPARTURE_RESTRICTION",
            verdict="pass",
            message="Night departure allowed due to mobility exception.",
            citations=refs,
        )

    return PolicyCheckResult(
        rule_id="NIGHT_DEPARTURE_RESTRICTION",
        verdict="requires_approval",
        message="Night departure requires additional approval.",
        citations=refs,
    )


def _mobility_exception_rule(
    refs: list[PolicyChunkRef],
    profile: EmployeeProfile | None,
) -> PolicyCheckResult:
    if _is_mobility_exception_applicable(profile):
        return PolicyCheckResult(
            rule_id="SPECIAL_MOBILITY_EXCEPTION",
            verdict="pass",
            message="Mobility exception is active for this employee profile.",
            citations=refs,
        )
    return PolicyCheckResult(
        rule_id="SPECIAL_MOBILITY_EXCEPTION",
        verdict="warning",
        message="Mobility exception is not applicable.",
        citations=refs,
    )


def evaluate_option(
    option: TravelOption,
    refs: list[PolicyChunkRef],
    profile: EmployeeProfile | None = None,
) -> list[PolicyCheckResult]:
    """Apply deterministic policy rules to a single travel option."""
    return [
        _transport_class_rule(option, refs, profile),
        _hotel_limit_rule(option, refs, profile),
        _preapproval_rule(option, refs, profile),
        _night_departure_rule(option, refs, profile),
        _mobility_exception_rule(refs, profile),
    ]
