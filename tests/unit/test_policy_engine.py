from __future__ import annotations

from datetime import UTC, datetime

from src.domain.models import EmployeeProfile, PolicyChunkRef, TravelOption
from src.policy.engine import evaluate_option


def _profile(**updates) -> EmployeeProfile:
    base = EmployeeProfile(
        employee_id="EMP-TEST",
        employee_category="standard",
        cost_center="CC-TEST",
        grade="G6",
        home_city="Kazan",
        policy_tier="default",
        max_hotel_per_night=7000,
        max_flight_class="economy",
        requires_preapproval_over_amount=45000,
        accessibility_requirements=[],
    )
    return base.model_copy(update=updates)


def _option(**updates) -> TravelOption:
    base = TravelOption(
        option_id="OPT-TEST",
        transport_type="flight",
        outbound_departure=datetime(2026, 9, 18, 8, 0, tzinfo=UTC),
        outbound_arrival=datetime(2026, 9, 18, 10, 0, tzinfo=UTC),
        inbound_departure=datetime(2026, 9, 19, 18, 0, tzinfo=UTC),
        inbound_arrival=datetime(2026, 9, 19, 20, 0, tzinfo=UTC),
        carrier="Mock Air",
        fare_class="economy",
        hotel_name="Hotel",
        hotel_nightly_price=6000,
        total_price=42000,
        segments=1,
        refundable=False,
    )
    return base.model_copy(update=updates)


def _by_rule(results):
    return {item.rule_id: item for item in results}


def test_transport_class_limit_fail_for_prohibited_class():
    profile = _profile(max_flight_class="economy")
    option = _option(fare_class="business")
    results = evaluate_option(option, [], profile)
    rule = _by_rule(results)["TRANSPORT_CLASS_LIMIT"]
    assert rule.verdict == "fail"


def test_hotel_limit_requires_approval_when_exceeded():
    profile = _profile(max_hotel_per_night=7000)
    option = _option(hotel_nightly_price=12000)
    results = evaluate_option(option, [], profile)
    rule = _by_rule(results)["HOTEL_LIMIT_STANDARD"]
    assert rule.verdict == "requires_approval"


def test_preapproval_threshold_requires_approval():
    profile = _profile(requires_preapproval_over_amount=45000)
    option = _option(total_price=61000)
    results = evaluate_option(option, [], profile)
    rule = _by_rule(results)["PREAPPROVAL_OVER_THRESHOLD"]
    assert rule.verdict == "requires_approval"


def test_night_departure_requires_approval_for_standard_profile():
    profile = _profile()
    option = _option(outbound_departure=datetime(2026, 9, 18, 2, 0, tzinfo=UTC))
    results = evaluate_option(option, [], profile)
    rule = _by_rule(results)["NIGHT_DEPARTURE_RESTRICTION"]
    assert rule.verdict == "requires_approval"


def test_mobility_exception_allows_night_departure():
    profile = _profile(
        employee_category="mobility",
        max_flight_class="premium_economy",
        accessibility_requirements=["wheelchair_access"],
    )
    option = _option(outbound_departure=datetime(2026, 9, 18, 2, 0, tzinfo=UTC))
    results = evaluate_option(option, [PolicyChunkRef(document_id="D", section_id="S", source_path="p", score=0.9)], profile)
    rules = _by_rule(results)
    assert rules["NIGHT_DEPARTURE_RESTRICTION"].verdict == "pass"
    assert rules["SPECIAL_MOBILITY_EXCEPTION"].verdict == "pass"
