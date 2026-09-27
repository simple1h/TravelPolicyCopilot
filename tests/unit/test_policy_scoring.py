from __future__ import annotations

from datetime import UTC, date, datetime

from src.domain.models import BudgetCheckResult, PolicyCheckResult, TravelOption, TripRequest
from src.policy.scoring import rank_travel_options


def _option(option_id: str, *, fare_class: str, total_price: int, segments: int, dep_hour: int) -> TravelOption:
    return TravelOption(
        option_id=option_id,
        transport_type="flight",
        outbound_departure=datetime(2026, 9, 18, dep_hour, 0, tzinfo=UTC),
        outbound_arrival=datetime(2026, 9, 18, dep_hour + 2, 0, tzinfo=UTC),
        inbound_departure=datetime(2026, 9, 19, 18, 0, tzinfo=UTC),
        inbound_arrival=datetime(2026, 9, 19, 20, 0, tzinfo=UTC),
        carrier="Mock Air",
        fare_class=fare_class,
        hotel_name="Hotel",
        hotel_nightly_price=6000,
        total_price=total_price,
        segments=segments,
        refundable=False,
    )


def _budget(total_price: int, threshold: int = 45000) -> BudgetCheckResult:
    overage = max(0, total_price - threshold)
    verdict = "within_limit" if overage == 0 else "requires_approval"
    return BudgetCheckResult(
        verdict=verdict,
        total_price=total_price,
        threshold=threshold,
        overage=overage,
    )


def test_policy_fail_option_is_not_primary_recommendation():
    allowed = _option("OPT-ALLOWED", fare_class="economy", total_price=34000, segments=1, dep_hour=10)
    prohibited = _option("OPT-FAIL", fare_class="business", total_price=25000, segments=1, dep_hour=10)

    policy_results_by_option = {
        "OPT-ALLOWED": [
            PolicyCheckResult(rule_id="TRANSPORT_CLASS_LIMIT", verdict="pass", message="ok", citations=[]),
        ],
        "OPT-FAIL": [
            PolicyCheckResult(rule_id="TRANSPORT_CLASS_LIMIT", verdict="fail", message="too high", citations=[]),
        ],
    }
    budget_results_by_option = {
        "OPT-ALLOWED": _budget(allowed.total_price),
        "OPT-FAIL": _budget(prohibited.total_price),
    }
    trip_request = TripRequest(
        origin_city="Казань",
        destination_city="Москва",
        departure_date=date(2026, 9, 18),
        return_date=date(2026, 9, 19),
        transport_preferences=["flight"],
    )

    recommendations = rank_travel_options(
        [allowed, prohibited],
        policy_results_by_option,
        budget_results_by_option,
        [],
        trip_request=trip_request,
    )

    recommended = next(item for item in recommendations if item.is_recommended)
    assert recommended.option_id == "OPT-ALLOWED"
    for item in recommendations:
        assert 0.0 <= item.score_price <= 1.0
        assert 0.0 <= item.score_schedule <= 1.0
        assert 0.0 <= item.score_preference <= 1.0
        assert 0.0 <= item.score_policy <= 1.0
        assert 0.0 <= item.score_total <= 1.0


def test_all_policy_fail_options_have_no_primary_recommendation():
    first = _option("OPT-FAIL-1", fare_class="business", total_price=31000, segments=1, dep_hour=10)
    second = _option("OPT-FAIL-2", fare_class="business", total_price=32000, segments=1, dep_hour=12)
    policy_results_by_option = {
        "OPT-FAIL-1": [PolicyCheckResult(rule_id="TRANSPORT_CLASS_LIMIT", verdict="fail", message="blocked", citations=[])],
        "OPT-FAIL-2": [PolicyCheckResult(rule_id="TRANSPORT_CLASS_LIMIT", verdict="fail", message="blocked", citations=[])],
    }
    budget_results_by_option = {
        "OPT-FAIL-1": _budget(first.total_price),
        "OPT-FAIL-2": _budget(second.total_price),
    }
    recommendations = rank_travel_options(
        [first, second],
        policy_results_by_option,
        budget_results_by_option,
        [],
        trip_request=TripRequest(
            origin_city="Казань",
            destination_city="Москва",
            departure_date=date(2026, 9, 18),
            return_date=date(2026, 9, 19),
        ),
    )
    assert recommendations
    assert not any(item.is_recommended for item in recommendations)
