from __future__ import annotations

from src.domain.models import (
    BudgetCheckResult,
    PolicyCheckResult,
    PolicyChunkRef,
    Recommendation,
    TravelOption,
    TripRequest,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _price_score(option: TravelOption, options: list[TravelOption]) -> float:
    prices = [item.total_price for item in options]
    min_price = min(prices)
    max_price = max(prices)
    if min_price == max_price:
        return 1.0
    return _clamp01(1.0 - (option.total_price - min_price) / (max_price - min_price))


def _schedule_score(option: TravelOption) -> float:
    segment_score = 1.0 if option.segments == 1 else max(0.5, 1.0 - (option.segments - 1) * 0.2)
    travel_duration_hours = (option.outbound_arrival - option.outbound_departure).total_seconds() / 3600
    duration_score = _clamp01(1.0 - max(0.0, travel_duration_hours - 2.0) / 10.0)
    night_penalty = 0.25 if option.outbound_departure.hour < 6 or option.outbound_departure.hour >= 23 else 0.0
    return _clamp01(0.6 * segment_score + 0.4 * duration_score - night_penalty)


def _preference_score(option: TravelOption, trip_request: TripRequest | None) -> float:
    if trip_request is None:
        return 0.75

    score = 0.7
    if trip_request.transport_preferences:
        if option.transport_type in trip_request.transport_preferences:
            score += 0.2
        else:
            score -= 0.25

    if option.outbound_departure.hour < 6 or option.outbound_departure.hour >= 23:
        score -= 0.15
    else:
        score += 0.05

    return _clamp01(score)


def _policy_score(results: list[PolicyCheckResult]) -> float:
    if any(result.verdict == "fail" for result in results):
        return 0.0
    if any(result.verdict == "requires_approval" for result in results):
        return 0.45
    if any(result.verdict == "warning" for result in results):
        return 0.8
    return 1.0


def _budget_summary(result: BudgetCheckResult) -> str:
    if result.verdict == "within_limit":
        return f"В пределах лимита ({result.total_price} / {result.threshold} RUB)."
    return f"Требуется согласование: превышение на {result.overage} RUB."


def _policy_summary(results: list[PolicyCheckResult]) -> str:
    if not results:
        return "Политика не проверялась."
    if any(result.verdict == "fail" for result in results):
        return "Есть нарушения корпоративной политики."
    if any(result.verdict == "requires_approval" for result in results):
        return "Требуется дополнительное согласование по политике."
    return "Политика соблюдена."


def rank_travel_options(
    options: list[TravelOption],
    policy_results_by_option: dict[str, list[PolicyCheckResult]],
    budget_results_by_option: dict[str, BudgetCheckResult],
    citations: list[PolicyChunkRef],
    trip_request: TripRequest | None = None,
) -> list[Recommendation]:
    recommendations: list[Recommendation] = []

    for option in options:
        policy_results = policy_results_by_option.get(option.option_id, [])
        budget_result = budget_results_by_option[option.option_id]

        price_score = _price_score(option, options)
        schedule_score = _schedule_score(option)
        preference_score = _preference_score(option, trip_request)
        policy_score = _policy_score(policy_results)

        score_total = (
            0.40 * price_score
            + 0.30 * schedule_score
            + 0.20 * preference_score
            + 0.10 * policy_score
        )
        if any(result.verdict == "fail" for result in policy_results):
            score_total = min(score_total, 0.15)
        if budget_result.verdict in {"over_limit", "requires_approval"}:
            score_total = max(0.0, score_total - 0.1)
        score_total = _clamp01(score_total)

        recommendations.append(
            Recommendation(
                option_id=option.option_id,
                score_total=round(score_total, 4),
                score_price=round(price_score, 4),
                score_schedule=round(schedule_score, 4),
                score_preference=round(preference_score, 4),
                score_policy=round(policy_score, 4),
                is_recommended=False,
                explanation=(
                    f"{option.carrier}, {option.transport_type}, "
                    f"итого {option.total_price} RUB."
                ),
                policy_summary=_policy_summary(policy_results),
                budget_summary=_budget_summary(budget_result),
                citations=citations,
            )
        )

    recommendations.sort(key=lambda item: item.score_total, reverse=True)
    if recommendations:
        best_allowed_index = None
        for index, recommendation in enumerate(recommendations):
            option_policy_results = policy_results_by_option.get(recommendation.option_id, [])
            if not any(item.verdict == "fail" for item in option_policy_results):
                best_allowed_index = index
                break
        if best_allowed_index is not None:
            top = recommendations[best_allowed_index].model_copy(update={"is_recommended": True})
            recommendations[best_allowed_index] = top

    return recommendations
