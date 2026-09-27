from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.domain.models import EmployeeProfile
from src.rag.retrieval import retrieve_policy_chunks


@dataclass
class RetrievalEvalCaseResult:
    case_id: str
    matched: bool
    expected_document_ids: list[str]
    actual_document_ids: list[str]


@dataclass
class RetrievalEvalReport:
    total_cases: int
    matched_cases: int
    hit_rate: float
    source_accuracy: float
    faithfulness_proxy: float
    results: list[RetrievalEvalCaseResult]


def _profile(category: str) -> EmployeeProfile:
    return EmployeeProfile(
        employee_id=f"EMP-EVAL-{category}",
        employee_category=category,  # type: ignore[arg-type]
        cost_center="CC-EVAL",
        grade="G6",
        home_city="Kazan",
        policy_tier="default",
        max_hotel_per_night=7000,
        max_flight_class="economy",
        requires_preapproval_over_amount=45000,
        accessibility_requirements=["wheelchair_access"] if category == "mobility" else [],
    )


def run_golden_retrieval_eval(dataset_path: str = "data/rag/golden_retrieval_cases.yaml") -> RetrievalEvalReport:
    data = yaml.safe_load(Path(dataset_path).read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    results: list[RetrievalEvalCaseResult] = []
    source_accuracy_sum = 0.0
    faithfulness_proxy_sum = 0.0

    for case in cases:
        profile = _profile(case["employee_category"])
        chunks = retrieve_policy_chunks(case["query"], profile)
        actual_ids = [chunk.document_id for chunk in chunks]
        expected_ids = list(case.get("expected_document_ids", []))
        matched_expected = sum(1 for doc_id in expected_ids if doc_id in actual_ids)
        source_accuracy_sum += 1.0 if actual_ids and actual_ids[0] in expected_ids else 0.0
        faithfulness_proxy_sum += matched_expected / max(1, len(expected_ids))
        matched = all(doc_id in actual_ids for doc_id in expected_ids)
        results.append(
            RetrievalEvalCaseResult(
                case_id=case["id"],
                matched=matched,
                expected_document_ids=expected_ids,
                actual_document_ids=actual_ids,
            )
        )

    total = len(results)
    matched_cases = sum(1 for item in results if item.matched)
    hit_rate = (matched_cases / total) if total else 0.0
    return RetrievalEvalReport(
        total_cases=total,
        matched_cases=matched_cases,
        hit_rate=hit_rate,
        source_accuracy=(source_accuracy_sum / total) if total else 0.0,
        faithfulness_proxy=(faithfulness_proxy_sum / total) if total else 0.0,
        results=results,
    )
