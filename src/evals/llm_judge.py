from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import re
import hashlib
from pathlib import Path
from typing import Any

import yaml

from src.llm.gateway import LLMGatewayError, OpenAILLMGateway, get_default_llm_gateway


@dataclass
class JudgeCaseResult:
    case_id: str
    status: str
    relevance: float | None
    clarity: float | None
    completeness: float | None
    faithfulness: float | None
    average: float | None
    citation_required: bool
    citation_ok: bool
    model: str
    mode: str
    passed: bool
    required: bool
    answer_hash: str | None
    trace_id: str | None
    run_id: str | None
    error_code: str | None


@dataclass
class JudgeEvalReport:
    total_cases: int
    passed_cases: int
    pass_rate: float
    average_score: float
    average_faithfulness: float
    citation_success_rate: float
    judge_model: str
    mode: str
    threshold_average: float
    threshold_faithfulness: float
    dataset_hash: str
    prompt_version: str
    timestamp: str
    commit_sha: str | None
    required_cases_total: int
    required_cases_passed: int
    required_cases_failed: int
    incomplete_cases: int
    live_cases: int
    heuristic_cases: int
    results: list[JudgeCaseResult]


def _extract_json_payload(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise json.JSONDecodeError("No JSON object found.", text, 0)
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise json.JSONDecodeError("JSON object must be a dict.", text, 0)
    return payload


def _clamp_score(value: Any) -> float:
    score = float(value)
    if score < 1.0:
        return 1.0
    if score > 5.0:
        return 5.0
    return round(score, 2)


def _has_citation_markers(response_text: str, expected_sources: list[str]) -> bool:
    lowered = response_text.lower()
    if "источник" in lowered or "[" in response_text and "]" in response_text:
        return True
    return any(source.lower() in lowered for source in expected_sources)


def _heuristic_score(
    response_text: str,
    expected_facts: list[str],
    require_citation: bool,
    expected_sources: list[str],
) -> tuple[dict[str, float], bool]:
    text = response_text.lower()
    matched = sum(1 for fact in expected_facts if fact.lower() in text)
    coverage = matched / max(1, len(expected_facts))
    relevance = min(5.0, 2.0 + coverage * 3.0)
    clarity = 4.0 if len(response_text) >= 25 else 3.0
    completeness = min(5.0, 2.0 + coverage * 3.0)
    faithfulness = min(5.0, 2.5 + coverage * 2.5)
    citation_ok = (not require_citation) or _has_citation_markers(response_text, expected_sources)
    if require_citation and not citation_ok:
        faithfulness = max(1.0, faithfulness - 1.0)
    return {
        "relevance": round(relevance, 2),
        "clarity": round(clarity, 2),
        "completeness": round(completeness, 2),
        "faithfulness": round(faithfulness, 2),
    }, citation_ok


def _llm_score(
    *,
    answer_text: str,
    expected_facts: list[str],
    expected_sources: list[str],
    require_citation: bool,
    grounded_facts: list[str],
    citations: list[str],
    gateway: OpenAILLMGateway,
) -> tuple[dict[str, float], str, bool]:
    expected = ", ".join(expected_facts)
    expected_sources_text = ", ".join(expected_sources)
    grounded_facts_text = ", ".join(grounded_facts)
    citations_text = ", ".join(citations)
    citation_instruction = (
        "Citation is REQUIRED in answer style."
        if require_citation
        else "Citation is optional for this case."
    )
    response = gateway.complete(
        system_prompt=(
            "You are an evaluator for travel policy copilot responses. "
            "Return strict JSON object with fields: relevance, clarity, completeness, faithfulness, citation_ok. "
            "Scores must be numbers in range 1..5. citation_ok must be true/false. "
            "Scoring rubric: "
            "completeness=5 when all expected facts are present; "
            "completeness=4 when most expected facts are present; "
            "completeness<=3 when expected facts are partially missing. "
            "faithfulness=5 when answer does not contradict grounded facts or citations."
        ),
        user_prompt=(
            "Evaluate assistant answer for the travel policy copilot.\n"
            f"Expected facts: {expected}\n"
            f"Expected sources: {expected_sources_text}\n"
            f"Grounded facts from this run: {grounded_facts_text}\n"
            f"Citations from this run: {citations_text}\n"
            f"{citation_instruction}\n"
            f"Answer: {answer_text}\n"
            "Return JSON only. No prose."
        ),
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    payload = _extract_json_payload(response.content)
    scores = {
        "relevance": _clamp_score(payload["relevance"]),
        "clarity": _clamp_score(payload["clarity"]),
        "completeness": _clamp_score(payload["completeness"]),
        "faithfulness": _clamp_score(payload["faithfulness"]),
    }
    citation_ok = bool(payload.get("citation_ok", not require_citation))
    return scores, response.model, citation_ok


def _dataset_hash(dataset_path: str) -> str:
    raw = Path(dataset_path).read_text(encoding="utf-8")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _extract_grounded_facts(benchmark_case: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    tool_calls = benchmark_case.get("tool_calls", []) or []
    if "retrieve_travel_policy" in tool_calls:
        facts.append("policy_retrieval_executed")
    if "check_budget" in tool_calls:
        facts.append("budget_check_executed")
    if benchmark_case.get("approval_request_id"):
        facts.append("approval_request_created")
    if benchmark_case.get("final_step"):
        facts.append(f"final_step={benchmark_case.get('final_step')}")
    return facts


def _extract_citations(benchmark_case: dict[str, Any]) -> list[str]:
    citations: set[str] = set()
    for event in benchmark_case.get("tool_events", []) or []:
        if event.get("tool_name") != "retrieve_travel_policy":
            continue
        chunks = ((event.get("response") or {}).get("chunks") or [])
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            document_id = str(chunk.get("document_id") or "").strip()
            section_id = str(chunk.get("section_id") or "").strip()
            if document_id and section_id:
                citations.add(f"{document_id}:{section_id}")
            elif document_id:
                citations.add(document_id)
    return sorted(citations)


def run_llm_judge_eval(
    *,
    dataset_path: str = "data/evals/llm_judge_cases.yaml",
    threshold_average: float = 4.0,
    threshold_faithfulness: float = 4.0,
    benchmark_report: dict[str, Any] | None = None,
    require_live_llm: bool = True,
    gateway: OpenAILLMGateway | None = None,
) -> JudgeEvalReport:
    cases = yaml.safe_load(Path(dataset_path).read_text(encoding="utf-8")).get("cases", [])
    judge_gateway = gateway or get_default_llm_gateway()
    benchmark_results = {item["id"]: item for item in (benchmark_report or {}).get("results", [])}
    run_id = (benchmark_report or {}).get("run_id")
    commit_sha = (benchmark_report or {}).get("commit_sha")

    results: list[JudgeCaseResult] = []
    llm_case_count = 0
    fallback_case_count = 0
    incomplete_case_count = 0
    judge_models: set[str] = set()

    for case in cases:
        case_id = str(case.get("case_id", ""))
        required_case = bool(case.get("required", True))
        require_citation = bool(case.get("require_citation", False))
        expected_sources = list(case.get("expected_sources", []))
        expected_facts = list(case.get("expected_facts", []))
        benchmark_case = benchmark_results.get(case_id)
        if benchmark_case is None:
            incomplete_case_count += 1
            results.append(
                JudgeCaseResult(
                    case_id=case_id,
                    status="incomplete",
                    relevance=None,
                    clarity=None,
                    completeness=None,
                    faithfulness=None,
                    average=None,
                    citation_required=require_citation,
                    citation_ok=False,
                    model="unavailable",
                    mode="incomplete",
                    passed=False,
                    required=required_case,
                    answer_hash=None,
                    trace_id=None,
                    run_id=run_id,
                    error_code="BENCHMARK_CASE_NOT_FOUND",
                )
            )
            continue

        answer_text = str(benchmark_case.get("answer_text") or "")
        answer_hash = str(benchmark_case.get("answer_hash") or "")
        trace_id = benchmark_case.get("trace_id")
        if not answer_text:
            incomplete_case_count += 1
            results.append(
                JudgeCaseResult(
                    case_id=case_id,
                    status="incomplete",
                    relevance=None,
                    clarity=None,
                    completeness=None,
                    faithfulness=None,
                    average=None,
                    citation_required=require_citation,
                    citation_ok=False,
                    model="unavailable",
                    mode="incomplete",
                    passed=False,
                    required=required_case,
                    answer_hash=answer_hash or None,
                    trace_id=trace_id,
                    run_id=run_id,
                    error_code="BENCHMARK_ANSWER_MISSING",
                )
            )
            continue

        grounded_facts = _extract_grounded_facts(benchmark_case)
        citations = _extract_citations(benchmark_case)
        try:
            scores, resolved_model, citation_ok = _llm_score(
                answer_text=answer_text,
                expected_facts=expected_facts,
                expected_sources=expected_sources,
                require_citation=require_citation,
                grounded_facts=grounded_facts,
                citations=citations,
                gateway=judge_gateway,
            )
            llm_case_count += 1
            judge_models.add(resolved_model)
            mode = "live_llm"
            status = "live_llm"
            error_code = None
        except (LLMGatewayError, ValueError, KeyError, json.JSONDecodeError) as exc:
            fallback_case_count += 1
            scores, citation_ok = _heuristic_score(
                answer_text,
                expected_facts,
                require_citation,
                expected_sources,
            )
            resolved_model = "heuristic-fallback"
            mode = "heuristic_fallback"
            status = "incomplete" if require_live_llm else "heuristic_fallback"
            error_code = getattr(exc, "code", type(exc).__name__)

        average = (
            scores["relevance"] + scores["clarity"] + scores["completeness"] + scores["faithfulness"]
        ) / 4.0
        passed = (
            status == "live_llm"
            and average >= threshold_average
            and scores["faithfulness"] >= threshold_faithfulness
            and (citation_ok or not require_citation)
        )
        if status == "incomplete":
            incomplete_case_count += 1
        results.append(
            JudgeCaseResult(
                case_id=case_id,
                status=status,
                relevance=scores["relevance"],
                clarity=scores["clarity"],
                completeness=scores["completeness"],
                faithfulness=scores["faithfulness"],
                average=round(average, 3),
                citation_required=require_citation,
                citation_ok=citation_ok,
                model=resolved_model,
                mode=mode,
                passed=passed,
                required=required_case,
                answer_hash=answer_hash or None,
                trace_id=trace_id,
                run_id=run_id,
                error_code=error_code,
            )
        )

    total = len(results)
    passed_cases = sum(1 for item in results if item.passed)
    pass_rate = passed_cases / total if total else 0.0
    scored = [item for item in results if item.average is not None]
    avg_score = (sum(float(item.average) for item in scored) / len(scored)) if scored else 0.0
    faithfulness_values = [float(item.faithfulness) for item in results if item.faithfulness is not None]
    avg_faithfulness = (
        sum(faithfulness_values) / len(faithfulness_values)
        if faithfulness_values
        else 0.0
    )
    citation_required_count = sum(1 for case in cases if bool(case.get("require_citation", False)))
    citation_success = (
        sum(1 for item in results if item.citation_required and item.citation_ok) / citation_required_count
        if citation_required_count
        else 1.0
    )
    aggregate_mode = "live_llm"
    if incomplete_case_count > 0:
        aggregate_mode = "incomplete"
    elif fallback_case_count > 0:
        aggregate_mode = "heuristic_fallback"
    judge_model = "heuristic-fallback"
    if judge_models:
        judge_model = ",".join(sorted(judge_models))
    required_total = sum(1 for item in results if item.required)
    required_passed = sum(1 for item in results if item.required and item.passed)
    required_failed = required_total - required_passed
    return JudgeEvalReport(
        total_cases=total,
        passed_cases=passed_cases,
        pass_rate=round(pass_rate, 4),
        average_score=round(avg_score, 4),
        average_faithfulness=round(avg_faithfulness, 4),
        citation_success_rate=round(citation_success, 4),
        judge_model=judge_model,
        mode=aggregate_mode,
        threshold_average=threshold_average,
        threshold_faithfulness=threshold_faithfulness,
        dataset_hash=_dataset_hash(dataset_path),
        prompt_version="judge_prompt_v2_live_answers",
        timestamp=datetime.now(UTC).isoformat(),
        commit_sha=commit_sha,
        required_cases_total=required_total,
        required_cases_passed=required_passed,
        required_cases_failed=required_failed,
        incomplete_cases=incomplete_case_count,
        live_cases=llm_case_count,
        heuristic_cases=fallback_case_count,
        results=results,
    )
