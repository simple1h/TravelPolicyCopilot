from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from scripts.run_benchmark import run_benchmark
from src.evals.llm_judge import run_llm_judge_eval
from src.observability.tracing import create_langfuse_score
from src.rag.evaluation import run_golden_retrieval_eval


def run_all_evals(output_path: str = "data/evals/last_eval_report.json") -> dict[str, Any]:
    benchmark_report = run_benchmark()
    rag_report = run_golden_retrieval_eval()
    judge_report = run_llm_judge_eval(benchmark_report=benchmark_report, require_live_llm=True)

    benchmark_results = {item["id"]: item for item in benchmark_report.get("results", [])}
    aligned_cases: list[dict[str, Any]] = []
    judge_scores_exported = 0
    for item in judge_report.results:
        deterministic = benchmark_results.get(item.case_id)
        trace_id = deterministic.get("trace_id") if deterministic else None
        common_meta = {
            "evaluator": "travelcopilot-judge",
            "case_id": item.case_id,
            "mode": item.mode,
            "model": item.model,
            "status": item.status,
            "answer_hash": item.answer_hash,
            "run_id": item.run_id,
        }
        if trace_id:
            judge_scores_exported += int(
                create_langfuse_score(
                    name="travelcopilot.judge.average",
                    value=item.average,
                    trace_id=trace_id,
                    data_type="NUMERIC",
                    metadata=common_meta,
                )
            )
            judge_scores_exported += int(
                create_langfuse_score(
                    name="travelcopilot.judge.faithfulness",
                    value=item.faithfulness,
                    trace_id=trace_id,
                    data_type="NUMERIC",
                    metadata=common_meta,
                )
            )
            judge_scores_exported += int(
                create_langfuse_score(
                    name="travelcopilot.judge.pass",
                    value=1.0 if item.passed else 0.0,
                    trace_id=trace_id,
                    data_type="NUMERIC",
                    metadata=common_meta,
                )
            )
        aligned_cases.append(
            {
                "case_id": item.case_id,
                "deterministic_success": deterministic.get("success") if deterministic else None,
                "judge_passed": item.passed,
                "judge_status": item.status,
                "judge_average": item.average,
                "judge_faithfulness": item.faithfulness,
                "answer_hash": item.answer_hash,
                "trace_id": item.trace_id,
                "run_id": item.run_id,
            }
        )

    rag_hit_rate_min = 0.90
    rag_source_accuracy_min = 0.60
    rag_faithfulness_proxy_min = 0.9
    judge_citation_success_rate_min = 0.85

    rag_pass = (
        rag_report.hit_rate >= rag_hit_rate_min
        and rag_report.source_accuracy >= rag_source_accuracy_min
        and rag_report.faithfulness_proxy >= rag_faithfulness_proxy_min
    )
    judge_pass = (
        judge_report.mode == "live_llm"
        and judge_report.incomplete_cases == 0
        and judge_report.average_score >= judge_report.threshold_average
        and judge_report.average_faithfulness >= judge_report.threshold_faithfulness
        and judge_report.citation_success_rate >= judge_citation_success_rate_min
        and judge_report.required_cases_failed == 0
    )
    deterministic_pass = bool(benchmark_report.get("overall_pass", False))
    benchmark_success_rate_min = 0.90
    benchmark_pass = (
        deterministic_pass
        and float(benchmark_report.get("success_rate", 0.0)) >= benchmark_success_rate_min
        and float(benchmark_report.get("tool_call_correctness", 0.0)) >= 0.95
        and int(benchmark_report.get("total_cases", 0)) >= 20
        and not benchmark_report.get("fallback_used_cases", [])
        and float(benchmark_report.get("trace_coverage_rate", 0.0)) >= float(benchmark_report.get("trace_coverage_min", 1.0))
        and not benchmark_report.get("trace_unverified_cases", [])
    )
    overall_pass = bool(benchmark_pass and rag_pass and judge_pass)

    report = {
        "summary": {
            "deterministic_success_rate": benchmark_report.get("success_rate"),
            "rag_hit_rate": round(rag_report.hit_rate, 4),
            "rag_source_accuracy": round(rag_report.source_accuracy, 4),
            "rag_faithfulness_proxy": round(rag_report.faithfulness_proxy, 4),
            "judge_pass_rate": judge_report.pass_rate,
            "judge_average_score": judge_report.average_score,
            "judge_average_faithfulness": judge_report.average_faithfulness,
            "judge_citation_success_rate": judge_report.citation_success_rate,
            "judge_mode": judge_report.mode,
            "judge_incomplete_cases": judge_report.incomplete_cases,
            "deterministic_pass": benchmark_pass,
            "rag_pass": rag_pass,
            "judge_pass": judge_pass,
            "overall_pass": overall_pass,
        },
        "thresholds": {
            "benchmark_success_rate_min": benchmark_success_rate_min,
            "rag_hit_rate_min": rag_hit_rate_min,
            "rag_source_accuracy_min": rag_source_accuracy_min,
            "rag_faithfulness_proxy_min": rag_faithfulness_proxy_min,
            "judge_average_min": judge_report.threshold_average,
            "judge_faithfulness_min": judge_report.threshold_faithfulness,
            "judge_citation_success_rate_min": judge_citation_success_rate_min,
            "trace_coverage_min": float(benchmark_report.get("trace_coverage_min", 1.0)),
        },
        "judge_meta": {
            "model": judge_report.judge_model,
            "mode": judge_report.mode,
            "dataset_hash": judge_report.dataset_hash,
            "prompt_version": judge_report.prompt_version,
            "timestamp": judge_report.timestamp,
            "commit_sha": judge_report.commit_sha,
            "required_cases_total": judge_report.required_cases_total,
            "required_cases_passed": judge_report.required_cases_passed,
            "required_cases_failed": judge_report.required_cases_failed,
            "incomplete_cases": judge_report.incomplete_cases,
            "live_cases": judge_report.live_cases,
            "heuristic_cases": judge_report.heuristic_cases,
            "scores_exported": judge_scores_exported,
        },
        "alignment": aligned_cases,
        "deterministic": benchmark_report,
        "rag": {
            "total_cases": rag_report.total_cases,
            "matched_cases": rag_report.matched_cases,
            "hit_rate": round(rag_report.hit_rate, 4),
            "source_accuracy": round(rag_report.source_accuracy, 4),
            "faithfulness_proxy": round(rag_report.faithfulness_proxy, 4),
            "results": [asdict(item) for item in rag_report.results],
        },
        "llm_judge": {
            "total_cases": judge_report.total_cases,
            "passed_cases": judge_report.passed_cases,
            "pass_rate": judge_report.pass_rate,
            "average_score": judge_report.average_score,
            "average_faithfulness": judge_report.average_faithfulness,
            "citation_success_rate": judge_report.citation_success_rate,
            "dataset_hash": judge_report.dataset_hash,
            "prompt_version": judge_report.prompt_version,
            "timestamp": judge_report.timestamp,
            "commit_sha": judge_report.commit_sha,
            "required_cases_total": judge_report.required_cases_total,
            "required_cases_passed": judge_report.required_cases_passed,
            "required_cases_failed": judge_report.required_cases_failed,
            "incomplete_cases": judge_report.incomplete_cases,
            "live_cases": judge_report.live_cases,
            "heuristic_cases": judge_report.heuristic_cases,
            "results": [asdict(item) for item in judge_report.results],
        },
    }

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
