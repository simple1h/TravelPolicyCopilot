from __future__ import annotations

from src.evals.runner import run_all_evals


def test_run_all_evals_generates_aligned_report(tmp_path, monkeypatch):
    output = tmp_path / "eval_report.json"

    class _FakeJudge:
        total_cases = 1
        passed_cases = 1
        pass_rate = 1.0
        average_score = 4.5
        average_faithfulness = 4.6
        citation_success_rate = 1.0
        judge_model = "openai/gpt-4.1-nano"
        mode = "live_llm"
        threshold_average = 4.0
        threshold_faithfulness = 4.0
        dataset_hash = "hash"
        prompt_version = "v2"
        timestamp = "2026-01-01T00:00:00+00:00"
        commit_sha = "abc"
        required_cases_total = 1
        required_cases_passed = 1
        required_cases_failed = 0
        incomplete_cases = 0
        live_cases = 1
        heuristic_cases = 0
        results = []

    class _FakeRag:
        total_cases = 1
        matched_cases = 1
        hit_rate = 1.0
        source_accuracy = 1.0
        faithfulness_proxy = 1.0
        results = []

    monkeypatch.setattr("src.evals.runner.run_llm_judge_eval", lambda **kwargs: _FakeJudge())
    monkeypatch.setattr("src.evals.runner.run_golden_retrieval_eval", lambda: _FakeRag())
    monkeypatch.setattr(
        "src.evals.runner.run_benchmark",
        lambda: {
            "overall_pass": True,
            "success_rate": 1.0,
            "tool_call_correctness": 1.0,
            "total_cases": 20,
            "fallback_used_cases": [],
            "trace_coverage_rate": 1.0,
            "trace_coverage_min": 1.0,
            "trace_unverified_cases": [],
            "results": [{"id": "case-1", "success": True, "trace_id": None}],
        },
    )

    report = run_all_evals(output_path=str(output))
    assert output.exists()
    assert report["summary"]["overall_pass"] is True
    assert "judge_meta" in report
    assert "alignment" in report
    assert report["summary"]["judge_citation_success_rate"] == 1.0
