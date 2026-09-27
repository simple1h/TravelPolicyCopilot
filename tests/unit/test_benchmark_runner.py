from scripts.run_benchmark import run_benchmark
import pytest


def test_benchmark_runner_produces_report(monkeypatch, tmp_path):
    report_path = tmp_path / "benchmark_report.json"
    monkeypatch.setattr("scripts.run_benchmark._strict_preconditions", lambda: (True, []))
    report = run_benchmark(output_path=report_path)

    assert report["total_cases"] >= 15
    assert report["successful_cases"] >= 1
    assert 0.0 <= report["success_rate"] <= 1.0
    assert "latency_avg_ms" in report
    assert "latency_p50_ms" in report
    assert "latency_p95_ms" in report
    assert "trace_coverage_rate" in report
    assert "security_case_count" in report
    assert report["security_case_count"] >= 4
    assert isinstance(report["results"], list)
    first = report["results"][0]
    assert "trace_id" in first
    assert "trace_url" in first
    assert "trace_verified" in first
    assert "tool_events" in first
    assert "answer_text" in first
    assert "answer_hash" in first
    assert "quality_bucket" in first


def test_benchmark_runner_fails_when_strict_preconditions_missing(monkeypatch, tmp_path):
    report_path = tmp_path / "benchmark_report.json"
    monkeypatch.setattr("scripts.run_benchmark._strict_preconditions", lambda: (False, ["bad mode"]))
    with pytest.raises(RuntimeError):
        run_benchmark(output_path=report_path)
