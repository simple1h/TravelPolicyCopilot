from __future__ import annotations

from src.evals.llm_judge import LLMGatewayError
from src.evals.llm_judge import run_llm_judge_eval


def test_llm_judge_runner_produces_report_without_live_llm(monkeypatch):
    class _FakeGateway:
        model = "openai/gpt-4.1-nano"

        def complete(self, **kwargs):
            raise LLMGatewayError(code="LLM_PROVIDER_ERROR", message="offline", retryable=True)

    monkeypatch.setattr("src.evals.llm_judge.get_default_llm_gateway", lambda: _FakeGateway())
    benchmark_report = {
        "run_id": "run-1",
        "commit_sha": "abc123",
        "results": [
            {
                "id": "happy_path_recommendations_ready",
                "answer_text": "Найдено вариантов: 2. Рекомендуем OPT-001.",
                "answer_hash": "hash-1",
                "trace_id": "f" * 32,
                "tool_calls": ["retrieve_travel_policy"],
                "tool_events": [],
                "approval_request_id": None,
                "final_step": "present_recommendations",
            }
        ],
    }
    report = run_llm_judge_eval(benchmark_report=benchmark_report, require_live_llm=True)
    assert report.total_cases >= 4
    assert 0.0 <= report.pass_rate <= 1.0
    assert 0.0 <= report.citation_success_rate <= 1.0
    assert report.mode == "incomplete"
    assert report.incomplete_cases >= 1
