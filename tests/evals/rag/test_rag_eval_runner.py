from __future__ import annotations

from src.rag.evaluation import run_golden_retrieval_eval
from src.rag.ingestion import ingest_policies


def test_rag_eval_runner_returns_hit_rate():
    ingest_policies(with_qdrant=False)
    report = run_golden_retrieval_eval()
    assert report.total_cases >= 4
    assert 0.0 <= report.hit_rate <= 1.0
    assert 0.0 <= report.source_accuracy <= 1.0
    assert 0.0 <= report.faithfulness_proxy <= 1.0
