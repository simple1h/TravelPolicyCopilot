from __future__ import annotations

from src.rag.evaluation import run_golden_retrieval_eval
from src.rag.ingestion import ingest_policies


def test_golden_retrieval_eval_has_nonzero_hit_rate():
    ingest_policies(with_qdrant=False)
    report = run_golden_retrieval_eval()
    assert report.total_cases >= 4
    assert 0.0 <= report.hit_rate <= 1.0
    assert report.hit_rate >= 0.5
