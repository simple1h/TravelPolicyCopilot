from __future__ import annotations

import pytest

from src.domain.models import EmployeeProfile
from src.rag.ingestion import ingest_policies
from src.rag.retrieval import retrieve_policy_chunks


def _profile() -> EmployeeProfile:
    return EmployeeProfile(
        employee_id="EMP-RAG-QDRANT",
        employee_category="standard",
        cost_center="CC-RAG",
        grade="G6",
        home_city="Kazan",
        policy_tier="default",
        max_hotel_per_night=7000,
        max_flight_class="economy",
        requires_preapproval_over_amount=45000,
        accessibility_requirements=[],
    )


def test_qdrant_retrieval_returns_chunks_with_metadata():
    try:
        result = ingest_policies(with_qdrant=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Qdrant is not available: {exc}")
    if not result.qdrant_upserted:
        pytest.skip(f"Qdrant ingestion was not completed: {result.qdrant_error}")

    chunks = retrieve_policy_chunks("ночные вылеты и согласование", _profile())
    assert len(chunks) >= 1
    first = chunks[0]
    assert first.document_id
    assert first.section_id
    assert first.source_path
    assert first.chunk_text
    assert first.rule_type is not None
    assert first.employee_category in {"all", "standard"}
    assert first.score >= 0.0
