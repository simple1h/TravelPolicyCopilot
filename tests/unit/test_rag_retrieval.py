from __future__ import annotations

import types

from src.domain.models import EmployeeProfile
from src.rag.ingestion import ingest_policies
from src.rag.retrieval import retrieve_policy_chunks


def _profile(category: str) -> EmployeeProfile:
    return EmployeeProfile(
        employee_id="EMP-RAG",
        employee_category=category,  # type: ignore[arg-type]
        cost_center="CC-RAG",
        grade="G6",
        home_city="Kazan",
        policy_tier="default",
        max_hotel_per_night=7000,
        max_flight_class="economy",
        requires_preapproval_over_amount=45000,
        accessibility_requirements=[],
    )


def test_retrieval_returns_multiple_chunks():
    ingest_policies(with_qdrant=False)
    chunks = retrieve_policy_chunks("ночные вылеты и согласование", _profile("standard"))
    assert len(chunks) >= 1
    assert chunks[0].document_id
    assert chunks[0].section_id
    assert chunks[0].source_path


def test_retrieval_applies_employee_category_filter():
    ingest_policies(with_qdrant=False)
    chunks = retrieve_policy_chunks("mobility exception wheelchair_access", _profile("standard"))
    assert chunks
    # Для standard не должны приоритетно попадать чисто mobility-only записи.
    assert any(chunk.employee_category in {"all", "standard"} for chunk in chunks)


def test_retrieval_prefers_qdrant_results_when_available(monkeypatch):
    qdrant_like_items = [
        {
            "id": "point-1",
            "vector": [0.1] * 96,
            "payload": {
                "document_id": "POLICY-TRANSPORT",
                "section_id": "POLICY-TRANSPORT-S01-C001",
                "source_path": "data/policies/transport_policy.md",
                "document_title": "Transport Policy",
                "policy_version": "v0.2",
                "section_title": "Класс обслуживания",
                "effective_from": "2026-08-14",
                "employee_category": "all",
                "rule_type": "transport",
                "chunk_text": "Для категории standard допустим класс economy.",
            },
        }
    ]

    monkeypatch.setattr("src.rag.retrieval._retrieve_from_qdrant", lambda *args, **kwargs: qdrant_like_items)
    monkeypatch.setattr("src.rag.retrieval._load_records", lambda: [])

    chunks = retrieve_policy_chunks("допустимый класс economy", _profile("standard"))
    assert len(chunks) == 1
    assert chunks[0].document_id == "POLICY-TRANSPORT"


def test_retrieval_uses_qdrant_query_points_not_scroll(monkeypatch):
    monkeypatch.setattr("src.config.settings.embedding_provider", "local_hash")
    monkeypatch.setattr("src.config.settings.rag_corpus_version", "v1")

    called = {"query_points": False, "query_filter": None}

    class _FakeFilter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FieldCondition:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _MatchAny:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _MatchValue:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakePoint:
        id = "point-1"
        vector = [0.01] * 1024
        score = 0.77
        payload = {
            "document_id": "POLICY-TRANSPORT",
            "section_id": "POLICY-TRANSPORT-S01-C001",
            "source_path": "data/policies/transport_policy.md",
            "document_title": "Transport Policy",
            "policy_version": "v0.2",
            "section_title": "Класс обслуживания",
            "effective_from": "2026-08-14",
            "employee_category": "all",
            "rule_type": "transport",
            "chunk_text": "Класс economy допустим.",
            "corpus_version": "v1",
        }

    class _FakeClient:
        def __init__(self, **kwargs):
            _ = kwargs

        def query_points(self, **kwargs):
            called["query_points"] = True
            called["query_filter"] = kwargs.get("query_filter")
            return types.SimpleNamespace(points=[_FakePoint()])

        def scroll(self, **kwargs):  # pragma: no cover - should not be called
            _ = kwargs
            raise AssertionError("scroll must not be used")

    fake_models = types.SimpleNamespace(
        FieldCondition=_FieldCondition,
        Filter=_FakeFilter,
        MatchAny=_MatchAny,
        MatchValue=_MatchValue,
    )
    fake_module = types.SimpleNamespace(QdrantClient=_FakeClient)

    monkeypatch.setitem(__import__("sys").modules, "qdrant_client", fake_module)
    monkeypatch.setitem(__import__("sys").modules, "qdrant_client.models", fake_models)

    chunks = retrieve_policy_chunks("эконом класс", _profile("standard"))
    assert called["query_points"] is True
    query_filter = called["query_filter"]
    assert query_filter is not None
    assert getattr(query_filter, "kwargs", {}).get("must")
    assert getattr(query_filter, "kwargs", {}).get("must_not")
    assert getattr(query_filter, "kwargs", {}).get("should")
    assert chunks
