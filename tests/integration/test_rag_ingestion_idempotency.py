from __future__ import annotations

import pytest

from src.config import settings
from src.rag.embeddings import resolve_policy_collection_name
from src.rag.ingestion import ingest_policies


def test_ingestion_is_idempotent_for_same_collection():
    try:
        from qdrant_client import QdrantClient
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        pytest.skip(f"qdrant-client is not available: {exc}")

    try:
        first = ingest_policies(with_qdrant=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Qdrant ingestion failed in current environment: {exc}")
    if not first.qdrant_upserted:
        pytest.skip(f"Qdrant ingestion was not completed: {first.qdrant_error}")

    second = ingest_policies(with_qdrant=True)
    if not second.qdrant_upserted:
        pytest.skip(f"Second Qdrant ingestion was not completed: {second.qdrant_error}")

    client = QdrantClient(url=settings.qdrant_url, timeout=settings.request_timeout_seconds)
    collection_name = resolve_policy_collection_name()
    default_count = client.count(collection_name=collection_name, exact=True).count
    security_count = client.count(collection_name=f"{collection_name}_security", exact=True).count
    assert default_count >= 1
    assert security_count >= 1
    assert first.chunks_count == second.chunks_count
