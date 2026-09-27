from __future__ import annotations

import json
from pathlib import Path

from src.rag.ingestion import build_policy_index_records, ingest_policies


def test_ingestion_builds_chunks_with_required_metadata():
    records = build_policy_index_records(
        policies_dir=Path("data/policies"),
        chunk_size=600,
        chunk_overlap=120,
        embedding_size=64,
    )
    assert records
    first = records[0]
    payload = first["payload"]
    for key in (
        "document_id",
        "document_title",
        "policy_version",
        "section_id",
        "section_title",
        "effective_from",
        "employee_category",
        "rule_type",
        "source_path",
        "chunk_text",
    ):
        assert key in payload
    assert len(first["vector"]) == 64


def test_ingestion_writes_local_index(tmp_path):
    output = tmp_path / "policy_index.json"
    result = ingest_policies(output_path=str(output), with_qdrant=False)
    assert result.chunks_count > 0
    assert output.exists()
    data = json.loads(output.read_text(encoding="utf-8"))
    assert "records" in data
    assert len(data["records"]) == result.chunks_count
