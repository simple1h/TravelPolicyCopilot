from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import re
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from src.config import settings
from src.rag.embeddings import EmbeddingError, embed_texts, format_bge_document, resolve_policy_collection_name


@dataclass
class IngestionResult:
    chunks_count: int
    output_path: str
    qdrant_upserted: bool
    qdrant_error: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    collection_name: str | None = None
    security_collection_name: str | None = None
    embedding_usage: dict[str, Any] | None = None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _embedding(text: str, *, size: int) -> list[float]:
    _ = size
    vectors, _usage = embed_texts([text], is_query=False)
    return vectors[0]


def _chunk_text(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    clean = _normalize_text(text)
    if not clean:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunks.append(clean[start:end])
        if end >= len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def _extract_sections(markdown: str) -> list[tuple[str, str]]:
    lines = markdown.splitlines()
    sections: list[tuple[str, str]] = []
    current_title = "General"
    current_body: list[str] = []
    for line in lines:
        if line.startswith("#"):
            if current_body:
                sections.append((current_title, "\n".join(current_body)))
                current_body = []
            current_title = line.lstrip("#").strip() or "General"
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_title, "\n".join(current_body)))
    return sections


def _policy_metadata_by_filename(file_name: str) -> dict[str, str]:
    mapping = {
        "general_policy.md": {"document_id": "POLICY-GENERAL", "rule_type": "general", "employee_category": "all"},
        "transport_policy.md": {"document_id": "POLICY-TRANSPORT", "rule_type": "transport", "employee_category": "all"},
        "hotel_limits.md": {"document_id": "POLICY-HOTEL", "rule_type": "hotel_limit", "employee_category": "all"},
        "budget_policy.md": {"document_id": "POLICY-BUDGET", "rule_type": "budget", "employee_category": "all"},
        "exceptions_policy.md": {"document_id": "POLICY-EXCEPTIONS", "rule_type": "exception", "employee_category": "mobility"},
        "malicious_policy_injection.md": {
            "document_id": "POLICY-MALICIOUS",
            "rule_type": "security_test",
            "employee_category": "all",
        },
    }
    return mapping.get(
        file_name,
        {"document_id": f"POLICY-{file_name.upper()}", "rule_type": "general", "employee_category": "all"},
    )


def build_policy_index_records(
    *,
    policies_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_size: int | None = None,
    trace_id: str | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    pending_texts: list[str] = []
    pending_indexes: list[int] = []
    for path in sorted(policies_dir.glob("*.md")):
        markdown = path.read_text(encoding="utf-8")
        meta = _policy_metadata_by_filename(path.name)
        document_scope = "security_test" if meta["rule_type"] == "security_test" else "default"
        sections = _extract_sections(markdown)
        for section_num, (section_title, section_text) in enumerate(sections, start=1):
            chunks = _chunk_text(section_text, chunk_size=chunk_size, overlap=chunk_overlap)
            for chunk_num, chunk in enumerate(chunks, start=1):
                section_id = f"{meta['document_id']}-S{section_num:02d}-C{chunk_num:03d}"
                point_id = str(uuid5(NAMESPACE_URL, section_id))
                records.append(
                    {
                        "id": point_id,
                        "vector": [],
                        "payload": {
                            "document_id": meta["document_id"],
                            "document_title": path.stem.replace("_", " ").title(),
                            "policy_version": "v0.2",
                            "section_id": section_id,
                            "section_title": section_title,
                            "effective_from": date(2026, 8, 14).isoformat(),
                            "employee_category": meta["employee_category"],
                            "rule_type": meta["rule_type"],
                            "test_namespace": document_scope,
                            "source_path": str(path),
                            "chunk_text": format_bge_document(chunk),
                            "embedding_provider": settings.embedding_provider,
                            "embedding_model": settings.embedding_model,
                            "embedding_dimension": settings.embedding_dimension,
                            "collection_version": settings.rag_collection_version,
                            "corpus_version": settings.rag_corpus_version,
                        },
                    }
                )
                pending_texts.append(chunk)
                pending_indexes.append(len(records) - 1)

    batch_size = 32
    for start in range(0, len(pending_texts), batch_size):
        batch_texts = pending_texts[start : start + batch_size]
        vectors, _usage = embed_texts(batch_texts, is_query=False, trace_id=trace_id)
        for local_index, vector in enumerate(vectors):
            record_index = pending_indexes[start + local_index]
            records[record_index]["vector"] = vector

    if records and embedding_size is not None:
        # Keep test-time override support without changing configured runtime dimension.
        for record in records:
            record["vector"] = record["vector"][:embedding_size] + [0.0] * max(0, embedding_size - len(record["vector"]))
    return records


def _upsert_to_qdrant(records: list[dict[str, Any]]) -> tuple[bool, str | None]:
    if not records:
        return False, "No records to upsert."
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        return False, f"qdrant-client import failed: {type(exc).__name__}"

    def _upsert_collection(
        client: Any,
        *,
        collection_name: str,
        collection_records: list[dict[str, Any]],
    ) -> None:
        embedding_size = len(collection_records[0]["vector"])
        collections = {item.name for item in client.get_collections().collections}
        if collection_name in collections:
            client.delete_collection(collection_name=collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=item["id"], vector=item["vector"], payload=item["payload"])
            for item in collection_records
        ]
        client.upsert(collection_name=collection_name, points=points)

    try:
        client = QdrantClient(url=settings.qdrant_url, timeout=settings.request_timeout_seconds)
        default_collection = resolve_policy_collection_name()
        security_collection = f"{default_collection}_security"
        default_records = [item for item in records if item.get("payload", {}).get("test_namespace") != "security_test"]
        security_records = [item for item in records if item.get("payload", {}).get("test_namespace") == "security_test"]
        if default_records:
            _upsert_collection(client, collection_name=default_collection, collection_records=default_records)
        if security_records:
            _upsert_collection(client, collection_name=security_collection, collection_records=security_records)
        else:
            collections = {item.name for item in client.get_collections().collections}
            if security_collection in collections:
                client.delete_collection(collection_name=security_collection)
        return True, None
    except Exception as exc:  # pragma: no cover - external service runtime
        return False, f"qdrant upsert failed: {type(exc).__name__}: {exc}"


def ingest_policies(
    *,
    policies_dir: str = "data/policies",
    output_path: str | None = None,
    with_qdrant: bool = True,
) -> IngestionResult:
    target_output = output_path or settings.rag_local_index_path
    policies_path = Path(policies_dir)
    try:
        records = build_policy_index_records(
            policies_dir=policies_path,
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            embedding_size=settings.embedding_dimension,
        )
    except EmbeddingError as exc:
        raise RuntimeError(f"{exc.code}: {exc.safe_message}") from exc

    output = Path(target_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "meta": {
                    "embedding_provider": settings.embedding_provider,
                    "embedding_model": settings.embedding_model,
                    "embedding_dimension": settings.embedding_dimension,
                    "chunk_size": settings.rag_chunk_size,
                    "chunk_overlap": settings.rag_chunk_overlap,
                    "collection_name": resolve_policy_collection_name(),
                    "security_collection_name": f"{resolve_policy_collection_name()}_security",
                    "collection_version": settings.rag_collection_version,
                    "corpus_version": settings.rag_corpus_version,
                },
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    qdrant_upserted = False
    qdrant_error: str | None = None
    if with_qdrant:
        qdrant_upserted, qdrant_error = _upsert_to_qdrant(records)

    return IngestionResult(
        chunks_count=len(records),
        output_path=str(output),
        qdrant_upserted=qdrant_upserted,
        qdrant_error=qdrant_error,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        embedding_dimension=settings.embedding_dimension,
        collection_name=resolve_policy_collection_name(),
        security_collection_name=f"{resolve_policy_collection_name()}_security",
        embedding_usage={"mode": "remote_or_local"},
    )
