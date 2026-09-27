from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from src.config import is_strict_mode, settings
from src.domain.models import EmployeeProfile, PolicyChunkRef
from src.rag.embeddings import EmbeddingError, embed_texts, format_bge_document, resolve_policy_collection_name
from src.rag.ingestion import ingest_policies
from src.observability.tracing import traced_span


class RetrievalSettings:
    top_k: int = settings.rag_top_k
    top_n: int = settings.rag_top_n


class RagUnavailableError(RuntimeError):
    def __init__(self, *, code: str, safe_message: str, retryable: bool, component: str) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable
        self.component = component


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _query_terms(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower()))


def _rerank_score(query_tokens: set[str], dense_score: float, item: dict[str, Any]) -> float:
    payload = item.get("payload", {})
    chunk_tokens = _query_terms(payload.get("chunk_text", ""))
    lexical = (len(query_tokens & chunk_tokens) / len(query_tokens)) if query_tokens else 0.0
    return 0.85 * dense_score + 0.15 * lexical


def _to_chunk_ref(score: float, item: dict[str, Any]) -> PolicyChunkRef:
    payload = item["payload"]
    return PolicyChunkRef(
        document_id=payload["document_id"],
        section_id=payload["section_id"],
        source_path=payload["source_path"],
        score=round(score, 4),
        document_title=payload.get("document_title"),
        policy_version=payload.get("policy_version"),
        section_title=payload.get("section_title"),
        effective_from=payload.get("effective_from"),
        employee_category=payload.get("employee_category"),
        rule_type=payload.get("rule_type"),
        chunk_text=payload.get("chunk_text"),
    )


def _load_records() -> list[dict[str, Any]]:
    index_path = Path(settings.rag_local_index_path)
    if not index_path.exists():
        ingest_policies(with_qdrant=False)
    if not index_path.exists():
        return []
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    return payload.get("records", [])


def _apply_metadata_filter(records: list[dict[str, Any]], profile: EmployeeProfile) -> list[dict[str, Any]]:
    category = profile.employee_category
    target_version = settings.rag_corpus_version.strip() or "v1"
    target_policy_version = "v0.2"
    result = []
    for item in records:
        payload = item.get("payload", {})
        allowed_category = payload.get("employee_category", "all")
        rule_type = payload.get("rule_type")
        corpus_version = payload.get("corpus_version")
        policy_version = payload.get("policy_version")
        if (
            allowed_category in {"all", category}
            and rule_type != "security_test"
            and corpus_version == target_version
            and policy_version == target_policy_version
        ):
            result.append(item)
    return result


def _build_metadata_filter(profile: EmployeeProfile):
    from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
    target_version = settings.rag_corpus_version.strip() or "v1"
    target_policy_version = "v0.2"

    return Filter(
        must=[
            FieldCondition(key="corpus_version", match=MatchValue(value=target_version)),
            FieldCondition(key="policy_version", match=MatchValue(value=target_policy_version)),
        ],
        must_not=[
            FieldCondition(key="rule_type", match=MatchValue(value="security_test")),
        ],
        should=[
            FieldCondition(key="employee_category", match=MatchAny(any=["all", profile.employee_category])),
        ]
    )


def _retrieve_from_qdrant(
    query_vector: list[float],
    profile: EmployeeProfile,
    *,
    trace_id: str | None = None,
) -> list[dict[str, Any]] | None:
    try:
        from qdrant_client import QdrantClient
    except Exception:
        return None

    try:
        client = QdrantClient(url=settings.qdrant_url, timeout=settings.request_timeout_seconds)
        with traced_span(
            "travelcopilot.rag.retrieval.qdrant_query",
            trace_id=trace_id,
            input_data={
                "collection_name": resolve_policy_collection_name(),
                "top_k": RetrievalSettings.top_k,
                "employee_category": profile.employee_category,
            },
            metadata={"component": "rag_retrieval"},
        ) as span:
            response = client.query_points(
                collection_name=resolve_policy_collection_name(),
                query=query_vector,
                query_filter=_build_metadata_filter(profile),
                limit=RetrievalSettings.top_k * 4,
                with_payload=True,
                with_vectors=True,
            )
        points = list(getattr(response, "points", []) or [])
        span.update(output_data={"points_count": len(points)})
        if not points:
            return []
        items: list[dict[str, Any]] = []
        for point in points:
            vector = point.vector
            payload = point.payload or {}
            if isinstance(vector, list):
                point_vector = vector
            elif isinstance(vector, dict):
                # Named vectors support.
                point_vector = next(iter(vector.values())) if vector else []
            else:
                point_vector = []
            items.append(
                {
                    "id": str(point.id),
                    "vector": point_vector,
                    "payload": payload,
                    "qdrant_score": float(getattr(point, "score", 0.0) or 0.0),
                }
            )
        return items
    except Exception:
        return None


def retrieve_policy_chunks_with_diagnostics(
    query: str,
    profile: EmployeeProfile,
    *,
    trace_id: str | None = None,
) -> tuple[list[PolicyChunkRef], dict[str, Any]]:
    try:
        vectors, embedding_usage = embed_texts([query], is_query=True, trace_id=trace_id)
    except EmbeddingError as exc:
        raise RagUnavailableError(
            code=exc.code,
            safe_message=exc.safe_message,
            retryable=exc.retryable,
            component=exc.component,
        ) from exc
    query_vector = vectors[0]
    qdrant_items = _retrieve_from_qdrant(query_vector, profile, trace_id=trace_id)
    if qdrant_items is None:
        if is_strict_mode():
            raise RagUnavailableError(
                code="RAG_UNAVAILABLE",
                safe_message="Policy retrieval backend is unavailable.",
                retryable=True,
                component="qdrant",
            )
        records = _load_records()
        if not records:
            return []
        filtered = _apply_metadata_filter(records, profile)
    else:
        filtered = qdrant_items

    query_tokens = _query_terms(query)

    with traced_span(
        "travelcopilot.rag.retrieval.rerank",
        trace_id=trace_id,
        input_data={"candidates": len(filtered)},
        metadata={"component": "rag_rerank"},
    ) as span:
        ranked: list[tuple[float, dict[str, Any]]] = []
        for item in filtered:
            dense_score = float(item.get("qdrant_score", 0.0))
            if not dense_score and item.get("vector"):
                dense_score = _cosine(query_vector, item.get("vector", []))
            rerank_score = _rerank_score(query_tokens, dense_score, item)
            ranked.append((rerank_score, item))
        span.update(output_data={"reranked_count": len(ranked)})

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    top_k = ranked[: RetrievalSettings.top_k]
    top_n = top_k[: RetrievalSettings.top_n]

    chunks = [_to_chunk_ref(score, item) for score, item in top_n]
    diagnostics = {
        "embedding_usage": {
            "provider": embedding_usage.provider,
            "model": embedding_usage.model,
            "mode": embedding_usage.mode,
            "input_tokens": embedding_usage.input_tokens,
            "estimated_cost_usd": embedding_usage.estimated_cost_usd,
            "cost_known": embedding_usage.cost_known,
            "latency_ms": embedding_usage.latency_ms,
        },
        "retrieval": {
            "collection_name": resolve_policy_collection_name(),
            "top_k": RetrievalSettings.top_k,
            "top_n": RetrievalSettings.top_n,
            "retrieved_candidates": len(filtered),
        },
    }
    return chunks, diagnostics


def retrieve_policy_chunks(query: str, profile: EmployeeProfile) -> list[PolicyChunkRef]:
    chunks, _ = retrieve_policy_chunks_with_diagnostics(query, profile)
    return chunks
