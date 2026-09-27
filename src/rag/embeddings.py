from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from time import perf_counter
import unicodedata
from typing import Any

from src.config import get_backend_mode, is_strict_mode, settings
from src.observability.tracing import sanitize_for_logging, traced_span


@dataclass
class EmbeddingUsage:
    input_tokens: int
    estimated_cost_usd: float
    latency_ms: int
    provider: str
    model: str
    mode: str
    cost_known: bool


@dataclass
class EmbeddingError(RuntimeError):
    code: str
    safe_message: str
    retryable: bool
    component: str

    def __str__(self) -> str:
        return self.safe_message


def normalize_embedding_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = normalized.strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def format_bge_query(text: str) -> str:
    normalized = normalize_embedding_text(text)
    return f"Represent this sentence for searching relevant passages: {normalized}"


def format_bge_document(text: str) -> str:
    return normalize_embedding_text(text)


def _hash_embedding(text: str, *, size: int) -> list[float]:
    vec = [0.0] * size
    tokens = re.findall(r"[a-zA-Zа-яА-ЯёЁ0-9]+", text.lower())
    if not tokens:
        return vec
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % size
        vec[idx] += 1.0
    norm = math.sqrt(sum(value * value for value in vec))
    if norm > 0:
        vec = [value / norm for value in vec]
    return vec


def _estimate_embedding_cost_usd(input_tokens: int) -> float:
    return round((input_tokens / 1_000_000) * settings.embedding_input_cost_per_1m_tokens_usd, 8)


def _sanitize_model_for_collection(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def resolve_policy_collection_name() -> str:
    model_name = _sanitize_model_for_collection(settings.embedding_model)
    version = settings.rag_collection_version.strip() or "v1"
    return f"{settings.qdrant_collection_name}_{model_name}_{version}"


def _should_use_local_embeddings() -> bool:
    mode = get_backend_mode()
    provider = settings.embedding_provider.strip().lower()
    if mode == "strict":
        return False
    if provider == "local_hash":
        return True
    if provider == "router.ai":
        return not bool(settings.embedding_api_key.strip())
    return True


def _ensure_routerai_available() -> None:
    provider = settings.embedding_provider.strip().lower()
    if provider != "router.ai":
        raise EmbeddingError(
            code="EMBEDDING_PROVIDER_UNAVAILABLE",
            safe_message="Embedding provider must be router.ai.",
            retryable=False,
            component="embedding_provider",
        )
    if not settings.embedding_api_key.strip():
        raise EmbeddingError(
            code="EMBEDDING_PROVIDER_UNAVAILABLE",
            safe_message="Embedding provider API key is not configured.",
            retryable=False,
            component="embedding_provider",
        )


def _validate_vectors(vectors: list[list[float]], expected_size: int) -> None:
    if not vectors:
        raise EmbeddingError(
            code="EMBEDDING_PROVIDER_INVALID_RESPONSE",
            safe_message="Embedding provider returned no vectors.",
            retryable=True,
            component="embedding_provider",
        )
    for vector in vectors:
        if len(vector) != expected_size:
            raise EmbeddingError(
                code="EMBEDDING_DIMENSION_MISMATCH",
                safe_message=f"Embedding vector dimension mismatch: expected {expected_size}.",
                retryable=False,
                component="embedding_provider",
            )
        if any(not math.isfinite(value) for value in vector):
            raise EmbeddingError(
                code="EMBEDDING_PROVIDER_INVALID_RESPONSE",
                safe_message="Embedding provider returned non-finite values.",
                retryable=True,
                component="embedding_provider",
            )


def _coerce_vector(raw_vector: Any) -> list[float]:
    if not isinstance(raw_vector, list):
        raise EmbeddingError(
            code="EMBEDDING_PROVIDER_INVALID_RESPONSE",
            safe_message="Embedding provider returned malformed vector payload.",
            retryable=True,
            component="embedding_provider",
        )
    vector: list[float] = []
    for item in raw_vector:
        if not isinstance(item, (int, float)):
            raise EmbeddingError(
                code="EMBEDDING_PROVIDER_INVALID_RESPONSE",
                safe_message="Embedding provider returned non-numeric vector values.",
                retryable=True,
                component="embedding_provider",
            )
        vector.append(float(item))
    return vector


def _parse_embeddings_response(data_items: list[Any], expected_count: int) -> list[list[float]]:
    if len(data_items) != expected_count:
        raise EmbeddingError(
            code="EMBEDDING_PROVIDER_INVALID_RESPONSE",
            safe_message=(
                f"Embedding provider returned unexpected batch size: expected {expected_count}, got {len(data_items)}."
            ),
            retryable=True,
            component="embedding_provider",
        )
    indexed_vectors: dict[int, list[float]] = {}
    for item in data_items:
        index_raw = getattr(item, "index", None)
        embedding_raw = getattr(item, "embedding", None)
        if not isinstance(index_raw, int):
            raise EmbeddingError(
                code="EMBEDDING_PROVIDER_INVALID_RESPONSE",
                safe_message="Embedding provider response item has invalid index.",
                retryable=True,
                component="embedding_provider",
            )
        if index_raw in indexed_vectors:
            raise EmbeddingError(
                code="EMBEDDING_PROVIDER_INVALID_RESPONSE",
                safe_message="Embedding provider response contains duplicate indexes.",
                retryable=True,
                component="embedding_provider",
            )
        indexed_vectors[index_raw] = _coerce_vector(embedding_raw)
    if sorted(indexed_vectors.keys()) != list(range(expected_count)):
        raise EmbeddingError(
            code="EMBEDDING_PROVIDER_INVALID_RESPONSE",
            safe_message="Embedding provider response indexes are incomplete or malformed.",
            retryable=True,
            component="embedding_provider",
        )
    return [indexed_vectors[index] for index in range(expected_count)]


def _routerai_embed(texts: list[str], *, is_query: bool, trace_id: str | None) -> tuple[list[list[float]], EmbeddingUsage]:
    _ensure_routerai_available()
    try:
        from openai import OpenAI
    except Exception as exc:  # pragma: no cover - runtime dependency guard
        raise EmbeddingError(
            code="EMBEDDING_CLIENT_IMPORT_ERROR",
            safe_message=f"Failed to import OpenAI client: {type(exc).__name__}",
            retryable=False,
            component="embedding_provider",
        ) from exc

    prepared = [format_bge_query(text) if is_query else format_bge_document(text) for text in texts]
    client = OpenAI(
        api_key=settings.embedding_api_key,
        base_url=settings.embedding_base_url,
        timeout=settings.embedding_timeout_seconds,
        max_retries=settings.embedding_max_retries,
    )
    with traced_span(
        "travelcopilot.rag.embedding.remote",
        trace_id=trace_id,
        input_data={
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "items": len(prepared),
            "is_query": is_query,
            "sample": sanitize_for_logging(prepared[0] if prepared else ""),
        },
        metadata={"component": "embedding"},
    ) as span:
        try:
            response = client.embeddings.create(model=settings.embedding_model, input=prepared)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            retryable = True
            if isinstance(status_code, int) and 400 <= status_code < 500:
                retryable = False
            raise EmbeddingError(
                code="EMBEDDING_PROVIDER_UNAVAILABLE",
                safe_message=f"Embedding provider request failed: {type(exc).__name__}",
                retryable=retryable,
                component="embedding_provider",
            ) from exc

        data = getattr(response, "data", []) or []
        vectors = _parse_embeddings_response(list(data), len(prepared))
        _validate_vectors(vectors, settings.embedding_dimension)
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "total_tokens", 0) or 0)
        cost_known = settings.embedding_input_cost_per_1m_tokens_usd > 0
        embedding_usage = EmbeddingUsage(
            input_tokens=input_tokens,
            estimated_cost_usd=_estimate_embedding_cost_usd(input_tokens) if cost_known else 0.0,
            latency_ms=0,
            provider="router.ai",
            model=settings.embedding_model,
            mode="remote",
            cost_known=cost_known,
        )
        span.update(
            output_data={"vector_count": len(vectors), "dimension": len(vectors[0]) if vectors else 0},
            metadata={
                "input_tokens": embedding_usage.input_tokens,
                "estimated_cost_usd": embedding_usage.estimated_cost_usd,
                "provider": embedding_usage.provider,
                "model": embedding_usage.model,
                "cost_known": embedding_usage.cost_known,
            },
        )
        return vectors, embedding_usage


def _local_embed(texts: list[str], *, is_query: bool, trace_id: str | None) -> tuple[list[list[float]], EmbeddingUsage]:
    if is_strict_mode():
        raise EmbeddingError(
            code="EMBEDDING_PROVIDER_UNAVAILABLE",
            safe_message="Strict mode requires external RouterAI embeddings provider.",
            retryable=False,
            component="embedding_provider",
        )
    prepared = [format_bge_query(text) if is_query else format_bge_document(text) for text in texts]
    with traced_span(
        "travelcopilot.rag.embedding.local",
        trace_id=trace_id,
        input_data={"items": len(prepared), "is_query": is_query},
        metadata={"component": "embedding"},
    ):
        vectors = [_hash_embedding(text, size=settings.embedding_dimension) for text in prepared]
    _validate_vectors(vectors, settings.embedding_dimension)
    input_tokens = sum(len(text.split()) for text in prepared)
    return vectors, EmbeddingUsage(
        input_tokens=input_tokens,
        estimated_cost_usd=0.0,
        latency_ms=0,
        provider="local_hash",
        model=f"hash-{settings.embedding_dimension}",
        mode="local",
        cost_known=False,
    )


def embed_texts(
    texts: list[str],
    *,
    is_query: bool,
    trace_id: str | None = None,
) -> tuple[list[list[float]], EmbeddingUsage]:
    if not texts:
        raise EmbeddingError(
            code="EMBEDDING_INPUT_ERROR",
            safe_message="Embedding input is empty.",
            retryable=False,
            component="embedding_provider",
        )
    started_from = perf_counter()
    if _should_use_local_embeddings():
        vectors, usage = _local_embed(texts, is_query=is_query, trace_id=trace_id)
    else:
        vectors, usage = _routerai_embed(texts, is_query=is_query, trace_id=trace_id)
    elapsed_ms = int((perf_counter() - started_from) * 1000)
    usage.latency_ms = elapsed_ms
    return vectors, usage
