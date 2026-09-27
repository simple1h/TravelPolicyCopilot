from __future__ import annotations

import pytest

from src.config import settings
from src.rag.embeddings import (
    EmbeddingError,
    embed_texts,
    format_bge_document,
    format_bge_query,
    normalize_embedding_text,
    resolve_policy_collection_name,
)


def test_bge_query_and_document_format_contract():
    query = format_bge_query("  Привет,\nмир  ")
    document = format_bge_document("  Привет,\nмир  ")
    assert query == "Represent this sentence for searching relevant passages: Привет, мир"
    assert document == "Привет, мир"


def test_normalize_embedding_text_uses_nfkc():
    text = "ＡＢС   test"
    assert normalize_embedding_text(text) == "ABС test"


def test_collection_name_contains_model_and_version(monkeypatch):
    monkeypatch.setattr(settings, "qdrant_collection_name", "travel_policy")
    monkeypatch.setattr(settings, "embedding_model", "baai/bge-m3")
    monkeypatch.setattr(settings, "rag_collection_version", "v2")
    assert resolve_policy_collection_name() == "travel_policy_baai_bge_m3_v2"


def test_strict_mode_fails_without_embedding_key(monkeypatch):
    monkeypatch.setattr(settings, "backend_mode", "strict")
    monkeypatch.setattr(settings, "embedding_provider", "router.ai")
    monkeypatch.setattr(settings, "embedding_api_key", "")
    with pytest.raises(EmbeddingError) as exc_info:
        embed_texts(["test"], is_query=True)
    assert exc_info.value.code == "EMBEDDING_PROVIDER_UNAVAILABLE"
