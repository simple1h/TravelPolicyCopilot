from __future__ import annotations

import types

import pytest

from src.config import settings
from src.rag.embeddings import EmbeddingError, embed_texts


class _FakeItem:
    def __init__(self, index: int, embedding):
        self.index = index
        self.embedding = embedding


class _FakeUsage:
    def __init__(self, prompt_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.total_tokens = prompt_tokens


class _FakeResponse:
    def __init__(self, data, prompt_tokens: int = 12):
        self.data = data
        self.usage = _FakeUsage(prompt_tokens)


def _configure_strict_embedding(monkeypatch):
    monkeypatch.setattr(settings, "backend_mode", "strict")
    monkeypatch.setattr(settings, "embedding_provider", "router.ai")
    monkeypatch.setattr(settings, "embedding_api_key", "test-key")
    monkeypatch.setattr(settings, "embedding_dimension", 4)


def _patch_openai(monkeypatch, create_fn):
    class _FakeEmbeddings:
        @staticmethod
        def create(**kwargs):
            return create_fn(**kwargs)

    class _FakeClient:
        embeddings = _FakeEmbeddings()

    class _OpenAI:
        def __init__(self, **kwargs):
            _ = kwargs

        @property
        def embeddings(self):
            return _FakeClient.embeddings

    monkeypatch.setitem(__import__("sys").modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))


def test_embeddings_accepts_reordered_batch_and_restores_order(monkeypatch):
    _configure_strict_embedding(monkeypatch)

    def _create(**kwargs):
        _ = kwargs
        return _FakeResponse(
            data=[
                _FakeItem(1, [0.1, 0.2, 0.3, 0.4]),
                _FakeItem(0, [0.9, 0.8, 0.7, 0.6]),
            ],
            prompt_tokens=24,
        )

    _patch_openai(monkeypatch, _create)
    vectors, usage = embed_texts(["doc one", "doc two"], is_query=False)
    assert vectors[0] == [0.9, 0.8, 0.7, 0.6]
    assert vectors[1] == [0.1, 0.2, 0.3, 0.4]
    assert usage.input_tokens == 24


def test_embeddings_rejects_incomplete_batch(monkeypatch):
    _configure_strict_embedding(monkeypatch)
    _patch_openai(
        monkeypatch,
        lambda **kwargs: _FakeResponse(data=[_FakeItem(0, [0.1, 0.2, 0.3, 0.4])]),
    )
    with pytest.raises(EmbeddingError) as exc_info:
        embed_texts(["a", "b"], is_query=False)
    assert exc_info.value.code == "EMBEDDING_PROVIDER_INVALID_RESPONSE"


def test_embeddings_rejects_dimension_mismatch(monkeypatch):
    _configure_strict_embedding(monkeypatch)
    _patch_openai(
        monkeypatch,
        lambda **kwargs: _FakeResponse(data=[_FakeItem(0, [0.1, 0.2]), _FakeItem(1, [0.3, 0.4])]),
    )
    with pytest.raises(EmbeddingError) as exc_info:
        embed_texts(["a", "b"], is_query=False)
    assert exc_info.value.code == "EMBEDDING_DIMENSION_MISMATCH"


def test_embeddings_rejects_malformed_vector(monkeypatch):
    _configure_strict_embedding(monkeypatch)
    _patch_openai(
        monkeypatch,
        lambda **kwargs: _FakeResponse(data=[_FakeItem(0, "not-a-vector")]),
    )
    with pytest.raises(EmbeddingError) as exc_info:
        embed_texts(["a"], is_query=True)
    assert exc_info.value.code == "EMBEDDING_PROVIDER_INVALID_RESPONSE"


def test_embeddings_timeout_and_5xx_surface_unavailable(monkeypatch):
    _configure_strict_embedding(monkeypatch)

    class _TimeoutError(Exception):
        pass

    class _ServerError(Exception):
        status_code = 503

    _patch_openai(monkeypatch, lambda **kwargs: (_ for _ in ()).throw(_TimeoutError()))
    with pytest.raises(EmbeddingError) as timeout_exc:
        embed_texts(["a"], is_query=True)
    assert timeout_exc.value.code == "EMBEDDING_PROVIDER_UNAVAILABLE"
    assert timeout_exc.value.retryable is True

    _patch_openai(monkeypatch, lambda **kwargs: (_ for _ in ()).throw(_ServerError()))
    with pytest.raises(EmbeddingError) as server_exc:
        embed_texts(["a"], is_query=True)
    assert server_exc.value.code == "EMBEDDING_PROVIDER_UNAVAILABLE"
    assert server_exc.value.retryable is True
