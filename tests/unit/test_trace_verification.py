from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.observability import tracing


class _EventsOnlyReadError(Exception):
    status_code = 404

    def __str__(self) -> str:
        return (
            "NotFoundError: status_code: 404, body: "
            "{'message': 'This endpoint is not available on deployments running in Langfuse v4 events_only mode.'}"
        )


@pytest.fixture(autouse=True)
def _reset_tracing_state():
    tracing._LANGFUSE_CLIENT = None
    tracing._LANGFUSE_INIT_ERROR = None
    tracing._LANGFUSE_RUNTIME_ERROR = "stale"
    tracing._LANGFUSE_DIAGNOSTIC_WARNINGS = []
    yield
    tracing._LANGFUSE_CLIENT = None
    tracing._LANGFUSE_INIT_ERROR = None
    tracing._LANGFUSE_RUNTIME_ERROR = None
    tracing._LANGFUSE_DIAGNOSTIC_WARNINGS = []


def test_verify_trace_read_api_success(monkeypatch):
    client = SimpleNamespace(
        get_trace=lambda trace_id: {"id": trace_id},
    )
    monkeypatch.setattr(tracing, "_init_langfuse_client", lambda: client)

    result = tracing.verify_trace("a" * 32)

    assert result.verified is True
    assert result.mode == "read_api"
    assert tracing.get_tracing_diagnostics()["runtime_error"] is None


def test_verify_trace_events_only_fallback(monkeypatch):
    calls = {"scores": 0}

    def _get_trace(trace_id: str):
        raise _EventsOnlyReadError()

    def _create_score(**kwargs):
        calls["scores"] += 1

    def _flush():
        return None

    client = SimpleNamespace(get_trace=_get_trace, create_score=_create_score, flush=_flush)
    monkeypatch.setattr(tracing, "_init_langfuse_client", lambda: client)

    result = tracing.verify_trace("b" * 32)

    assert result.verified is True
    assert result.mode == "events_write_ack"
    assert calls["scores"] == 1
    diagnostics = tracing.get_tracing_diagnostics()
    assert diagnostics["runtime_error"] is None
    assert diagnostics["diagnostic_warnings"]


def test_verify_trace_auth_error_is_not_verified(monkeypatch):
    def _get_trace(trace_id: str):
        raise RuntimeError("401 Unauthorized")

    client = SimpleNamespace(get_trace=_get_trace)
    monkeypatch.setattr(tracing, "_init_langfuse_client", lambda: client)

    result = tracing.verify_trace("c" * 32)

    assert result.verified is False
    assert result.mode == "failed"
    assert "401" in str(tracing.get_tracing_diagnostics()["runtime_error"])
