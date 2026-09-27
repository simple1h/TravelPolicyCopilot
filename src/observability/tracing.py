from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

TraceVerificationMode = Literal["read_api", "events_write_ack", "failed", "skipped"]

_LANGFUSE_CLIENT = None
_LANGFUSE_INIT_ERROR: str | None = None
_LANGFUSE_RUNTIME_ERROR: str | None = None
_LANGFUSE_DIAGNOSTIC_WARNINGS: list[str] = []
_SENSITIVE_KEYWORDS = ("password", "secret", "token", "api_key", "apikey", "authorization")
_SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?i)\b(password|secret|token|api[_-]?key|authorization)\b\s*[:=]\s*([^\s,;]+)"
)


def _resolve_langfuse_credentials() -> tuple[str, str, str]:
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
    host = os.getenv("LANGFUSE_HOST", "").strip()

    if public_key and secret_key and host:
        return public_key, secret_key, host

    try:
        from src.config import settings as app_settings

        public_key = public_key or str(getattr(app_settings, "langfuse_public_key", "")).strip()
        secret_key = secret_key or str(getattr(app_settings, "langfuse_secret_key", "")).strip()
        host = host or str(getattr(app_settings, "langfuse_host", "")).strip()
    except Exception:
        pass

    return public_key, secret_key, host


def _normalize_trace_id(trace_id: str | None) -> str | None:
    if not trace_id:
        return None
    candidate = trace_id.strip().replace("-", "").lower()
    if len(candidate) != 32:
        return None
    try:
        int(candidate, 16)
    except ValueError:
        return None
    return candidate


def _init_langfuse_client():
    global _LANGFUSE_CLIENT
    global _LANGFUSE_INIT_ERROR

    if _LANGFUSE_CLIENT is not None or _LANGFUSE_INIT_ERROR is not None:
        return _LANGFUSE_CLIENT

    public_key, secret_key, host = _resolve_langfuse_credentials()
    if not public_key or not secret_key or not host:
        return None

    try:
        from langfuse import get_client

        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", public_key)
        os.environ.setdefault("LANGFUSE_SECRET_KEY", secret_key)
        os.environ.setdefault("LANGFUSE_HOST", host)
        _LANGFUSE_CLIENT = get_client()
    except Exception as exc:  # pragma: no cover - depends on runtime env
        _LANGFUSE_INIT_ERROR = str(exc)
        return None
    return _LANGFUSE_CLIENT


def get_tracing_diagnostics() -> dict[str, Any]:
    client = _init_langfuse_client()
    return {
        "client_ready": client is not None,
        "init_error": _LANGFUSE_INIT_ERROR,
        "runtime_error": _LANGFUSE_RUNTIME_ERROR,
        "diagnostic_warnings": list(_LANGFUSE_DIAGNOSTIC_WARNINGS),
    }


def _is_langfuse_events_only_read_unavailable(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "events_only",
        "events-only",
        "not available on deployments running in langfuse v4",
    )
    if any(marker in message for marker in markers):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code == 404 and "404" in message:
        return True
    return False


@dataclass(frozen=True)
class TraceVerificationResult:
    verified: bool
    mode: TraceVerificationMode
    diagnostic_warning: str | None = None


def verify_trace(trace_id: str | None) -> TraceVerificationResult:
    global _LANGFUSE_RUNTIME_ERROR
    global _LANGFUSE_DIAGNOSTIC_WARNINGS

    client = _init_langfuse_client()
    normalized_trace_id = _normalize_trace_id(trace_id)
    if client is None or normalized_trace_id is None:
        return TraceVerificationResult(verified=False, mode="failed")

    try:
        trace_obj = None
        if hasattr(client, "get_trace"):
            trace_obj = client.get_trace(trace_id=normalized_trace_id)
        elif hasattr(client, "api") and hasattr(client.api, "trace"):
            trace_obj = client.api.trace.get(normalized_trace_id)
        else:
            return TraceVerificationResult(verified=False, mode="failed")

        if trace_obj is not None:
            _LANGFUSE_RUNTIME_ERROR = None
            return TraceVerificationResult(verified=True, mode="read_api")
        return TraceVerificationResult(verified=False, mode="failed")
    except Exception as exc:
        read_error = f"{type(exc).__name__}: {exc}"
        if not _is_langfuse_events_only_read_unavailable(exc):
            _LANGFUSE_RUNTIME_ERROR = read_error
            return TraceVerificationResult(verified=False, mode="failed", diagnostic_warning=read_error)

        diagnostic_warning = read_error
        try:
            client.create_score(
                name="travelcopilot.trace.verification",
                value=1.0,
                trace_id=normalized_trace_id,
                data_type="NUMERIC",
                comment="trace verification ping (events_only fallback)",
            )
            client.flush()
            _LANGFUSE_RUNTIME_ERROR = None
            if diagnostic_warning not in _LANGFUSE_DIAGNOSTIC_WARNINGS:
                _LANGFUSE_DIAGNOSTIC_WARNINGS.append(diagnostic_warning)
            return TraceVerificationResult(
                verified=True,
                mode="events_write_ack",
                diagnostic_warning=diagnostic_warning,
            )
        except Exception as nested_exc:
            nested_error = f"{type(nested_exc).__name__}: {nested_exc}"
            _LANGFUSE_RUNTIME_ERROR = nested_error
            return TraceVerificationResult(
                verified=False,
                mode="failed",
                diagnostic_warning=f"{read_error}; fallback_failed={nested_error}",
            )


def verify_trace_exists(trace_id: str | None) -> bool:
    return verify_trace(trace_id).verified


def sanitize_for_logging(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            lowered = key.lower()
            if any(keyword in lowered for keyword in _SENSITIVE_KEYWORDS):
                sanitized[key] = "***REDACTED***"
            else:
                sanitized[key] = sanitize_for_logging(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_for_logging(item) for item in value]
    if isinstance(value, str):
        return _SENSITIVE_VALUE_PATTERN.sub(r"\1=***REDACTED***", value)
    return value


def create_langfuse_score(
    *,
    name: str,
    value: float | str,
    trace_id: str | None = None,
    observation_id: str | None = None,
    session_id: str | None = None,
    data_type: str | None = None,
    comment: str | None = None,
    metadata: Any | None = None,
) -> bool:
    global _LANGFUSE_RUNTIME_ERROR
    client = _init_langfuse_client()
    if client is None:
        return False

    kwargs: dict[str, Any] = {"name": name, "value": value}
    normalized_trace_id = _normalize_trace_id(trace_id)
    if normalized_trace_id:
        kwargs["trace_id"] = normalized_trace_id
    if observation_id:
        kwargs["observation_id"] = observation_id
    if session_id:
        kwargs["session_id"] = session_id
    if data_type:
        kwargs["data_type"] = data_type
    if comment:
        kwargs["comment"] = comment
    if metadata is not None:
        kwargs["metadata"] = metadata

    try:
        client.create_score(**kwargs)
        client.flush()
        return True
    except Exception as exc:
        _LANGFUSE_RUNTIME_ERROR = f"{type(exc).__name__}: {exc}"
        return False


@dataclass
class TraceHandle:
    trace_id: str | None = None
    enabled: bool = False
    _observation: Any = None

    def update(self, *, input_data: Any | None = None, output_data: Any | None = None, metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled or self._observation is None:
            return
        payload: dict[str, Any] = {}
        if input_data is not None:
            payload["input"] = input_data
        if output_data is not None:
            payload["output"] = output_data
        if metadata is not None:
            payload["metadata"] = metadata
        if not payload:
            return
        try:
            self._observation.update(**payload)
        except Exception:
            return


@dataclass
class SpanHandle:
    trace_id: str | None = None
    enabled: bool = False
    _observation: Any = None

    def update(
        self,
        *,
        input_data: Any | None = None,
        output_data: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or self._observation is None:
            return
        payload: dict[str, Any] = {}
        if input_data is not None:
            payload["input"] = input_data
        if output_data is not None:
            payload["output"] = output_data
        if metadata is not None:
            payload["metadata"] = metadata
        if not payload:
            return
        try:
            self._observation.update(**payload)
        except Exception:
            return


@contextmanager
def traced_workflow(
    *,
    name: str,
    trace_id: str | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
    input_data: Any | None = None,
):
    global _LANGFUSE_RUNTIME_ERROR
    client = _init_langfuse_client()
    if client is None:
        yield TraceHandle(trace_id=trace_id, enabled=False)
        return

    normalized_trace_id = _normalize_trace_id(trace_id)
    trace_context = {"trace_id": normalized_trace_id} if normalized_trace_id else None
    kwargs: dict[str, Any] = {"as_type": "span", "name": name}
    if trace_context is not None:
        kwargs["trace_context"] = trace_context
    if input_data is not None:
        kwargs["input"] = input_data
    if session_id is not None:
        kwargs["session_id"] = session_id
    if user_id is not None:
        kwargs["user_id"] = user_id

    try:
        try:
            observation_cm = client.start_as_current_observation(**kwargs)
        except TypeError:
            # Compatibility fallback for older/newer langfuse client signatures.
            fallback_kwargs = dict(kwargs)
            metadata = {}
            if session_id is not None:
                fallback_kwargs.pop("session_id", None)
                metadata["session_id"] = session_id
            if user_id is not None:
                fallback_kwargs.pop("user_id", None)
                metadata["user_id"] = user_id
            if metadata:
                fallback_kwargs["metadata"] = {
                    **dict(fallback_kwargs.get("metadata") or {}),
                    **metadata,
                }
            observation_cm = client.start_as_current_observation(**fallback_kwargs)
    except Exception as exc:
        _LANGFUSE_RUNTIME_ERROR = f"{type(exc).__name__}: {exc}"
        yield TraceHandle(trace_id=trace_id, enabled=False)
        return

    try:
        with observation_cm as observation:
            resolved_trace_id = getattr(observation, "trace_id", None) or normalized_trace_id
            handle = TraceHandle(
                trace_id=resolved_trace_id,
                enabled=True,
                _observation=observation,
            )
            yield handle
    finally:
        try:
            client.flush()
        except Exception:
            pass


@contextmanager
def traced_span(
    name: str,
    trace_id: str | None = None,
    input_data: Any | None = None,
    metadata: dict[str, Any] | None = None,
):
    start = perf_counter()
    global _LANGFUSE_RUNTIME_ERROR
    client = _init_langfuse_client()

    if client is None:
        try:
            yield SpanHandle(trace_id=trace_id, enabled=False)
        finally:
            _ = perf_counter() - start
        return

    kwargs: dict[str, Any] = {"as_type": "span", "name": name}
    normalized_trace_id = _normalize_trace_id(trace_id)
    if normalized_trace_id:
        kwargs["trace_context"] = {"trace_id": normalized_trace_id}
    if input_data is not None:
        kwargs["input"] = input_data
    if metadata is not None:
        kwargs["metadata"] = metadata

    try:
        observation_cm = client.start_as_current_observation(**kwargs)
    except Exception as exc:
        _LANGFUSE_RUNTIME_ERROR = f"{type(exc).__name__}: {exc}"
        yield SpanHandle(trace_id=trace_id, enabled=False)
        return

    try:
        with observation_cm as observation:
            handle = SpanHandle(
                trace_id=getattr(observation, "trace_id", trace_id),
                enabled=True,
                _observation=observation,
            )
            yield handle
            elapsed_ms = int((perf_counter() - start) * 1000)
            try:
                observation.update(metadata={"latency_ms": elapsed_ms})
            except Exception:
                pass
    finally:
        try:
            client.flush()
        except Exception:
            pass
