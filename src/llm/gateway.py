from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable

from src.config import Settings, settings
from src.observability.tracing import sanitize_for_logging, traced_span


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


@dataclass
class LLMResponse:
    content: str
    model: str
    finish_reason: str | None
    usage: LLMUsage
    response_id: str | None = None


class LLMGatewayError(RuntimeError):
    def __init__(self, *, code: str, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable


class OpenAILLMGateway:
    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        base_url: str,
        timeout_seconds: float,
        max_retries: int,
        max_tokens: int,
        temperature: float,
        input_cost_per_1m_tokens_usd: float,
        output_cost_per_1m_tokens_usd: float,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.input_cost_per_1m_tokens_usd = input_cost_per_1m_tokens_usd
        self.output_cost_per_1m_tokens_usd = output_cost_per_1m_tokens_usd
        self._client_factory = client_factory
        self._client: Any | None = None

    def _estimate_cost_usd(self, *, prompt_tokens: int, completion_tokens: int) -> float:
        input_cost = (prompt_tokens / 1_000_000) * self.input_cost_per_1m_tokens_usd
        output_cost = (completion_tokens / 1_000_000) * self.output_cost_per_1m_tokens_usd
        return round(input_cost + output_cost, 8)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client

        if not self.api_key:
            raise LLMGatewayError(
                code="LLM_CONFIG_ERROR",
                message="LLM API key is not configured.",
                retryable=False,
            )

        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - dependency/runtime guard
            raise LLMGatewayError(
                code="LLM_CLIENT_IMPORT_ERROR",
                message=f"Failed to import OpenAI client: {type(exc).__name__}",
                retryable=False,
            ) from exc

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        return self._client

    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> LLMResponse:
        if not system_prompt.strip():
            raise LLMGatewayError(
                code="LLM_INPUT_ERROR",
                message="system_prompt must not be empty.",
                retryable=False,
            )
        if not user_prompt.strip():
            raise LLMGatewayError(
                code="LLM_INPUT_ERROR",
                message="user_prompt must not be empty.",
                retryable=False,
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format

        with traced_span(
            "travelcopilot.llm.complete",
            trace_id=trace_id,
            input_data={
                "model": self.model,
                "system_prompt": sanitize_for_logging(system_prompt),
                "user_prompt": sanitize_for_logging(user_prompt),
                "response_format": response_format,
            },
            metadata={
                "provider": self.provider,
                "base_url": self.base_url,
                "timeout_seconds": self.timeout_seconds,
                "max_retries": self.max_retries,
            },
        ) as span:
            client = self._get_client()
            try:
                response = client.chat.completions.create(**payload)
            except TimeoutError as exc:
                span.update(
                    output_data={"error_code": "LLM_TIMEOUT"},
                    metadata={"error_type": type(exc).__name__},
                )
                raise LLMGatewayError(
                    code="LLM_TIMEOUT",
                    message="LLM request timed out.",
                    retryable=True,
                ) from exc
            except Exception as exc:
                span.update(
                    output_data={"error_code": "LLM_PROVIDER_ERROR"},
                    metadata={"error_type": type(exc).__name__},
                )
                raise LLMGatewayError(
                    code="LLM_PROVIDER_ERROR",
                    message=f"LLM provider request failed: {type(exc).__name__}",
                    retryable=True,
                ) from exc

            choices = getattr(response, "choices", None) or []
            if not choices:
                span.update(output_data={"error_code": "LLM_EMPTY_RESPONSE"})
                raise LLMGatewayError(
                    code="LLM_EMPTY_RESPONSE",
                    message="LLM returned no choices.",
                    retryable=True,
                )

            first_choice = choices[0]
            message = getattr(first_choice, "message", None)
            content = getattr(message, "content", None)
            if not content:
                span.update(output_data={"error_code": "LLM_EMPTY_CONTENT"})
                raise LLMGatewayError(
                    code="LLM_EMPTY_CONTENT",
                    message="LLM returned empty content.",
                    retryable=True,
                )

            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
            estimated_cost_usd = self._estimate_cost_usd(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            usage_data = LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                estimated_cost_usd=estimated_cost_usd,
            )
            span.update(
                output_data={
                    "response_id": getattr(response, "id", None),
                    "model": getattr(response, "model", self.model) or self.model,
                    "finish_reason": getattr(first_choice, "finish_reason", None),
                    "content": sanitize_for_logging(content),
                },
                metadata={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                    "estimated_cost_usd": estimated_cost_usd,
                },
            )
            return LLMResponse(
                content=content,
                model=getattr(response, "model", self.model) or self.model,
                finish_reason=getattr(first_choice, "finish_reason", None),
                usage=usage_data,
                response_id=getattr(response, "id", None),
            )


def build_llm_gateway(app_settings: Settings = settings) -> OpenAILLMGateway:
    return OpenAILLMGateway(
        provider=app_settings.llm_provider,
        model=app_settings.llm_model,
        api_key=app_settings.llm_api_key,
        base_url=app_settings.llm_base_url,
        timeout_seconds=app_settings.llm_timeout_seconds,
        max_retries=app_settings.llm_max_retries,
        max_tokens=app_settings.llm_max_tokens,
        temperature=app_settings.llm_temperature,
        input_cost_per_1m_tokens_usd=app_settings.llm_input_cost_per_1m_tokens_usd,
        output_cost_per_1m_tokens_usd=app_settings.llm_output_cost_per_1m_tokens_usd,
    )


@lru_cache(maxsize=1)
def get_default_llm_gateway() -> OpenAILLMGateway:
    return build_llm_gateway(settings)
