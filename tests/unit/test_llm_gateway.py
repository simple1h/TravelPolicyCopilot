from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.llm.gateway import LLMGatewayError, OpenAILLMGateway


def _build_gateway_with_client(fake_client):
    return OpenAILLMGateway(
        provider="router.ai",
        model="gpt-4o-mini",
        api_key="test-key",
        base_url="https://routerai.ru/api/v1",
        timeout_seconds=10.0,
        max_retries=1,
        max_tokens=200,
        temperature=0.1,
        input_cost_per_1m_tokens_usd=0.15,
        output_cost_per_1m_tokens_usd=0.60,
        client_factory=lambda: fake_client,
    )


def test_gateway_complete_success():
    response = SimpleNamespace(
        id="resp-1",
        model="gpt-4o-mini",
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=20, total_tokens=32),
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content="structured output"),
            )
        ],
    )
    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=lambda **_: response,
            )
        )
    )
    gateway = _build_gateway_with_client(fake_client)

    result = gateway.complete(system_prompt="You are parser", user_prompt="parse this")

    assert result.content == "structured output"
    assert result.model == "gpt-4o-mini"
    assert result.finish_reason == "stop"
    assert result.usage.total_tokens == 32
    assert result.usage.estimated_cost_usd > 0
    assert result.response_id == "resp-1"


def test_gateway_complete_provider_error():
    def _raise_error(**_):
        raise RuntimeError("provider boom")

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=_raise_error,
            )
        )
    )
    gateway = _build_gateway_with_client(fake_client)

    with pytest.raises(LLMGatewayError) as exc_info:
        gateway.complete(system_prompt="You are parser", user_prompt="parse this")

    assert exc_info.value.code == "LLM_PROVIDER_ERROR"
    assert exc_info.value.retryable is True
