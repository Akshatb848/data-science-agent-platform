import pytest

from llm.client import OpenAIClient, AnthropicClient, validate_llm_client, LLMValidationResult


@pytest.mark.asyncio
async def test_openai_key_format_rejected(monkeypatch):
    client = OpenAIClient(api_key="not-a-key", model="gpt-4o-mini")
    result = await validate_llm_client(client)
    assert result.status == "AUTH_FAILED"
    assert result.state == "auth_failed"


@pytest.mark.asyncio
async def test_anthropic_key_format_rejected(monkeypatch):
    client = AnthropicClient(api_key="bad-key", model="claude-3-5-sonnet-20240620")
    result = await validate_llm_client(client)
    assert result.status == "AUTH_FAILED"
    assert result.state == "auth_failed"


@pytest.mark.asyncio
async def test_valid_probe_accepts_non_empty_response(monkeypatch):
    client = OpenAIClient(api_key="sk-valid", model="gpt-4o-mini")

    async def fake_chat(*args, **kwargs):
        return "OK"

    monkeypatch.setattr(client, "chat", fake_chat)
    result = await validate_llm_client(client)
    assert result.status == "SUCCESS"
    assert result.state == "connected"


@pytest.mark.asyncio
async def test_rate_limit_classification(monkeypatch):
    client = OpenAIClient(api_key="sk-valid", model="gpt-4o-mini")

    async def fake_chat(*args, **kwargs):
        raise RuntimeError("429 rate limit exceeded")

    monkeypatch.setattr(client, "chat", fake_chat)
    result = await validate_llm_client(client)
    assert result.status == "RATE_LIMITED"
    assert result.state == "rate_limited"
