import pytest

from agents.coordinator_agent import CoordinatorAgent


@pytest.mark.asyncio
async def test_intent_blocked_without_llm():
    coordinator = CoordinatorAgent(llm_client=None)
    result = await coordinator.analyze_user_intent("analyze data", {})
    assert result["intent"] == "llm_unavailable"
    assert result["blocked"] is True
