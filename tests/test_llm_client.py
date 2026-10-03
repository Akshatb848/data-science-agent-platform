"""Tests for core.llm_client: env-var resolution and offline (rule-based) behaviour.

No network calls are made; the OpenAI client is replaced with a mock where needed.
"""

from unittest.mock import MagicMock

import pytest

from core import llm_client as llm_mod
from core.llm_client import LLMClient, resolve_openrouter_config

_ENV_VARS = (
    "OPENROUTER_API_KEY",
    "OPENROUTER_BASE_URL",
    "AI_INTEGRATIONS_OPENROUTER_API_KEY",
    "AI_INTEGRATIONS_OPENROUTER_BASE_URL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    return monkeypatch


class TestResolveConfig:

    def test_nothing_set(self):
        assert resolve_openrouter_config() == (None, None)

    def test_standard_key_defaults_base_url(self, clean_env):
        clean_env.setenv("OPENROUTER_API_KEY", "test-key")
        assert resolve_openrouter_config() == (llm_mod.DEFAULT_OPENROUTER_BASE_URL, "test-key")

    def test_standard_vars_take_precedence(self, clean_env):
        clean_env.setenv("OPENROUTER_API_KEY", "std-key")
        clean_env.setenv("OPENROUTER_BASE_URL", "https://std.example/v1")
        clean_env.setenv("AI_INTEGRATIONS_OPENROUTER_API_KEY", "replit-key")
        clean_env.setenv("AI_INTEGRATIONS_OPENROUTER_BASE_URL", "https://replit.example/v1")
        assert resolve_openrouter_config() == ("https://std.example/v1", "std-key")

    def test_replit_vars_fallback(self, clean_env):
        clean_env.setenv("AI_INTEGRATIONS_OPENROUTER_API_KEY", "replit-key")
        clean_env.setenv("AI_INTEGRATIONS_OPENROUTER_BASE_URL", "https://replit.example/v1")
        assert resolve_openrouter_config() == ("https://replit.example/v1", "replit-key")


class TestOfflineBehaviour:

    def test_unavailable_without_key(self):
        client = LLMClient()
        assert client.is_available() is False
        assert client.chat("system", "hello") == ""

    def test_rule_based_dataset_analysis(self):
        profile = {
            "problem_type": "regression",
            "target_column": "price",
            "quality_score": 40.0,
            "schema": {"a": "numeric", "b": "numeric", "c": "categorical"},
            "stats": {"missing_pct": {"a": 50.0}},
            "id_columns": ["id"],
        }
        result = LLMClient().analyze_dataset(profile)
        assert result["analysis_source"] == "rule_based"
        assert result["suggested_kpis"] == ["rmse", "mae", "r2_score"]
        text = " ".join(result["recommendations"])
        assert "missing" in text and "quality" in text and "ID columns" in text

    def test_rule_based_results_summary(self):
        ctx = {
            "profile": {"stats": {"row_count": 10, "column_count": 3},
                        "problem_type": "binary_classification", "quality_score": 90},
            "model_results": {"champion_name": "RandomForest", "champion_score": 0.9,
                              "leaderboard": [{"name": "RandomForest", "score": 0.9}]},
        }
        text = LLMClient().analyze_results(ctx)
        assert "RandomForest" in text
        assert "Binary Classification" in text


class TestWithMockedClient:

    def _client_with(self, clean_env, fake_openai):
        clean_env.setenv("OPENROUTER_API_KEY", "test-key")
        client = LLMClient()
        client._client = fake_openai
        return client

    def test_chat_strips_thinking_tags(self, clean_env):
        fake = MagicMock()
        fake.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="<think>internal</think>  Final answer"))
        ]
        client = self._client_with(clean_env, fake)
        assert client.is_available() is True
        assert client.chat("sys", "q") == "Final answer"

    def test_chat_falls_back_to_second_model(self, clean_env):
        fake = MagicMock()
        ok = MagicMock()
        ok.choices = [MagicMock(message=MagicMock(content="from fallback"))]
        fake.chat.completions.create.side_effect = [RuntimeError("rate limited"), ok]
        client = self._client_with(clean_env, fake)
        assert client.chat("sys", "q") == "from fallback"
        models = [c.kwargs["model"] for c in fake.chat.completions.create.call_args_list]
        assert models == [llm_mod.PRIMARY_MODEL, llm_mod.FALLBACK_MODEL]

    def test_llm_dataset_analysis_parses_json(self, clean_env):
        fake = MagicMock()
        fake.chat.completions.create.return_value.choices = [MagicMock(message=MagicMock(
            content='Here you go: {"problem_type": "regression", "target_column": "y", '
                    '"suggested_kpis": ["rmse"], "recommendations": ["scale"]}'
        ))]
        client = self._client_with(clean_env, fake)
        result = client.analyze_dataset({"stats": {}})
        assert result["analysis_source"] == "llm"
        assert result["suggested_kpis"] == ["rmse"]
