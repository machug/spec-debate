"""Tests for provider config, cost lookup, and credential validation."""

from __future__ import annotations

import pytest
from providers import get_model_cost, validate_model_credentials


# ---------------------------------------------------------------------------
# get_model_cost
# ---------------------------------------------------------------------------


class TestGetModelCost:
    def test_cli_tools_are_free(self):
        assert get_model_cost("codex/gpt-5.3-codex") == {"input": 0.0, "output": 0.0}
        assert get_model_cost("codex/latest") == {"input": 0.0, "output": 0.0}
        assert get_model_cost("gemini-cli/gemini-3.1-pro-preview") == {"input": 0.0, "output": 0.0}

    def test_unknown_model_returns_default(self):
        cost = get_model_cost("totally-unknown-model-xyz")
        assert cost == {"input": 5.00, "output": 15.00}

    def test_foundry_model_returns_default(self):
        # Foundry models aren't in litellm's registry
        cost = get_model_cost("foundry/gpt-5-mini")
        assert cost["input"] >= 0
        assert cost["output"] >= 0

    def test_known_model_returns_nonzero(self):
        # gpt-5.4 should be in litellm's registry
        cost = get_model_cost("gpt-5.4")
        assert cost["input"] > 0
        assert cost["output"] > 0


# ---------------------------------------------------------------------------
# validate_model_credentials
# ---------------------------------------------------------------------------


class TestValidateModelCredentials:
    def test_openai_valid_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        valid, invalid = validate_model_credentials(["gpt-5.4"])
        assert "gpt-5.4" in valid

    def test_openai_invalid_without_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        valid, invalid = validate_model_credentials(["gpt-5.4"])
        assert "gpt-5.4" in invalid

    def test_anthropic_valid(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        valid, invalid = validate_model_credentials(["claude-opus-4-6"])
        assert "claude-opus-4-6" in valid

    def test_gemini_valid(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        valid, invalid = validate_model_credentials(["gemini/gemini-3.1-pro-preview"])
        assert "gemini/gemini-3.1-pro-preview" in valid

    def test_xai_valid(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        valid, invalid = validate_model_credentials(["xai/grok-4.20-0309-reasoning"])
        assert "xai/grok-4.20-0309-reasoning" in valid

    def test_foundry_valid_with_key(self, monkeypatch):
        monkeypatch.setenv("AZURE_AI_API_KEY", "test-key")
        valid, invalid = validate_model_credentials(["foundry/gpt-5-mini"])
        assert "foundry/gpt-5-mini" in valid

    def test_foundry_invalid_without_key(self, monkeypatch):
        monkeypatch.delenv("AZURE_AI_API_KEY", raising=False)
        valid, invalid = validate_model_credentials(["foundry/gpt-5-mini"])
        assert "foundry/gpt-5-mini" in invalid

    def test_o_series_needs_openai_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        valid, invalid = validate_model_credentials(["o3", "o4-mini"])
        assert "o3" in valid
        assert "o4-mini" in valid

    def test_multiple_models_mixed(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        valid, invalid = validate_model_credentials(["gpt-5.4", "claude-opus-4-6"])
        assert "gpt-5.4" in valid
        assert "claude-opus-4-6" in invalid

    def test_zai_valid_with_key(self, monkeypatch):
        monkeypatch.setenv("ZAI_API_KEY", "test-key")
        valid, invalid = validate_model_credentials(["zai/glm-5"])
        assert "zai/glm-5" in valid

    def test_zai_invalid_without_key(self, monkeypatch):
        monkeypatch.delenv("ZAI_API_KEY", raising=False)
        valid, invalid = validate_model_credentials(["zai/glm-4.7"])
        assert "zai/glm-4.7" in invalid

    def test_moonshot_valid_with_key(self, monkeypatch):
        monkeypatch.setenv("MOONSHOT_API_KEY", "test-key")
        valid, invalid = validate_model_credentials(["moonshot/kimi-k2.5"])
        assert "moonshot/kimi-k2.5" in valid

    def test_moonshot_invalid_without_key(self, monkeypatch):
        monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
        valid, invalid = validate_model_credentials(["moonshot/kimi-k2-thinking"])
        assert "moonshot/kimi-k2-thinking" in invalid

    def test_unknown_prefix_assumed_valid(self, monkeypatch):
        # Unknown provider prefixes pass through (for openrouter, etc.)
        valid, invalid = validate_model_credentials(["openrouter/openai/gpt-5.2-pro"])
        assert "openrouter/openai/gpt-5.2-pro" in valid
