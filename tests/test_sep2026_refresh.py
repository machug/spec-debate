"""Tests for the September 2026 model refresh: new defaults, Fable 5.1
temperature handling, and live Anthropic model discovery."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import models
import providers


def test_xai_default_is_grok_47(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "x")
    defaults = {name: model for name, _, model in providers.get_available_providers()}
    assert defaults["xAI"] == "xai/grok-4.7"


def test_fable_51_omits_temperature():
    assert models.claude_version("claude-fable-5-1") == (5, 1)
    assert models.is_reasoning_model("claude-fable-5-1")
    assert not models.uses_max_completion_tokens("claude-fable-5-1")


def test_discover_models_anthropic_is_live(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY", "ZAI_API_KEY",
                "MISTRAL_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY",
                "MOONSHOT_API_KEY", "MINIMAX_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(providers, "ANTIGRAVITY_AVAILABLE", False)
    body = json.dumps({"data": [{"id": "claude-fable-5-1"}, {"id": "claude-opus-5"}]})

    def fake_urlopen(req, timeout=10):
        assert req.full_url.startswith("https://api.anthropic.com/v1/models")
        assert req.get_header("X-api-key") == "k"
        return io.BytesIO(body.encode())

    with patch("urllib.request.urlopen", fake_urlopen):
        result = providers.discover_models()
    assert result["Anthropic"] == ["claude-fable-5-1", "claude-opus-5"]
