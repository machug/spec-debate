"""Tests for the October 2026 model refresh: Gemini 4 Argon discovery, the
GPT-6 Sol/Luna lineup, Claude 5.5 temperature handling, and the gpt-5.5
retirement from ChatGPT-account Codex."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

import models
import providers


def test_discover_models_gemini_includes_gemini4(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "XAI_API_KEY", "ZAI_API_KEY",
                "MISTRAL_API_KEY", "GROQ_API_KEY", "DEEPSEEK_API_KEY",
                "MOONSHOT_API_KEY", "MINIMAX_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(providers, "ANTIGRAVITY_AVAILABLE", False)
    body = json.dumps({"models": [
        {"name": "models/gemini-4-argon", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemini-3.8-flash", "supportedGenerationMethods": ["generateContent"]},
        {"name": "models/gemma-4-31b-it", "supportedGenerationMethods": ["generateContent"]},
    ]})

    with patch("urllib.request.urlopen", lambda req, timeout=10: io.BytesIO(body.encode())):
        result = providers.discover_models()
    assert "gemini/gemini-4-argon" in result["Google Gemini"]
    assert "gemini/gemma-4-31b-it" not in result["Google Gemini"]


def test_gemini4_argon_keeps_temperature():
    # verified 2026-10-14: gemini-4-argon accepts temperature and max_tokens
    assert not models.is_reasoning_model("gemini/gemini-4-argon")


def test_gpt6_sol_luna_are_reasoning_models():
    for m in ("gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna", "codex/gpt-6.1-sol"):
        assert models.is_reasoning_model(m)
        assert models.uses_max_completion_tokens(m)
    assert models.gpt5_tuning_params("gpt-6.1-sol") == {"extra_body": {"reasoning_effort": "medium"}}


def test_claude_55_omits_temperature():
    for m in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"):
        assert models.claude_version(m) == (5, 5)
        assert models.is_reasoning_model(m)
        assert not models.uses_max_completion_tokens(m)


def test_codex_chatgpt_lineup():
    for m in ("gpt-6.1-sol", "gpt-6-sol", "gpt-6-luna", "gpt-6-astra"):
        assert m in providers.CODEX_CHATGPT_MODELS
    assert "gpt-5.5" not in providers.CODEX_CHATGPT_MODELS
