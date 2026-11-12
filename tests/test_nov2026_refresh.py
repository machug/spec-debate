"""Tests for the November 2026 model refresh: Gemini 4 Carbon, GPT-6.1 Terra,
Claude Fable 5.2, Grok Build, Kimi K3.1 Pro, and MiniMax M3.1 Flash.
Parameter rules verified live 2026-11-12."""

from __future__ import annotations

import models
import providers


def test_new_models_keep_temperature():
    # accepted temperature=0.7 live on 2026-11-12
    for m in ("gemini/gemini-4-carbon", "xai/grok-build-0.1", "minimax/MiniMax-M3.1-Flash"):
        assert not models.is_reasoning_model(m)


def test_gpt61_terra_is_reasoning_model():
    for m in ("gpt-6.1-terra", "codex/gpt-6.1-terra"):
        assert models.is_reasoning_model(m)
        assert models.uses_max_completion_tokens(m)


def test_claude_fable_52_omits_temperature():
    assert models.claude_version("claude-fable-5-2") == (5, 2)
    assert models.is_reasoning_model("claude-fable-5-2")


def test_codex_chatgpt_lineup_has_terra():
    assert "gpt-6.1-terra" in providers.CODEX_CHATGPT_MODELS


def test_google_default_stays_argon(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    defaults = {name: model for name, _, model in providers.get_available_providers()}
    assert defaults["Google"] == "gemini/gemini-4-argon"
