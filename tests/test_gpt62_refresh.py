"""Tests for the GPT-6.2 model refresh: GPT-6.2 Sol and GPT-6.2 Astra.
Both verified live via Codex CLI 0.169.0 and the OpenAI API 2026-11-18."""

from __future__ import annotations

import models
import providers


def test_gpt62_models_are_reasoning_models():
    for m in ("gpt-6.2-sol", "gpt-6.2-astra", "codex/gpt-6.2-sol", "codex/gpt-6.2-astra"):
        assert models.is_reasoning_model(m)
        assert models.uses_max_completion_tokens(m)


def test_codex_chatgpt_lineup_has_gpt62():
    for m in ("gpt-6.2-sol", "gpt-6.2-astra"):
        assert m in providers.CODEX_CHATGPT_MODELS


def test_defaults_move_to_gpt62_sol(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setattr(providers, "CODEX_AVAILABLE", True)
    defaults = {name: model for name, _, model in providers.get_available_providers()}
    assert defaults["OpenAI"] == "gpt-6.2-sol"
    assert defaults["Codex CLI"] == "codex/gpt-6.2-sol"
