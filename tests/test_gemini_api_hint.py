"""gemini-api CLI installed + GEMINI_API_KEY unset -> one-time hint.

`gemini-api configure` stores the key in the OS keychain, so the skill
skipped Gemini without saying why."""

from __future__ import annotations

import providers


def test_hint_when_cli_installed_and_key_unset(monkeypatch):
    monkeypatch.setattr(providers, "GEMINI_API_CLI_AVAILABLE", True)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert "export GEMINI_API_KEY" in providers.gemini_api_key_hint()


def test_no_hint_when_key_set_or_cli_missing(monkeypatch):
    monkeypatch.setattr(providers, "GEMINI_API_CLI_AVAILABLE", True)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert providers.gemini_api_key_hint() is None
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setattr(providers, "GEMINI_API_CLI_AVAILABLE", False)
    assert providers.gemini_api_key_hint() is None


def test_auto_select_prints_hint_once(monkeypatch, capsys):
    monkeypatch.setattr(providers, "GEMINI_API_CLI_AVAILABLE", True)
    monkeypatch.setattr(providers, "_gemini_api_hint_shown", False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    providers.get_available_providers()
    providers.get_available_providers()
    assert capsys.readouterr().err.count("gemini-api is installed") == 1
