"""Tests for session state management and checkpointing."""

from __future__ import annotations

import argparse
import json

import pytest
from session import SessionState, save_checkpoint


class TestSessionState:
    def test_save_and_load(self, tmp_path, monkeypatch):
        import session

        monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path)

        state = SessionState(
            session_id="test-session",
            spec="# My Spec",
            round=2,
            doc_type="tech",
            models=["gpt-5.4", "claude-opus-4-6"],
            focus="security",
        )
        state.save()

        loaded = SessionState.load("test-session")
        assert loaded.session_id == "test-session"
        assert loaded.spec == "# My Spec"
        assert loaded.round == 2
        assert loaded.doc_type == "tech"
        assert loaded.models == ["gpt-5.4", "claude-opus-4-6"]
        assert loaded.focus == "security"
        assert loaded.updated_at != ""

    def test_load_nonexistent_raises(self, tmp_path, monkeypatch):
        import session

        monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path)

        with pytest.raises(FileNotFoundError):
            SessionState.load("does-not-exist")

    def test_list_sessions_empty(self, tmp_path, monkeypatch):
        import session

        monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path)
        assert SessionState.list_sessions() == []

    def test_list_sessions_returns_saved(self, tmp_path, monkeypatch):
        import session

        monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path)

        state = SessionState(
            session_id="my-debate",
            spec="content",
            round=3,
            doc_type="prd",
            models=["gpt-5.4"],
        )
        state.save()

        sessions = SessionState.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["id"] == "my-debate"
        assert sessions[0]["round"] == 3

    def test_path_traversal_blocked_save(self, tmp_path, monkeypatch):
        import session

        monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path)

        state = SessionState(
            session_id="../../../etc/evil",
            spec="content",
            round=1,
            doc_type="tech",
            models=[],
        )
        with pytest.raises(ValueError, match="Invalid session ID"):
            state.save()

    def test_path_traversal_blocked_load(self, tmp_path, monkeypatch):
        import session

        monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path)

        with pytest.raises(ValueError, match="Invalid session ID"):
            SessionState.load("../../../etc/passwd")


class TestSaveCheckpoint:
    def test_creates_checkpoint_file(self, tmp_path, monkeypatch):
        import session

        monkeypatch.setattr(session, "CHECKPOINTS_DIR", tmp_path)

        save_checkpoint("# My Spec v2", 2)

        files = list(tmp_path.glob("*.md"))
        assert len(files) == 1
        assert "round-2" in files[0].name
        assert files[0].read_text() == "# My Spec v2"

    def test_checkpoint_with_session_prefix(self, tmp_path, monkeypatch):
        import session

        monkeypatch.setattr(session, "CHECKPOINTS_DIR", tmp_path)

        save_checkpoint("content", 3, session_id="my-debate")

        files = list(tmp_path.glob("*.md"))
        assert "my-debate-round-3" in files[0].name

    def test_path_traversal_blocked(self, tmp_path, monkeypatch):
        import session

        monkeypatch.setattr(session, "CHECKPOINTS_DIR", tmp_path)

        with pytest.raises(ValueError, match="Invalid session ID"):
            save_checkpoint("content", 1, session_id="../../etc/evil")


class TestCritiquePersistedInHistory:
    """run_critique must write the argument, not just the verdict.

    Testing strip_spec_block alone would not catch the original bug: the
    critique text existed, run_critique simply never stored it.
    """

    def _run(self, tmp_path, monkeypatch, response, spec="# Spec\n"):
        import debate
        import session
        from models import ModelResponse

        monkeypatch.setattr(session, "SESSIONS_DIR", tmp_path)
        monkeypatch.setattr(session, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
        monkeypatch.setattr(
            debate,
            "call_models_parallel",
            lambda *a, **k: [
                ModelResponse(
                    model="stub", response=response, agreed=False, spec="# Revised\n"
                )
            ],
        )

        state = SessionState(
            session_id="wiring", spec=spec, round=1, doc_type="tech", models=["stub"]
        )
        args = argparse.Namespace(
            review_only=False,
            press=False,
            focus=None,
            persona=None,
            preserve_intent=False,
            codex_search=False,
            codex_reasoning="high",
            claude_effort="low",
            timeout=600,
            round=1,
            doc_type="tech",
            session="wiring",
            telegram=False,
            json=True,
        )
        debate.run_critique(args, spec, ["stub"], state, None, False, None)
        return json.loads((tmp_path / "wiring.json").read_text())

    def test_history_stores_the_critique_text(self, tmp_path, monkeypatch, capsys):
        saved = self._run(
            tmp_path, monkeypatch, "1. No retry budget\n2. No rollback\n"
        )
        entry = saved["history"][0]["models"][0]
        assert entry["model"] == "stub"
        assert "No retry budget" in entry["critique"]

    def test_history_omits_the_reemitted_document(self, tmp_path, monkeypatch, capsys):
        saved = self._run(
            tmp_path,
            monkeypatch,
            "1. No retry budget\n\n[SPEC]\n# A very long revised document\n[/SPEC]",
        )
        critique = saved["history"][0]["models"][0]["critique"]
        assert "No retry budget" in critique
        assert "very long revised document" not in critique
