"""Tests for emit-plan subcommand helpers."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock

import pytest

import debate
from debate import (
    _derive_title_hint,
    _strip_spec_suffix,
    default_plan_out_path,
    detect_pr_labels,
    handle_detect_prs,
    handle_emit_plan,
)


class TestDetectPrLabels:
    def test_empty_spec_returns_empty_list(self):
        assert detect_pr_labels("") == []

    def test_spec_with_no_pr_markers_returns_empty_list(self):
        spec = "# My Spec\n\nNo PR decomposition here."
        assert detect_pr_labels(spec) == []

    def test_detects_single_pr(self):
        spec = "## 15. Deployment Strategy\n\nPR-1 ships everything."
        assert detect_pr_labels(spec) == ["PR-1"]

    def test_detects_stacked_prs_sorted_and_unique(self):
        spec = """
## 15. Deployment Strategy

1. PR-1 — data engine
2. PR-2 — extraction
3. PR-3 — UI
4. PR-4 — generator

See PR-1 again below in the rollback section.
See PR-3 cross-reference.
"""
        assert detect_pr_labels(spec) == ["PR-1", "PR-2", "PR-3", "PR-4"]

    def test_numeric_sort_not_lexicographic(self):
        spec = "PR-1 PR-2 PR-10 PR-11"
        assert detect_pr_labels(spec) == ["PR-1", "PR-2", "PR-10", "PR-11"]

    def test_ignores_non_pr_numeric_patterns(self):
        spec = "CVE-2024-1234 issue-5 bug-99 PR-1"
        assert detect_pr_labels(spec) == ["PR-1"]

    def test_word_boundary_not_substring(self):
        spec = "GRPR-1 is not a PR label but PR-1 is."
        assert detect_pr_labels(spec) == ["PR-1"]


class TestDefaultPlanOutPath:
    def test_stdin_spec_writes_to_cwd(self):
        assert default_plan_out_path(None, "PR-1") == "./pr-1.plan.md"

    def test_sibling_path_with_spec_debate_final_suffix_stripped(self):
        result = default_plan_out_path(
            "docs/plans/2026-04-22-feature.spec-debate-final.md", "PR-1"
        )
        assert result == "docs/plans/2026-04-22-feature-pr-1.plan.md"

    def test_sibling_path_with_spec_suffix_stripped(self):
        result = default_plan_out_path("docs/foo.spec.md", "PR-2")
        assert result == "docs/foo-pr-2.plan.md"

    def test_sibling_path_with_no_known_suffix(self):
        result = default_plan_out_path("docs/plan.md", "PR-1")
        assert result == "docs/plan-pr-1.plan.md"

    def test_pr_label_lowercased_in_filename(self):
        result = default_plan_out_path("spec.md", "PR-3")
        assert result == "spec-pr-3.plan.md"

    def test_custom_label_slugged(self):
        result = default_plan_out_path("spec.md", "PR-5")
        assert result == "spec-pr-5.plan.md"


class TestEmitPlanPromptTemplate:
    """Smoke tests: ensure the prompt template has the placeholders we pass."""

    def test_prompt_has_required_placeholders(self):
        from prompts import EMIT_PLAN_PROMPT

        assert "{pr_label}" in EMIT_PLAN_PROMPT
        assert "{pr_scope}" in EMIT_PLAN_PROMPT
        assert "{title_hint}" in EMIT_PLAN_PROMPT
        assert "{spec}" in EMIT_PLAN_PROMPT

    def test_prompt_formats_without_keyerror(self):
        from prompts import EMIT_PLAN_PROMPT

        out = EMIT_PLAN_PROMPT.format(
            pr_label="PR-1",
            pr_scope="data engine",
            title_hint="My Feature",
            spec="# Spec\n\nBody.",
        )
        assert "PR-1" in out
        assert "data engine" in out
        assert "My Feature" in out
        assert "# Spec" in out

    def test_prompt_requires_spec_citation_not_duplication(self):
        from prompts import EMIT_PLAN_PROMPT

        assert "Cite, don't duplicate" in EMIT_PLAN_PROMPT
        assert "REQUIRED SUB-SKILL" in EMIT_PLAN_PROMPT
        assert "superpowers:executing-plans" in EMIT_PLAN_PROMPT


class TestStripSpecSuffix:
    def test_strips_spec_debate_final(self):
        assert _strip_spec_suffix("foo.spec-debate-final") == "foo"

    def test_strips_spec(self):
        assert _strip_spec_suffix("foo.spec") == "foo"

    def test_strips_dash_spec(self):
        assert _strip_spec_suffix("foo-spec") == "foo"

    def test_strips_dash_final(self):
        assert _strip_spec_suffix("foo-final") == "foo"

    def test_returns_unchanged_when_no_match(self):
        assert _strip_spec_suffix("plain-name") == "plain-name"

    def test_only_strips_once(self):
        # foo.spec-debate-final ends in both patterns; longest wins (ordered first)
        assert _strip_spec_suffix("foo.spec-debate-final") == "foo"


class TestDeriveTitleHint:
    def test_none_path_returns_feature(self):
        assert _derive_title_hint(None) == "Feature"

    def test_strips_suffix_before_converting_separators(self):
        # Regression: dots in .spec-debate-final used to survive into the title
        assert _derive_title_hint("2026-04-22-feature.spec-debate-final.md") == "2026 04 22 feature"

    def test_underscores_and_dashes_become_spaces(self):
        assert _derive_title_hint("my_cool-feature.md") == "my cool feature"

    def test_plain_name(self):
        assert _derive_title_hint("feature.md") == "feature"

    def test_empty_stem_falls_back_to_feature(self):
        # stem after stripping could be empty in pathological case
        assert _derive_title_hint(".spec.md") == "Feature"


class TestHandleDetectPrs:
    def _args(self, spec: str | None = None, json: bool = False) -> argparse.Namespace:
        return argparse.Namespace(spec=spec, json=json)

    def test_text_output_one_per_line(self, tmp_path, capsys):
        spec_file = tmp_path / "s.md"
        spec_file.write_text("PR-1 and PR-2 and PR-10.")

        handle_detect_prs(self._args(spec=str(spec_file)))

        out = capsys.readouterr().out.strip().splitlines()
        assert out == ["PR-1", "PR-2", "PR-10"]

    def test_json_output(self, tmp_path, capsys):
        spec_file = tmp_path / "s.md"
        spec_file.write_text("PR-3 PR-1 PR-2")

        handle_detect_prs(self._args(spec=str(spec_file), json=True))

        out = capsys.readouterr().out.strip()
        assert out == '["PR-1", "PR-2", "PR-3"]'

    def test_empty_spec_prints_nothing(self, tmp_path, capsys):
        spec_file = tmp_path / "s.md"
        spec_file.write_text("no PR markers here")

        handle_detect_prs(self._args(spec=str(spec_file)))

        assert capsys.readouterr().out == ""

    def test_stdin_fallback(self, monkeypatch, capsys):
        import io

        monkeypatch.setattr("sys.stdin", io.StringIO("PR-1 content"))
        handle_detect_prs(self._args(spec=None))

        assert capsys.readouterr().out.strip() == "PR-1"

    def test_missing_file_exits(self, capsys):
        with pytest.raises(SystemExit) as exc:
            handle_detect_prs(self._args(spec="/nonexistent/path.md"))
        assert exc.value.code == 1


class TestHandleEmitPlanIntegration:
    """Integration tests for handle_emit_plan with mocked model call."""

    def _make_response(self, content: str = "# Plan\n\nTask 1...", prompt_tokens: int = 100, completion_tokens: int = 200):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = content
        resp.usage.prompt_tokens = prompt_tokens
        resp.usage.completion_tokens = completion_tokens
        return resp

    def _args(self, **overrides) -> argparse.Namespace:
        defaults = {
            "spec": None,
            "pr_label": "PR-1",
            "pr_scope": None,
            "title_hint": "",
            "plan_out": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_writes_plan_to_default_sibling_path(self, tmp_path, monkeypatch):
        spec_file = tmp_path / "feature.spec-debate-final.md"
        spec_file.write_text("# Spec\n\nPR-1 stuff.")

        mock_completion = MagicMock(return_value=self._make_response("# Emitted plan body"))
        monkeypatch.setattr(debate, "completion", mock_completion)
        monkeypatch.setattr(debate, "cost_tracker", MagicMock())
        monkeypatch.setattr(debate, "is_reasoning_model", lambda m: False)

        handle_emit_plan(self._args(spec=str(spec_file)), ["claude-opus-4-6"])

        expected = tmp_path / "feature-pr-1.plan.md"
        assert expected.exists()
        assert expected.read_text() == "# Emitted plan body\n"

    def test_passes_formatted_prompt_to_completion(self, tmp_path, monkeypatch):
        spec_file = tmp_path / "spec.md"
        spec_file.write_text("SPEC BODY TOKEN")

        mock_completion = MagicMock(return_value=self._make_response())
        monkeypatch.setattr(debate, "completion", mock_completion)
        monkeypatch.setattr(debate, "cost_tracker", MagicMock())
        monkeypatch.setattr(debate, "is_reasoning_model", lambda m: False)

        handle_emit_plan(
            self._args(
                spec=str(spec_file),
                pr_label="PR-2",
                pr_scope="extraction layer",
                title_hint="MyFeature",
            ),
            ["claude-opus-4-6"],
        )

        kwargs = mock_completion.call_args.kwargs
        prompt = kwargs["messages"][0]["content"]
        assert "PR-2" in prompt
        assert "extraction layer" in prompt
        assert "MyFeature" in prompt
        assert "SPEC BODY TOKEN" in prompt
        # Non-reasoning model uses max_tokens + temperature
        assert kwargs["max_tokens"] == debate.EMIT_PLAN_MAX_TOKENS
        assert kwargs["temperature"] == debate.EMIT_PLAN_TEMPERATURE
        assert "max_completion_tokens" not in kwargs

    def test_reasoning_model_uses_max_completion_tokens(self, tmp_path, monkeypatch):
        spec_file = tmp_path / "spec.md"
        spec_file.write_text("body")

        mock_completion = MagicMock(return_value=self._make_response())
        monkeypatch.setattr(debate, "completion", mock_completion)
        monkeypatch.setattr(debate, "cost_tracker", MagicMock())
        monkeypatch.setattr(debate, "is_reasoning_model", lambda m: True)

        handle_emit_plan(self._args(spec=str(spec_file)), ["o1-preview"])

        kwargs = mock_completion.call_args.kwargs
        assert kwargs["max_completion_tokens"] == debate.EMIT_PLAN_MAX_REASONING_TOKENS
        assert "max_tokens" not in kwargs
        assert "temperature" not in kwargs

    def test_empty_spec_exits(self, monkeypatch):
        import io
        monkeypatch.setattr("sys.stdin", io.StringIO(""))

        with pytest.raises(SystemExit) as exc:
            handle_emit_plan(self._args(), ["claude-opus-4-6"])
        assert exc.value.code == 1

    def test_missing_spec_file_exits(self):
        with pytest.raises(SystemExit) as exc:
            handle_emit_plan(
                self._args(spec="/nonexistent/spec.md"),
                ["claude-opus-4-6"],
            )
        assert exc.value.code == 1

    def test_custom_plan_out_overrides_default(self, tmp_path, monkeypatch):
        spec_file = tmp_path / "feature.md"
        spec_file.write_text("body")
        custom_out = tmp_path / "subdir" / "custom-plan.md"

        mock_completion = MagicMock(return_value=self._make_response("CUSTOM"))
        monkeypatch.setattr(debate, "completion", mock_completion)
        monkeypatch.setattr(debate, "cost_tracker", MagicMock())
        monkeypatch.setattr(debate, "is_reasoning_model", lambda m: False)

        handle_emit_plan(
            self._args(spec=str(spec_file), plan_out=str(custom_out)),
            ["claude-opus-4-6"],
        )

        assert custom_out.exists()
        assert custom_out.read_text() == "CUSTOM\n"

    def test_completion_exception_exits(self, tmp_path, monkeypatch):
        spec_file = tmp_path / "spec.md"
        spec_file.write_text("body")

        def boom(**_kwargs):
            raise RuntimeError("API down")

        monkeypatch.setattr(debate, "completion", boom)
        monkeypatch.setattr(debate, "cost_tracker", MagicMock())
        monkeypatch.setattr(debate, "is_reasoning_model", lambda m: False)

        with pytest.raises(SystemExit) as exc:
            handle_emit_plan(self._args(spec=str(spec_file)), ["claude-opus-4-6"])
        assert exc.value.code == 1
