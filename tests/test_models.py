"""Tests for model utilities: reasoning detection, CostTracker, response parsing."""

from __future__ import annotations

import pytest

import models
from unittest.mock import MagicMock
from models import (
    CostTracker,
    detect_agreement,
    extract_spec,
    extract_tasks,
    generate_diff,
    get_critique_summary,
    gpt5_tuning_params,
    dropped_headings,
    has_unterminated_fence,
    is_reasoning_model,
    markdown_headings,
    should_warn_missing_spec,
    strip_spec_block,
    warn_dropped_headings,
)
from prompts import get_system_prompt


# ---------------------------------------------------------------------------
# is_reasoning_model
# ---------------------------------------------------------------------------


class TestIsReasoningModel:
    def test_o1_models(self):
        assert is_reasoning_model("o1")
        assert is_reasoning_model("o1-preview")

    def test_o3_models(self):
        assert is_reasoning_model("o3")
        assert is_reasoning_model("o3-mini")
        assert is_reasoning_model("o3-pro")

    def test_o4_models(self):
        assert is_reasoning_model("o4-mini")

    def test_o_series_with_provider_prefix(self):
        assert is_reasoning_model("openai/o3")
        assert is_reasoning_model("openrouter/openai/o3-mini")

    def test_gpt5_family(self):
        assert is_reasoning_model("gpt-5.4")
        assert is_reasoning_model("gpt-5.4-pro")
        assert is_reasoning_model("gpt-5-mini")
        assert is_reasoning_model("gpt-5-nano")

    def test_gpt5_with_prefix(self):
        assert is_reasoning_model("openrouter/openai/gpt-5.4")

    def test_regular_models_are_not_reasoning(self):
        assert not is_reasoning_model("claude-opus-4-6")
        assert not is_reasoning_model("claude-sonnet-4-6")
        assert not is_reasoning_model("gemini/gemini-3.1-pro-preview")
        assert not is_reasoning_model("xai/grok-4-1-fast-non-reasoning")
        assert not is_reasoning_model("xai/grok-4.20-0309-non-reasoning")
        assert not is_reasoning_model("mistral/mistral-large")

    def test_xai_reasoning_models(self):
        assert is_reasoning_model("xai/grok-4-1-fast-reasoning")
        assert is_reasoning_model("xai/grok-4-fast-reasoning")
        assert is_reasoning_model("xai/grok-4.20-0309-reasoning")
        assert not is_reasoning_model("xai/grok-4-0709")
        assert not is_reasoning_model("xai/grok-4-1-fast-non-reasoning")

    def test_moonshot_kimi_reasoning(self):
        assert is_reasoning_model("moonshot/kimi-k2.5")
        assert is_reasoning_model("moonshot/kimi-k2.6")
        assert is_reasoning_model("moonshot/kimi-k2.7-code")
        assert is_reasoning_model("moonshot/kimi-k2.7-code-highspeed")
        assert is_reasoning_model("moonshot/kimi-k3")
        # Future versions past k3 stay covered
        assert is_reasoning_model("moonshot/kimi-k3.1")
        assert is_reasoning_model("moonshot/kimi-k4")

    def test_moonshot_kimi_non_reasoning(self):
        # Pre-2.5 generations accepted temperature
        assert not is_reasoning_model("moonshot/kimi-k2")
        assert not is_reasoning_model("moonshot/kimi-k2-thinking")
        assert not is_reasoning_model("moonshot/kimi-k1.5")
        # No version segment after "kimi-k" — must not match
        assert not is_reasoning_model("moonshot/kimi-latest")
        # "k3" as an arbitrary substring must not match
        assert not is_reasoning_model("moonshot/kimi-rk3-mini")

    def test_grok_45_not_reasoning(self):
        # grok-4.5 returns reasoning_content but accepts temperature —
        # treat as standard so temperature/max_tokens handling applies
        assert not is_reasoning_model("xai/grok-4.5")

    def test_case_insensitive(self):
        assert is_reasoning_model("GPT-5.4")
        assert is_reasoning_model("O3-Mini")


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


class TestCostTracker:
    def test_add_accumulates(self):
        ct = CostTracker()
        ct.add("gpt-5.4", 1000, 500)
        ct.add("gpt-5.4", 2000, 1000)
        assert ct.total_input_tokens == 3000
        assert ct.total_output_tokens == 1500
        assert ct.total_cost > 0

    def test_multiple_models(self):
        ct = CostTracker()
        ct.add("gpt-5.4", 1000, 500)
        ct.add("claude-opus-4-6", 1000, 500)
        assert len(ct.by_model) == 2
        assert "gpt-5.4" in ct.by_model
        assert "claude-opus-4-6" in ct.by_model

    def test_add_returns_cost(self):
        ct = CostTracker()
        cost = ct.add("codex/gpt-5.3-codex", 1000, 500)
        # Codex is free (subscription-based)
        assert cost == 0.0

    def test_summary_format(self):
        ct = CostTracker()
        ct.add("gpt-5.4", 1000, 500)
        summary = ct.summary()
        assert "Cost Summary" in summary
        assert "Total tokens" in summary
        assert "Total cost" in summary

    def test_summary_shows_breakdown_for_multiple_models(self):
        ct = CostTracker()
        ct.add("gpt-5.4", 1000, 500)
        ct.add("claude-opus-4-6", 2000, 1000)
        summary = ct.summary()
        assert "By model:" in summary
        assert "gpt-5.4" in summary
        assert "claude-opus-4-6" in summary


# ---------------------------------------------------------------------------
# detect_agreement
# ---------------------------------------------------------------------------


class TestDetectAgreement:
    def test_agree_present(self, sample_agree_response):
        assert detect_agreement(sample_agree_response) is True

    def test_agree_absent(self, sample_spec_response):
        assert detect_agreement(sample_spec_response) is False

    def test_empty_string(self):
        assert detect_agreement("") is False

    def test_agree_in_spec_block(self):
        # [AGREE] inside a spec block should still be detected
        assert detect_agreement("[SPEC]something[/SPEC]\n[AGREE]") is True


# ---------------------------------------------------------------------------
# extract_spec
# ---------------------------------------------------------------------------


class TestExtractSpec:
    def test_basic_extraction(self, sample_spec_response):
        spec = extract_spec(sample_spec_response)
        assert spec is not None
        assert "Rate Limiter Service" in spec
        assert "Error Handling" in spec

    def test_no_spec_tags(self):
        assert extract_spec("Just a critique with no spec") is None

    def test_empty_string(self):
        assert extract_spec("") is None

    def test_missing_closing_tag(self):
        assert extract_spec("[SPEC]\nsome content") is None

    def test_missing_opening_tag(self):
        assert extract_spec("some content\n[/SPEC]") is None

    def test_empty_spec(self):
        result = extract_spec("[SPEC]\n[/SPEC]")
        assert result == ""

    def test_whitespace_stripped(self):
        result = extract_spec("[SPEC]\n  hello world  \n[/SPEC]")
        assert result == "hello world"

    def test_multiline_content(self):
        result = extract_spec("[SPEC]\nline 1\nline 2\nline 3\n[/SPEC]")
        assert "line 1" in result
        assert "line 3" in result


# ---------------------------------------------------------------------------
# extract_tasks
# ---------------------------------------------------------------------------


class TestExtractTasks:
    def test_basic_extraction(self, sample_tasks_output):
        tasks = extract_tasks(sample_tasks_output)
        assert len(tasks) == 2

    def test_task_fields(self, sample_tasks_output):
        tasks = extract_tasks(sample_tasks_output)
        task = tasks[0]
        assert task["title"] == "Implement Redis connection pool"
        assert task["type"] == "backend"
        assert task["priority"] == "high"
        assert "connection pool" in task["description"]

    def test_acceptance_criteria_is_list(self, sample_tasks_output):
        tasks = extract_tasks(sample_tasks_output)
        criteria = tasks[0]["acceptance_criteria"]
        assert isinstance(criteria, list)
        assert len(criteria) == 3
        assert "Pool size is configurable" in criteria

    def test_empty_input(self):
        assert extract_tasks("") == []

    def test_no_task_blocks(self):
        assert extract_tasks("Just some text without tasks") == []

    def test_missing_closing_tag(self):
        assert extract_tasks("[TASK]\ntitle: Incomplete task\n") == []

    def test_missing_title_skipped(self):
        tasks = extract_tasks("[TASK]\ntype: backend\npriority: high\n[/TASK]")
        assert len(tasks) == 0

    def test_minimal_task(self):
        tasks = extract_tasks("[TASK]\ntitle: Simple task\n[/TASK]")
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Simple task"


# ---------------------------------------------------------------------------
# get_critique_summary
# ---------------------------------------------------------------------------


class TestGetCritiqueSummary:
    def test_extracts_before_spec(self, sample_spec_response):
        summary = get_critique_summary(sample_spec_response)
        assert "gaps" in summary
        assert "[SPEC]" not in summary

    def test_no_spec_returns_full(self):
        text = "This is just a critique"
        assert get_critique_summary(text) == text

    def test_truncation(self):
        long_text = "x" * 500
        summary = get_critique_summary(long_text, max_length=100)
        assert len(summary) == 103  # 100 + "..."
        assert summary.endswith("...")

    def test_empty_string(self):
        assert get_critique_summary("") == ""


# ---------------------------------------------------------------------------
# generate_diff
# ---------------------------------------------------------------------------


class TestGenerateDiff:
    def test_identical_specs(self):
        assert generate_diff("same\n", "same\n") == ""

    def test_added_line(self):
        diff = generate_diff("line 1\n", "line 1\nline 2\n")
        assert "+line 2" in diff

    def test_removed_line(self):
        diff = generate_diff("line 1\nline 2\n", "line 1\n")
        assert "-line 2" in diff

    def test_changed_line(self):
        diff = generate_diff("old text\n", "new text\n")
        assert "-old text" in diff
        assert "+new text" in diff


# ---------------------------------------------------------------------------
# gpt5_tuning_params
# ---------------------------------------------------------------------------


class TestGpt5TuningParams:
    def test_non_gpt5_returns_empty(self):
        assert gpt5_tuning_params("claude-opus-4-7") == {}
        assert gpt5_tuning_params("gemini/gemini-3.1-pro-preview") == {}
        assert gpt5_tuning_params("xai/grok-4.3") == {}
        assert gpt5_tuning_params("o3-mini") == {}

    def test_gpt5_sets_low_verbosity(self):
        params = gpt5_tuning_params("gpt-5.5")
        assert params["extra_body"]["text"]["verbosity"] == "low"

    def test_gpt5_non_pro_sets_medium_reasoning_effort(self):
        params = gpt5_tuning_params("gpt-5.5")
        assert params["reasoning_effort"] == "medium"

    def test_gpt56_variants_covered(self):
        for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            params = gpt5_tuning_params(model)
            assert params["extra_body"]["text"]["verbosity"] == "low"
            assert params["reasoning_effort"] == "medium"

    def test_gpt5_mini_sets_medium_reasoning_effort(self):
        params = gpt5_tuning_params("gpt-5-mini")
        assert params["reasoning_effort"] == "medium"

    def test_pro_keeps_default_reasoning_effort(self):
        # Pro is a deep reasoner; don't force medium effort. Still cap verbosity.
        params = gpt5_tuning_params("gpt-5.5-pro")
        assert "reasoning_effort" not in params
        assert params["extra_body"]["text"]["verbosity"] == "low"

    def test_works_with_provider_prefix(self):
        params = gpt5_tuning_params("openrouter/openai/gpt-5.5")
        assert params["reasoning_effort"] == "medium"
        assert params["extra_body"]["text"]["verbosity"] == "low"

    def test_pro_with_provider_prefix(self):
        params = gpt5_tuning_params("openrouter/openai/gpt-5.5-pro")
        assert "reasoning_effort" not in params

    def test_case_insensitive(self):
        params = gpt5_tuning_params("GPT-5.5")
        assert params["reasoning_effort"] == "medium"


# ---------------------------------------------------------------------------
# should_warn_missing_spec
# ---------------------------------------------------------------------------


class TestShouldWarnMissingSpec:
    def test_warns_when_no_agree_and_no_spec(self):
        assert should_warn_missing_spec(agreed=False, extracted=None, review_only=False)

    def test_no_warn_when_agreed(self):
        assert not should_warn_missing_spec(agreed=True, extracted=None, review_only=False)

    def test_no_warn_when_spec_present(self):
        assert not should_warn_missing_spec(agreed=False, extracted="some spec", review_only=False)

    def test_review_only_never_warns(self):
        # Reviewers are not expected to re-emit a spec.
        assert not should_warn_missing_spec(agreed=False, extracted=None, review_only=True)


# ---------------------------------------------------------------------------
# get_system_prompt — review-only mode
# ---------------------------------------------------------------------------


class TestReviewOnlyPrompt:
    def test_review_only_instructs_no_reemit(self):
        prompt = get_system_prompt("tech", review_only=True)
        lowered = prompt.lower()
        assert "[agree]" in lowered
        assert "review-only" in lowered
        # Must tell the model NOT to reproduce the spec.
        assert "do not reproduce" in lowered or "do not output the document" in lowered

    def test_review_only_works_for_prd(self):
        prompt = get_system_prompt("prd", review_only=True)
        assert "[AGREE]" in prompt

    def test_default_still_requests_spec_tags(self):
        prompt = get_system_prompt("tech")
        assert "[SPEC]" in prompt

    def test_review_only_overrides_persona(self):
        # Even with a persona, review-only must apply the no-re-emit directive.
        prompt = get_system_prompt("tech", persona="security-engineer", review_only=True)
        assert "REVIEW-ONLY" in prompt


# ---------------------------------------------------------------------------
# get_system_prompt — press + review-only (spec-debate-294)
# ---------------------------------------------------------------------------


class TestPressReviewOnlyPrompt:
    """--press must not be cancelled by the review-only system prompt."""

    def test_press_review_only_does_not_forbid_extra_output(self):
        # The plain reviewer suffix says "Nothing else", which cancels press.
        plain = get_system_prompt("tech", review_only=True)
        assert "Nothing else" in plain
        pressed = get_system_prompt("tech", review_only=True, press=True)
        assert "Nothing else" not in pressed

    def test_press_review_only_demands_a_section_list(self):
        pressed = get_system_prompt("tech", review_only=True, press=True)
        lowered = pressed.lower()
        assert "bare [agree] is not an acceptable answer" in lowered
        assert "three named sections" in lowered

    def test_press_review_only_still_forbids_reemitting_the_spec(self):
        pressed = get_system_prompt("tech", review_only=True, press=True)
        assert "REVIEW-ONLY" in pressed
        assert "Do NOT use [SPEC] tags" in pressed

    def test_press_without_review_only_is_unchanged(self):
        assert get_system_prompt("tech", press=True) == get_system_prompt("tech")


# ---------------------------------------------------------------------------
# preserve-intent heading backstop (spec-debate-xp7)
# ---------------------------------------------------------------------------


EVIDENCE_SPEC = """# Warehouse Sync

## Findings

### F1

## Decisions

## Design
"""


class TestDroppedHeadings:
    def test_identical_document_drops_nothing(self):
        assert dropped_headings(EVIDENCE_SPEC, EVIDENCE_SPEC) == []

    def test_added_sections_are_allowed(self):
        revised = EVIDENCE_SPEC + "\n## Open Questions\n"
        assert dropped_headings(EVIDENCE_SPEC, revised) == []

    def test_generic_template_rewrite_is_reported(self):
        # The observed failure: structure replaced wholesale, findings gone.
        revised = "# Warehouse Sync\n\n## Overview\n\n## Goals\n\n## Non-Goals\n"
        assert dropped_headings(EVIDENCE_SPEC, revised) == [
            "Findings",
            "Decisions",
            "Design",
        ]

    def test_ignores_third_level_headings(self):
        # F1 is an h3; only h1/h2 are tracked.
        revised = "# Warehouse Sync\n\n## Findings\n\n## Decisions\n\n## Design\n"
        assert dropped_headings(EVIDENCE_SPEC, revised) == []

    def test_reports_each_dropped_heading_once(self):
        original = "## Findings\n\n## Findings\n\n## Design\n"
        assert dropped_headings(original, "## Design\n") == ["Findings"]

    def test_warn_prints_to_stderr(self, capsys):
        revised = "# Warehouse Sync\n\n## Overview\n"
        missing = warn_dropped_headings("xai/grok-4.6", EVIDENCE_SPEC, revised)
        assert missing == ["Findings", "Decisions", "Design"]
        err = capsys.readouterr().err
        assert "preserve-intent" in err
        assert "Findings" in err

    def test_warn_is_silent_when_nothing_dropped(self, capsys):
        assert warn_dropped_headings("m", EVIDENCE_SPEC, EVIDENCE_SPEC) == []
        assert capsys.readouterr().err == ""


# ---------------------------------------------------------------------------
# critique prose persistence (spec-debate-px2)
# ---------------------------------------------------------------------------


class TestStripSpecBlock:
    def test_keeps_critique_and_drops_the_reemitted_document(self):
        response = "1. Missing rate limits\n2. No retries\n\n[SPEC]\n# Doc\n[/SPEC]"
        assert strip_spec_block(response) == "1. Missing rate limits\n2. No retries"

    def test_response_without_spec_tags_is_kept_whole(self, sample_agree_response):
        assert strip_spec_block(sample_agree_response) == sample_agree_response.strip()

    def test_empty_response_stays_empty(self):
        assert strip_spec_block("") == ""


# ---------------------------------------------------------------------------
# code-fence tracking (shared by the plan truncation check and heading diff)
# ---------------------------------------------------------------------------


class TestHasUnterminatedFence:
    def test_balanced_fences_are_closed(self):
        assert not has_unterminated_fence("# Doc\n\n```bash\nls\n```\n")

    def test_open_fence_at_eof_is_detected(self):
        assert has_unterminated_fence("# Doc\n\n```bash\nls\n")

    def test_fence_inside_a_string_literal_does_not_count(self):
        body = '# Doc\n\n```python\nx = "```"\n```\n'
        assert body.count("```") % 2 == 1
        assert not has_unterminated_fence(body)

    def test_four_backticks_may_wrap_three(self):
        assert not has_unterminated_fence("````md\n```sh\nls\n```\n````\n")

    def test_tilde_fence_is_tracked(self):
        assert has_unterminated_fence("~~~python\nx = 1\n")

    def test_backticks_do_not_close_a_tilde_fence(self):
        assert has_unterminated_fence("~~~python\nx = 1\n```\n")

    def test_no_fences_at_all(self):
        assert not has_unterminated_fence("# Doc\n\nJust prose.\n")


# ---------------------------------------------------------------------------
# heading extraction edge cases
# ---------------------------------------------------------------------------


class TestMarkdownHeadings:
    def test_hash_comments_inside_a_code_block_are_not_headings(self):
        body = "# Real\n\n```bash\n# install deps\n## run it\nnpm i\n```\n\n## Also Real\n"
        assert markdown_headings(body) == ["Real", "Also Real"]

    def test_closed_atx_heading_normalizes(self):
        assert markdown_headings("## Design ##\n") == ["Design"]

    def test_setext_headings_are_found(self):
        assert markdown_headings("Findings\n========\n\nDecisions\n---------\n") == [
            "Findings",
            "Decisions",
        ]

    def test_third_level_headings_are_ignored(self):
        assert markdown_headings("### F1\n") == []

    def test_indented_hash_beyond_three_spaces_is_not_a_heading(self):
        assert markdown_headings("    # not a heading\n") == []


class TestDroppedHeadingsFalsePositives:
    def test_rewriting_only_a_code_block_reports_nothing(self):
        original = "# Doc\n\n```bash\n# install deps\n```\n\n## Design\n"
        revised = "# Doc\n\n```bash\n# install everything\n```\n\n## Design\n"
        assert dropped_headings(original, revised) == []

    def test_closing_hashes_are_not_a_dropped_heading(self):
        assert dropped_headings("## Design ##\n", "## Design\n") == []


# ---------------------------------------------------------------------------
# strip_spec_block edge cases
# ---------------------------------------------------------------------------


class TestStripSpecBlockEdgeCases:
    def test_prose_mention_of_the_tag_does_not_eat_the_critique(self):
        # The press prompt puts the literal "[SPEC]" in the user message, so
        # models echo it. Splitting on the first occurrence loses everything.
        response = (
            "You asked for the doc between [SPEC] and [/SPEC] tags. My concerns:\n"
            "1. No retry budget\n\n"
            "[SPEC]\n# Doc\n[/SPEC]"
        )
        out = strip_spec_block(response)
        assert "No retry budget" in out
        assert "# Doc" not in out

    def test_critique_after_the_closing_tag_survives(self):
        response = "Before.\n[SPEC]\n# Doc\n[/SPEC]\nAfter: one more concern."
        out = strip_spec_block(response)
        assert "Before." in out
        assert "After: one more concern." in out
        assert "# Doc" not in out

    def test_unclosed_spec_tag_is_cut_from_the_last_opening(self):
        response = "Critique stands.\n\n[SPEC]\n# Half a doc that got cut off"
        assert strip_spec_block(response) == "Critique stands."


# ---------------------------------------------------------------------------
# wiring: the fixes must actually be reached from the call path
# ---------------------------------------------------------------------------


class TestPressReachesTheSystemPrompt:
    """Regression guard for the original bug: press never reached judge mode.

    Testing get_system_prompt alone would not catch it — the v1.12.0 defect was
    the call site dropping the argument, not the prompt text.
    """

    def _stub_response(self):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "[AGREE]"
        resp.choices[0].finish_reason = "stop"
        resp.usage.prompt_tokens = 10
        resp.usage.completion_tokens = 5
        return resp

    def _spy(self, monkeypatch):
        seen = {}

        def spy(doc_type, persona=None, review_only=False, press=False):
            seen["review_only"] = review_only
            seen["press"] = press
            return "SYSTEM"

        monkeypatch.setattr(models, "get_system_prompt", spy)
        monkeypatch.setattr(
            models, "completion", MagicMock(return_value=self._stub_response())
        )
        monkeypatch.setattr(models, "cost_tracker", MagicMock())
        return seen

    def test_press_and_review_only_both_reach_the_system_prompt(self, monkeypatch):
        seen = self._spy(monkeypatch)
        models.call_single_model(
            "gpt-4o", "# Spec", 5, "tech", press=True, review_only=True
        )
        assert seen == {"review_only": True, "press": True}

    def test_defaults_are_passed_through_unchanged(self, monkeypatch):
        seen = self._spy(monkeypatch)
        models.call_single_model("gpt-4o", "# Spec", 1, "tech")
        assert seen == {"review_only": False, "press": False}

    def test_pressed_judge_user_message_does_not_request_spec_tags(self, monkeypatch):
        # The user message must not contradict the reviewer suffix.
        mock = MagicMock(return_value=self._stub_response())
        monkeypatch.setattr(models, "completion", mock)
        monkeypatch.setattr(models, "cost_tracker", MagicMock())
        models.call_single_model(
            "gpt-4o", "# Spec", 5, "tech", press=True, review_only=True
        )
        user_message = mock.call_args.kwargs["messages"][1]["content"]
        assert "between [SPEC] and [/SPEC] tags" not in user_message
        assert "List at least 3 specific sections" in user_message

    def test_pressed_editor_still_requests_spec_tags(self, monkeypatch):
        # Without review_only the model IS the editor and must re-emit.
        mock = MagicMock(return_value=self._stub_response())
        monkeypatch.setattr(models, "completion", mock)
        monkeypatch.setattr(models, "cost_tracker", MagicMock())
        models.call_single_model("gpt-4o", "# Spec", 5, "tech", press=True)
        user_message = mock.call_args.kwargs["messages"][1]["content"]
        assert "between [SPEC] and [/SPEC] tags" in user_message


class TestPreserveIntentBackstopIsWired:
    """The heading diff must run from call_models_parallel, not just exist."""

    def _result(self, spec):
        return models.ModelResponse(
            model="stub", response="critique", agreed=False, spec=spec
        )

    def test_warns_when_a_revision_drops_headings(self, monkeypatch, capsys):
        monkeypatch.setattr(
            models,
            "call_single_model",
            lambda *a, **k: self._result("# Doc\n\n## Overview\n"),
        )
        models.call_models_parallel(
            ["stub"], "# Doc\n\n## Findings\n", 1, "tech", preserve_intent=True
        )
        assert "Findings" in capsys.readouterr().err

    def test_silent_when_preserve_intent_is_off(self, monkeypatch, capsys):
        monkeypatch.setattr(
            models,
            "call_single_model",
            lambda *a, **k: self._result("# Doc\n\n## Overview\n"),
        )
        models.call_models_parallel(["stub"], "# Doc\n\n## Findings\n", 1, "tech")
        assert "Findings" not in capsys.readouterr().err

    def test_silent_when_the_model_agreed_without_re_emitting(
        self, monkeypatch, capsys
    ):
        monkeypatch.setattr(
            models, "call_single_model", lambda *a, **k: self._result(None)
        )
        models.call_models_parallel(
            ["stub"], "# Doc\n\n## Findings\n", 1, "tech", preserve_intent=True
        )
        assert "Findings" not in capsys.readouterr().err
