"""Tests for model utilities: reasoning detection, CostTracker, response parsing."""

from __future__ import annotations

import pytest
from models import (
    CostTracker,
    detect_agreement,
    extract_spec,
    extract_tasks,
    generate_diff,
    get_critique_summary,
    is_reasoning_model,
)


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
        assert not is_reasoning_model("gemini/gemini-2.5-flash")
        assert not is_reasoning_model("xai/grok-4-1-fast")
        assert not is_reasoning_model("mistral/mistral-large")

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
