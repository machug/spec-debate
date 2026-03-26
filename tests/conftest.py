"""Shared fixtures for spec-debate tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts dir to path so we can import the modules under test
SCRIPTS_DIR = Path(__file__).parent.parent / "skills" / "spec-debate" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_spec_response():
    """Model response with critique and [SPEC] block."""
    return (
        "The spec has several gaps:\n"
        "1. No rate limiting strategy defined\n"
        "2. Missing error handling for Redis connection failures\n\n"
        "[SPEC]\n"
        "# Rate Limiter Service\n\n"
        "## Overview\n"
        "A Redis-backed rate limiting service.\n\n"
        "## Error Handling\n"
        "On Redis connection failure, fall back to in-memory limiter.\n"
        "[/SPEC]"
    )


@pytest.fixture
def sample_agree_response():
    """Model response indicating agreement."""
    return (
        "The spec now addresses all my concerns. The error handling "
        "strategy is sound and the rate limiting algorithm is well-defined.\n\n"
        "[AGREE]"
    )


@pytest.fixture
def sample_tasks_output():
    """Raw text with [TASK] blocks."""
    return (
        "Here are the tasks:\n\n"
        "[TASK]\n"
        "title: Implement Redis connection pool\n"
        "type: backend\n"
        "priority: high\n"
        "description: Create a connection pool manager for Redis\n"
        "with configurable pool size and timeout.\n"
        "acceptance_criteria:\n"
        "- Pool size is configurable\n"
        "- Connections are reused\n"
        "- Timeout on connection failure\n"
        "[/TASK]\n\n"
        "[TASK]\n"
        "title: Add rate limit headers\n"
        "type: backend\n"
        "priority: medium\n"
        "description: Return X-RateLimit-* headers in responses\n"
        "acceptance_criteria:\n"
        "- X-RateLimit-Limit header present\n"
        "- X-RateLimit-Remaining header present\n"
        "[/TASK]\n"
    )
