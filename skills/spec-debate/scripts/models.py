"""Model calling, cost tracking, and response handling."""

from __future__ import annotations

import concurrent.futures
import difflib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ["LITELLM_LOG"] = "ERROR"

try:
    import litellm
    from litellm import completion

    litellm.suppress_debug_info = True
except ImportError:
    print(
        "Error: litellm package not installed. Run: pip install litellm",
        file=sys.stderr,
    )
    sys.exit(1)

from prompts import (
    FOCUS_AREAS,
    PRESERVE_INTENT_PROMPT,
    PRESS_PROMPT_TEMPLATE,
    REVIEW_PROMPT_TEMPLATE,
    get_doc_type_name,
    PRESS_REVIEW_ONLY_PROMPT_TEMPLATE,
    get_system_prompt,
)
from providers import (
    ANTIGRAVITY_AVAILABLE,
    ANTIGRAVITY_PATH,
    CODEX_AVAILABLE,
    CODEX_PATH,
    DEFAULT_CLAUDE_EFFORT,
    DEFAULT_CODEX_REASONING,
    GEMINI_CLI_AVAILABLE,
    GEMINI_CLI_PATH,
    get_model_cost,
)

MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds

# Error substrings that retrying cannot fix: bad model id, wrong auth mode,
# rejected/revoked credentials. These are deterministic 4xx-class failures —
# retrying just burns time and spams warnings.
NON_RETRYABLE_PATTERNS = (
    "not supported when using codex with a chatgpt account",
    "invalid_request_error",
    "model_not_found",
    "does not exist or you do not have access",
    "authenticationerror",
    "invalid api key",
    "incorrect api key",
    "notfounderror",
    # Antigravity CLI deterministic failures
    "is not authenticated",
    "invalid model selection",
    # Bedrock messages rewritten in the litellm retry loop below
    "model not enabled in your bedrock account",
    "invalid bedrock model id",
)

CODEX_CHATGPT_HINT = (
    "Codex is authenticated with a ChatGPT account, which only serves: "
    "gpt-6-astra (eligible plans), gpt-5.6-sol, gpt-5.6-terra, gpt-5.6-luna, gpt-5.5 (until 2026-10-14) "
    "(gpt-5.3-codex-spark needs ChatGPT Pro; gpt-5.4/-mini retired 2026-08-31). "
    "For other models authenticate Codex with an API key or use the "
    "OPENAI_API_KEY litellm route (e.g. --models gpt-5.5-pro)."
)


def is_non_retryable_error(error_msg: str) -> bool:
    """Whether an error is deterministic (4xx-class) and not worth retrying."""
    lower = error_msg.lower()
    return any(p in lower for p in NON_RETRYABLE_PATTERNS)


# Anthropic models from this version up reject any temperature but 1
# (verified 2026-08-31: claude-opus-4-7/-4-8, claude-opus-5, claude-sonnet-5 and
# claude-fable-5 all raise UnsupportedParamsError on temperature=0; sonnet-4-6,
# opus-4-6 and haiku-4-5 still accept it).
CLAUDE_FIXED_TEMPERATURE_FROM = (4, 7)

# Effort (output_config.effort) is supported from Claude 4.6 up; every model
# that rejects temperature also takes effort. Debaters re-emit the whole spec,
# so they run at "medium" — Anthropic's documented step-down from the "high"
# default — instead of burning the full output budget on thinking.
CLAUDE_EFFORT_FROM = (4, 6)

# Matches "claude-opus-5", "claude-opus-4-8", "claude-sonnet-4-6-20250627-v1:0",
# "anthropic.claude-opus-4-7-...", "antigravity/claude-sonnet-4-6". Deliberately
# does NOT match the legacy "claude-3-5-sonnet" ordering, which is pre-4.7.
_CLAUDE_VERSION_RE = re.compile(r"claude-(?:opus|sonnet|haiku|fable)-(\d+)(?:[-.](\d+))?")


def claude_version(model: str) -> Optional[tuple[int, int]]:
    """Return (major, minor) for a Claude model id, or None if not one."""
    m = _CLAUDE_VERSION_RE.search(model.lower())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2) or 0))


def is_reasoning_model(model: str) -> bool:
    """
    Check if a model is a reasoning model (o-series, gpt-5/gpt-6, Claude 4.7+).

    Reasoning models differ from standard models:
    - They ignore or reject the temperature parameter (fixed internally)
    - They use max_completion_tokens instead of max_tokens

    Args:
        model: Model identifier string.

    Returns:
        True if the model is a reasoning model.
    """
    model_lower = model.lower()
    # O-series: o1, o3, o4, etc.
    if model_lower.startswith(("o1", "o3", "o4")) or "/o1" in model_lower or "/o3" in model_lower or "/o4" in model_lower:
        return True
    # GPT-5/GPT-6 family: gpt-5, gpt-5.5, gpt-5-mini, gpt-6-astra, etc.
    # (verified 2026-09-22: gpt-6-astra rejects temperature != 1 and max_tokens)
    if "gpt-5" in model_lower or "gpt-6" in model_lower:
        return True
    # xAI reasoning models: grok-*-reasoning but NOT *-non-reasoning
    if "xai/" in model_lower and model_lower.endswith("-reasoning") and not model_lower.endswith("-non-reasoning"):
        return True
    # Moonshot Kimi reasoning models (kimi-k2.5 and later reject temperature,
    # only allow 1). Anchor on the version segment after "kimi-k" so arbitrary
    # "k3" substrings elsewhere in a model id don't match.
    if "moonshot/" in model_lower:
        m = re.search(r"kimi-k(\d+(?:\.\d+)?)", model_lower)
        if m and float(m.group(1)) >= 2.5:
            return True
    # Anthropic Claude 4.7 and newer only accept temperature=1
    version = claude_version(model_lower)
    if version and version >= CLAUDE_FIXED_TEMPERATURE_FROM:
        return True
    return False


def uses_max_completion_tokens(model: str) -> bool:
    """Check if a model uses max_completion_tokens instead of max_tokens.

    Most reasoning models use max_completion_tokens, but some providers
    still use max_tokens (litellm doesn't support max_completion_tokens for them).
    """
    if not is_reasoning_model(model):
        return False
    # xAI, Moonshot and Anthropic use max_tokens even for reasoning models
    if model.lower().startswith(("xai/", "moonshot/")):
        return False
    if claude_version(model):
        return False
    return True


def output_token_budget(model: str) -> int:
    """Output-token budget for a model.

    Reasoning models spend hidden reasoning tokens out of the same budget as
    visible output. If the budget is too low, deep reasoners (notably the
    `-pro` tier) exhaust it on reasoning and the API hard-fails with
    "unable to complete request: max_output_tokens" instead of returning
    truncated text. Give reasoning models — especially pro — far more room.
    """
    model_lower = model.lower()
    if is_reasoning_model(model):
        # Pro/deep reasoners burn the most reasoning tokens before output.
        if "-pro" in model_lower or "pro-" in model_lower:
            return 64000
        return 32000
    return 16000


def gpt5_tuning_params(model: str) -> dict:
    """Output-control params for GPT-5 family models.

    Reasoning models spend hidden reasoning tokens out of the same budget as
    visible output. The proper levers (vs. a prose "be brief" instruction) are:
    - verbosity=low: shortens *visible* output without touching reasoning depth.
    - reasoning_effort=medium: caps reasoning spend for the in-loop debaters so a
      full spec re-emit fits the budget.

    The `-pro` tier is a deep reasoner; forcing medium effort defeats its
    purpose, so pro keeps its default effort and only gets verbosity capped.
    (Pro should generally run as a review-only judge, where it never re-emits.)

    Returns kwargs to merge into the litellm completion call. Empty for
    non-GPT-5 models.
    """
    model_lower = model.lower()
    if "gpt-6" in model_lower:
        # gpt-6-astra (verified 2026-09-22): chat completions reject the
        # `text.verbosity` block ("Unknown parameter: 'text'"), and litellm
        # 1.98 (the requirements.txt floor) does not list reasoning_effort for
        # it (1.102 does), so pass effort through extra_body, which works on
        # both.
        return {"extra_body": {"reasoning_effort": "medium"}}
    if "gpt-5" not in model_lower:
        return {}
    params: dict = {"extra_body": {"text": {"verbosity": "low"}}}
    is_pro = "-pro" in model_lower or "pro-" in model_lower
    if not is_pro:
        params["reasoning_effort"] = "medium"
    return params


def claude_tuning_params(
    model: str, review_only: bool, effort: str = DEFAULT_CLAUDE_EFFORT
) -> dict:
    """Effort control for Claude models, mirroring gpt5_tuning_params.

    Claude 4.6+ defaults to `high` effort (adaptive thinking always on), which
    on a full spec re-emit burns tens of thousands of tokens and minutes of
    wall time. See DEFAULT_CLAUDE_EFFORT for the measured trade-off.

    Judges (review_only) never re-emit the spec, so their output is short
    already — leave them at Anthropic's `high` default, where thoroughness is
    the point.

    LiteLLM maps `reasoning_effort` to Anthropic's `output_config.effort`.
    Returns kwargs to merge into the litellm completion call; empty for
    non-Claude models and for models older than 4.7 (no effort support).
    """
    if review_only:
        return {}
    version = claude_version(model)
    if not version or version < CLAUDE_EFFORT_FROM:
        return {}
    return {"reasoning_effort": effort}


def should_warn_missing_spec(
    agreed: bool, extracted: Optional[str], review_only: bool
) -> bool:
    """Whether to warn that a response lacked [SPEC] tags.

    In review-only mode the model is a judge and is not expected to re-emit the
    spec, so a missing [SPEC] block is normal — never warn. Otherwise warn when
    the model neither agreed nor produced a spec (likely malformed output).
    """
    if review_only:
        return False
    return not agreed and not extracted


@dataclass
class ModelResponse:
    """Response from a model critique."""

    model: str
    response: str
    agreed: bool
    spec: Optional[str]
    error: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass
class CostTracker:
    """Track token usage and costs across model calls."""

    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost: float = 0.0
    by_model: dict = field(default_factory=dict)

    def add(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Add usage for a model call and return the cost."""
        costs = get_model_cost(model)
        cost = (input_tokens / 1_000_000 * costs["input"]) + (
            output_tokens / 1_000_000 * costs["output"]
        )

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost

        if model not in self.by_model:
            self.by_model[model] = {"input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        self.by_model[model]["input_tokens"] += input_tokens
        self.by_model[model]["output_tokens"] += output_tokens
        self.by_model[model]["cost"] += cost

        return cost

    def summary(self) -> str:
        """Generate cost summary string."""
        lines = ["", "=== Cost Summary ==="]
        lines.append(
            f"Total tokens: {self.total_input_tokens:,} in / {self.total_output_tokens:,} out"
        )
        lines.append(f"Total cost: ${self.total_cost:.4f}")
        if len(self.by_model) > 1:
            lines.append("")
            lines.append("By model:")
            for model, data in self.by_model.items():
                lines.append(
                    f"  {model}: ${data['cost']:.4f} ({data['input_tokens']:,} in / {data['output_tokens']:,} out)"
                )
        return "\n".join(lines)


# Global cost tracker instance
cost_tracker = CostTracker()


def load_context_files(context_paths: list[str]) -> str:
    """Load and format context files for inclusion in prompts."""
    if not context_paths:
        return ""

    sections = []
    for path in context_paths:
        try:
            content = Path(path).read_text()
            sections.append(f"### Context: {path}\n```\n{content}\n```")
        except Exception as e:
            sections.append(f"### Context: {path}\n[Error loading file: {e}]")

    return (
        "## Additional Context\nThe following documents are provided as context:\n\n"
        + "\n\n".join(sections)
    )


def detect_agreement(response: str) -> bool:
    """Check if response indicates agreement."""
    return "[AGREE]" in response


def extract_spec(response: str) -> Optional[str]:
    """Extract spec content from [SPEC]...[/SPEC] tags."""
    if "[SPEC]" not in response or "[/SPEC]" not in response:
        return None
    start = response.find("[SPEC]") + len("[SPEC]")
    end = response.find("[/SPEC]")
    return response[start:end].strip()


FENCE_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
ATX_HEADING_RE = re.compile(r"^ {0,3}(#{1,2})[ \t]+(.+?)[ \t]*$")
SETEXT_UNDERLINE_RE = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")


def _fence_state(line: str, open_fence: Optional[str]) -> tuple[Optional[str], bool]:
    """Advance code-fence state for one line.

    Returns (new open fence or None, whether this line was a fence marker).
    """
    m = FENCE_RE.match(line)
    if not m:
        return open_fence, False
    fence = m.group("fence")
    if open_fence is None:
        return fence, True
    closes = (
        fence[0] == open_fence[0]
        and len(fence) >= len(open_fence)
        and not m.group("info").strip()
    )
    return (None if closes else open_fence), True


def has_unterminated_fence(text: str) -> bool:
    """True if the document ends inside an open ``` or ~~~ code fence.

    Counting occurrences of ``` does not work. A fence sequence can appear
    inside a string literal in a code sample (`assert "```" in body`), and
    CommonMark lets a four-backtick fence wrap three-backtick content. Both are
    normal in an emitted plan and both break parity, so a naive count reports a
    complete plan as truncated. Track opens and closes per line instead.
    """
    open_fence = None
    for line in text.split("\n"):
        open_fence, _ = _fence_state(line, open_fence)
    return open_fence is not None


def strip_spec_block(response: str) -> str:
    """Return the model's critique prose with any re-emitted [SPEC] block removed.

    This is the part of a response worth persisting: the argument, not the copy
    of the document the caller already holds.

    Cuts the tagged span rather than splitting on the first "[SPEC]".
    PRESS_PROMPT_TEMPLATE puts the literal string "[SPEC]" in the user message,
    so a model that echoes its instructions before critiquing would otherwise
    lose the entire critique. An unclosed [SPEC] means a truncated re-emit; cut
    from the last opening tag so the prose before it survives.
    """
    cleaned = re.sub(r"\[SPEC\].*?\[/SPEC\]", "", response, flags=re.DOTALL)
    if "[SPEC]" in cleaned and "[/SPEC]" not in cleaned:
        cleaned = cleaned[: cleaned.rindex("[SPEC]")]
    return cleaned.strip()


def _normalize_heading(text: str) -> str:
    """Heading text without a closing ### sequence or runs of whitespace."""
    return re.sub(r"\s+", " ", re.sub(r"[ \t]+#+[ \t]*$", "", text)).strip()


def markdown_headings(text: str) -> list[str]:
    """Top-level (#) and second-level (##) headings, in document order.

    Skips fenced code blocks: `# install deps` inside a bash sample is a
    comment, not a heading, and counting it produces false "dropped heading"
    warnings on the ordinary case — which teaches the reader to ignore the one
    real warning this exists to raise. Handles setext (underlined) headings and
    closed ATX headings too, since a spec written either way needs the same
    protection.
    """
    headings = []
    open_fence = None
    lines = text.split("\n")
    for i, line in enumerate(lines):
        open_fence, was_fence = _fence_state(line, open_fence)
        if was_fence or open_fence is not None:
            continue
        atx = ATX_HEADING_RE.match(line)
        if atx:
            headings.append(_normalize_heading(atx.group(2)))
            continue
        if (
            line.strip()
            and i + 1 < len(lines)
            and SETEXT_UNDERLINE_RE.match(lines[i + 1])
        ):
            headings.append(_normalize_heading(line))
    return headings


def dropped_headings(original: str, revised: str) -> list[str]:
    """Headings present in `original` but missing from `revised`, in order.

    Mechanical backstop for --preserve-intent: a model can obey every wording
    rule in PRESERVE_INTENT_PROMPT and still swap the document's whole skeleton
    for a generic template, taking the evidence sections with it.
    """
    kept = set(markdown_headings(revised))
    seen: set[str] = set()
    missing = []
    for h in markdown_headings(original):
        if h not in kept and h not in seen:
            seen.add(h)
            missing.append(h)
    return missing


def warn_dropped_headings(model: str, original: str, revised: str) -> list[str]:
    """Print a warning naming headings the revision dropped. Returns them."""
    missing = dropped_headings(original, revised)
    if missing:
        shown = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        print(
            f"Warning: {model} dropped {len(missing)} heading(s) from the input "
            f"despite --preserve-intent: {shown}{more}",
            file=sys.stderr,
        )
    return missing


def extract_tasks(response: str) -> list[dict]:
    """Extract tasks from export-tasks response."""
    tasks = []
    parts = response.split("[TASK]")
    for part in parts[1:]:
        if "[/TASK]" not in part:
            continue
        task_text = part.split("[/TASK]")[0].strip()
        task: dict[str, str | list[str]] = {}
        current_key: Optional[str] = None
        current_value: list[str] = []

        for line in task_text.split("\n"):
            line = line.strip()
            if line.startswith("title:"):
                if current_key:
                    task[current_key] = (
                        "\n".join(current_value).strip()
                        if len(current_value) > 1
                        else current_value[0]
                        if current_value
                        else ""
                    )
                current_key = "title"
                current_value = [line[6:].strip()]
            elif line.startswith("type:"):
                if current_key:
                    task[current_key] = (
                        "\n".join(current_value).strip()
                        if len(current_value) > 1
                        else current_value[0]
                        if current_value
                        else ""
                    )
                current_key = "type"
                current_value = [line[5:].strip()]
            elif line.startswith("priority:"):
                if current_key:
                    task[current_key] = (
                        "\n".join(current_value).strip()
                        if len(current_value) > 1
                        else current_value[0]
                        if current_value
                        else ""
                    )
                current_key = "priority"
                current_value = [line[9:].strip()]
            elif line.startswith("description:"):
                if current_key:
                    task[current_key] = (
                        "\n".join(current_value).strip()
                        if len(current_value) > 1
                        else current_value[0]
                        if current_value
                        else ""
                    )
                current_key = "description"
                current_value = [line[12:].strip()]
            elif line.startswith("acceptance_criteria:"):
                if current_key:
                    task[current_key] = (
                        "\n".join(current_value).strip()
                        if len(current_value) > 1
                        else current_value[0]
                        if current_value
                        else ""
                    )
                current_key = "acceptance_criteria"
                current_value = []
            elif line.startswith("- ") and current_key == "acceptance_criteria":
                current_value.append(line[2:])
            elif current_key:
                current_value.append(line)

        if current_key:
            task[current_key] = (
                current_value
                if current_key == "acceptance_criteria"
                else "\n".join(current_value).strip()
            )

        if task.get("title"):
            tasks.append(task)

    return tasks


def get_critique_summary(response: str, max_length: int = 300) -> str:
    """Get a summary of the critique portion of a response."""
    spec_start = response.find("[SPEC]")
    if spec_start > 0:
        critique = response[:spec_start].strip()
    else:
        critique = response

    if len(critique) > max_length:
        critique = critique[:max_length] + "..."
    return critique


def generate_diff(previous: str, current: str) -> str:
    """Generate unified diff between two specs."""
    prev_lines = previous.splitlines(keepends=True)
    curr_lines = current.splitlines(keepends=True)

    diff = difflib.unified_diff(
        prev_lines, curr_lines, fromfile="previous", tofile="current", lineterm=""
    )
    return "".join(diff)


def call_foundry_model(
    system_prompt: str,
    user_message: str,
    model: str,
    timeout: int = 600,
) -> tuple[str, int, int]:
    """Call Azure AI Foundry v2 using the azure-ai-inference SDK.

    Args:
        system_prompt: System instructions for the model.
        user_message: User prompt to send.
        model: Model name with foundry/ prefix (e.g., "foundry/gpt-5-mini").
        timeout: Timeout in seconds.

    Returns:
        Tuple of (response_text, input_tokens, output_tokens).
    """
    from azure.ai.inference import ChatCompletionsClient
    from azure.ai.inference.models import SystemMessage, UserMessage
    from azure.core.credentials import AzureKeyCredential

    api_key = os.environ.get("AZURE_AI_API_KEY")
    api_base = os.environ.get("AZURE_AI_API_BASE", "")

    if not api_key:
        raise ValueError("AZURE_AI_API_KEY environment variable not set")

    # Derive the /models endpoint from the base URL
    endpoint = api_base.rstrip("/")
    if not endpoint.endswith("/models"):
        # Strip project path if present and append /models
        # e.g., https://x.services.ai.azure.com/api/projects/foo -> https://x.services.ai.azure.com/models
        parts = endpoint.split(".services.ai.azure.com")
        if len(parts) == 2:
            endpoint = parts[0] + ".services.ai.azure.com/models"
        else:
            endpoint = endpoint + "/models"

    # Strip foundry/ prefix to get deployment name
    deployment_name = model.split("/", 1)[1] if "/" in model else model

    client = ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
    )

    response = client.complete(
        messages=[
            SystemMessage(content=system_prompt),
            UserMessage(content=user_message),
        ],
        model=deployment_name,
    )

    content = response.choices[0].message.content or ""
    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0

    return content, input_tokens, output_tokens


def call_codex_model(
    system_prompt: str,
    user_message: str,
    model: str,
    reasoning_effort: str = DEFAULT_CODEX_REASONING,
    timeout: int = 600,
    search: bool = False,
) -> tuple[str, int, int]:
    """
    Call Codex CLI in headless mode using ChatGPT subscription.

    Args:
        system_prompt: System instructions for the model
        user_message: User prompt to send
        model: Model name (e.g., "codex/gpt-5.3-codex" -> uses "gpt-5.3-codex")
        reasoning_effort: Thinking level (minimal, low, medium, high, xhigh). Default: xhigh
        timeout: Timeout in seconds (default 10 minutes)
        search: Enable web search capability for Codex

    Returns:
        Tuple of (response_text, input_tokens, output_tokens)

    Raises:
        RuntimeError: If Codex CLI is not available or fails
    """
    if not CODEX_AVAILABLE:
        raise RuntimeError(
            "Codex CLI not found. Install with: npm install -g @openai/codex"
        )

    # Extract actual model name from "codex/model" format
    actual_model = model.split("/", 1)[1] if "/" in model else model

    # Combine system prompt and user message for Codex
    full_prompt = f"""SYSTEM INSTRUCTIONS:
{system_prompt}

USER REQUEST:
{user_message}"""

    try:
        cmd = [
            CODEX_PATH,
            "exec",
            "--json",
            "--sandbox",
            "workspace-write",
            "--skip-git-repo-check",
            "--model",
            actual_model,
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
        ]
        if search:
            cmd.extend(["--enable", "web_search"])
        cmd.append(full_prompt)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )

        # Parse JSONL output to extract agent messages and structured errors.
        # Codex CLI emits API errors as `{"type":"error",...}` events on stdout
        # while stderr carries deprecation warnings and unrelated skill-load
        # noise — prefer the structured error over raw stderr.
        response_text = ""
        input_tokens = 0
        output_tokens = 0
        structured_error: Optional[str] = None

        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue

            event_type = event.get("type")
            if event_type == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    response_text = item.get("text", "")
            elif event_type == "turn.completed":
                usage = event.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
            elif event_type in ("error", "turn.failed"):
                # The "error" field may be a dict, a bare string, or null.
                err = event.get("error")
                msg = event.get("message") or (
                    err.get("message") if isinstance(err, dict) else err
                )
                if msg:
                    structured_error = msg

        if result.returncode != 0 or structured_error:
            error_msg = (
                structured_error
                or result.stderr.strip()
                or f"Codex exited with code {result.returncode}"
            )
            raise RuntimeError(f"Codex CLI failed: {error_msg}")

        if not response_text:
            raise RuntimeError("No agent message found in Codex output")

        return response_text, input_tokens, output_tokens

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Codex CLI timed out after {timeout}s")
    except FileNotFoundError:
        raise RuntimeError("Codex CLI not found in PATH")


# One-shot flag so the retirement notice doesn't repeat on every call and retry.
_gemini_retirement_warned = False


def call_gemini_cli_model(
    system_prompt: str,
    user_message: str,
    model: str,
    timeout: int = 600,
) -> tuple[str, int, int]:
    """
    Call Gemini CLI for model inference using Google account authentication.

    Args:
        system_prompt: System instructions for the model
        user_message: User prompt to send
        model: Model name (e.g., "gemini-cli/gemini-3.1-pro-preview" -> uses "gemini-3.1-pro-preview")
        timeout: Timeout in seconds (default 10 minutes)

    Returns:
        Tuple of (response_text, input_tokens, output_tokens)
        Note: Gemini CLI doesn't report token usage, so tokens are estimated.

    Raises:
        RuntimeError: If Gemini CLI is not available or fails
    """
    if not GEMINI_CLI_AVAILABLE:
        raise RuntimeError(
            "Gemini CLI not found. Note: Gemini CLI was retired for consumer "
            "accounts on 2026-06-18 — use antigravity/<model> (agy CLI) or "
            "gemini/<model> (GEMINI_API_KEY) instead."
        )

    global _gemini_retirement_warned
    if not _gemini_retirement_warned:
        _gemini_retirement_warned = True
        print(
            "Warning: Gemini CLI consumer service was retired 2026-06-18 in favor of "
            "Antigravity CLI. If this call fails, switch to antigravity/<model> "
            "(agy CLI) or gemini/<model> (GEMINI_API_KEY).",
            file=sys.stderr,
        )

    # Extract actual model name from "gemini-cli/model" format
    actual_model = model.split("/", 1)[1] if "/" in model else model

    # Combine system prompt and user message
    full_prompt = f"""SYSTEM INSTRUCTIONS:
{system_prompt}

USER REQUEST:
{user_message}"""

    try:
        # Use gemini CLI with the prompt passed via stdin and -p flag
        cmd = [
            GEMINI_CLI_PATH,
            "-m",
            actual_model,
            "-y",
        ]  # -y for auto-approve (no tool calls expected)

        result = subprocess.run(
            cmd, input=full_prompt, capture_output=True, text=True, timeout=timeout
        )

        if result.returncode != 0:
            error_msg = (
                result.stderr.strip()
                or f"Gemini CLI exited with code {result.returncode}"
            )
            raise RuntimeError(f"Gemini CLI failed: {error_msg}")

        response_text = result.stdout.strip()

        # Filter out noise lines from gemini CLI output
        lines = response_text.split("\n")
        filtered_lines = []
        skip_prefixes = ("Loaded cached", "Server ", "Loading extension")
        for line in lines:
            if not any(line.startswith(prefix) for prefix in skip_prefixes):
                filtered_lines.append(line)
        response_text = "\n".join(filtered_lines).strip()

        if not response_text:
            raise RuntimeError("No response from Gemini CLI")

        # Estimate tokens (Gemini CLI doesn't report actual usage)
        # Rough estimate: 4 chars per token
        input_tokens = len(full_prompt) // 4
        output_tokens = len(response_text) // 4

        return response_text, input_tokens, output_tokens

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Gemini CLI timed out after {timeout}s")
    except FileNotFoundError:
        raise RuntimeError("Gemini CLI not found in PATH")


def resolve_antigravity_model(model: str) -> Optional[str]:
    """Extract the agy model slug from an antigravity/<slug> model string.

    `agy --model` accepts slugs exactly as listed by `agy models`
    (e.g. gemini-3.1-pro-high, claude-sonnet-4-6, gpt-oss-120b-medium).
    Returns None for a bare "antigravity" (use agy's default model).
    """
    slug = model.split("/", 1)[1] if "/" in model else ""
    return slug or None


def call_antigravity_model(
    system_prompt: str,
    user_message: str,
    model: str,
    timeout: int = 600,
) -> tuple[str, int, int]:
    """
    Call Antigravity CLI (agy) in headless print mode using Google account auth.

    Sign in once interactively (`agy`) before headless use — print mode reuses
    cached credentials and cannot complete the OAuth flow itself.

    Args:
        system_prompt: System instructions for the model
        user_message: User prompt to send
        model: Model name (e.g., "antigravity/gemini-3.5-flash"; bare
            "antigravity" uses agy's default model)
        timeout: Timeout in seconds (default 10 minutes)

    Returns:
        Tuple of (response_text, input_tokens, output_tokens)
        Token counts come from agy JSON metadata when present, else estimated.

    Raises:
        RuntimeError: If Antigravity CLI is not available or fails
    """
    if not ANTIGRAVITY_AVAILABLE:
        raise RuntimeError(
            "Antigravity CLI not found. Install with: "
            "curl -fsSL https://antigravity.google/cli/install.sh | bash "
            "— then run `agy` once to sign in."
        )

    display_model = resolve_antigravity_model(model)

    full_prompt = f"""SYSTEM INSTRUCTIONS:
{system_prompt}

USER REQUEST:
{user_message}"""

    cmd = [
        ANTIGRAVITY_PATH,
        "-p",
        full_prompt,
        "--output-format",
        "json",
        "--print-timeout",
        f"{timeout}s",
    ]
    if display_model:
        cmd.extend(["--model", display_model])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 30,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Antigravity CLI timed out after {timeout}s")
    except FileNotFoundError:
        raise RuntimeError("Antigravity CLI not found in PATH")

    stdout = result.stdout.strip()

    # agy prints an interactive OAuth prompt when credentials are missing —
    # detect its fixed prompt strings, not URL fragments (which could appear
    # in legitimate model output).
    if (
        "Waiting for authentication" in result.stdout
        or "paste the authorization code" in result.stdout
    ):
        raise RuntimeError(
            "Antigravity CLI is not authenticated. Run `agy` interactively once "
            "to complete Google sign-in, then retry."
        )

    if result.returncode != 0:
        error_msg = (
            result.stderr.strip()
            or stdout
            or f"Antigravity CLI exited with code {result.returncode}"
        )
        raise RuntimeError(f"Antigravity CLI failed: {error_msg}")

    response_text = ""
    input_tokens = 0
    output_tokens = 0

    # JSON output is a single object; schema may evolve, so probe common keys.
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, dict):
        status = payload.get("status", "")
        if status and status != "SUCCESS":
            raise RuntimeError(
                f"Antigravity CLI returned status {status}: "
                f"{payload.get('error') or payload.get('response') or stdout[:200]}"
            )
        for key in ("response", "result", "text", "output", "message"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                response_text = value.strip()
                break
        usage = payload.get("usage") or payload.get("metadata") or {}
        if isinstance(usage, dict):
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)

    if not response_text and payload is None:
        # Fall back to raw stdout only when it wasn't JSON at all
        # (e.g. --output-format ignored by an older agy)
        response_text = stdout

    if not response_text:
        raise RuntimeError(
            "No response text in Antigravity CLI output: " + stdout[:200]
        )

    if not input_tokens:
        input_tokens = len(full_prompt) // 4
    if not output_tokens:
        output_tokens = len(response_text) // 4

    return response_text, input_tokens, output_tokens


def _call_cli_provider_with_retries(
    model: str,
    call_fn,
    review_only: bool,
) -> ModelResponse:
    """Shared retry loop for CLI/SDK providers (codex, gemini-cli, antigravity,
    foundry).

    Retries transient failures with exponential backoff; deterministic errors
    (unknown model, wrong auth mode, bad credentials) fail fast. Codex
    ChatGPT-account model rejections get an actionable hint appended.
    """
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            content, input_tokens, output_tokens = call_fn()
            agreed = "[AGREE]" in content
            extracted = extract_spec(content)

            if should_warn_missing_spec(agreed, extracted, review_only):
                print(
                    f"Warning: {model} provided critique but no [SPEC] tags found. Response may be malformed.",
                    file=sys.stderr,
                )

            cost = cost_tracker.add(model, input_tokens, output_tokens)

            return ModelResponse(
                model=model,
                response=content,
                agreed=agreed,
                spec=extracted,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
            )
        except Exception as e:
            last_error = str(e)
            lower = last_error.lower()
            if (
                "not supported when using codex with a chatgpt account" in lower
            ):
                last_error = f"{last_error}\n  Hint: {CODEX_CHATGPT_HINT}"
            if is_non_retryable_error(last_error):
                print(
                    f"Error: {model} failed (non-retryable): {last_error}",
                    file=sys.stderr,
                )
                break
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                print(
                    f"Warning: {model} failed (attempt {attempt + 1}/{MAX_RETRIES}): {last_error}. Retrying in {delay:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                print(
                    f"Error: {model} failed after {MAX_RETRIES} attempts: {last_error}",
                    file=sys.stderr,
                )

    return ModelResponse(
        model=model, response="", agreed=False, spec=None, error=last_error
    )


def call_single_model(
    model: str,
    spec: str,
    round_num: int,
    doc_type: str,
    press: bool = False,
    focus: Optional[str] = None,
    persona: Optional[str] = None,
    context: Optional[str] = None,
    preserve_intent: bool = False,
    codex_reasoning: str = DEFAULT_CODEX_REASONING,
    codex_search: bool = False,
    timeout: int = 600,
    bedrock_mode: bool = False,
    bedrock_region: Optional[str] = None,
    review_only: bool = False,
    claude_effort: str = DEFAULT_CLAUDE_EFFORT,
) -> ModelResponse:
    """Send spec to a single model and return response with retry on failure.

    When review_only is True, the model acts as a final reviewer/judge: it emits
    [AGREE] or a short critique and never re-emits the spec. This keeps deep
    reasoners (gpt-5.5-pro) and Opus inside the debate as acceptance gates
    without exhausting their output budget re-typing a long document.
    """
    # Handle Bedrock routing
    actual_model = model
    if bedrock_mode:
        if bedrock_region:
            os.environ["AWS_REGION"] = bedrock_region
        if not model.startswith("bedrock/"):
            actual_model = f"bedrock/{model}"

    system_prompt = get_system_prompt(doc_type, persona, review_only, press)
    doc_type_name = get_doc_type_name(doc_type)

    focus_section = ""
    if focus and focus.lower() in FOCUS_AREAS:
        focus_section = FOCUS_AREAS[focus.lower()]
    elif focus:
        focus_section = f"**CRITICAL FOCUS: {focus.upper()}**\nPrioritize analysis of {focus} concerns above all else."

    if preserve_intent:
        focus_section = PRESERVE_INTENT_PROMPT + "\n\n" + focus_section

    context_section = context if context else ""

    if press:
        # In judge mode the plain press template's closing "emit the final spec
        # between [SPEC] tags" contradicts the reviewer suffix's "do NOT use
        # [SPEC] tags". Leaving both in place only moves the conflict.
        template = (
            PRESS_REVIEW_ONLY_PROMPT_TEMPLATE if review_only else PRESS_PROMPT_TEMPLATE
        )
    else:
        template = REVIEW_PROMPT_TEMPLATE
    user_message = template.format(
        round=round_num,
        doc_type_name=doc_type_name,
        spec=spec,
        focus_section=focus_section,
        context_section=context_section,
    )

    # Route Codex CLI models to dedicated handler
    if model.startswith("codex/"):
        return _call_cli_provider_with_retries(
            model,
            lambda: call_codex_model(
                system_prompt=system_prompt,
                user_message=user_message,
                model=model,
                reasoning_effort=codex_reasoning,
                timeout=timeout,
                search=codex_search,
            ),
            review_only,
        )

    # Route Antigravity CLI models to dedicated handler
    if model == "antigravity" or model.startswith("antigravity/"):
        return _call_cli_provider_with_retries(
            model,
            lambda: call_antigravity_model(
                system_prompt=system_prompt,
                user_message=user_message,
                model=model,
                timeout=timeout,
            ),
            review_only,
        )

    # Route Gemini CLI models to dedicated handler (retired 2026-06-18;
    # kept for enterprise-license users, warns and points at antigravity/)
    if model.startswith("gemini-cli/"):
        return _call_cli_provider_with_retries(
            model,
            lambda: call_gemini_cli_model(
                system_prompt=system_prompt,
                user_message=user_message,
                model=model,
                timeout=timeout,
            ),
            review_only,
        )

    # Route Azure AI Foundry models to dedicated handler
    if model.startswith("foundry/"):
        return _call_cli_provider_with_retries(
            model,
            lambda: call_foundry_model(
                system_prompt=system_prompt,
                user_message=user_message,
                model=model,
                timeout=timeout,
            ),
            review_only,
        )

    # Standard litellm path for all other providers
    last_error = None
    display_model = model

    for attempt in range(MAX_RETRIES):
        try:
            # Build completion kwargs
            completion_kwargs = {
                "model": actual_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "timeout": timeout,
            }

            budget = output_token_budget(actual_model)
            if uses_max_completion_tokens(actual_model):
                completion_kwargs["max_completion_tokens"] = budget
            else:
                completion_kwargs["max_tokens"] = budget
            if not is_reasoning_model(actual_model):
                completion_kwargs["temperature"] = 0.7

            # GPT-5 family: cap visible verbosity and (non-pro) reasoning effort
            # so reasoning + output fit the budget. Replaces the old prose hint.
            completion_kwargs.update(gpt5_tuning_params(actual_model))
            # Claude 4.6+: step effort down for in-loop debaters (same reason).
            completion_kwargs.update(
                claude_tuning_params(actual_model, review_only, claude_effort)
            )

            response = completion(**completion_kwargs)
            content = response.choices[0].message.content or ""
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if finish_reason == "length" and not content.strip():
                raise RuntimeError(
                    f"{actual_model} exhausted its {budget}-token output budget on "
                    "reasoning before producing visible output. Budget too low for this model."
                )
            agreed = "[AGREE]" in content
            extracted = extract_spec(content)

            if should_warn_missing_spec(agreed, extracted, review_only):
                print(
                    f"Warning: {display_model} provided critique but no [SPEC] tags found. Response may be malformed.",
                    file=sys.stderr,
                )

            input_tokens = response.usage.prompt_tokens if response.usage else 0
            output_tokens = response.usage.completion_tokens if response.usage else 0

            cost = cost_tracker.add(display_model, input_tokens, output_tokens)

            return ModelResponse(
                model=display_model,
                response=content,
                agreed=agreed,
                spec=extracted,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
            )
        except Exception as e:
            last_error = str(e)
            if bedrock_mode:
                if "AccessDeniedException" in last_error:
                    last_error = (
                        f"Model not enabled in your Bedrock account: {display_model}"
                    )
                elif "ValidationException" in last_error:
                    last_error = f"Invalid Bedrock model ID: {display_model}"

            if is_non_retryable_error(last_error):
                print(
                    f"Error: {display_model} failed (non-retryable): {last_error}",
                    file=sys.stderr,
                )
                break

            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                print(
                    f"Warning: {display_model} failed (attempt {attempt + 1}/{MAX_RETRIES}): {last_error}. Retrying in {delay:.1f}s...",
                    file=sys.stderr,
                )
                time.sleep(delay)
            else:
                print(
                    f"Error: {display_model} failed after {MAX_RETRIES} attempts: {last_error}",
                    file=sys.stderr,
                )

    return ModelResponse(
        model=display_model, response="", agreed=False, spec=None, error=last_error
    )


def call_models_parallel(
    models: list[str],
    spec: str,
    round_num: int,
    doc_type: str,
    press: bool = False,
    focus: Optional[str] = None,
    persona: Optional[str] = None,
    context: Optional[str] = None,
    preserve_intent: bool = False,
    codex_reasoning: str = DEFAULT_CODEX_REASONING,
    codex_search: bool = False,
    timeout: int = 600,
    bedrock_mode: bool = False,
    bedrock_region: Optional[str] = None,
    review_only: bool = False,
    claude_effort: str = DEFAULT_CLAUDE_EFFORT,
) -> list[ModelResponse]:
    """Call multiple models in parallel and collect responses."""
    if not models:
        return []
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
        future_to_model = {
            executor.submit(
                call_single_model,
                model,
                spec,
                round_num,
                doc_type,
                press,
                focus,
                persona,
                context,
                preserve_intent,
                codex_reasoning,
                codex_search,
                timeout,
                bedrock_mode,
                bedrock_region,
                review_only,
                claude_effort,
            ): model
            for model in models
        }
        for future in concurrent.futures.as_completed(future_to_model):
            results.append(future.result())

    if preserve_intent:
        for r in results:
            if r.spec:
                warn_dropped_headings(r.model, spec, r.spec)

    return results
