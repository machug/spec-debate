# spec-debate

A Claude Code plugin that iteratively refines product specifications through multi-model debate until consensus is reached.

> **Attribution**: Originally based on [adversarial-spec](https://github.com/zscole/adversarial-spec) by [Zak Cole](https://github.com/zscole). This fork adds GPT-5/O3/O4 compatibility fixes, updated model defaults and pricing for 2026, and is actively maintained.

**Key insight:** A single LLM reviewing a spec will miss things. Multiple LLMs debating a spec will catch gaps, challenge assumptions, and surface edge cases that any one model would overlook. The result is a document that has survived rigorous adversarial review.

**Claude is an active participant**, not just an orchestrator. Claude provides independent critiques, challenges opponent models, and contributes substantive improvements alongside external models.

## What Changed from the Original

This fork addresses compatibility issues and stale defaults in the upstream project:

- **`is_o_series_model()` -> `is_fixed_temperature_model()`** - Expanded to handle O3, O4, and GPT-5 family models (none support custom temperature)
- **GPT-5 `max_tokens` fix** - GPT-5 models reject `max_tokens`/`max_output_tokens`; conditionally omitted
- **Null safety fix** - `response.choices[0].message.content or ""` prevents NoneType errors
- **Updated all model defaults** to current generation (GPT-5.4, Claude 4.6, Gemini 2.5, Grok 4, Codex 5.3)
- **Updated cost table** with 2026 pricing across all providers
- **Added O3/O4 to credential validation** provider map
- **Test suite updated** with coverage for all new models and edge cases

## Quick Start

```bash
# 1. Add the marketplace and install the plugin
claude plugin marketplace add machug/marketplace
claude plugin install spec-debate@machug

# 2. Set at least one API key
export OPENAI_API_KEY="sk-..."
# Or use OpenRouter for access to multiple providers with one key
export OPENROUTER_API_KEY="sk-or-..."

# 3. Run it
/spec-debate "Build a rate limiter service with Redis backend"
```

## How It Works

```
You describe product --> Claude drafts spec --> Multiple LLMs critique in parallel
        |                                              |
        |                                              v
        |                              Claude synthesizes + adds own critique
        |                                              |
        |                                              v
        |                              Revise and repeat until ALL agree
        |                                              |
        +--------------------------------------------->|
                                                       v
                                            User review period
                                                       |
                                                       v
                                            Final document output
```

1. Describe your product concept or provide an existing document
2. (Optional) Start with an in-depth interview to capture requirements
3. Claude drafts the initial document (PRD or tech spec)
4. Document is sent to opponent models (GPT, Gemini, Grok, etc.) for parallel critique
5. Claude provides independent critique alongside opponent feedback
6. Claude synthesizes all feedback and revises
7. Loop continues until ALL models AND Claude agree
8. User review period: request changes or run additional cycles
9. Final converged document is output

## Requirements

- Python 3.10+
- `litellm` package: `pip install litellm`
- API key for at least one LLM provider

## Supported Models

| Provider   | Env Var                | Example Models                               |
|------------|------------------------|----------------------------------------------|
| OpenAI     | `OPENAI_API_KEY`       | `gpt-5.4`, `gpt-5.4-pro`, `gpt-5-mini`      |
| Anthropic  | `ANTHROPIC_API_KEY`    | `claude-sonnet-4-6-20250627`, `claude-opus-4-6-20250627` |
| Google     | `GEMINI_API_KEY`       | `gemini/gemini-2.5-pro`, `gemini/gemini-2.5-flash` |
| xAI        | `XAI_API_KEY`          | `xai/grok-4-0709`, `xai/grok-4-fast-reasoning` |
| Mistral    | `MISTRAL_API_KEY`      | `mistral/mistral-large`, `mistral/codestral` |
| Groq       | `GROQ_API_KEY`         | `groq/llama-3.3-70b-versatile`               |
| OpenRouter | `OPENROUTER_API_KEY`   | `openrouter/openai/gpt-5.4`, `openrouter/anthropic/claude-sonnet-4-6-20250627` |
| Codex CLI  | ChatGPT subscription   | `codex/gpt-5.3-codex`, `codex/gpt-5.2-codex` |
| Gemini CLI | Google account         | `gemini-cli/gemini-3.1-pro-preview`, `gemini-cli/gemini-3-flash-preview` |
| Deepseek   | `DEEPSEEK_API_KEY`     | `deepseek/deepseek-chat`                     |
| Zhipu      | `ZHIPUAI_API_KEY`      | `zhipu/glm-4`, `zhipu/glm-4-plus`            |

Check which keys are configured:

```bash
python3 "$(find ~/.claude -name debate.py -path '*spec-debate*' 2>/dev/null | head -1)" providers
```

## AWS Bedrock Support

For enterprise users who need to route all model calls through AWS Bedrock:

```bash
python3 "$(find ~/.claude -name debate.py -path '*spec-debate*' 2>/dev/null | head -1)" bedrock enable --region us-east-1
python3 "$(find ~/.claude -name debate.py -path '*spec-debate*' 2>/dev/null | head -1)" bedrock add-model claude-3-sonnet
python3 "$(find ~/.claude -name debate.py -path '*spec-debate*' 2>/dev/null | head -1)" bedrock status
```

When Bedrock is enabled, **all model calls route through Bedrock** - no direct API calls are made. Configuration is stored at `~/.claude/spec-debate/config.json`.

## OpenRouter Support

[OpenRouter](https://openrouter.ai) provides unified access to multiple LLM providers through a single API:

```bash
export OPENROUTER_API_KEY="sk-or-..."
python3 debate.py critique --models openrouter/openai/gpt-5.4,openrouter/anthropic/claude-sonnet-4-6-20250627 < spec.md
```

## Codex CLI Support

[Codex CLI](https://github.com/openai/codex) allows ChatGPT Pro subscribers to use OpenAI models without separate API credits:

```bash
npm install -g @openai/codex
python3 debate.py critique --models codex/gpt-5.3-codex,gemini/gemini-2.5-flash < spec.md
```

## Gemini CLI Support

[Gemini CLI](https://github.com/google-gemini/gemini-cli) allows Google account holders to use Gemini models without API credits:

```bash
npm install -g @google/gemini-cli && gemini auth
python3 debate.py critique --models gemini-cli/gemini-3.1-pro-preview < spec.md
```

## Usage

**Start from scratch:**

```
/spec-debate "Build a rate limiter service with Redis backend"
```

**Refine an existing document:**

```
/spec-debate ./docs/my-spec.md
```

## Document Types

- **PRD** (Product Requirements Document) - Business/product focus for stakeholders and PMs
- **Technical Specification** - Engineering focus for developers and architects

## Core Features

- **Interview Mode** - In-depth requirements gathering before the debate
- **Claude's Active Participation** - Independent critiques alongside external models
- **Early Agreement Verification** - Anti-laziness checks for models that agree too quickly
- **User Review Period** - Accept, request changes, or run additional cycles
- **PRD to Tech Spec Flow** - Continue from PRD directly into a technical specification

## Advanced Features

- **Focus Modes**: `--focus security|scalability|performance|ux|reliability|cost`
- **Model Personas**: `--persona security-engineer|oncall-engineer|junior-developer|...`
- **Context Injection**: `--context ./existing-api.md --context ./schema.sql`
- **Session Persistence**: `--session my-spec` / `--resume my-spec`
- **Preserve Intent Mode**: `--preserve-intent` (require justification for removals)
- **Cost Tracking**: Per-round token usage and estimated cost
- **Saved Profiles**: `save-profile` / `--profile`
- **Diff Between Rounds**: `diff --previous old.md --current new.md`
- **Export to Task List**: `export-tasks --doc-type prd [--json]`
- **Telegram Integration**: Real-time notifications and human-in-the-loop feedback

## CLI Reference

```bash
debate.py critique --models MODEL_LIST --doc-type TYPE [OPTIONS] < spec.md
debate.py critique --resume SESSION_ID
debate.py diff --previous OLD.md --current NEW.md
debate.py export-tasks --models MODEL --doc-type TYPE [--json] < spec.md
debate.py providers | focus-areas | personas | profiles | sessions
debate.py save-profile NAME --models ... [--focus ...] [--persona ...]
```

## File Structure

```
spec-debate/
├── README.md
├── LICENSE
├── requirements.txt
└── skills/
    └── spec-debate/
        ├── SKILL.md
        └── scripts/
            ├── debate.py
            ├── models.py
            ├── prompts.py
            ├── providers.py
            ├── session.py
            └── telegram_bot.py
```

## License

MIT - See [LICENSE](LICENSE) for details.

Original work Copyright (c) Zak Cole.
Modifications Copyright (c) Hugh McIntyre.
