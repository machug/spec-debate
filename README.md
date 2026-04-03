# spec-debate

```
                 ▄▄                          ▄▄
       ▄▄▄▄▄▄▄▄██▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄██▄▄▄▄▄▄▄▄
      ▐░░░▒▒▒▓▓                                ▓▓▒▒▒░░░▌
      ▐░    ████ ████ ████ ████                      ░▌
      ▐░    █    █  █ █    █                         ░▌
      ▐░    ████ ████ ███  █                         ░▌
      ▐░       █ █    █    █                         ░▌
      ▐░    ████ █    ████ ████                      ░▌
      ▐░                                             ░▌
      ▐░    ████ ████ ████ ████ █████ ████           ░▌
      ▐░    █  █ █    █  █ █  █   █   █              ░▌
      ▐░    █  █ ███  ████ ████   █   ███            ░▌
      ▐░    █  █ █    █  █ █  █   █   █              ░▌
      ▐░    ████ ████ ████ █  █   █   ████           ░▌
      ▐░░░▒▒▒▓▓                                ▓▓▒▒▒░░░▌
       ▀▀▀▀▀▀▀▀██▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀██▀▀▀▀▀▀▀▀
                 ▀▀                          ▀▀
```

Refine product specs through multi-model adversarial debate until every model agrees the document is complete. Supports PRDs and technical specifications.

Claude doesn't just orchestrate — it critiques alongside the opponent models, challenges their reasoning, and contributes its own improvements. The debate continues until genuine consensus is reached, not just polite agreement.

> **Attribution**: Originally forked from [adversarial-spec](https://github.com/zscole/adversarial-spec) by [Zak Cole](https://github.com/zscole). This project has since been substantially rewritten with current-gen model support, Azure AI Foundry integration, focus modes, personas, session persistence, Telegram integration, dynamic cost tracking, and more.

## Install

**Claude Code marketplace** (auto-updates):

```bash
claude plugin marketplace add machug/marketplace
claude plugin install spec-debate@machug
```

**skills.sh** (works across Claude Code, Cursor, Copilot, Gemini CLI):

```bash
npx skills add machug/spec-debate
```

**Dependencies:**

```bash
pip install litellm                  # Required
pip install azure-ai-inference       # Optional, for Azure AI Foundry
```

## Quick Start

```bash
# 1. Set at least one API key (or use CLI tools — see Providers below)
export OPENAI_API_KEY="sk-..."

# 2. Check what's available
/spec-debate providers

# 3. Run it
/spec-debate "Design a webhook delivery system with retry and dead-letter queues"
```

Claude will ask you to pick a document type (PRD or tech spec), select opponent models, and optionally run an interview session before drafting. Then the debate begins.

## How It Works

1. You describe what you want to build (or provide an existing spec)
2. Optionally, Claude interviews you to gather detailed requirements
3. Claude drafts the initial document
4. The document is sent to opponent models for parallel critique
5. Claude provides its own independent critique alongside
6. Claude synthesizes all feedback and revises the document
7. Repeat until ALL models AND Claude agree — max 10 rounds per cycle
8. You review: accept, request changes, or run additional debate cycles
9. For PRDs, optionally continue into a technical specification

## Providers

The skill auto-detects available providers at runtime. Run `/spec-debate providers` to see what's configured.

| Provider | Env Var | Example Models |
|----------|---------|----------------|
| OpenAI | `OPENAI_API_KEY` | `gpt-5.4`, `gpt-5.4-pro`, `o3`, `o4-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-opus-4-6`, `claude-sonnet-4-6` |
| Google | `GEMINI_API_KEY` | `gemini/gemini-3.1-pro-preview`, `gemini/gemini-3.1-pro-preview` |
| xAI | `XAI_API_KEY` | `xai/grok-4.20-0309-reasoning`, `xai/grok-4.20-0309-non-reasoning` |
| Azure AI Foundry | `AZURE_AI_API_KEY` + `AZURE_AI_API_BASE` | `foundry/claude-opus-4-6`, `foundry/grok-4.20`, `foundry/Phi-4-reasoning` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/openai/gpt-5.2-pro` |
| Mistral | `MISTRAL_API_KEY` | `mistral/mistral-large`, `mistral/codestral` |
| Groq | `GROQ_API_KEY` | `groq/llama-3.3-70b-versatile` |
| Deepseek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-chat` |
| ZAI (GLM) | `ZAI_API_KEY` | `zai/glm-5`, `zai/glm-4.7` |
| Moonshot (Kimi) | `MOONSHOT_API_KEY` | `moonshot/kimi-k2.5`, `moonshot/kimi-k2-thinking` |
| Codex CLI | ChatGPT subscription | `codex/gpt-5.3-codex` |
| Gemini CLI | Google account | `gemini-cli/gemini-3.1-pro-preview` |

**No API key?** Install [Codex CLI](https://github.com/openai/codex) (`npm install -g @openai/codex`) or [Gemini CLI](https://github.com/google-gemini/gemini-cli) (`npm install -g @google/gemini-cli`) to use your existing ChatGPT or Google subscription.

**Model costs** are discovered dynamically from [LiteLLM's model registry](https://github.com/BerriAI/litellm) — no hardcoded pricing to go stale.

### Azure AI Foundry

Set `AZURE_AI_API_KEY` and `AZURE_AI_API_BASE` (your Foundry project endpoint). Models use the `foundry/` prefix:

```bash
export AZURE_AI_API_KEY="your-key"
export AZURE_AI_API_BASE="https://your-resource.services.ai.azure.com/api/projects/your-project"
```

### AWS Bedrock

Route all model calls through Bedrock for enterprise security/compliance:

```bash
python3 debate.py bedrock enable --region us-east-1
python3 debate.py bedrock add-model claude-sonnet-4.6
python3 debate.py bedrock status
```

Friendly model names (e.g., `claude-sonnet-4.6`) are mapped to Bedrock model IDs automatically. Supports Claude, Llama, Mistral, and Cohere families. Config stored at `~/.claude/spec-debate/config.json`.

## Document Types

### PRD (Product Requirements Document)

Business and product-focused. Covers executive summary, problem statement, personas, user stories, functional and non-functional requirements, success metrics, scope boundaries, dependencies, and risks.

### Technical Specification

Engineering-focused. Covers architecture, component design, API contracts with full schemas, data models, infrastructure, security, error handling, performance targets, observability, testing strategy, deployment, and migration.

## Features

### Focus Modes

Direct critique toward specific concerns:

```bash
/spec-debate "Auth service redesign" --focus security
```

Available: `security`, `scalability`, `performance`, `ux`, `reliability`, `cost`

### Model Personas

Shift the critique perspective:

```bash
/spec-debate "Payment processing API" --persona oncall-engineer
```

Available: `security-engineer`, `oncall-engineer`, `junior-developer`, `qa-engineer`, `site-reliability`, `product-manager`, `data-engineer`, `mobile-developer`, `accessibility-specialist`, `legal-compliance`, or any custom persona.

### Interview Mode

Before drafting, Claude runs a thorough requirements gathering session — problem context, user types, constraints, tradeoffs, success criteria. Opt-in at the start of each debate.

### Context Injection

Include existing documents as reference:

```bash
python3 debate.py critique --models gpt-5.4 --context ./existing-api.md --context ./schema.sql --doc-type tech < spec.md
```

### Session Persistence

Long debates survive interruptions:

```bash
# Start a named session
python3 debate.py critique --models gpt-5.4 --session auth-redesign --doc-type tech < spec.md

# Resume where you left off
python3 debate.py critique --resume auth-redesign

# List all sessions
python3 debate.py sessions
```

Each round is auto-checkpointed to `.spec-debate-checkpoints/` for rollback.

### Preserve Intent Mode

Prevents convergence toward lowest-common-denominator specs. Models must quote what they want to remove, justify the harm, and distinguish errors from preferences:

```bash
python3 debate.py critique --models gpt-5.4 --preserve-intent --doc-type tech < spec.md
```

### Anti-Laziness Checks

If a model agrees too quickly (rounds 1-2), Claude presses it to confirm it actually read the full document, list specific sections reviewed, and explain why it agrees.

### PRD to Tech Spec Flow

After a PRD reaches consensus, optionally continue directly into a technical specification that implements it — same session, same opponent models.

### Cost Tracking

Token usage and estimated cost (USD) reported per round and per model, sourced from LiteLLM's dynamic pricing registry:

```
=== Cost Summary ===
Total tokens: 12,543 in / 3,221 out
Total cost: $0.0847

By model:
  gpt-5.4: $0.0523 (8,234 in / 2,100 out)
  gemini/gemini-3.1-pro-preview: $0.0324 (4,309 in / 1,121 out)
```

### Saved Profiles

Save frequently used configurations:

```bash
python3 debate.py save-profile strict-security --models gpt-5.4,gemini/gemini-3.1-pro-preview --focus security --doc-type tech
python3 debate.py critique --profile strict-security < spec.md
python3 debate.py profiles
```

### Diff Between Rounds

```bash
python3 debate.py diff --previous round1.md --current round2.md
```

### Export to Task List

Convert a finalized spec into structured tasks for your issue tracker:

```bash
python3 debate.py export-tasks --models gpt-5.4 --doc-type prd < spec-output.md
python3 debate.py export-tasks --models gpt-5.4 --doc-type prd --json < spec-output.md
```

### Telegram Integration

Real-time notifications and human-in-the-loop feedback during debates:

```bash
# Setup
python3 telegram_bot.py setup

# Run with Telegram enabled
python3 debate.py critique --models gpt-5.4 --doc-type tech --telegram < spec.md
```

After each round, the bot sends a summary to Telegram and waits 60 seconds for your feedback before continuing. Configurable via `--poll-timeout`.

## CLI Reference

All commands assume you're in the scripts directory (`skills/spec-debate/scripts/`):

```bash
# Core
python3 debate.py critique --models MODEL_LIST --doc-type TYPE [OPTIONS] < spec.md
python3 debate.py critique --resume SESSION_ID

# Diagnostics
python3 debate.py providers
python3 debate.py test
python3 debate.py test --models gpt-5.4,foundry/gpt-5-mini

# Utilities
python3 debate.py diff --previous OLD.md --current NEW.md
python3 debate.py export-tasks --models MODEL --doc-type TYPE [--json] < spec.md
python3 debate.py focus-areas
python3 debate.py personas
python3 debate.py profiles
python3 debate.py sessions
python3 debate.py save-profile NAME --models ... [--focus ...] [--persona ...]

# Bedrock
python3 debate.py bedrock enable --region us-east-1
python3 debate.py bedrock add-model claude-sonnet-4.6
python3 debate.py bedrock status
python3 debate.py bedrock list-models

# Telegram
python3 debate.py send-final --models MODEL_LIST --doc-type TYPE --rounds N < spec.md
```

### Critique Options

| Flag | Description |
|------|-------------|
| `--models, -m` | Comma-separated model list |
| `--doc-type, -d` | `prd` or `tech` (default: tech) |
| `--focus, -f` | Focus area for critique |
| `--persona` | Professional persona for critique |
| `--context, -c` | Context file (repeatable) |
| `--preserve-intent` | Require justification for removals |
| `--session, -s` | Session ID for persistence |
| `--resume` | Resume a previous session |
| `--profile` | Load saved profile |
| `--press, -p` | Anti-laziness check |
| `--telegram, -t` | Enable Telegram notifications |
| `--poll-timeout` | Telegram reply timeout in seconds (default: 60) |
| `--codex-reasoning` | Reasoning effort for Codex CLI (low/medium/high/xhigh) |
| `--codex-search` | Enable web search for Codex CLI models |
| `--timeout` | API/CLI call timeout in seconds (default: 600) |
| `--json, -j` | Output as JSON |

## Project Structure

```
spec-debate/
├── .claude-plugin/
│   └── plugin.json            # Plugin metadata
├── skills/
│   └── spec-debate/
│       ├── SKILL.md            # Claude skill definition
│       └── scripts/
│           ├── debate.py       # Main orchestrator + CLI
│           ├── models.py       # LLM calls, cost tracking, parallel execution
│           ├── prompts.py      # System prompts + critique templates
│           ├── providers.py    # Provider config, Bedrock, profiles
│           ├── session.py      # State persistence + checkpointing
│           └── telegram_bot.py # Telegram notifications (optional)
├── tests/
├── README.md
├── requirements.txt
└── LICENSE
```

## Author

[Hugh McIntyre](https://hughtec.com/) ([X](https://x.com/mmhughmm))

## License

MIT — See [LICENSE](LICENSE) for details.

Original work Copyright (c) Zak Cole.
Modifications Copyright (c) Hugh McIntyre.
