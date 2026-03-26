# spec-debate

A Claude Code plugin that iteratively refines product specifications through multi-model adversarial debate until all models reach consensus.

> **Attribution**: Originally based on [adversarial-spec](https://github.com/zscole/adversarial-spec) by [Zak Cole](https://github.com/zscole). This fork adds current-gen model support, Azure AI Foundry integration, dynamic cost tracking, and is actively maintained.

**Key insight:** A single LLM reviewing a spec will miss things. Multiple LLMs debating a spec will catch gaps, challenge assumptions, and surface edge cases that any one model would overlook. The result is a document that has survived rigorous adversarial review.

**Claude is an active participant**, not just an orchestrator. Claude provides independent critiques, challenges opponent models, and contributes substantive improvements alongside external models.

## Quick Start

```bash
# 1. Install the plugin
claude plugin marketplace add machug/marketplace
claude plugin install spec-debate@machug

# 2. Install dependencies
pip install litellm azure-ai-inference

# 3. Set at least one API key
export OPENAI_API_KEY="sk-..."
# Or use OpenRouter for access to multiple providers with one key
export OPENROUTER_API_KEY="sk-or-..."

# 4. Test your providers
/spec-debate providers

# 5. Run it
/spec-debate "Build a rate limiter service with Redis backend"
```

Also available via [skills.sh](https://skills.sh) for cross-platform agent support:
```bash
npx skills add machug/spec-debate
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
2. (Optional) Interview mode for in-depth requirements gathering
3. Claude drafts the initial document (PRD or tech spec)
4. Document sent to opponent models for parallel critique
5. Claude provides independent critique alongside opponent feedback
6. Claude synthesizes all feedback and revises the document
7. Loop continues until ALL models AND Claude agree (max 10 rounds)
8. User review: accept, request changes, or run additional debate cycles
9. Final converged document output

## Requirements

- Python 3.10+
- `pip install litellm` (required)
- `pip install azure-ai-inference` (optional, for Azure AI Foundry)
- API key for at least one LLM provider

## Supported Providers

| Provider | Env Var | Example Models |
|----------|---------|----------------|
| OpenAI | `OPENAI_API_KEY` | `gpt-5.4`, `gpt-5.4-pro`, `o3`, `o4-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5` |
| Google | `GEMINI_API_KEY` | `gemini/gemini-2.5-pro`, `gemini/gemini-2.5-flash` |
| xAI | `XAI_API_KEY` | `xai/grok-4-1-fast`, `xai/grok-4-0709` |
| Azure AI Foundry | `AZURE_AI_API_KEY` + `AZURE_AI_API_BASE` | `foundry/<your-deployment>` |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/openai/gpt-5.2-pro`, `openrouter/anthropic/claude-opus-4.6` |
| Mistral | `MISTRAL_API_KEY` | `mistral/mistral-large`, `mistral/codestral` |
| Groq | `GROQ_API_KEY` | `groq/llama-3.3-70b-versatile` |
| Deepseek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-chat` |
| ZAI (GLM) | `ZAI_API_KEY` | `zai/glm-5`, `zai/glm-4.7` |
| Moonshot (Kimi) | `MOONSHOT_API_KEY` | `moonshot/kimi-k2.5`, `moonshot/kimi-k2-thinking` |
| Codex CLI | ChatGPT subscription | `codex/gpt-5.3-codex` |
| Gemini CLI | Google account | `gemini-cli/gemini-3.1-pro-preview` |

**Model costs are dynamic** via LiteLLM's community-maintained registry -- no hardcoded pricing to go stale.

Check configured providers and test connectivity:

```bash
cd skills/spec-debate/scripts
python3 debate.py providers
python3 debate.py test
python3 debate.py test --models gpt-5.4,foundry/gpt-5-mini
```

## Document Types

- **PRD** (Product Requirements Document) -- business/product focus with personas, user stories, KPIs, scope, and risks
- **Tech Spec** (Technical Specification) -- engineering focus with architecture, APIs, data models, security, and deployment

## Features

### Core
- **Multi-LLM adversarial debate** -- 2+ models critique in parallel until consensus
- **Claude's active participation** -- independent critiques, not just orchestration
- **Anti-laziness checks** -- presses models that agree too quickly
- **Interview mode** -- in-depth requirements gathering before drafting
- **User review period** -- accept, request changes, or run additional cycles
- **PRD to tech spec flow** -- continue from PRD directly into a technical specification

### Advanced
- **Focus modes** -- direct critique toward specific concerns: `security`, `scalability`, `performance`, `ux`, `reliability`, `cost`
- **Model personas** -- shift perspective: `security-engineer`, `oncall-engineer`, `junior-developer`, `qa-engineer`, `site-reliability`, `product-manager`, `data-engineer`, `mobile-developer`, `accessibility-specialist`, `legal-compliance`, or custom
- **Context injection** -- include existing docs: `--context ./api.md --context ./schema.sql`
- **Session persistence** -- resume interrupted debates: `--session my-spec` / `--resume my-spec`
- **Auto-checkpointing** -- per-round spec snapshots for rollback
- **Preserve intent mode** -- require justification for any section removals
- **Cost tracking** -- per-round token usage and estimated cost via LiteLLM registry
- **Saved profiles** -- reuse model/focus/persona configs across debates
- **Diff between rounds** -- unified diff showing what changed
- **Export to task list** -- convert specs to structured tasks (JSON)
- **Telegram integration** -- real-time notifications and human-in-the-loop feedback

### Enterprise
- **AWS Bedrock routing** -- route all calls through Bedrock with friendly model names
- **Azure AI Foundry v2** -- native SDK integration for Foundry-deployed models
- **OpenRouter** -- unified access to 100+ models with a single key

## CLI Reference

```bash
# Core commands
python3 debate.py critique --models MODEL_LIST --doc-type TYPE [OPTIONS] < spec.md
python3 debate.py critique --resume SESSION_ID

# Test and diagnostics
python3 debate.py test                        # Ping all configured providers
python3 debate.py test --models gpt-5.4       # Test specific models
python3 debate.py providers                   # Show provider status

# Utilities
python3 debate.py diff --previous OLD.md --current NEW.md
python3 debate.py export-tasks --models MODEL --doc-type TYPE [--json] < spec.md
python3 debate.py focus-areas
python3 debate.py personas
python3 debate.py profiles
python3 debate.py sessions
python3 debate.py save-profile NAME --models ... [--focus ...] [--persona ...]

# Bedrock management
python3 debate.py bedrock enable --region us-east-1
python3 debate.py bedrock add-model claude-sonnet-4.6
python3 debate.py bedrock status
python3 debate.py bedrock list-models
```

## File Structure

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
├── README.md
├── LICENSE
└── requirements.txt
```

## License

MIT -- See [LICENSE](LICENSE) for details.

Original work Copyright (c) Zak Cole.
Modifications Copyright (c) Hugh McIntyre.
