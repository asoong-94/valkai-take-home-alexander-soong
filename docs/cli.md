# CLI guide

Run the agent as an interactive terminal chat with pluggable memory strategies. See the [core README](../README.md) for initial setup.

## Start

```bash
# Default: baseline strategy, no cross-session memory
uv run chat

# Semantic memory (vector-based, persists across sessions)
uv run chat --memory semantic --user-id alex

# Structured memory (fixed-schema JSON profile)
uv run chat --memory structured --user-id alex

# Use a different LLM provider
uv run chat --memory semantic --model openai:gpt-4o

# Custom data directory for persistent storage
uv run chat --memory structured --data-dir ./my-data
```

Type `quit` or `exit` to end the session.

## CLI arguments

| Flag | Default | Description |
|---|---|---|
| `--memory` | `baseline` | Memory strategy: `baseline`, `semantic`, or `structured` |
| `--user-id` | `default-user` | User ID that scopes cross-session memory |
| `--model` | `anthropic:claude-haiku-4-5-20251001` | LLM provider:model string |
| `--data-dir` | `./data` | Directory for persistent memory storage |

## How it works

`src/agent/cli.py` creates a `MemoryStrategy` instance via the registry and generates a random `thread_id` per session. Each user message is routed through `strategy.chat()`, which handles both in-session conversation history and cross-session memory depending on the strategy.

On startup the CLI prints the active strategy, model, user ID, and thread ID:

```
Strategy: semantic | Model: anthropic:claude-haiku-4-5-20251001
User: alex | Thread: a1b2c3d4e5f6
Type 'quit' to exit.
```

## Relevant files

```
src/agent/
├── cli.py              # REPL loop, argument parsing
├── core.py             # agent factory (used by baseline)
├── schemas.py          # Pydantic models for extraction
└── strategies/
    ├── base.py         # MemoryStrategy ABC
    ├── baseline.py     # In-process only
    ├── semantic.py     # ChromaDB + LangGraph pipeline
    ├── structured.py   # JSON profile + LangGraph pipeline
    └── __init__.py     # REGISTRY + make_strategy()
```

## Running the evaluation harness

```bash
uv run harness
```

Runs 5 scripted scenarios (identity recall, preference overwrite, exhaustive recall, schema-free facts, mutable fact update) against all three strategies and prints a comparison matrix. Each run is isolated with its own data directory that is cleaned up afterward. Takes ~90 seconds.
