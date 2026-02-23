# CLI guide

Run the agent as an interactive terminal chat. See the [core README](../README.md) for initial setup.

## Start

```bash
# Default model (Anthropic Claude Haiku)
uv run chat

# OpenAI
uv run chat --model openai:gpt-4o

# Google
uv run chat --model google_genai:gemini-2.5-flash

# Custom system prompt
uv run chat --system "You are a helpful coding assistant."
```

Type `quit` or `exit` to end the session.

## How it works

`src/agent/cli.py` keeps a running `messages` list in memory for the duration of the session, appending each user/assistant turn before passing the full history to `agent.invoke()`.

## Relevant files

```
src/agent/
├── core.py     # agent factory (shared)
└── cli.py      # REPL loop, argument parsing
```
