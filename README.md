# Memory Strategy Agent

A conversational LLM agent that implements and compares four cross-session memory strategies: **baseline** (no memory), **semantic** (vector-based), **structured** (fixed-schema profiles), and **hybrid** (both stores with source-of-truth prompting). Built on LangChain, LangGraph, ChromaDB, and sentence-transformers.

- [CLI guide](docs/cli.md) — interactive terminal chat with memory
- [Fullstack guide](docs/fullstack.md) — FastAPI server + React frontend
- [Approach & findings](docs/approach.md) — design decisions, trade-offs, harness results

---

## Quick start

```bash
git clone <repo-url>
cd valkai-take-home-alexander-soong
uv sync
cp .env.example .env
# Fill in your API key(s) in .env
```

### Run the CLI

```bash
# Baseline (no cross-session memory — control group)
uv run chat --memory baseline

# Semantic memory (vector similarity via ChromaDB)
uv run chat --memory semantic --user-id alex

# Structured memory (fixed-schema JSON profile)
uv run chat --memory structured --user-id alex

# Hybrid memory (profile as source of truth + vector store for context)
uv run chat --memory hybrid --user-id alex
```

### Run the evaluation harness

```bash
uv run harness
```

Runs 6 scripted scenarios against all 4 strategies and prints a side-by-side comparison matrix. Takes ~4 min (makes real LLM calls). See [Approach & findings](docs/approach.md) for detailed results.

### Run the fullstack app

**Terminal 1 — backend:**

```bash
uv run serve
```

**Terminal 2 — frontend:**

```bash
cd frontend && npm install && npm run dev
```

Open `http://localhost:3000`. Use the strategy picker and user dropdown to switch between memory types and see the stored memory panel update live.

---

## Memory strategies

| Strategy | Storage | Cross-session? | Write policy | Read policy |
|---|---|---|---|---|
| **Baseline** | In-process dict | No | Append to thread | Full thread history |
| **Semantic** | ChromaDB vectors | Yes | Extract facts → embed → append | Top-k similarity search |
| **Structured** | JSON files on disk | Yes | Extract profile → merge → overwrite | Load full profile |
| **Hybrid** | JSON profile + ChromaDB | Yes | Profile overwrites + facts append | Profile (source of truth) + top-k facts |

### Suggested conversations to try

**Baseline** — start here to see the control group:

```
You: My name is Alex and I work at Stripe.
You: What's my name?           # ✓ works (same session)
# restart the CLI
You: What's my name?           # ✗ no recall (new session)
```

**Semantic** — good at open-ended facts, struggles with updates:

```
uv run chat --memory semantic --user-id demo
You: My favorite food is ramen and I love skiing.
You: I drink cortados every morning.
# quit and restart
You: What do you know about me?     # ✓ recalls ramen, skiing, cortados
You: I moved from SF to New York.
# quit and restart
You: Where do I live?               # ⚠ may mention both SF and NY
```

**Structured** — clean overwrites, but drops facts outside the schema:

```
uv run chat --memory structured --user-id demo
You: I'm a backend engineer at Stripe in San Francisco.
# quit and restart
You: What do you know about me?     # ✓ name, role, company, location
You: My favorite movie is Interstellar.
# quit and restart
You: What's my favorite movie?      # ✗ not in the profile schema
```

**Hybrid** — best of both worlds, profile is source of truth + vector store for everything else:

```
uv run chat --memory hybrid --user-id demo
You: I grew up in Chicago and went to college in Boston.
You: I moved to San Francisco after graduating.
You: I spent two years in Seattle, then a year in Austin.
You: My favorite food is ramen.
# quit and restart
uv run chat --memory hybrid --user-id demo
You: I just moved to New York.
# quit and restart
uv run chat --memory hybrid --user-id demo
You: Where do I currently live?     # ✓ New York (from profile, no hedging)
You: Where have I lived before?     # ✓ Chicago, Boston, SF, Seattle, Austin (from vector store)
You: What's my favorite food?       # ✓ ramen (from vector store, not in profile schema)
```

---

## Project structure

```
├── README.md
├── docs/
│   ├── cli.md              # CLI usage guide
│   ├── fullstack.md        # server + frontend guide
│   └── approach.md         # design write-up & harness results
├── pyproject.toml
├── .env.example
├── src/agent/
│   ├── core.py             # LangChain agent factory
│   ├── cli.py              # CLI entry point (uv run chat)
│   ├── server.py           # FastAPI server (uv run serve)
│   ├── schemas.py          # Pydantic models (MemoryExtraction, UserProfile)
│   └── strategies/
│       ├── base.py         # MemoryStrategy ABC
│       ├── baseline.py     # No cross-session memory
│       ├── semantic.py     # ChromaDB + sentence-transformers
│       ├── structured.py   # JSON profile with fixed schema
│       ├── hybrid.py       # Both stores + source-of-truth prompting
│       ├── nodes.py        # LangGraph node name enum
│       └── __init__.py     # Registry + factory
├── evals/
│   ├── harness.py          # Comparison harness (uv run harness)
│   └── test_agent.py       # pytest smoke tests
├── frontend/
│   └── src/App.jsx         # React chat UI with memory panel
└── data/                   # Runtime storage (gitignored)
    ├── chroma/             # Semantic memory vector DB
    └── profiles/           # Structured memory JSON files
```

## Supported providers

| Provider  | Model string example                            | Required env var    |
|-----------|------------------------------------------------|---------------------|
| Anthropic | `anthropic:claude-haiku-4-5-20251001` (default) | `ANTHROPIC_API_KEY` |
| OpenAI    | `openai:gpt-4o`                                 | `OPENAI_API_KEY`    |
| Google    | `google_genai:gemini-2.5-flash`                 | `GOOGLE_API_KEY`    |

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- Node.js 18+ (for frontend only)
- At least one LLM provider API key
