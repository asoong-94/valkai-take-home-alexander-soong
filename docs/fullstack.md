# Fullstack guide

Run the agent as a FastAPI server with a React chat frontend that lets you switch memory strategies and inspect stored memory live. See the [core README](../README.md) for initial setup.

## Prerequisites

- Everything in the core README
- Node.js 18+

## Start

Run both processes in separate terminals.

**Terminal 1 — backend:**

```bash
uv run serve
```

Server starts at `http://localhost:8000`.

**Terminal 2 — frontend:**

```bash
cd frontend
npm install   # first time only
npm run dev
```

UI opens at `http://localhost:3000`.

## API

### `POST /chat`

Send a message using a specific memory strategy.

```json
// Request
{
  "message": "My name is Alex and I work at Stripe.",
  "memory": "structured",
  "user_id": "alex",
  "thread_id": null
}

// Response
{
  "reply": "Nice to meet you, Alex! ...",
  "thread_id": "a1b2c3d4e5f6"
}
```

The server manages conversation history per `thread_id`. On the first request, omit `thread_id` (or send `null`) and the server generates one. Send it back on subsequent requests to continue the same conversation.

### `GET /strategies`

List available memory strategies.

```json
{ "strategies": ["baseline", "semantic", "structured"] }
```

### `GET /users`

Discover existing user IDs from persistent storage (profile files and ChromaDB collections).

```json
{ "users": ["alex", "default-user", "demo"] }
```

### `GET /inspect?memory=semantic&user_id=alex`

View what a strategy has stored for a user.

```json
// Semantic
{ "type": "semantic", "stored": ["Name: Alex", "Role: Senior Backend Engineer", "Company: Stripe"] }

// Structured
{ "type": "structured", "stored": { "name": "Alex", "role": "Senior Backend Engineer", "company": "Stripe", "projects": [] } }

// Baseline
{ "type": "none", "stored": [] }
```

## Frontend features

- **Strategy picker** — dropdown to switch between baseline, semantic, and structured
- **User dropdown** — select from existing users or create a new one with the "+ new user" option
- **New Session button** — clears the conversation and starts a fresh `thread_id`
- **Stored Memory panel** — left sidebar that updates live after every message:
  - Baseline: shows "Baseline stores nothing cross-session"
  - Semantic: scrollable list of extracted facts
  - Structured: dictionary table showing profile fields (Name, Role, Company, Location, etc.) with empty fields displayed as "—"
- **Refresh Memory button** — manually re-fetch the stored memory state

## How it works

`src/agent/server.py` lazily initializes one strategy instance per strategy name. The React frontend (`frontend/src/App.jsx`) manages `thread_id` and sends a single message per request. After each response, the frontend automatically calls `/inspect` to refresh the memory panel.

## Relevant files

```
src/agent/
├── server.py         # FastAPI app, /chat, /strategies, /users, /inspect
├── core.py           # agent factory (shared)
├── schemas.py        # Pydantic models
└── strategies/       # memory strategy implementations

frontend/
├── src/
│   ├── App.jsx       # chat UI with strategy picker + memory panel
│   └── main.jsx      # React entry point
├── index.html
├── vite.config.js
└── package.json
```
