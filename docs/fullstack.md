# Fullstack guide

Run the agent as a FastAPI server with a React chat frontend. See the [core README](../README.md) for initial setup.

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

```
POST /chat
Content-Type: application/json

{
  "messages": [
    { "role": "user", "content": "Hello!" }
  ]
}
```

```json
{
  "reply": "Hi! How can I help you?"
}
```

The frontend sends the full conversation history on each request. The server is stateless — no session storage.

## How it works

`src/agent/server.py` initializes the agent once at startup using the same `make_agent()` factory from `core.py`. The React frontend (`frontend/src/App.jsx`) manages conversation state locally and posts the full message list on every send.

## Relevant files

```
src/agent/
├── core.py       # agent factory (shared)
└── server.py     # FastAPI app, POST /chat endpoint

frontend/
├── src/
│   ├── App.jsx   # chat UI component
│   └── main.jsx  # React entry point
├── index.html
├── vite.config.js
└── package.json
```
