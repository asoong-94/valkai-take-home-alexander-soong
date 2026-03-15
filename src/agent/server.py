import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.strategies import REGISTRY, make_strategy
from agent.strategies.base import MemoryStrategy

load_dotenv()

_MODEL = os.getenv("MODEL", "anthropic:claude-haiku-4-5-20251001")
_DATA_DIR = os.getenv("DATA_DIR", "./data")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_strategies: dict[str, MemoryStrategy] = {}


def _get_strategy(name: str) -> MemoryStrategy:
    if name not in _strategies:
        _strategies[name] = make_strategy(name, _MODEL, data_dir=_DATA_DIR)
    return _strategies[name]


class ChatRequest(BaseModel):
    message: str
    memory: str = "baseline"
    user_id: str = "default-user"
    thread_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    thread_id: str


@app.post("/chat")
def chat(req: ChatRequest) -> ChatResponse:
    thread_id = req.thread_id or uuid.uuid4().hex[:12]
    strategy = _get_strategy(req.memory)
    reply = strategy.chat(req.message, user_id=req.user_id, thread_id=thread_id)
    return ChatResponse(reply=reply, thread_id=thread_id)


@app.get("/strategies")
def list_strategies():
    return {"strategies": list(REGISTRY.keys())}


@app.get("/users")
def list_users():
    """Discover user IDs from stored profile files and chroma collections."""
    user_ids: set[str] = set()

    profiles_dir = Path(_DATA_DIR) / "profiles"
    if profiles_dir.exists():
        for f in profiles_dir.glob("*.json"):
            stem = f.stem
            if stem.startswith("hybrid_"):
                stem = stem[7:]
            user_ids.add(stem)

    chroma_dir = Path(_DATA_DIR) / "chroma"
    if chroma_dir.exists():
        import chromadb

        try:
            client = chromadb.PersistentClient(path=str(chroma_dir))
            for col in client.list_collections():
                name = col.name if hasattr(col, "name") else str(col)
                for prefix in ("user_", "hybrid_"):
                    if name.startswith(prefix):
                        user_ids.add(name[len(prefix):])
        except Exception:
            pass

    return {"users": sorted(user_ids)}


@app.get("/inspect")
def inspect_memory(
    memory: str = Query(..., description="Strategy name"),
    user_id: str = Query("default-user", description="User ID"),
):
    strategy = _get_strategy(memory)
    return strategy.inspect(user_id)


def main():
    import uvicorn

    uvicorn.run("agent.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
