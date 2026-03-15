import json
import re
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from agent.schemas import UserProfile
from agent.strategies.base import MemoryStrategy
from agent.strategies.nodes import Node

_EXTRACT_PROMPT = """\
Extract structured profile information about the user from this conversation.

Only extract facts the user has explicitly stated. Do not infer or guess.
If a field was not mentioned, leave it as null / empty."""

_SCALAR_FIELDS = ("name", "role", "company", "location", "response_style", "preferred_language")
_LIST_FIELDS = ("projects",)


def _merge_profiles(current: dict, extracted: dict) -> dict:
    """Merge an extracted profile into the current one.

    Scalars: new non-null value overwrites.
    Lists: union with deduplication, never discard existing entries.
    """
    merged = dict(current)
    for key in _SCALAR_FIELDS:
        if extracted.get(key):
            merged[key] = extracted[key]
    for key in _LIST_FIELDS:
        existing = set(merged.get(key) or [])
        new_items = set(extracted.get(key) or [])
        merged[key] = list(existing | new_items)
    return merged


class StructuredState(TypedDict):
    """Graph state flowing through the load_profile → llm → extract_profile pipeline."""

    messages: list
    user_id: str
    profile: dict


def _sanitize(name: str) -> str:
    """Sanitize a string into a safe filesystem name."""
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return s or "unknown"


class StructuredMemory(MemoryStrategy):
    """JSON-profile-based cross-session memory with fixed schema extraction.

    Implements a LangGraph pipeline: START → load_profile → llm → extract_profile → END.
    User profiles are stored as human-readable JSON files. On each turn the
    profile is loaded and injected as a system message; after the LLM responds,
    new profile fields are extracted and merged.
    """

    def __init__(
        self,
        model_str: str = "anthropic:claude-haiku-4-5-20251001",
        data_dir: str = "./data",
        **_kwargs: object,
    ) -> None:
        super().__init__(model_str)
        self._profiles_dir = Path(data_dir) / "profiles"
        self._profiles_dir.mkdir(parents=True, exist_ok=True)
        self._threads: dict[str, list] = {}
        self._graph = self._build_graph()

    @property
    def name(self) -> str:
        return "structured"

    def _profile_path(self, user_id: str) -> Path:
        """Return the JSON file path for a user's profile."""
        return self._profiles_dir / f"{_sanitize(user_id)}.json"

    def _load_profile_file(self, user_id: str) -> dict:
        """Read a user's profile from disk, returning an empty dict if it doesn't exist."""
        path = self._profile_path(user_id)
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def _save_profile_file(self, user_id: str, profile: dict) -> None:
        """Write a user's profile to disk as formatted JSON."""
        path = self._profile_path(user_id)
        path.write_text(json.dumps(profile, indent=2))

    def _load_profile_node(self, state: StructuredState) -> dict:
        """Read the user's profile from disk into graph state."""
        profile = self._load_profile_file(state["user_id"])
        return {"profile": profile}

    def _llm_node(self, state: StructuredState) -> dict:
        """Invoke the LLM with the user's profile injected as a system message.

        Only non-empty profile fields are included in the prompt.
        Normalizes response content that may be ``str`` or ``list[dict]``.
        """
        messages = list(state["messages"])

        profile = state.get("profile") or {}
        profile_lines = []
        for key, val in profile.items():
            if not val:
                continue
            label = key.replace("_", " ").title()
            if isinstance(val, list):
                profile_lines.append(f"{label}: {', '.join(val)}")
            else:
                profile_lines.append(f"{label}: {val}")

        if profile_lines:
            messages.insert(
                0,
                SystemMessage(content="User profile:\n" + "\n".join(profile_lines)),
            )

        response = self.llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return {"messages": messages + [AIMessage(content=content)]}

    def _extract_profile_node(self, state: StructuredState) -> dict:
        """Extract structured profile fields from the conversation, merge, and persist.

        Handles both ``UserProfile`` instances and raw ``dict`` returns from
        ``with_structured_output``. Extraction failures are silently swallowed.
        """
        try:
            extractor = self.llm.with_structured_output(UserProfile)
            extraction = extractor.invoke(
                [SystemMessage(content=_EXTRACT_PROMPT)] + state["messages"]
            )
            if extraction is not None:
                extracted = (
                    extraction.model_dump(exclude_none=True)
                    if isinstance(extraction, UserProfile)
                    else extraction
                )
                merged = _merge_profiles(state.get("profile") or {}, extracted)
                self._save_profile_file(state["user_id"], merged)
                return {"profile": merged}
        except Exception:
            pass
        return {}

    def _build_graph(self) -> CompiledStateGraph:
        """Compile the load_profile → llm → extract_profile LangGraph pipeline."""
        g = StateGraph(StructuredState)
        g.add_node(Node.LOAD_PROFILE, self._load_profile_node)
        g.add_node(Node.LLM, self._llm_node)
        g.add_node(Node.EXTRACT_PROFILE, self._extract_profile_node)
        g.add_edge(START, Node.LOAD_PROFILE)
        g.add_edge(Node.LOAD_PROFILE, Node.LLM)
        g.add_edge(Node.LLM, Node.EXTRACT_PROFILE)
        g.add_edge(Node.EXTRACT_PROFILE, END)
        return g.compile()

    def chat(self, message: str, *, user_id: str, thread_id: str) -> str:
        """Send a message and return the assistant reply.

        Maintains per-thread conversation history in-process and invokes the
        full load_profile → llm → extract_profile graph each turn.
        """
        history = self._threads.setdefault(thread_id, [])
        history.append(HumanMessage(content=message))

        result = self._graph.invoke({
            "messages": list(history),
            "user_id": user_id,
            "profile": {},
        })

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        reply = ai_msgs[-1].content if ai_msgs else ""
        history.append(AIMessage(content=reply))
        return reply

    def inspect(self, user_id: str) -> dict:
        """Return the stored JSON profile for a user."""
        profile = self._load_profile_file(user_id)
        return {"type": "structured", "stored": profile}
