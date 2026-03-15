import json
import re
from pathlib import Path

import chromadb
from chromadb import Collection
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from agent.schemas import MemoryExtraction, UserProfile
from agent.strategies.base import MemoryStrategy
from agent.strategies.nodes import Node
from agent.strategies.semantic import _LocalEF, _sanitize as _sanitize_collection

_SCALAR_FIELDS = ("name", "role", "company", "location", "response_style", "preferred_language")
_LIST_FIELDS = ("projects",)

_EXTRACT_PROMPT = """\
Extract structured profile information about the user from this conversation.

Only extract facts the user has explicitly stated. Do not infer or guess.
If a field was not mentioned, leave it as null / empty."""

_FACTS_PROMPT = """\
Extract factual information about the user from this conversation.

STORE: name, role, company, location, preferences, interests, projects, skills,
       personal facts, hobbies, history, past experiences.
DO NOT STORE: greetings, temporary requests, questions, assistant output, opinions about weather/time.

Return only facts explicitly stated by the user."""


def _sanitize_path(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return s or "unknown"


def _merge_profiles(current: dict, extracted: dict) -> dict:
    """Scalars: new non-null value overwrites. Lists: union with dedup."""
    merged = dict(current)
    for key in _SCALAR_FIELDS:
        if extracted.get(key):
            merged[key] = extracted[key]
    for key in _LIST_FIELDS:
        existing = set(merged.get(key) or [])
        new_items = set(extracted.get(key) or [])
        merged[key] = list(existing | new_items)
    return merged


class HybridState(TypedDict):
    messages: list
    user_id: str
    profile: dict
    recalled: list[str]


class HybridMemory(MemoryStrategy):
    """Combined semantic + structured memory.

    Maintains both a fixed-schema JSON profile (authoritative current state)
    and a ChromaDB vector store (open-ended facts and history). On each turn
    both stores are consulted: the profile provides definitive answers for
    known fields while recalled facts supply supplementary context, personal
    details, and historical information the profile schema can't capture.

    LangGraph pipeline: START → load_profile → recall → llm → memorize → extract_profile → END
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
        self._chroma = chromadb.PersistentClient(path=f"{data_dir}/chroma")
        self._ef = _LocalEF()
        self._threads: dict[str, list] = {}
        self._graph = self._build_graph()

    @property
    def name(self) -> str:
        return "hybrid"

    def _collection(self, user_id: str) -> Collection:
        return self._chroma.get_or_create_collection(
            name=f"hybrid_{_sanitize_collection(user_id)}",
            embedding_function=self._ef,
        )

    def _profile_path(self, user_id: str) -> Path:
        return self._profiles_dir / f"hybrid_{_sanitize_path(user_id)}.json"

    def _load_profile_file(self, user_id: str) -> dict:
        path = self._profile_path(user_id)
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def _save_profile_file(self, user_id: str, profile: dict) -> None:
        path = self._profile_path(user_id)
        path.write_text(json.dumps(profile, indent=2))

    # ── graph nodes ──────────────────────────────────────────────────────

    def _load_profile_node(self, state: HybridState) -> dict:
        return {"profile": self._load_profile_file(state["user_id"])}

    def _recall_node(self, state: HybridState) -> dict:
        col = self._collection(state["user_id"])
        if col.count() == 0:
            return {"recalled": []}
        last_msg = state["messages"][-1].content
        results = col.query(
            query_texts=[last_msg],
            n_results=min(10, col.count()),
        )
        docs = results["documents"][0] if results["documents"] else []
        return {"recalled": docs}

    def _llm_node(self, state: HybridState) -> dict:
        messages = list(state["messages"])
        parts: list[str] = []

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
            parts.append(
                "User profile (authoritative current state — these are the latest values):\n"
                + "\n".join(profile_lines)
            )

        if state["recalled"]:
            memory_text = "\n".join(f"- {fact}" for fact in state["recalled"])
            parts.append(
                "Recalled facts (supplementary — includes historical info, personal details, "
                "and context the profile doesn't cover; use these to answer questions about "
                "the user's past, preferences, and interests):\n" + memory_text
            )

        if parts:
            messages.insert(0, SystemMessage(content="\n\n".join(parts)))

        response = self.llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return {"messages": messages + [AIMessage(content=content)]}

    def _memorize_node(self, state: HybridState) -> dict:
        try:
            extractor = self.llm.with_structured_output(MemoryExtraction)
            extraction = extractor.invoke(
                [SystemMessage(content=_FACTS_PROMPT)] + state["messages"]
            )
            if extraction and extraction.facts:
                col = self._collection(state["user_id"])
                for fact in extraction.facts:
                    col.add(
                        documents=[fact],
                        ids=[f"{state['user_id']}_{col.count()}"],
                    )
        except Exception:
            pass
        return {}

    def _extract_profile_node(self, state: HybridState) -> dict:
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
        g = StateGraph(HybridState)
        g.add_node(Node.LOAD_PROFILE, self._load_profile_node)
        g.add_node(Node.RECALL, self._recall_node)
        g.add_node(Node.LLM, self._llm_node)
        g.add_node(Node.MEMORIZE, self._memorize_node)
        g.add_node(Node.EXTRACT_PROFILE, self._extract_profile_node)
        g.add_edge(START, Node.LOAD_PROFILE)
        g.add_edge(Node.LOAD_PROFILE, Node.RECALL)
        g.add_edge(Node.RECALL, Node.LLM)
        g.add_edge(Node.LLM, Node.MEMORIZE)
        g.add_edge(Node.MEMORIZE, Node.EXTRACT_PROFILE)
        g.add_edge(Node.EXTRACT_PROFILE, END)
        return g.compile()

    def chat(self, message: str, *, user_id: str, thread_id: str) -> str:
        history = self._threads.setdefault(thread_id, [])
        history.append(HumanMessage(content=message))

        result = self._graph.invoke({
            "messages": list(history),
            "user_id": user_id,
            "profile": {},
            "recalled": [],
        })

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        reply = ai_msgs[-1].content if ai_msgs else ""
        history.append(AIMessage(content=reply))
        return reply

    def inspect(self, user_id: str) -> dict:
        profile = self._load_profile_file(user_id)
        col = self._collection(user_id)
        facts = col.get().get("documents", [])
        return {"type": "hybrid", "stored": {"profile": profile, "facts": facts}}
