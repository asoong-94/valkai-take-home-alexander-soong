import re

import chromadb
from chromadb import Collection, EmbeddingFunction, Embeddings, Documents
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sentence_transformers import SentenceTransformer
from typing_extensions import TypedDict

from agent.schemas import MemoryExtraction
from agent.strategies.base import MemoryStrategy
from agent.strategies.nodes import Node

_ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

_EXTRACT_PROMPT = """\
Extract factual information about the user from this conversation.

STORE: name, role, company, location, preferences, interests, projects, skills.
DO NOT STORE: greetings, temporary requests, questions, assistant output, opinions about weather/time.

Return only facts explicitly stated by the user."""


class _LocalEF(EmbeddingFunction):
    """Wraps sentence-transformers for Chroma so we don't need an API key for embeddings."""

    def __call__(self, input: Documents) -> Embeddings:
        return _ST_MODEL.encode(input, show_progress_bar=False).tolist()


class SemanticState(TypedDict):
    """Graph state flowing through the recall → llm → memorize pipeline."""

    messages: list
    user_id: str
    recalled: list[str]


def _sanitize(name: str) -> str:
    """Sanitize a string into a valid Chroma collection name (3-63 alphanumeric/underscore/hyphen chars)."""
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    if len(s) < 3:
        s = s + "_" * (3 - len(s))
    return s[:63]


class SemanticMemory(MemoryStrategy):
    """Vector-based cross-session memory using ChromaDB + sentence-transformers.

    Implements a LangGraph pipeline: START → recall → llm → memorize → END.
    Facts are embedded and stored per-user in ChromaDB; on each turn the top-3
    most similar memories are retrieved and injected as a system message.
    """

    def __init__(
        self,
        model_str: str = "anthropic:claude-haiku-4-5-20251001",
        data_dir: str = "./data",
        **_kwargs: object,
    ) -> None:
        super().__init__(model_str)
        self._chroma = chromadb.PersistentClient(path=f"{data_dir}/chroma")
        self._ef = _LocalEF()
        self._threads: dict[str, list] = {}
        self._graph = self._build_graph()

    @property
    def name(self) -> str:
        return "semantic"

    def _collection(self, user_id: str) -> Collection:
        """Get or create a per-user Chroma collection with local embeddings."""
        return self._chroma.get_or_create_collection(
            name=f"user_{_sanitize(user_id)}",
            embedding_function=self._ef,
        )

    def _recall_node(self, state: SemanticState) -> dict:
        """Embed the latest user message and retrieve the top-3 most similar stored facts.

        The top-k=3 ceiling is intentional — it demonstrates the exhaustive-recall
        limitation of vector retrieval in the eval harness.
        """
        col = self._collection(state["user_id"])
        if col.count() == 0:
            return {"recalled": []}

        last_msg = state["messages"][-1].content
        results = col.query(
            query_texts=[last_msg],
            n_results=min(3, col.count()),
        )
        docs = results["documents"][0] if results["documents"] else []
        return {"recalled": docs}

    def _llm_node(self, state: SemanticState) -> dict:
        """Invoke the LLM with recalled memories prepended as a system message.

        Normalizes response content that may be ``str`` or ``list[dict]``
        (varies by provider) into a plain string.
        """
        messages = list(state["messages"])

        if state["recalled"]:
            memory_text = "\n".join(f"- {fact}" for fact in state["recalled"])
            messages.insert(
                0,
                SystemMessage(
                    content=f"Relevant things you remember about this user:\n{memory_text}"
                ),
            )

        response = self.llm.invoke(messages)
        content = response.content
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return {"messages": messages + [AIMessage(content=content)]}

    def _memorize_node(self, state: SemanticState) -> dict:
        """Extract declarative facts from the conversation and persist them to ChromaDB.

        Uses ``with_structured_output(MemoryExtraction)`` to pull facts through
        the write-policy prompt. Extraction failures are silently swallowed so
        they never break the conversation flow.
        """
        try:
            extractor = self.llm.with_structured_output(MemoryExtraction)
            extraction = extractor.invoke(
                [SystemMessage(content=_EXTRACT_PROMPT)] + state["messages"]
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

    def _build_graph(self) -> CompiledStateGraph:
        """Compile the recall → llm → memorize LangGraph pipeline."""
        g = StateGraph(SemanticState)
        g.add_node(Node.RECALL, self._recall_node)
        g.add_node(Node.LLM, self._llm_node)
        g.add_node(Node.MEMORIZE, self._memorize_node)
        g.add_edge(START, Node.RECALL)
        g.add_edge(Node.RECALL, Node.LLM)
        g.add_edge(Node.LLM, Node.MEMORIZE)
        g.add_edge(Node.MEMORIZE, END)
        return g.compile()

    def chat(self, message: str, *, user_id: str, thread_id: str) -> str:
        """Send a message and return the assistant reply.

        Maintains per-thread conversation history in-process and invokes the
        full recall → llm → memorize graph each turn.
        """
        history = self._threads.setdefault(thread_id, [])
        history.append(HumanMessage(content=message))

        result = self._graph.invoke({
            "messages": list(history),
            "user_id": user_id,
            "recalled": [],
        })

        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        reply = ai_msgs[-1].content if ai_msgs else ""
        history.append(AIMessage(content=reply))
        return reply

    def inspect(self, user_id: str) -> dict:
        """Return all stored memory documents for a user."""
        col = self._collection(user_id)
        data = col.get()
        return {"type": "semantic", "stored": data.get("documents", [])}
