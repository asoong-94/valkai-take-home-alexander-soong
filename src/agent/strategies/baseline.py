from langchain_core.messages import AIMessage, HumanMessage

from agent.strategies.base import MemoryStrategy


class Baseline(MemoryStrategy):
    """No cross-session memory — conversation window only."""

    def __init__(self, model_str: str = "anthropic:claude-haiku-4-5-20251001", **_kwargs):
        super().__init__(model_str)
        self._threads: dict[str, list] = {}

    @property
    def name(self) -> str:
        return "baseline"

    def chat(self, message: str, *, user_id: str, thread_id: str) -> str:
        history = self._threads.setdefault(thread_id, [])
        history.append(HumanMessage(content=message))

        response = self.llm.invoke(history)

        content = response.content
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )

        history.append(AIMessage(content=content))
        return content

    def inspect(self, user_id: str) -> dict:
        return {"type": "none", "stored": []}
