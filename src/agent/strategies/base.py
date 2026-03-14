from abc import ABC, abstractmethod

from langchain.chat_models import init_chat_model


class MemoryStrategy(ABC):
    def __init__(self, model_str: str = "anthropic:claude-haiku-4-5-20251001"):
        self.llm = init_chat_model(model_str)
        self._model_str = model_str

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def chat(self, message: str, *, user_id: str, thread_id: str) -> str: ...

    @abstractmethod
    def inspect(self, user_id: str) -> dict: ...
