from abc import ABC, abstractmethod

from langchain.chat_models import init_chat_model


class MemoryStrategy(ABC):
    """Abstract base class for all memory strategies.

    Subclasses implement a ``chat`` method that handles a single user message
    (with cross-session memory semantics determined by the strategy) and an
    ``inspect`` method that exposes the raw stored state for debugging/eval.
    """

    def __init__(self, model_str: str = "anthropic:claude-haiku-4-5-20251001") -> None:
        self.llm = init_chat_model(model_str)
        self._model_str = model_str

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy identifier."""
        ...

    @abstractmethod
    def chat(self, message: str, *, user_id: str, thread_id: str) -> str:
        """Process a user message and return the assistant reply.

        Args:
            message: The user's message text.
            user_id: Scopes cross-session memory to this user.
            thread_id: Scopes in-session conversation history to this thread.

        Returns:
            The assistant's reply as a plain string.
        """
        ...

    @abstractmethod
    def inspect(self, user_id: str) -> dict:
        """Return the raw stored memory state for a given user.

        Used by the eval harness and inspect endpoint to surface what
        each strategy has actually persisted.
        """
        ...
