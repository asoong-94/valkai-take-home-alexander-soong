from agent.strategies.base import MemoryStrategy
from agent.strategies.baseline import Baseline
from agent.strategies.hybrid import HybridMemory
from agent.strategies.semantic import SemanticMemory
from agent.strategies.structured import StructuredMemory

REGISTRY: dict[str, type[MemoryStrategy]] = {
    "baseline": Baseline,
    "semantic": SemanticMemory,
    "structured": StructuredMemory,
    "hybrid": HybridMemory,
}


def make_strategy(
    name: str,
    model_str: str = "anthropic:claude-haiku-4-5-20251001",
    **kwargs,
) -> MemoryStrategy:
    cls = REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy {name!r}. Choose from: {list(REGISTRY)}")
    return cls(model_str=model_str, **kwargs)
