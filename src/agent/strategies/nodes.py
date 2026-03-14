from enum import StrEnum


class Node(StrEnum):
    LLM = "llm"
    RECALL = "recall"
    MEMORIZE = "memorize"
    LOAD_PROFILE = "load_profile"
    EXTRACT_PROFILE = "extract_profile"
