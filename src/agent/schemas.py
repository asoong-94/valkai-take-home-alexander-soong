from pydantic import BaseModel, Field


class MemoryExtraction(BaseModel):
    """Schema for LLM-extracted facts from a conversation.

    Used by semantic memory's memorize node with ``with_structured_output``
    to pull declarative user facts out of the conversation history.
    """

    facts: list[str] = Field(
        default_factory=list,
        description="Short declarative statements about the user worth storing long-term. "
        "Include only: name, role, preferences, projects, interests. "
        "Return an empty list if nothing is worth storing.",
    )
