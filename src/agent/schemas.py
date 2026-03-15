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


class UserProfile(BaseModel):
    """Fixed-schema profile extracted from conversations by structured memory.

    Deliberately omits catch-all fields like ``interests`` or ``facts`` —
    this forces schema blindness for information that doesn't fit the
    predefined fields, which is the key trade-off structured memory demonstrates.
    """

    name: str | None = None
    role: str | None = None
    company: str | None = None
    location: str | None = None
    response_style: str | None = None
    projects: list[str] = Field(default_factory=list)
    preferred_language: str | None = None
