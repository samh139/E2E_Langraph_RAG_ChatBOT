"""State passed between nodes in the retrieval LangGraph."""

from typing import Any

from pydantic import BaseModel, Field


class RetrieverState(BaseModel):
    session_id: str
    query_id: str
    query: str
    user_id: str = "default_user"

    intent: str | None = None
    refined_query: str = ""
    short_term_memory: list[dict[str, str]] = Field(default_factory=list)
    long_term_memory: list[dict[str, str]] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    response: str = ""
    error: str | None = None
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
