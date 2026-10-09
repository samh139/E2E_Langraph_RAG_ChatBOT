"""State passed between nodes in the retrieval LangGraph."""

from typing import Any

from pydantic import BaseModel, Field


class RetrieverState(BaseModel):
    session_id: str
    query_id: str
    query: str
    user_id: str = "default_user"

    intent: str | None = None
    intent_meta: dict[str, Any] = Field(default_factory=dict)
    refined_query: str = ""
    should_refine: bool = False
    refinement_reason: str = ""
    short_term_memory: dict[str, Any] = Field(default_factory=dict)
    long_term_memory: list[dict[str, Any]] = Field(default_factory=list)

    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    retrieval_score: float = 0.0
    retrieval_threshold: float = 0.5
    retrieval_passed: bool = False
    fallback_reason: str | None = None
    retrieval_warning: str | None = None

    response: str = ""
    error: str | None = None
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
