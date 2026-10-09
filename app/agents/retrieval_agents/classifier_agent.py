# app/agents/retrieval_agents/classifier_agent.py
from __future__ import annotations
import json
from typing import Any
from pydantic import BaseModel, Field

from app.workflow.retrieval_state import RetrieverState
from app.configs.llm_config import ask_gemini_structured, load_intent_taxonomy

# Define a strict schema response for Gemini
class IntentClassification(BaseModel):
    intent: str = Field(description="The matching key from the intent taxonomy.")
    confidence: float = Field(description="Confidence value between 0.0 and 1.0.")

class ClassifierAgent:
    """Detect the exact intent of an incoming user query using Gemini and a dynamic taxonomy."""

    def __init__(self, payload: RetrieverState):
        self.payload = payload
        # Load the custom file rules you want to reuse
        self.taxonomy = load_intent_taxonomy()

    def _build_system_prompt(self) -> str:
        """Injects your file configuration examples into the system context."""
        taxonomy_dump = json.dumps(self.taxonomy, indent=2)
        
        return f"""You are an elite intent classification routing agent for an enterprise system.
Your job is to match the user's incoming message into exactly ONE of the configured intent families provided below.

Here is the authoritative taxonomy configuration rules containing keys, examples, and intent tags:
{taxonomy_dump}

INSTRUCTIONS:
1. Carefully analyze the semantics, tone, and goals of the user's input.
2. Select the absolute best matching key from the provided taxonomy configuration.
3. If no key clearly matches, fallback to 'generic_query'.
"""

    async def _detect_intent(self, query: str) -> dict[str, Any]:
        """Call Gemini using the structured response wrapper."""
        system_prompt = self._build_system_prompt()
        user_prompt = f"User Message to Classify: '{query}'"
        
        # Invoke Gemini with schema verification
        result = await ask_gemini_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=IntentClassification
        )
        return result

    async def execute(self) -> dict[str, Any]:
        """Return a precise state patch suitable for downstream LangGraph nodes."""
        if not isinstance(self.payload, RetrieverState):
            raise TypeError("ClassifierAgent payload must be a RetrieverState")
        state = self.payload
        query = state.query.strip()

        classification = await self._detect_intent(query)
        if not isinstance(classification, dict):
            classification = {}

        intent = classification.get("intent", "generic_query")
        if intent not in self.taxonomy:
            intent = "generic_query"
        try:
            confidence = min(1.0, max(0.0, float(classification.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0

        return {
            "session_id": state.session_id,
            "query_id": state.query_id,
            "user_id": state.user_id,
            "query": query,
            "intent": intent,
            "intent_meta": {"confidence": confidence},
        }
