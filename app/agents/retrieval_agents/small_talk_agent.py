from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.configs.llm_config import ask_gemini_structured
from app.workflow.retrieval_state import RetrieverState


logger = logging.getLogger(__name__)


class SmallTalkResponse(BaseModel):
    response: str = Field(description="A brief, natural response to the user.")


class SmallTalkAgent:
    """Handle small talk and the retrieval workflow's no-answer fallback."""

    def __init__(self, payload: RetrieverState):
        self.payload = payload

    def _build_system_prompt(self, intent: str) -> str:
        return f"""You are a polite, helpful banking information assistant.
The user's intent was classified as {intent!r}. Reply briefly and naturally.
Do not claim to access accounts, execute transactions, or perform actions.
For off-topic questions, respond warmly and explain that you can help with
banking information from the available documents."""

    async def execute(self) -> dict[str, Any]:
        if not isinstance(self.payload, RetrieverState):
            raise TypeError("SmallTalkAgent payload must be a RetrieverState")
        state = self.payload
        if state.fallback_reason == "low_relevance":
            return {
                "response": (
                    "I couldn’t find a sufficiently relevant answer in the available "
                    "documents. Could you rephrase the question or mention the bank or product?"
                )
            }
        if state.fallback_reason == "retrieval_error":
            return {
                "response": (
                    "I’m having trouble searching the available documents right now. "
                    "Please try again in a moment."
                )
            }

        try:
            result = await ask_gemini_structured(
                system_prompt=self._build_system_prompt(state.intent or "greeting"),
                user_prompt=f"Respond briefly to this message: {state.query}",
                response_schema=SmallTalkResponse,
            )
            response = result.get("response") if isinstance(result, dict) else None
            if isinstance(response, str) and response.strip():
                return {"response": response.strip()}
        except Exception:
            logger.exception("Small-talk response generation failed")

        # ask_gemini_structured returns a generic fallback object on API errors;
        # this deterministic response also prevents None from reaching memory storage.
        intent = state.intent or "greeting"
        if intent == "gratitude_ack":
            response = "You’re very welcome!"
        elif intent == "farewell":
            response = "Goodbye! Feel free to return whenever you need help."
        elif intent == "apology_ack":
            response = "No worries. How can I help?"
        elif intent == "offtopic_query":
            response = "I’m here to help with banking information from the available documents."
        else:
            response = "Hello! How can I help with your banking question today?"
        return {"response": response}
