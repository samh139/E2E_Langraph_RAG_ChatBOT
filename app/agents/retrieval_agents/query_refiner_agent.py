from __future__ import annotations

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from app.configs.llm_config import ask_gemini_structured
from app.memory_service.memory_team.ltm.ltm_service import retrieve_ltm_context
from app.memory_service.memory_team.stm.store import get_stm_summary
from app.workflow.retrieval_state import RetrieverState


logger = logging.getLogger(__name__)


class RefinedQueryResponse(BaseModel):
    should_refine: bool = Field(
        description="True only when the query needs supplied conversation context."
    )
    refined_query: str = Field(
        description="A concise, natural retrieval query with any necessary context."
    )
    reason: str = Field(description="Brief reason for the refinement decision.")


class QueryRefinerAgent:
    def __init__(self, payload: RetrieverState):
        self.payload = payload

    def _build_system_prompt(self) -> str:
            return """
            You are a query refinement component in a Retrieval-Augmented Generation system.

            Your job is to convert the user's current query into a retrieval-ready query
            ONLY when conversational context is required to understand the query.

            You may receive:
            1. The current user query.
            2. Short-Term Memory (STM), representing the cumulative summary of the
            current session.
            3. Long-Term Memory (LTM), containing relevant previous conversation turns.

            IMPORTANT PRINCIPLES:

            1. SYNTACTIC NORMALIZATION IS ALLOWED.
            If a query is a collection of jumbled keywords or fragmented phrases (e.g., "NEFT fees for SBI Bank what is"),
            smooth it out into a clean, natural search query (e.g., "What are the NEFT fees for SBI Bank?")..

            2. PRESERVE THE ORIGINAL MEANING.
            Never introduce facts, entities, products, organizations, locations,
            transaction types, policies, or terminology that are not supported by
            the current query or supplied memory.

            3. USE MEMORY ONLY WHEN NECESSARY.
            Memory should be used when the current query contains an unresolved
            reference, continuation, omission, pronoun, or contextual dependency.

            4. DO NOT FORCE CONTEXT.
            If the current query is already understandable and independently
            searchable, return it unchanged.

            5. MINIMAL REFINEMENT.
            When refinement is necessary, make the smallest change required to make
            the query independently understandable for retrieval.

            6. PRESERVE RETRIEVAL TERMS.
            Important terminology from the user's query must remain in the refined
            query. Do not replace specific terms with vague synonyms.

            7. DO NOT SUMMARIZE THE CONVERSATION.
            Produce one retrieval query, not an explanation of the conversation.

            8. DO NOT ANSWER THE USER'S QUESTION.
            Your output is only a retrieval-ready query.

            9. DO NOT INVENT MISSING INFORMATION.
            If STM/LTM does not provide enough information to resolve an ambiguous
            reference, leave the query unchanged rather than guessing.

            10. STM AND LTM HAVE DIFFERENT ROLES.
                Prefer STM for immediate conversational context.
                Use LTM when it contains relevant context that STM does not provide.
                Do not incorporate unrelated memories.

            11. A REFINED QUERY SHOULD REMAIN NATURAL.
                It should look like a query that could realistically be sent to a
                document retriever.

            DECISION:

            - If the query is independently understandable:
                should_refine = false
                refined_query = original query

            - If the query depends on previous conversation and the supplied memory
            resolves that dependency:
                should_refine = true
                refined_query = minimally rewritten query containing the necessary
                context

            - If the query depends on previous conversation but the supplied memory
            cannot reliably resolve it:
                should_refine = false
                refined_query = original query

            Return only the structured response requested by the schema.
            """

    def _build_user_prompt(
        self, query: str, stm_context: Any, ltm_context: Any
    ) -> str:
        return (
            f"CURRENT USER QUERY:\n{query}\n\n"
            f"SHORT-TERM MEMORY:\n{stm_context or '[No STM available]'}\n\n"
            f"LONG-TERM MEMORY:\n{ltm_context or '[No relevant LTM available]'}\n\n"
            "Return one retrieval-ready query. If context is insufficient, do not guess."
        )

    async def execute(self) -> dict[str, Any]:
        if not isinstance(self.payload, RetrieverState):
            raise TypeError("QueryRefinerAgent payload must be a RetrieverState")
        query = self.payload.query.strip()
        session_id = self.payload.session_id
        user_id = self.payload.user_id
        if not query:
            return {
                "refined_query": "",
                "should_refine": False,
                "refinement_reason": "Empty query.",
            }

        stm_context: dict[str, Any] = {}
        ltm_context: list[dict[str, Any]] = []
        if session_id:
            stm_result, ltm_result = await asyncio.gather(
                get_stm_summary(session_id=session_id, user_id=user_id),
                retrieve_ltm_context(query=query, session_id=session_id, user_id=user_id),
                return_exceptions=True,
            )
            if isinstance(stm_result, BaseException):
                logger.warning("STM lookup failed; continuing without STM: %s", stm_result)
            elif isinstance(stm_result, dict):
                stm_context = stm_result

            if isinstance(ltm_result, BaseException):
                logger.warning("LTM lookup failed; continuing without LTM: %s", ltm_result)
            elif isinstance(ltm_result, list):
                ltm_context = ltm_result

        result = await ask_gemini_structured(
            system_prompt=self._build_system_prompt(),
            user_prompt=self._build_user_prompt(query, stm_context, ltm_context),
            response_schema=RefinedQueryResponse,
        )

        refined_value = result.get("refined_query") if isinstance(result, dict) else None
        refined_query = refined_value.strip() if isinstance(refined_value, str) else ""
        if not refined_query:
            refined_query = query

        reason = result.get("reason", "") if isinstance(result, dict) else ""
        should_refine = bool(result.get("should_refine", False)) if isinstance(result, dict) else False
        logger.info(
            "Refined query=%r should_refine=%s reason=%s",
            refined_query,
            should_refine,
            reason,
        )
        return {
            "refined_query": refined_query,
            "should_refine": should_refine,
            "refinement_reason": str(reason or ""),
            "short_term_memory": stm_context,
            "long_term_memory": ltm_context,
        }
