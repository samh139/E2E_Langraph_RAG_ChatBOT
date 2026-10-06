# query_refiner.py
from __future__ import annotations
import asyncio
from collections.abc import Mapping
from typing import Any
from pydantic import BaseModel, Field
from app.workflow.retrieval_state import RetrieverState
from app.configs.llm_config import ask_gemini_structured
from app.memory_service.memory_team.stm.store import get_stm_summary
from app.memory_service.memory_team.ltm.ltm_service import retrieve_ltm_context


class RefinedQueryResponse(BaseModel):
    should_refine: bool = Field(
        description=(
            "True only when the current query depends on information from "
            "the supplied STM/LTM context."
        )
    )
    refined_query: str = Field(
        description=(
            "Retrieval-ready query. If memory is not required, return the "
            "original user query unchanged."
        )
    )
    reason: str = Field(
        description=(
            "Short explanation of why the query was refined or left unchanged."
        )
    )


class QueryRefinerAgent:
    def __init__(self, payload: str | Mapping[str, Any] | RetrieverState):
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

        1. BE CONSERVATIVE.
        Do not rewrite a query merely to make it sound more formal or complete.

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

    def _get_context(self, state: RetrieverState) -> tuple[Any, Any]:
        """
        Fixed: Correctly reading fields defined in RetrieverState schema.
        """
        stm_context = getattr(state, "short_term_memory", None)
        ltm_context = getattr(state, "long_term_memory", None)
        return stm_context, ltm_context

    def _build_user_prompt(self, query: str, stm_context: Any, ltm_context: Any) -> str:
        return f"""
CURRENT USER QUERY:
{query}

SHORT-TERM MEMORY:
{stm_context if stm_context else "[No STM available]"}

LONG-TERM MEMORY:
{ltm_context if ltm_context else "[No relevant LTM available]"}

Determine whether the current query actually requires conversational context.
If it does, resolve only the missing context required for retrieval.
If it does not, return the current query unchanged.
If the memory is insufficient to resolve an ambiguous reference, do not guess.
"""

    # Fixed: Marked method as async to handle await statements inside
    async def execute(self) -> dict[str, Any]:
        """
        Refine the current knowledge query using available STM/LTM context.
        """
        # 1. Parse operational variables depending on the input payload configuration
        if isinstance(self.payload, RetrieverState):
            query = (self.payload.query or "").strip()
            session_id = self.payload.session_id
            user_id = self.payload.user_id
        elif isinstance(self.payload, Mapping):
            query = str(self.payload.get("query", "")).strip()
            session_id = str(self.payload.get("session_id", ""))
            user_id = str(self.payload.get("user_id", "default_user"))
        else:
            query = str(self.payload).strip()
            session_id = ""
            user_id = "default_user"

        # Guard clause for empty queries
        if not query:
            return {
                "refined_query": "",
                "should_refine": False,
                "refinement_reason": "Empty query."
            }

        # 2. Fetch context if execution session data is available
        stm_context = None
        ltm_context = None
        
        if session_id:
            stm_task = get_stm_summary(session_id=session_id, user_id=user_id)
            ltm_task = retrieve_ltm_context(query=query, session_id=session_id, user_id=user_id)
            stm_context, ltm_context = await asyncio.gather(stm_task, ltm_task)

        # 3. Request LLM refinement structure using the populated parameters
        result = await ask_gemini_structured(
            system_prompt=self._build_system_prompt(),
            user_prompt=self._build_user_prompt(
                query=query,
                stm_context=stm_context,
                ltm_context=ltm_context,
            ),
            response_schema=RefinedQueryResponse,
        )

        # 4. Fallback safeguard verification
        refined_query = (result.get("refined_query") or query).strip()
        if not refined_query:
            refined_query = query

        return {
            "refined_query": refined_query,
            "should_refine": bool(result.get("should_refine", False)),
            "refinement_reason": result.get("reason", ""),
        }
