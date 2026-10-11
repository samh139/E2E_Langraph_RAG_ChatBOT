"""Route small talk directly; retrieve knowledge with role filters before synthesis."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from app.agents.retrieval_agents.classifier_agent import ClassifierAgent
from app.agents.retrieval_agents.small_talk_agent import SmallTalkAgent
from app.agents.retrieval_agents.query_refiner_agent import QueryRefinerAgent
from app.agents.retrieval_agents.retrieval_agent import RetrievalAgent
from app.agents.retrieval_agents.synthesizer_agent import SynthesizerAgent
from app.utils.answer_cache import cache_answer, get_cached_answer
from app.workflow.retrieval_state import RetrieverState


class RetrievalWorkflow:
    """Classifier → small talk, or refiner → hybrid retrieval → answer/fallback."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k
        workflow = StateGraph(RetrieverState)

        workflow.add_node("classifier_agent", self.classifier_agent)
        workflow.add_node("small_talk_agent", self.small_talk_agent)
        workflow.add_node("query_refiner", self.query_refiner_agent)
        workflow.add_node("retrieval_agent", self.retrieval_agent)
        workflow.add_node("synthesizer_agent", self.synthesizer_agent)

        workflow.set_entry_point("classifier_agent")
        workflow.add_conditional_edges(
            "classifier_agent",
            self.route_intent,
            {
                "small_talk": "small_talk_agent",
                "knowledge": "query_refiner",
            },
        )
        workflow.add_edge("small_talk_agent", END)
        workflow.add_conditional_edges(
            "query_refiner",
            self.route_cache,
            {"hit": END, "miss": "retrieval_agent"},
        )
        workflow.add_conditional_edges(
            "retrieval_agent",
            self.route_retrieval,
            {
                "answer": "synthesizer_agent",
                "fallback": "small_talk_agent",
            },
        )
        workflow.add_edge("synthesizer_agent", END)
        self.compiled_graph = workflow.compile()

    @staticmethod
    async def classifier_agent(state: RetrieverState) -> dict[str, Any]:
        validated_state = RetrieverState.model_validate(state)
        return await ClassifierAgent(payload=validated_state).execute()

    @staticmethod
    def route_intent(state: RetrieverState) -> str:
        state = RetrieverState.model_validate(state)
        intent = state.intent
        small_talk_intents = {
            "greeting",
            "farewell",
            "rapport_smalltalk",
            "gratitude_ack",
            "apology_ack",
            "offtopic_query",
        }
        return "small_talk" if intent in small_talk_intents else "knowledge"

    @staticmethod
    async def small_talk_agent(state: RetrieverState) -> dict[str, Any]:
        validated_state = RetrieverState.model_validate(state)
        return await SmallTalkAgent(payload=validated_state).execute()

    @staticmethod
    async def query_refiner_agent(state: RetrieverState) -> dict[str, Any]:
        validated_state = RetrieverState.model_validate(state)
        refinement = await QueryRefinerAgent(payload=validated_state).execute()
        # Follow-up answers can depend on private session memory, so never share them.
        if refinement.get("should_refine"):
            return refinement
        cached = await get_cached_answer(
            refinement.get("refined_query") or validated_state.query,
            validated_state.memory_scope_id,
        )
        return {**refinement, **(cached or {})}

    @staticmethod
    def route_cache(state: RetrieverState) -> str:
        state = RetrieverState.model_validate(state)
        return "hit" if state.response and state.retrieval_passed else "miss"

    async def retrieval_agent(self, state: RetrieverState) -> dict[str, Any]:
        validated_state = RetrieverState.model_validate(state)
        return await RetrievalAgent(payload=validated_state, top_k=self.top_k).execute()

    @staticmethod
    def route_retrieval(state: RetrieverState) -> str:
        state = RetrieverState.model_validate(state)
        return "answer" if state.retrieval_passed else "fallback"

    @staticmethod
    async def synthesizer_agent(state: RetrieverState) -> dict[str, Any]:
        validated_state = RetrieverState.model_validate(state)
        result = await SynthesizerAgent(payload=validated_state).execute()
        response = result.get("response")
        if (
            not validated_state.should_refine
            and validated_state.retrieval_passed
            and isinstance(response, str)
            and response.strip()
        ):
            await cache_answer(
                validated_state.refined_query or validated_state.query,
                validated_state.memory_scope_id,
                {
                    **result,
                    "retrieved_chunks": validated_state.retrieved_chunks,
                    "retrieval_score": validated_state.retrieval_score,
                    "retrieval_threshold": validated_state.retrieval_threshold,
                    "retrieval_passed": True,
                },
            )
        return result

    async def ainvoke(self, graph_input: dict[str, Any]) -> dict[str, Any]:
        """Run the compiled graph asynchronously for request-router callers."""
        return await self.compiled_graph.ainvoke(graph_input)
