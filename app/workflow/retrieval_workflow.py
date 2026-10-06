"""LangGraph workflow for routing chat requests through retrieval.

The graph keeps integrations injectable: an application can supply a retrieval
callable and memory loader, while the default retrieval node queries the
project's Elasticsearch chunk index using a lexical ``match`` query.
"""

import asyncio
import os
import re
import time
from collections.abc import Callable, Mapping
from typing import Any

from langgraph.graph import END, StateGraph

from app.workflow.retrieval_state import RetrieverState
# ✅ Import your new Gemini-powered classification agent code here
from app.agents.retrieval_agents.classifier_agent import ClassifierAgent 
from app.agents.retrieval_agents.greeting_agent import GreetingAgent
from app.agents.retrieval_agents.query_refiner_agent import QueryRefinerAgent

class RetrievalWorkflow:
    """Build and run the classifier → refine → knowledge → retrieval graph."""

    def __init__(
        self,
        top_k: int = 5,
    ) -> None:
        self.top_k = top_k
        workflow = StateGraph(RetrieverState)

        # 1. Maintain your primary agent node bindings
        workflow.add_node("classifier_agent", self.classifier_agent)
        workflow.add_node("greeting", self.greeting_agent)
        workflow.add_node("query_refiner", self.query_refiner_agent)
        workflow.add_node("knowledge_agent", self.knowledge_agent)
        workflow.add_node("retrieval", self.retrieval_agent)

        workflow.set_entry_point("classifier_agent")
        
        # 2. ✅ UPDATED: The routing map directs the 12 granular intents cleanly
        # into your two primary downstream execution branches
        workflow.add_conditional_edges(
            "classifier_agent",
            self.route_intent,
            {
                "small_talk_stream": "greeting", 
                "knowledge_stream": "query_refiner"
            },
        )
        workflow.add_edge("greeting", END)
        workflow.add_edge("query_refiner", "knowledge_agent")
        workflow.add_edge("knowledge_agent", "retrieval")
        workflow.add_edge("retrieval", END)
        self.compiled_graph = workflow.compile()

    @staticmethod
    async def classifier_agent(state: RetrieverState) -> dict[str, Any]:
        """✅ UPDATED: Invokes the Gemini ClassifierAgent to populate the state patch dynamically."""
        agent = ClassifierAgent(payload=state)
        return await agent.execute()

    @staticmethod
    def route_intent(state: RetrieverState) -> str:
        """✅ UPDATED: Groups the 12 semantic taxonomy intents into high-level graph streams."""
        small_talk_intents = {
            "greeting", 
            "farewell", 
            "rapport_smalltalk", 
            "gratitude_ack", 
            "apology_ack", 
            "offtopic_query"
        }
        
        knowledge_intents = {
            "lookup_knowledge", 
            "lookup_knowledge_plus_reasoning", 
            "get_file_template", 
            "task_execute",      # Map execution to refiner/knowledge stream for RAG injection
            "customer_insight",  # Map insight queries to refiner/knowledge stream for context building
            "generic_query"
        }
        
        if state.intent in small_talk_intents:
            return "small_talk_stream"
            
        return "knowledge_stream"

    # Ensure your remaining structural node stubs stay declared below...
    async def greeting_agent(self, state: RetrieverState) -> dict[str, Any]:
        agent = GreetingAgent(payload=state)
        return await agent.execute()

    async def query_refiner_agent(self, state: RetrieverState) -> dict[str, Any]:
        """
        Pure orchestration stub. LangGraph triggers this, 
        the agent runs its logic, and returns the patch state.
        """
        agent = QueryRefinerAgent(payload=state)
        return await agent.execute() 

    async def knowledge_agent(self, state: RetrieverState) -> dict[str, Any]:
        return {}

    async def retrieval_agent(self, state: RetrieverState) -> dict[str, Any]:
        return {"retrieved_chunks": []}

    async def ainvoke(self, graph_input: dict[str, Any]) -> dict[str, Any]:
        """Run the compiled graph asynchronously for request-router callers."""
        return await self.compiled_graph.ainvoke(graph_input)
