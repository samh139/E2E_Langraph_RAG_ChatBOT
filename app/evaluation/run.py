"""Run DeepEval RAG metrics for one query without affecting chat traffic.

Usage from the project directory:
    python -m app.evaluation.run --query "What are SBI NEFT charges?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _load_environment() -> None:
    """Load the same local config file used by the application, if present."""
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / "app" / "configs" / ".env", override=False)


def _make_judge(model_name: str):
    """Create a Gemini judge using this project's Vertex AI configuration."""
    from deepeval.models import GeminiModel

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    judge_options: dict[str, Any] = {
        "model": model_name,
        "project": os.getenv("GOOGLE_CLOUD_PROJECT", "e2e-langragh-chatbot-510511"),
        "location": os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        "use_vertexai": True,
        "temperature": 0.0,
    }
    if credentials_path:
        try:
            credentials_json = Path(credentials_path).read_text(encoding="utf-8")
            json.loads(credentials_json)  # Validate without ever logging the secret.
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "GOOGLE_APPLICATION_CREDENTIALS must point to a readable service-account JSON file."
            ) from exc
        # DeepEval's Gemini Vertex integration expects the JSON contents here,
        # even though GOOGLE_APPLICATION_CREDENTIALS itself is a file path.
        judge_options["service_account_key"] = credentials_json
    return GeminiModel(**judge_options)


async def _evaluate_query(
    query: str,
    session_id: str,
    user_id: str,
    judge_model: str,
) -> dict[str, Any]:
    # Import after loading .env because llm_config initializes its Gemini client
    # and resolves GOOGLE_APPLICATION_CREDENTIALS during import.
    from app.workflow.retrieval_state import RetrieverState
    from app.workflow.retrieval_workflow import RetrievalWorkflow

    state = RetrieverState(
        session_id=session_id,
        query_id=str(uuid.uuid4()),
        user_id=user_id,
        query=query,
    )
    result = await RetrievalWorkflow().ainvoke(state.model_dump())

    retrieved_chunks = result.get("retrieved_chunks") or []
    retrieval_context = [
        content
        for chunk in retrieved_chunks
        if isinstance((content := chunk.get("content")), str) and content.strip()
    ]
    actual_output = result.get("response") or ""
    refined_query = result.get("refined_query") or query

    if not actual_output.strip():
        raise RuntimeError("The retrieval workflow returned an empty response.")
    if not retrieval_context:
        raise RuntimeError(
            "This query produced no retrieved chunk content, so RAG metrics cannot "
            "be calculated. Check the retrieval result and score threshold first."
        )

    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    judge = _make_judge(judge_model)
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
    )
    metrics = [
        FaithfulnessMetric(threshold=None, model=judge),
        AnswerRelevancyMetric(threshold=None, model=judge),
        ContextualRelevancyMetric(threshold=None, model=judge),
    ]

    metric_results = []
    for metric in metrics:
        metric.measure(test_case)
        metric_results.append(
            {
                "name": metric.__class__.__name__,
                "score": metric.score,
                "reason": metric.reason,
            }
        )

    return {
        "query_id": result.get("query_id", state.query_id),
        "query": query,
        "refined_query": refined_query,
        "response": actual_output,
        "retrieved_chunk_count": len(retrieval_context),
        "metrics": metric_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run DeepEval RAG metrics once for a query; this is not part of chat serving."
    )
    parser.add_argument("--query", required=True, help="Question to run through retrieval")
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session ID. Defaults to a new isolated evaluation session.",
    )
    parser.add_argument(
        "--user-id",
        default="deepeval-evaluation",
        help="User ID used for memory lookup; defaults to an isolated evaluation user.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Vertex Gemini judge model (defaults to GEMINI_MODEL or gemini-2.5-flash).",
    )
    args = parser.parse_args()
    _load_environment()

    # Ensure llm_config sets its application credential fallback before we
    # initialize DeepEval's separate Gemini judge client.
    from app.configs.llm_config import GEMINI_MODEL  # noqa: F401

    judge_model = args.model or os.getenv("DEEPEVAL_MODEL") or GEMINI_MODEL
    session_id = args.session_id or f"deepeval-{uuid.uuid4()}"
    try:
        result = asyncio.run(
            _evaluate_query(
                query=args.query,
                session_id=session_id,
                user_id=args.user_id,
                judge_model=judge_model,
            )
        )
    except Exception as exc:
        parser.exit(1, f"DeepEval run failed: {exc}\n")

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
