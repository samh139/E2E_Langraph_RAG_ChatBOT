"""Run an explicit offline DeepEval benchmark and save results to MongoDB.

Input can be plain text (one question per line) or JSONL with input,
expected_output, and optional source_documents fields.
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
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / "app" / "configs" / ".env", override=False)


def _make_judge(model_name: str):
    from deepeval.models import GeminiModel

    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    judge_options: dict[str, Any] = {
        "model": model_name,
        "project": os.getenv("GOOGLE_CLOUD_PROJECT", "e2e-langragh-chatbot-510511"),
        "location": os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
        "use_vertexai": True,
        "temperature": 0.1,
    }
    if credentials_path:
        try:
            credentials_json = Path(credentials_path).read_text(encoding="utf-8")
            json.loads(credentials_json)
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "GOOGLE_APPLICATION_CREDENTIALS must point to a readable service-account JSON file."
            ) from exc
        # DeepEval needs the JSON contents, not the GOOGLE_APPLICATION_CREDENTIALS path.
        judge_options["service_account_key"] = credentials_json
    return GeminiModel(**judge_options)


async def _run_evaluation(
    query: str,
    expected_output: str | None,
    source_documents: list[str],
    query_id: str,
    session_id: str,
    user_id: str,
    judge_model: str,
) -> dict[str, Any]:
    from app.workflow.retrieval_state import RetrieverState
    from app.workflow.retrieval_workflow import RetrievalWorkflow

    state = RetrieverState(
        session_id=session_id,
        query_id=query_id,
        user_id=user_id,
        query=query,
    )
    result = await RetrievalWorkflow().ainvoke(state.model_dump())

    retrieved_chunks = result.get("retrieved_chunks") or []
    retrieved_sources = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "file_name": chunk.get("file_name"),
            "content": content,
        }
        for chunk in retrieved_chunks
        if isinstance((content := chunk.get("content")), str) and content.strip()
    ]
    retrieval_context = [source["content"] for source in retrieved_sources]
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
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.test_case import LLMTestCase, SingleTurnParams

    judge = _make_judge(judge_model)
    test_case = LLMTestCase(
        input=query,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=expected_output or None,
    )
    metrics = [
        ("Faithfulness", FaithfulnessMetric(threshold=None, model=judge, async_mode=False)),
        ("Answer Relevancy", AnswerRelevancyMetric(threshold=None, model=judge, async_mode=False)),
        ("Contextual Relevancy", ContextualRelevancyMetric(threshold=None, model=judge, async_mode=False)),
    ]
    has_reference = bool(expected_output and expected_output.strip())
    if has_reference:
        metrics.extend(
            [
                (
                    "Contextual Precision",
                    ContextualPrecisionMetric(
                        threshold=None, model=judge, async_mode=False
                    ),
                ),
                (
                    "Contextual Recall",
                    ContextualRecallMetric(
                        threshold=None, model=judge, async_mode=False
                    ),
                ),
                (
                    "Answer Correctness [GEval]",
                    GEval(
                        name="Answer Correctness",
                        criteria=(
                            "Assess whether the actual answer correctly and completely "
                            "answers the input question compared with the expected answer. "
                            "Accept semantically equivalent wording. Reduce the score for "
                            "incorrect or contradictory facts, or missing material facts. "
                            "Do not require an exact wording match, and do not penalize "
                            "accurate additional details solely because they are absent "
                            "from the expected answer."
                        ),
                        evaluation_params=[
                            SingleTurnParams.INPUT,
                            SingleTurnParams.ACTUAL_OUTPUT,
                            SingleTurnParams.EXPECTED_OUTPUT,
                        ],
                        threshold=None,
                        model=judge,
                        async_mode=False,
                    ),
                ),
            ]
        )

    metric_results = []
    for metric_name, metric in metrics:
        metric.measure(test_case)
        metric_results.append(
            {
                "name": metric_name,
                "status": "completed",
                "score": metric.score,
                "reason": metric.reason,
            }
        )
    if not has_reference:
        metric_results.extend(
            {
                "name": metric_name,
                "status": "skipped",
                "score": None,
                "reason": "This metric requires a non-empty expected_output in the golden case.",
            }
            for metric_name in (
                "Contextual Precision",
                "Contextual Recall",
                "Answer Correctness [GEval]",
            )
        )

    return {
        "query_id": query_id,
        "query": query,
        "expected_output": expected_output,
        "source_documents": source_documents,
        "refined_query": refined_query,
        "response": actual_output,
        "retrieved_chunk_count": len(retrieval_context),
        "retrieved_chunks": retrieved_sources,
        "intent": result.get("intent"),
        "retrieval_score": result.get("retrieval_score"),
        "retrieval_threshold": result.get("retrieval_threshold"),
        "retrieval_passed": result.get("retrieval_passed"),
        "metrics": metric_results,
    }


async def _evaluate_queries(
    cases: list[dict[str, Any]],
    run_id: str,
    user_id: str,
    judge_model: str,
    result_store,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    from app.configs.llm_config import close_gemini_client

    results: list[dict[str, Any]] = []
    try:
        for case_number, case in enumerate(cases, start=1):
            query = case["input"]
            query_id = str(uuid.uuid4())
            try:
                result = await _run_evaluation(
                    query=query,
                    expected_output=case.get("expected_output"),
                    source_documents=case.get("source_documents", []),
                    query_id=query_id,
                    session_id=session_id or f"deepeval-{run_id}-{case_number}",
                    user_id=user_id,
                    judge_model=judge_model,
                )
                record = {
                    **result,
                    "run_id": run_id,
                    "case_number": case_number,
                    "golden_case_id": case.get("id"),
                    "source_url": case.get("source_url"),
                    "status": "completed",
                    "source": "offline",
                    "judge_model": judge_model,
                }
            except Exception as exc:
                record = {
                    "run_id": run_id,
                    "query_id": query_id,
                    "case_number": case_number,
                    "query": query,
                    "golden_case_id": case.get("id"),
                    "expected_output": case.get("expected_output"),
                    "source_documents": case.get("source_documents", []),
                    "source_url": case.get("source_url"),
                    "status": "failed",
                    "source": "offline",
                    "judge_model": judge_model,
                    "error": str(exc),
                }
                result_store.save(record)
                raise

            result_store.save(record)
            results.append(record)
            print(
                f"Saved benchmark case {case_number}/{len(cases)} "
                f"(query_id={query_id}) to MongoDB.",
                flush=True,
            )
        return results
    finally:
        # This CLI owns the app Gemini client's lifecycle; the long-running chat
        # service continues to reuse its client without closing it per request.
        await close_gemini_client()


def _load_queries(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[dict[str, Any]]:
    if args.query is not None:
        cases = [{"id": "single-query", "input": args.query.strip()}]
    else:
        try:
            contents = args.queries_file.read_text(encoding="utf-8")
        except OSError as exc:
            parser.error(f"Could not read query file: {exc}")
        if args.queries_file.suffix.lower() == ".jsonl":
            cases = []
            for line_number, line in enumerate(contents.splitlines(), start=1):
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                try:
                    case = json.loads(line)
                except json.JSONDecodeError as exc:
                    parser.error(f"Invalid JSONL at line {line_number}: {exc}")
                if not isinstance(case, dict) or not isinstance(case.get("input"), str):
                    parser.error(
                        f"JSONL line {line_number} must be an object with a string 'input'."
                    )
                case["input"] = case["input"].strip()
                if not case["input"]:
                    parser.error(f"JSONL line {line_number} has an empty 'input'.")
                expected_output = case.get("expected_output", "")
                if not isinstance(expected_output, str):
                    parser.error(
                        f"JSONL line {line_number} 'expected_output' must be a string."
                    )
                source_documents = case.get("source_documents", [])
                if not isinstance(source_documents, list) or not all(
                    isinstance(source, str) for source in source_documents
                ):
                    parser.error(
                        f"JSONL line {line_number} 'source_documents' must be a list of strings."
                    )
                cases.append(case)
        else:
            cases = [
                {"id": f"case-{index}", "input": line.strip()}
                for index, line in enumerate(contents.splitlines(), start=1)
                if line.strip() and not line.lstrip().startswith("#")
            ]
    if not cases:
        parser.error("Provide at least one non-empty query.")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run offline DeepEval RAG metrics and save them to MongoDB."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--query", help="Run one question")
    inputs.add_argument(
        "--queries-file",
        type=Path,
        help="Text file (one question per line) or JSONL file with input and optional reference fields",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Optional session ID for memory lookup, primarily for a single query.",
    )
    parser.add_argument(
        "--user-id",
        default=None,
        help="User ID for memory lookup; defaults to an isolated ID for this run.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Vertex Gemini judge model (defaults to GEMINI_MODEL or gemini-2.5-flash).",
    )
    args = parser.parse_args()
    _load_environment()
    queries = _load_queries(args, parser)

    from app.evaluation.mongo_store import EvaluationResultStore

    result_store = None
    try:
        result_store = EvaluationResultStore(
            uri=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            database=os.getenv("MONGO_DATABASE", "rag_evaluation"),
            collection=os.getenv("MONGO_COLLECTION", "deepeval_results"),
        )

        # Importing this module initializes the app's Vertex client and applies
        # its GOOGLE_APPLICATION_CREDENTIALS fallback before model selection.
        from app.configs.llm_config import GEMINI_MODEL

        judge_model = args.model or os.getenv("DEEPEVAL_MODEL") or GEMINI_MODEL
        run_id = str(uuid.uuid4())
        user_id = args.user_id or f"deepeval-{run_id}"
        results = asyncio.run(
            _evaluate_queries(
                cases=queries,
                run_id=run_id,
                user_id=user_id,
                judge_model=judge_model,
                result_store=result_store,
                session_id=args.session_id,
            )
        )
    except Exception as exc:
        parser.exit(1, f"DeepEval run failed: {exc}\n")
    finally:
        if result_store is not None:
            result_store.close()

    #print(json.dumps({"run_id": run_id, "results": results}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
