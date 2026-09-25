import asyncio
import time

from langgraph.graph import END, StateGraph

from app.agents.chunking_agent import ChunkingAgent
from app.agents.docx_extractor_agent import DocxExtractionAgent
from app.agents.file_classifier_agent import FileClassifierAgent
from app.agents.pdf_extractor_agent import PDFExtractorAgent
from app.agents.pptx_extractor_agent import PPTXExtractorAgent
from app.workflow.state import IngestionState


async def _timed(
    agent_cls,
    state: IngestionState,
    stage: str,
) -> IngestionState:
    """
    Execute one agent and record its execution time safely using Pydantic mutations.
    """
    started = time.perf_counter() ## Performance counter for benchmarking

    try:
        agent = agent_cls()

        # Run the agent node cleanly 
        result = await asyncio.to_thread(
            agent.execute,
            state,
        )

        if result is None:
            result = {}

        # ----------------------------------------------------------------
        # FIXED: Create a fresh state copy and update properties safely
        # ----------------------------------------------------------------
        updated_state = state.model_copy(deep=True)
        
        # Apply the returned dictionary patch to our fresh Pydantic state copy
        for key, val in result.items():
            if hasattr(updated_state, key):
                setattr(updated_state, key, val)

        # Update timings safely using the object attributes
        current_timings = dict(updated_state.stage_timings_ms or {})
        current_timings[stage] = round((time.perf_counter() - started) * 1000, 3)
        updated_state.stage_timings_ms = current_timings

        return updated_state

    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        raise RuntimeError(f"{stage} failed after {elapsed_ms} ms: {exc}") from exc


async def run_file_classifier(state: IngestionState) -> IngestionState:
    return await _timed(FileClassifierAgent, state, "classifier")


async def run_pdf_extractor(state: IngestionState) -> IngestionState:
    return await _timed(PDFExtractorAgent, state, "extractor")


async def run_docx_extractor(state: IngestionState) -> IngestionState:
    return await _timed(DocxExtractionAgent, state, "extractor")


async def run_pptx_extractor(state: IngestionState) -> IngestionState:
    return await _timed(PPTXExtractorAgent, state, "extractor")


async def run_chunking(state: IngestionState) -> IngestionState:
    return await _timed(ChunkingAgent, state, "chunker")


def route_to_extractor(state: IngestionState) -> str:
    routes = {
        "pdf": "pdf_extractor",
        "docx": "docx_extractor",
        "pptx": "pptx_extractor",
    }

    # FIXED: Replaced .get("file_type") dictionary lookups with clean object attributes
    file_type = (state.file_type or "").lower().strip().lstrip(".")

    if file_type not in routes:
        raise ValueError(
            f"Unsupported file type: {file_type!r}. "
            f"File extension: {state.file_extension!r}"
        )

    return routes[file_type]


# Initialize Graph Framework
workflow = StateGraph(IngestionState)

workflow.add_node("file_classifier", run_file_classifier)
workflow.add_node("pdf_extractor", run_pdf_extractor)
workflow.add_node("docx_extractor", run_docx_extractor)
workflow.add_node("pptx_extractor", run_pptx_extractor)
workflow.add_node("chunking", run_chunking)

workflow.set_entry_point("file_classifier")

workflow.add_conditional_edges(
    "file_classifier",
    route_to_extractor,
    {
        "pdf_extractor": "pdf_extractor",
        "docx_extractor": "docx_extractor",
        "pptx_extractor": "pptx_extractor",
    },
)

workflow.add_edge("pdf_extractor", "chunking")
workflow.add_edge("docx_extractor", "chunking")
workflow.add_edge("pptx_extractor", "chunking")
workflow.add_edge("chunking", END)

graph = workflow.compile()
