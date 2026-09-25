import asyncio
import time
from langgraph.graph import END, StateGraph, CompiledGraph
from app.agents.chunking_agent import ChunkingAgent
from app.agents.docx_extractor_agent import DocxExtractionAgent
from app.agents.file_classifier_agent import FileClassifierAgent
from app.agents.pdf_extractor_agent import PDFExtractorAgent
from app.agents.pptx_extractor_agent import PPTXExtractorAgent
from app.workflow.state import IngestionState

class IngestionWorkflow:
    def __init__(self):
        self.workflow = StateGraph(IngestionState)
        self._build_graph()
        self.compiled_graph = self.workflow.compile()

    async def _timed(self, agent_cls, state: IngestionState, stage: str) -> IngestionState:
        started = time.perf_counter()
        try:
            agent = agent_cls()
            result = await asyncio.to_thread(agent.execute, state)
            if result is None:
                result = {}

            updated_state = state.model_copy(deep=True)
            for key, val in result.items():
                if hasattr(updated_state, key):
                    setattr(updated_state, key, val)

            current_timings = dict(updated_state.stage_timings_ms or {})
            current_timings[stage] = round((time.perf_counter() - started) * 1000, 3)
            updated_state.stage_timings_ms = current_timings
            return updated_state
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            raise RuntimeError(f"{stage} failed after {elapsed_ms} ms: {exc}") from exc

    async def run_file_classifier(self, state: IngestionState) -> IngestionState:
        return await self._timed(FileClassifierAgent, state, "classifier")

    async def run_pdf_extractor(self, state: IngestionState) -> IngestionState:
        return await self._timed(PDFExtractorAgent, state, "extractor")

    async def run_docx_extractor(self, state: IngestionState) -> IngestionState:
        return await self._timed(DocxExtractionAgent, state, "extractor")

    async def run_pptx_extractor(self, state: IngestionState) -> IngestionState:
        return await self._timed(PPTXExtractorAgent, state, "extractor")

    async def run_chunking(self, state: IngestionState) -> IngestionState:
        return await self._timed(ChunkingAgent, state, "chunker")

    def route_to_extractor(self, state: IngestionState) -> str:
        routes = {
            "pdf": "pdf_extractor",
            "docx": "docx_extractor",
            "pptx": "pptx_extractor",
        }
        file_type = (state.file_type or "").lower().strip().lstrip(".")
        if file_type not in routes:
            raise ValueError(f"Unsupported file type: {file_type!r}. File extension: {state.file_extension!r}")
        return routes[file_type]

    def _build_graph(self):
        self.workflow.add_node("file_classifier", self.run_file_classifier)
        self.workflow.add_node("pdf_extractor", self.run_pdf_extractor)
        self.workflow.add_node("docx_extractor", self.run_docx_extractor)
        self.workflow.add_node("pptx_extractor", self.run_pptx_extractor)
        self.workflow.add_node("chunking", self.run_chunking)

        self.workflow.set_entry_point("file_classifier")

        self.workflow.add_conditional_edges(
            "file_classifier",
            self.route_to_extractor,
            {
                "pdf_extractor": "pdf_extractor",
                "docx_extractor": "docx_extractor",
                "pptx_extractor": "pptx_extractor",
            },
        )

        self.workflow.add_edge("pdf_extractor", "chunking")
        self.workflow.add_edge("docx_extractor", "chunking")
        self.workflow.add_edge("pptx_extractor", "chunking")
        self.workflow.add_edge("chunking", END)

    def compile(self) -> CompiledGraph:
        return self.compiled_graph

# Exported compiled graph instance instance to match expected imports
graph = IngestionWorkflow().compile()
