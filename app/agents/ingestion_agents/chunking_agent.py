# app/agents/chunking_agent.py

from app.workflow.ingestion_state import IngestionState
from app.utils.chunking_engine import ChunkingEngine
from app.app_logger import LoggerFactory 

class ChunkingAgent:

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)

    def execute(self, state: IngestionState) -> dict:
        file_name = state.file_name
        file_type = state.file_type or ""
        run_id = state.run_id

        self.logger.info(f"[Chunking Agent] Processing: {file_name} | Run ID: {run_id}")

        # 🎯 HARDCODED SYSTEM DEFAULTS OPTIMIZED FOR YOUR RAG CHATBOT
        MAX_CHARS = 1000
        OVERLAP = 250

        metadata = state.extraction_metadata or {}
        elements = metadata.get("elements", [])

        if elements:
            raw_chunks = ChunkingEngine(max_chars=MAX_CHARS, overlap=OVERLAP).chunk(
                elements=elements,
                file_name=file_name
            )
        else:
            raise ValueError(f"[Chunking Agent] No elements found in metadata for file: {file_name}.")

        self.logger.info(f"[Chunking Agent] Created {len(raw_chunks)} raw chunks.")

        # ------------------------------------------------------------------
        # FIXED METADATA SCHEMA CONTRACT
        # Strips layout variables (pages, headers) to maximize RAG search performance
        # ------------------------------------------------------------------
        standardized_chunks = []
        for idx, chunk_item in enumerate(raw_chunks):
            if isinstance(chunk_item, dict):
                content_str = chunk_item.get("content", chunk_item.get("text_content", ""))
                raw_chunk_id = chunk_item.get("id", chunk_item.get("chunk_id"))
            else:
                content_str = getattr(chunk_item, "content", getattr(chunk_item, "text_content", str(chunk_item)))
                raw_chunk_id = getattr(chunk_item, "id", getattr(chunk_item, "chunk_id", None))

            clean_chunk = {
                "chunk_id": raw_chunk_id or f"{run_id}_c{idx}",
                "doc_id": run_id,
                "file_name": file_name,
                "file_type": file_type.lower().replace(".", "").strip(),
                "content": content_str.strip()
            }
            standardized_chunks.append(clean_chunk)

        # try:
        #     StorageService.save_chunks(standardized_chunks, run_id=run_id)
        # except Exception as storage_err:
        #     self.logger.error(f"[Chunking Agent Storage Error] Failed to write chunks: {storage_err}")

        return {
            "chunks": standardized_chunks,
            "status": "chunked",
            "error": None
        }
