from app.workflow.state import IngestionState
from app.utils.chunking_engine import ChunkingEngine
from app.utils.chunk_output_writer import ChunkOutputWriter
from app.storage.storage_service import StorageService 
from app.app_logger import LoggerFactory 

class ChunkingAgent:

    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)

    # FIXED: Return type hint changed to dict to follow the LangGraph patch pattern contract
    def execute(self, state: IngestionState) -> dict:
        file_name = state.file_name
        file_type = state.file_type
        run_id = state.run_id or StorageService.generate_run_id()

        self.logger.info(f"[Chunking Agent] Processing: {file_name} | Run ID: {run_id}")

        # --- Read values dynamically from your workflow configuration ---
        config = state.chunking_config or {}
        max_chars = config.get("max_chars", 2500)
        overlap = config.get("overlap", 250)

        self.logger.info(f"[Chunking Agent] Configuration applied -> max_chars: {max_chars}, overlap: {overlap}")

        # Safely capture metadata fields from our workflow state
        metadata = state.extraction_metadata or {}
        elements = metadata.get("elements", [])

        # BLOCK 1: Preferred Processing Path (Rich Layout Items)
        if elements:
            chunks = ChunkingEngine(max_chars=max_chars, overlap=overlap).chunk(
                elements=elements,
                file_name=file_name
            )

        # BLOCK 2: Fallback Processing Path (Flat String Alternative)
        else:
            extracted_text = state.extracted_text or ""

            # Stop immediately if there's no data available to chunk
            if not extracted_text.strip():
                # FIXED: Return a clean patch update dictionary if chunking fails
                return {
                    "chunks": [],
                    "status": "chunking_failed",
                    "error": "No extracted content available for chunking."
                }

            # Map the flat string to a standard paragraph dictionary to reuse engine logic
            fallback_elements = [{
                "text": extracted_text,
                "element_type": "paragraph",
                "page": None,
            }]

            chunks = ChunkingEngine(max_chars=max_chars, overlap=overlap).chunk(
                elements=fallback_elements,
                file_name=file_name
            )

        # BLOCK 3: File Output and Serialization IO
        writer = ChunkOutputWriter()
        output_path = writer.write(
            file_name=file_name,
            file_type=file_type,
            chunks=chunks,
        )

        self.logger.info(f"[Chunking Agent] Created {len(chunks)} chunks")
        self.logger.info(f"[Chunking Agent] Saved chunks to file system at: {output_path}")

        # --------------------------------------------------------
        # INTEGRATION BLOCK: PERSIST CHUNKS TO MONGODB
        # --------------------------------------------------------
        try:
            # Pass chunks list and run_id directly into the StorageService
            mongo_ids = StorageService.save_chunks(chunks, run_id=run_id)
            self.logger.info(f"[Chunking Agent] Successfully saved {len(chunks)} chunks to MongoDB.")
        except Exception as storage_err:
            self.logger.error(f"[Chunking Agent Storage Error] Failed to write chunks to local database: {storage_err}")

        # CORRECT: Returns a clean patch dictionary for LangGraph to merge into state automatically
        return {
            "chunks": chunks,
            "status": "chunked",
            "error": None
        }
