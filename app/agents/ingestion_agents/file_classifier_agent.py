from pathlib import Path
from app.workflow.state import IngestionState
from app.utils.file_utils import get_file_type, get_extension
from app.app_logger import LoggerFactory

class FileClassifierAgent:
    def __init__(self):
        self.logger = LoggerFactory.get_logger(__name__)

    # FIXED: Changed return type hint to dict to match the dictionary patch layout
    def execute(self, state: IngestionState) -> dict:
        file_path = state.file_path

        extension = get_extension(file_path)
        file_type = get_file_type(file_path)

        self.logger.info(
            f"[File Classifier] "
            f"file={Path(file_path).name} "
            f"type={file_type}"
        )

        # Correct: Returns a clean state patch dictionary for LangGraph to merge automatically
        return {
            "file_extension": extension,
            "file_type": file_type,
            "status": "classified"
        }
