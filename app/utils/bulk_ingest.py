import os
import sys
import time
from pathlib import Path
import asyncio
from dotenv import load_dotenv

# Set paths for standalone scripts workspace configurations
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

# Load configurations cleanly using python-dotenv
ENV_PATH = ROOT_DIR / "app" / "config" / "secrets.env"
load_dotenv(dotenv_path=ENV_PATH)

from app.services.ingestion_service import IngestionService
from app.utils.file_utils import get_file_type

def run_bulk_ingest(target_directory: str):
    print("====================================================")
    print(f"🚀 ENTERPRISE INGESTION BATCH RUN STARTED AT: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("====================================================")

    dir_path = Path(target_directory)
    if not dir_path.exists():
        print(f"[X] Staging directory path does not exist: {target_directory}")
        return

    files_to_process = []
    for path in dir_path.iterdir():
        if not path.is_file():
            continue
        try:
            get_file_type(str(path))
            files_to_process.append(str(path))
        except ValueError:
            continue

    if not files_to_process:
        print("[*] No processing candidates found inside folder.")
        return

    print(f"[*] Located {len(files_to_process)} target documents for pipeline ingestion.")
    start_time = time.time()
    service = IngestionService()
    processed = asyncio.run(service.bulk_ingest(target_directory))

    duration = time.time() - start_time
    print("====================================================")
    print(f"✨ BATCH INGESTION COMPLETE. Processed {processed} documents in {duration:.2f}s")
    print("====================================================")

if __name__ == "__main__":
    # Point this to a local testing directory filled with sample PDFs/Docx files
    DATA_STAGING_VAULT = "./data_staging"
    run_bulk_ingest(DATA_STAGING_VAULT)
