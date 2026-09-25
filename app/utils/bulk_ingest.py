import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Set paths for standalone scripts workspace configurations
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Load configurations cleanly using python-dotenv
ENV_PATH = ROOT_DIR / "app" / "config" / "secrets.env"
load_dotenv(dotenv_path=ENV_PATH)

from app.ingestion.services.ingestion_producer import ProductionIngestionCoordinator

def run_bulk_ingest(target_directory: str):
    print("====================================================")
    print(f"🚀 ENTERPRISE INGESTION BATCH RUN STARTED AT: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("====================================================")

    dir_path = Path(target_directory)
    if not dir_path.exists():
        print(f"[X] Staging directory path does not exist: {target_directory}")
        return

    # Scan for supported document patterns matching our extraction handlers
    supported_extensions = {'.pdf', '.docx', '.pptx'}
    files_to_process = [
        str(f) for f in dir_path.iterdir() 
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if not files_to_process:
        print("[*] No processing candidates found inside folder.")
        return

    print(f"[*] Located {len(files_to_process)} target documents for pipeline ingestion.")
    coordinator = ProductionIngestionCoordinator()

    # Production Threading Framework: Runs 4 extractions simultaneously 
    MAX_WORKERS = 4 
    print(f"[*] Spawning Asynchronous Execution Pool [Workers: {MAX_WORKERS}]")
    
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(coordinator.process_single_file, files_to_process)

    duration = time.time() - start_time
    print("====================================================")
    print(f"✨ BATCH INGESTION WORKFLOW COMPLETE. Processing Duration: {duration:.2f}s")
    print("====================================================")

if __name__ == "__main__":
    # Point this to a local testing directory filled with sample PDFs/Docx files
    DATA_STAGING_VAULT = "./data_staging"
    run_bulk_ingest(DATA_STAGING_VAULT)
