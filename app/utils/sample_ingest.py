# scripts/sample_ingest.py
import sys
from app.ingestion.services.chunks_store_service import Job, ingest_file

if __name__ == "__main__":
    path = sys.argv[1]
    owner = sys.argv[2] if len(sys.argv) > 2 else "sameer@example.com"
    job = Job(file_path=path, owner=owner)
    n = ingest_file(job)
    print(f"Published {n} chunks from {path}")
