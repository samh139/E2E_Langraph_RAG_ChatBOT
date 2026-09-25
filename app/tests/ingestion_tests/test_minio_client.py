import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# 1. Resolve paths cleanly for imports
ROOT_DIR = Path(__file__).resolve().parents[3] # Adjusted to reach workspace root
sys.path.append(str(ROOT_DIR))

# 2. Production Pattern: Explicitly load environment variables before running any code
ENV_PATH = ROOT_DIR / "app" / "configs" / ".env"
load_dotenv(dotenv_path=ENV_PATH)

from app.ingestion.minio_client import MinioStorageClient

def run_integration_test():
    print("=== 🚀 Starting MinIO Storage Integration Test (dotenv) ===")
    
    # Verify environment values were loaded into process memory
    print(f"[*] Target Endpoint Configured: {os.getenv('MINIO_ENDPOINT')}")
    
    try:
        storage_client = MinioStorageClient()
        print(f"[✓] Successfully initialized connection to target bucket: '{storage_client.bucket_name}'")
    except Exception as e:
        print(f"[X] Failed initialization: {e}")
        return

    # [The rest of your test code remains exactly the same...]
    local_test_file = "interview_sample.txt"
    with open(local_test_file, "w", encoding="utf-8") as f:
        f.write("Hello World! This is an enterprise RAG system validation file streaming test.")
    
    try:
        object_key = "test_ingestion/interview_sample.txt"
        uploaded_key = storage_client.upload_document(
            local_file_path=local_test_file, 
            destination_name=object_key
        )
        print(f"[✓] Upload test passed. Object registered under key: {uploaded_key}")

        print("[*] Testing file byte stream verification tracking...")
        stream_buffer = storage_client.get_document_stream(uploaded_key)
        retrieved_content = stream_buffer.getvalue().decode('utf-8')
        
        print(f"[✓] Extracted content from MinIO stream: '{retrieved_content}'")
        print("\n=== ✨ MINIO INTEGRATION TEST SUCCESSFUL! ENGINE OK ===")

    except Exception as e:
        print(f"\n[X] Test failed during storage operations lifecycle: {e}")
    finally:
        if os.path.exists(local_test_file):
            os.remove(local_test_file)
            print("[*] Local workspace test cache cleaned.")

if __name__ == "__main__":
    run_integration_test()
