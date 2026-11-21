import os
from io import BytesIO
from pathlib import Path
from google.cloud import storage


class MinioStorageClient:
    def __init__(self):
        self.client = storage.Client()
        self.bucket_name = os.getenv("MINIO_BUCKET", "e2e-langragh-chatbot-510511-rag-docs")
        self.bucket = self.client.bucket(self.bucket_name)

    def upload_document(self, local_file_path: str, destination_name: str = None) -> str:
        """
        Uploads a raw document into Google Cloud Storage.
        Returns the destination object key.
        """
        path = Path(local_file_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found at {local_file_path}")

        obj_name = destination_name or path.name

        try:
            print(f"[*] Uploading {path.name} to GCS bucket '{self.bucket_name}'...")
            blob = self.bucket.blob(obj_name)
            blob.upload_from_filename(str(path))
            print(f"[✓] Successfully stored '{obj_name}' in GCS.")
            return obj_name
        except Exception as e:
            print(f"[X] Failed to upload to GCS: {e}")
            raise e

    def get_document_stream(self, object_name: str) -> BytesIO:
        """Streams document directly into a memory buffer."""
        try:
            blob = self.bucket.blob(object_name)
            return BytesIO(blob.download_as_bytes())
        except Exception as e:
            print(f"[X] Error reading stream for '{object_name}': {e}")
            raise e

    def download_to_local_temp(self, object_name: str, temp_dir: str = "/tmp/rag_ingest") -> str:
        """
        Downloads the file out of GCS to a local temporary path for extractors.
        """
        os.makedirs(temp_dir, exist_ok=True)
        local_dest_path = os.path.join(temp_dir, object_name)
        os.makedirs(os.path.dirname(local_dest_path), exist_ok=True)

        try:
            blob = self.bucket.blob(object_name)
            blob.download_to_filename(local_dest_path)
            return local_dest_path
        except Exception as e:
            print(f"[X] Failed downloading local copy for processing: {e}")
            raise e
