import os
from io import BytesIO
from pathlib import Path
from minio import Minio

class MinioStorageClient:
    def __init__(self):
        self.client = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT", "localhost:9010"),
            access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
            secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
            secure=False
        )
        self.bucket_name = os.getenv("MINIO_BUCKET", "rag-raw-documents")
        
        # Ensure our bucket exists before performing operations
        try:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)
        except Exception as e:
            print(f"[-] MinIO initialization bucket check warning: {e}")

    def upload_document(self, local_file_path: str, destination_name: str = None) -> str:
        """
        Uploads a raw document from the host machine/worker into MinIO storage.
        Returns the destination name (object key).
        """
        path = Path(local_file_path)
        if not path.exists():
            raise FileNotFoundError(f"Local file not found at {local_file_path}")
            
        obj_name = destination_name or path.name
        
        try:
            print(f"[*] Uploading {path.name} to MinIO bucket '{self.bucket_name}'...")
            self.client.fput_object(
                bucket_name=self.bucket_name,
                object_name=obj_name,
                file_path=str(path)
            )
            print(f"[✓] Successfully stored '{obj_name}' in MinIO.")
            return obj_name
        except Exception as e:
            print(f"[X] Failed to upload to MinIO: {e}")
            raise e

    def get_document_stream(self, object_name: str) -> BytesIO:
        """Streams document directly into a memory buffer."""
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            return BytesIO(response.read())
        except Exception as e:
            print(f"[X] Error reading stream for '{object_name}': {e}")
            raise e
        finally:
            if 'response' in locals():
                response.close()
                response.release_conn()

    def download_to_local_temp(self, object_name: str, temp_dir: str = "/tmp/rag_ingest") -> str:
        """
        Downloads the file out of MinIO and saves it to a local temporary path.
        This provides the string file path your extractors need to process data.
        """
        os.makedirs(temp_dir, exist_ok=True)
        local_dest_path = os.path.join(temp_dir, object_name)
        
        try:
            self.client.fget_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                file_path=local_dest_path
            )
            return local_dest_path
        except Exception as e:
            print(f"[X] Failed downloading local copy for processing: {e}")
            raise e
