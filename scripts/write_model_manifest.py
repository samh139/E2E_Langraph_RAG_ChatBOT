"""Write deployment-time model identifiers as a release artifact."""

import json
import os
from pathlib import Path


provider = os.getenv("EMBEDDING_PROVIDER", "ollama").strip().lower()
manifest = {
    "release": os.getenv("APP_RELEASE", "local"),
    "app_revision": os.getenv("APP_GIT_SHA", "unknown"),
    "generation": {
        "provider": "vertex-ai",
        "model_id": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        "location": os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
    },
    "embedding": {
        "provider": provider,
        "model_id": (
            os.getenv("VERTEX_EMBEDDING_MODEL", "nomic-embed-text")
            if provider == "vertex"
            else os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        ),
        "model_version": os.getenv("VERTEX_EMBEDDING_VERSION", "unknown"),
        "endpoint_id": os.getenv("VERTEX_EMBEDDING_ENDPOINT_ID", ""),
        "dimensions": int(os.getenv("EMBEDDING_DIMENSIONS", "768")),
    },
}

output = Path(os.getenv("MODEL_MANIFEST_PATH", "build/model-manifest.json"))
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(output)
