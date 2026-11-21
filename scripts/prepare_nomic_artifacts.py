"""Download a pinned Hugging Face model revision for Vertex model artifacts."""

import os
from pathlib import Path

from huggingface_hub import snapshot_download


model_id = os.getenv("NOMIC_HF_MODEL", "nomic-ai/nomic-embed-text-v1.5")
revision = os.getenv("NOMIC_MODEL_REVISION")
if not revision:
    raise SystemExit("Set NOMIC_MODEL_REVISION to an immutable Hugging Face commit SHA.")

destination = Path(os.getenv("NOMIC_ARTIFACT_DIR", "build/nomic-model"))
destination.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=model_id,
    revision=revision,
    local_dir=str(destination),
)
print(f"Downloaded {model_id}@{revision} into {destination}")
