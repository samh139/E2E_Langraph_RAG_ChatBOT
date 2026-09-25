from uuid import uuid4

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class IngestionState(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    file_path: str
    file_name: str
    file_extension: str
    mime_type: Optional[str] = None

    file_type: str = ""

    extracted_text: str = ""
    extraction_metadata: Dict[str, Any] = Field(default_factory=dict)

    status: str = "uploaded"
    error: Optional[str] = None

    language: Optional[str] = None
    normalized_content: Optional[str] = None
    chunks: Optional[List[Dict[str, Any]]] = None

    # Benchmark/experiment configuration and measurements.
    chunking_config: Dict[str, Any] = Field(default_factory=dict)
    stage_timings_ms: Dict[str, float] = Field(default_factory=dict)
