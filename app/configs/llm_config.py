# app/configs/llm_config.py
import os
import json
from typing import Dict, Any
from google import genai
from google.genai import types
from app.observability_metrics import dependency_operation, record_model_tokens

# 1. Point to your Service Account JSON file path
# Ensure you set the path to your e2e-langragh...6b38e63.json credential file
# Dynamic path approach
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # points to app/
CREDENTIALS_PATH = os.path.join(BASE_DIR, "configs", "e2e-langragh-chatbot-510511-cb0dd6b38e63.json")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS", 
    CREDENTIALS_PATH
)


# Initialize the official Gemini Client (auto-picks up credentials from the environment variable)
_client = genai.Client(
    vertexai=True,
    project=os.getenv("GOOGLE_CLOUD_PROJECT", "e2e-langragh-chatbot-510511"),
    location=os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1"),
)

# Set your preferred Gemini model (e.g., gemini-2.5-flash for speed/classification)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def _record_gemini_usage(operation: str, response: Any) -> None:
    """Record provider-reported token usage when Gemini returns it."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return
    record_model_tokens(operation, "input", getattr(usage, "prompt_token_count", 0) or 0)
    record_model_tokens(operation, "output", getattr(usage, "candidates_token_count", 0) or 0)


async def close_gemini_client() -> None:
    """Release both transports owned by this module's Google GenAI client.

    Long-running services should keep the shared client open for reuse. One-shot
    commands, such as the DeepEval CLI, should call this during shutdown.
    """
    await _client.aio.aclose()
    _client.close()

def load_intent_taxonomy() -> dict:
    """Load the intent taxonomy file containing examples and tags."""
    file_path = os.path.join(BASE_DIR, "intent_classification.json")
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load intent taxonomy file: {e}")
        return {}

async def ask_gemini_structured(system_prompt: str, user_prompt: str, response_schema: Any) -> Dict[str, Any]:
    """
    Sends a request to Google Gemini forcing a structured JSON output 
    matching the provided Pydantic schema.
    """
    try:
        with dependency_operation("gemini", "structured_generation"):
            response = await _client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.1, # Keep it low for predictable classification
                ),
            )
        _record_gemini_usage("structured_generation", response)
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        # Return fallback values if the API crashes
        return {"intent": "generic_query", "confidence": 0.0, "reasoning": "Fallback due to API error"}

async def fire_fast_modal_request_chat(
    system_prompt: str,
    user_prompt: str,
    model: str = GEMINI_MODEL,
) -> str:
    """
    Send a normal text request to Gemini Flash and return plain text.
    """

    try:
        with dependency_operation("gemini", "answer_generation"):
            response = await _client.aio.models.generate_content(
                model=model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.1,
                ),
            )
        _record_gemini_usage("answer_generation", response)

        return response.text.strip()

    except Exception as e:
        print(f"Gemini API Error: {e}")
        return ""
