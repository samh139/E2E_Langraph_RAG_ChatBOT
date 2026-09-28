# app/ingestion/clustering/cluster_summarizer.py

import os
import requests
import torch
from typing import List, Tuple
from sentence_transformers import SentenceTransformer

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://docker.internal")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "gemma3:12b")

SYSTEM_PROMPT = """
You are generating business capability summaries for a banking knowledge system.
Your goal is to describe what customer questions this cluster can fully answer.

Rules:
- Output ONE short summary (1-2 sentences)
- Explicitly mention in summary banking services products fees charges limits or rules if present
- Use generic capability language not specific dates amounts or examples
- The summary should help decide whether this cluster alone can answer a question
- Do NOT mention documents chunks embeddings or clustering
- Do NOT use bullet points
- Do NOT use punctuation
"""

# Hardware engine accelerator resolution (Synchronized with your EmbedWorker parameters)
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Cluster Summarizer Embedding Engine initialized on: {DEVICE.upper()}")
EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-small-en-v1.5", device=DEVICE)

def summarize_cluster(chunk_texts: List[str]) -> Tuple[str, List[float]]:
    """
    Generates an LLM summary string AND its matching 384-dimensional dense vector array.
    Returns: Tuple[summary_text, summary_vector]
    """
    excerpts = "\n\n".join(f"- {text[:500]}" for text in chunk_texts)

    prompt = f"""
{SYSTEM_PROMPT}

Document excerpts:
{excerpts}

Cluster label:
"""

    payload = {
        "model": SUMMARY_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        summary_text = resp.json().get("response", "").strip()
    except Exception as e:
        print(f"[X] Ollama Summary execution failed: {e}. Falling back to default baseline text.")
        summary_text = "Banking service capability summary baseline node cluster entry"

    # Natively compute the 384-dimensional dense vector from the summary string text
    summary_vector = EMBEDDING_MODEL.encode(summary_text, convert_to_numpy=True, normalize_embeddings=True).tolist()

    return summary_text, summary_vector
