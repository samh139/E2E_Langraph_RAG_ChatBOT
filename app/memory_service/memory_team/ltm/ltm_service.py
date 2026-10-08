# store_ltm.py
from datetime import datetime
import os
from pathlib import Path

from elasticsearch import AsyncElasticsearch 
import ollama
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / "configs" / ".env")

es_host="http://elasticsearch:9200"
es_ltm_index="conversations"
es = AsyncElasticsearch(es_host)

EMBEDDING_MODEL_NAME = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBEDDING_DIMS = 768
OLLAMA_URL = os.getenv(
    "OLLAMA_URL", os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
)
EMBEDDING_CLIENT = ollama.AsyncClient(
    host=OLLAMA_URL,
    timeout=float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")),
)
print(
    f"[*] LTM embedding client configured: {EMBEDDING_MODEL_NAME} "
    f"({EMBEDDING_DIMS} dimensions) at {OLLAMA_URL}"
)


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts with the same Ollama model used for indexed documents."""
    if not texts:
        return []

    response = await EMBEDDING_CLIENT.embed(model=EMBEDDING_MODEL_NAME, input=texts)
    embeddings = [list(vector) for vector in response.embeddings]
    if len(embeddings) != len(texts):
        raise RuntimeError(
            f"Ollama returned {len(embeddings)} vectors for {len(texts)} input texts"
        )
    for index, vector in enumerate(embeddings):
        if len(vector) != EMBEDDING_DIMS:
            raise RuntimeError(
                f"Ollama model {EMBEDDING_MODEL_NAME!r} returned vector {index} "
                f"with {len(vector)} dimensions; expected {EMBEDDING_DIMS}"
            )
    return embeddings

async def create_index_if_not_exists() -> None:
    """Create the LTM conversation index with fields used by lexical and KNN retrieval."""
    if await es.indices.exists(index=es_ltm_index):
        return

    mappings = {
        "properties": {
            "user_id": {"type": "keyword"},
            "session_id": {"type": "keyword"},
            "user_message": {"type": "text"},
            "bot_response": {"type": "text"},
            "user_embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMS,
                "index": False,
            },
            "bot_embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMS,
                "index": False,
            },
            "combined_embedding": {
                "type": "dense_vector",
                "dims": EMBEDDING_DIMS,
                "index": True,
                "similarity": "cosine",
            },
            "timestamp": {"type": "date"},
        }
    }

    try:
        await es.indices.create(index=es_ltm_index, mappings=mappings)
        print(f"Created Elasticsearch index '{es_ltm_index}'.")
    except Exception:
        # Another request may have created it after the exists check.
        if await es.indices.exists(index=es_ltm_index):
            return
        raise



def normalize_scores(items: List[Dict], score_key: str) -> None:
    """
    Normalize scores in-place between 0 and 1 for the given key.
    """
    if not items:
        return
    
    scores = [item.get(score_key, 0) for item in items]
    min_score = min(scores)
    max_score = max(scores)
    
    # Avoid division by zero
    if max_score == min_score:
        for item in items:
            item[score_key] = 1.0  # if all scores are equal, set to 1
    else:
        for item in items:
            item[score_key] = (item.get(score_key, 0) - min_score) / (max_score - min_score)

async def store_conversation_to_es(user_id: str, session_id: str, user_message: str, bot_response: str) -> str:
    print("Storing conversation to ES...")
    await create_index_if_not_exists()
    
    combined_text = f"{user_message} {bot_response}"
    user_emb, bot_emb, combined_emb = await embed_texts(
        [user_message, bot_response, combined_text]
    )

    doc = {
        "user_id": user_id,
        "session_id": session_id,
        "user_message": user_message,
        "bot_response": bot_response,
        "user_embedding": user_emb,
        "bot_embedding": bot_emb,
        "combined_embedding": combined_emb,
        "timestamp": datetime.utcnow().isoformat()
    }

    # 2. Await the database index operation using the async client
    res = await es.index(index=es_ltm_index, document=doc)
    return res["_id"]
    
def _parse_ltm_hits(hits: list[dict], score_key: str = "semantic_score") -> list[dict]:
    """
    Convert Elasticsearch hits into a standard candidate dict list.
    score_key: either 'bm_score' or 'semantic_score'
    """
    candidates = []
    for hit in hits:
        src = hit["_source"]
        candidates.append({
            "id": hit["_id"],
            score_key: hit["_score"],
            "user_id": src.get("user_id"),
            "session_id": src.get("session_id"),
            "user_message": src.get("user_message", ""),
            "bot_response": src.get("bot_response", ""),
            "timestamp": src.get("timestamp")
        })
    return candidates

async def fetch_from_ltm( query: str, user_id: str, session_id: str, top_k: int = 10) -> list[dict]:
    request_body = {
        "query": {
            "bool": {
                "must": [
                    {"multi_match": {"query": query, "fields": ["user_message", "bot_response"]}}
                ],
                "filter": [
                    {"term": {"user_id": user_id}},
                    {"term": {"session_id": session_id}}
                ]
            }
        },
        "size": top_k,
        "_source": {"excludes": ["user_embedding", "bot_embedding", "combined_embedding"]}
    }

    res = await es.search(index=es_ltm_index, body=request_body)
    return _parse_ltm_hits(res["hits"]["hits"], score_key="bm_score")

async def fetch_knn_from_ltm( query: str, user_id: str, session_id: str, top_k: int = 10) -> list[dict]:
    query_vec = (await embed_texts([query]))[0]
    
    request_body = {
        "size": top_k,
        "knn": {
            "field": "combined_embedding",
            "query_vector": query_vec,
            "k": top_k * 3,
            "num_candidates": top_k * 5,
            "filter": [
                {"term": {"user_id": user_id}},
                {"term": {"session_id": session_id}}
            ]
        },
        "_source": {"excludes": ["user_embedding", "bot_embedding", "combined_embedding"]}
    }

    res = await es.search(index=es_ltm_index, body=request_body)
    candidates = _parse_ltm_hits(res["hits"]["hits"], score_key="semantic_score")
    print(f"KNN candidates before filtering: {len(candidates)}")
    # Filter based on raw semantic score threshold before normalization
    candidates = [c for c in candidates if c["semantic_score"] >= 0.6]
    print(f"KNN candidates after filtering: {len(candidates)}")
    return candidates

def rrf_fuse_ltm(bm25_list: list[dict], knn_list: list[dict], k: int = 60) -> list[dict]:
    score_dict = {}

    # BM25 contribution
    for rank, chunk in enumerate(bm25_list, start=1):
        score_dict[chunk["id"]] = score_dict.get(chunk["id"], 0) + 1 / (k + rank)

    # KNN contribution
    for rank, chunk in enumerate(knn_list, start=1):
        score_dict[chunk["id"]] = score_dict.get(chunk["id"], 0) + 1 / (k + rank)

    # Merge actual candidate data
    merged = {c["id"]: c for c in bm25_list + knn_list}

    fused_candidates = []
    for chunk_id, score in score_dict.items():
        bm_chunk = next((c for c in bm25_list if c["id"] == chunk_id), None)
        knn_chunk = next((c for c in knn_list if c["id"] == chunk_id), None)

        fused_candidates.append({
            "chunk": merged[chunk_id],
            "rrf_score": score,
            "bm_score": bm_chunk.get("bm_score", 0) if bm_chunk else 0,
            "semantic_score": knn_chunk.get("semantic_score", 0) if knn_chunk else 0
        })

    # Sort descending by RRF score
    fused_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused_candidates

async def retrieve_ltm_context( query: str, session_id: str, user_id: str ="12345",
                        top_k_bm25: int = 10, top_k_knn: int = 10, top_n: int = 3) -> list[dict]:
    print("Retrieving LTM context...")
    await create_index_if_not_exists()
    
    # Step 1: Fetch BM25 candidates
    bm25_candidates = await fetch_from_ltm(query, user_id, session_id, top_k=top_k_bm25)
    normalize_scores(bm25_candidates, "bm_score")
    bm25_candidates.sort(key=lambda x: x["bm_score"], reverse=True)

    # Step 2: Fetch KNN candidates
    knn_candidates = await fetch_knn_from_ltm(query, user_id, session_id, top_k=top_k_knn)
    normalize_scores(knn_candidates, "semantic_score")
    knn_candidates.sort(key=lambda x: x["semantic_score"], reverse=True)

    # Step 3: Fuse BM25 + KNN results using RRF
    fused_results = rrf_fuse_ltm(bm25_candidates, knn_candidates)
    normalize_scores(fused_results, "rrf_score")

    # Step 4: Return only top N fused conversation turns
    top_results =  fused_results[:top_n]

    # Keep only relevant conversation fields
    final_results = [
        {
            "user_id": r["chunk"].get("user_id"),
            "session_id": r["chunk"].get("session_id"),
            "rrf_score": r["rrf_score"],
            "bm_score": r["bm_score"],
            "semantic_score": r["semantic_score"],
            "user_message": r["chunk"].get("user_message"),
            "bot_response": r["chunk"].get("bot_response"),
            "timestamp": r["chunk"].get("timestamp")
        }
        for r in top_results
    ]

    # --- Step 5: Filter based on semantic_score threshold ---
    filtered_results = [r for r in final_results if r["semantic_score"] >= 0.6]

    return filtered_results
