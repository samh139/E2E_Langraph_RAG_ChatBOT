"""
Utility: Nightly Cluster Refiner (Qwen-generated CHUNK tags + Vectors)
---------------------------------------------------------------------
Responsibilities:
  • Agglomerative reclustering (cosine distance)
  • Qwen generates *chunk-level* tags (semantic topics)
  • Embeds those tags with MiniLM → stored in chunks_es.chunk_metadata.tag_vector
  • Aggregates chunk tags per cluster → cluster.tags
  • Qwen generates cluster.summary
  • Immediately upserts both cluster docs and enriched chunk docs
  • Logs detailed progress with per-step and total durations
  • Designed for nohup background runs
"""

import datetime, time, numpy as np
from elasticsearch import Elasticsearch, helpers
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
import requests
import asyncio
import httpx
import numpy as np

CONCURRENCY_LIMIT = 8 
semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
ES_URL = "http://localhost:9200"
CHUNKS_INDEX = "dsprawl_documents"
CLUSTERS_INDEX = "clusters_v2"

ES = Elasticsearch(ES_URL)

# ----------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------

async def async_fire_ollama(client, endpoint, payload):
    async with semaphore:
        try:
            # Note: /api/embeddings and /api/generate have different response keys
            response = await client.post(f"http://localhost:11434/api/{endpoint}", json=payload, timeout=300)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error hitting {endpoint}: {e}")
            return None
        
async def batch_embed(texts, model="nomic-embed-text"):
    async with httpx.AsyncClient() as client:
        tasks = [async_fire_ollama(client, "embeddings", {"model": model, "prompt": t}) for t in texts]
        results = await asyncio.gather(*tasks)
        
        vectors = []
        for r in results:
            if r and "embedding" in r:
                vec = np.array(r["embedding"])
                norm = np.linalg.norm(vec)
                vectors.append(vec / norm if norm > 0 else vec)
        return np.array(vectors)
    

async def batch_tags(texts, model="qwen2.5vl:latest"):
    async with httpx.AsyncClient() as client:
        tasks = []
        for text in texts:
            prompt = f"From the following text, extract 5–8 short meaningful tags (topics, actions, entities). Comma-separated only:\n\n{text[:2000]}"
            tasks.append(async_fire_ollama(client, "generate", {"model": model, "prompt": prompt, "stream": False}))
        
        results = await asyncio.gather(*tasks)
        
        all_tags_lists = []
        for r in results:
            if r and "response" in r:
                raw = r["response"]
                tags = [t.strip().strip('"').strip("'").title() for t in raw.split(",") if len(t.strip()) > 1]
                all_tags_lists.append(list(dict.fromkeys(tags))[:8])
            else:
                all_tags_lists.append([])
        return all_tags_lists
    

# ----------------------------------------------------------------------
# Summary generator
# ----------------------------------------------------------------------

def summary_for_cluster(title: str, texts):

    sample = "\n\n".join(texts[:5])

    prompt = f"""
Summarize the following document cluster in 3–4 sentences.

Cluster title: {title}

Content sample:
{sample}
"""

    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5vl:latest",
            "prompt": prompt,
            "stream": False,
        },
        timeout=300,
    )

    r.raise_for_status()

    return r.json()["response"]



# ----------------------------------------------------------------------
# ES helpers
# ----------------------------------------------------------------------
def es_search(index: str, query: dict, size: int = 10000):
    res = ES.search(index=index, body=query, size=size)
    return res["hits"]["hits"]

def es_update_or_create(index: str, doc_id: str, body: dict):
    """Safe update/upsert wrapper."""
    try:
        ES.update(index=index, id=doc_id, doc=body, doc_as_upsert=True)
    except Exception as e:
        print(f"⚠️ Update failed for {doc_id}: {e}")

# ----------------------------------------------------------------------
# Core refinement
# ----------------------------------------------------------------------
async def process_cluster_async(idx, cid, group, refresh_summary):
    """Handles the heavy lifting for a single cluster asynchronously."""
    c_start = time.time()
    docs = group["docs"]
    vset = np.vstack(group["vecs"])
    cluster_texts = [d["_source"]["content"] for d in docs]
    centroid = np.mean(vset, axis=0)
    cluster_id = f"CL_{cid}_{datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    print(f"▶️ Cluster {idx} — {len(docs)} chunks: Generating tags...")
    
    # 1. Generate tags and tag embeddings in parallel for this cluster
    all_chunk_tags = await batch_tags(cluster_texts)
    tag_strings = [" ".join(tags) if tags else "" for tags in all_chunk_tags]
    all_tag_vecs = await batch_embed(tag_strings)

    # 2. Prepare Bulk Updates for ES
    chunk_actions = []
    all_tags = []
    for d, tags, vec in zip(docs, all_chunk_tags, all_tag_vecs):
        if not tags: continue
        chunk_actions.append({
            "_op_type": "update",
            "_index": CHUNKS_INDEX,
            "_id": d["_id"],
            "doc": {"chunk_metadata": {"tags": tags}, "tag_vector": vec.tolist()},
            "doc_as_upsert": True
        })
        all_tags.extend(tags)

    if chunk_actions:
        helpers.bulk(ES, chunk_actions, raise_on_error=False)

    # 3. Summary (Can also be made async if you wrap summary_for_cluster)
    summary = ""
    if refresh_summary:
        # Assuming you keep summary_for_cluster synchronous for now
        summary = summary_for_cluster(cluster_id, cluster_texts)

    cluster_doc = {
        "cluster_id": cluster_id,
        "summary": summary,
        "tags": sorted(set(all_tags)),
        "vector": centroid.tolist(),
        "last_updated": datetime.datetime.utcnow().isoformat()
    }
    
    es_update_or_create(CLUSTERS_INDEX, cluster_id, cluster_doc)
    print(f"✅ Cluster {cluster_id} done — {time.time()-c_start:.2f}s")

async def refine_clusters_async(refresh_summary=True, cluster_count=None):
    job_start = time.time()
    
    # 1. Fetch
    chunks = es_search(CHUNKS_INDEX, {"query": {"match_all": {}}})
    if not chunks: return
    texts = [c["_source"]["content"] for c in chunks]

    # 2. Initial Embedding (Batch all at once)
    print(f"🧠 Embedding {len(texts)} chunks...")
    vecs = await batch_embed(texts)

    # 3. Clustering (CPU Bound - keep synchronous)
    n_clusters = cluster_count or max(5, len(chunks) // 50)
    dist = cosine_distances(vecs)
    labels = AgglomerativeClustering(n_clusters=n_clusters, metric="precomputed", linkage="average").fit_predict(dist)

    clusters = {}
    for lab, doc, v in zip(labels, chunks, vecs):
        clusters.setdefault(lab, {"docs": [], "vecs": []})
        clusters[lab]["docs"].append(doc)
        clusters[lab]["vecs"].append(v)

    # 4. Process clusters (One by one to avoid overwhelming GPU, but internal tags are parallel)
    for idx, (cid, group) in enumerate(clusters.items(), start=1):
        await process_cluster_async(idx, cid, group, refresh_summary)

    print(f"\n🏁 Finished in {time.time() - job_start:.2f}s")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("******* Start time *******", datetime.datetime.utcnow())
    asyncio.run(refine_clusters_async())
    print("******* End time *******", datetime.datetime.utcnow())
 