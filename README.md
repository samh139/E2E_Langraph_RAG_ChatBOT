# E2E_Langraph_RAG_ChatBOT

# End-to-End Enterprise RAG Chatbot Platform

An event-driven, production-grade Retrieval-Augmented Generation (RAG) platform. The architecture is decoupled into asynchronous document ingestion pipelines and reactive user question-answering loops using LangGraph, Apache Kafka, Elasticsearch, Redis, and MongoDB.

---

## 🏗️ Architecture Overview

The system is split into two primary operational lifecycle layers:
1. **Document Ingestion & Indexing Pipeline:** Automatically reads raw documents out of object storage, splits text into optimal chunks, captures high-dimensional semantic vector embeddings, and registers index payloads to a vector database engine.
2. **Retrieval & Answer Generation Pipeline:** Accepts customer socket questions, classifies intents, executes contextual search expansions against vector documents, handles dialogue ambiguities, and maps grounded responses back to users.

---

## 🛠️ Technology Stack Components
* **Core Application Worker:** Python system layer orchestrating ingestion engines and LangGraph state machines.
* **Streaming Backbone (Apache Kafka & Zookeeper):** Handles asynchronous decoupling of document processing blocks and live client message queues.
* **Vector Database Engine (Elasticsearch):** Manages high-performance lexical search and dense vector search capabilities.
* **Raw Object Storage Tier (MinIO):** S3-compatible enterprise file storage engine protecting source assets.
* **Stateful Checkpoint Memory (Redis):** Tracks user session transactions, chat state limits, and short-term operational histories.
* **Offline Evaluation Results (MongoDB):** Stores DeepEval benchmark inputs, retrieved chunks, scores, and judge explanations for later review. Conversation memory currently uses Elasticsearch and Redis.

## 🔐 End-to-End Role-Based Access Control

RBAC is enforced across the chat request path, from WebSocket authentication through document retrieval:

1. The WebSocket gateway validates the JWT signature, issuer, audience, expiry, and role claims.
2. Verified JWT roles are mapped through `app/configs/rbac_roles.yaml` to permitted document roles. Connections without valid permissions are rejected.
3. Each Kafka request carries the verified identity and permissions in an HMAC-signed context. The retrieval router checks the signature, request age, and that permissions match the configured policy.
4. Elasticsearch applies the permitted document-role filter to both BM25 and vector search, preventing unauthorized chunks from reaching reranking or answer synthesis.
5. Conversation memory is stored and retrieved within a permission-derived scope, so memory created under a different access scope is not reused.

The policy separates document-role-to-filename assignments (`roles_mapping`) from JWT-role-to-document-role permissions (`role_permissions`). See [the RBAC request and retrieval flow](docs/rbac_jwt_retrieval_flow.md) for a component-level walkthrough.

### Test the authenticated WebSocket locally

Start the application and its dependencies:

```bash
docker compose up -d --build
```

Generate a short-lived development JWT using a role configured in `app/configs/rbac_roles.yaml`:

```bash
docker compose run --rm --no-deps --entrypoint python retrieval-router \
  -m app.security.create_test_token --role retail_banking_staff
```

In Postman, create a WebSocket request to `ws://localhost:8000/ws/rbac-smoke` and add this request header before connecting:

```text
Authorization: Bearer <paste-the-generated-token>
```

After connecting, send a message such as:

```json
{"message": "What are SBI NEFT charges?"}
```

The test-token command is restricted to local development and requires the development JWT settings in `app/configs/.env`. Do not use it as a production identity provider.

## ⚡ Permission-Scoped Answer Cache

The retrieval workflow uses Redis to cache successful answers for standalone knowledge queries. The lookup happens after query refinement and before retrieval. A cache hit returns the answer and its source chunks without repeating retrieval or synthesis. Queries that require conversation memory bypass the shared cache.

Cache keys include the normalized refined query, permission-derived `memory_scope_id`, Elasticsearch index alias, and current model and retrieval settings. Users with the same effective document permissions can share cached results; different permission scopes use different keys. Entries expire after 15 minutes by default. Set `ANSWER_CACHE_TTL_SECONDS` to change the TTL, and bump `ANSWER_CACHE_VERSION` when prompts or answer behavior change.

To see the cache working, connect to the authenticated WebSocket as described above and send the same standalone question twice. The first request should log `RAG answer cache miss`; the next should log `RAG answer cache hit`:

```bash
docker compose logs -f retrieval-router
```

Cache entries include retrieved source chunks and are stored in Redis. Protect Redis access as you would other application data. Evaluation runs should be interpreted with care when they use cached results, since the cache reuses the original retrieved context.

The retrieval LangGraph is compiled with `AsyncRedisSaver` in the live router.
Each query uses its own `thread_id` (`query_id`) so checkpoint state cannot leak
between turns or authorization scopes; the conversation memory remains in the
existing STM/LTM stores. Checkpoints expire after 24 hours by default. Set
`LANGGRAPH_CHECKPOINT_TTL_MINUTES` to change the retention period. Redis
checkpointer indices are initialized when the router starts.

---

## 🚀 Quick Start Deployment Guide

Follow these sequential setup steps to initialize the environment from absolute zero.

### 1. Prerequisites & Environment Variables
Ensure you have Docker and Docker Compose installed on your system. Before booting up the stack, verify that your local environment configurations exist at `./app/configs/.env` to feed credentials into the data layers.

### 2. Force Clear Stale System States
If you are iterating rapidly or experiencing local container name resource allocation errors, wipe out old daemon states completely:
```bash
docker rm -f es zookeeper kafka minio minio_mc redis redisinsight mongo chatbot_worker || true
docker system prune -f
```

### 3. Build and Orchestrate the Topology
Execute the Docker build engine to compile the local workspace manifests and bring up the microservice nodes in detached mode:
```bash
docker compose up -d --build
```

### 4. Verify Platform Cluster Health
Wait roughly 15 seconds for Elasticsearch and Kafka clusters to pass internal boot checks, then execute:
```bash
docker compose ps
```
*Ensure that both `es` and `redis` display a `(healthy)` status line in your terminal.*

---

## 🔬 Infrastructure Verification & Inspection

To verify that the automated bootstrap provisioning scripts completed successfully, validate your storage and queue engines using these commands:

### A. Inspect Event Stream Channels (Kafka)
Confirm that the platform's four core system topics were explicitly established with appropriate scaling partitions:
```bash
docker exec -it kafka kafka-topics --list --bootstrap-server localhost:9092
```

**Expected Output includes:**
```text
chat-requests
chat-responses
es.chunks
es.chunks.dlq
es.embedded_chunks
es.embedded_chunks.dlq
```

### Embedding and indexing recovery

The embedding and indexing consumers use manual Kafka commits. The embed worker
commits a source chunk only after its vector is confirmed in Kafka; the indexer
commits only after Elasticsearch accepts the document. Both use `chunk_id` as
the Elasticsearch document ID, so replay after a crash is idempotent.

Transient Ollama and Elasticsearch failures are retried with exponential backoff.
After retries are exhausted, the worker exits without committing that source
offset, so the message remains available to the same consumer group on restart.
Malformed records and permanent Elasticsearch 4xx rejections are copied to
their respective DLQ topics before the source offset is committed. Configure
`KAFKA_PROCESSING_MAX_ATTEMPTS` (default `5`), `KAFKA_RETRY_INITIAL_SECONDS`
(default `1`), and `KAFKA_RETRY_MAX_SECONDS` (default `30`) to tune retries.

DLQ records contain the original chunk payload, which may include sensitive
banking data. Production Kafka deployments must apply appropriate access
controls, encryption, and retention to both DLQ topics. The single-broker,
replication-factor-one Compose configuration is for local development; it does
not provide broker-failure durability.

### B. Verify Object Storage Buckets (MinIO)
The stack uses `minio_mc` to automatically configure local directory structures on startup. Verify the dashboard console is reachable by browsing to:
* **Console URL:** http://localhost:9001
* **API Endpoint:** http://localhost:9010

---

## 👨‍💻 Local Python App Workspace Setup

To start developing components (such as the `BulkIngestWorker` or `EmbedWorker`), configure a localized virtual environment on your host machine:

```bash
# 1. Create a dedicated virtual environment
python3 -m venv rag_chat_env

# 2. Activate the virtual environment
source rag_chat_env/bin/activate  # On Windows use: rag_chat_env\Scripts\activate

# 3. Install core production runtime dependencies

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

## If you want to reinstall all packages
#pip install -r requirements.txt

# 4. Run the docker image-
docker compose up -d --build

#5. Restart and run the application
docker compose restart app-worker
docker compose logs -f app-worker

#6. To stop the running container
docker compose down

#7. To prune all volumes and remove all orphans(Resets everything, then build again)
docker compose down --remove-orphans -v
docker volume prune --force
```

## Retrieval
## Manually create the retrieval kafka topics
docker compose exec kafka kafka-topics \
  --create --if-not-exists \
  --bootstrap-server kafka:9092 \
  --partitions 3 --replication-factor 1 \
  --topic chat-requests

docker compose exec kafka kafka-topics \
  --create --if-not-exists \
  --bootstrap-server kafka:9092 \
  --partitions 3 --replication-factor 1 \
  --topic chat-responses

## Run the entire project with:
docker compose up -d
## Run ingestion only, along with its Kafka, Elasticsearch, and Redis dependencies:
docker compose up -d app-worker
## Run the chat/retrieval flow only, along with its dependencies:
## 1st terminal
docker compose up websocket 
## 2nd terminal
docker compose up retrieval-router
## For routine restarts, use docker compose restart app-worker, docker compose restart websocket retrieval-router, or docker compose restart for everything. You don’t need --build unless the Dockerfile or dependencies changed.
## For authenticated WebSocket testing, follow “Test the authenticated WebSocket locally” above.
---

## 💬 Interview Discussion Guide (How to explain this start)

When asked about your setup and initialization choices during technical rounds, highlight these architectural talking points:

*"Instead of relying on unstable runtime anti-patterns like Kafka's default topic auto-creation or manual administrative setup, I engineered infrastructure-as-code automation directly into the compose lifecycle. I built autonomous setup containers that block execution until core storage and message brokers stabilize, programmatically injecting optimized partition topologies from day one. This guarantees that our data ingestion microservices map to decoupled, horizontally-scalable streams without configuration drift across development and deployment environments."*
