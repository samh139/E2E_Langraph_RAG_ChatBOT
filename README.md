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
* **Metadata & Logging Layer (MongoDB):** Long-term historical conversation store, deep audit log tracker, and analytics warehouse.

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

**Expected Output:**
```text
chat.requests
chat.responses
dsprawl.chunks
dsprawl.embedded_chunks
```

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
pip install -r requirements.txt
```

---

## 💬 Interview Discussion Guide (How to explain this start)

When asked about your setup and initialization choices during technical rounds, highlight these architectural talking points:

*"Instead of relying on unstable runtime anti-patterns like Kafka's default topic auto-creation or manual administrative setup, I engineered infrastructure-as-code automation directly into the compose lifecycle. I built autonomous setup containers that block execution until core storage and message brokers stabilize, programmatically injecting optimized partition topologies from day one. This guarantees that our data ingestion microservices map to decoupled, horizontally-scalable streams without configuration drift across development and deployment environments."*

