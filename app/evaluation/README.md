# Offline DeepEval benchmark

This is an explicitly run, one-shot benchmark. It does not run for normal chat requests. Inputs can be plain text questions or JSONL golden cases with a reference answer:

```text
What are the NEFT charges for SBI Bank?
How can I replace a damaged debit card?
Are ATM fees different for other banks?
```

The repository includes three practice cases in `app/evaluation/golden_dataset.jsonl`. Each JSONL row contains an `id`, `input`, `expected_output`, `source_documents`, an optional `role`, and official SBI `source_url`. A case-level role lets one benchmark cover documents assigned to different RBAC roles; cases without it use the CLI `--role` value.

For golden cases with a non-empty `expected_output`, the runner scores Faithfulness, Answer Relevancy, Contextual Relevancy, Contextual Precision, Contextual Recall, and Answer Correctness (implemented with DeepEval `GEval`). It also calculates document-level Precision@5, Recall@5, and MRR@5 from the ranked retrieved chunks and `source_documents` labels. Repeated chunks from one file count as one ranked document. Plain-text query files do not include reference labels, so ranking metrics are recorded as skipped; the three existing reference-based DeepEval metrics are skipped as well.

Run the practice dataset from the directory containing `docker-compose.yml`:

```bash
docker compose --profile evaluation run --build --rm evaluation-runner \
  --queries-file app/evaluation/golden_dataset.jsonl
```

For your own questions, save one question per line in `app/evaluation/queries.txt` and run:

```bash
docker compose --profile evaluation run --build --rm evaluation-runner \
  --queries-file app/evaluation/queries.txt
```

For a single question:

```bash
docker compose --profile evaluation run --rm evaluation-runner \
  --query "What are the NEFT charges for SBI Bank?"
```

The runner starts Elasticsearch, Redis, and MongoDB as dependencies. MongoDB is only a dependency of this opt-in Compose profile; it does not gate the normal retrieval-router. Results go to database `rag_evaluation`, collection `deepeval_results`.

Each record includes a benchmark run ID, query ID, question, expected answer (if supplied), refined query, response, retrieved chunk IDs/names/text, retrieval score, judge model, and metric scores/reasons. Failures are saved with their query and error before the run stops. The retrieved text and answer are stored to make a score reviewable, so keep the local MongoDB volume and query file within your intended data-handling boundary.

Inspect recent results with:

```bash
docker compose exec mongo mongosh rag_evaluation --quiet --eval \
  'db.deepeval_results.find({}, {query:1, status:1, metrics:1, error:1, created_at:1}).sort({created_at:-1}).limit(10).forEach(printjson)'
```

## WebSocket load test with Locust

Install the optional Locust dependencies in the active environment:

```bash
pip install -r requirements-locust.txt
```

Start the chat services (`docker compose up -d websocket retrieval-router`) and provide a valid JWT for a configured role. For local development, the repository's test-token helper can print one when test token creation is enabled in `app/configs/.env`:

```bash
docker compose run --rm --no-deps --entrypoint python retrieval-router \
  -m app.security.create_test_token --role retail_banking_staff
```

Then run Locust from the repository root:

```bash
LOCUST_JWT='<token>' locust -f app/evaluation/locustfile.py \
  --host ws://localhost:8000
```

Open <http://localhost:8089>, set the user count and spawn rate, and start the run. Each simulated user holds its own authenticated WebSocket and sends one chat question at a time, measuring until the corresponding asynchronous chat response arrives. Set `LOCUST_QUESTION`, `LOCUST_WS_URL`, or `LOCUST_RESPONSE_TIMEOUT_SECONDS` to change the question, gateway URL, or response timeout. Use a short-lived development token only in a local environment.
