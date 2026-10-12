# Local OpenTelemetry

The local OpenTelemetry Collector receives traces and metrics from the chat
services. It sends traces to Jaeger and exposes metrics for Prometheus to scrape.

Start the chat path and observability services with:

```sh
docker compose up -d --build websocket retrieval-router prometheus
```

Compose also starts the Collector, Jaeger, Kafka, Elasticsearch, and Redis as
dependencies. Ollama must be reachable at `OLLAMA_URL` (by default,
`http://host.docker.internal:11434`).

- Jaeger traces: <http://localhost:16686>
- Prometheus metrics and query UI: <http://localhost:9090>

Send a chat request to generate traffic. Prometheus scrapes the Collector every
15 seconds; metric export from each app process runs every 10 seconds.

## Metrics included

- Cache lookup and write counts, outcomes, and durations.
- Retrieval outcome and duration, chunk counts by stage, top relevance score,
  and fallback counts.
- Redis, Elasticsearch, Ollama, and Gemini operation counts, durations, and
  outcomes (`success`, `error`, `timeout`, or `rate_limited`).
- Gemini input and output tokens when the provider returns usage metadata.

Metric attributes use bounded operation and outcome names. They do not include
queries, prompts, answers, user identifiers, roles, or document contents.

Example PromQL for cache hit ratio:

```promql
sum(rate(rag_cache_operations_total{operation="lookup",outcome="hit"}[5m]))
/
sum(rate(rag_cache_operations_total{operation="lookup",outcome=~"hit|miss"}[5m]))
```

Example PromQL for dependency errors, timeouts, and rate limits:

```promql
sum by (provider, operation, outcome) (
  rate(rag_dependency_operations_total{outcome=~"error|timeout|rate_limited"}[5m])
)
```

Use the Prometheus query autocomplete with the `rag_` prefix to inspect the
other counters and histogram buckets. Availability is currently reflected by
the success and failure of operations the application performs; there are no
periodic synthetic health probes. Internal SDK retry attempts and estimated
provider cost are not recorded. Gemini token usage is provider-reported and may
be absent from responses.

The trace path propagates W3C trace context across Kafka. It intentionally
excludes message text, prompts, answers, JWTs, user IDs, and document contents.
Jaeger uses local in-memory storage, so traces are lost when it restarts.
