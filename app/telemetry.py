"""Small OpenTelemetry setup and Kafka trace-context helpers."""

from __future__ import annotations

import atexit
import os
from typing import Any

from opentelemetry import metrics, propagate, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


_provider: TracerProvider | None = None
_meter_provider: MeterProvider | None = None


def configure_telemetry(service_name: str) -> None:
    """Configure OTLP tracing once for this Python process."""
    global _provider, _meter_provider
    if _provider is not None or os.getenv("OTEL_SDK_DISABLED", "false").lower() == "true":
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "otel-collector:4317")
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": "e2e-langgraph-rag",
        }
    )
    _provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        insecure=os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true",
    )
    _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)
    atexit.register(_provider.shutdown)

    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(
            endpoint=endpoint,
            insecure=os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "true").lower() == "true",
        ),
        export_interval_millis=int(os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "10000")),
    )
    _meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(_meter_provider)
    atexit.register(_meter_provider.shutdown)


def inject_trace_headers() -> list[tuple[str, str]]:
    """Serialize the current W3C trace context into Kafka headers."""
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    return list(carrier.items())


def extract_trace_context(headers: list[tuple[str, Any]] | None):
    """Extract W3C trace context from confluent-kafka message headers."""
    carrier: dict[str, str] = {}
    for key, value in headers or []:
        if value is None:
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            carrier[key] = value
    return propagate.extract(carrier)
