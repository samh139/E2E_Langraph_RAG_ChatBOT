import asyncio

from confluent_kafka import Consumer
import json
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from app.telemetry import configure_telemetry, extract_trace_context

configure_telemetry("websocket")
tracer = trace.get_tracer(__name__)

consumer = Consumer({
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'websocket-group',
    'auto.offset.reset': 'earliest'})

consumer.subscribe(['chat-responses'])

async def consume_loop(manager):
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            await asyncio.sleep(0.1)
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue

        context = extract_trace_context(msg.headers())
        with tracer.start_as_current_span(
            "chat.websocket.response",
            context=context,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                data = json.loads(msg.value().decode())
                await manager.send(data["session_id"], data)
            except Exception as exc:
                span.set_attribute("error.type", type(exc).__name__)
                span.set_status(Status(StatusCode.ERROR))
                raise
