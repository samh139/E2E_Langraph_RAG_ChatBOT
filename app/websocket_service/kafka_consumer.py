import asyncio

from confluent_kafka import Consumer
import json

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

        data = json.loads(msg.value().decode())
        await manager.send(data["session_id"], data)