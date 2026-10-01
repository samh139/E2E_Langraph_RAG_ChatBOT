import asyncio
import json
import uuid
from app.app_logger import LoggerFactory
from confluent_kafka import Consumer, Producer
from app.workflow.retrieval_worflow import *

logger = LoggerFactory.get_logger(__name__)

class RequestResponseRouter:
    def __init__(self, kafka_bootstrap_servers='kafka:9092'):
        self.consumer = Consumer({
            'bootstrap.servers': kafka_bootstrap_servers,
            'group.id': 'request_response_router_group',
            'auto.offset.reset': 'earliest'
        })
        self.producer = Producer({'bootstrap.servers': kafka_bootstrap_servers})
        self.consumer.subscribe(['chat-requests'])
        self.pending_requests:dict[str, asyncio.Future] = {}

    async def start(self):
        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                await asyncio.sleep(0.1)
                continue
            if msg.error():
                logger.error(f"Consumer error: {msg.error()}")
                continue

            try:
                request_data = json.loads(msg.value().decode('utf-8'))
                logger.info(f"Received request: {request_data}")

                # Process the request and generate a response
                response_data = await self.process_request(request_data)

                self.producer.produce('chat-responses',
                                  key=response_data["session_id"], value=json.dumps(response_data))

                # Send the response back to Kafka
                self.producer.flush()
                logger.info(f"Sent response: {response_data} to chat-responses topic")
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    async def process_request(self, request_data:dict):
        session_id = request_data.get("session_id")
        if not session_id:
            logger.error("Missing session_id in request data")
            return {"error": "Missing session_id"}
        user_message = request_data.get("message")
        if not user_message:
            logger.error("Missing message in request data")
            return {"error": "Missing message"}
        user_id = request_data.get("user_id","default_user")
        query_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.pending_requests[session_id] = future

        ## Trigger Langraph Runtime
        



        return {"response": request_data}

    