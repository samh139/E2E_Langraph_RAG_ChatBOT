import asyncio
import json
import uuid
from app.app_logger import LoggerFactory
from confluent_kafka import Consumer, Producer
from app.workflow.retrieval_worflow import *

logger = LoggerFactory.get_logger(__name__)

class RequestResponseRouter:
    def __init__(self, kafka_bootstrap_servers='kafka:9092',max_concurrency=10):
        self.consumer = Consumer({
            'bootstrap.servers': kafka_bootstrap_servers,
            'group.id': 'request_response_router_group',
            'auto.offset.reset': 'earliest'
        })
        self.producer = Producer({'bootstrap.servers': kafka_bootstrap_servers})
        self.consumer.subscribe(['chat-requests'])
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.tasks = set()
        # self.pending_requests:dict[str, asyncio.Future] = {}

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
                task = asyncio.create_task(
                self._handle_request(request_data)
            )

                self.tasks.add(task)
                task.add_done_callback(self.tasks.discard)

            except Exception as e:
                logger.error(f"Error receiving Kafka message: {e}")

    async def _handle_request(self, request_data: dict):
        async with self.semaphore:
            response_data = await self.process_request(request_data)
            response_topic = 'chat-responses'
            self.producer.produce(
                "chat-responses",
                key=response_data["session_id"],
                value=json.dumps(response_data),
            )

            self.producer.flush()
            logger.info(f"Sent response: {response_data} to topic: {response_topic}")

    async def process_request(self, request_data:dict):
        session_id = request_data["session_id"]
        if not session_id:
            logger.error("Missing session_id in request data")
            return {"error": "Missing session_id"}
        user_message = request_data["message"]
        if not user_message:
            logger.error("Missing message in request data")
            return {"error": "Missing message"}
        user_id = request_data.get("user_id","default_user")
        query_id = str(uuid.uuid4())
        
        ## Trigger Langraph Runtime
        graph_input = {
            "session_id": session_id,
            "query_id": query_id,
            "user_id": user_id,
            "message": user_message,
            }

        # result = await retrieval_graph.ainvoke(graph_input)

        # return {
        #     "session_id": session_id,
        #     "query_id": query_id,
        #     "response": result["response"],
        # }
            



        #     return {"response": request_data}

    