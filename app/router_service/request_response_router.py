import asyncio
import os
import json
import uuid
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from app.app_logger import LoggerFactory
from confluent_kafka import Consumer, Producer
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from app.telemetry import configure_telemetry, extract_trace_context, inject_trace_headers

configure_telemetry("retrieval-router")
tracer = trace.get_tracer(__name__)

from app.workflow.retrieval_workflow import RetrievalWorkflow
from app.workflow.retrieval_state import RetrieverState
from app.memory_service.memory_team.ltm.ltm_service import store_conversation_to_es
from app.memory_service.memory_team.stm.agent import (
    update_session_stm
)
from app.security.access_control import (
    memory_scope_id,
    verify_internal_request,
)

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
        self.retrieval_workflow = None
        # self.pending_requests:dict[str, asyncio.Future] = {}

    async def start(self):
        redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        checkpoint_ttl_minutes = int(
            os.getenv("LANGGRAPH_CHECKPOINT_TTL_MINUTES", "1440")
        )
        async with AsyncRedisSaver.from_conn_string(
            redis_url,
            ttl={"default_ttl": checkpoint_ttl_minutes, "refresh_on_read": True},
        ) as checkpointer:
            await checkpointer.setup()
            self.retrieval_workflow = RetrievalWorkflow(checkpointer=checkpointer)
            logger.info(
                "Retrieval LangGraph Redis checkpointing enabled; TTL=%s minutes",
                checkpoint_ttl_minutes,
            )
            await self._consume_requests()

    async def _consume_requests(self):
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
                logger.info(
                    "Received chat request: session_id=%s connection_id=%s",
                    request_data.get("session_id"),
                    request_data.get("connection_id"),
                )

                # Process the request and generate a response
                task = asyncio.create_task(
                    self._handle_request(
                        request_data,
                        extract_trace_context(msg.headers()),
                    )
                )

                self.tasks.add(task)
                task.add_done_callback(self._on_request_task_done)

            except Exception as e:
                logger.error(f"Error receiving Kafka message: {e}")

    def _on_request_task_done(self, task: asyncio.Task) -> None:
        self.tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Request handler task failed")

    async def _handle_request(self, request_data: dict, trace_context=None):
        with tracer.start_as_current_span(
            "chat.request.process",
            context=trace_context,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            async with self.semaphore:
                try:
                    response_data = await self.process_request(request_data)
                except Exception as exc:
                    span.set_attribute("error.type", type(exc).__name__)
                    span.set_status(Status(StatusCode.ERROR))
                    logger.exception("Unable to process authenticated chat request")
                    response_data = {
                        "session_id": request_data.get("session_id", ""),
                        "connection_id": request_data.get("connection_id", ""),
                        "response": "I couldn’t securely process that request. Please reconnect and try again.",
                    }
                self.producer.produce(
                    "chat-responses",
                    key=response_data["session_id"],
                    value=json.dumps(response_data),
                    headers=inject_trace_headers(),
                )

                self.producer.flush()

    async def process_request(self, request_data:dict):
        # Kafka requests are accepted only when signed by the authenticated gateway.
        signed_context = verify_internal_request(request_data)
        session_id = signed_context["session_id"]
        user_message = signed_context["message"]
        user_id = signed_context["user_id"]
        user_roles = signed_context["roles"]
        allowed_roles = signed_context["allowed_roles"]
        scope_id = memory_scope_id(allowed_roles)
        query_id = str(uuid.uuid4())
        
        ## Trigger Langraph Runtime
        graph_input = RetrieverState(
            session_id=session_id,
            query_id=query_id,
            user_id=user_id,
            user_roles=user_roles,
            allowed_roles=allowed_roles,
            memory_scope_id=scope_id,
            connection_id=signed_context["connection_id"],
            query=user_message,
        ).model_dump()

        with tracer.start_as_current_span(
            "retrieval.workflow",
            record_exception=False,
            set_status_on_exception=False,
        ):
            result = await self.retrieval_workflow.ainvoke(
                graph_input,
                config={"configurable": {"thread_id": query_id}},
            )

        bot_response = result.get("response")

        await store_conversation_to_es(
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            bot_response=bot_response,
            scope_id=scope_id,
        )
        await update_session_stm(
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            bot_response=bot_response,
            scope_id=scope_id,
        )
        return {
            "session_id": session_id,
            "connection_id": signed_context["connection_id"],
            "query_id": query_id,
            "user_id": user_id,
            "response": result.get("response", ""),
            "intent": result.get("intent"),
            "sources": [
                {"file_name": chunk.get("file_name"), "chunk_id": chunk.get("chunk_id")}
                for chunk in result.get("retrieved_chunks", [])
            ],
        }

    
