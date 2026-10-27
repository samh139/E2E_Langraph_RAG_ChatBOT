"""Locust scenario for the authenticated, asynchronous chat WebSocket."""

from __future__ import annotations

import json
import os
import time
import uuid

from gevent import monkey

monkey.patch_all()

import websocket
from locust import User, between, events, task
from locust.exception import StopUser


class ChatWebSocketUser(User):
    """One authenticated WebSocket session per simulated user."""

    wait_time = between(1, 3)

    def on_start(self) -> None:
        token = os.getenv("LOCUST_JWT", "").strip()
        if not token:
            raise StopUser("Set LOCUST_JWT to a valid application JWT.")

        self.session_id = f"locust-{uuid.uuid4()}"
        self.url = os.getenv("LOCUST_WS_URL", "ws://localhost:8000").rstrip("/")
        self.socket = None
        try:
            self.socket = websocket.create_connection(
                f"{self.url}/ws/{self.session_id}",
                header=[f"Authorization: Bearer {token}"],
                timeout=float(os.getenv("LOCUST_RESPONSE_TIMEOUT_SECONDS", "120")),
                enable_multithread=True,
            )
        except Exception as exc:
            events.request.fire(
                request_type="WS",
                name="connect",
                response_time=0,
                response_length=0,
                exception=exc,
                context={},
            )
            raise StopUser() from exc
        events.request.fire(
            request_type="WS",
            name="connect",
            response_time=0,
            response_length=0,
            exception=None,
            context={},
        )

    @task
    def send_chat_message(self) -> None:
        question = os.getenv("LOCUST_QUESTION", "What are SBI NEFT charges?")
        started = time.perf_counter()
        response_length = 0
        error = None
        try:
            self.socket.send(json.dumps({"message": question}))
            payload = self.socket.recv()
            response_length = len(payload)
            response = json.loads(payload)
            if response.get("session_id") != self.session_id:
                raise RuntimeError("Received a response for a different session.")
            if response.get("error") or not response.get("response"):
                raise RuntimeError(response.get("error") or "Chat response was empty.")
        except Exception as exc:
            error = exc
        events.request.fire(
            request_type="WS",
            name="chat response",
            response_time=(time.perf_counter() - started) * 1000,
            response_length=response_length,
            exception=error,
            context={},
        )

    def on_stop(self) -> None:
        if self.socket is not None:
            self.socket.close()
