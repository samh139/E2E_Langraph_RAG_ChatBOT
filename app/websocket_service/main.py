import asyncio
import os
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from dotenv import load_dotenv

from app.security.access_control import (
    effective_permissions,
    sign_internal_request,
    validate_access_token,
)
from app.websocket_service.kafka_consumer import consume_loop
from app.websocket_service.kafka_producer import send_message

load_dotenv(Path(__file__).resolve().parents[1] / "configs" / ".env")

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    async def connect(self, session_id: str, websocket: WebSocket, principal: dict):
        async with self._lock:
            if session_id in self.active_connections:
                await websocket.close(code=4409, reason="Session is already connected")
                return None
            connection_id = str(uuid.uuid4())
            await websocket.accept()
            self.active_connections[session_id] = {
                "websocket": websocket,
                "principal": principal,
                "connection_id": connection_id,
            }
            return connection_id

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)

    async def send(self, session_id: str, message: dict):
        connection = self.active_connections.get(session_id)
        if connection and connection["connection_id"] == message.get("connection_id"):
            await connection["websocket"].send_json(message)

manager = ConnectionManager()

## Start the Kafka consumer loop in the background
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(consume_loop(manager))

## websocket endpoint
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    origin = websocket.headers.get("origin", "").strip()
    allowed_origins = {
        item.strip()
        for item in os.getenv("WEBSOCKET_ALLOWED_ORIGINS", "").split(",")
        if item.strip()
    }
    if origin and origin not in allowed_origins:
        await websocket.close(code=4403, reason="Origin is not allowed")
        return

    authorization = websocket.headers.get("authorization", "")
    token = ""
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    if not token:
        token = websocket.cookies.get("access_token", "")
    if not token:
        await websocket.close(code=4401, reason="Authentication required")
        return
    if websocket.cookies.get("access_token") and not origin:
        await websocket.close(code=4403, reason="Cookie authentication requires an Origin")
        return

    try:
        principal = await asyncio.to_thread(validate_access_token, token)
        principal["allowed_roles"] = effective_permissions(principal["roles"])
        if not principal["allowed_roles"]:
            await websocket.close(code=4403, reason="No configured role permissions")
            return
    except Exception as exc:
        print(f"Rejected WebSocket authentication: {exc}")
        await websocket.close(code=4401, reason="Invalid authentication token")
        return

    connection_id = await manager.connect(session_id, websocket, principal)
    if connection_id is None:
        return
    try:
        while True:
            data = await websocket.receive_json()
            if not isinstance(data, dict) or not isinstance(data.get("message"), str):
                await websocket.send_json({"error": "Message must contain a text 'message' field."})
                continue
            message = data["message"].strip()
            if not message:
                await websocket.send_json({"error": "Message cannot be empty."})
                continue

            active = manager.active_connections.get(session_id)
            if not active or active["connection_id"] != connection_id:
                break
            if active["principal"]["expires_at"] <= int(time.time()):
                await websocket.close(code=4401, reason="Authentication token expired")
                break
            signed_context = sign_internal_request(
                session_id=session_id,
                connection_id=connection_id,
                message=message,
                principal=active["principal"],
            )
            send_message("chat-requests", key=session_id, value=signed_context)
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        print(f"Client disconnected: {session_id}")
    except Exception:
        manager.disconnect(session_id)
        raise

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
