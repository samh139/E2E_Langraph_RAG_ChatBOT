from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
import asyncio
from app.websocket_service.kafka_consumer import consume_loop
from app.websocket_service.kafka_producer import send_message

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)

    async def send(self, session_id: str, message: dict):
        websocket = self.active_connections.get(session_id)
        if websocket:
            await websocket.send_json(message)

manager = ConnectionManager()

## Start the Kafka consumer loop in the background
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(consume_loop(manager))

## websocket endpoint
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(session_id, websocket)
    try:
        while True:
            data = await websocket.receive_json()
            # Send the received message to Kafka
            send_message("chat-requests", key=session_id, value  = {
                "session_id": session_id,
                "message": data["message"]
            })
    except WebSocketDisconnect:
        manager.disconnect(session_id)
        print(f"Client disconnected: {session_id}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)