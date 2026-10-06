import json
from typing import Dict, List
from datetime import datetime, timedelta
# Updated: Import from redis.asyncio instead of standard redis
import redis.asyncio as aioredis

REDIS_URL =  "redis://redis:6379/0"

# Updated: Use the async client configuration
redis_client = aioredis.Redis.from_url(REDIS_URL, decode_responses=True)

SLIDING_WINDOW_SIZE = 10

# Updated to async def
async def get_stm(session_id: str, user_id: str = "12345"):
    key = f"stm:{user_id}:{session_id}"
    # Added await
    data = await redis_client.get(key)
    if data:
        return json.loads(data)
    return {
        "turn_number": 0,
        "topic": "",
        "context_summary": "",
        "entities": [],
        "last_user_message": "",
        "last_bot_message": ""
    }

# Updated to async def
async def save_stm(session_id: str, stm: dict, user_id: str = "12345"):
    key = f"stm:{user_id}:{session_id}"
    # Added await
    await redis_client.set(key, json.dumps(stm))

# Updated to async def
async def add_conversation(session_id: str, user_message: str, bot_message: str, user_id: str = "12345"):
    """
    Push a new conversation turn to Redis and maintain a sliding window of last N turns,
    including turn_number for proper sequencing.
    """
    # Added await since we are calling another async function
    last_turns = await get_last_n_conversations(session_id, user_id, n=1)
    previous_turn_number = last_turns[-1]["turn_number"] if last_turns else 0

    new_turn_number = previous_turn_number + 1

    turn = {
        "turn_number": new_turn_number,
        "user_message": user_message,
        "bot_message": bot_message,
    }
    key = f"stm_history:{user_id}:{session_id}"
    
    # Added await
    await redis_client.lpush(key, json.dumps(turn))
    
    # Added await
    await redis_client.ltrim(key, 0, SLIDING_WINDOW_SIZE - 1)

# Updated to async def
async def get_last_n_conversations(session_id: str, user_id: str = "12345", n: int = SLIDING_WINDOW_SIZE) -> List[Dict]:
    """
    Get last N conversation turns in chronological order (oldest first).
    """
    key = f"stm_history:{user_id}:{session_id}"

    # Added await
    raw_list = await redis_client.lrange(key, 0, n - 1)
    
    conversations = []
    for item in reversed(raw_list):
        try:
            conversations.append(json.loads(item))
        except json.JSONDecodeError:
            continue
    
    return conversations

# Updated to async def
async def get_stm_summary(session_id: str, user_id: str = "12345"):
    # Added await
    stm = await get_stm(session_id, user_id)
    return {
        "conversation_summary": stm.get("context_summary", ""),
        "conversation_entities": stm.get("entities", []),
        "last_user_message": stm.get("last_user_message", "")
    }
