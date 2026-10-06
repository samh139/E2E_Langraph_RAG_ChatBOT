import json

from app.configs.llm_config import (
    fire_fast_modal_request_chat
)

from .store import (
    get_stm,
    save_stm,
    add_conversation,
)


STM_SYSTEM_PROMPT = """
You are a conversation memory summarization agent.

Your job is to maintain a concise summary of the ENTIRE conversation
for the current session.

You will receive:
1. The existing session summary.
2. The latest user message.
3. The latest assistant response.

Update the existing summary using the new turn.

IMPORTANT:
- Preserve important facts, decisions, preferences, entities, and context
  from the existing summary.
- Do not forget information just because it is old.
- Add important information from the latest turn.
- Remove irrelevant conversational noise.
- Do not invent information.
- Keep the summary concise.
- This summary represents the complete conversation history of the session.

Return ONLY valid JSON in this format:

{
    "topic": "main topic of the conversation",
    "context_summary": "concise summary of the entire conversation",
    "entities": [],
    "last_user_message": "",
    "last_bot_message": ""
}
"""


async def update_session_stm(
    user_id: str,
    session_id: str,
    user_message: str,
    bot_response: str,
):
    """
    Update the session-level STM using the previous STM summary
    and the latest conversation turn.
    """

    # Get the existing summary for this session.
    previous_stm = await get_stm(
        session_id=session_id,
        user_id=user_id,
    )

    if not previous_stm:
        previous_stm = {
            "topic": "",
            "context_summary": "",
            "entities": [],
            "last_user_message": "",
            "last_bot_message": "",
        }

    user_prompt = f"""
EXISTING SESSION SUMMARY:
{previous_stm}

LATEST USER MESSAGE:
{user_message}

LATEST ASSISTANT RESPONSE:
{bot_response}

Update the existing session summary with this latest turn.
Remember that the updated summary must represent the ENTIRE session.
"""

    updated_stm_text = await fire_fast_modal_request_chat(
        system_prompt=STM_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    # Persist the fields consumed by get_stm_summary on subsequent requests.
    try:
        updated_stm = json.loads(updated_stm_text)
        if not isinstance(updated_stm, dict):
            raise ValueError("STM response must be a JSON object")
    except (json.JSONDecodeError, TypeError, ValueError):
        updated_stm = previous_stm.copy()

    updated_stm.update(
        {
            "topic": updated_stm.get("topic", previous_stm.get("topic", "")),
            "context_summary": updated_stm.get(
                "context_summary", previous_stm.get("context_summary", "")
            ),
            "entities": updated_stm.get("entities", previous_stm.get("entities", [])),
            "last_user_message": user_message,
            "last_bot_message": bot_response,
        }
    )

    await save_stm(
        session_id=session_id,
        stm=updated_stm,
        user_id=user_id,
    )

    # Keep the individual turn history as well.
    # This is useful for debugging/auditing and does NOT define STM.
    await add_conversation(
        session_id=session_id,
        user_message=user_message,
        bot_message=bot_response,
        user_id=user_id,
    )

    return updated_stm
