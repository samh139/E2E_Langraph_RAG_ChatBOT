import asyncio
import argparse
import mimetypes
import os
from pathlib import Path

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.workflow.ingestion_state import IngestionState
from app.workflow.ingestion_workflow import IngestionWorkflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one file through the checkpointed ingestion graph.")
    parser.add_argument(
        "file",
        nargs="?",
        default="data/SBI_ATM_Charges.pdf",
        help="Path to a PDF, DOCX, or PPTX file (default: data/SBI_ATM_Charges.pdf)",
    )
    parser.add_argument(
        "--thread-id",
        help="LangGraph checkpoint thread ID. Defaults to a stable ID derived from the file path.",
    )
    return parser.parse_args()

async def main():
    args = parse_args()
    file_path = Path(args.file).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    file_type = suffix.lstrip(".")
    if file_type not in {"pdf", "docx", "pptx"}:
        raise ValueError(f"Unsupported file type {suffix!r}; expected .pdf, .docx, or .pptx")

    initial_state = IngestionState(
        file_path=str(file_path),
        file_name=file_path.name,
        file_extension=suffix,
        mime_type=mimetypes.guess_type(file_path.name)[0],
        file_type=file_type,
        status="pending",
    )

    # In Compose, "redis" is the Redis service hostname. Override REDIS_URL
    # when running this script directly on the host (for example localhost:6379).
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    thread_id = args.thread_id or f"ingestion-{file_path.stem}"
    config = {"configurable": {"thread_id": thread_id}}

    async with AsyncRedisSaver.from_conn_string(redis_url) as checkpointer:
        # Create the Redis indices required by the checkpointer before graph use.
        await checkpointer.setup()
        workflow_app = IngestionWorkflow(checkpointer=checkpointer).compile()

        print(f"Starting ingestion for {file_path.name} (thread_id={thread_id})...")
        final_state = await workflow_app.ainvoke(initial_state, config=config)
        print("Workflow finished successfully!")
        print("Status:", final_state.get("status"))
        print("Chunks:", len(final_state.get("chunks") or []))
        print("Stage timings (ms):", final_state.get("stage_timings_ms", {}))

if __name__ == "__main__":
    asyncio.run(main())
