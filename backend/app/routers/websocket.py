"""
websocket.py — WebSocket endpoint for real-time batch progress updates.

Polls the in-memory batch_jobs state store instead of Redis pub/sub.
"""

import asyncio
import json
import logging
import uuid as _uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.schemas import BatchStatus
from app import state

logger = logging.getLogger(__name__)
router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/batch/{task_id}")
async def batch_progress_ws(websocket: WebSocket, task_id: str):
    """
    WebSocket that streams batch processing progress updates.

    Polls the in-memory batch_jobs dict every 0.3 seconds and
    forwards status updates to the client.
    """
    await websocket.accept()
    logger.info("WebSocket connected for batch %s", task_id)

    try:
        task_uuid = _uuid.UUID(task_id)
    except ValueError:
        await websocket.send_text(json.dumps({"error": "Invalid task ID"}))
        await websocket.close()
        return

    last_processed = -1

    try:
        while True:
            job = state.batch_jobs.get(task_uuid)
            if job is None:
                await websocket.send_text(json.dumps({"error": "Job not found"}))
                break

            current_processed = job.get("processed_rows", 0)
            total = job.get("total_rows", 0)
            status = job.get("status", BatchStatus.PENDING)

            # Only send update when something changed
            if current_processed != last_processed or status in (
                BatchStatus.COMPLETED, BatchStatus.FAILED
            ):
                data = {
                    "status": status.value if isinstance(status, BatchStatus) else status,
                    "processed": current_processed,
                    "total": total,
                }

                if status == BatchStatus.FAILED:
                    data["error"] = job.get("error_message", "Unknown error")

                await websocket.send_text(json.dumps(data))
                last_processed = current_processed

                # If status is terminal, close
                if status in (BatchStatus.COMPLETED, BatchStatus.FAILED):
                    await asyncio.sleep(0.5)
                    break

            await asyncio.sleep(0.3)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for batch %s", task_id)
