"""
websocket.py — WebSocket endpoint for real-time batch progress updates.
"""

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.dependencies import get_redis

logger = logging.getLogger(__name__)
router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/batch/{task_id}")
async def batch_progress_ws(websocket: WebSocket, task_id: str):
    """
    WebSocket that streams batch processing progress updates.

    The background task publishes messages to a Redis channel
    `batch_progress:{task_id}`.  This WebSocket subscribes and
    forwards every message to the client.
    """
    await websocket.accept()
    logger.info("WebSocket connected for batch %s", task_id)

    redis = await get_redis()
    pubsub = redis.pubsub()
    channel = f"batch_progress:{task_id}"
    await pubsub.subscribe(channel)

    try:
        while True:
            message = await pubsub.get_message(
                ignore_subscribe_messages=True, timeout=1.0
            )
            if message and message["type"] == "message":
                data = message["data"]
                await websocket.send_text(data)

                # If status is terminal, close
                try:
                    parsed = json.loads(data)
                    if parsed.get("status") in ("COMPLETED", "FAILED"):
                        await asyncio.sleep(0.5)
                        break
                except (json.JSONDecodeError, TypeError):
                    pass

            await asyncio.sleep(0.3)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for batch %s", task_id)
    finally:
        await pubsub.unsubscribe(channel)
        await pubsub.close()
