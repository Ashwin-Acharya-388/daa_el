"""
dependencies.py — FastAPI dependencies for auth, rate limiting, and Redis.
"""

import time
from typing import Optional

import redis.asyncio as aioredis
from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.config import settings

# ── API-key security scheme ─────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> str:
    """
    Validate the X-API-Key header. Returns the key on success.
    Raises 401 if missing / invalid.
    """
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# ── Redis connection pool ───────────────────────────────────────────

_redis_pool: Optional[aioredis.Redis] = None


async def get_redis() -> aioredis.Redis:
    """Return a shared async Redis client."""
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            max_connections=20,
        )
    return _redis_pool


async def close_redis():
    """Shutdown Redis pool — call on app shutdown."""
    global _redis_pool
    if _redis_pool is not None:
        await _redis_pool.close()
        _redis_pool = None


# ── Rate limiter (sliding window via Redis) ─────────────────────────

async def rate_limiter(
    request: Request,
    api_key: str = Depends(verify_api_key),
    redis_client: aioredis.Redis = Depends(get_redis),
):
    """
    Enforce per-key rate limiting using a Redis sorted-set sliding window.
    Raises 429 when limit is exceeded.
    """
    key = f"rate_limit:{api_key}"
    now = time.time()
    window = 60  # seconds

    pipe = redis_client.pipeline()
    pipe.zremrangebyscore(key, 0, now - window)  # prune old entries
    pipe.zadd(key, {str(now): now})               # add current
    pipe.zcard(key)                                # count in window
    pipe.expire(key, window + 1)                   # auto-expire
    results = await pipe.execute()

    count = results[2]
    if count > settings.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: {settings.RATE_LIMIT_PER_MINUTE} "
                f"requests per minute. Try again shortly."
            ),
        )
