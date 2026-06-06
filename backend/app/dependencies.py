"""
dependencies.py — FastAPI dependencies for auth and rate limiting.

All state is in-memory — no external services required.
"""

import time
from collections import defaultdict
from typing import Optional

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


# ── In-memory rate limiter (sliding window) ─────────────────────────

_rate_limit_store: dict[str, list[float]] = defaultdict(list)


async def rate_limiter(
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """
    Enforce per-key rate limiting using an in-memory sliding window.
    Raises 429 when limit is exceeded.
    """
    now = time.time()
    window = 60  # seconds

    # Prune old entries
    timestamps = _rate_limit_store[api_key]
    _rate_limit_store[api_key] = [t for t in timestamps if t > now - window]

    # Add current request
    _rate_limit_store[api_key].append(now)

    if len(_rate_limit_store[api_key]) > settings.RATE_LIMIT_PER_MINUTE:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Rate limit exceeded: {settings.RATE_LIMIT_PER_MINUTE} "
                f"requests per minute. Try again shortly."
            ),
        )
