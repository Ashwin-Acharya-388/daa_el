"""
main.py — FastAPI application entry point.

Starts the Financial Risk Analysis API with:
  - Model pre-loading at startup
  - Database table creation
  - CORS for the React frontend
  - All API routers mounted
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.ml.model_loader import load_model

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan events ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    # STARTUP
    logger.info("=" * 60)
    logger.info("  Financial Risk Analysis API — Starting")
    logger.info("=" * 60)

    # Pre-load ML model
    load_model()
    logger.info("Model loaded into memory")

    yield

    # SHUTDOWN
    logger.info("Shutdown complete")


# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production API for DenseNet-based financial risk prediction "
        "with SHAP explainability and in-memory state management."
    ),
    lifespan=lifespan,
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ──
from app.routers import predict, explain, model_info, history, websocket  # noqa: E402

app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(model_info.router)
app.include_router(history.router)
app.include_router(websocket.router)


# ── Root health check ──
@app.get("/", tags=["Health"])
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    from app.ml.model_loader import is_model_healthy
    return {
        "status": "healthy" if is_model_healthy() else "unhealthy",
        "model_loaded": is_model_healthy(),
    }
