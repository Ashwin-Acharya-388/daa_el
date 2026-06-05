"""
model_info.py — Router for model metadata and health.
"""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import rate_limiter
from app.models import Prediction
from app.schemas import ModelInfoResponse
from app.ml.model_loader import (
    get_feature_names,
    get_metrics,
    is_model_healthy,
)
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/model", tags=["Model Info"])


@router.get("/info", response_model=ModelInfoResponse, dependencies=[Depends(rate_limiter)])
async def model_info(db: AsyncSession = Depends(get_db)):
    """Return model metadata, performance metrics, and health status."""
    feature_names = get_feature_names()
    metrics = get_metrics()

    # Count total predictions in DB
    result = await db.execute(select(func.count(Prediction.id)))
    total_predictions = result.scalar() or 0

    healthy = is_model_healthy()

    return ModelInfoResponse(
        model_name="DenseNet Tabular — Financial Risk",
        version=settings.APP_VERSION,
        feature_count=len(feature_names),
        feature_names=feature_names,
        metrics=metrics,
        total_predictions=total_predictions,
        model_status="healthy" if healthy else "unhealthy",
        architecture={
            "type": "DenseNet (adapted for tabular data)",
            "dense_blocks": 3,
            "layers_per_block": 4,
            "growth_rate": 32,
            "compression": 0.5,
            "dropout": 0.3,
            "classifier_head": "Dense(128) → Dense(64) → Sigmoid",
        },
    )
