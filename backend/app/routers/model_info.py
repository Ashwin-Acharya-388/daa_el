"""
model_info.py — Router for model metadata and health.
"""

import logging
from fastapi import APIRouter, Depends

from app.dependencies import rate_limiter
from app.schemas import ModelInfoResponse
from app.ml.model_loader import (
    get_feature_names,
    get_metrics,
    get_metadata,
    is_model_healthy,
)
from app.config import settings
from app import state

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/model", tags=["Model Info"])


@router.get("/info", response_model=ModelInfoResponse, dependencies=[Depends(rate_limiter)])
async def model_info():
    """Return model metadata, performance metrics, and health status."""
    feature_names = get_feature_names()
    metrics = get_metrics()
    metadata = get_metadata()

    # Count total predictions from in-memory store
    total_predictions = state.total_predictions

    healthy = is_model_healthy()

    # Extract feature selection info from metadata
    fs_info = metadata.get("feature_selection", {})

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
        feature_selection={
            "algorithm": fs_info.get("algorithm", "SFFS"),
            "n_features_selected": fs_info.get("n_features_selected", len(feature_names)),
            "final_auc": fs_info.get("final_auc", 0),
            "total_evaluations": fs_info.get("total_evaluations", 0),
        },
        training_medians=metadata.get("training_medians", {}),
    )
