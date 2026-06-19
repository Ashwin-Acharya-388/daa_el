"""
predict.py — Router for single predictions.

Endpoints:
  POST /api/predict              — single prediction
"""

import uuid
import logging
from datetime import datetime, timezone

from fastapi import (
    APIRouter,
    Depends,
)

from app.dependencies import rate_limiter
from app.schemas import (
    PredictionRequest,
    PredictionResponse,
)
from app.ml.predictor import predict_single
from app.ml.explainer import compute_explanation
from app import state

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/predict", tags=["Predictions"])


# ────────────── Single Prediction ───────────────────────────────────

@router.post("", response_model=PredictionResponse, dependencies=[Depends(rate_limiter)])
async def create_prediction(
    body: PredictionRequest,
):
    """Run a single prediction through the DenseNet model."""
    # 1. Predict
    result = await predict_single(body.features)

    # 2. SHAP explanation (optional)
    shap_json = None
    explanation_out = None
    if body.include_shap:
        try:
            shap_json = await compute_explanation(body.features)
            from app.schemas import ShapExplanation, ShapFactor
            explanation_out = ShapExplanation(
                base_probability=shap_json["base_probability"],
                top_factors_toward_default=[
                    ShapFactor(**f) for f in shap_json["top_factors_toward_default"]
                ],
                top_factors_toward_healthy=[
                    ShapFactor(**f) for f in shap_json["top_factors_toward_healthy"]
                ],
                summary=shap_json["summary"],
            )
        except Exception as e:
            logger.warning("SHAP computation failed: %s", e)

    # 3. Persist to in-memory store
    record_id = uuid.uuid4()
    created_at = datetime.now(timezone.utc)

    # Full detail
    state.predictions_details[record_id] = {
        "id": record_id,
        "features_json": body.features,
        "prediction": result["prediction"],
        "probability": result["probability_of_default"],
        "decision": result["decision"],
        "confidence": result["confidence"],
        "shap_json": shap_json,
        "created_at": created_at,
    }

    # Summary for history list
    state.prediction_history.insert(0, {
        "id": record_id,
        "probability": result["probability_of_default"],
        "prediction": result["prediction"],
        "decision": result["decision"],
        "confidence": result["confidence"],
        "has_shap": shap_json is not None,
        "created_at": created_at,
    })

    state.total_predictions += 1

    return PredictionResponse(
        id=record_id,
        probability_of_default=result["probability_of_default"],
        prediction=result["prediction"],
        decision=result["decision"],
        confidence=result["confidence"],
        explanation=explanation_out,
        created_at=created_at,
    )

