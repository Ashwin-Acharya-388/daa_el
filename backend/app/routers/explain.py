"""
explain.py — Router for SHAP explanations.
"""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import rate_limiter, get_redis
from app.schemas import ExplainRequest, ExplainResponse, ShapExplanation, ShapFactor
from app.ml.explainer import compute_explanation

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/explain", tags=["Explainability"])


@router.post("", response_model=ExplainResponse, dependencies=[Depends(rate_limiter)])
async def explain_prediction(
    body: ExplainRequest,
    redis=Depends(get_redis),
):
    """Compute a SHAP explanation for a given set of features."""
    shap_result = await compute_explanation(body.features, redis_client=redis)

    explanation = ShapExplanation(
        base_probability=shap_result["base_probability"],
        top_factors_toward_default=[
            ShapFactor(**f) for f in shap_result["top_factors_toward_default"]
        ],
        top_factors_toward_healthy=[
            ShapFactor(**f) for f in shap_result["top_factors_toward_healthy"]
        ],
        summary=shap_result["summary"],
    )

    return ExplainResponse(
        prediction=shap_result["prediction"],
        explanation=explanation,
    )
