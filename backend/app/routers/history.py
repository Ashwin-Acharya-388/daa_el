"""
history.py — Router for prediction history (paginated, filterable, exportable).

Uses in-memory state instead of a database.
"""

import io
import logging
import math
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException

from app.dependencies import rate_limiter
from app.schemas import (
    HistoryResponse,
    PredictionSummary,
    PredictionDetail,
)
from app import state

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/predictions", tags=["History"])


def _apply_filters(items, risk_level, min_confidence, date_from, date_to):
    """Apply optional filters to the in-memory prediction list."""
    filtered = items

    if risk_level == "high":
        filtered = [r for r in filtered if r["prediction"] == 1]
    elif risk_level == "low":
        filtered = [r for r in filtered if r["prediction"] == 0]

    if min_confidence is not None:
        filtered = [r for r in filtered if r["confidence"] >= min_confidence]

    if date_from:
        filtered = [r for r in filtered if r["created_at"] >= date_from]
    if date_to:
        filtered = [r for r in filtered if r["created_at"] <= date_to]

    return filtered


@router.get("/history", response_model=HistoryResponse, dependencies=[Depends(rate_limiter)])
async def list_predictions(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    risk_level: Optional[str] = Query(None, pattern="^(high|low)$"),
    min_confidence: Optional[float] = Query(None, ge=0, le=100),
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
):
    """Return paginated prediction history with optional filters."""
    filtered = _apply_filters(
        state.prediction_history, risk_level, min_confidence, date_from, date_to
    )
    total = len(filtered)

    # Paginate
    offset = (page - 1) * per_page
    page_items = filtered[offset : offset + per_page]

    items = [
        PredictionSummary(
            id=r["id"],
            probability=r["probability"],
            prediction=r["prediction"],
            decision=r["decision"],
            confidence=r["confidence"],
            has_shap=r.get("has_shap", False),
            created_at=r["created_at"],
        )
        for r in page_items
    ]

    return HistoryResponse(
        items=items,
        total=total,
        page=page,
        per_page=per_page,
        total_pages=math.ceil(total / per_page) if per_page else 0,
    )


@router.get("/{prediction_id}", response_model=PredictionDetail, dependencies=[Depends(rate_limiter)])
async def get_prediction_detail(
    prediction_id: str,
):
    """Return full details of a single prediction, including SHAP data."""
    import uuid as _uuid
    try:
        pid = _uuid.UUID(prediction_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid prediction ID")

    pred = state.predictions_details.get(pid)
    if pred is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return PredictionDetail(
        id=pred["id"],
        features_json=pred["features_json"],
        probability=pred["probability"],
        prediction=pred["prediction"],
        decision=pred["decision"],
        confidence=pred["confidence"],
        shap_json=pred.get("shap_json"),
        created_at=pred["created_at"],
    )


@router.get("/export/csv", dependencies=[Depends(rate_limiter)])
async def export_predictions(
    risk_level: Optional[str] = Query(None, pattern="^(high|low)$"),
    min_confidence: Optional[float] = Query(None, ge=0, le=100),
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
):
    """Export filtered predictions as a CSV download."""
    import pandas as pd
    from fastapi.responses import StreamingResponse

    filtered = _apply_filters(
        state.prediction_history, risk_level, min_confidence, date_from, date_to
    )

    # Limit to 10k rows
    filtered = filtered[:10000]

    records = [
        {
            "id": str(r["id"]),
            "probability": r["probability"],
            "prediction": r["prediction"],
            "decision": r["decision"],
            "confidence": r["confidence"],
            "created_at": r["created_at"].isoformat() if r.get("created_at") else "",
        }
        for r in filtered
    ]

    df = pd.DataFrame(records)
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions_export.csv"},
    )
