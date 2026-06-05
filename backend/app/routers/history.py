"""
history.py — Router for prediction history (paginated, filterable, exportable).
"""

import io
import logging
import math
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import rate_limiter
from app.models import Prediction
from app.schemas import (
    HistoryResponse,
    PredictionSummary,
    PredictionDetail,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/predictions", tags=["History"])


def _apply_filters(query, risk_level, min_confidence, date_from, date_to):
    """Apply optional filters to the prediction query."""
    if risk_level == "high":
        query = query.where(Prediction.prediction == 1)
    elif risk_level == "low":
        query = query.where(Prediction.prediction == 0)

    if min_confidence is not None:
        query = query.where(Prediction.confidence >= min_confidence)

    if date_from:
        query = query.where(Prediction.created_at >= date_from)
    if date_to:
        query = query.where(Prediction.created_at <= date_to)

    return query


@router.get("/history", response_model=HistoryResponse, dependencies=[Depends(rate_limiter)])
async def list_predictions(
    db: AsyncSession = Depends(get_db),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    risk_level: Optional[str] = Query(None, regex="^(high|low)$"),
    min_confidence: Optional[float] = Query(None, ge=0, le=100),
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
):
    """Return paginated prediction history with optional filters."""
    # Count total
    count_q = select(func.count(Prediction.id))
    count_q = _apply_filters(count_q, risk_level, min_confidence, date_from, date_to)
    total = (await db.execute(count_q)).scalar() or 0

    # Fetch page
    offset = (page - 1) * per_page
    items_q = (
        select(Prediction)
        .order_by(desc(Prediction.created_at))
        .offset(offset)
        .limit(per_page)
    )
    items_q = _apply_filters(items_q, risk_level, min_confidence, date_from, date_to)
    result = await db.execute(items_q)
    rows = result.scalars().all()

    items = [
        PredictionSummary(
            id=r.id,
            probability=r.probability,
            prediction=r.prediction,
            decision=r.decision,
            confidence=r.confidence,
            has_shap=r.shap_json is not None,
            created_at=r.created_at,
        )
        for r in rows
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
    db: AsyncSession = Depends(get_db),
):
    """Return full details of a single prediction, including SHAP data."""
    import uuid as _uuid
    try:
        pid = _uuid.UUID(prediction_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid prediction ID")

    result = await db.execute(
        select(Prediction).where(Prediction.id == pid)
    )
    pred = result.scalar_one_or_none()
    if pred is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return PredictionDetail(
        id=pred.id,
        features_json=pred.features_json,
        probability=pred.probability,
        prediction=pred.prediction,
        decision=pred.decision,
        confidence=pred.confidence,
        shap_json=pred.shap_json,
        created_at=pred.created_at,
    )


@router.get("/export/csv", dependencies=[Depends(rate_limiter)])
async def export_predictions(
    db: AsyncSession = Depends(get_db),
    risk_level: Optional[str] = Query(None, regex="^(high|low)$"),
    min_confidence: Optional[float] = Query(None, ge=0, le=100),
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
):
    """Export filtered predictions as a CSV download."""
    import pandas as pd

    query = select(Prediction).order_by(desc(Prediction.created_at)).limit(10000)
    query = _apply_filters(query, risk_level, min_confidence, date_from, date_to)
    result = await db.execute(query)
    rows = result.scalars().all()

    records = [
        {
            "id": str(r.id),
            "probability": r.probability,
            "prediction": r.prediction,
            "decision": r.decision,
            "confidence": r.confidence,
            "created_at": r.created_at.isoformat() if r.created_at else "",
        }
        for r in rows
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
