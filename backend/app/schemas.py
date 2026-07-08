"""
schemas.py — Pydantic models for request/response validation.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums (moved from models.py) ────────────────────────────────────

class BatchStatus(str, PyEnum):
    """Status of a batch prediction job."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ────────────────── Prediction ──────────────────────────────────────

class PredictionRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Feature-name → numeric-value mapping. Missing features are filled with training medians.",
        examples=[{"x1": 1.2, "x2": 0.11, "x3": 0.63}],
    )
    include_shap: bool = Field(
        default=True,
        description="Whether to compute a SHAP explanation (slower).",
    )


class ShapFactor(BaseModel):
    feature: str
    shap_value: float
    feature_value: float


class ShapExplanation(BaseModel):
    base_probability: float
    top_factors_toward_default: list[ShapFactor]
    top_factors_toward_healthy: list[ShapFactor]
    summary: str


class PredictionResponse(BaseModel):
    id: uuid.UUID
    probability_of_default: float
    prediction: int
    decision: str
    confidence: float
    explanation: Optional[ShapExplanation] = None
    created_at: datetime


# ────────────────── Batch ───────────────────────────────────────────

class BatchUploadResponse(BaseModel):
    task_id: uuid.UUID
    filename: str
    total_rows: int
    status: str


class BatchStatusResponse(BaseModel):
    task_id: uuid.UUID
    filename: str
    status: str
    total_rows: int
    processed_rows: int
    progress_percent: float
    result_download_url: Optional[str] = None
    error_message: Optional[str] = None


# ────────────────── Model Info ──────────────────────────────────────

class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    feature_count: int
    feature_names: list[str]
    metrics: dict[str, Any]
    total_predictions: int
    model_status: str
    architecture: dict[str, Any]
    feature_selection: dict[str, Any] = {}
    training_medians: dict[str, float] = {}
    training_stds: dict[str, float] = {}
    training_mins: dict[str, float] = {}
    training_maxs: dict[str, float] = {}


# ────────────────── Explain ─────────────────────────────────────────

class ExplainRequest(BaseModel):
    features: dict[str, float] = Field(
        ...,
        description="Feature-name → numeric-value mapping.",
    )


class ExplainResponse(BaseModel):
    prediction: dict[str, Any]
    explanation: ShapExplanation


# ────────────────── History ─────────────────────────────────────────

class PredictionSummary(BaseModel):
    id: uuid.UUID
    probability: float
    prediction: int
    decision: str
    confidence: float
    has_shap: bool
    created_at: datetime


class HistoryResponse(BaseModel):
    items: list[PredictionSummary]
    total: int
    page: int
    per_page: int
    total_pages: int


class PredictionDetail(BaseModel):
    id: uuid.UUID
    features_json: dict[str, Any]
    probability: float
    prediction: int
    decision: str
    confidence: float
    shap_json: Optional[dict[str, Any]] = None
    created_at: datetime
