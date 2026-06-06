"""
predict.py — Router for single and batch predictions.

Endpoints:
  POST /api/predict              — single prediction
  POST /api/predict/batch        — CSV upload → background task
  GET  /api/predict/batch/{id}   — check batch status
  GET  /api/predict/batch/{id}/download — download result CSV
"""

import uuid
import io
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse

from app.config import settings, BATCH_RESULTS_DIR
from app.dependencies import rate_limiter
from app.schemas import (
    BatchStatus,
    PredictionRequest,
    PredictionResponse,
    BatchUploadResponse,
    BatchStatusResponse,
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


# ────────────── Batch Upload ────────────────────────────────────────

@router.post("/batch", response_model=BatchUploadResponse, dependencies=[Depends(rate_limiter)])
async def upload_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a CSV file for batch prediction. Processing runs in the background."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    # Read file
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    # Save uploaded CSV
    task_id = uuid.uuid4()
    upload_path = BATCH_RESULTS_DIR / f"{task_id}_input.csv"
    with open(upload_path, "wb") as f:
        f.write(contents)

    # Create batch job record in memory
    state.batch_jobs[task_id] = {
        "id": task_id,
        "filename": file.filename,
        "total_rows": len(df),
        "processed_rows": 0,
        "status": BatchStatus.PENDING,
        "error_message": None,
        "result_path": None,
        "created_at": datetime.now(timezone.utc),
        "completed_at": None,
    }

    # Schedule background processing
    from app.tasks.batch_processor import process_batch
    background_tasks.add_task(process_batch, str(task_id), str(upload_path))

    return BatchUploadResponse(
        task_id=task_id,
        filename=file.filename,
        total_rows=len(df),
        status="PENDING",
    )


# ────────────── Batch Status ────────────────────────────────────────

@router.get("/batch/{task_id}", response_model=BatchStatusResponse, dependencies=[Depends(rate_limiter)])
async def get_batch_status(
    task_id: uuid.UUID,
):
    """Check the status of a batch prediction job."""
    job = state.batch_jobs.get(task_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Batch job not found")

    progress = (
        round(job["processed_rows"] / job["total_rows"] * 100, 1)
        if job["total_rows"] > 0 else 0.0
    )

    download_url = None
    if job["status"] == BatchStatus.COMPLETED and job["result_path"]:
        download_url = f"/api/predict/batch/{task_id}/download"

    return BatchStatusResponse(
        task_id=job["id"],
        filename=job["filename"],
        status=job["status"].value,
        total_rows=job["total_rows"],
        processed_rows=job["processed_rows"],
        progress_percent=progress,
        result_download_url=download_url,
        error_message=job["error_message"],
    )


# ────────────── Batch Download ──────────────────────────────────────

@router.get("/batch/{task_id}/download", dependencies=[Depends(rate_limiter)])
async def download_batch_result(
    task_id: uuid.UUID,
):
    """Download the result CSV of a completed batch job."""
    job = state.batch_jobs.get(task_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Batch job not found")
    if job["status"] != BatchStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Batch job not yet completed")
    if not job["result_path"] or not Path(job["result_path"]).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(
        path=job["result_path"],
        filename=f"predictions_{job['filename']}",
        media_type="text/csv",
    )
