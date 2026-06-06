"""
batch_processor.py — Background task for processing batch CSV predictions.

Runs inside FastAPI BackgroundTasks.  Updates the in-memory batch_jobs
store with progress so the WebSocket endpoint can relay updates to the
frontend.  Includes per-row error resilience and memory-efficient processing.
"""

import logging
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.config import BATCH_RESULTS_DIR
from app.ml.predictor import predict_batch_sync
from app.schemas import BatchStatus
from app import state

logger = logging.getLogger(__name__)

# Maximum CSV file size to process (50 MB)
MAX_CSV_SIZE_BYTES = 50 * 1024 * 1024


def process_batch(task_id: str, input_csv_path: str):
    """
    Synchronous batch processing function — called by BackgroundTasks.

    Reads the uploaded CSV, runs predictions in chunks with progress
    updates, and saves the result CSV.  All state is stored in-memory.
    """
    task_uuid = _uuid.UUID(task_id)

    try:
        job = state.batch_jobs.get(task_uuid)
        if job is None:
            logger.error("Batch job %s not found in state", task_id)
            return

        # Check file size
        input_path = Path(input_csv_path)
        if input_path.stat().st_size > MAX_CSV_SIZE_BYTES:
            job["status"] = BatchStatus.FAILED
            job["error_message"] = (
                f"File too large ({input_path.stat().st_size / 1024 / 1024:.1f} MB). "
                f"Maximum allowed: {MAX_CSV_SIZE_BYTES / 1024 / 1024:.0f} MB."
            )
            logger.warning("Batch %s rejected: file too large", task_id)
            return

        # Update status → PROCESSING
        job["status"] = BatchStatus.PROCESSING

        # Read input CSV
        try:
            df = pd.read_csv(input_csv_path)
        except Exception as e:
            job["status"] = BatchStatus.FAILED
            job["error_message"] = f"Failed to parse CSV: {e}"
            logger.error("Batch %s CSV parse failed: %s", task_id, e)
            return

        if df.empty:
            job["status"] = BatchStatus.FAILED
            job["error_message"] = "CSV file is empty"
            return

        total = len(df)
        job["total_rows"] = total

        def on_progress(processed, total_count):
            job["processed_rows"] = processed

        # Run predictions
        result_df = predict_batch_sync(df, progress_callback=on_progress)

        # Save result CSV
        output_path = BATCH_RESULTS_DIR / f"{task_id}_result.csv"
        result_df.to_csv(str(output_path), index=False)

        # Mark job complete
        job["status"] = BatchStatus.COMPLETED
        job["processed_rows"] = total
        job["result_path"] = str(output_path)
        job["completed_at"] = datetime.now(timezone.utc)

        logger.info("Batch %s completed — %d rows processed", task_id, total)

    except Exception as e:
        logger.exception("Batch %s FAILED: %s", task_id, e)

        try:
            job = state.batch_jobs.get(task_uuid)
            if job:
                job["status"] = BatchStatus.FAILED
                job["error_message"] = str(e)
        except Exception:
            pass
