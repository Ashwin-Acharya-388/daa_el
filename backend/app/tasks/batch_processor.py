"""
batch_processor.py — Background task for processing batch CSV predictions.

Runs inside FastAPI BackgroundTasks.  Updates the batch_jobs table with
progress and publishes updates to a Redis pub/sub channel so the
WebSocket endpoint can relay them to the frontend.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import redis as sync_redis

from app.config import settings, BATCH_RESULTS_DIR
from app.ml.predictor import predict_batch_sync

logger = logging.getLogger(__name__)


def process_batch(task_id: str, input_csv_path: str):
    """
    Synchronous batch processing function — called by BackgroundTasks.

    Reads the uploaded CSV, runs predictions in chunks with progress
    updates, and saves the result CSV.  All DB and Redis operations
    use synchronous clients (because BackgroundTasks runs in a thread).
    """
    # ── Synchronous Redis for pub/sub ──
    try:
        r = sync_redis.from_url(settings.REDIS_URL, decode_responses=True)
    except Exception:
        r = None

    channel = f"batch_progress:{task_id}"

    def publish(data: dict):
        if r:
            try:
                r.publish(channel, json.dumps(data))
            except Exception:
                pass

    # ── Synchronous DB access ──
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    sync_url = settings.DATABASE_URL.replace("+asyncpg", "")
    engine = create_engine(sync_url)

    from app.models import BatchJob, Prediction, BatchStatus

    try:
        # Update status → PROCESSING
        with Session(engine) as session:
            import uuid as _uuid
            job = session.get(BatchJob, _uuid.UUID(task_id))
            if job is None:
                logger.error("Batch job %s not found in DB", task_id)
                return
            job.status = BatchStatus.PROCESSING
            session.commit()

        publish({"status": "PROCESSING", "processed": 0, "total": 0})

        # Read input CSV
        df = pd.read_csv(input_csv_path)
        total = len(df)

        with Session(engine) as session:
            job = session.get(BatchJob, _uuid.UUID(task_id))
            job.total_rows = total
            session.commit()

        def on_progress(processed, total_count):
            with Session(engine) as session:
                job = session.get(BatchJob, _uuid.UUID(task_id))
                job.processed_rows = processed
                session.commit()
            publish({
                "status": "PROCESSING",
                "processed": processed,
                "total": total_count,
            })

        # Run predictions
        result_df = predict_batch_sync(df, progress_callback=on_progress)

        # Save result CSV
        output_path = BATCH_RESULTS_DIR / f"{task_id}_result.csv"
        result_df.to_csv(str(output_path), index=False)

        # Store individual predictions in DB
        with Session(engine) as session:
            import uuid as _uuid
            for _, row in result_df.iterrows():
                pred = Prediction(
                    features_json={},  # Batch predictions don't store full features
                    prediction=int(row.get("predicted_class", 0)),
                    probability=float(row.get("default_probability", 0)),
                    decision=str(row.get("decision", "")),
                    confidence=round(
                        float(row.get("default_probability", 0))
                        if row.get("predicted_class", 0) == 1
                        else (1 - float(row.get("default_probability", 0))),
                        1
                    ) * 100,
                    batch_id=_uuid.UUID(task_id),
                )
                session.add(pred)
            session.commit()

        # Mark job complete
        with Session(engine) as session:
            job = session.get(BatchJob, _uuid.UUID(task_id))
            job.status = BatchStatus.COMPLETED
            job.processed_rows = total
            job.result_path = str(output_path)
            job.completed_at = datetime.now(timezone.utc)
            session.commit()

        publish({
            "status": "COMPLETED",
            "processed": total,
            "total": total,
        })

        logger.info("Batch %s completed — %d rows processed", task_id, total)

    except Exception as e:
        logger.exception("Batch %s FAILED: %s", task_id, e)

        try:
            with Session(engine) as session:
                import uuid as _uuid
                job = session.get(BatchJob, _uuid.UUID(task_id))
                if job:
                    job.status = BatchStatus.FAILED
                    job.error_message = str(e)
                    session.commit()
        except Exception:
            pass

        publish({"status": "FAILED", "error": str(e)})

    finally:
        if r:
            r.close()
        engine.dispose()
