"""
models.py — SQLAlchemy ORM models for predictions and batch jobs.
"""

import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text, Enum, ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.database import Base


class BatchStatus(str, PyEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    features_json = Column(JSONB, nullable=False)
    prediction = Column(Integer, nullable=False)          # 0 or 1
    probability = Column(Float, nullable=False)            # 0.0–1.0
    decision = Column(String(50), nullable=False)          # "DEFAULT / HIGH RISK" etc.
    confidence = Column(Float, nullable=False)             # 0–100 %
    shap_json = Column(JSONB, nullable=True)               # full SHAP explanation
    batch_id = Column(
        UUID(as_uuid=True),
        ForeignKey("batch_jobs.id", ondelete="SET NULL"),
        nullable=True,
    )
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    batch_job = relationship("BatchJob", back_populates="predictions")

    def __repr__(self):
        return (
            f"<Prediction {self.id} prob={self.probability:.4f} "
            f"decision={self.decision}>"
        )


class BatchJob(Base):
    __tablename__ = "batch_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    total_rows = Column(Integer, nullable=False, default=0)
    processed_rows = Column(Integer, nullable=False, default=0)
    status = Column(
        Enum(BatchStatus, name="batch_status"),
        nullable=False,
        default=BatchStatus.PENDING,
    )
    error_message = Column(Text, nullable=True)
    result_path = Column(String(500), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)

    predictions = relationship(
        "Prediction", back_populates="batch_job", cascade="all, delete-orphan"
    )

    def __repr__(self):
        return (
            f"<BatchJob {self.id} status={self.status} "
            f"{self.processed_rows}/{self.total_rows}>"
        )
