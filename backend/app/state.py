"""
state.py — In-memory data store for predictions and batch jobs.

Replaces PostgreSQL persistence with simple Python data structures.
All data lives in-process and is lost on server restart.
"""

import uuid
from typing import Any

# Global in-memory storage
prediction_history: list[dict[str, Any]] = []  # List of summary dicts (newest first)
predictions_details: dict[uuid.UUID, dict[str, Any]] = {}  # UUID -> full detail dict
batch_jobs: dict[uuid.UUID, dict[str, Any]] = {}  # UUID -> batch job info
total_predictions: int = 0
