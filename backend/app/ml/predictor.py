"""
predictor.py — Single and batch prediction logic.

Wraps the loaded DenseNet model for inference.  Supports Redis caching
for identical inputs.
"""

import hashlib
import json
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from app.ml.model_loader import (
    get_model,
    get_scaler,
    get_feature_names,
    get_training_medians,
)

logger = logging.getLogger(__name__)


def _cache_key(features: dict) -> str:
    """Deterministic hash of feature values for caching."""
    canon = json.dumps(features, sort_keys=True)
    return f"pred:{hashlib.sha256(canon.encode()).hexdigest()}"


def _build_feature_vector(
    features: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    """
    Build a scaled (1, n_features) numpy array from a dict of features.
    Missing features are filled with training medians.

    Returns (X_scaled, list_of_missing_features).
    """
    feature_names = get_feature_names()
    medians = get_training_medians()
    scaler = get_scaler()

    row: dict[str, float] = {}
    missing: list[str] = []
    for col in feature_names:
        if col in features:
            row[col] = float(features[col])
        else:
            row[col] = medians.get(col, 0.0)
            missing.append(col)

    df = pd.DataFrame([row], columns=feature_names)
    X_scaled = scaler.transform(df.values)
    return X_scaled, missing


async def predict_single(
    features: dict[str, float],
    redis_client=None,
) -> dict[str, Any]:
    """
    Run a single prediction through the DenseNet model.

    Parameters
    ----------
    features : dict  — feature name → value
    redis_client : optional async Redis — if provided, caches the result

    Returns
    -------
    dict with keys: probability_of_default, prediction, decision, confidence
    """
    # ── Check cache ──
    cache_key = _cache_key(features)
    if redis_client:
        cached = await redis_client.get(cache_key)
        if cached:
            logger.info("Cache HIT for prediction")
            return json.loads(cached)

    model = get_model()
    X_scaled, missing = _build_feature_vector(features)

    if missing:
        logger.info(
            "Filled %d missing features with training medians", len(missing)
        )

    proba = float(model.predict(X_scaled, verbose=0).flatten()[0])
    pred_class = 1 if proba >= 0.5 else 0
    decision = "DEFAULT / HIGH RISK" if proba >= 0.5 else "APPROVED / LOW RISK"
    confidence = round((proba if proba >= 0.5 else 1 - proba) * 100, 1)

    result = {
        "probability_of_default": round(proba, 4),
        "prediction": pred_class,
        "decision": decision,
        "confidence": confidence,
    }

    # ── Store in cache ──
    if redis_client:
        from app.config import settings
        await redis_client.setex(
            cache_key,
            settings.CACHE_TTL_SECONDS,
            json.dumps(result),
        )

    return result


def predict_batch_sync(
    df: pd.DataFrame,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Synchronous batch prediction — used by the background task.

    Parameters
    ----------
    df : pd.DataFrame — input CSV data
    progress_callback : callable(processed, total)

    Returns
    -------
    pd.DataFrame with added columns: default_probability, predicted_class, decision
    """
    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()
    medians = get_training_medians()

    # Prepare feature matrix
    X = pd.DataFrame()
    for col in feature_names:
        if col in df.columns:
            X[col] = df[col].astype(float)
        else:
            X[col] = medians.get(col, 0.0)

    # Fill remaining NaN
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(medians.get(col, 0.0), inplace=True)

    # Scale
    X_scaled = scaler.transform(X.values)

    # Predict in chunks for progress reporting
    chunk_size = 100
    probas = np.empty(len(X_scaled))
    total = len(X_scaled)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        probas[start:end] = model.predict(
            X_scaled[start:end], verbose=0
        ).flatten()
        if progress_callback:
            progress_callback(end, total)

    df = df.copy()
    df["default_probability"] = np.round(probas, 4)
    df["predicted_class"] = (probas >= 0.5).astype(int)
    df["decision"] = np.where(
        probas >= 0.5, "DEFAULT / HIGH RISK", "APPROVED / LOW RISK"
    )

    return df
