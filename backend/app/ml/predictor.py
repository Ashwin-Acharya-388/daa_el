"""
predictor.py — Single and batch prediction logic.

Wraps the loaded DenseNet model for inference with input validation,
error handling, and performance logging.
"""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from app.ml.model_loader import (
    get_model,
    get_scaler,
    get_feature_names,
    get_training_medians,
)

logger = logging.getLogger(__name__)


def _build_feature_vector(
    features: dict[str, float],
) -> tuple[np.ndarray, list[str]]:
    """
    Build a scaled (1, n_features) numpy array from a dict of features.
    Missing features are filled with training medians.
    Non-numeric values are rejected.

    Returns (X_scaled, list_of_missing_features).
    """
    feature_names = get_feature_names()
    medians = get_training_medians()
    scaler = get_scaler()

    row: dict[str, float] = {}
    missing: list[str] = []
    for col in feature_names:
        if col in features:
            try:
                val = float(features[col])
                if np.isnan(val) or np.isinf(val):
                    logger.warning(
                        "Feature '%s' has invalid value (NaN/Inf), using median", col
                    )
                    val = medians.get(col, 0.0)
            except (ValueError, TypeError):
                logger.warning(
                    "Feature '%s' is non-numeric (%r), using median", col, features[col]
                )
                val = medians.get(col, 0.0)
            row[col] = val
        else:
            row[col] = medians.get(col, 0.0)
            missing.append(col)

    df = pd.DataFrame([row], columns=feature_names)
    X_scaled = scaler.transform(df.values)
    return X_scaled, missing


async def predict_single(
    features: dict[str, float],
    **kwargs,
) -> dict[str, Any]:
    """
    Run a single prediction through the DenseNet model.

    Parameters
    ----------
    features : dict  — feature name → value

    Returns
    -------
    dict with keys: probability_of_default, prediction, decision, confidence
    """
    start = time.perf_counter()

    try:
        model = get_model()
        X_scaled, missing = _build_feature_vector(features)

        if missing:
            logger.info(
                "Filled %d missing features with training medians", len(missing)
            )

        proba = float(model.predict(X_scaled, verbose=0).flatten()[0])

        # Clamp probability to [0, 1] for safety
        proba = max(0.0, min(1.0, proba))

        pred_class = 1 if proba >= 0.5 else 0
        decision = "DEFAULT / HIGH RISK" if proba >= 0.5 else "APPROVED / LOW RISK"
        confidence = round((proba if proba >= 0.5 else 1 - proba) * 100, 1)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Prediction: prob=%.4f class=%d confidence=%.1f%% latency=%.1fms",
            proba, pred_class, confidence, elapsed_ms,
        )

        result = {
            "probability_of_default": round(proba, 4),
            "prediction": pred_class,
            "decision": decision,
            "confidence": confidence,
        }

        return result

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception("Prediction failed after %.1fms: %s", elapsed_ms, e)
        raise RuntimeError(f"Model prediction failed: {e}") from e


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
    start = time.perf_counter()

    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()
    medians = get_training_medians()

    # Prepare feature matrix
    X = pd.DataFrame()
    for col in feature_names:
        if col in df.columns:
            X[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
        else:
            X[col] = medians.get(col, 0.0)

    # Fill remaining NaN with medians
    for col in X.columns:
        if X[col].isnull().any():
            X[col].fillna(medians.get(col, 0.0), inplace=True)

    # Replace inf values
    X.replace([np.inf, -np.inf], 0.0, inplace=True)

    # Scale
    X_scaled = scaler.transform(X.values)

    # Predict in chunks for progress reporting
    chunk_size = 100
    probas = np.empty(len(X_scaled))
    total = len(X_scaled)
    errors = []

    for chunk_start in range(0, total, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total)
        try:
            probas[chunk_start:chunk_end] = model.predict(
                X_scaled[chunk_start:chunk_end], verbose=0
            ).flatten()
        except Exception as e:
            logger.warning(
                "Batch chunk %d-%d failed: %s. Filling with -1.0",
                chunk_start, chunk_end, e,
            )
            probas[chunk_start:chunk_end] = -1.0
            errors.append(f"rows {chunk_start}-{chunk_end}: {e}")
        if progress_callback:
            progress_callback(chunk_end, total)

    # Clamp valid probabilities
    valid_mask = probas >= 0
    probas[valid_mask] = np.clip(probas[valid_mask], 0.0, 1.0)

    df = df.copy()
    df["default_probability"] = np.round(probas, 4)
    df["predicted_class"] = np.where(probas >= 0.5, 1, np.where(probas < 0, -1, 0))
    df["decision"] = np.where(
        probas >= 0.5, "DEFAULT / HIGH RISK",
        np.where(probas < 0, "ERROR", "APPROVED / LOW RISK"),
    )

    elapsed = time.perf_counter() - start
    logger.info(
        "Batch prediction: %d rows in %.1fs (%.1f rows/sec), %d errors",
        total, elapsed, total / max(elapsed, 0.001), len(errors),
    )

    return df
