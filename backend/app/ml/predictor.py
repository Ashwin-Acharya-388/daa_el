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

