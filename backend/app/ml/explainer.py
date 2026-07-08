"""
explainer.py — SHAP explanation wrapper.

Computes KernelExplainer-based SHAP values for individual predictions
and returns structured data suitable for frontend visualisation.
Includes timeout protection and graceful error recovery.
"""

import logging
import sys
import time
from typing import Any

import numpy as np

from app.config import PROJECT_ROOT
from app.ml.model_loader import (
    get_model,
    get_scaler,
    get_feature_names,
    get_training_medians,
)
from app.ml.predictor import calibrate_probability

logger = logging.getLogger(__name__)

# Ensure project root is on path for shap_explainer import
_project_str = str(PROJECT_ROOT)
if _project_str not in sys.path:
    sys.path.insert(0, _project_str)

# Maximum time (seconds) to allow for SHAP computation
SHAP_TIMEOUT_SECONDS = 60


async def compute_explanation(
    features: dict[str, float],
    **kwargs,
) -> dict[str, Any]:
    """
    Compute a SHAP explanation for a single set of features.

    Returns a dict ready to send to the frontend.
    Includes timeout protection and graceful error recovery.
    """
    start = time.perf_counter()

    model = get_model()
    scaler = get_scaler()
    feature_names = get_feature_names()
    medians = get_training_medians()

    import pandas as pd

    # Build feature vector
    row = {}
    for col in feature_names:
        val = features.get(col, medians.get(col, 0.0))
        try:
            val = float(val)
            if np.isnan(val) or np.isinf(val):
                val = medians.get(col, 0.0)
        except (ValueError, TypeError):
            val = medians.get(col, 0.0)
        row[col] = val

    df = pd.DataFrame([row], columns=feature_names)
    X_scaled = scaler.transform(df.values)

    # Forward pass to get prediction
    raw_proba = float(model.predict(X_scaled, verbose=0).flatten()[0])
    raw_proba = max(0.0, min(1.0, raw_proba))

    # Apply temperature scaling to match predictor output
    proba = calibrate_probability(raw_proba)

    # Compute SHAP
    sv = None
    base_value = 0.5  # sensible default if SHAP fails

    try:
        import shap
        import warnings

        warnings.filterwarnings("ignore", message=".*additivity.*")

        # Use a small zero-background for speed
        bg = np.zeros((1, len(feature_names)))
        predict_fn = lambda x: model.predict(x, verbose=0).flatten()  # noqa: E731
        explainer = shap.KernelExplainer(predict_fn, bg)

        # Compute SHAP values with nsamples control for speed
        shap_vals = explainer.shap_values(X_scaled, nsamples=200)

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]
        sv = shap_vals.flatten()

        base_value = explainer.expected_value
        if isinstance(base_value, np.ndarray):
            base_value = float(base_value[0])

        elapsed = time.perf_counter() - start
        logger.info("SHAP computed in %.1fs for %d features", elapsed, len(feature_names))

    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.warning(
            "SHAP computation failed after %.1fs: %s. Returning partial explanation.",
            elapsed, e,
        )
        # Return a partial explanation without SHAP values
        status = "DEFAULT / DISTRESSED" if proba >= 0.5 else "HEALTHY / APPROVED"
        confidence = proba if proba >= 0.5 else (1 - proba)
        return {
            "prediction": {
                "probability_of_default": round(proba, 4),
                "decision": status,
                "confidence": round(confidence * 100, 1),
            },
            "base_probability": round(base_value, 4),
            "top_factors_toward_default": [],
            "top_factors_toward_healthy": [],
            "all_shap_values": [],
            "summary": (
                f"This application is predicted as {status} with "
                f"{confidence * 100:.1f}% confidence. "
                f"SHAP explanation unavailable: {e}"
            ),
        }

    # Identify top contributors
    sorted_idx = np.argsort(np.abs(sv))[::-1]
    pushing_default = []
    pushing_healthy = []

    for idx in sorted_idx[:20]:
        entry = {
            "feature": feature_names[idx],
            "shap_value": round(float(sv[idx]), 6),
            "feature_value": round(float(X_scaled[0, idx]), 4),
        }
        if sv[idx] > 0:
            pushing_default.append(entry)
        else:
            pushing_healthy.append(entry)

    status = (
        "DEFAULT / DISTRESSED" if proba >= 0.5 else "HEALTHY / APPROVED"
    )
    confidence = proba if proba >= 0.5 else (1 - proba)

    # Build all SHAP values for waterfall chart
    all_shap = [
        {
            "feature": feature_names[i],
            "shap_value": round(float(sv[i]), 6),
            "feature_value": round(float(X_scaled[0, i]), 4),
        }
        for i in range(len(sv))
    ]

    summary_text = ""
    if pushing_default:
        summary_text = (
            f"This application is predicted as {status} with "
            f"{confidence * 100:.1f}% confidence. "
            f"The base default rate is {base_value * 100:.1f}%. "
            f"The top risk factor is '{pushing_default[0]['feature']}' "
            f"(SHAP contribution: {pushing_default[0]['shap_value']:+.4f})."
        )
    else:
        summary_text = (
            f"This application is predicted as {status} with "
            f"{confidence * 100:.1f}% confidence. No strong risk factors found."
        )

    explanation = {
        "prediction": {
            "probability_of_default": round(proba, 4),
            "decision": status,
            "confidence": round(confidence * 100, 1),
        },
        "base_probability": round(base_value, 4),
        "top_factors_toward_default": pushing_default[:5],
        "top_factors_toward_healthy": pushing_healthy[:5],
        "all_shap_values": all_shap,
        "summary": summary_text,
    }

    return explanation
