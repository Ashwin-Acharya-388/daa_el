"""
model_loader.py — Singleton loader for the DenseNet model, scaler, and metadata.

Loads the trained model once at app startup and keeps it in memory.
Registers the custom Keras layers from the project's densenet_model.py
so that keras.models.load_model can deserialize the .h5 correctly.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from app.config import settings, PROJECT_ROOT, REPORTS_DIR

logger = logging.getLogger(__name__)

# ── Ensure the project root is on sys.path so we can import the original
#    densenet_model.py (which defines the custom Keras layers).
_project_str = str(PROJECT_ROOT)
if _project_str not in sys.path:
    sys.path.insert(0, _project_str)


class _ModelStore:
    """Private singleton that holds model artifacts in memory."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata: dict = {}
        self.metrics: dict = {}
        self.feature_names: list[str] = []
        self.training_medians: dict[str, float] = {}
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self):
        """Load all artifacts from disk.  Called once during app startup."""
        if self._loaded:
            return

        logger.info("Loading model artifacts …")

        # 1. Import custom layers so Keras can deserialize them
        try:
            from densenet_model import (  # noqa: F401
                BottleneckDenseLayer,
                DenseBlock,
                TransitionLayer,
            )
        except ImportError:
            logger.warning(
                "Could not import custom layers from densenet_model.py. "
                "Model loading may fail."
            )

        # 2. Load Keras model
        import tensorflow as tf

        custom_objects = {}
        try:
            from densenet_model import (
                BottleneckDenseLayer,
                DenseBlock,
                TransitionLayer,
            )
            custom_objects = {
                "BottleneckDenseLayer": BottleneckDenseLayer,
                "DenseBlock": DenseBlock,
                "TransitionLayer": TransitionLayer,
            }
        except Exception:
            pass

        self.model = tf.keras.models.load_model(
            settings.MODEL_PATH,
            custom_objects=custom_objects,
            compile=False,
        )
        logger.info("  ✓ Keras model loaded from %s", settings.MODEL_PATH)

        # 3. Load scaler
        self.scaler = joblib.load(settings.SCALER_PATH)
        logger.info("  ✓ Scaler loaded from %s", settings.SCALER_PATH)

        # 4. Load metadata
        with open(settings.METADATA_PATH, "r") as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata["feature_names"]
        self.training_medians = self.metadata.get("training_medians", {})
        logger.info(
            "  ✓ Metadata loaded — %d features", len(self.feature_names)
        )

        # 5. CRITICAL: Validate feature count matches model input dimension
        model_input_dim = self.model.input_shape[1]
        metadata_dim = len(self.feature_names)

        if model_input_dim != metadata_dim:
            logger.warning(
                "  ⚠ DIMENSION MISMATCH: model expects %d features but "
                "metadata declares %d. Attempting fallback to selected_features.json …",
                model_input_dim, metadata_dim,
            )
            # Fallback: load selected_features.json
            fallback_path = str(REPORTS_DIR / "selected_features.json")
            try:
                with open(fallback_path, "r") as f:
                    sel = json.load(f)
                fallback_features = sel["selected_features"]
                if len(fallback_features) == model_input_dim:
                    self.feature_names = fallback_features
                    # Rebuild medians for these features (use 0.0 as default)
                    self.training_medians = {
                        fn: self.training_medians.get(fn, 0.0)
                        for fn in fallback_features
                    }
                    logger.info(
                        "  ✓ Fallback successful — using %d features from selected_features.json",
                        len(fallback_features),
                    )
                else:
                    raise ValueError(
                        f"Fallback also mismatched: selected_features.json has "
                        f"{len(fallback_features)} but model expects {model_input_dim}"
                    )
            except Exception as e:
                raise RuntimeError(
                    f"FATAL: Cannot resolve feature dimension mismatch. "
                    f"Model expects {model_input_dim} features but metadata "
                    f"declares {metadata_dim}. Fallback failed: {e}"
                ) from e
        else:
            logger.info(
                "  ✓ Dimension check passed — model and metadata both use %d features",
                model_input_dim,
            )

        # 6. Load saved evaluation metrics
        try:
            with open(settings.METRICS_PATH, "r") as f:
                self.metrics = json.load(f)
            logger.info("  ✓ Metrics loaded")
        except FileNotFoundError:
            self.metrics = {}
            logger.warning("  ⚠ metrics.json not found — skipping")

        self._loaded = True
        logger.info("All model artifacts loaded successfully.")


# Module-level singleton
_store = _ModelStore()


def load_model():
    """Explicitly load — call during app lifespan startup."""
    _store.load()


def get_model():
    """Return the loaded Keras model."""
    if not _store.is_loaded:
        _store.load()
    return _store.model


def get_scaler():
    """Return the fitted StandardScaler."""
    if not _store.is_loaded:
        _store.load()
    return _store.scaler


def get_metadata() -> dict:
    if not _store.is_loaded:
        _store.load()
    return _store.metadata


def get_feature_names() -> list[str]:
    if not _store.is_loaded:
        _store.load()
    return _store.feature_names


def get_training_medians() -> dict[str, float]:
    if not _store.is_loaded:
        _store.load()
    return _store.training_medians


def get_metrics() -> dict:
    if not _store.is_loaded:
        _store.load()
    return _store.metrics


def is_model_healthy() -> bool:
    """Quick health check — try a dummy forward pass."""
    try:
        m = get_model()
        n = len(get_feature_names())
        dummy = np.zeros((1, n), dtype=np.float32)
        m.predict(dummy, verbose=0)
        return True
    except Exception:
        return False
