"""
Backend configuration — environment-based settings via Pydantic.

All paths point to the existing trained-model artifacts so the API
can load the DenseNet, scaler, and metadata without re-training.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings

# ── Resolve project root (two levels up from this file) ──
_THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _THIS_DIR.parent.parent          # daa financial risk analysis EL/
MODEL_DIR = PROJECT_ROOT / "outputs" / "model"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
BATCH_RESULTS_DIR = PROJECT_ROOT / "outputs" / "batch_results"

# Ensure batch results dir exists
BATCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


class Settings(BaseSettings):
    """Application settings — override via environment variables."""

    # ── App ──
    APP_NAME: str = "Financial Risk Analysis API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # ── Database ──
    DATABASE_URL: str = (
        "postgresql+asyncpg://risk_user:risk_password_2024@localhost:5432/financial_risk"
    )

    # ── Redis ──
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 86400  # 24 hours

    # ── Security ──
    API_KEY: str = "fra-dev-key-2024"
    RATE_LIMIT_PER_MINUTE: int = 100

    # ── Model paths ──
    MODEL_PATH: str = str(MODEL_DIR / "densenet_model.h5")
    SCALER_PATH: str = str(MODEL_DIR / "scaler.joblib")
    METADATA_PATH: str = str(MODEL_DIR / "metadata.json")
    METRICS_PATH: str = str(REPORTS_DIR / "metrics.json")
    SHAP_EXPLANATIONS_PATH: str = str(REPORTS_DIR / "shap_explanations.json")
    SELECTED_FEATURES_PATH: str = str(REPORTS_DIR / "selected_features.json")

    # ── CORS ──
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
