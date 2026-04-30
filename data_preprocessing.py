"""
data_preprocessing.py — Module A: Data Loading, Cleaning, and Feature Engineering.

Handles:
  - CSV/Excel loading with configurable target column
  - Dropping non-feature columns (Company, Time, etc.)
  - Binarizing continuous target variables
  - Missing value imputation (median for numeric, mode for categorical)
  - One-hot encoding of categorical features
  - Feature scaling with StandardScaler
  - Stratified train-test split
  - Persisting scaler and column metadata for inference
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

import config


def load_dataset(path: str = None) -> pd.DataFrame:
    """
    Load a dataset from CSV or Excel file.

    Parameters
    ----------
    path : str, optional
        File path. Defaults to config.DATA_PATH.

    Returns
    -------
    pd.DataFrame
    """
    path = path or config.DATA_PATH
    ext = os.path.splitext(path)[1].lower()

    if ext in (".csv",):
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .xlsx")

    print(f"[Preprocessing] Loaded dataset: {path}")
    print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def clean_and_prepare(df: pd.DataFrame,
                      target_col: str = None,
                      cols_to_drop: list = None,
                      binarize: bool = None,
                      threshold: float = None) -> tuple:
    """
    Clean the dataset and prepare features + target.

    Steps:
      1. Drop non-feature columns
      2. Binarize target (if continuous)
      3. Impute missing values
      4. One-hot encode categorical columns

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    target_col : str
        Name of the target column.
    cols_to_drop : list
        Column names to remove before modelling.
    binarize : bool
        Whether to convert target to binary 0/1.
    threshold : float
        Binarization threshold (values < threshold → 1).

    Returns
    -------
    X : pd.DataFrame — feature matrix
    y : pd.Series    — binary target
    """
    target_col = target_col or config.TARGET_COLUMN
    cols_to_drop = cols_to_drop if cols_to_drop is not None else config.COLUMNS_TO_DROP
    binarize = binarize if binarize is not None else config.BINARIZE_TARGET
    threshold = threshold if threshold is not None else config.BINARIZE_THRESHOLD

    df = df.copy()

    # ── 1. Drop non-feature columns ──
    existing_drops = [c for c in cols_to_drop if c in df.columns]
    if existing_drops:
        df.drop(columns=existing_drops, inplace=True)
        print(f"[Preprocessing] Dropped columns: {existing_drops}")

    # ── 2. Separate target ──
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found. "
                       f"Available: {list(df.columns)}")

    y = df[target_col].copy()
    X = df.drop(columns=[target_col])

    # ── 3. Binarize target ──
    if binarize:
        y = (y < threshold).astype(int)
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        print(f"[Preprocessing] Binarized target (threshold={threshold}):")
        print(f"  Class 0 (healthy):    {n_neg} ({n_neg/len(y)*100:.1f}%)")
        print(f"  Class 1 (distressed): {n_pos} ({n_pos/len(y)*100:.1f}%)")

    # ── 4. Impute missing values ──
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    n_missing = X.isnull().sum().sum()
    if n_missing > 0:
        print(f"[Preprocessing] Imputing {n_missing} missing values...")
        for col in numeric_cols:
            if X[col].isnull().any():
                X[col].fillna(X[col].median(), inplace=True)
        for col in categorical_cols:
            if X[col].isnull().any():
                X[col].fillna(X[col].mode()[0], inplace=True)
    else:
        print("[Preprocessing] No missing values found.")

    # ── 5. One-hot encode categorical columns ──
    if categorical_cols:
        print(f"[Preprocessing] One-hot encoding {len(categorical_cols)} "
              f"categorical columns: {categorical_cols}")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    # Convert all to float
    X = X.astype(np.float32)

    print(f"[Preprocessing] Final feature matrix: {X.shape[0]} rows × {X.shape[1]} features")
    return X, y


def scale_and_split(X: pd.DataFrame,
                    y: pd.Series,
                    test_size: float = None,
                    seed: int = None) -> dict:
    """
    Scale features and perform stratified train-test split.

    Saves the fitted scaler and feature column names for inference.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    test_size : float
    seed : int

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test,
        scaler, feature_names
    """
    test_size = test_size or config.TEST_SIZE
    seed = seed or config.RANDOM_SEED

    feature_names = X.columns.tolist()

    # ── Stratified split ──
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )
    print(f"[Preprocessing] Train/Test split ({1-test_size:.0%}/{test_size:.0%}):")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")

    # ── Fit scaler on training data only ──
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ── Save scaler and metadata ──
    scaler_path = os.path.join(config.MODEL_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # Save feature names and training medians (for missing value handling at inference)
    metadata = {
        "feature_names": feature_names,
        "training_medians": X_train.median().to_dict(),
        "target_column": config.TARGET_COLUMN,
        "binarize_threshold": config.BINARIZE_THRESHOLD,
        "columns_dropped": config.COLUMNS_TO_DROP
    }
    meta_path = os.path.join(config.MODEL_DIR, "metadata.json")
    # Convert numpy types for JSON serialization
    serializable = {}
    for k, v in metadata.items():
        if isinstance(v, dict):
            serializable[k] = {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                               for kk, vv in v.items()}
        else:
            serializable[k] = v
    with open(meta_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"[Preprocessing] Scaler saved to: {scaler_path}")
    print(f"[Preprocessing] Metadata saved to: {meta_path}")

    return {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.values,
        "y_test": y_test.values,
        "scaler": scaler,
        "feature_names": feature_names
    }


def run_preprocessing(data_path: str = None) -> dict:
    """
    Full preprocessing pipeline: load → clean → scale → split.

    Returns
    -------
    dict with train/test arrays, scaler, and feature names.
    """
    df = load_dataset(data_path)
    X, y = clean_and_prepare(df)
    result = scale_and_split(X, y)
    return result


# ─────────────────────── Standalone Test ─────────────────────────────
if __name__ == "__main__":
    data = run_preprocessing()
    print(f"\nPreprocessing complete.")
    print(f"  X_train shape: {data['X_train'].shape}")
    print(f"  X_test shape:  {data['X_test'].shape}")
    print(f"  y_train distribution: {np.bincount(data['y_train'])}")
    print(f"  y_test distribution:  {np.bincount(data['y_test'])}")
