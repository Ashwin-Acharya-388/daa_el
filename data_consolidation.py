"""
data_consolidation.py — Dataset Consolidation Module.

Merges the Financial Distress dataset (3,672 rows × 83 features) with the
Taiwanese Bankruptcy Prediction dataset (6,819 rows × 95 features) into a
single consolidated dataset for training.

Strategy:
  Since both datasets use different feature names (x1-x83 vs named ratios),
  we create a UNION schema:
    - All features from both datasets are included
    - Rows from one dataset get 0 for features exclusive to the other
    - A 'source' column tracks origin (for analysis)
    - Both targets are unified into a single binary 'target' column

  The Greedy Feature Selection algorithm (greedy_feature_selection.py) then
  selects the most predictive features from this consolidated set.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config


def load_financial_distress(path: str = None) -> tuple:
    """
    Load and prepare the Financial Distress dataset.

    Returns
    -------
    X : pd.DataFrame — features with standardized column names (fd_x1, fd_x2, ...)
    y : pd.Series    — binary target (1 = distressed, 0 = healthy)
    """
    path = path or config.DATA_PATH
    df = pd.read_csv(path)

    # Drop non-feature columns
    df = df.drop(columns=["Company", "Time"], errors="ignore")

    # Separate target
    y = (df["Financial Distress"] < config.BINARIZE_THRESHOLD).astype(int)
    X = df.drop(columns=["Financial Distress"])

    # Rename features to avoid collision: x1 → fd_x1
    X.columns = [f"fd_{col}" for col in X.columns]

    print(f"[Consolidation] Financial Distress: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"  Healthy: {(y==0).sum()}, Distressed: {(y==1).sum()}")

    return X, y


def load_taiwanese_bankruptcy(path: str = None) -> tuple:
    """
    Load and prepare the Taiwanese Bankruptcy dataset.

    Returns
    -------
    X : pd.DataFrame — features with standardized column names (tw_f1, tw_f2, ...)
    y : pd.Series    — binary target (1 = bankrupt, 0 = not bankrupt)
    """
    path = path or os.path.join(config.BASE_DIR, "taiwanese_bankruptcy.csv")
    df = pd.read_csv(path)

    # Separate target
    y = df["Bankrupt?"].astype(int)
    X = df.drop(columns=["Bankrupt?"])

    # Clean column names (remove leading spaces, special chars)
    clean_names = []
    for i, col in enumerate(X.columns):
        clean_names.append(f"tw_f{i+1}")
    X.columns = clean_names

    print(f"[Consolidation] Taiwanese Bankruptcy: {X.shape[0]} rows × {X.shape[1]} features")
    print(f"  Not Bankrupt: {(y==0).sum()}, Bankrupt: {(y==1).sum()}")

    return X, y


def consolidate_datasets(fd_path: str = None, tw_path: str = None,
                         save: bool = True) -> dict:
    """
    Consolidate both datasets into a single unified dataset.

    Strategy: UNION of all features. Missing features filled with 0.
    This allows the greedy feature selection to pick the best features
    regardless of which dataset they came from.

    Returns
    -------
    dict with keys:
        X : pd.DataFrame — consolidated feature matrix
        y : pd.Series    — unified binary target
        feature_names : list — all feature column names
        source : pd.Series — 'fd' or 'tw' for each row
        fd_features : list — feature names from Financial Distress
        tw_features : list — feature names from Taiwanese dataset
    """
    print(f"\n{'═'*60}")
    print("DATASET CONSOLIDATION")
    print(f"{'═'*60}\n")

    # Load both datasets
    X_fd, y_fd = load_financial_distress(fd_path)
    X_tw, y_tw = load_taiwanese_bankruptcy(tw_path)

    fd_features = X_fd.columns.tolist()
    tw_features = X_tw.columns.tolist()

    # Create source tracking
    source_fd = pd.Series(["fd"] * len(X_fd), name="source")
    source_tw = pd.Series(["tw"] * len(X_tw), name="source")

    # UNION merge: concatenate with fill_value=0 for missing features
    X_fd_expanded = X_fd.reindex(columns=fd_features + tw_features, fill_value=0.0)
    X_tw_expanded = X_tw.reindex(columns=fd_features + tw_features, fill_value=0.0)

    # Stack vertically
    X_combined = pd.concat([X_fd_expanded, X_tw_expanded], axis=0, ignore_index=True)
    y_combined = pd.concat([y_fd, y_tw], axis=0, ignore_index=True)
    source_combined = pd.concat([source_fd, source_tw], axis=0, ignore_index=True)

    # Convert to float32
    X_combined = X_combined.astype(np.float32)

    # Fill any remaining NaN
    X_combined = X_combined.fillna(0.0)

    all_features = X_combined.columns.tolist()

    print(f"\n[Consolidation] ═══ CONSOLIDATED DATASET ═══")
    print(f"  Total rows:     {X_combined.shape[0]:,}")
    print(f"  Total features: {X_combined.shape[1]}")
    print(f"  FD features:    {len(fd_features)}")
    print(f"  TW features:    {len(tw_features)}")
    print(f"  Target distribution:")
    print(f"    Healthy/Not Bankrupt (0): {(y_combined==0).sum():,}")
    print(f"    Distressed/Bankrupt  (1): {(y_combined==1).sum():,}")
    print(f"    Imbalance ratio: 1:{(y_combined==0).sum()/(y_combined==1).sum():.1f}")

    # Save consolidated CSV
    if save:
        consolidated_df = X_combined.copy()
        consolidated_df["target"] = y_combined.values
        consolidated_df["source"] = source_combined.values
        save_path = os.path.join(config.BASE_DIR, "consolidated_dataset.csv")
        consolidated_df.to_csv(save_path, index=False)
        print(f"\n[Consolidation] Saved to: {save_path}")

    return {
        "X": X_combined,
        "y": y_combined,
        "feature_names": all_features,
        "source": source_combined,
        "fd_features": fd_features,
        "tw_features": tw_features
    }


# ─────────────────────── Standalone Test ─────────────────────────────
if __name__ == "__main__":
    result = consolidate_datasets()
    print(f"\nConsolidation complete.")
    print(f"  X shape: {result['X'].shape}")
    print(f"  y distribution: {np.bincount(result['y'].values)}")
    print(f"  Sources: {result['source'].value_counts().to_dict()}")
