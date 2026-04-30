"""
data_balancing.py — Module B: SMOTE-ENN Data Balancing.

Handles:
  - SMOTE oversampling of minority class (distressed/default)
  - ENN (Edited Nearest Neighbors) cleaning of noisy samples
  - Visualization of class distribution before/after balancing
  - Only applied to TRAINING data (test set never touched)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours

import config


def apply_smote_enn(X_train: np.ndarray,
                    y_train: np.ndarray,
                    sampling_strategy: str = None,
                    smote_k: int = None,
                    enn_n: int = None,
                    seed: int = None) -> tuple:
    """
    Apply SMOTE-ENN to balance the training dataset.

    SMOTE creates synthetic minority samples; ENN then removes noisy
    samples from BOTH classes using nearest-neighbor editing.

    Parameters
    ----------
    X_train : np.ndarray — training features
    y_train : np.ndarray — training labels (0/1)
    sampling_strategy : str — SMOTE strategy ("auto" = equalize classes)
    smote_k : int — k-neighbors for SMOTE
    enn_n : int — k-neighbors for ENN
    seed : int — random seed

    Returns
    -------
    X_resampled, y_resampled : np.ndarray
    """
    sampling_strategy = sampling_strategy or config.SMOTE_SAMPLING_STRATEGY
    smote_k = smote_k or config.SMOTE_K_NEIGHBORS
    enn_n = enn_n or config.ENN_N_NEIGHBORS
    seed = seed or config.RANDOM_SEED

    # Count original distribution
    unique, counts = np.unique(y_train, return_counts=True)
    original_dist = dict(zip(unique.astype(int), counts.astype(int)))
    print(f"\n[Balancing] Original training distribution: {original_dist}")
    print(f"  Imbalance ratio: 1:{counts.max()/counts.min():.1f}")

    # Check if minority class has enough samples for SMOTE
    min_count = counts.min()
    if min_count < smote_k + 1:
        smote_k = max(1, min_count - 1)
        print(f"[Balancing] Adjusted SMOTE k_neighbors to {smote_k} "
              f"(minority class has only {min_count} samples)")

    # Build SMOTE-ENN pipeline
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=smote_k,
        random_state=seed
    )
    enn = EditedNearestNeighbours(
        n_neighbors=enn_n
    )
    smote_enn = SMOTEENN(
        smote=smote,
        enn=enn,
        random_state=seed
    )

    print("[Balancing] Applying SMOTE-ENN... (this may take a moment)")
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)

    # Count new distribution
    unique_new, counts_new = np.unique(y_resampled, return_counts=True)
    new_dist = dict(zip(unique_new.astype(int), counts_new.astype(int)))
    print(f"[Balancing] Resampled training distribution: {new_dist}")
    print(f"  Total samples: {len(y_train)} → {len(y_resampled)} "
          f"({len(y_resampled)-len(y_train):+d})")

    # Generate visualization
    _plot_distribution(original_dist, new_dist)

    return X_resampled, y_resampled


def _plot_distribution(original: dict, resampled: dict):
    """
    Plot side-by-side bar charts of class distribution before/after SMOTE-ENN.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    classes = ["Healthy (0)", "Distressed (1)"]
    colors = ["#2ecc71", "#e74c3c"]

    # Before
    orig_vals = [original.get(0, 0), original.get(1, 0)]
    bars1 = axes[0].bar(classes, orig_vals, color=colors, edgecolor="black", alpha=0.85)
    axes[0].set_title("Before SMOTE-ENN", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Number of Samples", fontsize=12)
    for bar, val in zip(bars1, orig_vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     str(val), ha="center", va="bottom", fontweight="bold", fontsize=11)

    # After
    res_vals = [resampled.get(0, 0), resampled.get(1, 0)]
    bars2 = axes[1].bar(classes, res_vals, color=colors, edgecolor="black", alpha=0.85)
    axes[1].set_title("After SMOTE-ENN", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Number of Samples", fontsize=12)
    for bar, val in zip(bars2, res_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     str(val), ha="center", va="bottom", fontweight="bold", fontsize=11)

    # Styling
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=11)

    plt.suptitle("Class Distribution: SMOTE-ENN Balancing", fontsize=16,
                 fontweight="bold", y=1.02)
    plt.tight_layout()

    save_path = os.path.join(config.PLOTS_DIR, "class_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Balancing] Distribution plot saved to: {save_path}")


# ─────────────────────── Standalone Test ─────────────────────────────
if __name__ == "__main__":
    from data_preprocessing import run_preprocessing

    data = run_preprocessing()
    X_bal, y_bal = apply_smote_enn(data["X_train"], data["y_train"])
    print(f"\nBalancing complete.")
    print(f"  X_balanced shape: {X_bal.shape}")
    print(f"  y_balanced distribution: {np.bincount(y_bal)}")
