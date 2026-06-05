"""
greedy_feature_selection.py — Greedy Forward Feature Selection (DAA Algorithm).

Implements the Greedy Forward Selection algorithm for finding the optimal
subset of features from the consolidated financial risk dataset.

Algorithm (Greedy Paradigm):
  ┌──────────────────────────────────────────────────────────────────┐
  │  GREEDY-FORWARD-SELECT(X, y, all_features):                     │
  │    selected ← ∅                                                 │
  │    remaining ← all_features                                     │
  │    best_score ← 0                                               │
  │                                                                  │
  │    WHILE remaining ≠ ∅:                                         │
  │      best_candidate ← NULL                                      │
  │      best_candidate_score ← best_score                          │
  │                                                                  │
  │      FOR EACH feature f IN remaining:                           │
  │        score ← CV_SCORE(X[selected ∪ {f}], y)                  │
  │        IF score > best_candidate_score + ε:                     │
  │          best_candidate ← f                                      │
  │          best_candidate_score ← score                            │
  │                                                                  │
  │      IF best_candidate ≠ NULL:                                  │
  │        selected ← selected ∪ {best_candidate}                   │
  │        remaining ← remaining - {best_candidate}                 │
  │        best_score ← best_candidate_score                        │
  │      ELSE:                                                       │
  │        BREAK  // No improvement → greedy stopping condition     │
  │                                                                  │
  │    RETURN selected                                               │
  └──────────────────────────────────────────────────────────────────┘

Time Complexity Analysis:
  - Let n = number of features, k = number selected
  - Outer loop: at most n iterations (usually k << n)
  - Inner loop: n, n-1, n-2, ... remaining features
  - Each CV evaluation: O(m) where m = dataset size
  - Total: O(k × n × CV_folds × model_fit_time)
  - Worst case: O(n² × CV_folds × model_fit_time)

Space Complexity: O(n × m) for the feature matrix

Greedy Choice Property:
  At each step, we select the feature that gives the MAXIMUM marginal
  improvement to the cross-validated score. This greedy choice doesn't
  guarantee the global optimum, but empirically yields near-optimal
  subsets in polynomial time vs. the exponential brute-force (2^n subsets).

Reference: 
  - Cormen et al. "Introduction to Algorithms" — Greedy Algorithms (Ch. 16)
  - Guyon & Elisseeff (2003) "An Introduction to Variable and Feature Selection"
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier

import config


def evaluate_feature_set(X: np.ndarray, y: np.ndarray,
                         feature_indices: list,
                         cv_folds: int = 5,
                         seed: int = 42) -> float:
    """
    Evaluate a feature subset using stratified k-fold cross-validation.

    Uses GradientBoosting (fast, handles imbalance well) as the evaluation
    model. Returns mean AUC-ROC score across folds.

    Parameters
    ----------
    X : np.ndarray — full feature matrix
    y : np.ndarray — binary labels
    feature_indices : list — indices of features to evaluate
    cv_folds : int — number of CV folds
    seed : int — random seed

    Returns
    -------
    float — mean AUC-ROC score
    """
    X_subset = X[:, feature_indices]

    # Scale features
    scaler = StandardScaler()
    X_subset = scaler.fit_transform(X_subset)

    # Fast classifier for evaluation
    clf = GradientBoostingClassifier(
        n_estimators=50,        # Reduced for speed
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=seed
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Use AUC-ROC for evaluation (better for imbalanced data)
    scores = cross_val_score(clf, X_subset, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    return scores.mean()


def greedy_forward_selection(X: np.ndarray, y: np.ndarray,
                             feature_names: list,
                             max_features: int = None,
                             min_features: int = 25,
                             min_improvement: float = 0.001,
                             cv_folds: int = 5,
                             seed: int = 42,
                             verbose: bool = True) -> dict:
    """
    Greedy Forward Feature Selection Algorithm.

    At each iteration, greedily selects the single feature that maximizes
    the cross-validated AUC-ROC score when added to the current set.

    Two-phase stopping:
      Phase 1 (Forced): Always picks the best candidate until min_features
                        is reached, regardless of improvement size.
      Phase 2 (Greedy): After min_features, stops when no feature improves
                        the score by at least `min_improvement` (ε).

    Parameters
    ----------
    X : np.ndarray — feature matrix (n_samples × n_features)
    y : np.ndarray — binary target labels
    feature_names : list — names of all features
    max_features : int — maximum features to select (default: n_features // 2)
    min_features : int — minimum features to select (forced, ignores ε)
    min_improvement : float — minimum AUC improvement to continue after
                              min_features is reached (ε)
    cv_folds : int — cross-validation folds
    seed : int — random seed
    verbose : bool — print progress

    Returns
    -------
    dict with keys:
        selected_features : list — names of selected features
        selected_indices : list — indices of selected features
        scores_history : list — AUC score after each feature addition
        all_evaluations : list — detailed evaluation log
        total_time : float — total runtime in seconds
    """
    n_samples, n_features = X.shape
    max_features = max_features or min(n_features // 2, 40)
    min_features = min(min_features, max_features)

    if verbose:
        print(f"\n{'═'*60}")
        print("  GREEDY FORWARD FEATURE SELECTION")
        print(f"{'═'*60}")
        print(f"  Total features:     {n_features}")
        print(f"  Min to select:      {min_features} (forced)")
        print(f"  Max to select:      {max_features}")
        print(f"  Min improvement (ε): {min_improvement} (applied after min)")
        print(f"  CV folds:           {cv_folds}")
        print(f"  Dataset size:       {n_samples:,} samples")
        print(f"{'═'*60}\n")

    selected_indices = []
    selected_names = []
    remaining_indices = list(range(n_features))
    scores_history = []
    all_evaluations = []
    best_score = 0.0

    start_time = time.time()

    for iteration in range(max_features):
        iter_start = time.time()
        n_selected = len(selected_indices)
        in_forced_phase = n_selected < min_features

        phase_label = "FORCED" if in_forced_phase else "GREEDY"

        if verbose:
            print(f"  ── Iteration {iteration + 1}/{max_features} "
                  f"[{phase_label}] "
                  f"({len(remaining_indices)} candidates remaining) ──")

        best_candidate_idx = None
        best_candidate_score = -np.inf if in_forced_phase else best_score
        best_candidate_name = None
        iteration_evals = []

        # ── GREEDY CHOICE: Try each remaining feature ──
        for i, feat_idx in enumerate(remaining_indices):
            # Candidate set = current selected + this feature
            candidate_set = selected_indices + [feat_idx]
            score = evaluate_feature_set(X, y, candidate_set, cv_folds, seed)

            iteration_evals.append({
                "feature": feature_names[feat_idx],
                "index": feat_idx,
                "score": score,
                "improvement": score - best_score
            })

            # In FORCED phase: always pick the highest-scoring candidate
            # In GREEDY phase: only pick if improvement > ε
            if in_forced_phase:
                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate_idx = feat_idx
                    best_candidate_name = feature_names[feat_idx]
            else:
                if score > best_candidate_score + min_improvement:
                    best_candidate_score = score
                    best_candidate_idx = feat_idx
                    best_candidate_name = feature_names[feat_idx]

            # Progress indicator (every 20 features)
            if verbose and (i + 1) % 20 == 0:
                print(f"    Evaluated {i+1}/{len(remaining_indices)} features...")

        # ── STOPPING CONDITION ──
        if best_candidate_idx is None:
            if verbose:
                print(f"\n  ✗ NO feature improves score by ≥ {min_improvement}")
                print(f"    Greedy stopping condition met after {n_selected} features.\n")
            break

        # ── ADD BEST CANDIDATE ──
        selected_indices.append(best_candidate_idx)
        selected_names.append(best_candidate_name)
        remaining_indices.remove(best_candidate_idx)
        prev_score = best_score
        best_score = best_candidate_score
        scores_history.append(best_score)

        iter_time = time.time() - iter_start
        improvement = best_score - prev_score
        all_evaluations.append({
            "iteration": iteration + 1,
            "phase": phase_label,
            "selected_feature": best_candidate_name,
            "auc_score": best_score,
            "improvement": improvement,
            "candidates_evaluated": len(iteration_evals),
            "time_seconds": iter_time,
            "details": sorted(iteration_evals, key=lambda x: x["score"], reverse=True)[:5]
        })

        if verbose:
            print(f"  ✓ [{phase_label}] Selected: {best_candidate_name}")
            print(f"    AUC-ROC: {best_score:.4f} (+{improvement:.4f})")
            print(f"    Time: {iter_time:.1f}s")
            print(f"    Total selected: {len(selected_indices)} features\n")

    total_time = time.time() - start_time

    result = {
        "selected_features": selected_names,
        "selected_indices": selected_indices,
        "scores_history": scores_history,
        "all_evaluations": all_evaluations,
        "total_time": total_time,
        "final_score": best_score,
        "n_features_selected": len(selected_indices)
    }

    if verbose:
        print(f"{'═'*60}")
        print(f"  GREEDY SELECTION COMPLETE")
        print(f"{'═'*60}")
        print(f"  Features selected: {len(selected_indices)} / {n_features}")
        print(f"  Final AUC-ROC:     {best_score:.4f}")
        print(f"  Total time:        {total_time/60:.1f} minutes")
        print(f"  Total evaluations: {sum(e['candidates_evaluated'] for e in all_evaluations)}")
        print(f"\n  Selected features (in order of selection):")
        for i, (name, score) in enumerate(zip(selected_names, scores_history)):
            phase = all_evaluations[i]["phase"]
            print(f"    {i+1:2d}. [{phase:6s}] {name:<35s} AUC={score:.4f}")
        print(f"{'═'*60}")

    return result


def plot_selection_history(result: dict, save_path: str = None):
    """
    Plot the greedy selection progress: AUC-ROC vs number of features.
    """
    save_path = save_path or os.path.join(config.PLOTS_DIR, "greedy_selection.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── AUC vs Features ──
    x = range(1, len(result["scores_history"]) + 1)
    ax1.plot(x, result["scores_history"], "o-", color="#3498db",
             linewidth=2, markersize=6, label="AUC-ROC")
    ax1.fill_between(x, result["scores_history"], alpha=0.15, color="#3498db")
    ax1.set_xlabel("Number of Features Selected", fontsize=12)
    ax1.set_ylabel("Cross-Validated AUC-ROC", fontsize=12)
    ax1.set_title("Greedy Forward Selection Progress", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)

    # Mark the optimal point
    best_idx = np.argmax(result["scores_history"])
    ax1.axvline(x=best_idx + 1, color="#e74c3c", linestyle="--", alpha=0.7,
                label=f"Best: {len(result['scores_history'])} features")

    # ── Marginal Improvement ──
    improvements = [result["scores_history"][0]]
    for i in range(1, len(result["scores_history"])):
        improvements.append(result["scores_history"][i] - result["scores_history"][i-1])

    colors = ["#2ecc71" if imp > 0.005 else "#f39c12" if imp > 0.001 else "#e74c3c"
              for imp in improvements]
    ax2.bar(x, improvements, color=colors, edgecolor="black", alpha=0.85)
    ax2.axhline(y=0.001, color="red", linestyle="--", alpha=0.5, label="ε threshold")
    ax2.set_xlabel("Feature Added (Order)", fontsize=12)
    ax2.set_ylabel("Marginal AUC Improvement", fontsize=12)
    ax2.set_title("Marginal Improvement per Feature", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)

    # Styling
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("DAA Greedy Algorithm — Feature Selection Analysis",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Greedy] Selection plot saved to: {save_path}")


def save_results(result: dict, save_dir: str = None):
    """
    Save greedy selection results to JSON and update config.
    """
    save_dir = save_dir or config.REPORTS_DIR

    # Save detailed results
    results_path = os.path.join(save_dir, "greedy_selection_results.json")

    # Make JSON-serializable
    serializable = {
        "selected_features": result["selected_features"],
        "selected_indices": result["selected_indices"],
        "scores_history": [float(s) for s in result["scores_history"]],
        "final_score": float(result["final_score"]),
        "n_features_selected": result["n_features_selected"],
        "total_time_seconds": float(result["total_time"]),
        "evaluations": []
    }
    for ev in result["all_evaluations"]:
        serializable["evaluations"].append({
            "iteration": ev["iteration"],
            "selected_feature": ev["selected_feature"],
            "auc_score": float(ev["auc_score"]),
            "improvement": float(ev["improvement"]),
            "candidates_evaluated": ev["candidates_evaluated"],
            "time_seconds": float(ev["time_seconds"])
        })

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"[Greedy] Results saved to: {results_path}")

    # Save selected feature list (for easy loading)
    features_path = os.path.join(save_dir, "selected_features.json")
    with open(features_path, "w") as f:
        json.dump({
            "selected_features": result["selected_features"],
            "selected_indices": result["selected_indices"],
            "final_auc": float(result["final_score"])
        }, f, indent=2)
    print(f"[Greedy] Feature list saved to: {features_path}")


def run_greedy_pipeline(max_features: int = None,
                        min_features: int = None,
                        min_improvement: float = None,
                        cv_folds: int = None) -> dict:
    """
    Full pipeline: consolidate datasets → greedy feature selection.
    """
    from data_consolidation import consolidate_datasets

    # Step 1: Consolidate datasets
    data = consolidate_datasets()
    X = data["X"].values
    y = data["y"].values
    feature_names = data["feature_names"]

    # Step 2: Use config defaults if not provided
    max_features = max_features or config.GREEDY_MAX_FEATURES
    min_features = min_features or config.GREEDY_MIN_FEATURES
    min_improvement = min_improvement or config.GREEDY_MIN_IMPROVEMENT
    cv_folds = cv_folds or config.GREEDY_CV_FOLDS

    # Step 3: Run greedy forward selection
    result = greedy_forward_selection(
        X, y, feature_names,
        max_features=max_features,
        min_features=min_features,
        min_improvement=min_improvement,
        cv_folds=cv_folds,
        seed=config.RANDOM_SEED
    )

    # Step 3: Save results and plots
    plot_selection_history(result)
    save_results(result)

    return result


# ─────────────────────── Main Entry Point ────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DAA Greedy Forward Feature Selection for Financial Risk"
    )
    parser.add_argument("--max-features", type=int, default=None,
                        help="Maximum features to select")
    parser.add_argument("--min-features", type=int, default=None,
                        help="Minimum features to select (forced phase)")
    parser.add_argument("--min-improvement", type=float, default=None,
                        help="Minimum AUC improvement threshold ε")
    parser.add_argument("--cv-folds", type=int, default=None,
                        help="Cross-validation folds")
    args = parser.parse_args()

    print("=" * 60)
    print("  DAA — Greedy Forward Feature Selection")
    print("  Financial Risk Analysis Tool")
    print("=" * 60)

    result = run_greedy_pipeline(
        max_features=args.max_features,
        min_features=args.min_features,
        min_improvement=args.min_improvement,
        cv_folds=args.cv_folds
    )

    print(f"\n{'='*60}")
    print("  DONE — Use selected features in training:")
    print(f"  python train.py --use-selected-features")
    print(f"{'='*60}")
