"""
sffs_feature_selection.py — Sequential Floating Forward Selection (SFFS).

Implements the SFFS algorithm (Pudil et al., 1994) for finding a near-global
optimum subset of features from the consolidated financial risk dataset.

SFFS overcomes the nesting effect of plain Greedy Forward Selection by adding
a conditional backward (exclusion) step after each forward (inclusion) step.
This allows features that were useful early on to be removed later if superior
combinations are discovered, enabling the search to escape local optima.

Algorithm (SFFS):
  ┌──────────────────────────────────────────────────────────────────────┐
  │  SFFS(X, y, all_features):                                         │
  │    Y ← ∅              // selected set                              │
  │    k ← 0              // current subset size                       │
  │    best_J[k] ← -∞     // best score seen for each subset size k    │
  │                                                                     │
  │    WHILE k < max_features:                                          │
  │      ── INCLUSION STEP ──                                           │
  │      x⁺ ← argmax_{f ∈ remaining} J(Y ∪ {f})                       │
  │      Y ← Y ∪ {x⁺}                                                 │
  │      k ← k + 1                                                     │
  │      Update best_J[k] if improved                                   │
  │                                                                     │
  │      ── CONDITIONAL EXCLUSION (FLOATING) STEP ──                    │
  │      WHILE k > 2:                                                   │
  │        x⁻ ← argmax_{f ∈ Y} J(Y \ {f})                             │
  │        IF J(Y \ {x⁻}) > best_J[k-1]:                               │
  │          Y ← Y \ {x⁻}                                              │
  │          k ← k - 1                                                  │
  │          best_J[k] ← J(Y)                                          │
  │        ELSE:                                                        │
  │          BREAK                                                      │
  │                                                                     │
  │    RETURN Y                                                         │
  └──────────────────────────────────────────────────────────────────────┘

Time Complexity Analysis:
  - Each inclusion step: O(|remaining| × CV_cost)
  - Each exclusion step: O(|selected| × CV_cost)
  - Worst case per outer iteration: O(n × CV_cost) for inclusion +
    O(k × CV_cost) for potentially repeated exclusions
  - Total: O(max_features × n × CV_cost) in practice

Space Complexity: O(n × m) for the feature matrix + O(n) for score tracking

Advantage over Greedy Forward Selection:
  The floating backward step breaks the nesting property.  A feature that
  was optimal at step 3 can be dropped at step 10 if the remaining nine
  features form a stronger combination without it.  This yields near-global
  optimum subsets in practice.

References:
  - Pudil, Novovičová & Kittler (1994). "Floating search methods in
    feature selection." Pattern Recognition Letters, 15(11), 1119-1125.
  - Cormen et al. "Introduction to Algorithms" — Greedy Algorithms (Ch. 16)
"""

import os
import sys
import time
import json
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

import config


# ════════════════════════════════════════════════════════════════════════
#  Feature Subset Evaluator
# ════════════════════════════════════════════════════════════════════════

def evaluate_feature_set(X: np.ndarray, y: np.ndarray,
                         feature_indices: list,
                         cv_folds: int = 5,
                         seed: int = 42) -> float:
    """
    Evaluate a feature subset using stratified k-fold cross-validation.

    Uses GradientBoosting (fast, handles imbalance well) as the evaluation
    model.  Returns mean AUC-ROC score across folds.

    Parameters
    ----------
    X : np.ndarray — full feature matrix (n_samples × n_features)
    y : np.ndarray — binary labels
    feature_indices : list — indices of features to evaluate
    cv_folds : int — number of CV folds
    seed : int — random seed

    Returns
    -------
    float — mean AUC-ROC score
    """
    X_subset = X[:, feature_indices]

    scaler = StandardScaler()
    X_subset = scaler.fit_transform(X_subset)

    clf = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        random_state=seed,
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X_subset, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    return scores.mean()


# ════════════════════════════════════════════════════════════════════════
#  SFFS Algorithm
# ════════════════════════════════════════════════════════════════════════

def sffs_selection(X: np.ndarray, y: np.ndarray,
                   feature_names: list,
                   max_features: int = None,
                   min_features: int = 25,
                   min_improvement: float = 0.001,
                   cv_folds: int = 5,
                   seed: int = 42,
                   verbose: bool = True) -> dict:
    """
    Sequential Floating Forward Selection (SFFS).

    At each outer iteration:
      1. INCLUSION — greedily add the single feature that maximises CV AUC.
      2. EXCLUSION — repeatedly remove the least-useful feature from the
         current set IF doing so produces a score that exceeds the
         historical best for that (smaller) subset size.
         Continue removing while the condition holds and k > 2.

    Two-phase stopping:
      Phase 1 (Forced):  Always performs inclusion until min_features
                         features are selected (exclusion still runs).
      Phase 2 (Greedy):  After min_features, stops when the inclusion
                         step fails to improve the score by at least ε.

    Parameters
    ----------
    X : np.ndarray — feature matrix (n_samples × n_features)
    y : np.ndarray — binary target labels
    feature_names : list — names of all features
    max_features : int — maximum features to select
    min_features : int — minimum features to select before ε-stopping
    min_improvement : float — ε threshold after min_features reached
    cv_folds : int — cross-validation folds
    seed : int — random seed
    verbose : bool — print progress

    Returns
    -------
    dict with keys:
        selected_features : list — names of selected features
        selected_indices : list — indices of selected features
        scores_history : list — AUC score after each inclusion step
        exclusion_log : list — detailed exclusion event log
        total_time : float — total runtime in seconds
        final_auc : float — final CV AUC-ROC of the selected set
    """
    n_samples, n_features = X.shape
    max_features = max_features or min(n_features // 2, 40)
    min_features = min(min_features, max_features)

    if verbose:
        print(f"\n{'═' * 65}")
        print("  SEQUENTIAL FLOATING FORWARD SELECTION (SFFS)")
        print(f"{'═' * 65}")
        print(f"  Total features:       {n_features}")
        print(f"  Min to select:        {min_features} (forced)")
        print(f"  Max to select:        {max_features}")
        print(f"  Min improvement (ε):  {min_improvement} (applied after min)")
        print(f"  CV folds:             {cv_folds}")
        print(f"  Dataset size:         {n_samples:,} samples")
        print(f"{'═' * 65}\n")

    # ── State ──
    selected_indices: list[int] = []         # Y_k
    remaining_indices: list[int] = list(range(n_features))

    # best_scores[k] = best AUC-ROC ever achieved by ANY subset of size k
    best_scores: dict[int, float] = {0: 0.0}
    # best_subsets[k] = the feature index list that achieved best_scores[k]
    best_subsets: dict[int, list[int]] = {0: []}

    scores_history: list[float] = []         # score after each inclusion step
    inclusion_log: list[dict] = []
    exclusion_log: list[dict] = []
    total_exclusions = 0
    total_evaluations = 0

    start_time = time.time()
    outer_iteration = 0

    while len(selected_indices) < max_features and remaining_indices:
        outer_iteration += 1
        k = len(selected_indices)
        in_forced_phase = k < min_features
        phase_label = "FORCED" if in_forced_phase else "GREEDY"

        # ════════════════════════════════════════════════════════════════
        #  INCLUSION STEP — find x⁺ = argmax_{f ∈ remaining} J(Y ∪ {f})
        # ════════════════════════════════════════════════════════════════
        if verbose:
            print(f"  ── Iteration {outer_iteration} "
                  f"[{phase_label}] INCLUSION  "
                  f"(k={k}, {len(remaining_indices)} candidates) ──")

        best_incl_score = -np.inf
        best_incl_idx = None
        best_incl_name = None
        incl_evals = 0

        for i, feat_idx in enumerate(remaining_indices):
            candidate = selected_indices + [feat_idx]
            score = evaluate_feature_set(X, y, candidate, cv_folds, seed)
            incl_evals += 1

            if score > best_incl_score:
                best_incl_score = score
                best_incl_idx = feat_idx
                best_incl_name = feature_names[feat_idx]

            # Progress indicator
            if verbose and (i + 1) % 20 == 0:
                print(f"    Evaluated {i + 1}/{len(remaining_indices)} "
                      f"candidates …")

        total_evaluations += incl_evals

        # ── Check ε-stopping (only after min_features reached) ──
        prev_best = best_scores.get(k, 0.0)
        improvement = best_incl_score - prev_best
        if not in_forced_phase and improvement < min_improvement:
            if verbose:
                print(f"\n  ✗ Inclusion improvement ({improvement:.5f}) < "
                      f"ε ({min_improvement})")
                print(f"    SFFS stopping condition met at k={k}.\n")
            break

        # ── Accept inclusion ──
        selected_indices.append(best_incl_idx)
        remaining_indices.remove(best_incl_idx)
        k = len(selected_indices)

        current_score = best_incl_score
        if current_score > best_scores.get(k, -np.inf):
            best_scores[k] = current_score
            best_subsets[k] = list(selected_indices)

        scores_history.append(current_score)

        inclusion_log.append({
            "iteration": outer_iteration,
            "phase": phase_label,
            "action": "INCLUDE",
            "feature": best_incl_name,
            "feature_idx": best_incl_idx,
            "auc_score": current_score,
            "subset_size": k,
            "candidates_evaluated": incl_evals,
        })

        if verbose:
            print(f"  ✓ INCLUDED: {best_incl_name}")
            print(f"    AUC-ROC: {current_score:.4f}  "
                  f"(+{improvement:.4f})   k={k}")

        # ════════════════════════════════════════════════════════════════
        #  EXCLUSION (FLOATING) STEP
        #  While k > 2:
        #    x⁻ = argmax_{f ∈ Y_k} J(Y_k \ {f})
        #    if J(Y_k \ {x⁻}) > best_scores[k-1] → remove x⁻, k -= 1
        #    else → break
        # ════════════════════════════════════════════════════════════════
        excl_round = 0
        while k > 2:
            if verbose:
                print(f"    ── Exclusion check (k={k}) ──")

            best_excl_score = -np.inf
            best_excl_idx = None
            best_excl_name = None
            excl_evals = 0

            for feat_idx in selected_indices:
                candidate = [fi for fi in selected_indices if fi != feat_idx]
                score = evaluate_feature_set(X, y, candidate, cv_folds, seed)
                excl_evals += 1

                if score > best_excl_score:
                    best_excl_score = score
                    best_excl_idx = feat_idx
                    best_excl_name = feature_names[feat_idx]

            total_evaluations += excl_evals

            # Compare against best score ever seen at size k-1
            prev_best_at_km1 = best_scores.get(k - 1, -np.inf)

            if best_excl_score > prev_best_at_km1:
                # ── Accept exclusion ──
                selected_indices.remove(best_excl_idx)
                remaining_indices.append(best_excl_idx)
                k = len(selected_indices)
                total_exclusions += 1
                excl_round += 1

                best_scores[k] = best_excl_score
                best_subsets[k] = list(selected_indices)

                exclusion_log.append({
                    "iteration": outer_iteration,
                    "excl_round": excl_round,
                    "action": "EXCLUDE",
                    "feature": best_excl_name,
                    "feature_idx": best_excl_idx,
                    "auc_score": best_excl_score,
                    "prev_best_at_k": prev_best_at_km1,
                    "subset_size": k,
                    "candidates_evaluated": excl_evals,
                })

                if verbose:
                    print(f"    ✗ EXCLUDED: {best_excl_name}")
                    print(f"      AUC-ROC: {best_excl_score:.4f} > "
                          f"prev best@k={k}: {prev_best_at_km1:.4f}")
                    print(f"      k → {k}")
            else:
                if verbose:
                    print(f"    ─ No exclusion improves over best@k={k-1} "
                          f"({prev_best_at_km1:.4f})")
                break

        if verbose:
            print()

    # ═══════════════════════════════════════════════════════════════════
    #  FINAL RESULT
    # ═══════════════════════════════════════════════════════════════════
    total_time = time.time() - start_time
    k_final = len(selected_indices)
    final_score = evaluate_feature_set(X, y, selected_indices, cv_folds, seed)
    selected_names = [feature_names[i] for i in selected_indices]

    result = {
        "selected_features": selected_names,
        "selected_indices": selected_indices,
        "scores_history": scores_history,
        "inclusion_log": inclusion_log,
        "exclusion_log": exclusion_log,
        "best_scores": {str(k): v for k, v in best_scores.items()},
        "total_time": total_time,
        "final_auc": final_score,
        "n_features_selected": k_final,
        "total_exclusions": total_exclusions,
        "total_evaluations": total_evaluations,
    }

    if verbose:
        print(f"{'═' * 65}")
        print(f"  SFFS COMPLETE")
        print(f"{'═' * 65}")
        print(f"  Features selected:    {k_final} / {n_features}")
        print(f"  Final AUC-ROC:        {final_score:.4f}")
        print(f"  Total exclusions:     {total_exclusions}")
        print(f"  Total CV evaluations: {total_evaluations}")
        print(f"  Total time:           {total_time / 60:.1f} minutes")
        print(f"\n  Selected features (final subset):")
        for i, (name, idx) in enumerate(zip(selected_names, selected_indices)):
            print(f"    {i + 1:2d}. {name:<35s} (idx={idx})")
        print(f"{'═' * 65}")

    return result


# ════════════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════════════

def plot_sffs_history(result: dict, save_path: str = None):
    """
    Plot SFFS selection progress:
      - Left:  AUC-ROC vs. iteration (inclusion steps), with exclusion
               events marked.
      - Right: Best AUC-ROC achieved for each subset size k, showing
               the floating search explored many sizes.
    """
    save_path = save_path or os.path.join(config.PLOTS_DIR, "sffs_selection.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left panel: AUC-ROC over inclusion iterations ──
    n_incl = len(result["scores_history"])
    x_incl = list(range(1, n_incl + 1))
    ax1.plot(x_incl, result["scores_history"], "o-", color="#2980b9",
             linewidth=2, markersize=5, label="After inclusion step")
    ax1.fill_between(x_incl, result["scores_history"], alpha=0.12,
                     color="#2980b9")

    # Mark exclusion events on the timeline
    excl_iters = [e["iteration"] for e in result["exclusion_log"]]
    if excl_iters:
        unique_iters = sorted(set(excl_iters))
        for it in unique_iters:
            count = excl_iters.count(it)
            if it <= n_incl:
                ax1.axvline(x=it, color="#e74c3c", linestyle="--",
                            alpha=0.35, linewidth=1)
        # Single legend entry for exclusion lines
        ax1.axvline(x=-1, color="#e74c3c", linestyle="--", alpha=0.6,
                    label=f"Exclusion events ({result['total_exclusions']})")

    ax1.set_xlabel("Inclusion Iteration", fontsize=12)
    ax1.set_ylabel("CV AUC-ROC", fontsize=12)
    ax1.set_title("SFFS — AUC-ROC per Inclusion Step", fontsize=14,
                  fontweight="bold")
    ax1.legend(fontsize=10, loc="lower right")
    ax1.grid(True, alpha=0.3)

    # ── Right panel: Best score per subset size k ──
    best_scores = result["best_scores"]
    ks = sorted(int(k) for k in best_scores.keys() if int(k) > 0)
    scores_at_k = [best_scores[str(k)] for k in ks]

    colors = ["#27ae60" if s > 0.90 else "#f39c12" if s > 0.85
              else "#e74c3c" for s in scores_at_k]
    ax2.bar(ks, scores_at_k, color=colors, edgecolor="black", alpha=0.85,
            width=0.8)
    ax2.set_xlabel("Subset Size k", fontsize=12)
    ax2.set_ylabel("Best CV AUC-ROC Achieved", fontsize=12)
    ax2.set_title("Best Score per Subset Size (Floating Search)",
                  fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    # Highlight the final subset size
    final_k = result["n_features_selected"]
    if final_k in ks:
        idx_in_ks = ks.index(final_k)
        ax2.bar([final_k], [scores_at_k[idx_in_ks]], color="#8e44ad",
                edgecolor="black", alpha=0.9, width=0.8,
                label=f"Final (k={final_k})")
        ax2.legend(fontsize=10)

    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("DAA Sequential Floating Forward Selection (SFFS)",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SFFS] Selection plot saved to: {save_path}")


# ════════════════════════════════════════════════════════════════════════
#  Save Results
# ════════════════════════════════════════════════════════════════════════

def save_results(result: dict, save_dir: str = None):
    """
    Save SFFS results in the same JSON schema expected by the training
    pipeline (selected_features.json) and a detailed log.
    """
    save_dir = save_dir or config.REPORTS_DIR

    # ── Detailed SFFS results ──
    detailed_path = os.path.join(save_dir, "sffs_selection_results.json")
    serializable = {
        "algorithm": "SFFS (Sequential Floating Forward Selection)",
        "selected_features": result["selected_features"],
        "selected_indices": result["selected_indices"],
        "scores_history": [float(s) for s in result["scores_history"]],
        "final_auc": float(result["final_auc"]),
        "n_features_selected": result["n_features_selected"],
        "total_exclusions": result["total_exclusions"],
        "total_evaluations": result["total_evaluations"],
        "total_time_seconds": float(result["total_time"]),
        "best_scores_per_k": {
            k: float(v) for k, v in result["best_scores"].items()
        },
        "inclusion_log": [
            {
                "iteration": e["iteration"],
                "feature": e["feature"],
                "auc_score": float(e["auc_score"]),
                "subset_size": e["subset_size"],
            }
            for e in result["inclusion_log"]
        ],
        "exclusion_log": [
            {
                "iteration": e["iteration"],
                "feature": e["feature"],
                "auc_score": float(e["auc_score"]),
                "prev_best": float(e["prev_best_at_k"]),
                "subset_size": e["subset_size"],
            }
            for e in result["exclusion_log"]
        ],
    }
    with open(detailed_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"[SFFS] Detailed results saved to: {detailed_path}")

    # ── Pipeline-compatible feature list (same schema as greedy) ──
    features_path = os.path.join(save_dir, "selected_features.json")
    with open(features_path, "w") as f:
        json.dump({
            "selected_features": result["selected_features"],
            "selected_indices": result["selected_indices"],
            "final_auc": float(result["final_auc"]),
        }, f, indent=2)
    print(f"[SFFS] Feature list saved to: {features_path}")


# ════════════════════════════════════════════════════════════════════════
#  Pipeline Entry Point
# ════════════════════════════════════════════════════════════════════════

def run_sffs_pipeline(max_features: int = None,
                      min_features: int = None,
                      min_improvement: float = None,
                      cv_folds: int = None) -> dict:
    """
    Full pipeline: consolidate datasets → SFFS feature selection.
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

    # Step 3: Run SFFS
    result = sffs_selection(
        X, y, feature_names,
        max_features=max_features,
        min_features=min_features,
        min_improvement=min_improvement,
        cv_folds=cv_folds,
        seed=config.RANDOM_SEED,
    )

    # Step 4: Save results and plot
    plot_sffs_history(result)
    save_results(result)

    return result


# ════════════════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DAA Sequential Floating Forward Selection (SFFS) "
                    "for Financial Risk Feature Selection"
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

    print("=" * 65)
    print("  DAA — Sequential Floating Forward Selection (SFFS)")
    print("  Financial Risk Analysis Tool")
    print("=" * 65)

    result = run_sffs_pipeline(
        max_features=args.max_features,
        min_features=args.min_features,
        min_improvement=args.min_improvement,
        cv_folds=args.cv_folds,
    )

    print(f"\n{'=' * 65}")
    print("  DONE — Use selected features in training:")
    print(f"  python train.py --consolidated --use-selected-features")
    print(f"{'=' * 65}")
