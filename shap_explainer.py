"""
shap_explainer.py — Module D: SHAP Explainability.

Provides:
  - Global feature importance (SHAP summary + bar plots)
  - Local instance-level explanations (force + waterfall plots)
  - JSON-formatted explanations for each prediction
  - Human-readable "why was this loan approved/rejected" output

Uses KernelExplainer (model-agnostic) for compatibility with Keras models.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

# Suppress SHAP warnings about additivity checks
warnings.filterwarnings("ignore", message=".*additivity.*")


def create_explainer(model, X_background: np.ndarray, feature_names: list):
    """
    Create a SHAP KernelExplainer using a summarized background dataset.

    Parameters
    ----------
    model : keras.Model — trained DenseNet
    X_background : np.ndarray — training data subsample for background
    feature_names : list — column names

    Returns
    -------
    shap.KernelExplainer
    """
    import shap

    n_bg = min(config.SHAP_BACKGROUND_SAMPLES, len(X_background))
    print(f"[SHAP] Summarizing background data ({n_bg} samples via k-means)...")
    background = shap.kmeans(X_background, min(n_bg, 50))

    def predict_fn(x):
        return model.predict(x, verbose=0).flatten()

    explainer = shap.KernelExplainer(predict_fn, background)
    print("[SHAP] KernelExplainer created.")
    return explainer


def compute_shap_values(explainer, X_explain: np.ndarray,
                        feature_names: list) -> np.ndarray:
    """
    Compute SHAP values for a set of instances.

    Parameters
    ----------
    explainer : shap.KernelExplainer
    X_explain : np.ndarray — instances to explain
    feature_names : list

    Returns
    -------
    shap_values : np.ndarray — shape (n_samples, n_features)
    """
    import shap
    print(f"[SHAP] Computing SHAP values for {len(X_explain)} samples...")
    print("       (This may take a few minutes for KernelExplainer)")
    shap_values = explainer.shap_values(X_explain, nsamples=200)
    return shap_values


def plot_global_importance(shap_values: np.ndarray,
                           X_explain: np.ndarray,
                           feature_names: list):
    """
    Generate global SHAP feature importance plots:
      1. Summary beeswarm plot
      2. Bar plot of mean |SHAP| values
    """
    import shap

    # ── Bar plot of mean absolute SHAP values ──
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[-config.SHAP_TOP_FEATURES:][::-1]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = mean_abs[top_idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_names)))
    bars = ax.barh(range(len(top_names)), top_vals[::-1], color=colors[::-1],
                   edgecolor="black", alpha=0.9)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=11)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Top Feature Importance (SHAP)", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    bar_path = os.path.join(config.PLOTS_DIR, "shap_feature_importance.png")
    plt.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Feature importance bar plot saved to: {bar_path}")

    # ── Summary beeswarm plot ──
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_explain,
            feature_names=feature_names,
            max_display=config.SHAP_TOP_FEATURES,
            show=False
        )
        summary_path = os.path.join(config.PLOTS_DIR, "shap_summary_beeswarm.png")
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Summary beeswarm plot saved to: {summary_path}")
    except Exception as e:
        print(f"[SHAP] Warning: Could not generate beeswarm plot: {e}")


def explain_single_prediction(explainer, instance: np.ndarray,
                              prediction: float, feature_names: list,
                              instance_idx: int = 0) -> dict:
    """
    Generate a human-readable SHAP explanation for a single prediction.

    Parameters
    ----------
    explainer : SHAP explainer
    instance : np.ndarray — single instance (1, n_features)
    prediction : float — model's predicted probability of default
    feature_names : list
    instance_idx : int — index for naming files

    Returns
    -------
    explanation : dict — structured explanation
    """
    import shap

    if instance.ndim == 1:
        instance = instance.reshape(1, -1)

    shap_vals = explainer.shap_values(instance, nsamples=200)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    sv = shap_vals.flatten()

    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = float(base_value[0])

    # Identify top contributors
    sorted_idx = np.argsort(np.abs(sv))[::-1]
    top_n = config.SHAP_TOP_FEATURES

    pushing_default = []
    pushing_healthy = []

    for idx in sorted_idx[:top_n * 2]:
        entry = {
            "feature": feature_names[idx],
            "shap_value": round(float(sv[idx]), 6),
            "feature_value": round(float(instance[0, idx]), 4)
        }
        if sv[idx] > 0:
            pushing_default.append(entry)
        else:
            pushing_healthy.append(entry)

    # Build explanation
    status = "DEFAULT / DISTRESSED" if prediction >= 0.5 else "HEALTHY / APPROVED"
    confidence = prediction if prediction >= 0.5 else (1 - prediction)

    explanation = {
        "prediction": {
            "probability_of_default": round(float(prediction), 4),
            "decision": status,
            "confidence": round(float(confidence * 100), 1)
        },
        "base_probability": round(float(base_value), 4),
        "top_factors_toward_default": pushing_default[:5],
        "top_factors_toward_healthy": pushing_healthy[:5],
        "summary": (
            f"This application is predicted as {status} with "
            f"{confidence*100:.1f}% confidence. "
            f"The base default rate is {base_value*100:.1f}%. "
            f"The top risk factor is '{pushing_default[0]['feature']}' "
            f"(SHAP contribution: {pushing_default[0]['shap_value']:+.4f})."
            if pushing_default else
            f"This application is predicted as {status} with "
            f"{confidence*100:.1f}% confidence. No strong risk factors found."
        )
    }

    # Save waterfall plot
    try:
        shap_explanation = shap.Explanation(
            values=sv,
            base_values=base_value,
            data=instance.flatten(),
            feature_names=feature_names
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_explanation, max_display=10, show=False)
        wf_path = os.path.join(config.PLOTS_DIR,
                               f"shap_waterfall_sample_{instance_idx}.png")
        plt.savefig(wf_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[SHAP] Waterfall plot saved: {wf_path}")
    except Exception as e:
        print(f"[SHAP] Warning: Could not generate waterfall plot: {e}")

    return explanation


def generate_explanations(model, explainer, X_test: np.ndarray,
                          feature_names: list,
                          n_samples: int = None) -> list:
    """
    Generate SHAP explanations for multiple test samples.

    Parameters
    ----------
    model : keras.Model
    explainer : SHAP explainer
    X_test : np.ndarray
    feature_names : list
    n_samples : int — number of samples to explain

    Returns
    -------
    list of explanation dicts
    """
    n_samples = n_samples or config.SHAP_NUM_EXPLANATIONS
    n_samples = min(n_samples, len(X_test))

    # Pick a mix of likely defaults and likely healthy
    preds = model.predict(X_test, verbose=0).flatten()
    sorted_idx = np.argsort(preds)

    # Pick some from each end + middle
    indices = list(sorted_idx[:n_samples//2]) + list(sorted_idx[-(n_samples - n_samples//2):])
    indices = indices[:n_samples]

    explanations = []
    for i, idx in enumerate(indices):
        print(f"\n[SHAP] Explaining sample {i+1}/{n_samples} "
              f"(pred={preds[idx]:.4f})...")
        exp = explain_single_prediction(
            explainer, X_test[idx], preds[idx],
            feature_names, instance_idx=i
        )
        explanations.append(exp)

    # Save all explanations to JSON
    json_path = os.path.join(config.REPORTS_DIR, "shap_explanations.json")
    with open(json_path, "w") as f:
        json.dump(explanations, f, indent=2)
    print(f"\n[SHAP] All explanations saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SAMPLE SHAP EXPLANATIONS")
    print("=" * 60)
    for i, exp in enumerate(explanations):
        print(f"\n--- Sample {i+1} ---")
        print(f"  Decision: {exp['prediction']['decision']}")
        print(f"  Default Probability: {exp['prediction']['probability_of_default']:.2%}")
        print(f"  {exp['summary']}")

    return explanations


# ─────────────────────── Standalone Test ─────────────────────────────
if __name__ == "__main__":
    print("SHAP module loaded. Use via train.py for full pipeline.")
